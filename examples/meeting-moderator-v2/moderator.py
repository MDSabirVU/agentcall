#!/usr/bin/env python3
"""
AgentCall - Meeting Moderator (Gemini, 1-on-1 round-trip)

A minimal STT -> LLM -> TTS loop:

    You speak in Google Meet
        -> AgentCall transcribes your speech (STT)
        -> moderator.py forwards each utterance to Gemini
        -> Gemini replies in plain text
        -> moderator.py sends the reply back to AgentCall as a tts.speak
        -> AgentCall plays the reply through the bot's mic in Meet

This is the "prove the round-trip works" iteration. The bot answers every
user utterance. Moderator-specific gating (only speak when X happens,
agenda enforcement, speaker fairness, etc.) is intentionally NOT here yet -
it will be layered on once this loop is stable.

Usage:
    # In ~/Repositories/agentcall/.env:
    #     GEMINI_API_KEY=your-key
    # AgentCall key in env or ~/.agentcall/config.json
    pip install google-genai python-dotenv  # plus requests/websockets already installed
    python examples/meeting-moderator/moderator.py "https://meet.google.com/abc-def-ghi"

    # Custom name and voice:
    python examples/meeting-moderator/moderator.py "https://meet.google.com/abc" --name "Mod" --voice bm_george
"""

import argparse
import asyncio
import difflib
import json
import os
import re
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests
import websockets
from dotenv import load_dotenv

# Load GEMINI_API_KEY (and SLACK_BOT_TOKEN, once configured) from a .env file
# at the repo root (or the cwd). google-genai picks GEMINI_API_KEY up
# automatically when genai.Client() runs, so we just need it to be present
# in os.environ. SLACK_BOT_TOKEN we read explicitly below.
load_dotenv()

from google import genai
from google.genai import types

# Slack delivery lives in a sibling module so this file stays focused on the
# AgentCall <-> Gemini round-trip. Block 1 only uses post_action_items.
from slack_client import post_action_items

# Atlassian (Jira) integration is a sibling module too. We import lazily-
# tolerant: if atlassian-python-api isn't installed yet, the module-level
# import still succeeds and individual functions raise JiraClientError on
# first use. Lets people run the script in dry-run / no-Jira mode without
# the SDK installed.
from atlassian_client import (  # noqa: E402
    JiraClientError,
    STORY_POINTS_FIELD,
    SPRINT_FIELD,
    make_jira_client,
    jira_ping,
    jira_create_issue,
    jira_get_issue,
    jira_edit_issue,
    jira_resolve_account_id,
    jira_list_sprints,
    jira_add_to_sprint,
    jira_list_components,
    jira_list_boards_for_project,
    jira_add_comment,
    jira_search_jql,
    jira_search_jql_total,
    format_issue_url,
    format_issue_for_chat,
    format_created_issue_for_chat,
    format_issue_for_slack,
)

# ----------------------------------------------------------------------------
# AgentCall API config (mirrors examples/support-agent/agent.py)
# ----------------------------------------------------------------------------

_cfg = {}
_cfg_path = os.path.join(os.path.expanduser("~"), ".agentcall", "config.json")
if os.path.exists(_cfg_path):
    try:
        _cfg = json.loads(open(_cfg_path).read())
    except (json.JSONDecodeError, OSError):
        pass

API_BASE = os.environ.get("AGENTCALL_API_URL", "") or _cfg.get("api_url", "") or "https://api.agentcall.dev"
API_KEY = os.environ.get("AGENTCALL_API_KEY", "") or _cfg.get("api_key", "")

if not API_KEY:
    print("Error: set AGENTCALL_API_KEY env var or save it to ~/.agentcall/config.json")
    sys.exit(1)

if not os.environ.get("GEMINI_API_KEY"):
    print("Error: set GEMINI_API_KEY (in env or in a .env file at the repo root)")
    sys.exit(1)

# Slack is optional. If the token isn't set, we just skip the post-call Slack
# delivery entirely - the rest of the pipeline (transcript markdown, REST
# cleanup) still works. We only complain if --slack-channel is passed without
# a token; that's a clear user error and worth catching at startup.
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")

# Atlassian (Jira) is optional. All three values must be set together for any
# Jira capability to light up; we check the trio at the moment each Jira flag
# is used (--show-jira-ping, --jira-project) and fail loudly with a hint.
# Loading them at module level (rather than inside main()) so the dry-run
# path can also reach them when 4c-4e mid-meeting tool-use lands.
ATLASSIAN_EMAIL = os.environ.get("ATLASSIAN_EMAIL", "")
ATLASSIAN_API_TOKEN = os.environ.get("ATLASSIAN_API_TOKEN", "")
ATLASSIAN_BASE_URL = os.environ.get("ATLASSIAN_BASE_URL", "")

HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

# ----------------------------------------------------------------------------
# Gemini config
# ----------------------------------------------------------------------------

# gemini-3.5-flash — Google's frontier Flash (released 2026-05-19). Knowledge
# cutoff Jan 2025; pricing $1.50 input / $9.00 output per million tokens.
# Drop back to gemini-2.5-flash if you hit quota or want a cheaper run.
MODEL = "gemini-3.5-flash"

# System prompt is sent ONCE at chat-session creation (system_instruction).
# Every subsequent send_message call reuses it implicitly - we don't re-prepend
# it on every turn. That's the whole point of using a chat session here: the
# SDK keeps the rolling history for us so we can stay focused on one-shot
# user-turn -> assistant-turn calls.
#
# Block 2 prompt: turns the bot from a "friendly companion" into a
# MOSTLY-SILENT moderator. Every turn returns a structured JSON object so we
# can (a) decide whether to actually speak via tts.speak, (b) capture action
# items deterministically as the meeting unfolds, and (c) detect off-topic
# drift in real time. The previous chatty behaviour is gone - the bot now
# speaks only when one of five trigger conditions fires.
SYSTEM_PROMPT = """
You are Juno, a virtual meeting moderator who has joined a video call.
Your job is to keep the meeting focused, capture action items, draw out
quiet participants, and rein in dominators. STAY MOSTLY SILENT.
Listening is your default. Most turns you should output speak: null
AND intent: null.

You intervene in FIVE situations. THREE of them you do directly
(AGENDA-SETTING, DIRECT ADDRESS, WRAP-UP RECAP). TWO of them
(OFF-TOPIC REDIRECT, SILENT-PARTICIPANT NUDGE) use a TWO-PHASE
"raise-hand" pattern: first politely ask permission to speak, then
deliver the actual intervention only after a human acknowledges you
("yes Juno", "go ahead Mod", "what is it"). NEVER barge into the
conversation directly for off-topic or silent-nudge - it feels rude.

While your hand is raised (the per-turn context will show
"Pending intervention: <type>"), do NOT raise it again - just wait
silently for someone to acknowledge you.

THE RULES:

1. AGENDA-SETTING — direct (intent="open")
   When: the meeting just started and you don't know the agenda yet.
   How: greet warmly ("Hi everyone,") and ask what the agenda is for
        the call. Do this ONCE at the opening, then stay silent until
        someone answers. Once the agenda is set, never ask again.

2. OFF-TOPIC REDIRECT — TWO PHASES
   When: "Off-topic streak" reaches FIFTEEN cumulative turns across all
         participants. Single drifts and brief tangents are fine —
         only sustained drift earns a redirect.

   Phase 2a (intent="raise_hand", speak=null): SILENT hand-raise.
   Emit ONLY the intent — "speak" MUST be null. The system will
   trigger Google Meet's native ✋ raise-hand reaction on your
   behalf. NO words, NO TTS — purely visual. The "Pending
   intervention" line in your next turn's context will remind you
   that your hand is raised.

   Phase 2b (intent="redirect"): fires only when the per-turn context
   shows "Pending intervention: off_topic" AND the current speaker
   acknowledges you (phrases like "yes Juno", "go ahead Mod", "what
   is it", "yeah?", "sure", "tell us"). NOW deliver the redirect
   politely, mentioning the agenda:
     - "Thanks. I just wanted to gently flag we've drifted from
        <agenda>. Mind if we steer back?"
     - "Thank you. Could we please come back to <agenda>?"

   If no acknowledgment, stay silent (speak: null, intent: null) and
   keep waiting.

3. SILENT-PARTICIPANT NUDGE — TWO PHASES
   When: a participant appears in "Silent participants eligible for
         nudge" in the per-turn context. That list ALREADY filters
         for "0 turns AND at least 20 turns elapsed SINCE THEY
         JOINED" — every name in it is fair game. The plain "Silent
         participants" line shows everyone with 0 turns, including
         late joiners still in their grace window; DO NOT nudge
         those, only the ones on the "eligible" line. NEVER nudge
         someone already in "Already nudged".

   Phase 3a (intent="raise_hand", speak=null): same silent hand-raise
   as Phase 2a. NO words, NO TTS — just the intent. Don't name the
   silent person yet; that comes in Phase 3b.

   FIRE IMMEDIATELY — do not wait for a "more natural" moment. On
   the FIRST turn where ALL of these hold, emit intent="raise_hand"
   that same turn:
     • "Silent participants eligible for nudge" lists at least one
       name (the per-participant grace has already been enforced
       upstream — every name on that line is fair game RIGHT NOW)
     • "Pending intervention" is "none (hand is down)"
     • No other rule is firing this turn: the current utterance is
       NOT a wrap-up closing phrase (Rule 5), NOT a direct address
       to you (Rule 4), and the off-topic streak is < 15 (Rule 2
       takes priority if both apply)
   Once you raise the hand, "Pending intervention" will flip to
   "silent_nudge for <Name>" on the next turn — wait there for
   acknowledgment, then deliver Phase 3b.

   COEXISTS WITH ACTION-ITEM CAPTURE: intent="raise_hand" lives in
   a SEPARATE field from new_action_items. If the current utterance
   contains a commitment (e.g. "I'll handle the sales coordination"),
   capture it in new_action_items AND set intent="raise_hand" on the
   SAME turn. You are not choosing between them — both fields can
   be populated together. Do NOT skip the raise_hand just because
   there's a commitment to log.

   Phase 3b (intent="nudge"): fires only when the per-turn context
   shows "Pending intervention: silent_nudge for <Name>" AND the
   current speaker acknowledges you. NOW deliver the nudge by name:
     - "Thanks. Alice, we'd love to hear your thoughts — please
        jump in whenever."
     - "Thank you. Bob, you've been quiet — anything you'd like
        to add?"

   If no acknowledgment, stay silent and keep waiting.

4. DIRECT ADDRESS — direct (intent="answer")
   When: someone says "Juno, …" or "Mod, …" with a question or
         request that isn't acknowledgment of a pending hand-raise.
   How: answer helpfully. Keep regular questions under 25 words;
        recap-style questions ("what did we discuss?", "what's been
        covered?", "summarize the last 5 minutes") may use up to
        60 words. No hand-raise.

   For RECAP requests, include EVERYTHING that was discussed — the
   on-topic action items AND any brief off-topic tangents (mention
   them honestly: "We also briefly touched on X"). Don't censor
   tangents — they're part of the meeting and the asker wants a
   complete picture.

   For CAPABILITY questions ("what can you do?", "what are you for?",
   "are you an AI?"), use the YOUR CAPABILITIES section below. Do
   NOT improvise capabilities that aren't listed there.

   For OUT-OF-SCOPE requests ("write me a poem", "what's the
   weather", "schedule a meeting", "send an email"), politely decline
   and offer to capture an action item for a human to do it instead.
   Example: "That's outside what I can do here, but I can note it
   as an action item if you'd like."

   IMPORTANT EXCEPTION: if "Pending intervention" is set AND the
   speaker's utterance reads as acknowledgment ("yes Juno", "what
   is it", "go ahead") rather than a substantive new question, treat
   it as Phase 2b/3b instead — deliver the pending intervention.

YOUR CAPABILITIES (use this when asked "what can you do?"):

   Describe what you DO for participants — the benefits — NOT the
   internal mechanisms. The audience doesn't care about "raising
   hands" or "state machines"; they care about what value you add
   to their meeting.

   In this meeting I can help by:
     1. Capturing action items as people commit to tasks during the
        discussion, with owners and due dates
     2. Gently bringing the conversation back on track when it drifts
        too far from the agenda
     3. Inviting quieter participants to share their thoughts when
        they've been silent for a while
     4. Summarizing what's been discussed at any point during the
        call, including a sense of WHEN things happened — so you
        can ask things like "what did we cover in the last 5 minutes?"
        and get an accurate answer (including stretches where the
        room was quiet)
     5. Creating Jira tickets in our SHP board on request — just say
        something like "Mod, create a ticket called X, assigned to Y,
        with high priority and 3 story points under epic SHP-15162".
        I read the details back and only file the ticket after you
        say "confirm"
     6. Looking up existing Jira tickets if you mention the key —
        e.g. "Mod, what's in SHP-17182?" — I post the card to Meet
        chat with status, assignee, priority, and a link
     7. Editing existing tickets — title, priority, assignee,
        reporter, story points, epic, sprint, components, labels,
        due date, or description — e.g. "Mod, set SHP-17182 to high
        priority", "reassign SHP-17182 to Yaman", "add SHP-17182 to
        the current sprint", "set SHP-17182 components to STACK and
        CORE", "set reporter to Joas", or "change the title of
        SHP-17182 to 'Switch from Gemini 2.5 family to 3.0'". You
        can also refer to the most recent ticket without re-stating
        its key — e.g. "edit this ticket to high priority" or "add
        Alexander as assignee on the ticket you just created".
        Sprint references can name a team (e.g. "team core current
        sprint" / "team ux next sprint"); without a team, the Stack
        team's board is used. Same confirm step before anything
        changes.
     8. Adding comments to a ticket — e.g. "Mod, add a comment to
        SHP-17182 saying 'still investigating, will follow up
        Wednesday'". Same confirm step.
     9. Answering simple board-level questions — e.g. "Mod, how many
        open tickets are there for team stack?", "Mod, list the open
        tickets for team core", "Mod, how many open tickets does
        Alexander have?", or "Mod, open tickets per person on team
        stack in this sprint". I'll run a JQL search on the SHP
        board and either speak the count, post a list to chat, or
        break it down by person across the roster. No confirmation
        needed (read-only). Team and sprint filters are optional and
        combine freely.
    10. Recapping action items and posting a meeting summary to
        Slack when we wrap up

   CRITICAL: When ASKED "what can you do?" / "what are you for?" /
   similar capability questions, state ONLY the POSITIVES from the
   list above. DO NOT append "I cannot..." or "but I can't..."
   disclaimers — those make the answer feel defensive and undersell
   the bot. Keep the spoken answer roughly 80-100 words (more than
   the pre-Jira budget because the Jira surface keeps growing), upbeat,
   and forward-looking. End on a positive note (e.g. "...when we wrap
   up.") — never with a limitation. Always mention the Jira
   capabilities (create / lookup / edit / comment / search counts &
   lists by team) when summarising — they're often the most demo-
   relevant abilities. Group the Jira capability mentions together
   for clarity.

   IF THE SAME PERSON ASKS A SECOND OR THIRD TIME: vary your
   wording materially. Don't recite the same sentences in a
   slightly different order — open differently, reorder the
   capability list, and on the third+ ask drop the recital
   altogether in favour of one concrete example they could try
   right now ("Want me to file a ticket called X, or count the
   open tickets on team stack?"). The per-turn context will tell
   you when this is a repeat.

   The list below is for YOUR INTERNAL REFERENCE ONLY. Use it to
   recognize and politely decline out-of-scope requests (Rule 4
   OUT-OF-SCOPE handling). NEVER recite this list in response to
   a "what can you do?" question — only mention a limitation when
   the speaker has specifically asked for that exact out-of-scope
   thing:

   Internal reference — out-of-scope items (don't volunteer these):
     - Scheduling meetings, sending emails, controlling external apps
       OTHER than Jira (Jira create/lookup/edit/comment/search IS in
       scope — see items 5-9 above; sprint, components, labels, due
       date, and reporter are all configurable via items 5-7;
       counting / listing open tickets on a team board is item 9)
     - Looking up info outside this conversation (no web/document
       search beyond Jira tickets / boards)
     - Remembering anything from previous meetings
     - Speaking any language other than English
     - Deleting Jira tickets or anything destructive (read + create
       + edit + comment only; no DELETE)
     - Changing Jira ticket status (To Do → In Progress workflow
       transitions) or linking tickets via relationships other than
       epic/parent (no blocks / is-blocked-by / relates-to links)

   When asked about capabilities, describe them CONCISELY (under
   60 words). Frame them as BENEFITS — what I do for the meeting —
   never as how I do them. Phrases to AVOID: "I raise my hand",
   "I have a state machine", "I emit JSON", "my prompt", "the model",
   "TTS", "WebSocket", AND any "I cannot..." / "I can't..." /
   "but I can't..." disclaimers in capability answers. Phrases to
   USE: "I help by...", "I can bring you back on topic when...",
   "I gently nudge quieter participants when...", "I can summarize
   what we've covered".

5. WRAP-UP RECAP — direct (intent="wrap_up")
   When: the meeting is winding down. Fire IMMEDIATELY on ANY of:
     - "thanks all", "thanks everyone", "thank you all"
     - "that's everything", "that's it", "that's all"
     - "let's end here", "let's wrap up", "we're done"
     - "have a good one", "talk soon", "catch you later"
     - "bye", "goodbye", "see you"
   How: speak directly (no hand-raise) using NATURAL closing phrasing.
        LEAD WITH a wrap-up phrase, then recap action items grouped
        by owner. Examples of good lead-ins:
          - "Just to wrap up what we discussed today, ..."
          - "Just to recap, ..."
          - "Before we close, here's a quick recap — ..."
          - "Quick recap before we go: ..."
   If there are NO action items, summarize the main discussion
   points (priorities, decisions, themes) with the same lead-in.

WHEN IN DOUBT: stay silent. speak: null AND intent: null is the right
answer for any turn where you wouldn't naturally jump into a real
meeting as a human moderator.

YOU MUST RESPOND IN THIS EXACT JSON SHAPE ON EVERY TURN. No prose,
no markdown code fences, no preamble — just the raw JSON object:

{
  "speak": null OR "the words to say out loud",
  "intent": null OR "open" OR "raise_hand" OR "redirect" OR "nudge" OR "answer" OR "wrap_up",
  "agenda_items": [] OR ["item 1", "item 2"],
  "new_action_items": [] OR [
    {"owner": "<first name>", "task": "<short description>",
     "due": null OR "YYYY-MM-DD"}
  ],
  "off_topic": false OR true,
  "follow_up_meeting": null OR {
    "topic": "<short description>",
    "attendees": ["<name>", ...]
  }
}

Field semantics:
- speak: literal words AgentCall TTS will play. null on silent turns
  AND null when intent is "raise_hand" (raise_hand is purely visual).
  Word caps:
    redirect / nudge / open     — under 25 words
    answer (regular question)   — under 25 words
    answer (recap or capability) — under 60 words
    wrap_up                     — under 60 words
- intent: which rule fired this turn. Values:
    open       - Rule 1 (agenda-setting)             — speak REQUIRED
    raise_hand - Rule 2a / 3a (silent native Meet ✋) — speak MUST be null
    redirect   - Rule 2b (off-topic redirect)        — speak REQUIRED
    nudge      - Rule 3b (silent-participant nudge)  — speak REQUIRED
    answer     - Rule 4 (direct address)             — speak REQUIRED
    wrap_up    - Rule 5 (closing recap)              — speak REQUIRED
  intent="raise_hand" is the ONLY case where intent is set but speak
  is null. For any other intent, speak MUST be a non-empty string.
- agenda_items: populate ONLY on the turn where the agenda is being
  established. After that, leave empty.
- new_action_items: capture every commitment, decision, priority,
  or concrete task mentioned in THIS turn. Interpret LIBERALLY:
    * Priorities ("we should prioritize feature X" → owner=speaker,
      task="prioritize feature X").
    * Decisions ("we should retire feature Z" → owner=speaker).
    * Aspirations ("I think a launch webinar is worth doing" →
      owner=speaker, task="organize launch webinar").
    * Explicit assignments ("Alice will own the pricing audit by
      Friday" → owner="Alice", task="pricing audit", due="<friday-iso>").
  When in doubt: CAPTURE IT.

  CRITICAL — DO NOT RE-CAPTURE: check the "Existing action items
  captured so far" block in the per-turn context. If the current
  utterance is essentially RESTATING a commitment already in that
  list (same intent for the same owner, even with different
  wording), leave new_action_items EMPTY for this turn. Listing
  the same commitment twice with slight wording variations reads
  as repetitive padding. Only emit a new item when there's a
  genuinely NEW task or commitment, OR when the current utterance
  adds significant detail to an existing one (in which case
  re-emit the EXPANDED version — the dedup layer will merge it
  with the existing entry and keep the more detailed wording).

  CRITICAL — OWNER RESOLUTION: the owner MUST be the first name of
  an actual call participant. The per-turn context lists current
  participants under "Participants in this call". If a name is
  garbled by STT or ambiguous, pick the closest match from that
  list. If the speaker uses "we" or "I", set owner to the current
  speaker's first name (from "Current speaker" in the context).
  NEVER invent a name that isn't a participant.

  Empty list ONLY if the turn was purely conversational with zero
  forward-looking content.

  Use ISO YYYY-MM-DD for due dates. CRITICAL: ALL due dates MUST
  be in the current year shown in "Today's date", OR later. NEVER
  emit a due date earlier than today. Resolve relative dates
  ("Friday", "end of Q3", "28th of May") against "Today's date" in
  context, not against your training cutoff.
- off_topic: true if the current utterance is off-topic vs the
  stated agenda. False otherwise (and false if no agenda is set).
- follow_up_meeting: populate when someone proposes a follow-up
  session. Otherwise null.

Never say "as an AI" or talk about being a language model.
""".strip()


# Module-level Gemini client. We keep a strong reference here so its
# underlying httpx connection pool doesn't get torn down while the chat
# is still in use - if `client` only existed inside build_chat_session()
# it would be garbage-collected on return and every subsequent
# chat.send_message would fail with "Cannot send a request, as the
# client has been closed."
_gemini_client: Optional["genai.Client"] = None


def build_chat_session():
    """
    Create a Gemini chat session.

    Why a chat session and not raw generate_content() per turn?
      - The session keeps the rolling conversation history on Google's side,
        so we don't have to manage a messages list ourselves.
      - The system_instruction is set once at create time, so it isn't billed
        on every turn (cached by Gemini between calls within the session).
      - send_message(text) returns a response with a `.text` accessor that
        already concatenates all parts.

    The Client() picks up GEMINI_API_KEY from the environment automatically.
    """
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client()
    chat = _gemini_client.chats.create(
        model=MODEL,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            # JSON mode at chat-level. Every turn the moderator emits the
            # structured object described in SYSTEM_PROMPT. Setting this
            # once at session creation means we don't have to override
            # per-call.
            response_mime_type="application/json",
            # Bumped 400 -> 800 -> 1500 across iterations. 800 was tripping
            # mid-string truncations once the chat history and cumulative
            # action-item list had grown into the back half of a real
            # meeting (~10-15 minutes in). 1500 fits the full envelope
            # with comfortable headroom for a long action-item list and
            # a 25-word "speak" line.
            max_output_tokens=1500,
            # Lower than Block 1's 0.7 because structured output benefits
            # from less creativity. Still enough headroom for Gemini to
            # pick varied "speak" phrasings.
            temperature=0.4,
        ),
    )
    return chat


def speaker_name(raw) -> str:
    """
    Normalise a speaker / participant field to a plain string.

    AgentCall sometimes delivers these as a plain string ("Alice") and
    sometimes as a nested object ({"id": "spaces/.../devices/123",
    "name": "Alice"}). We want just the human-readable name.
    """
    if isinstance(raw, dict):
        return raw.get("name") or raw.get("id") or "Unknown"
    if isinstance(raw, str):
        return raw or "Unknown"
    return "Unknown"


def ask_gemini(chat, user_text: str) -> str:
    """
    Send one user utterance to the Gemini chat session and return the reply.

    The chat object keeps the full history internally, so we only pass the
    latest user turn each time. Gemini sometimes returns an empty `.text`
    (safety filter, empty candidate list, etc.); callers must handle "" by
    skipping the tts.speak rather than crashing or sending blank audio.

    NOTE: Block 2 onward, the chat session emits JSON. Use
    ask_moderator_structured() instead - this raw text helper now exists
    only for legacy --dry-run prompt-tuning compatibility (Block 1 code
    paths) and will likely be removed once the new dry-run is stable.
    """
    try:
        response = chat.send_message(user_text)
        return (response.text or "").strip()
    except Exception as exc:
        # Don't crash the whole call if one Gemini request fails - log and
        # return empty so the loop continues listening.
        print(f"  [gemini error] {exc}")
        return ""


# Shape returned when ask_moderator_structured() can't parse a response.
# Mirrors the SYSTEM_PROMPT schema exactly so callers can treat it as a
# no-op turn without special-casing.
_EMPTY_TURN = {
    "speak": None,
    "intent": None,
    "agenda_items": [],
    "new_action_items": [],
    "off_topic": False,
    "follow_up_meeting": None,
}

# Whitelist of valid intent strings the model can emit. Anything else
# coerces to None - the raise-hand / pending logic only acts on values
# from this set.
_VALID_INTENTS = {"open", "raise_hand", "redirect", "nudge", "answer", "wrap_up"}


def ask_moderator_structured(chat, prompt: str) -> dict:
    """
    Send one enriched turn-context prompt to the moderator chat session and
    return the parsed JSON object as a dict.

    The chat is configured (in build_chat_session) with
    response_mime_type="application/json", so Gemini emits valid JSON with
    no markdown fences and no preamble. We still defend against:
      - empty response (safety filter etc.) -> _EMPTY_TURN
      - JSON parse failure -> log + _EMPTY_TURN
      - missing/wrong-type fields -> coerce or default

    Returns the EXACT _EMPTY_TURN shape on any failure so callers can
    blindly read response["speak"], response["new_action_items"], etc.
    without worrying about KeyErrors.
    """
    try:
        response = chat.send_message(prompt)
        raw = (response.text or "").strip()
    except Exception as exc:
        print(f"  [moderator gemini error] {exc}")
        return dict(_EMPTY_TURN)

    if not raw:
        # Safety filter or empty candidate - treat as a no-op turn.
        return dict(_EMPTY_TURN)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        # Should be rare now that we're in JSON mode, but if it ever
        # happens we'd rather skip the turn than crash the call.
        print(f"  [moderator parse error] {exc}")
        print(f"  [moderator raw output] {raw[:200]!r}{'...' if len(raw) > 200 else ''}")
        return dict(_EMPTY_TURN)

    # Coerce each field to the expected type. Gemini is reliable but not
    # perfect - defensive coercion costs nothing and saves us from "got
    # None where I expected a list" bugs downstream.
    speak_raw = data.get("speak")
    speak = speak_raw.strip() if isinstance(speak_raw, str) and speak_raw.strip() else None

    # intent must come from the whitelist; otherwise it can't drive the
    # pending-intervention state machine and we treat it as missing.
    intent_raw = data.get("intent")
    intent = intent_raw.strip() if isinstance(intent_raw, str) else None
    if intent not in _VALID_INTENTS:
        intent = None
    # Belt-and-braces: speak and intent must agree, with ONE exception —
    # raise_hand is purely visual (native Meet ✋ reaction), so it MUST
    # have speak=null. All other intents require a non-empty speak.
    if intent == "raise_hand":
        # If the model accidentally generated a speak with raise_hand,
        # drop it: the hand is the action, not the words.
        speak = None
    elif speak and not intent:
        intent = "answer"  # safe default - "answer" is the most generic
    elif intent and not speak:
        intent = None

    agenda_items_raw = data.get("agenda_items") or []
    agenda_items = [str(x).strip() for x in agenda_items_raw if str(x).strip()] if isinstance(agenda_items_raw, list) else []

    new_action_items_raw = data.get("new_action_items") or []
    new_action_items = []
    if isinstance(new_action_items_raw, list):
        for item in new_action_items_raw:
            if not isinstance(item, dict):
                continue
            owner = str(item.get("owner", "")).strip()
            task = str(item.get("task", "")).strip()
            if not task:
                continue  # an action item with no task is useless
            due_raw = item.get("due")
            due = str(due_raw).strip() if due_raw and str(due_raw).strip().lower() != "null" else None
            new_action_items.append({"owner": owner or "Unknown", "task": task, "due": due})

    off_topic = bool(data.get("off_topic", False))

    follow_up_raw = data.get("follow_up_meeting")
    follow_up = None
    if isinstance(follow_up_raw, dict):
        topic = str(follow_up_raw.get("topic", "")).strip()
        attendees_raw = follow_up_raw.get("attendees") or []
        attendees = [str(a).strip() for a in attendees_raw if str(a).strip()] if isinstance(attendees_raw, list) else []
        if topic:
            follow_up = {"topic": topic, "attendees": attendees}

    return {
        "speak": speak,
        "intent": intent,
        "agenda_items": agenda_items,
        "new_action_items": new_action_items,
        "off_topic": off_topic,
        "follow_up_meeting": follow_up,
    }


def merge_turn_into_state(state: dict, turn: dict) -> None:
    """
    Merge one structured turn response into the rolling state dict.

    Why merge instead of overwrite?
      - agenda_items: only the first non-empty list wins. Once an agenda
        is set, subsequent turns shouldn't be able to replace it (we tell
        the model this in SYSTEM_PROMPT, but defensive code reinforces).
      - action_items: cumulative across the whole meeting.
      - speaker_turns / off_topic_streak: tracked outside this helper by
        the caller (they need the current speaker name etc.).
      - follow_up_meeting: last writer wins (latest proposal is what we
        act on at end-of-call).
    """
    if turn.get("agenda_items") and not state.get("agenda"):
        state["agenda"] = list(turn["agenda_items"])

    new_items = turn.get("new_action_items") or []
    if new_items:
        existing = state.setdefault("action_items", [])
        for item in new_items:
            owner = (item.get("owner") or "").strip()
            task = (item.get("task") or "").strip()
            if not owner or not task:
                continue

            # Two-stage dedup. Same commitment often gets captured twice
            # across turns with slightly different wording — "safety
            # evals" on turn 12, then "own the safety eval plan" on
            # turn 13. Listing both reads as repetitive padding.
            #
            #   Stage 1: exact match on (owner first-name, task) — kept
            #   for cheapness when the model emits an identical
            #   duplicate.
            #
            #   Stage 2: fuzzy match per-owner via SequenceMatcher
            #   (cutoff 0.6). If a similar task already exists for the
            #   same owner first-name, MERGE rather than append:
            #     - keep the longer task description (more detail wins)
            #     - preserve any due date from either side (existing
            #       wins if both have one, since the earlier capture
            #       was typically the explicit commitment)
            owner_first = owner.split()[0].lower()

            # Stage 1: exact match
            if any(
                (ex.get("owner") or "").split()[0].lower() == owner_first
                and (ex.get("task") or "").lower() == task.lower()
                for ex in existing
            ):
                continue

            # Stage 2: fuzzy match against existing items from same owner.
            # _task_similarity uses MAX of SequenceMatcher ratio and
            # token-coverage ratio so it catches both "same words slightly
            # rearranged" and "same key nouns, different framing verbs".
            matched_idx = -1
            for i, ex in enumerate(existing):
                ex_owner_first = (ex.get("owner") or "").split()[0].lower()
                if ex_owner_first != owner_first:
                    continue
                if _task_similarity(task, ex.get("task") or "") >= 0.6:
                    matched_idx = i
                    break

            if matched_idx >= 0:
                ex = existing[matched_idx]
                # Prefer the more detailed task description.
                if len(task) > len(ex.get("task") or ""):
                    ex["task"] = task
                # Fill in due date if the existing item didn't have one.
                if not ex.get("due") and item.get("due"):
                    ex["due"] = item["due"]
                continue

            existing.append({"owner": owner, "task": task, "due": item.get("due")})

    if turn.get("follow_up_meeting"):
        state["follow_up"] = turn["follow_up_meeting"]


# ----------------------------------------------------------------------------
# Raise-hand / pending intervention state machine
# ----------------------------------------------------------------------------
#
# Rules 2 (off-topic redirect) and 3 (silent-participant nudge) use a
# two-phase pattern: the bot first raises its hand with a short polite
# "may I chime in?" line, then on a later turn — once a human
# acknowledges it — actually delivers the redirect or nudge.
#
# state["pending_intervention"] tracks that "hand raised, waiting" state
# between turns. None when nothing is pending. A dict otherwise:
#   {"type": "off_topic" | "silent_nudge", "target": "<name or agenda>"}
#
# The functions below own all reads/writes to this state so the live
# (run_moderator) and dry-run (run_dry) paths stay in lockstep.


# Per-participant nudge grace: a late joiner shouldn't be nudged the
# moment they arrive into a room that's already past N total turns.
# Each participant gets their own N-turn buffer measured from the
# total-turn count at the moment they joined. SILENT_NUDGE_GRACE is
# both the threshold (in update_pending_intervention) and the buffer
# size (in this picker) - keeping it in one named constant means the
# two stay in lockstep if we ever tune the value.
#
# Bumped 10 -> 20 (2026-05-21) per user feedback: meetings tend to
# warm up gradually; nudging silent attendees after only ten turns
# was firing too early in real calls.
SILENT_NUDGE_GRACE = 20

# Off-topic streak threshold: the bot raises a silent hand once this
# many consecutive off-topic turns accumulate across all participants.
# Bumped 8 -> 15 (2026-05-21) — eight was too eager, the redirect was
# firing on natural mid-meeting tangents that resolved themselves.
OFF_TOPIC_STREAK_THRESHOLD = 15


def _silent_nudge_candidates(state: dict) -> List[str]:
    """
    Return the silent participants who are eligible to be nudged
    RIGHT NOW: zero turns, not already nudged, AND at least
    SILENT_NUDGE_GRACE turns have happened SINCE THEY JOINED.

    Per-participant grace replaces the old global "total_turns >= 10"
    gate. A late joiner walking into a room with 30 prior turns isn't
    immediately nudge-eligible — they get the same 10-turn buffer the
    original participants got from the meeting's start. joined_at_turn
    captures the total-turn count at the moment each participant was
    first seen (via participant.joined in the live path, /join in
    dry-run, or implicit add via transcript.final).
    """
    nudged = state.get("nudged_set", set())
    joined_at_turn = state.get("joined_at_turn") or {}
    total_turns = sum(state.get("speaker_turns", {}).values())
    candidates: List[str] = []
    for name, count in state.get("speaker_turns", {}).items():
        if count != 0 or name in nudged:
            continue
        # Default to 0 for participants seeded BEFORE joined_at_turn
        # tracking existed (or any path that forgot to seed) — this
        # matches the previous global-gate behaviour, so we never
        # regress to "harder to nudge than before".
        join_turn = joined_at_turn.get(name, 0)
        if (total_turns - join_turn) >= SILENT_NUDGE_GRACE:
            candidates.append(name)
    return candidates


def _pick_silent_nudge_target(state: dict) -> Optional[str]:
    """
    Identify the most likely silent participant the bot is about to
    nudge. Used to populate pending_intervention["target"] when the
    model emits intent="raise_hand" without telling us who.

    Returns the FIRST nudge-eligible candidate (silent, not nudged,
    grace period elapsed), else None.
    """
    candidates = _silent_nudge_candidates(state)
    return candidates[0] if candidates else None


def update_pending_intervention(state: dict, turn: dict) -> None:
    """
    Set or clear state["pending_intervention"] based on what the model
    just emitted.

      - intent="raise_hand" → set pending based on which rule's
        conditions are currently met. Off-topic streak >=
        OFF_TOPIC_STREAK_THRESHOLD (15) takes priority over
        silent-nudge if both happen to be true.
      - intent in ("redirect", "nudge") → clear pending (delivered).
      - intent in ("answer", "wrap_up", "open") → leave pending alone.
      - intent is None (silent turn) → leave pending alone.
    """
    intent = turn.get("intent")
    if intent == "raise_hand":
        # GUARD A: if the bot's hand is ALREADY up, this is a model
        # violation (SYSTEM_PROMPT explicitly says "do not raise again
        # while pending"). Suppress so we don't accidentally toggle
        # the Meet ✋ off via a duplicate meeting.raise_hand send.
        if state.get("pending_intervention"):
            turn["intent"] = None
            return

        # GUARD B: validate that one of the two real triggers actually
        # holds. The model sometimes raises hand early (e.g. at
        # off-topic streak=4 instead of 15). If neither rule's hard
        # threshold is crossed, treat this as a misfire and suppress —
        # bot stays silent, no Meet ✋ sent, no pending set. The
        # threshold gets enforced server-side regardless of how the
        # model's judgment drifts under prompt updates.
        streak = state.get("off_topic_streak") or 0
        # _pick_silent_nudge_target now enforces the per-participant
        # grace internally (a candidate is only returned once they
        # personally have >= SILENT_NUDGE_GRACE turns since joining),
        # so we no longer need the old global "total_turns >= 10"
        # gate here. If no candidate qualifies, the picker returns
        # None and we fall through to the misfire-cancel branch below.
        silent_target = _pick_silent_nudge_target(state)
        if streak >= OFF_TOPIC_STREAK_THRESHOLD:
            agenda_list = state.get("agenda") or []
            agenda_str = ", ".join(agenda_list) if agenda_list else "the agenda"
            state["pending_intervention"] = {
                "type": "off_topic",
                "target": agenda_str,
            }
        elif silent_target:
            state["pending_intervention"] = {
                "type": "silent_nudge",
                "target": silent_target,
            }
        else:
            # Misfire: model raised hand prematurely. Cancel.
            turn["intent"] = None
            return
    elif intent == "redirect":
        # Delivering the redirect clears pending AND resets the
        # off-topic streak. The redirect has done its job; if the
        # room drifts again afterwards, we want a fresh 8-turn
        # buffer before raising hand a second time. Without this
        # reset, the streak counter from the just-delivered drift
        # carries forward and the second raise_hand would trigger
        # almost immediately on the next off-topic utterance.
        state["pending_intervention"] = None
        state["off_topic_streak"] = 0
        state["off_topic_last_speaker"] = None
    elif intent == "nudge":
        # Nudge clears pending. The off-topic streak is unrelated
        # to silent-participant handling, so it's left alone.
        state["pending_intervention"] = None


# Minimal stopword set for action-item dedup. Kept short on purpose —
# we strip these so the token-overlap similarity below focuses on
# content words (nouns + verbs). NOT a full NLP stopword list; just
# the connective tissue that shows up in commitment phrasing.
_TASK_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for",
    "with", "by", "from", "into", "at", "as",
    "is", "are", "be", "been", "being", "was", "were", "have", "has",
    "had", "will", "would", "should", "can", "could", "may", "might",
    "i", "we", "you", "they", "he", "she", "it", "us", "our", "your",
    "this", "that", "these", "those", "add", "own", "do",
}


def _normalize_task_tokens(task: str) -> set:
    """
    Tokenize a task description for similarity comparison.

    Lowercases, strips punctuation, removes stopwords, and applies
    crude stemming (-ing, trailing -s) so "benchmark" matches
    "benchmarking" and "evals" matches "eval". Returns a set of
    content tokens.
    """
    tokens: set = set()
    for word in task.lower().split():
        word = "".join(c for c in word if c.isalnum())
        if not word or word in _TASK_STOPWORDS or len(word) <= 2:
            continue
        if word.endswith("ing") and len(word) > 5:
            word = word[:-3]
        elif word.endswith("s") and len(word) > 3:
            word = word[:-1]
        tokens.add(word)
    return tokens


def _task_similarity(task_a: str, task_b: str) -> float:
    """
    Similarity score for action-item dedup, in [0, 1].

    Combines two signals via max:
      - SequenceMatcher ratio (catches near-identical strings)
      - Token-coverage ratio: shared_tokens / min(tokens_a, tokens_b)
        (catches "same key nouns, different verbs" — e.g. "add safety
        evals — refusal rates and tone consistency" vs "own the
        safety eval plan", which share {safety, eval} = 2 out of
        min(8, 3) = 0.67).

    Using max means EITHER signal can trigger a merge. The cost is
    occasional false positives when two genuinely distinct tasks share
    a key noun (e.g. "review prompt templates" vs "migrate prompt
    templates"). The prompt-level "Existing action items" guard
    catches most of those before they reach this function.
    """
    seq = difflib.SequenceMatcher(None, task_a.lower(), task_b.lower()).ratio()
    tokens_a = _normalize_task_tokens(task_a)
    tokens_b = _normalize_task_tokens(task_b)
    if not tokens_a or not tokens_b:
        coverage = 0.0
    else:
        intersection = len(tokens_a & tokens_b)
        coverage = intersection / min(len(tokens_a), len(tokens_b))
    return max(seq, coverage)


def _format_existing_action_items(action_items: List[dict]) -> str:
    """
    Format the cumulative action items captured so far for the per-turn
    prompt. The model uses this to avoid re-capturing the same
    commitment in slightly different wording on a later turn.

    Grouped by owner first name, one bullet per task. Returns an empty
    string when nothing is captured yet (no point spending tokens on
    an empty list).
    """
    if not action_items:
        return ""
    grouped: "dict[str, List[str]]" = {}
    for item in action_items:
        owner = (item.get("owner") or "Unknown").split()[0]
        task = (item.get("task") or "").strip()
        if not task:
            continue
        grouped.setdefault(owner, []).append(task)
    if not grouped:
        return ""
    lines = ["Existing action items captured so far (DO NOT re-capture these):"]
    for owner, tasks in grouped.items():
        for t in tasks:
            lines.append(f"  - {owner}: {t}")
    return "\n".join(lines) + "\n"


def _format_elapsed(started_at: datetime) -> str:
    """
    Format meeting-elapsed time for the per-turn prompt.

    Used so the moderator can answer time-relative questions accurately
    ("what did we discuss in the last 5 minutes?"). Each per-turn
    prompt includes this string at the top, so the model's chat
    history accumulates a timestamp for every utterance — letting it
    infer silent gaps from consecutive deltas and bound "the last N
    minutes" against the current elapsed value.

    Output examples: "42s", "8m 17s", "1h 23m".
    """
    seconds = max(0, int((datetime.now() - started_at).total_seconds()))
    if seconds < 60:
        return f"{seconds}s"
    minutes, rem_s = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {rem_s}s"
    hours, rem_m = divmod(minutes, 60)
    return f"{hours}h {rem_m}m"


# Phrases that mean "tell me what you can do". We treat any of these
# (in a bot-addressed turn) as a capability question, bump a counter,
# and feed the count into the per-turn prompt so the model varies its
# wording on the 2nd, 3rd, ... ask.
_CAPABILITY_QUESTION_PATTERN = re.compile(
    r"\b(?:"
    r"what(?:\s+(?:type|kind|sort)s?\s+of)?\s+(?:tasks?|things?|stuff|"
    r"work)?\s*(?:can|do|are\s+you\s+able\s+to)\s+you\s+do"
    r"|what\s+are\s+you\s+(?:capable\s+of|for|able\s+to\s+do)"
    r"|what\s+(?:can|do)\s+you\s+do"
    r"|what(?:'s|\s+is)\s+your\s+(?:job|role|function|purpose)"
    r"|(?:tell|remind)\s+me\s+(?:again\s+)?(?:what|how)\s+you"
    r"|(?:repeat|tell\s+me\s+again)\s+(?:what|the)\s+(?:type|kinds?|tasks?)"
    r"|can\s+you\s+(?:tell\s+me|describe|explain)\s+"
    r"(?:what|the\s+(?:tasks?|things?))"
    r"|how\s+can\s+you\s+help"
    r"|what\s+do\s+you\s+do"
    r")\b",
    re.IGNORECASE,
)


def _is_capability_question(text: str, bot_name: str) -> bool:
    """
    True when the utterance reads as a "what can you do?" style
    question. We deliberately DROP the bot-addressing requirement
    here — STT often strips the bot name on follow-up turns ("once
    again, what tasks can you do?") and we'd undercount otherwise.
    The pattern is specific enough that incidental human-to-human
    "what do you do here?" chatter is unlikely to trip it; even if
    it does, the worst outcome is one extra variation directive
    on the next bot-addressed turn.
    """
    if not text:
        return False
    return bool(_CAPABILITY_QUESTION_PATTERN.search(text))


def _format_capability_variation_directive(count: int) -> str:
    """
    Build a per-turn directive that tells the model to vary its
    wording when the same speaker keeps asking what the bot can do.
    Returns "" for the first ask (no variation needed) so the
    baseline answer template still applies cleanly.

    The directive escalates: 2nd ask = "use a fresh opener and
    reorder", 3rd = "shorter, headline form", 4th+ = "be playful,
    suggest examples". Idea: the user gets DIFFERENT useful info
    each time, not a slightly-reworded re-recital.
    """
    if count <= 1:
        return ""
    if count == 2:
        return (
            "IMPORTANT: this is the SECOND time this caller has asked "
            "what you can do this meeting. Do NOT repeat the previous "
            "phrasing verbatim. Open with a fresh sentence (e.g. "
            "'Sure — here's another angle on it:' or 'Happy to recap "
            "differently:'), reorder the capabilities, lead with the "
            "Jira-related ones since they're the most demo-relevant, "
            "and trim any phrasing you used last time.\n"
        )
    if count == 3:
        return (
            "IMPORTANT: this is the THIRD time this caller has asked. "
            "Keep it SHORT and headline-style (under 40 words) — they "
            "already know the long form. Lead with one concrete example "
            "they could try right now (e.g. 'You could ask me to file "
            "a ticket called X, or look up SHP-N'), then a one-line "
            "summary of everything else.\n"
        )
    # 4th and beyond: gentle de-escalation, the user might be testing.
    return (
        f"IMPORTANT: this caller has asked what you can do {count} times "
        "this meeting. Acknowledge that lightly ('I think you've got the "
        "shape of it — anything specific you want to try?'), offer ONE "
        "concrete suggestion (e.g. creating a ticket or running an open-"
        "ticket count for a team), and stop. Do NOT recite the full "
        "capability list again.\n"
    )


def _format_pending_for_prompt(pending: Optional[dict]) -> str:
    """
    Format state["pending_intervention"] for the per-turn prompt.

    The bot reads this line and uses it to decide whether to deliver
    (Phase 2b / 3b) or keep waiting. Wording is deliberately
    instructional rather than just a value dump.
    """
    if not pending:
        return "none (hand is down)"
    ptype = pending.get("type")
    target = pending.get("target") or "the agenda"
    if ptype == "off_topic":
        return (
            f"off_topic — hand raised, waiting for acknowledgment. "
            f"If current speaker acknowledges you, deliver an off-topic "
            f"redirect to: {target}."
        )
    if ptype == "silent_nudge":
        return (
            f"silent_nudge for {target} — hand raised, waiting for "
            f"acknowledgment. If current speaker acknowledges you, "
            f"deliver a polite nudge inviting {target} to speak."
        )
    return "unknown (hand raised, unclear purpose — keep waiting)"


def _format_state_diagnostics(state: dict, turn: dict) -> str:
    """
    One-line per-turn diagnostic string showing internal counters
    the bot uses to decide whether to intervene. Mirrors the dry-run
    [state] line so live testing exposes the same signals.

    Empty pieces are omitted, so a quiet turn with no action items
    and no off-topic still gets a useful counter readout.
    """
    parts: List[str] = []
    speaker_turns = state.get("speaker_turns") or {}
    if speaker_turns:
        counts = " ".join(f"{n}={c}" for n, c in speaker_turns.items())
        parts.append(f"turns({counts})")
    silent = [n for n, c in speaker_turns.items() if c == 0]
    if silent:
        parts.append(f"silent({', '.join(silent)})")
        # Show how many turns have happened since each silent person
        # joined — this is the per-participant grace clock. Helps
        # debug "why didn't the bot nudge X?" by surfacing whether
        # X is still in their grace window. Format: "since_joined
        # (Name=3/10)" meaning 3 of 10 turns elapsed since they
        # joined. Once the numerator hits the denominator they're
        # nudge-eligible (assuming they're not already nudged).
        joined_at_turn = state.get("joined_at_turn") or {}
        total_turns = sum(speaker_turns.values())
        grace_parts = []
        for name in silent:
            join_turn = joined_at_turn.get(name, 0)
            elapsed = total_turns - join_turn
            grace_parts.append(f"{name}={elapsed}/{SILENT_NUDGE_GRACE}")
        if grace_parts:
            parts.append(f"since_joined({' '.join(grace_parts)})")
    streak = state.get("off_topic_streak") or 0
    parts.append(f"streak={streak}")
    pending = state.get("pending_intervention")
    if pending:
        parts.append(f"pending={pending.get('type')}")
    if turn.get("agenda_items"):
        parts.append(f"agenda={turn['agenda_items']}")
    if turn.get("new_action_items"):
        parts.append(f"+{len(turn['new_action_items'])} action item(s)")
    if turn.get("off_topic"):
        parts.append("off_topic=true")
    if turn.get("follow_up_meeting"):
        parts.append("follow_up suggested")
    return f"  [state] {' | '.join(parts)}"


# ----------------------------------------------------------------------------
# Owner normalization: map captured action-item owners to real participants
# ----------------------------------------------------------------------------

def _normalize_owner(captured: str, participants: List[str]) -> str:
    """
    Map a captured owner first-name (or full name) to the closest
    matching real participant from the call.

    STT is messy: Yaman gets transcribed as "Aman", Daniyal as
    "Danielle", etc. The captured owner string also might come from
    the model's interpretation of an "I"/"we" utterance and lack a
    surname. This function picks the best match against the actual
    participant list so the Slack post groups everyone correctly.

    Strategy (in order):
      1. No participants known → return captured unchanged.
      2. Exact case-insensitive match on first name → return full
         participant name.
      3. difflib fuzzy match on first name (cutoff 0.7) → return
         full participant name.
      4. No match → return captured unchanged so it shows up clearly
         in the Slack post and we know STT/model misfired.
    """
    if not captured or not participants:
        return captured
    captured_first = captured.strip().split()[0].lower() if captured.strip() else ""
    if not captured_first:
        return captured

    first_to_full = {p.split()[0].lower(): p for p in participants}
    if captured_first in first_to_full:
        return first_to_full[captured_first]

    matches = difflib.get_close_matches(
        captured_first, list(first_to_full.keys()), n=1, cutoff=0.7
    )
    if matches:
        return first_to_full[matches[0]]
    return captured


def normalize_action_item_owners(action_items: List[dict], participants: List[str]) -> List[dict]:
    """
    Return a new list of action items with every owner normalized to
    a real call participant where possible. Items are NOT dropped if
    a match isn't found - they keep their captured owner so the user
    sees the STT/model failure rather than silently losing the item.

    After normalization, duplicates (same normalized owner + same
    task text, case-insensitive) are deduped because the rename can
    collapse e.g. "Aman: do X" and "Yaman: do X" into the same item.
    """
    normalized: List[dict] = []
    for item in action_items:
        new_item = dict(item)
        new_item["owner"] = _normalize_owner(item.get("owner", ""), participants)
        # Dedup against what we've already emitted.
        dup = any(
            ex["owner"].lower() == new_item["owner"].lower()
            and ex["task"].lower() == new_item["task"].lower()
            for ex in normalized
        )
        if not dup:
            normalized.append(new_item)
    return normalized


# ----------------------------------------------------------------------------
# Avatar infrastructure (webpage-av mode)
# ----------------------------------------------------------------------------
# Mirror of scripts/python/bridge-visual.py minus the screenshare paths we
# don't use here. When the user passes --avatar, we need to:
#   1. Start a local HTTP server that serves ui-templates/<name>/index.html
#      plus the shared agentcall-audio.js. Random port.
#   2. Create the AgentCall call with ui_port=<that port>. The response
#      carries tunnel_id + tunnel_access_key + tunnel_url.
#   3. Open a WebSocket to AgentCall's tunnel endpoint and proxy HTTP
#      requests back to the local server. FirstCall's headless browser
#      loads the template via that tunnel.
# All three pieces have to be live for the bot's video tile to render the
# template. If any fails we fall back to audio-only with a warning.

# aiohttp is only required when --avatar is in use. We defer the import so
# audio-only runs (the common case) don't pay the dependency cost or fail
# noisily on an aiohttp-less venv.
def _require_aiohttp():
    """Import aiohttp.web on demand. Raises a friendly error if missing."""
    try:
        from aiohttp import web  # type: ignore
        return web
    except ImportError as exc:
        raise RuntimeError(
            "Avatar mode (--avatar) requires aiohttp. Install it with:\n"
            "  pip install -r examples/meeting-moderator-v2/requirements.txt\n"
            f"(original ImportError: {exc})"
        )


class _TemplateServer:
    """
    Aiohttp app that serves a single ui-template directory. Configured by
    `start_template_server` below. Routes:
      GET /                        -> index.html
      GET /agentcall-audio.js      -> the shared JS one level up
      GET /<anything-else>         -> static file inside the template dir

    Path-traversal guard: we realpath the resolved file and confirm it
    sits inside the template directory before serving.
    """

    def __init__(self, template_dir: str, shared_js_path: str):
        self.template_dir = template_dir
        self.shared_js_path = shared_js_path

    async def handle_index(self, request):
        web = _require_aiohttp()
        index_path = os.path.join(self.template_dir, "index.html")
        if not os.path.exists(index_path):
            return web.Response(status=404, text="index.html not found")
        with open(index_path, "r", encoding="utf-8") as f:
            return web.Response(text=f.read(), content_type="text/html")

    async def handle_shared_js(self, request):
        web = _require_aiohttp()
        if not os.path.exists(self.shared_js_path):
            return web.Response(status=404)
        with open(self.shared_js_path, "r", encoding="utf-8") as f:
            return web.Response(
                text=f.read(), content_type="application/javascript",
            )

    async def handle_static(self, request):
        web = _require_aiohttp()
        filename = request.match_info.get("filename", "")
        filepath = os.path.realpath(os.path.join(self.template_dir, filename))
        if not filepath.startswith(os.path.realpath(self.template_dir)):
            return web.Response(status=403, text="Forbidden")
        if os.path.exists(filepath) and os.path.isfile(filepath):
            return web.FileResponse(filepath)
        return web.Response(status=404)


async def start_template_server(template_name: str):
    """
    Start a local aiohttp server for the named UI template. Returns
    (runner, port). The caller MUST `await runner.cleanup()` on exit.
    Returns (None, 0) if the template directory doesn't exist.
    """
    web = _require_aiohttp()

    # Templates live at <repo-root>/ui-templates/. moderator.py sits at
    # <repo-root>/examples/meeting-moderator-v2/moderator.py, so two
    # `os.path.dirname` calls climb back to the root.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(script_dir))
    templates_base = os.path.join(repo_root, "ui-templates")
    template_dir = os.path.join(templates_base, template_name)
    shared_js = os.path.join(templates_base, "agentcall-audio.js")

    if not os.path.isdir(template_dir):
        print(f"  [avatar] template '{template_name}' not found at {template_dir}")
        return None, 0

    server = _TemplateServer(template_dir, shared_js)
    app = web.Application()
    app.router.add_get("/", server.handle_index)
    app.router.add_get("/agentcall-audio.js", server.handle_shared_js)
    # Bridge-visual.py also registers this oddly-pathed variant because
    # some templates reference "../agentcall-audio.js" from a sub-route.
    app.router.add_get("/../agentcall-audio.js", server.handle_shared_js)
    app.router.add_get("/{filename:.+}", server.handle_static)

    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)  # 0 = OS picks a free port
    await site.start()
    # Pull the actual port we got from the underlying socket.
    port = site._server.sockets[0].getsockname()[1]
    return runner, port


class _BridgeTunnelClient:
    """
    AgentCall tunnel client — proxies HTTP requests from FirstCall's
    headless browser back to our local template server. Connects to
    `/internal/tunnel/connect` on the AgentCall WS, authenticates with
    the per-call tunnel_access_key (NOT the API key), then forwards
    `http.request` messages to the local port via aiohttp.

    Lifted from scripts/python/bridge-visual.py with the screenshare /
    webpage routing branches stripped — we only proxy the UI port.
    """

    def __init__(
        self,
        tunnel_ws_url: str,
        tunnel_id: str,
        access_key: str,
        ui_port: int,
    ):
        self.tunnel_ws_url = tunnel_ws_url
        self.tunnel_id = tunnel_id
        self.access_key = access_key
        self.ui_port = ui_port
        self._ws = None
        self._running = False
        self._tasks: List[asyncio.Task] = []

    async def connect(self):
        self._running = True
        self._ws = await websockets.connect(self.tunnel_ws_url)
        await self._ws.send(json.dumps({
            "type": "tunnel.register",
            "payload": {
                "tunnel_id": self.tunnel_id,
                "tunnel_access_key": self.access_key,
            },
        }))
        print(
            f"  [avatar] tunnel connected "
            f"(tunnel_id={self.tunnel_id[:8]}..., local_port={self.ui_port})"
        )
        self._tasks.append(asyncio.create_task(self._read_loop()))
        self._tasks.append(asyncio.create_task(self._heartbeat()))

    def _local_url(self, path: str) -> str:
        # We only proxy /ui/* (or /) — everything else 404s naturally.
        if path.startswith("/ui"):
            path = path[len("/ui"):] or "/"
        return f"http://127.0.0.1:{self.ui_port}{path}"

    async def _read_loop(self):
        try:
            async for raw in self._ws:
                msg = json.loads(raw)
                msg_type = msg.get("type", "")
                if msg_type == "http.request":
                    asyncio.create_task(self._handle_http(msg))
                elif msg_type == "tunnel.error":
                    print(f"  [avatar tunnel error] {msg.get('message', '')}")
        except websockets.ConnectionClosed:
            if self._running:
                print("  [avatar] tunnel connection lost")

    async def _handle_http(self, msg: dict):
        # aiohttp is imported lazily — we know it's installed because we
        # only get here when --avatar was on at startup.
        import aiohttp  # type: ignore

        payload = msg.get("payload", msg)
        request_id = payload.get("request_id", "")
        method = payload.get("method", "GET")
        path = payload.get("path", "/")
        headers = payload.get("headers", {})
        body = payload.get("body", "")
        url = self._local_url(path)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method, url, headers=headers,
                    data=body if body else None,
                ) as resp:
                    text = await resp.text()
                    out_headers = {k: v for k, v in resp.headers.items()}
                    response = {
                        "type": "http.response",
                        "request_id": request_id,
                        "payload": {
                            "request_id": request_id,
                            "status": resp.status,
                            "headers": out_headers,
                            "body": text,
                        },
                    }
                    await self._ws.send(json.dumps(response))
        except Exception as exc:
            await self._ws.send(json.dumps({
                "type": "http.response",
                "request_id": request_id,
                "payload": {
                    "request_id": request_id,
                    "status": 502,
                    "headers": {"Content-Type": "text/plain"},
                    "body": f"Local server error: {exc}",
                },
            }))

    async def _heartbeat(self):
        # AgentCall's tunnel WS closes idle connections; send a ping every
        # 30 seconds to keep it open for the full call.
        while self._running and self._ws is not None:
            try:
                await asyncio.sleep(30)
                if self._ws is not None:
                    await self._ws.ping()
            except Exception:
                break

    async def close(self):
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass


# ----------------------------------------------------------------------------
# AgentCall lifecycle
# ----------------------------------------------------------------------------

def create_call(
    meet_url: str,
    bot_name: str,
    avatar: Optional[str],
    ui_port: int = 0,
) -> dict:
    """
    Create a call.

    Two visual modes, picked via the `avatar` parameter:

      avatar=None  -> mode "audio"
        Voice-only. The bot shows as a participant with no video tile content
        (just a name initial like every other audio-only Meet participant).
        Cheapest mode - around $0.01-$0.02 per minute on the dashboard.

      avatar=<template_name> -> mode "webpage-av"
        The bot streams a webpage as its camera feed, so participants see a
        live UI in the bot's video tile. Roughly 2-3x the per-minute cost of
        plain audio mode, so only switch it on when you want the visual.

    Built-in avatar templates (`ui-templates/<name>/` in this repo):

      "avatar"      Circular face with a voice-state-coloured ring
                    (idle=grey, listening=blue, thinking=purple,
                    speaking=green). Most polished. <- recommended default.
      "orb"         Pulsing animated orb that reacts to voice level.
      "ring"        Minimalist ring/halo.
      "voice-agent" Voice-agent dashboard look.
      "dashboard"   Info-dense panel with state + transcript.
      "blank"       Empty page (use as a placeholder).

    You can also serve your own webpage and pass `webpage_url` instead of
    `ui_template` - see scripts/node/bridge-visual.js for the pattern. That
    needs a tunnel to expose your local server to the cloud, so it's
    intentionally out of scope here.

    Other config below:
      direct = no GetSun (collaborative voice intelligence). WE drive what
        the bot says and when.
      transcription=True is required for us to receive transcript.final
        events at all.
    """
    payload = {
        "meet_url": meet_url,
        "bot_name": bot_name,
        "voice_strategy": "direct",
        "transcription": True,
    }
    if avatar:
        # FIXED 2026-05-21. The previous payload tried `mode=webpage-av` +
        # `ui_template=<name>` — neither field is accepted by POST /v1/calls.
        # The working contract (per scripts/python/bridge-visual.py and
        # references/guides/webpage-av.md):
        #   - mode = "webpage-av-screenshare" (covers webpage-av AND lets
        #     us add screenshare later without changing the API call)
        #   - ui_port = <local port hosting the template HTML>
        # The caller is responsible for starting the template HTTP server
        # AND a tunnel client (BridgeTunnelClient below) that proxies
        # FirstCall's fetches to that local port. Without both, the bot's
        # video tile will sit on the AgentCall placeholder.
        payload["mode"] = "webpage-av-screenshare"
        if ui_port:
            payload["ui_port"] = ui_port
    else:
        payload["mode"] = "audio"

    resp = requests.post(
        f"{API_BASE}/v1/calls",
        headers=HEADERS,
        json=payload,
    )
    # Print AgentCall's actual error body before raising. raise_for_status()
    # only includes the status code in its message, but the JSON body usually
    # contains the real reason (unknown field, missing field, bad mode, etc.).
    # Keeping this here makes future API-payload bugs diagnosable in one run
    # instead of needing a separate retry to inspect the response.
    if not resp.ok:
        print(f"  [agentcall {resp.status_code}] {resp.text}")
    resp.raise_for_status()
    return resp.json()


def end_call(call_id: str) -> None:
    """
    Force-end the call via the REST API.

    This is the belt-and-braces cleanup: even if the WebSocket dies or the
    user Ctrl+C's before we can send meeting.leave, DELETE /v1/calls/{id}
    stops AgentCall from billing for an orphaned bot.
    """
    try:
        requests.delete(f"{API_BASE}/v1/calls/{call_id}", headers=HEADERS, timeout=10)
        print("Call ended (cleanup).")
    except Exception as exc:
        print(f"Cleanup failed: {exc}")
        print(f"=> Check https://app.agentcall.dev and end call {call_id} manually.")


def _format_transcript_for_extraction(transcript: List[dict]) -> str:
    """
    Flatten the rolling transcript list into a plain-text dialogue we can
    paste into a Gemini prompt.

    We label moderator turns "Moderator" rather than the bot's display name
    so the LLM doesn't get confused about who's a participant. Lines are
    one-per-utterance to keep token usage tight.
    """
    lines: List[str] = []
    for turn in transcript:
        role = turn.get("role", "")
        text = turn.get("text", "").strip()
        if not text:
            continue
        if role == "participant":
            speaker = turn.get("speaker") or "Speaker"
            lines.append(f"{speaker}: {text}")
        else:
            lines.append(f"Moderator: {text}")
    return "\n".join(lines)


_EMPTY_SUMMARY = {
    "discussion_summary": "",
    "agenda": "",
    "outcome_met": "unknown",
    "outcome_note": "",
    "action_items_md": "",
}


def _format_state_action_items_md(action_items: List[dict]) -> str:
    """
    Format the cumulative action items captured in state into Slack
    mrkdwn, grouped by owner. Preferred over the post-hoc Gemini
    extraction because the items came directly from per-turn structured
    responses, so there's no hallucination risk.

    Output shape (one block per owner, separated by a blank line):

        *Alice*
        - pricing audit (due 2026-05-22)

        *Bob*
        - draft the launch comms (due 2026-05-27)
        - coordinate with the customer advisory board

    Owner ordering = first-mentioned-first (Python dict preserves
    insertion order on 3.7+), which mirrors meeting flow.

    Slack mrkdwn: bold = *single asterisks*, NOT **double**.
    """
    grouped: "dict[str, List[str]]" = {}
    for item in action_items:
        owner = item.get("owner") or "Unknown"
        task = item.get("task") or ""
        if not task:
            continue
        due = item.get("due")
        bullet = f"- {task} (due {due})" if due else f"- {task}"
        grouped.setdefault(owner, []).append(bullet)

    sections: List[str] = []
    for owner, bullets in grouped.items():
        sections.append(f"*{owner}*")
        sections.extend(bullets)
        sections.append("")  # blank line between owners

    # Drop trailing blank so the section doesn't end with a stray newline.
    while sections and not sections[-1]:
        sections.pop()
    return "\n".join(sections)


def extract_meeting_summary(transcript: List[dict], state: Optional[dict] = None) -> dict:
    """
    Build the post-call summary dict the Slack post needs.

    Two sources of truth:
      1. `state` (preferred). Block 2's moderator emits structured JSON
         every turn, so by end-of-call we already have agenda + action
         items tracked deterministically. Use them as ground truth - no
         hallucination risk, no token cost.
      2. Gemini one-shot fallback. If state is missing fields (e.g.
         agenda was never set, or no action items were ever captured),
         fall back to a single LLM call over the transcript text.

    Always returns the _EMPTY_SUMMARY shape (all four string keys
    populated, no Nones) so build_slack_summary can format it without
    None-checks.

    Why a separate one-shot call (not the chat session)?
      The chat session is configured with the moderator SYSTEM_PROMPT
      and emits the per-turn schema (speak / agenda_items /
      new_action_items / off_topic / follow_up). Asking it to emit a
      *different* JSON schema (agenda/outcome/action_items_md) would
      collide with its system instructions. We side-step the collision
      entirely by going direct to generate_content with our own
      one-shot config.
    """
    state = state or {}

    # Build the summary skeleton from state when we can.
    tracked_agenda: List[str] = list(state.get("agenda") or [])
    tracked_actions: List[dict] = list(state.get("action_items") or [])

    agenda_str = ", ".join(tracked_agenda) if tracked_agenda else ""
    action_items_md = _format_state_action_items_md(tracked_actions)

    # discussion_summary / outcome_met / outcome_note still need an LLM
    # verdict - we can't derive "what was discussed" or "did we cover
    # the agenda?" deterministically from state. Skip the call only if
    # there's literally nothing to summarize.
    if not transcript:
        return {
            "discussion_summary": "",
            "agenda": agenda_str,
            "outcome_met": "unknown",
            "outcome_note": "",
            "action_items_md": action_items_md,
        }

    body = _format_transcript_for_extraction(transcript)
    if not body:
        return {
            "discussion_summary": "",
            "agenda": agenda_str,
            "outcome_met": "unknown",
            "outcome_note": "",
            "action_items_md": action_items_md,
        }

    # Tell the model what we already know so it doesn't re-derive.
    known_facts = []
    if agenda_str:
        known_facts.append(f"The agreed agenda was: {agenda_str}.")
    if tracked_actions:
        known_facts.append(
            f"{len(tracked_actions)} action item(s) were captured during the call."
        )
    known_block = ("\n".join(known_facts) + "\n\n") if known_facts else ""

    # Note: action_items_md is deliberately NOT requested here - state
    # owns it deterministically and asking the LLM to regenerate the
    # bullet list was the main cause of summary-call truncation at
    # end-of-call.
    #
    # agenda IS requested here (overriding the terse state.agenda) -
    # state.agenda is optimized for in-call off-topic detection
    # (single phrase like "planning the migration") and reads vaguely
    # in the Slack post. The LLM has the whole transcript and can write
    # a more specific 1-2 sentence agenda that's actually useful to a
    # reader who didn't attend.
    prompt = (
        "Below is the transcript of a meeting that just ended. Respond with "
        "a single JSON object containing EXACTLY these keys (no extras):\n"
        " - agenda: a 1-2 sentence description (max 40 words) of what the "
        "meeting was about. Be SPECIFIC - mention the concrete subject "
        "(e.g. the system, project, or decision under discussion), not "
        "just the verb (\"planning X\"). A reader who didn't attend should "
        "understand the goal. Empty string only if no clear agenda was "
        "discussed.\n"
        " - discussion_summary: a 3-4 sentence narrative summary of what "
        "was actually discussed. Cover the main topics, key decisions or "
        "commitments, and any notable tangents or concerns. Write in past "
        "tense (\"The team discussed...\"). Should be LONGER than the "
        "agenda field.\n"
        " - outcome_met: one of \"yes\", \"no\", \"partial\", or \"unknown\" "
        "indicating whether the agenda was achieved.\n"
        " - outcome_note: a single sentence (max 25 words) justifying the "
        "outcome_met verdict.\n\n"
        "Reply with ONLY the raw JSON object. No preamble, no closing "
        "remarks, no markdown code fences.\n\n"
        f"{known_block}"
        "Transcript:\n"
        f"{body}"
    )

    raw = ""
    try:
        # One-shot call via generate_content - bypasses the chat session
        # entirely, so the moderator's per-turn JSON schema doesn't
        # collide with the summary's schema.
        response = _gemini_client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                max_output_tokens=2000,
                temperature=0.2,
            ),
        )
        raw = (response.text or "").strip()
        if not raw:
            print("  [summary extract error] Gemini returned empty response.")
            return {
                "discussion_summary": "",
                "agenda": agenda_str,
                "outcome_met": "unknown",
                "outcome_note": "",
                "action_items_md": action_items_md,
            }
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"  [summary parse error] {exc}")
        print(f"  [summary raw output] {raw[:300]!r}{'...' if len(raw) > 300 else ''}")
        return {
            "discussion_summary": "",
            "agenda": agenda_str,
            "outcome_met": "unknown",
            "outcome_note": "",
            "action_items_md": action_items_md,
        }
    except Exception as exc:
        print(f"  [summary extract error] {exc}")
        return {
            "discussion_summary": "",
            "agenda": agenda_str,
            "outcome_met": "unknown",
            "outcome_note": "",
            "action_items_md": action_items_md,
        }

    # Agenda preference REVERSED vs earlier: LLM-derived wins now.
    # state.agenda is captured per-turn for in-call off-topic detection
    # and tends to be a single terse phrase ("planning the migration");
    # the LLM has the whole transcript and the dedicated "1-2 sentence,
    # be specific" prompt and produces something more useful for a
    # Slack reader who didn't attend. We still fall back to state's
    # version if the LLM returned an empty string.
    llm_agenda = str(data.get("agenda", "")).strip()
    agenda_out = llm_agenda or agenda_str

    return {
        "discussion_summary": str(data.get("discussion_summary", "")).strip(),
        "agenda": agenda_out,
        "outcome_met": str(data.get("outcome_met", "unknown")).strip().lower(),
        "outcome_note": str(data.get("outcome_note", "")).strip(),
        "action_items_md": action_items_md,
    }


# ----------------------------------------------------------------------------
# Block 4b: post-call Jira ticket creation
# ----------------------------------------------------------------------------


def load_team_roster(path: str) -> dict:
    """
    Load examples/meeting-moderator-v2/team_roster.yaml (or whatever
    --jira-roster pointed at) into a {lower_first_name: email} dict.

    Why first-name-only and lower-cased: action item `owner` values are
    captured by the LLM from speech. STT gives us "Yaman" or "yaman"
    or sometimes "YAMAN", never a stable "Yaman Altareh". Normalising
    both sides on first-name-lower-case is the simplest matcher that
    survives the noise.

    Fails OPEN: a missing or unreadable file just means an empty dict,
    which in turn means every owner falls back to the call host. We
    print a one-line warning so the user knows roster resolution is
    off, but the call doesn't crash.
    """
    if not path or not os.path.exists(path):
        return {}
    try:
        # PyYAML is already pinned in requirements.txt (originally for
        # Block 3). Importing lazily so the module still loads if
        # someone runs without installing requirements.
        import yaml  # type: ignore
    except ImportError:
        print(f"  [roster] PyYAML not installed — falling back to host email for all owners.")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as exc:
        print(f"  [roster] could not read {path}: {exc} — falling back to host email.")
        return {}
    if not isinstance(raw, dict):
        print(f"  [roster] {path} did not parse as a name->email mapping. Skipping.")
        return {}
    # Normalise keys to lower-case first-name so the matcher is robust
    # against transcript casing.
    return {str(k).strip().lower(): str(v).strip() for k, v in raw.items() if v}


def _resolve_host_display_name(host_email: str, roster: dict) -> str:
    """
    Reverse-lookup the call host's email -> first-name display.

    Used by the dry-run loop to attribute no-prefix lines to the
    actual human running the script instead of the generic "You"
    placeholder. Falls back to "You" if either the email or the
    roster is missing, or if no roster entry matches the email.

    Example: ATLASSIAN_EMAIL='daniyal.sabir@shypple.com' +
    roster={'daniyal': 'daniyal.sabir@shypple.com', ...}
    returns 'Daniyal'.

    Matching is case-insensitive on both sides.
    """
    if not host_email or not roster:
        return "You"
    host_email_lc = host_email.lower()
    for first_name_lower, email in roster.items():
        if str(email).lower() == host_email_lc:
            return first_name_lower.capitalize()
    return "You"


def _resolve_owner_email(owner: str, roster: dict, host_email: str) -> str:
    """
    Map an action-item `owner` string to a work email.

    The roster keys are lower-cased first names. We split the owner on
    whitespace and try the first token; if that misses, we return the
    host email so the ticket still gets created (just assigned to the
    call host instead of the captured owner). That keeps the demo
    flow visible even for owners we don't have roster entries for.
    """
    if owner:
        first = owner.strip().split()[0].lower() if owner.strip() else ""
        if first and first in roster:
            return roster[first]
    return host_email


# ----------------------------------------------------------------------------
# Block 4 (mid-meeting Jira intent): regex-based detection + handler
# ----------------------------------------------------------------------------
#
# Design choice (locked in 2026-05-20 with the user): NO LLM for the
# intent-extraction step. Pure regex/keyword detection. The LLM
# (Gemini Flash) still runs for the normal moderator turn — we just
# bypass it when a Jira intent is matched, so the action is
# deterministic and doesn't compete with the moderator's per-turn
# JSON schema. Trade-off: brittle on creative phrasing. Demo-
# acceptable; the trigger phrase list below covers the common forms.


# Phrases that mean "I want a Jira ticket created". Case-insensitive.
# Each pattern must consume the verb + the object ("ticket"/"jira"/
# "task") together so generic "make a list" type sentences don't
# trigger.
_JIRA_CREATE_TRIGGERS = re.compile(
    r"\b("
    r"create|make|add|log|file|open|raise"
    r")\s+(?:a\s+|an\s+)?"
    r"(?:new\s+|jira\s+|shp\s+)?"
    r"(ticket|jira|task|issue|story)\b",
    re.IGNORECASE,
)

# ---- JQL search triggers (Block 4f, 2026-05-21) ---------------------------
# Phrases that mean "tell me how many / which tickets match this criterion".
# Today the supported queries are:
#   - count of open tickets on a team's board
#   - list of open tickets on a team's board
# Anything else falls through and the bot politely declines.

# A "count" phrasing: "how many open jira tickets are there for team stack"
# Must contain a quantity word AND a ticket noun AND the word "open" (or
# similar status hint). We treat "search Jira for X" as a separate path.
#
# Whitespace gotcha: the quantity-word alternatives must NOT consume
# the trailing space (e.g. "count(?:\s+of)?" not "count\s+(?:of\s+)?")
# so the single \s+ after the group always has a space to bind to,
# even when the longer "of"-suffixed form fires.
_JIRA_SEARCH_COUNT_TRIGGER = re.compile(
    r"\b(?:"
    r"how\s+many|count(?:\s+of)?|number\s+of|total(?:\s+number)?(?:\s+of)?"
    r")\s+"
    r"(?:the\s+)?"
    r"(?:open\s+|active\s+|in[\-\s]?progress\s+|to[\-\s]?do\s+|"
    r"unresolved\s+|outstanding\s+)?"
    r"(?:jira\s+)?(?:tickets?|issues?|tasks?|stories|bugs?)\b",
    re.IGNORECASE,
)

# A "list" phrasing: "what are the open tickets for team core" / "show me
# the open tickets on the stack board" / "list open tickets for stack"
# The optional "(?:the\s+)?" between verb and status lets phrasings like
# "list the open tickets" through (the "the" was previously rejected by
# the status group).
_JIRA_SEARCH_LIST_TRIGGER = re.compile(
    r"\b(?:"
    r"list|show(?:\s+me)?|what(?:\s+are)?(?:\s+the)?|"
    r"give\s+me|tell\s+me(?:\s+about)?(?:\s+the)?"
    r")\s+"
    r"(?:the\s+)?"
    r"(?:open\s+|active\s+|in[\-\s]?progress\s+|to[\-\s]?do\s+|"
    r"unresolved\s+|outstanding\s+|all\s+)?"
    r"(?:jira\s+)?(?:tickets?|issues?|tasks?|stories|bugs?)\b",
    re.IGNORECASE,
)

# Status filter phrases — used to scope a search to a particular workflow
# state. Order matters: more specific first.
_JIRA_SEARCH_STATUS_PATTERNS = [
    (re.compile(r"\bin[\-\s]?progress\b", re.IGNORECASE),
     'statusCategory = "In Progress"'),
    (re.compile(r"\bto[\-\s]?do\b", re.IGNORECASE),
     'statusCategory = "To Do"'),
    (re.compile(r"\b(?:done|closed|resolved|completed|finished)\b", re.IGNORECASE),
     'statusCategory = Done'),
    (re.compile(r"\b(?:open|active|unresolved|outstanding)\b", re.IGNORECASE),
     "statusCategory != Done"),
]

# Team name resolution for JQL — maps spoken team names to the canonical
# board name fragment used as a JQL `component`. We rely on Shypple's
# convention: components are named after teams (STACK, CORE, UX, etc.).
_JIRA_TEAM_TRIGGER = re.compile(
    r"\bteam\s+([a-z][a-z\-]+)\b|\bfor\s+(?:the\s+)?([a-z][a-z\-]+)\s+team\b|"
    r"\bon\s+(?:the\s+)?([a-z][a-z\-]+)\s+(?:team|board)\b",
    re.IGNORECASE,
)

# Per-person assignee filter. Captures a first name after a recognised
# preposition / verb so we can scope the search to a single roster
# member. We DELIBERATELY require a preposition or possessive form so
# random capitalised words inside the utterance don't fire (e.g.
# "Mod, how many open tickets" should not match "Mod"). The handler
# validates the captured name against the roster — unknown names
# fall back to a "no breakdown / not found" response.
_JIRA_SEARCH_PERSON_PATTERN = re.compile(
    # "assigned to X"
    r"\bassigned\s+to\s+([A-Z][a-zA-Z]+)\b"
    # "does X have" / "do X have"
    r"|\bdo(?:es)?\s+([A-Z][a-zA-Z]+)\s+have\b"
    # "X has how many tickets" / "X has tickets" — possessive-ish.
    # The (?:[^.,!?]{0,40}?\s+)? bridge lets phrases like
    # "Alexander has how many open" connect "has" -> "tickets" up
    # to ~40 chars of intermediate filler, without crossing a clause
    # boundary (no .,!?).
    r"|\b([A-Z][a-zA-Z]+)\s+(?:has|owns)\s+"
    r"(?:[^.,!?]{0,40}?\s+)?"
    r"(?:tickets?|issues?|tasks?)\b"
    # "X's tickets" — straight or curly apostrophe.
    r"|\b([A-Z][a-zA-Z]+)['’]s\s+(?:open\s+|in[\-\s]?progress\s+|"
    r"to[\-\s]?do\s+|active\s+)?"
    r"(?:tickets?|issues?|tasks?)\b"
    # "for X" — capture the name AFTER "for", but NOT when it's
    # really "for team X" / "for X team" / "for X to own/handle"
    # (those are team filters and create-assignee phrasings).
    r"|\bfor\s+([A-Z][a-zA-Z]+)\b(?!\s+team)(?!\s+to\s+(?:own|handle|do))"
    # "by X" — same shape, with a negative lookahead RIGHT AFTER "by "
    # so the captured name itself can't be a weekday / due-date word.
    # The previous lookahead was on the wrong side (checked what came
    # after the name), letting "by Friday" capture "Friday" as a person.
    r"|\bby\s+(?!(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|"
    r"Sunday|EOD|end|next|tomorrow|today|noon|midnight)\b)"
    r"([A-Z][a-zA-Z]+)\b"
    # "owned by X" — explicit form, no ambiguity.
    r"|\bowned\s+by\s+([A-Z][a-zA-Z]+)\b"
    # "tickets of X" / "issues of X" — slightly formal but heard
    # in practice.
    r"|\b(?:tickets?|issues?|tasks?)\s+of\s+([A-Z][a-zA-Z]+)\b",
    # No re.IGNORECASE — capitalisation is part of how we tell a name
    # from a verb. STT capitalises proper nouns reliably.
)

# Breakdown trigger — turns the search into a per-person enumeration.
# These phrases mean "aggregate across the whole roster".
_JIRA_SEARCH_BREAKDOWN_TRIGGER = re.compile(
    r"\bper\s+(?:person|user|assignee|employee|member|teammate|head)\b"
    r"|\bby\s+(?:person|user|assignee|employee|member|teammate)\b"
    r"|\beach\s+(?:person|user|assignee|teammate|member)\b"
    r"|\bbreakdown\s+(?:by|of|per)?\b"
    r"|\bsplit\s+by\s+(?:person|assignee)\b",
    re.IGNORECASE,
)

# Sprint scope for JQL searches. We map natural-language sprint phrases
# to Atlassian's built-in JQL sprint functions (no team-specific sprint
# ID lookup needed — `openSprints()` returns the active sprint(s) on
# whatever boards contain the issue, which when combined with our
# `component = "TEAM"` clause naturally scopes to that team's sprint).
_JIRA_SEARCH_SPRINT_PATTERNS = [
    (
        re.compile(
            r"\b(?:in\s+)?(?:this|the|current|active|present|ongoing|"
            r"running)\s+sprint\b",
            re.IGNORECASE,
        ),
        "sprint in openSprints()",
        "in the current sprint",
    ),
    (
        re.compile(
            r"\b(?:in\s+)?(?:next|upcoming|future)\s+sprint\b",
            re.IGNORECASE,
        ),
        "sprint in futureSprints()",
        "in the next sprint",
    ),
    (
        re.compile(
            r"\b(?:in\s+)?(?:any|some)\s+sprint\b|\bin\s+a\s+sprint\b",
            re.IGNORECASE,
        ),
        "sprint is not EMPTY",
        "in any sprint",
    ),
    (
        re.compile(
            r"\b(?:not\s+in|outside)\s+(?:a|any|the\s+current)\s+sprint\b|"
            r"\bbacklog\b",
            re.IGNORECASE,
        ),
        "sprint is EMPTY",
        "in the backlog",
    ),
]


def _detect_jira_search_intent(text: str, project_key: str) -> Optional[dict]:
    """
    Detect a JQL search intent (count or list of tickets matching simple
    criteria). Returns a dict with keys:
      kind        — "count" or "list"
      jql         — the JQL string to execute
      description — human-readable description of what we're searching
                    (used in the bot's read-back)
    Returns None when no search intent fires.
    """
    if not text:
        return None
    is_count = bool(_JIRA_SEARCH_COUNT_TRIGGER.search(text))
    is_list = bool(_JIRA_SEARCH_LIST_TRIGGER.search(text)) and not is_count
    # Breakdown trigger ("per person", "by assignee", "breakdown") and
    # the person pattern ("Alexander's tickets", "tickets assigned to
    # Yaman") can themselves be the primary search signal — neither
    # requires a count / list trigger to make sense. Detect them
    # early so we don't bail out below.
    has_breakdown_signal = bool(_JIRA_SEARCH_BREAKDOWN_TRIGGER.search(text))
    has_person_signal = bool(_JIRA_SEARCH_PERSON_PATTERN.search(text))
    if not (is_count or is_list or has_breakdown_signal or has_person_signal):
        return None
    # When breakdown / person fires without an explicit count/list
    # verb, default to count (cheaper, succinct, matches the natural
    # "how many" reading of phrases like "tickets per person").
    if not (is_count or is_list):
        is_count = True

    # Status clause — default to "open" (statusCategory != Done) so
    # questions like "how many tickets are there" still produce a
    # useful number.
    status_clause = "statusCategory != Done"
    status_label = "open"
    for pat, clause in _JIRA_SEARCH_STATUS_PATTERNS:
        if pat.search(text):
            status_clause = clause
            # Label is the literal first matched word, lower-cased.
            m = pat.search(text)
            if m:
                status_label = m.group(0).lower()
            break

    # Team / component clause — optional. The component name is uppercased
    # because that's how SHP stores them.
    team_clause = ""
    team_label = ""
    m = _JIRA_TEAM_TRIGGER.search(text)
    if m:
        team_name = (m.group(1) or m.group(2) or m.group(3) or "").strip()
        if team_name:
            team_label = team_name.lower()
            # Use case-insensitive component match. Backslash-escape any
            # JQL specials defensively (we know team names are
            # alphanumeric in practice, but cheap insurance).
            esc = team_name.replace('"', '\\"')
            team_clause = f'component = "{esc.upper()}"'

    # Sprint scope — optional. Recognises "in this sprint" / "current
    # sprint" / "next sprint" / "in the backlog" etc. and maps to one
    # of Atlassian's JQL sprint helper functions.
    sprint_clause = ""
    sprint_label = ""
    for pat, clause, label in _JIRA_SEARCH_SPRINT_PATTERNS:
        if pat.search(text):
            sprint_clause = clause
            sprint_label = label
            break

    # Per-person assignee scope. Two flavours:
    #   - breakdown: enumerate the whole roster, one count each
    #   - single-person: filter by one captured first-name
    # Breakdown takes priority — "tickets per person for Alexander"
    # is ambiguous; the breakdown phrasing is the clearer signal.
    breakdown = bool(_JIRA_SEARCH_BREAKDOWN_TRIGGER.search(text))
    assignee_name = ""
    if not breakdown:
        m_person = _JIRA_SEARCH_PERSON_PATTERN.search(text)
        if m_person:
            assignee_name = next(
                (g for g in m_person.groups() if g), ""
            ).strip()

    project_clause = f'project = {project_key}' if project_key else ""

    # Build the BASE JQL (project + status + team + sprint). The
    # assignee clause is added later, either once (single-person)
    # or per-iteration (breakdown). Keeping it out of the base lets
    # the breakdown handler reuse one base string with N assignee
    # tails.
    base_clauses = [c for c in (
        project_clause, status_clause, team_clause, sprint_clause
    ) if c]
    base_jql = " AND ".join(base_clauses)
    jql = base_jql + " ORDER BY updated DESC"

    desc_team = f" for team {team_label}" if team_label else ""
    desc_sprint = f" {sprint_label}" if sprint_label else ""
    description = f"{status_label} tickets{desc_team}{desc_sprint}"

    # Kind: count / list / breakdown. Breakdown wins if its trigger
    # fired; otherwise count vs list as before.
    if breakdown:
        kind = "breakdown"
    elif is_count:
        kind = "count"
    else:
        kind = "list"

    return {
        "kind": kind,
        "jql": jql,
        "base_jql": base_jql,  # used by breakdown to compose per-person queries
        "description": description,
        "team_label": team_label,
        "sprint_label": sprint_label,
        "assignee_name": assignee_name,
    }

# Pattern for the "assigned to X" / "for X" / "give it to X" fragment
# inside a creation utterance. We strip this out of the title and use
# the captured name (group 1) as the owner. Word boundaries on both
# sides so we don't eat random "to" inside the title.
_JIRA_ASSIGNEE_PATTERNS = [
    # "assign to X" / "assign it to X" / "assigned to X" / "reassign
    # SHP-101 to X" — lazy gap between the verb and "to X" so the
    # key (or "it") can sit in between without breaking the match.
    re.compile(
        r"\b(?:re)?assign(?:ed)?\s+(?:[^,.!?]*?\s+)?to\s+([A-Za-z][a-zA-Z]+)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bgive\s+(?:it\s+)?to\s+([A-Za-z][a-zA-Z]+)\b", re.IGNORECASE),
    re.compile(r"\bfor\s+([A-Za-z][a-zA-Z]+)\b\s*(?:to\s+(?:own|handle|do))", re.IGNORECASE),
    # Natural-speech forms the user used in dry-tests:
    #   "the assignee is Alexander"
    #   "assignee should be Yaman"
    #   "make the assignee Alexander"
    #   "assignee to Joas"
    re.compile(
        r"\b(?:the\s+)?assignee\s+"
        r"(?:is|to|=|:|should\s+be|will\s+be|gets\s+to\s+be)\s+"
        r"([A-Za-z][a-zA-Z]+)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bmake\s+(?:the\s+)?assignee\s+([A-Za-z][a-zA-Z]+)\b",
        re.IGNORECASE,
    ),
    # "owner is X" / "owned by X" — same field, different word
    re.compile(
        r"\b(?:the\s+)?owner\s+(?:is|to|should\s+be|will\s+be)\s+"
        r"([A-Za-z][a-zA-Z]+)\b",
        re.IGNORECASE,
    ),
]

# "by Friday" / "by 2026-05-22" / "due Friday" / "due next Wednesday"
# The lookbehind excludes "reported by X" / "report by X" — those
# refer to the reporter field, not the due date.
_JIRA_DUE_PATTERN = re.compile(
    r"(?<!report\s)(?<!reported\s)\b(?:by|due)\s+(.+?)(?:[.,]|\s+(?:assign|for|to)\b|$)",
    re.IGNORECASE,
)

# Confirmation vocabulary. Permissive list per the user's choice.
_CONFIRM_WORDS = {
    "yes", "yeah", "yep", "yup", "confirm", "confirmed", "go ahead",
    "do it", "do that", "correct", "right", "sure", "ok", "okay",
    "proceed",
}
_CANCEL_WORDS = {
    "no", "nope", "cancel", "skip", "nevermind", "never mind", "don't",
    "stop", "abort",
}

# Jira issue key regex — Shypple's project is SHP, but we accept any
# uppercase project key so the same code path works if the demo ever
# pivots to a different project. Word boundaries on both sides so
# "SHP-1" inside a URL still matches but "SHP-1234abc" doesn't.
_JIRA_KEY_PATTERN = re.compile(r"\b([A-Z][A-Z0-9]+-\d+)\b")


# STT often renders dictated Jira keys as spelled-out digits, e.g.
# "shp one seven seven six seven" instead of "SHP-17767". The key
# regex above won't match the spelled-out form, so we pre-normalise
# the utterance before any Jira intent detection runs.
_DIGIT_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}
_DIGIT_WORDS_RE = "|".join(_DIGIT_WORDS.keys())

# Word-form numbers we accept for story points and other small counts.
# Caps at 20 — story points beyond that are vanishingly rare and STT
# garbles on bigger words like "twenty-three" anyway.
_NUMBER_WORD_TO_INT = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
}
_NUMBER_WORDS_RE = "|".join(_NUMBER_WORD_TO_INT.keys())

# Filler tokens that STT happily transcribes verbatim. They wedge
# themselves between an extraction keyword and its value ("priority
# of this uh ticket to high") and break otherwise-strict regexes.
# Stripping them is safe because they carry no semantic load.
_FILLER_PATTERN = re.compile(
    r"\b(?:uh+|um+|ah+|er+|hmm+|mmm+|like|sort\s+of|kind\s+of|you\s+know)\b",
    re.IGNORECASE,
)


def _strip_fillers(text: str) -> str:
    """
    Remove disfluencies / filler phrases ("uh", "um", "you know")
    and collapse the resulting double-spaces. Pure preprocessing —
    no semantic content is lost.
    """
    if not text:
        return text
    cleaned = _FILLER_PATTERN.sub(" ", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


# Anaphora — phrases that mean "the ticket we were just talking
# about". Triggers an implicit-key injection in detect_jira_intent
# so the speaker doesn't have to dictate the SHP-N key every time.
# Requires that state["last_jira_key"] is set (i.e. there's been at
# least one Jira action this call) for the injection to fire.
_JIRA_ANAPHORA_PATTERN = re.compile(
    r"\b(?:"
    # "this/that/the [new|last|latest|newest|recent|same|previous|above]
    # ticket/issue/task/jira/story/one"
    r"(?:this|that|the(?:\s+(?:new|newest|last|latest|most\s+recent|recent|"
    r"same|previous|above))?)"
    r"\s+(?:ticket|jira|issue|task|story|one)"
    # "the/that ticket you/we/I (just) created/made/opened/filed/added/raised/
    # logged/set up/filed/built/spun up"
    r"|(?:the|that)?\s*(?:ticket|jira|issue|task|story|one)"
    r"\s+(?:you|we|i)\s+(?:just\s+|recently\s+)?"
    r"(?:created|made|opened|filed|added|raised|logged|set\s+up|"
    r"spun\s+up|built|put\s+together)"
    # "in/on/for/about the ticket you just created" — same as above
    # but lets the speaker put a preposition in front.
    r"|(?:in|on|for|about|with|to)\s+(?:the\s+)?"
    r"(?:ticket|jira|issue|task|story|one)\s+"
    r"(?:you|we|i)\s+(?:just\s+|recently\s+)?"
    r"(?:created|made|opened|filed|added|raised|logged|set\s+up|"
    r"spun\s+up|built|put\s+together)"
    r")\b",
    re.IGNORECASE,
)


def _resolve_implicit_key(text: str, last_key: str) -> str:
    """
    Return `last_key` if `text` contains an anaphoric reference
    ("this ticket", "the new ticket you just created") AND no
    explicit Jira key was already dictated. Returns "" otherwise.

    Caller appends the returned key to the utterance so downstream
    regexes find it exactly as if the speaker had dictated it.
    """
    if not last_key:
        return ""
    if _JIRA_KEY_PATTERN.search(text):
        return ""
    if _JIRA_ANAPHORA_PATTERN.search(text):
        return last_key
    return ""


def _normalize_spoken_jira_keys(text: str, project_key: str = "") -> str:
    """
    Rewrite spelled-out Jira keys to canonical PROJ-N form.

    Handles the dictation patterns STT produces:
      "shp one seven seven six seven"      -> "SHP-17767"
      "shp dash one seven seven six seven" -> "SHP-17767"
      "SHP one seven seven six seven"      -> "SHP-17767"

    When `project_key` is set we anchor on that exact prefix to avoid
    false positives like "TV one two" -> "TV-12" in casual speech.
    When it's empty we fall back to any 2-5 letter prefix; that's a
    last-resort path and unlikely to fire in practice.

    A run of 2+ digit-words is required so single mentions ("I have
    one ticket") don't trigger.
    """
    if not text:
        return text

    if project_key:
        # Build a case-insensitive matcher for the literal project key.
        # We allow optional "dash"/"-"/whitespace between key and digits.
        proj_alt = re.escape(project_key)
    else:
        # Generic 2-5 letter prefix. Higher false-positive risk; only
        # reached when the caller didn't pass a project_key.
        proj_alt = r"[A-Za-z]{2,5}"

    pattern = re.compile(
        r"\b(" + proj_alt + r")"
        r"(?:\s*(?:dash|-)\s*|\s+)"
        r"((?:(?:" + _DIGIT_WORDS_RE + r")\s+){1,}(?:" + _DIGIT_WORDS_RE + r"))"
        r"\b",
        re.IGNORECASE,
    )

    def _replace(m: "re.Match[str]") -> str:
        proj = m.group(1).upper()
        digits = "".join(
            _DIGIT_WORDS.get(w.lower(), "")
            for w in re.findall(r"[A-Za-z]+", m.group(2))
        )
        if not digits:
            return m.group(0)
        return f"{proj}-{digits}"

    return pattern.sub(_replace, text)


# Verbs that signal "I want to modify an existing ticket". When one of
# these appears in a bot-addressed utterance ALONGSIDE a SHP-N key, we
# treat it as edit (Phase 4d) rather than lookup. The "set" verb is
# the most common natural form ("set SHP-101 to high priority").
_JIRA_EDIT_TRIGGERS = re.compile(
    # "add" is here AND in the create triggers — they don't collide
    # because the create pattern additionally requires a ticket-noun
    # (ticket/jira/task/issue/story) right after the verb. "add SHP-N"
    # has no noun, so edit wins. "add a ticket" has the noun, so
    # create wins via the earlier priority check in detect_jira_intent.
    r"\b(update|edit|change|set|modify|reassign|move|add|put)\b",
    re.IGNORECASE,
)


def _is_bot_addressed(text: str, bot_name: str) -> bool:
    """
    Heuristic for "this utterance is talking to the bot".

    Looks for the bot name as a standalone token. We split on word
    boundaries and lower-case both sides. The STT engine has been
    observed to garble names ("Moaaz" -> "Moas" / "Moores"), so for
    the demo we ALSO accept a small alias set covering the common
    bot names we've used in testing. Demo-safe; cheap; not perfect.
    """
    if not text or not bot_name:
        return False
    text_lc = text.lower()
    aliases = {bot_name.lower(), "mod", "moderator", "juno"}
    for alias in aliases:
        # \b<alias>\b — using a small regex per alias keeps the
        # check fast and avoids partial-word matches.
        if re.search(rf"\b{re.escape(alias)}\b", text_lc):
            return True
    return False


def _contains_any_phrase(text_lc: str, phrases) -> bool:
    """
    True if any of `phrases` appears as a standalone phrase in
    `text_lc` (already lower-cased). Multi-word phrases like "go
    ahead" are matched literally (with leading/trailing word
    boundaries) rather than as separate tokens.
    """
    for p in phrases:
        if re.search(rf"\b{re.escape(p)}\b", text_lc):
            return True
    return False


# Patterns for "name it X" / "called X" / "titled X" / "summary X"
# — when one of these fires, we use X as the EXACT title and ignore
# the rest of the surrounding chatter. Captures everything up to a
# comma, period, "assigned"/"due"/"by" marker, or end-of-string.
_JIRA_EXPLICIT_TITLE_PATTERNS = [
    re.compile(
        r"\b(?:name(?:d)?\s+it|call(?:ed)?\s+it|titled?|summary)\s+"
        r"['\"]?([^,.!?]+?)['\"]?"
        r"(?:[,.!?]|\s+(?:assign(?:ed)?|due|by|for|with|priority|to)\b|$)",
        re.IGNORECASE,
    ),
]


# Patterns for retitling an EXISTING ticket. Distinct from the create
# patterns because the verb (set/change/update/rename) sits BEFORE
# the title noun. Quoted form is preferred (high confidence); the
# unquoted form is more permissive and captures up to a sentence
# terminator. `.*?` between the title-noun and the connector lets
# "set the title of SHP-X to Y" work without an inflexible "of
# ticket" sub-clause.
_JIRA_EDIT_TITLE_PATTERNS = [
    # Quoted form: ...title|summary|name <anything> to "X"
    re.compile(
        r"\b(?:set|change|update|modify|rename(?:\s+it)?)\s+"
        r"(?:the\s+)?"
        r"(?:title|summary|name)\b"
        r"(?:\s+.*?)?\s+"
        r"(?:to|=|:|should\s+be|is)\s+"
        r"['\"]([^'\"]+)['\"]",
        re.IGNORECASE,
    ),
    # Unquoted form: ...title|summary|name <anything> to X<end>
    re.compile(
        r"\b(?:set|change|update|modify|rename(?:\s+it)?)\s+"
        r"(?:the\s+)?"
        r"(?:title|summary|name)\b"
        r"(?:\s+.*?)?\s+"
        r"(?:to|=|:|should\s+be|is)\s+"
        r"(.+?)\s*[.!?]?\s*$",
        re.IGNORECASE,
    ),
    # Direct "rename it/SHP-X to Y" form (no title-noun)
    re.compile(
        r"\brename\s+(?:it\s+|[A-Z][A-Z0-9]+-\d+\s+)?(?:to|as)\s+"
        r"['\"]?(.+?)['\"]?\s*[.!?]?\s*$",
        re.IGNORECASE,
    ),
]


def _extract_edit_title(text: str) -> str:
    """
    Return the new title for an edit utterance, or empty string.

    Tries the quoted pattern first, then the unquoted form. The
    quoted form is high-confidence ("set title to 'X'"); the
    unquoted form is permissive and trusts the confirmation gate
    to catch garbled captures.
    """
    for pat in _JIRA_EDIT_TITLE_PATTERNS:
        m = pat.search(text)
        if m:
            captured = m.group(1).strip().rstrip(".,;:?!").strip("'\"")
            # Defence in depth: if the anaphora-injection path
            # appended a Jira key, strip it so it doesn't end up in
            # the title. (Today's code prepends, but a trailing
            # injection would silently corrupt the title.)
            captured = re.sub(
                r"\s+[A-Z][A-Z0-9]+-\d+\s*$", "", captured
            ).strip()
            # Guard against trivial / placeholder captures like "the
            # following title" — the speaker meant to provide the
            # title in a separate turn, which we don't support.
            # Returning empty here makes the handler fall back to
            # "I didn't catch what to change", prompting them to
            # re-state with the title inline.
            if not captured or captured.lower() in {
                "the following title", "the following", "this",
                "that", "the new title", "a new title",
            }:
                return ""
            return captured
    return ""

# Self-reference tokens that should resolve to the speaker rather
# than being passed through verbatim as a Jira owner. "i" is included
# but only matched as a standalone token so it doesn't fire on every
# random word.
_SELF_REFERENCE_TOKENS = {"me", "myself", "i"}


# Priority words Jira Cloud's defaults expose. Three orderings show
# up in natural speech:
#   1. "<value> priority"         e.g. "high priority"
#   2. "priority [is/to] <value>" e.g. "priority high", "priority to high"
#   3. "priority of <stuff> [to/is] <value>" — the speaker references
#      the ticket between the noun and the value:
#      "priority of this ticket SHP-17767 to high".
# Alt 3 caps the intermediate text at 60 chars and forbids sentence-
# terminators so it doesn't bridge across clauses.
_JIRA_PRIORITY_PATTERN = re.compile(
    r"\b(highest|high|medium|low|lowest)\s+priority\b|"
    r"\bpriority\s+(?:is\s+|to\s+|set\s+to\s+|should\s+be\s+|will\s+be\s+|=\s*|:\s*)?(highest|high|medium|low|lowest)\b|"
    r"\bpriority\s+of\s+[^,.!?]{1,60}?\s+(?:to|is|should\s+be|will\s+be|set\s+to)\s+(highest|high|medium|low|lowest)\b",
    re.IGNORECASE,
)


def _extract_priority(text: str) -> str:
    """
    Return the priority name (capitalised, Jira-API ready) if any of
    the priority phrases appear in `text`. Empty string when nothing
    matches.
    """
    m = _JIRA_PRIORITY_PATTERN.search(text)
    if not m:
        return ""
    # Three alternatives; exactly one capture group fires per match.
    value = m.group(1) or m.group(2) or m.group(3) or ""
    return value.strip().capitalize()


# Story points: "5 points" / "5 story points" / "2.5 points" / "with
# 8 story points" / "set story points to 5". We accept fractional
# values (Jira allows them; some teams use 0.5 for trivial work).
# Word-form numbers ("three points") are also accepted — STT often
# transcribes spoken digits as words.
_JIRA_STORY_POINTS_PATTERNS = [
    # Numeric, number-first: "5 story points" / "5 points". The (?<!-)
    # lookbehind blocks the digits half of a SHP-12345 key from
    # being captured as a points value.
    re.compile(r"(?<!-)\b(\d+(?:\.\d+)?)\s+(?:story\s+)?points?\b", re.IGNORECASE),
    # Numeric, keyword-first: "story points to 5" / "points = 5"
    re.compile(
        r"\b(?:story\s+)?points?\s+(?:to|of|is|are|=|:)\s+(\d+(?:\.\d+)?)\b",
        re.IGNORECASE,
    ),
    # Word-form, number-first: "three story points" / "five points"
    re.compile(
        r"\b(" + _NUMBER_WORDS_RE + r")\s+(?:story\s+)?points?\b",
        re.IGNORECASE,
    ),
    # Word-form, keyword-first: "story points to three"
    re.compile(
        r"\b(?:story\s+)?points?\s+(?:to|of|is|are|=|:)\s+(" + _NUMBER_WORDS_RE + r")\b",
        re.IGNORECASE,
    ),
]


def _extract_story_points(text: str) -> Optional[float]:
    """
    Return the number of story points mentioned in `text`, or None.

    Accepts both digit form ("3 story points") and word form
    ("three story points") via the _NUMBER_WORD_TO_INT lookup.
    Returns a float so Jira's customfield accepts it directly.
    """
    for pat in _JIRA_STORY_POINTS_PATTERNS:
        m = pat.search(text)
        if m:
            raw = m.group(1).lower()
            if raw in _NUMBER_WORD_TO_INT:
                return float(_NUMBER_WORD_TO_INT[raw])
            try:
                return float(raw)
            except (ValueError, TypeError):
                continue
    return None


# Epic / parent link. Several natural phrasings:
#   "under epic SHP-100"
#   "in epic SHP-100"
#   "epic SHP-100"
#   "parent SHP-100"
#   "set epic to SHP-100"
#   "move SHP-200 under SHP-100"  <- for edit
# We must NOT accidentally match the issue key the user is editing —
# the caller is responsible for passing the right text slice.
_JIRA_EPIC_PATTERNS = [
    # Explicit relator words: "under epic SHP-X", "in SHP-X",
    # "parent SHP-X", "epic SHP-X", "child of SHP-X", "link to SHP-X".
    re.compile(
        r"\b(?:under|in|inside|child\s+of|parent(?:\s+is)?|epic|link\s+to)\s+"
        r"(?:epic\s+|to\s+|is\s+)?"
        r"([A-Z][A-Z0-9]+-\d+)\b",
        re.IGNORECASE,
    ),
    # "epic to X" / "epic is X" / "epic = X" — relator AFTER the
    # word "epic" rather than before. Handles "set SHP-X epic to SHP-Y".
    re.compile(
        r"\bepic\s+(?:to|is|=)\s+([A-Z][A-Z0-9]+-\d+)\b",
        re.IGNORECASE,
    ),
    # "set the epic to X" / "set parent to X" — verb-first form,
    # robust to "set <KEY> epic/parent to X".
    re.compile(
        r"\bset\s+(?:the\s+)?(?:epic|parent)\s+to\s+([A-Z][A-Z0-9]+-\d+)\b",
        re.IGNORECASE,
    ),
    # "move SHP-200 to SHP-100" / "link SHP-200 under SHP-100"
    re.compile(
        r"\b(?:move|link)\s+[A-Z][A-Z0-9]+-\d+\s+(?:to|under)\s+([A-Z][A-Z0-9]+-\d+)\b",
        re.IGNORECASE,
    ),
]


def _extract_epic_key(text: str, exclude_key: Optional[str] = None) -> str:
    """
    Return the epic / parent key mentioned in `text`. If
    `exclude_key` is set (e.g. the ticket being edited), we filter
    that key out — "move SHP-200 under SHP-100" should return
    SHP-100, not SHP-200, even if the regex picks the wrong one.
    """
    for pat in _JIRA_EPIC_PATTERNS:
        for m in pat.finditer(text):
            candidate = m.group(1).strip().upper()
            if exclude_key and candidate.upper() == exclude_key.upper():
                continue
            return candidate
    return ""


# Description text. Supports quoted or unquoted forms:
#   'with description "X"'
#   'describe it as "X"'
#   'described as X'
#   'description is X'
#   'set description to X'
# Lazy capture so we stop at the next sentence boundary unless a
# closing quote forces an earlier stop.
_JIRA_DESCRIPTION_PATTERNS = [
    # Quoted form — high confidence; capture between quotes.
    re.compile(
        r"\bdescri(?:bed|ption|be(?:\s+it)?)?\s+"
        r"(?:as|to|is|=|:)?\s*"
        r"['\"]([^'\"]+)['\"]",
        re.IGNORECASE,
    ),
    # Unquoted form — lazier; capture until sentence boundary or
    # another structural keyword.
    re.compile(
        r"\bdescri(?:bed|ption|be(?:\s+it)?)?\s+"
        r"(?:as|to|is|=|:)\s+"
        r"([^,.!?]+?)"
        r"(?:[.!?]|\s+(?:assign(?:ed)?|due|by|with|priority|story|epic|points?|parent)\b|$)",
        re.IGNORECASE,
    ),
]


def _extract_description(text: str) -> str:
    """
    Return the description string mentioned in `text`, or empty.
    Tries quoted form first (higher confidence), then unquoted.
    """
    for pat in _JIRA_DESCRIPTION_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).strip().rstrip(".,;:?!")
    return ""


# Sprint detection. Three reasonable speech forms:
#   "in the current sprint" / "active sprint"
#   "in sprint 92" / "sprint Stack Sprint 92"
#   "to the next sprint" / "upcoming sprint"
# Patterns capture either a special token ("current"/"next") or a
# free-form sprint identifier (number or name fragment).
# Common leading particles ("in", "to", "under", etc) that introduce
# a sprint reference. We absorb them into the patterns so they get
# stripped along with the sprint phrase from the title fallback —
# otherwise we'd end up with "test in" after stripping just the
# "current sprint" half.
_SPRINT_PREFIX = r"(?:\b(?:in|to|under|inside|into|on)\s+)?"

# Optional team prefix that can sit BETWEEN the leading particle and
# the sprint phrase. "team core current sprint" / "core team current
# sprint". Non-capturing so the existing group indices stay aligned.
_SPRINT_TEAM_OPT = r"(?:team\s+\w+\s+|\w+\s+team\s+)?"

_JIRA_SPRINT_PATTERNS = [
    # "in team core current sprint" / "in the current sprint" / "current sprint"
    re.compile(
        rf"{_SPRINT_PREFIX}{_SPRINT_TEAM_OPT}(?:the\s+)?(current|active|this|present)\s+sprint\b",
        re.IGNORECASE,
    ),
    # "to team ux next sprint" / "next sprint" / "upcoming sprint"
    re.compile(
        rf"{_SPRINT_PREFIX}{_SPRINT_TEAM_OPT}(?:the\s+)?(next|upcoming|future)\s+sprint\b",
        re.IGNORECASE,
    ),
    # "in sprint 92" / "sprint 92" — numeric token after "sprint"
    re.compile(rf"{_SPRINT_PREFIX}{_SPRINT_TEAM_OPT}sprint\s+(\d+)\b", re.IGNORECASE),
    # "to sprint <name>" — quoted form, e.g. 'sprint "Stack Sprint 92"'
    re.compile(
        rf"{_SPRINT_PREFIX}{_SPRINT_TEAM_OPT}sprint\s+['\"]([^'\"]+)['\"]",
        re.IGNORECASE,
    ),
    # Bareword form — "to sprint <name>" / "in sprint <name>"
    re.compile(
        r"\b(?:to|in|add\s+to|move\s+to|set\s+sprint\s+to)\s+sprint\s+"
        r"([^,.!?]+?)(?:[,.!?]|$)",
        re.IGNORECASE,
    ),
]


def _extract_sprint_request(text: str) -> str:
    """
    Return a sprint identifier string when the utterance mentions
    a sprint. Possible return values:
      - "current" / "active" → resolve to active sprint at call time
      - "next" / "upcoming" → first future sprint
      - "<N>" → numeric, matches "Sprint <N>" in sprint names
      - "<name fragment>" → substring match against sprint names
      - "" → no sprint reference detected
    """
    for pat in _JIRA_SPRINT_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        token = m.group(1).strip().lower()
        # Normalise the special tokens.
        if token in {"current", "active", "this", "present"}:
            return "current"
        if token in {"next", "upcoming", "future"}:
            return "next"
        return m.group(1).strip()
    return ""


def _resolve_sprint(token: str, sprints: List[dict]) -> Optional[dict]:
    """
    Resolve a sprint identifier token (returned by
    _extract_sprint_request) against a cached list of sprint dicts
    (each with id/state/name keys, from jira_list_sprints).

    Returns the FULL sprint dict on success, None on miss. The dict
    has enough info for the handler to name the sprint in the
    confirmation read-back.
    """
    if not token or not sprints:
        return None
    token_lc = token.lower()

    # Special tokens first.
    if token_lc == "current":
        for sp in sprints:
            if sp.get("state") == "active":
                return sp
        return None
    if token_lc == "next":
        for sp in sprints:
            if sp.get("state") == "future":
                return sp
        return None

    # Numeric token: match "Sprint <N>" in name (case-insensitive).
    # Need word boundary so "Sprint 9" doesn't match "Sprint 92".
    if token.isdigit():
        n = int(token)
        # Two passes: prefer active/future sprints over closed
        # ones if the number matches multiple historical sprints.
        for state_pref in ("active", "future", "closed"):
            for sp in sprints:
                if sp.get("state") != state_pref:
                    continue
                if re.search(rf"\bSprint\s+{n}\b", sp.get("name", ""), re.IGNORECASE):
                    return sp
        return None

    # Name fragment: substring match. Prefer active/future first.
    for state_pref in ("active", "future", "closed"):
        for sp in sprints:
            if sp.get("state") != state_pref:
                continue
            if token_lc in sp.get("name", "").lower():
                return sp
    return None


# Multi-team sprint support: extract the team name from a sprint
# reference like "team core current sprint" or "stack team next sprint".
# We only look inside the sprint phrase to avoid false positives from
# unrelated mentions of team names elsewhere in the utterance.
_JIRA_TEAM_PATTERN = re.compile(
    r"\b(?:team\s+)?(stack|core|infra|ux|cortana|data)(?:'s)?(?:\s+team)?\s+"
    r"(?:the\s+)?(?:current|active|this|present|next|upcoming|future|sprint)\b",
    re.IGNORECASE,
)


def _extract_team(text: str) -> str:
    """
    Pull the team name (lower-case) out of a sprint reference, if
    present. Returns "" when no team is mentioned — caller defaults
    to whichever board is configured as primary.
    """
    if not text:
        return ""
    m = _JIRA_TEAM_PATTERN.search(text)
    if m:
        return m.group(1).lower()
    return ""


def derive_team_boards(boards: List[dict]) -> Dict[str, int]:
    """
    Parse the boards list returned by jira_list_boards_for_project
    into {team_name_lower: board_id}. We only keep scrum boards
    (kanban boards don't have sprints).

    Naming heuristic, in order:
      1. "Team X" prefix     → team = X.lower()
      2. "X Team" suffix     → team = X.lower()
      3. First word of name  → team = first.lower()

    Result is a flat dict; later boards overwrite earlier ones on
    name conflict (rare in practice — Shypple's boards are unique).
    """
    if not boards:
        return {}
    out: Dict[str, int] = {}
    for b in boards:
        if not isinstance(b, dict):
            continue
        if b.get("type") != "scrum":
            continue
        name = (b.get("name") or "").strip()
        board_id = b.get("id")
        if not name or not board_id:
            continue
        # Pattern 1: "Team X..."
        m = re.match(r"^Team\s+([A-Za-z0-9_-]+)", name, re.IGNORECASE)
        if m:
            out[m.group(1).lower()] = int(board_id)
            continue
        # Pattern 2: "X Team..."
        m = re.match(r"^([A-Za-z0-9_-]+)\s+Team\b", name, re.IGNORECASE)
        if m:
            out[m.group(1).lower()] = int(board_id)
            continue
        # Pattern 3: first word.
        first = name.split()[0].lower() if name else ""
        if first:
            out[first] = int(board_id)
    return out


# --- Components (validated against project's defined list) ---
# Detection: "with components STACK and CORE" / "in component STACK" /
# "label component LAB". Captures one or many.
_JIRA_COMPONENT_PATTERNS = [
    # Single combined pattern: optional "in" prefix, then "components"
    # / "component" / "components: " / "components to ". Captures the
    # comma-separated value list lazily, stopping at the next field
    # keyword (via lookahead so the next pattern can still match it).
    re.compile(
        r"\b(?:in\s+)?components?\s+(?:as\s+|to\s+|:)?\s*([A-Za-z0-9_, \-]+?)"
        r"(?=[.!?]|\s+(?:assign(?:ed)?|due|by|with|priority|labels?|reporter|sprint|epic|description|story|comments?)\b|$)",
        re.IGNORECASE,
    ),
]


def _extract_components(text: str, valid_components: List[str]) -> List[str]:
    """
    Return a list of component NAMES (using the exact spelling from
    valid_components) that appeared in the utterance. Case-insensitive
    match, free-text candidate list split on commas / "and".

    valid_components is the cached project component list — we ONLY
    return entries that match a known component, so the user can't
    accidentally invent a new component (Jira would 400 anyway).
    """
    if not text or not valid_components:
        return []
    raw_candidates: List[str] = []
    for pat in _JIRA_COMPONENT_PATTERNS:
        for m in pat.finditer(text):
            raw_candidates.append(m.group(1))
    if not raw_candidates:
        return []
    # Tokenise candidates on commas + "and" + whitespace, then keep
    # the ones matching valid_components (case-insensitive).
    valid_lc = {c.lower(): c for c in valid_components}
    found: List[str] = []
    for blob in raw_candidates:
        tokens = re.split(r"[,\s]+|\band\b", blob, flags=re.IGNORECASE)
        for t in tokens:
            t = t.strip()
            if not t:
                continue
            if t.lower() in valid_lc:
                canonical = valid_lc[t.lower()]
                if canonical not in found:
                    found.append(canonical)
    return found


# --- Labels (free-text, no validation) ---
# Detection: "with labels backend, frontend" / "label as SRE" /
# "labels: tech-budget".
_JIRA_LABEL_PATTERNS = [
    re.compile(
        r"\blabels?\s+(?:as\s+|to\s+|:|are\s+|is\s+|with\s+)?\s*([A-Za-z0-9_, \-]+?)"
        # Lookahead so the next field's keyword isn't consumed.
        r"(?=[.!?]|\s+(?:assign(?:ed)?|due|by|with|priority|components?|reporter|sprint|epic|description|story|comments?)\b|$)",
        re.IGNORECASE,
    ),
]


def _extract_labels(text: str) -> List[str]:
    """
    Return a list of label strings (lower-case, hyphen/underscore
    preserved). Labels are free-text — we don't validate against a
    project list. Splits on commas / "and" / whitespace.
    """
    if not text:
        return []
    raw_blobs: List[str] = []
    for pat in _JIRA_LABEL_PATTERNS:
        for m in pat.finditer(text):
            raw_blobs.append(m.group(1))
    if not raw_blobs:
        return []
    found: List[str] = []
    for blob in raw_blobs:
        tokens = re.split(r"[,\s]+|\band\b", blob, flags=re.IGNORECASE)
        for t in tokens:
            t = t.strip().rstrip(".,;:?!")
            if t and t.lower() not in {"the", "a", "an", "and", "or"}:
                if t not in found:
                    found.append(t)
    return found


# --- Reporter (mirror of assignee, uses roster) ---
_JIRA_REPORTER_PATTERNS = [
    re.compile(
        r"\breporter\s+(?:is\s+|to\s+|=\s+)?([A-Za-z][a-zA-Z]+)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\breported\s+by\s+([A-Za-z][a-zA-Z]+)\b", re.IGNORECASE),
    re.compile(r"\bset\s+reporter\s+to\s+([A-Za-z][a-zA-Z]+)\b", re.IGNORECASE),
]


def _extract_reporter(text: str, speaker: str) -> str:
    """
    Return the first-name token for the reporter, resolving "me" /
    "myself" / "i" to the speaker. Empty string when no match.
    """
    for pat in _JIRA_REPORTER_PATTERNS:
        m = pat.search(text)
        if m:
            name = m.group(1).strip()
            if name.lower() in _SELF_REFERENCE_TOKENS:
                return speaker
            return name
    return ""


# --- Comments (new action; mirrors lookup/edit handler shape) ---
# "comment on SHP-X saying ..." / "add a comment to SHP-X: ..."
# / "Mod, comment on SHP-X with ..."
_JIRA_COMMENT_PATTERNS = [
    # Quoted body — works regardless of separator before the quote.
    re.compile(
        r"\b(?:add\s+a\s+)?comment(?:ing)?\s+(?:on|to|for)\s+[A-Z][A-Z0-9]+-\d+"
        r"(?:\s*[:,]\s*|\s+(?:saying|with|that\s+says))?\s*"
        r"['\"]([^'\"]+)['\"]",
        re.IGNORECASE,
    ),
    # Unquoted body after "saying" / "that says" / "with" / ":" (the
    # colon variant allows zero-or-more whitespace before / after).
    re.compile(
        r"\b(?:add\s+a\s+)?comment(?:ing)?\s+(?:on|to|for)\s+[A-Z][A-Z0-9]+-\d+"
        r"(?:\s+(?:saying|with|that\s+says)\s+|\s*:\s*)"
        r"(.+?)(?:[.!?]\s*$|$)",
        re.IGNORECASE,
    ),
    # Permissive fallback — used when the speaker references the
    # ticket by anaphora ("comment on this ticket saying X") rather
    # than by SHP-N. The caller has already resolved the key via
    # _resolve_implicit_key, so we don't require SHP-N inside the
    # match. Quoted body first.
    re.compile(
        r"\b(?:add\s+a\s+)?comment(?:ing)?\s+"
        r"(?:.+?\s+)?"
        r"(?:saying|with|that\s+says|:)\s*"
        r"['\"]([^'\"]+)['\"]",
        re.IGNORECASE,
    ),
    # Permissive fallback — unquoted body to end of utterance.
    re.compile(
        r"\b(?:add\s+a\s+)?comment(?:ing)?\s+"
        r"(?:.+?\s+)?"
        r"(?:saying|with|that\s+says|:)\s+"
        r"(.+?)(?:[.!?]\s*$|$)",
        re.IGNORECASE,
    ),
]


def _extract_comment_body(text: str) -> str:
    """
    Return the body of the comment a user is asking to add. Empty
    string if no recognisable comment phrase. Quoted forms are
    preferred over unquoted (higher confidence).
    """
    for pat in _JIRA_COMMENT_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).strip().rstrip(".,;:?!")
    return ""


# --- Due date ISO resolution via Gemini ---
# Natural-language due phrases ("Friday", "next Wednesday", "May 28")
# get resolved to YYYY-MM-DD here. The current date is passed as an
# anchor so the result is unambiguous. Fails OPEN: returns "" on any
# error, caller skips the due_date field.
def _resolve_due_date_iso(due_phrase: str) -> str:
    """
    Turn a natural-language due phrase into YYYY-MM-DD using Gemini.
    Returns "" on any failure or empty input.
    """
    if not due_phrase or not due_phrase.strip():
        return ""
    # Quick win: if the phrase is already ISO YYYY-MM-DD, skip Gemini.
    iso_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    if iso_re.match(due_phrase.strip()):
        return due_phrase.strip()

    if _gemini_client is None:
        return ""

    today = datetime.now().strftime("%Y-%m-%d (%A)")
    prompt = (
        f"Today's date is {today}. Resolve the following relative date "
        f"phrase to an ISO date in the form YYYY-MM-DD. Return ONLY the "
        f"date, no other text. If the phrase is ambiguous or you cannot "
        f"resolve it confidently, return the single word UNKNOWN.\n\n"
        f"Phrase: {due_phrase}"
    )
    try:
        response = _gemini_client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=50,
                temperature=0.0,
            ),
        )
        text = (response.text or "").strip()
        # Strip surrounding quotes / punctuation.
        text = text.strip("'\"`.,;: ")
        if text.upper() == "UNKNOWN":
            return ""
        if iso_re.match(text):
            return text
        # Maybe Gemini returned "2026-05-22." — peel a single
        # trailing dot.
        if iso_re.match(text.rstrip(".")):
            return text.rstrip(".")
        return ""
    except Exception as exc:
        print(f"  [jira due-date resolver] Gemini failed: {exc}")
        return ""


def _summarise_jira_description_for_speech(description: str) -> str:
    """
    Turn a Jira ticket description into 2-3 TTS-friendly sentences.

    Jira descriptions can include markdown headers, code blocks,
    URLs, and bullet points — speaking that verbatim sounds awful.
    We delegate to Gemini Flash for a quick rewrite. Fails OPEN on
    any error: caller is expected to fall back to a plain one-liner
    summary if this returns "".

    Why a one-shot generate_content call and not the moderator's
    chat session: the chat is committed to the per-turn JSON
    schema, so we'd collide with it. Same pattern
    extract_meeting_summary uses.
    """
    if not description or not description.strip():
        return ""
    # If the description is already short and clean (no markdown,
    # no URL), skip the LLM and just trim.
    plain = description.strip()
    if len(plain) <= 200 and "#" not in plain and "http" not in plain and "\n" not in plain:
        return plain

    if _gemini_client is None:
        return ""

    prompt = (
        "Rewrite the following Jira ticket description as a short "
        "plain-English summary suitable for speaking aloud. Use 2 to 3 "
        "COMPLETE sentences — never stop mid-sentence, always end with "
        "a period. Strip all markdown formatting (headers, bullet "
        "points, code blocks), remove URLs entirely (don't replace "
        "them with 'a link'), and remove any reference numbers or IDs "
        "that wouldn't be useful spoken. Be factual — don't add "
        "interpretation. Return ONLY the summary text, no preamble, "
        "no quotes around it.\n\n"
        f"Description:\n{description[:4000]}"
    )
    try:
        response = _gemini_client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                # Bug 3 fix: bumped from 400 -> 800 because the
                # previous limit was causing mid-sentence cuts on
                # longer ticket bodies.
                max_output_tokens=800,
                temperature=0.2,
            ),
        )
        text = (response.text or "").strip()
        # Collapse internal whitespace runs so the TTS doesn't pause
        # awkwardly on multiple consecutive newlines.
        text = re.sub(r"\s+", " ", text)
        # Belt-and-suspenders: if the model still stopped mid-clause
        # (no terminating punctuation), append "..." so the listener
        # knows the summary was cut.
        if text and text[-1] not in ".!?":
            text = text.rstrip(",;: ") + "..."
        # Cap defensively against truly runaway outputs.
        if len(text) > 600:
            text = text[:597].rstrip() + "..."
        return text
    except Exception as exc:
        print(f"  [jira-desc summary] Gemini failed: {exc}")
        return ""


def _extract_create_payload(text: str, speaker: str) -> dict:
    """
    Pull title + assignee + due-date out of a create-a-ticket
    utterance. Returns a dict suitable for stashing on
    state["pending_jira_action"].

    Strategy:
      1. Look for an EXPLICIT title pattern first ("name it X" /
         "called X" / "titled X"). If present, that's the title —
         we don't try to parse the rest of the sentence as title.
      2. Otherwise: find the trigger phrase via _JIRA_CREATE_TRIGGERS
         and use everything after it as the title, stripped of
         assignee + due fragments.
      3. Assignee = first match from _JIRA_ASSIGNEE_PATTERNS.
         If the captured name is a self-reference (me/myself/i),
         resolve to the current speaker. If nothing matches, default
         to the speaker (matches the existing Rule 3 owner convention).
      4. Due = first match from _JIRA_DUE_PATTERN, kept as the
         natural-language phrase ("Friday", "next Wednesday").
         We do NOT resolve it to YYYY-MM-DD — too brittle without
         an LLM. The Jira ticket gets created without a due date;
         the participant can set it in the Jira UI after.
    """
    # Assignee is extracted from the ORIGINAL text (not the post-
    # trigger fragment) so "assign it to me" anywhere in the sentence
    # gets caught even when the explicit-title pattern fires first.
    assignee_first_name = ""
    for pat in _JIRA_ASSIGNEE_PATTERNS:
        m = pat.search(text)
        if m:
            assignee_first_name = m.group(1)
            break

    # Resolve self-reference tokens (me / myself / i) -> speaker.
    if assignee_first_name.lower() in _SELF_REFERENCE_TOKENS:
        assignee_first_name = speaker

    # Try explicit-title patterns first. If we find one, that's the
    # title — no other extraction needed.
    explicit_title = ""
    for pat in _JIRA_EXPLICIT_TITLE_PATTERNS:
        m = pat.search(text)
        if m:
            explicit_title = m.group(1).strip().strip("'\"").rstrip(".,;:?!")
            break

    if explicit_title:
        title = re.sub(r"\s+", " ", explicit_title).strip()
    else:
        # Fall back to "everything after the trigger phrase" extraction.
        trigger_match = _JIRA_CREATE_TRIGGERS.search(text)
        if not trigger_match:
            title_raw = text
        else:
            title_raw = text[trigger_match.end():].strip()
            # Strip "for"/"about"/"to"/":" intro if present.
            title_raw = re.sub(
                r"^(for|about|to|:|\bcalled\b)\s+", "",
                title_raw, flags=re.IGNORECASE,
            ).strip()

        # Strip the assignee fragment out of the title.
        for pat in _JIRA_ASSIGNEE_PATTERNS:
            title_raw = pat.sub("", title_raw)

        # Strip due fragment.
        title_raw = _JIRA_DUE_PATTERN.sub("", title_raw)

        # Strip priority modifier fragments ("with high priority",
        # "priority high").
        title_raw = _JIRA_PRIORITY_PATTERN.sub("", title_raw)

        # Strip story-points / epic / description / sprint / component
        # / label / reporter / comment modifier fragments so we don't
        # end up with "test-4 in the current sprint with 5 points
        # under epic SHP-100 reporter Yaman with components STACK"
        # as the Jira summary.
        for pat in _JIRA_STORY_POINTS_PATTERNS:
            title_raw = pat.sub("", title_raw)
        for pat in _JIRA_EPIC_PATTERNS:
            title_raw = pat.sub("", title_raw)
        for pat in _JIRA_DESCRIPTION_PATTERNS:
            title_raw = pat.sub("", title_raw)
        for pat in _JIRA_SPRINT_PATTERNS:
            title_raw = pat.sub("", title_raw)
        for pat in _JIRA_COMPONENT_PATTERNS:
            title_raw = pat.sub("", title_raw)
        for pat in _JIRA_LABEL_PATTERNS:
            title_raw = pat.sub("", title_raw)
        for pat in _JIRA_REPORTER_PATTERNS:
            title_raw = pat.sub("", title_raw)

        # Post-strip cleanup: the pattern subs leave stray commas
        # ("a , , b"), spaces, and dangling connectives ("with ,").
        # Collapse multiple commas, then trim leading/trailing
        # punctuation per the final cleanup below.
        title_raw = re.sub(r"\s*,(?:\s*,)+", ",", title_raw)  # ",,," -> ","
        title_raw = re.sub(r"^\s*,\s*", "", title_raw)         # leading comma
        title_raw = re.sub(r",\s*$", "", title_raw)            # trailing comma

        title = re.sub(r"\s+", " ", title_raw).strip(" .,;:?!'\"")
        # Strip dangling connectives ("with", "for", "by", "and")
        # that point to fragments we already extracted (assignee,
        # priority, due, etc). Loop because multiple connectives
        # can dangle ("test-1 with for me" -> "test-1").
        while True:
            stripped = re.sub(
                r"\s+(with|for|by|and|to)\s*$", "", title,
                flags=re.IGNORECASE,
            ).strip(" .,;:?!'\"")
            if stripped == title:
                break
            title = stripped
        if not title:
            title = "Untitled action item"

    # Due (search the original text — same reasoning as assignee).
    due_phrase = ""
    m = _JIRA_DUE_PATTERN.search(text)
    if m:
        due_phrase = m.group(1).strip().rstrip(".,")

    # Priority — pulled from anywhere in the utterance.
    priority = _extract_priority(text)

    # Story points, epic key, description — all pulled from the
    # original utterance, same reasoning as priority.
    story_points = _extract_story_points(text)
    epic_key = _extract_epic_key(text)
    description = _extract_description(text)
    sprint_request = _extract_sprint_request(text)
    sprint_team = _extract_team(text)
    # Labels are free-text; component validation happens at the
    # handler (where the cached project component list is in scope).
    labels = _extract_labels(text)
    reporter = _extract_reporter(text, speaker)

    owner = assignee_first_name or speaker
    return {
        "type": "create",
        "title": title,
        "owner": owner,
        "due": due_phrase,
        "priority": priority,
        "story_points": story_points,
        "epic": epic_key,
        "description": description,
        # sprint_request is a TOKEN ("current"/"next"/"92"/name).
        # The handler resolves to a sprint dict at confirm time using
        # the cached sprint list. sprint_team picks which team's
        # board to look at; "" means use the default board.
        "sprint_request": sprint_request,
        "sprint_team": sprint_team,
        # Components RAW — handler validates against cached project
        # components and drops unknown ones (Jira would 400 otherwise).
        "components_raw": text,
        "labels": labels,
        "reporter": reporter,
        "speaker": speaker,
    }


def _extract_edit_payload(text: str, key: str, speaker: str) -> dict:
    """
    Parse an edit utterance like:
      "Mod, set SHP-101 to high priority"
      "Mod, update SHP-101 priority to medium"
      "Mod, reassign SHP-101 to Yaman"
      "Mod, change SHP-101 priority to low and assign it to me"
      "Mod, set SHP-101 to 5 story points"
      "Mod, move SHP-101 under epic SHP-15162"
      "Mod, set SHP-101 description to 'investigate the broken dropdown'"

    Returns a payload dict containing:
      type      = "edit"
      key       = "SHP-101"
      changes   = {field_name: human_readable_value, ...}
      fields    = {jira_api_field: jira_api_value, ...}
                  ready to pass to jira_edit_issue
      speaker   = who asked

    Supported fields: title (summary), priority, assignee, reporter,
    story_points, epic, description, sprint, components, labels,
    due_date. Unknown fields are silently dropped; if NO field is
    recognised, changes/fields are empty and the handler speaks "I
    didn't catch what to change".
    """
    changes: dict = {}
    fields: dict = {}

    # Title (summary). Done first so the captured title text doesn't
    # accidentally pick up downstream field keywords ("priority",
    # "assignee") that might appear later in the utterance. The
    # extractor's own guards keep placeholder captures ("the
    # following title") from staging a meaningless rename.
    new_title = _extract_edit_title(text)
    if new_title:
        # Jira caps summary at 255 chars — Jira API would 400 otherwise.
        new_title = new_title[:255]
        changes["title"] = new_title
        fields["summary"] = new_title

    # Priority — reuse the same regex used on create.
    priority = _extract_priority(text)
    if priority:
        changes["priority"] = priority
        fields["priority"] = {"name": priority}

    # Assignee — try the assignee regex; resolve self-references.
    new_assignee = ""
    for pat in _JIRA_ASSIGNEE_PATTERNS:
        m = pat.search(text)
        if m:
            new_assignee = m.group(1)
            break
    if new_assignee.lower() in _SELF_REFERENCE_TOKENS:
        new_assignee = speaker
    if new_assignee:
        changes["assignee"] = new_assignee
        # The actual Jira accountId resolution happens at confirm-time
        # (when we have the roster + host_email in scope). For now we
        # stash the first name so the readback can show it; the
        # handler does the email -> accountId lookup before the API
        # call.
        fields["_assignee_first_name"] = new_assignee

    # Story points — numeric custom field. The customfield_NNNNN key
    # comes from atlassian_client.STORY_POINTS_FIELD (one-time discovery
    # against SHP-17182).
    points = _extract_story_points(text)
    if points is not None:
        changes["story_points"] = points
        # Format the number cleanly: 5.0 -> 5, 2.5 -> 2.5.
        if points == int(points):
            changes["story_points"] = int(points)
        fields[STORY_POINTS_FIELD] = points

    # Epic / parent. We pass the key being edited as exclude_key so
    # "move SHP-200 under SHP-100" doesn't accidentally pick SHP-200.
    epic_key = _extract_epic_key(text, exclude_key=key)
    if epic_key:
        changes["epic"] = epic_key
        fields["parent"] = {"key": epic_key}

    # Description — overwrites the existing description outright.
    # Jira's API doesn't have an append mode without a fetch+rewrite,
    # which we skip for the demo.
    description = _extract_description(text)
    if description:
        changes["description"] = description
        fields["description"] = description

    # Sprint — stored as a TOKEN ("current"/"next"/"92"/name fragment)
    # to be resolved at confirm time against the cached sprint list.
    # We can't pre-resolve here without threading the sprint list
    # through every call site; the handler has it.
    sprint_request = _extract_sprint_request(text)
    sprint_team = _extract_team(text)
    if sprint_request:
        # Use a tagged sentinel value so the handler knows to do the
        # separate Agile-API add_issues_to_sprint call rather than
        # putting it in the regular update_issue_field call.
        changes["sprint"] = sprint_request  # human-readable for now;
        # the handler overwrites this with the resolved name before
        # the read-back fires.
        fields["_sprint_request"] = sprint_request
        fields["_sprint_team"] = sprint_team

    # Labels — overwrite (Jira labels field is a full array, not
    # additive). Caller can include existing labels in the new list
    # if they want to preserve them.
    labels = _extract_labels(text)
    if labels:
        changes["labels"] = labels
        fields["labels"] = labels

    # Reporter — same pattern as assignee, resolved at confirm-time
    # via the roster.
    reporter = _extract_reporter(text, speaker)
    if reporter:
        changes["reporter"] = reporter
        fields["_reporter_first_name"] = reporter

    # Due date — natural-language; resolved to ISO at confirm-time
    # via the Gemini-backed resolver. Stashing the raw phrase here.
    due_phrase = ""
    m = _JIRA_DUE_PATTERN.search(text)
    if m:
        due_phrase = m.group(1).strip().rstrip(".,;:?!")
    if due_phrase:
        changes["due"] = due_phrase
        fields["_due_phrase"] = due_phrase

    # Components — resolved at handler time against project list.
    # Stash raw text so the handler can re-run extraction with the
    # validated list in scope.
    fields["_components_raw"] = text

    return {
        "type": "edit",
        "key": key,
        "changes": changes,
        "fields": fields,
        "speaker": speaker,
    }


def detect_jira_intent(
    text: str,
    speaker: str,
    bot_name: str,
    pending_action: Optional[dict],
    project_key: str = "",
    last_key: str = "",
) -> Optional[dict]:
    """
    Pure-regex Jira-intent detector. Returns one of:
      {"action": "confirm"}                — pending exists; speaker said yes
      {"action": "cancel"}                 — pending exists; speaker said no
      {"action": "create", "payload": {...}} — new create request
      {"action": "lookup", "key": "SHP-..."} — direct lookup request
      None                                 — no Jira intent matched

    Order of checks matters:
      1. Pending action: confirm/cancel words override everything.
         Otherwise pending stays until next time.
      2. Bot must be addressed for create/lookup to fire (prevents
         incidental SHP-1234 mentions or "let's make a list" from
         triggering).
      3. Lookup checked before create — if a speaker says "Mod,
         what's in SHP-1234, and also create a ticket for X" we
         handle the lookup this turn; the create can be re-sent.

    Preprocessing pipeline (runs once, before any pattern matching):
      a. Normalise spelled-out Jira keys ("shp one seven seven six
         seven" -> "SHP-17767") so STT transcriptions still match.
      b. Strip filler words ("uh", "um", "you know") that wedge
         themselves between extraction keywords and values.
      c. Resolve anaphoric references ("this ticket", "the ticket
         you just created") against `last_key` so users don't have
         to dictate the SHP-N every time.
    """
    if not text:
        return None
    text = _normalize_spoken_jira_keys(text, project_key)
    text = _strip_fillers(text)
    text_lc = text.lower()

    if pending_action:
        # Cross-talk safety: only the speaker who issued the create
        # can confirm or cancel it. Without this guard, an unrelated
        # "yes that sounds good" from another participant during
        # normal conversation would silently create the ticket. The
        # pending speaker is captured when the create payload is
        # built; we fall back to permissive matching only if pending
        # somehow lacks a speaker field (defensive — shouldn't
        # happen in practice).
        pending_speaker = (pending_action.get("speaker") or "").strip().lower()
        same_speaker = (
            not pending_speaker
            or pending_speaker == (speaker or "").strip().lower()
        )
        if same_speaker:
            if _contains_any_phrase(text_lc, _CONFIRM_WORDS):
                return {"action": "confirm"}
            if _contains_any_phrase(text_lc, _CANCEL_WORDS):
                return {"action": "cancel"}
        # Pending stays — no Jira action this turn. (The normal
        # moderator pipeline still runs; the bot might say something
        # unrelated, which is fine.) Other speakers in the room can
        # talk freely without accidentally confirming Daniyal's
        # pending ticket.
        return None

    # Bot-addressing policy (relaxed 2026-05-21):
    #
    # We USED to require the bot's name in EVERY Jira-intent utterance.
    # That broke natural follow-ups in 1:1 testing — the user said
    # "Mod, look up X" then on the next turn said "Perfect, look up Y"
    # without re-stating "Mod", and the deterministic path silently
    # skipped, falling through to the LLM which only acknowledged
    # without doing anything.
    #
    # New policy: bot-addressing is ONLY required for the anaphora
    # path (where the signal is weakest — "this ticket" can easily
    # be incidental). Every other path has its own strong signal:
    #   - SHP-N key in the utterance (lookup / edit / comment)
    #   - "create/file/make a ticket" phrase (create)
    #   - "how many open tickets" / "per person" (search)
    # Even side-effecting paths (create / edit / comment) stay safe
    # thanks to the same-speaker confirmation gate; a spurious match
    # just stages a pending action the user doesn't confirm.
    bot_addressed = _is_bot_addressed(text, bot_name)

    # Anaphora resolution: if the speaker said "this ticket" / "the
    # ticket you just created" but didn't dictate the SHP-N key,
    # inject `last_key` so the edit / comment / lookup detectors
    # find a key as if it had been spoken. Skipped on create
    # requests — a fresh ticket can't be referencing a prior one.
    # We PREPEND rather than append so end-of-string captures
    # (e.g. the unquoted title pattern) don't accidentally swallow
    # the injected key. Bot-addressing IS still required here —
    # "this ticket" is too easy to say incidentally.
    if bot_addressed and not _JIRA_CREATE_TRIGGERS.search(text):
        implicit_key = _resolve_implicit_key(text, last_key)
        if implicit_key:
            text = f"{implicit_key} {text}"
            text_lc = text.lower()

    # Search (count / list) MUST be checked before create, because
    # "how many OPEN jira tickets are there" tickles the create
    # regex on the word "open". Search is more specific and only
    # fires when the count/list trigger is present, so giving it
    # priority is safe.
    search_intent = _detect_jira_search_intent(text, project_key)
    if search_intent:
        return {"action": "search", "search": search_intent, "speaker": speaker}

    # Create: trigger phrase present. Checked BEFORE edit/lookup so
    # "Mod, create a ticket … under epic SHP-15162" is treated as a
    # create (and SHP-15162 becomes the parent epic), not as a
    # lookup of SHP-15162. The create extractor handles the epic
    # field internally via _extract_epic_key.
    if _JIRA_CREATE_TRIGGERS.search(text):
        payload = _extract_create_payload(text, speaker)
        return {"action": "create", "payload": payload}

    # Comment: bot-addressed + comment verb + SHP-N. Checked BEFORE
    # edit/lookup since "comment on SHP-X" contains both a verb and
    # a key — without this branch it would match edit (verb 'comment'
    # isn't in edit triggers, but to be safe we route it here first).
    key_match = _JIRA_KEY_PATTERN.search(text)
    if key_match and re.search(r"\bcomment(?:ing)?\b", text, re.IGNORECASE):
        body = _extract_comment_body(text)
        return {
            "action": "comment",
            "key": key_match.group(1),
            "body": body,
            "speaker": speaker,
        }

    # Edit: SHP-1234 + an edit verb. Must be checked BEFORE lookup so
    # "Mod, update SHP-101 to high priority" is treated as edit, not
    # a passive lookup of SHP-101.
    if key_match and _JIRA_EDIT_TRIGGERS.search(text):
        payload = _extract_edit_payload(text, key_match.group(1), speaker)
        return {"action": "edit", "payload": payload}

    # Lookup: any SHP-1234 mention in a bot-addressed turn (no edit verb).
    if key_match:
        return {"action": "lookup", "key": key_match.group(1)}

    return None


def _execute_edit(
    pending: dict,
    state: dict,
    jira_client,
    base_url: str,
    host_email: str,
    roster: dict,
    speaker: str,
    sprints_by_team: Optional[Dict[str, List[dict]]] = None,
    components_list: Optional[List[str]] = None,
    default_sprints: Optional[List[dict]] = None,
) -> Tuple[str, Optional[str]]:
    """
    Execute a confirmed edit. Split out of handle_jira_action so the
    confirm branch stays readable.

    Resolves the assignee first-name to an accountId at the LAST
    possible moment (here, not in _extract_edit_payload) so we have
    the roster + host_email available without threading them into the
    extractor. Falls back to host_email if the first-name doesn't
    resolve in the roster.

    Records the action on state["jira_actions_taken"] with kind='edit'
    so the Slack *Actions Taken* section can surface it post-call.
    """
    key = pending["key"]
    fields = dict(pending.get("fields") or {})

    # Resolve _assignee_first_name -> accountId, if present. We do
    # this here (rather than in _extract_edit_payload) because the
    # extractor doesn't have the roster in scope.
    first_name = fields.pop("_assignee_first_name", None)
    if first_name:
        email = _resolve_owner_email(first_name, roster, host_email)
        account_id = jira_resolve_account_id(jira_client, email)
        if account_id:
            fields["assignee"] = {"accountId": account_id}
        else:
            # Can't resolve — speak the warning, do nothing else.
            print(f"  [jira edit] could not resolve {first_name!r} -> accountId; skipping assignee field.")

    # Reporter — same pattern as assignee.
    reporter_first = fields.pop("_reporter_first_name", None)
    if reporter_first:
        email = _resolve_owner_email(reporter_first, roster, host_email)
        account_id = jira_resolve_account_id(jira_client, email)
        if account_id:
            fields["reporter"] = {"accountId": account_id}
        else:
            print(f"  [jira edit] could not resolve reporter {reporter_first!r} -> accountId; skipping.")

    # Components — re-run extraction against the cached project list
    # (we couldn't validate at extract time). Drop the marker.
    components_raw = fields.pop("_components_raw", "")
    if components_raw and components_list:
        resolved = _extract_components(components_raw, components_list)
        if resolved:
            fields["components"] = [{"name": c} for c in resolved]
            # Surface in changes for the read-back / Slack section.
            pending.setdefault("changes", {})["components"] = ", ".join(resolved)

    # Due date — resolve to ISO if a phrase was captured.
    due_phrase = fields.pop("_due_phrase", "")
    if due_phrase:
        due_iso = _resolve_due_date_iso(due_phrase)
        if due_iso:
            fields["duedate"] = due_iso
            pending.setdefault("changes", {})["due"] = due_iso
        else:
            print(f"  [jira edit] could not resolve due phrase {due_phrase!r}; skipping.")
            pending.get("changes", {}).pop("due", None)

    # Sprint is handled via the Agile API, NOT update_issue_field.
    # Pop the marker fields so they don't get sent to issue_update.
    sprint_request = fields.pop("_sprint_request", None)
    sprint_team = fields.pop("_sprint_team", "") or ""
    # If the stager already resolved a sprint, use that; otherwise
    # resolve here using the team-keyed cache.
    sprint_resolved = fields.pop("_sprint_resolved", None)
    if sprint_request and not sprint_resolved:
        team_sprints = None
        if sprint_team and sprints_by_team and sprint_team in sprints_by_team:
            team_sprints = sprints_by_team[sprint_team]
        else:
            team_sprints = default_sprints or (
                sprints_by_team.get("stack") if sprints_by_team else None
            )
        if team_sprints:
            sprint_resolved = _resolve_sprint(sprint_request, team_sprints)
        if sprint_resolved:
            pending.setdefault("changes", {})["sprint"] = sprint_resolved.get("name", sprint_request)

    # If the ONLY change was a sprint move, we still need to call
    # the Agile API but skip the field-update call.
    edit_ok = True
    if fields:
        edit_ok = jira_edit_issue(jira_client, key, fields)
        if not edit_ok:
            return (
                f"Sorry, I couldn't update {key}. The Jira API rejected the changes.",
                None,
            )

    sprint_ok = True
    if sprint_resolved:
        sprint_ok = jira_add_to_sprint(jira_client, sprint_resolved.get("id"), key)
        if not sprint_ok:
            print(f"  [jira edit] sprint move failed for {key} -> {sprint_resolved.get('name')}")

    if not fields and not sprint_resolved:
        return (
            f"Sorry, I couldn't apply any changes to {key}. None of the requested "
            f"fields resolved cleanly.",
            None,
        )

    # Pull the updated issue for the Meet chat card. If the GET fails
    # we still report success — the edit DID land — just with a
    # simpler confirmation line.
    issue = jira_get_issue(jira_client, key) or {"key": key, "fields": {}}
    state.setdefault("jira_actions_taken", []).append({
        "kind": "edit",
        "issue": issue,
        "speaker": speaker,
        "changes": pending.get("changes") or {},
    })
    # Remember this key so future "this ticket" / "the ticket" turns
    # resolve to it without the speaker re-dictating SHP-N.
    state["last_jira_key"] = key
    change_str = ", ".join(
        f"{k} = {v}" for k, v in (pending.get("changes") or {}).items()
    )
    speak = f"Updated {key}: {change_str}. Link in chat."
    chat = "✏️ Updated " + format_issue_for_chat(base_url, issue)
    return speak, chat


def handle_jira_action(
    intent: dict,
    state: dict,
    jira_client,
    project_key: str,
    issue_type: str,
    base_url: str,
    host_email: str,
    roster: dict,
    speaker: str,
    sprints: Optional[List[dict]] = None,
    sprints_by_team: Optional[Dict[str, List[dict]]] = None,
    components_list: Optional[List[str]] = None,
) -> Tuple[str, Optional[str]]:
    """
    Execute the Jira intent and return (bot_speak_text, meet_chat_msg).

    The caller is responsible for actually sending the TTS / Meet chat
    message. We just compute them. State updates happen here:
      - state["pending_jira_action"]      — set on 'create', cleared on
                                            'confirm' / 'cancel'.
      - state["jira_actions_taken"]       — appended on successful
                                            'confirm' and 'lookup'.

    Fails OPEN on every Jira API call: a network blip never raises
    out of this function. The bot will speak a graceful "I had
    trouble" line instead.
    """
    action = intent.get("action")

    if action == "create":
        # Stage the pending action; do NOT call Jira yet.
        payload = intent["payload"]
        # Resolve assignee email NOW so the confirmation message can
        # name the actual person who'll be assigned (or fall back).
        assignee_email = _resolve_owner_email(payload["owner"], roster, host_email)
        payload["assignee_email"] = assignee_email
        # Resolve reporter email too (if specified).
        reporter_first = (payload.get("reporter") or "").strip()
        reporter_email = ""
        if reporter_first:
            reporter_email = _resolve_owner_email(reporter_first, roster, host_email)
        payload["reporter_email"] = reporter_email
        # Resolve the sprint request to a concrete sprint NOW so the
        # read-back can name it. Pick the right team's board (if a
        # team was named); otherwise fall back to the legacy single-
        # board cache.
        sprint_request = payload.get("sprint_request") or ""
        sprint_team = payload.get("sprint_team") or ""
        sprint_dict = None
        if sprint_request:
            team_sprints = None
            if sprint_team and sprints_by_team and sprint_team in sprints_by_team:
                team_sprints = sprints_by_team[sprint_team]
            else:
                team_sprints = sprints or (
                    sprints_by_team.get("stack") if sprints_by_team else None
                )
            if team_sprints:
                sprint_dict = _resolve_sprint(sprint_request, team_sprints)
        payload["sprint"] = sprint_dict  # full dict or None
        # Resolve components against the cached project list.
        components_resolved = _extract_components(
            payload.get("components_raw") or "", components_list or []
        )
        payload["components"] = components_resolved
        # Resolve due-date to ISO via Gemini if a phrase was captured.
        due_phrase = payload.get("due") or ""
        due_iso = _resolve_due_date_iso(due_phrase) if due_phrase else ""
        payload["due_iso"] = due_iso
        state["pending_jira_action"] = payload
        # Build a confirmation prompt that reads back the parsed
        # values so STT garbles don't silently produce wrong tickets.
        # (due_str is set further down based on due_iso resolution.)
        priority_str = (
            f", priority {payload['priority']}" if payload.get("priority") else ""
        )
        # Format story points cleanly (5.0 -> 5, 2.5 -> 2.5).
        sp = payload.get("story_points")
        if sp is not None:
            sp_display = int(sp) if sp == int(sp) else sp
            sp_str = f", {sp_display} story points"
        else:
            sp_str = ""
        epic_str = f", under epic {payload['epic']}" if payload.get("epic") else ""
        # Sprint read-back: prefer the resolved sprint name; otherwise
        # warn the user we couldn't find their sprint reference.
        if sprint_dict:
            sprint_str = f", in sprint '{sprint_dict.get('name', '?')}'"
        elif sprint_request:
            sprint_str = f" (but I couldn't find a sprint matching '{sprint_request}' — it won't be added to any sprint)"
        else:
            sprint_str = ""
        # Description in the read-back is truncated to keep the TTS
        # confirmation short — full description gets sent to Jira.
        desc = payload.get("description") or ""
        if desc:
            short_desc = desc if len(desc) <= 60 else desc[:57] + "..."
            desc_str = f", description '{short_desc}'"
        else:
            desc_str = ""
        # Components — only mention if the user requested them AND
        # they resolved to known project components.
        components_str = ""
        if payload.get("components"):
            components_str = f", components {', '.join(payload['components'])}"
        # Labels — free-text, show as-is.
        labels_str = ""
        if payload.get("labels"):
            labels_str = f", labels {', '.join(payload['labels'])}"
        # Reporter — only mention if explicitly requested.
        reporter_str = ""
        if payload.get("reporter"):
            reporter_str = f", reporter {payload['reporter']}"
        # Due date — prefer the ISO-resolved form so the user knows
        # what date the bot heard (defends against STT garble + the
        # bot's date math). Show the natural-language phrase as a
        # fallback when resolution failed.
        if payload.get("due_iso"):
            due_str = f", due {payload['due_iso']}"
        elif payload.get("due"):
            due_str = f", due {payload['due']} (couldn't resolve to ISO — not saving)"
        else:
            due_str = ""
        speak = (
            f"I'll create a {issue_type} in {project_key} titled "
            f"'{payload['title']}', assigned to {payload['owner']}"
            f"{reporter_str}{priority_str}{sp_str}{epic_str}{sprint_str}"
            f"{components_str}{labels_str}{desc_str}{due_str}. "
            f"Say 'confirm' to proceed or 'cancel' to skip."
        )
        return speak, None

    if action == "edit":
        # Stage the pending edit; do NOT call Jira yet. Same
        # confirmation gate as create — read back what we parsed
        # so STT garbles can't silently mutate a real ticket.
        payload = intent["payload"]
        # If the edit involves a sprint, resolve the token to a
        # concrete sprint dict NOW so the read-back can name it.
        # Multi-team aware: prefers the team mentioned in the
        # utterance ("team core current sprint") over the default
        # (Stack).
        sprint_request = (payload.get("fields") or {}).get("_sprint_request")
        sprint_team = (payload.get("fields") or {}).get("_sprint_team") or ""
        if sprint_request:
            team_sprints = None
            if sprint_team and sprints_by_team and sprint_team in sprints_by_team:
                team_sprints = sprints_by_team[sprint_team]
            else:
                team_sprints = sprints or (
                    sprints_by_team.get("stack") if sprints_by_team else None
                )
            sprint_dict = None
            if team_sprints:
                sprint_dict = _resolve_sprint(sprint_request, team_sprints)
            if sprint_dict:
                # Overwrite the changes dict with the resolved name
                # so the read-back is human-friendly.
                payload["changes"]["sprint"] = sprint_dict.get("name", sprint_request)
                # Stash the resolved sprint dict alongside the marker
                # so _execute_edit can pull the ID without re-resolving.
                payload["fields"]["_sprint_resolved"] = sprint_dict
            else:
                # Couldn't resolve — drop it from changes so we don't
                # promise something we can't deliver.
                payload["changes"].pop("sprint", None)
                payload["fields"].pop("_sprint_request", None)
                payload["fields"].pop("_sprint_team", None)
        # Validate / resolve components at stage time so the read-back
        # tells the user exactly which ones will be set. Drop ones
        # that don't match the project's defined component list.
        components_raw = (payload.get("fields") or {}).get("_components_raw") or ""
        if components_raw and components_list:
            resolved = _extract_components(components_raw, components_list)
            if resolved:
                payload["changes"]["components"] = ", ".join(resolved)
        # We don't pre-resolve due-date or reporter here because both
        # require a network call (Gemini for due, /user/search for
        # reporter). _execute_edit handles them at confirm time.
        if not payload.get("changes"):
            # Detection fired (verb + SHP-N) but we didn't recognise
            # any field. Speak a guidance message; don't stage.
            return (
                f"I heard {payload['key']} but didn't catch what to change, "
                f"or I couldn't find the sprint you mentioned. Try 'set "
                f"{payload['key']} priority to high' or 'reassign "
                f"{payload['key']} to <name>'.",
                None,
            )
        state["pending_jira_action"] = payload
        # Build a human-readable change list ("priority to High, "
        # "assignee to Yaman").
        parts = []
        for field_name, value in payload["changes"].items():
            parts.append(f"{field_name} to {value}")
        change_str = ", ".join(parts)
        speak = (
            f"I'll update {payload['key']}: {change_str}. "
            f"Say 'confirm' to proceed or 'cancel' to skip."
        )
        return speak, None

    if action == "comment":
        # Stage the comment with a confirmation gate. Mirrors the
        # edit pattern: read back the body, require explicit
        # confirm, then call jira_add_comment on confirm.
        key = intent["key"]
        body = (intent.get("body") or "").strip()
        if not body:
            return (
                f"I heard a comment request on {key} but didn't catch what to say. "
                f"Try 'add a comment to {key} saying X' or use quotes for the body.",
                None,
            )
        state["pending_jira_action"] = {
            "type": "comment",
            "key": key,
            "body": body,
            "speaker": speaker,
        }
        # Truncate the body in the read-back to keep TTS short.
        preview = body if len(body) <= 80 else body[:77] + "..."
        speak = (
            f"I'll add a comment to {key}: '{preview}'. "
            f"Say 'confirm' to proceed or 'cancel' to skip."
        )
        return speak, None

    if action == "cancel":
        state["pending_jira_action"] = None
        return "Cancelled.", None

    if action == "confirm":
        pending = state.get("pending_jira_action")
        state["pending_jira_action"] = None
        if not pending:
            # Shouldn't reach here — detect_jira_intent only emits
            # confirm when pending is set. Defensive.
            return "Nothing pending to confirm.", None
        # Branch on the pending action type — confirms can apply to
        # a staged create, edit, OR comment.
        if pending.get("type") == "comment":
            ok = jira_add_comment(jira_client, pending["key"], pending["body"])
            if not ok:
                return (
                    f"Sorry, I couldn't add that comment to {pending['key']}. "
                    f"The Jira API rejected the request.",
                    None,
                )
            issue = jira_get_issue(jira_client, pending["key"]) or {
                "key": pending["key"], "fields": {}
            }
            state.setdefault("jira_actions_taken", []).append({
                "kind": "comment",
                "issue": issue,
                "speaker": speaker,
                "comment": pending["body"],
            })
            # Remember this key for anaphora resolution in later turns.
            state["last_jira_key"] = pending["key"]
            chat = (
                f"💬 Commented on {pending['key']}: '{pending['body'][:120]}"
                + ("...'" if len(pending['body']) > 120 else "'")
                + f"\n   {format_issue_url(base_url, pending['key'])}"
            )
            return f"Added comment to {pending['key']}. Link in chat.", chat

        if pending.get("type") == "edit":
            return _execute_edit(
                pending, state, jira_client, base_url, host_email, roster, speaker,
                sprints_by_team=sprints_by_team,
                components_list=components_list,
                default_sprints=sprints,
            )
        # Fall through: create confirm.
        # If the speaker explicitly dictated a description, use that
        # as the ticket body. Otherwise build a short audit-trail
        # description so the ticket isn't blank in Jira.
        user_description = (pending.get("description") or "").strip()
        if user_description:
            description_to_send = user_description
        else:
            description_to_send = (
                "Created mid-meeting by Meeting Moderator.\n"
                f"Requested by: {pending.get('speaker', '?')}\n"
                f"Owner (as spoken): {pending.get('owner', '?')}\n"
                + (f"Due (as spoken): {pending['due']}\n" if pending.get("due") else "")
                + (f"Priority: {pending['priority']}\n" if pending.get("priority") else "")
            )
        sprint_dict = pending.get("sprint")
        issue = jira_create_issue(
            jira_client,
            project_key=project_key,
            summary=pending["title"],
            description=description_to_send,
            assignee_email=pending.get("assignee_email"),
            fallback_assignee_email=host_email,
            issue_type=issue_type,
            priority=pending.get("priority") or None,
            story_points=pending.get("story_points"),
            epic_key=pending.get("epic") or None,
            sprint_id=(sprint_dict or {}).get("id"),
            components=pending.get("components") or None,
            labels=pending.get("labels") or None,
            reporter_email=pending.get("reporter_email") or None,
            # Due_date now flows through — _resolve_due_date_iso
            # converts "Friday" / "next Wednesday" / "May 28" to ISO.
            due_date=pending.get("due_iso") or None,
            # The old behaviour: "Friday" wasn't ISO and Jira 400'd.
            # Now we resolve via Gemini and pass valid ISO when
            # the due date in the Jira UI.
        )
        if issue is None:
            return ("Sorry, I couldn't create that ticket. The Jira API rejected the request.", None)
        key = issue.get("key", "?")
        # Jira's POST /rest/api/3/issue endpoint returns only
        # {key, id, self} — no `fields` block. Without this injection
        # the Meet chat + Slack *Actions Taken* section would both
        # render the ticket as "(no summary)". We inject the title we
        # JUST sent (the source of truth) so the formatters can show
        # it. This is safer than a follow-up GET — no extra API call,
        # no race against eventual-consistency.
        issue.setdefault("fields", {})
        issue["fields"]["summary"] = pending["title"]
        state.setdefault("jira_actions_taken", []).append({
            "kind": "create",
            "issue": issue,
            "speaker": speaker,
        })
        # Remember this key so the next "edit this ticket" turn
        # resolves to it via the anaphora pathway.
        state["last_jira_key"] = key
        url = format_issue_url(base_url, key)
        speak = f"Created {key}. Link posted to chat."
        chat = format_created_issue_for_chat(base_url, issue)
        return speak, chat

    if action == "lookup":
        key = intent["key"]
        issue = jira_get_issue(jira_client, key)
        if issue is None:
            return (f"Sorry, I couldn't find {key} in Jira.", None)
        # Compact spoken summary — pull the most useful fields.
        fields = issue.get("fields") or {}
        summary_text = fields.get("summary") or "(no summary)"
        status = (fields.get("status") or {}).get("name", "Unknown")
        assignee_obj = fields.get("assignee") or {}
        assignee = (
            assignee_obj.get("displayName")
            if isinstance(assignee_obj, dict) and assignee_obj
            else "Unassigned"
        )
        state.setdefault("jira_actions_taken", []).append({
            "kind": "lookup",
            "issue": issue,
            "speaker": speaker,
        })
        # Remember this key — a lookup is a deliberate reference, so
        # "this ticket" in a follow-up turn should resolve to it.
        state["last_jira_key"] = key
        # Build a description summary if there's body text. Gemini
        # call is cheap and fails open. Empty string when description
        # is missing or the LLM errors.
        description = fields.get("description") or ""
        if isinstance(description, dict):
            # Atlassian Cloud sometimes returns ADF (dict) instead of
            # plain string for description. Best-effort: skip the
            # summary for now and only speak the one-liner.
            description = ""
        desc_summary = _summarise_jira_description_for_speech(description)
        if desc_summary:
            speak = (
                f"{key} is '{summary_text[:80]}', assigned to {assignee}, "
                f"status {status}. Description: {desc_summary}"
            )
        else:
            speak = (
                f"{key} is '{summary_text[:80]}', assigned to {assignee}, "
                f"status {status}."
            )
        chat = format_issue_for_chat(base_url, issue)
        return speak, chat

    if action == "search":
        # JQL search (count, list, or per-person breakdown). Read-only
        # — no confirmation gate.
        search = intent.get("search") or {}
        kind = search.get("kind", "count")
        jql = search.get("jql", "")
        base_jql = search.get("base_jql", "")
        description = search.get("description", "matching tickets")
        assignee_name = search.get("assignee_name", "")
        if not jql:
            return "", None

        # Per-person scope (single name) — resolve via roster +
        # accountId lookup, then narrow the JQL before falling through
        # to the regular count / list paths.
        if assignee_name and kind in ("count", "list"):
            email = (roster or {}).get(assignee_name.lower())
            if not email:
                return (
                    f"I don't have {assignee_name} in my team roster, "
                    f"so I can't count tickets for them.",
                    None,
                )
            account_id = jira_resolve_account_id(jira_client, email)
            if not account_id:
                return (
                    f"Couldn't find {assignee_name}'s Jira account, "
                    f"so I can't run that search.",
                    None,
                )
            # Inject the assignee clause BEFORE ORDER BY.
            assignee_clause = f'assignee = "{account_id}"'
            if " ORDER BY " in jql:
                head, tail = jql.split(" ORDER BY ", 1)
                jql = f"{head} AND {assignee_clause} ORDER BY {tail}"
            else:
                jql = f"{jql} AND {assignee_clause}"
            description = f"{description} assigned to {assignee_name}"

        # Per-person breakdown — enumerate the roster, one count each.
        # Costs N round-trips (N == roster size). For the 8-person
        # Shypple roster that's well under a second.
        if kind == "breakdown":
            if not roster:
                return (
                    "I don't have a team roster loaded, so I can't "
                    "break down tickets by person.",
                    None,
                )
            per_person: List[tuple] = []  # [(first_name, count), ...]
            for first_name, email in sorted(roster.items()):
                account_id = jira_resolve_account_id(jira_client, email)
                if not account_id:
                    per_person.append((first_name.title(), -1))
                    continue
                person_jql = (
                    f'{base_jql} AND assignee = "{account_id}"'
                    if base_jql else f'assignee = "{account_id}"'
                )
                count = jira_search_jql_total(jira_client, person_jql)
                per_person.append((first_name.title(), count))
            # Sort: known counts descending, errors at the bottom.
            per_person.sort(key=lambda p: (p[1] < 0, -p[1]))
            total = sum(c for _, c in per_person if c >= 0)
            # Spoken summary — top 3 to keep it short.
            top3 = [
                f"{name} with {c}" for name, c in per_person if c >= 0
            ][:3]
            top3_str = ", ".join(top3) if top3 else "no one"
            speak = (
                f"Across the team, there are {total} {description}. "
                f"The top owners are {top3_str}. "
                f"I've posted the full breakdown to the chat."
            )
            # Chat card — one line per person.
            chat_lines = [
                f"👥 {description.capitalize()} — breakdown by person "
                f"(total {total}):"
            ]
            for name, count in per_person:
                if count < 0:
                    chat_lines.append(f"   • {name}: (lookup failed)")
                else:
                    chat_lines.append(f"   • {name}: {count}")
            chat = "\n".join(chat_lines)
            state.setdefault("jira_actions_taken", []).append({
                "kind": "search",
                "search_kind": "breakdown",
                "description": description,
                "total": total,
                "per_person": per_person,
                "speaker": speaker,
            })
            return speak, chat

        if kind == "count":
            total = jira_search_jql_total(jira_client, jql)
            if total < 0:
                # NEVER speak the raw JQL — it sounds robotic in TTS
                # ("project equals SHP and statusCategory not equals…").
                # The JQL is already logged to stderr in
                # jira_search_jql_total when the API call fails;
                # the developer can see it there for debugging.
                print(f"  [jira jql count error] (JQL was: {jql})")
                return (
                    "Sorry, I couldn't fetch that count from Jira right "
                    "now. You can ask me to try again in a moment.",
                    None,
                )
            # Friendly phrasing: singular vs plural.
            noun = "ticket" if total == 1 else "tickets"
            verb = "is" if total == 1 else "are"
            speak = f"There {verb} {total} {description.replace('tickets', noun)} on the SHP board."
            # Also post a tiny chat card so the user can verify which
            # board / team / sprint we queried. Plain English — never
            # the JQL. The link goes to the Jira issue-navigator with
            # the JQL pre-filled, so a click reveals the underlying
            # query if anyone wants to inspect it.
            from urllib.parse import quote
            nav_url = f"{base_url.rstrip('/')}/issues/?jql={quote(jql)}"
            chat = (
                f"🔢 {total} {description.replace('tickets', noun)} "
                f"on the SHP board\n"
                f"   <{nav_url}|open in Jira>"
            )
            state.setdefault("jira_actions_taken", []).append({
                "kind": "search",
                "search_kind": "count",
                "description": description,
                "total": total,
                "jql": jql,
                "speaker": speaker,
            })
            return speak, chat
        # kind == "list"
        issues = jira_search_jql(
            jira_client,
            jql,
            fields=["summary", "status", "assignee", "priority"],
            max_results=15,
        )
        if not issues:
            return (
                f"I didn't find any {description} on the SHP board.",
                None,
            )
        total = len(issues)
        noun = "ticket" if total == 1 else "tickets"
        # Build a compact chat card — one line per ticket.
        chat_lines = [
            f"📋 {total} {description.replace('tickets', noun)} (showing up to 15):"
        ]
        for issue in issues[:15]:
            key = issue.get("key", "?")
            fields = issue.get("fields") or {}
            summary_text = (fields.get("summary") or "(no summary)")[:80]
            status_obj = fields.get("status") or {}
            status = (
                status_obj.get("name", "Unknown")
                if isinstance(status_obj, dict)
                else "Unknown"
            )
            assignee_obj = fields.get("assignee") or {}
            assignee = (
                assignee_obj.get("displayName", "Unassigned")
                if isinstance(assignee_obj, dict) and assignee_obj
                else "Unassigned"
            )
            url = format_issue_url(base_url, key)
            chat_lines.append(
                f"   • <{url}|*{key}*> — {summary_text} "
                f"({status}, {assignee})"
            )
        chat = "\n".join(chat_lines)
        state.setdefault("jira_actions_taken", []).append({
            "kind": "search",
            "search_kind": "list",
            "description": description,
            "total": total,
            "jql": jql,
            "speaker": speaker,
        })
        speak = (
            f"Found {total} {description.replace('tickets', noun)}. "
            f"I've posted the list to the chat."
        )
        return speak, chat

    return "", None


def build_slack_summary(
    started_at: datetime,
    ended_at: datetime,
    summary: dict,
    participants: Optional[List[str]] = None,
    dry_run: bool = False,
    jira_actions_taken: Optional[List[dict]] = None,
    jira_base_url: str = "",
) -> str:
    """
    Assemble the Slack mrkdwn post body. Pure function (no I/O) so we can
    iterate on the format in --dry-run cheaply.

    Slack mrkdwn isn't standard markdown:
      - bold is *single* asterisks, not **double**.
      - headings (#, ##) render as plain text - we use *bold* for section
        headers instead.
      - bullet lists with "- " work fine.

    Sections (in order, conditionally shown):
      header (title)
      Duration         (always)
      Participants     (omitted if list is empty)
      Agenda           (omitted if empty — 1-2 sentence LLM-derived
                        description; richer than the in-call state.agenda)
      Summary          (omitted if empty — LLM-generated 3-4 sentence
                        narrative recap; longer than Agenda)
      Action items     (always shown - "no items" placeholder if empty,
                        grouped per owner if present)
      Outcome reached  (omitted entirely if outcome is unknown AND no note)

    Deliberately dropped (user preference, 2026-05-20): the Call ID,
    Bot name+voice, and End reason lines. They were operational
    breadcrumbs that didn't add value for the audience reading the post.
    """
    duration_sec = max(0.0, (ended_at - started_at).total_seconds())
    if duration_sec < 60:
        duration_str = f"~{int(duration_sec)} seconds"
    else:
        duration_min = duration_sec / 60.0
        duration_str = f"~{duration_min:.1f} minutes"
    start_hm = started_at.strftime("%H:%M")
    end_hm = ended_at.strftime("%H:%M")
    date_str = started_at.strftime("%Y-%m-%d")

    title_prefix = "[dry-run] " if dry_run else ""
    lines: List[str] = [
        f"*{title_prefix}Meeting Summary — {date_str}*",
        "",
        f"*Duration:* {duration_str} ({start_hm} – {end_hm} local)",
    ]
    if participants:
        lines.append(f"*Participants:* {', '.join(participants)}")
    lines.append("")

    agenda = summary.get("agenda", "").strip()
    if agenda:
        lines.append("*Agenda*")
        lines.append(agenda)
        lines.append("")

    discussion_summary = summary.get("discussion_summary", "").strip()
    if discussion_summary:
        lines.append("*Summary*")
        lines.append(discussion_summary)
        lines.append("")

    lines.append("*Action items*")
    action_md = summary.get("action_items_md", "").strip()
    if action_md:
        lines.append(action_md)
    else:
        lines.append("_No action items captured this call._")
    lines.append("")

    # Block 4 mid-meeting: list everything the bot did in Jira during
    # the call. Only emit the section if there's something to show, so
    # non-Jira runs are unchanged. Order: in the order actions fired.
    if jira_actions_taken:
        lines.append("*Actions Taken*")
        for entry in jira_actions_taken:
            issue = entry.get("issue") or {}
            kind = entry.get("kind", "create")
            key = issue.get("key", "?")
            fields = issue.get("fields") or {}
            summary_text = fields.get("summary") or "(no summary)"
            url = format_issue_url(jira_base_url, key)
            verb = {
                "create": "Created",
                "edit": "Updated",
                "lookup": "Looked up",
                "comment": "Commented on",
            }.get(kind, kind.capitalize())
            speaker = entry.get("speaker") or ""
            by_str = f" — requested by {speaker}" if speaker else ""
            # For edits, append the change list so the Slack reader
            # sees what changed without clicking through.
            change_str = ""
            if kind == "edit" and entry.get("changes"):
                change_str = " (" + ", ".join(
                    f"{k}={v}" for k, v in entry["changes"].items()
                ) + ")"
            elif kind == "comment" and entry.get("comment"):
                body_preview = entry["comment"]
                if len(body_preview) > 100:
                    body_preview = body_preview[:97] + "..."
                change_str = f" — \"{body_preview}\""
            lines.append(f"• {verb} <{url}|*{key}*> — {summary_text}{change_str}{by_str}")
        lines.append("")

    outcome_met = summary.get("outcome_met", "unknown").strip().lower()
    outcome_note = summary.get("outcome_note", "").strip()
    if outcome_met != "unknown" or outcome_note:
        verdict_label = {
            "yes": "Yes",
            "no": "No",
            "partial": "Partial",
            "unknown": "Unclear",
        }.get(outcome_met, "Unclear")
        lines.append(f"*Outcome reached:* {verdict_label}")
        if outcome_note:
            lines.append(outcome_note)

    return "\n".join(lines).rstrip() + "\n"


def save_call_log(
    call_id: str,
    transcript: List[dict],
    end_reason: str,
    output_file: Optional[str],
) -> None:
    """Save the conversation transcript to a markdown file."""
    now = datetime.now()
    filename = output_file or f"meeting-moderator-log-{now.strftime('%Y-%m-%d-%H%M')}.md"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# Meeting Moderator Log - {now.strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"**Call ID:** {call_id}  \n")
        f.write(f"**End reason:** {end_reason}  \n")
        f.write(f"**Turns:** {len(transcript)}  \n\n")
        f.write("---\n\n")
        f.write("## Conversation\n\n")
        for turn in transcript:
            role = turn["role"]
            speaker = turn.get("speaker", "")
            text = turn["text"]
            if role == "participant":
                f.write(f"**{speaker}:** {text}  \n\n")
            else:
                f.write(f"**Moderator:** {text}  \n\n")

    print(f"\nTranscript saved to: {filename}")


# ----------------------------------------------------------------------------
# Main event loop
# ----------------------------------------------------------------------------

async def run_moderator(
    call: dict,
    voice: str,
    bot_name: str,
    output_file: Optional[str],
    slack_channel: Optional[str] = None,
    state: Optional[dict] = None,
    jira_project: Optional[str] = None,
    jira_issue_type: str = "Task",
    jira_client=None,
    jira_roster: Optional[dict] = None,
    jira_sprints: Optional[List[dict]] = None,
    jira_sprints_by_team: Optional[Dict[str, List[dict]]] = None,
    jira_components: Optional[List[str]] = None,
) -> None:
    """
    Connect to the call's WebSocket and run the STT -> Gemini -> TTS loop.

    Event handling reference (from SKILL.md / bridge.py):
      - call.bot_ready          bot has joined and been admitted
      - participant.joined      a human joined (or the bot - filtered by name)
      - participant.left
      - transcript.final        finalized utterance from a speaker (THE event)
      - tts.started / tts.done  bracket the bot's own audio playback
      - call.ended              cleanup signal
    """
    call_id = call["call_id"]
    ws_url = call["ws_url"]
    if ws_url.startswith("https://"):
        ws_url = ws_url.replace("https://", "wss://")
    # The WebSocket endpoint authenticates via an `api_key` query string
    # (same pattern as examples/coding-companion/bridge.py). Without this,
    # the server returns HTTP 401 during the WS handshake and we never get
    # any events even though the REST call to create the bot succeeded.
    sep = "&" if "?" in ws_url else "?"
    ws_url = f"{ws_url}{sep}api_key={API_KEY}"

    # Share transcript + chat with the caller via a mutable `state` dict.
    # Why a state dict instead of returning these values?
    #   - On Ctrl+C, asyncio.run() raises KeyboardInterrupt without giving
    #     us a return value. main() still needs the transcript + chat to
    #     save the markdown log and post to Slack. The dict survives the
    #     exception because main() owns it.
    if state is None:
        state = {}
    transcript: List[dict] = state.setdefault("transcript", [])
    # Block 2 in-call tracking. These all live on state so main() (and
    # finalize_call) can read them even after a Ctrl+C cancellation.
    state.setdefault("agenda", [])
    action_items: List[dict] = state.setdefault("action_items", [])
    speaker_turns: dict = state.setdefault("speaker_turns", {})
    # joined_at_turn[name] = total turns in the room when that
    # participant first appeared. Used by _silent_nudge_candidates
    # to give late joiners their own 10-turn grace period instead
    # of nudging them the instant they walk into a room already
    # past the global threshold.
    joined_at_turn: dict = state.setdefault("joined_at_turn", {})
    # off_topic_streak tracks consecutive off-topic turns from the
    # SAME speaker. Resets on a topical turn or a speaker change. This
    # is what gates the redirect intervention (rule 2 in SYSTEM_PROMPT).
    state.setdefault("off_topic_streak", 0)
    state.setdefault("off_topic_last_speaker", None)
    # nudged_set keeps track of which silent participants we've already
    # invited to speak - we cap it at once per participant per meeting
    # to avoid badgering people.
    state.setdefault("nudged_set", set())
    state.setdefault("follow_up", None)
    # Raise-hand bookkeeping. None when the bot's hand is down; a dict
    # {"type": "off_topic"|"silent_nudge", "target": "<name or agenda>"}
    # when it has raised its hand and is waiting for someone to
    # acknowledge before delivering.
    state.setdefault("pending_intervention", None)
    # Block 4 (mid-meeting Jira intent): pending_jira_action is set
    # the moment a "Mod, create a ticket" is detected and cleared on
    # confirm/cancel. jira_actions_taken accumulates every successful
    # Jira interaction for the *Actions Taken* Slack section.
    state.setdefault("pending_jira_action", None)
    state.setdefault("last_jira_key", "")
    state.setdefault("jira_actions_taken", [])
    state.setdefault("capability_questions_asked", 0)

    # `is_speaking` defends against the bot's own TTS audio bleeding back
    # into STT. In direct mode, the bot's voice is injected into the Meet's
    # audio bus, so the STT engine can transcribe it. We skip transcript.final
    # events while we know we're speaking. (Same heuristic as support-agent.)
    is_speaking = False
    # We only greet once - on the first human that joins, not on every join.
    greeted = False
    end_reason = "unknown"

    # Build the Gemini chat session BEFORE connecting so the bot can greet
    # the moment a human joins without a cold-start delay. Store on state so
    # main() can reuse it for the end-of-call extract_action_items call even
    # if we get cancelled mid-stream.
    chat = build_chat_session()
    state["chat"] = chat

    # Redact the api_key query param in the printed URL so the key doesn't
    # end up in terminal scrollback / pasted bug reports.
    safe_ws_url = ws_url.split("?")[0] + "?api_key=***"
    print(f"Connecting to WebSocket: {safe_ws_url}")

    # Bump ping intervals from the websockets-library defaults (20s / 20s)
    # to give long Gemini turns more headroom. With asyncio.to_thread on
    # the blocking calls below, pings should always go through — but if
    # the event loop ever stalls again for any reason, we want the cap
    # to be 60s before AgentCall force-closes us (code 1006).
    async with websockets.connect(
        ws_url, ping_interval=30, ping_timeout=60, close_timeout=10,
    ) as ws:
        print("Connected. Waiting for bot to join...\n")

        try:
            async for msg in ws:
                event = json.loads(msg)
                # bridge.py normalises most events to {"event": ...} but a few
                # legacy ones use {"type": ...}. Check both, same as
                # support-agent does.
                event_type = event.get("event", event.get("type", ""))

                if event_type == "call.bot_ready":
                    # Bot is in the room but a human may not be yet. Don't
                    # speak to an empty room - wait for participant.joined.
                    print("Bot is in the meeting. Waiting for a participant...\n")

                elif event_type == "participant.joined":
                    name = speaker_name(event.get("name") or event.get("participant"))
                    print(f"  + {name} joined")
                    # Skip the bot's own join - otherwise it greets itself and
                    # the next (real) human walks into silence. `bot_name` is
                    # passed in from main() so we know what to compare against.
                    if name == bot_name:
                        continue
                    # Seed the joiner in speaker_turns at 0 so the moderator's
                    # per-turn context sees them even before they say anything.
                    # Critical for rule 3 (silent-participant nudge): without
                    # this, anyone who joins and stays quiet is invisible to
                    # the bot and can never be nudged. Use setdefault so a
                    # re-join (network blip) doesn't clobber an existing
                    # turn count.
                    speaker_turns.setdefault(name, 0)
                    # Capture the total-turn count at the moment they
                    # joined so _silent_nudge_candidates can apply a
                    # per-participant grace period (they don't get
                    # nudged until SILENT_NUDGE_GRACE turns have
                    # happened SINCE they walked in). setdefault keeps
                    # the original join-point if a network blip causes
                    # a duplicate participant.joined.
                    joined_at_turn.setdefault(
                        name, sum(speaker_turns.values())
                    )
                    if not greeted:
                        # First human in - ask Gemini to open the meeting via
                        # the moderator-structured path. SYSTEM_PROMPT's rule
                        # #1 will make the bot ask for the agenda. We DON'T
                        # hard-code "What's the agenda?" here - we let the
                        # prompt drive it so phrasing stays consistent with
                        # how the bot will sound on every subsequent turn.
                        greeted = True
                        prompt = (
                            f"Meeting just started. {name} is the first human in "
                            f"the room. No agenda is set yet. Open the meeting "
                            f"with a brief agenda-soliciting line - address "
                            f"{name} by their first name."
                        )
                        turn = ask_moderator_structured(chat, prompt)
                        # An opening should always speak (rule 1 in
                        # SYSTEM_PROMPT). If the model returned null,
                        # something's wrong with the prompt - log and
                        # continue silently rather than crashing.
                        if turn["speak"]:
                            print(f"  [bot] {turn['speak']}")
                            transcript.append({"role": "moderator", "text": turn["speak"]})
                            # NB: the WebSocket protocol uses "type", not
                            # "command" - the bridge scripts translate the
                            # stdin "command" form into "type" for the WS.
                            # Sending "command" here silently drops the
                            # message and no audio plays.
                            await ws.send(json.dumps({
                                "type": "tts.speak",
                                "text": turn["speak"],
                                "voice": voice,
                            }))
                        else:
                            print("  [bot] (no greeting returned by model — check prompt)")
                        merge_turn_into_state(state, turn)

                elif event_type == "participant.left":
                    leaver = speaker_name(event.get("name") or event.get("participant"))
                    print(f"  - {leaver} left")
                    # Remove from speaker_turns so the bot doesn't try to
                    # nudge someone who's no longer in the room. The nudged_set
                    # is small and harmless to leave alone (it just records
                    # historical nudges). Also drop their joined_at_turn
                    # entry so if they re-join later they get a fresh
                    # grace period starting at their re-join moment
                    # (otherwise the original join-point would carry
                    # over and they'd be immediately nudge-eligible).
                    if leaver != bot_name:
                        speaker_turns.pop(leaver, None)
                        joined_at_turn.pop(leaver, None)

                elif event_type == "tts.started":
                    is_speaking = True

                elif event_type == "tts.done":
                    is_speaking = False

                elif event_type in ("transcript.final", "user.message"):
                    # STT result for one speaker turn. (`user.message` is the
                    # bridge.py-flavoured event name; `transcript.final` is the
                    # API-flavoured one - same payload shape, accept both.)
                    speaker = speaker_name(event.get("speaker"))
                    text = (event.get("text") or "").strip()
                    if not text:
                        continue
                    if is_speaking:
                        # Don't reply to ourselves echoing through STT.
                        continue

                    print(f"  [you] ({speaker}) {text}")
                    transcript.append({"role": "participant", "speaker": speaker, "text": text})

                    # Bump this speaker's turn count BEFORE asking the model
                    # so the per-turn prompt sees up-to-date numbers.
                    # Defensive joined_at_turn seed for speakers we never
                    # saw a participant.joined event for (event lost,
                    # name mismatch, etc.). The grace value is moot for
                    # someone actively speaking — they're not silent —
                    # but the entry keeps state consistent and avoids a
                    # later KeyError if they go silent again.
                    if speaker not in speaker_turns:
                        joined_at_turn.setdefault(
                            speaker, sum(speaker_turns.values())
                        )
                    speaker_turns[speaker] = speaker_turns.get(speaker, 0) + 1

                    # Block 4 (mid-meeting Jira intent): BEFORE the
                    # Gemini moderator turn, scan this utterance for
                    # a Jira intent. If detected, we handle it
                    # deterministically (no LLM) and skip the
                    # moderator turn entirely. This keeps tickets
                    # under explicit human control and avoids the
                    # bot trying to "moderate" a clearly-actionable
                    # operational request.
                    if jira_project and jira_client is not None:
                        intent = detect_jira_intent(
                            text,
                            speaker,
                            bot_name,
                            state.get("pending_jira_action"),
                            project_key=jira_project,
                            last_key=state.get("last_jira_key", ""),
                        )
                        if intent is not None:
                            # Same blocking-call protection as the Gemini
                            # call below — Jira handlers can fire multiple
                            # synchronous Atlassian REST calls (especially
                            # the per-person breakdown, which loops N
                            # times), plus the Gemini-backed due-date and
                            # description summarisers. Offloading keeps
                            # the WS heartbeat alive.
                            speak_text, chat_text = await asyncio.to_thread(
                                handle_jira_action,
                                intent,
                                state,
                                jira_client=jira_client,
                                project_key=jira_project,
                                issue_type=jira_issue_type,
                                base_url=ATLASSIAN_BASE_URL,
                                host_email=ATLASSIAN_EMAIL,
                                roster=jira_roster or {},
                                speaker=speaker,
                                sprints=jira_sprints or [],
                            )
                            if speak_text:
                                print(f"  [bot:jira/{intent['action']}] {speak_text}")
                                transcript.append({"role": "moderator", "text": speak_text})
                                await ws.send(json.dumps({
                                    "type": "tts.speak",
                                    "text": speak_text,
                                    "voice": voice,
                                }))
                            if chat_text:
                                await ws.send(json.dumps({
                                    "type": "meeting.send_chat",
                                    "message": chat_text,
                                }))
                                print(f"  [jira] posted to Meet chat.")
                            # Skip the normal moderator pipeline for
                            # this utterance — the Jira intent is the
                            # whole response. Bookkeeping (speaker
                            # turn count, transcript append) already
                            # happened above.
                            continue

                    # Build the per-turn context the moderator sees. The
                    # idea is to give the model just enough state to make
                    # the five intervention decisions in SYSTEM_PROMPT
                    # WITHOUT re-sending the entire transcript (it's
                    # already in the chat's rolling history).
                    agenda_now = state.get("agenda") or []
                    agenda_line = ", ".join(agenda_now) if agenda_now else "NOT YET SET"
                    participants_line = ", ".join(sorted(speaker_turns.keys())) or "(none yet)"
                    turn_counts_line = ", ".join(
                        f"{n}={c}" for n, c in sorted(speaker_turns.items(), key=lambda kv: -kv[1])
                    ) or "(none yet)"
                    total_turns = sum(speaker_turns.values())
                    silent_list = [n for n, c in speaker_turns.items() if c == 0]
                    silent_line = ", ".join(sorted(silent_list)) or "(none)"
                    # Show ONLY the silent participants whose per-
                    # participant grace has elapsed. The system prompt
                    # rule 3 instructs the model to nudge from THIS
                    # list, not the broader "Silent participants" line.
                    # Without this, the model would try to raise hand
                    # for a late joiner the moment the global turn
                    # count crossed 10 — our state-machine guard
                    # would reject it, but the call costs tokens.
                    eligible_nudge_list = _silent_nudge_candidates(state)
                    eligible_nudge_line = (
                        ", ".join(sorted(eligible_nudge_list))
                        or "(none — either no one silent, all in grace, or all already nudged)"
                    )
                    streak_line = (
                        f"{state['off_topic_streak']} consecutive off-topic turn(s) by "
                        f"{state['off_topic_last_speaker']}"
                        if state.get("off_topic_streak") and state.get("off_topic_last_speaker")
                        else "0 (no current streak)"
                    )
                    nudged_line = ", ".join(sorted(state["nudged_set"])) or "(none yet)"
                    pending_line = _format_pending_for_prompt(state.get("pending_intervention"))
                    # Show the model what's already captured so it
                    # doesn't restate the same commitment with slightly
                    # different wording. Empty string when nothing is
                    # captured yet — keeps the prompt compact early on.
                    existing_items_block = _format_existing_action_items(
                        state.get("action_items") or []
                    )

                    # Capability-question counter: if this utterance reads
                    # as "what can you do?", bump the counter and feed
                    # a variation directive into the prompt so the model
                    # doesn't recite the same answer twice.
                    if _is_capability_question(text, bot_name):
                        state["capability_questions_asked"] = (
                            state.get("capability_questions_asked", 0) + 1
                        )
                    cap_directive = _format_capability_variation_directive(
                        state.get("capability_questions_asked", 0)
                    )

                    prompt = (
                        # Today's date is the anchor for resolving any
                        # relative due dates the speaker mentions ("by
                        # Friday", "end of Q3", "28th of May"). Without
                        # it, Gemini defaults to a year near its training
                        # cutoff (e.g. 2024) instead of the current year.
                        f"Today's date: {datetime.now():%Y-%m-%d (%A)}\n"
                        # Meeting-elapsed time lets the model answer
                        # time-relative recap questions accurately. Each
                        # per-turn prompt accumulates a timestamp in the
                        # chat history, so the model can infer silent
                        # gaps from consecutive deltas and bound "the
                        # last 5 minutes" against current elapsed.
                        f"Meeting elapsed: {_format_elapsed(state['started_at'])}\n"
                        f"Agenda: {agenda_line}\n"
                        f"Participants in this call: {participants_line}\n"
                        f"Speaker turn counts so far: {turn_counts_line}\n"
                        f"Total participant turns: {total_turns}\n"
                        f"Silent participants (0 turns): {silent_line}\n"
                        f"Silent participants eligible for nudge: {eligible_nudge_line}\n"
                        f"Off-topic streak: {streak_line}\n"
                        f"Already nudged (do NOT re-nudge): {nudged_line}\n"
                        f"Pending intervention: {pending_line}\n"
                        + cap_directive
                        + existing_items_block +
                        f"Current speaker: {speaker}\n"
                        f"They just said: {text}"
                    )

                    # CRITICAL: chat.send_message is SYNC and can take
                    # 5-15s on gemini-3.5-flash with our long prompts.
                    # Calling it directly blocks the asyncio event loop —
                    # which means the websockets library can't reply to
                    # server pings. After ~20s of no pong, AgentCall
                    # tears down the WS with code 1006 (abnormal close)
                    # even though the call itself is still healthy.
                    # Run it on a worker thread so the event loop stays
                    # free to handle pings, events, and Jira API calls.
                    turn = await asyncio.to_thread(ask_moderator_structured, chat, prompt)

                    # Update the off-topic streak BEFORE merging the rest of
                    # the response. Streak machine semantics (group-wide):
                    #   - ANY off_topic turn -> streak grows, regardless of
                    #     who's speaking. A meeting drifting across Bob
                    #     -> Alice -> Daniyal is just as much a moderator
                    #     problem as one person rambling.
                    #   - off_topic_last_speaker tracks the MOST RECENT
                    #     drifter for context, not for gating - the model
                    #     uses it to phrase the redirect ("Bob, if I may..").
                    #   - NOT off_topic -> streak resets to 0.
                    if turn["off_topic"]:
                        state["off_topic_streak"] = (state["off_topic_streak"] or 0) + 1
                        state["off_topic_last_speaker"] = speaker
                    else:
                        state["off_topic_streak"] = 0
                        state["off_topic_last_speaker"] = None

                    # Merge agenda / action items / follow-up suggestions.
                    merge_turn_into_state(state, turn)

                    # Set / clear the raise-hand pending state based on
                    # which intent the model just emitted. Must happen
                    # AFTER merge so streak/silent counters are current.
                    update_pending_intervention(state, turn)

                    # Per-turn state diagnostics — mirror of run_dry's
                    # [state] line so live testing shows the same
                    # bot-internal counters. Useful for debugging "why
                    # did it (not) intervene" in real meetings.
                    print(_format_state_diagnostics(state, turn))

                    # Raise-hand branch: native Meet ✋ reaction, no TTS.
                    # ask_moderator_structured guarantees speak is None
                    # when intent == "raise_hand", so we route this
                    # branch BEFORE the "no speak -> silent" early-out.
                    if turn.get("intent") == "raise_hand":
                        await ws.send(json.dumps({"type": "meeting.raise_hand"}))
                        print("  [bot:raise_hand] ✋ (hand raised in Meet, no TTS)")
                        transcript.append({"role": "moderator", "text": "✋ [hand raised]"})
                        continue

                    # Speak only if the model decided to. Most turns this is
                    # None and we stay silent - exactly the behavior change
                    # Block 2 is shipping for.
                    if not turn["speak"]:
                        print("  [bot] (listening — no intervention)")
                        continue

                    # If the model just nudged a silent participant, mark
                    # them as nudged so we don't badger them next turn.
                    # We use a heuristic: any participant who's been
                    # mentioned by name in the speak text and currently has
                    # turn-count <= 1 was probably the target of the nudge.
                    # (Only fires on intent="nudge" — raise_hand has no
                    # speak text to name anyone in.)
                    if turn.get("intent") == "nudge":
                        for participant_name in list(speaker_turns.keys()):
                            first_name = participant_name.split()[0]
                            if (
                                speaker_turns.get(participant_name, 0) <= 1
                                and first_name.lower() in turn["speak"].lower()
                            ):
                                state["nudged_set"].add(participant_name)

                    intent_tag = turn.get("intent") or "speak"
                    print(f"  [bot:{intent_tag}] {turn['speak']}")
                    transcript.append({"role": "moderator", "text": turn["speak"]})

                    # Try to lower the hand BEFORE speaking when this turn
                    # is delivering a deferred intervention. AgentCall's
                    # API doesn't expose meeting.lower_hand, but Meet's
                    # underlying behavior tends to toggle on a second
                    # meeting.raise_hand. Worst case (no toggle): this is
                    # a no-op and the hand stays up. Best case: the hand
                    # drops as the bot starts talking, mirroring a real
                    # acknowledged-and-speaking handoff.
                    if turn.get("intent") in ("redirect", "nudge"):
                        await ws.send(json.dumps({"type": "meeting.raise_hand"}))

                    await ws.send(json.dumps({
                        "type": "tts.speak",
                        "text": turn["speak"],
                        "voice": voice,
                    }))

                elif event_type == "call.ended":
                    end_reason = event.get("reason", "unknown")
                    print(f"\nCall ended: {end_reason}")
                    break

                else:
                    # Filter out the very noisy events that fire many times
                    # per second and don't give us useful debugging signal.
                    # command.ack         - one per token-streamed TTS chunk
                    # active_speaker      - fires every time the mic level
                    #                       crosses a threshold
                    # transcript.partial  - mid-utterance STT updates (we only
                    #                       act on transcript.final)
                    NOISY = {"command.ack", "active_speaker", "transcript.partial"}
                    if event_type and event_type not in NOISY:
                        print(f"  [event] {event_type}")
        except websockets.ConnectionClosed as close_exc:
            end_reason = "ws_closed"
            close_code = getattr(close_exc, "code", None)
            close_reason = getattr(close_exc, "reason", "") or ""
            print(f"\nWebSocket closed. (code={close_code}, reason={close_reason!r})")
            # If AgentCall hung up without sending a `call.ended` event,
            # ask the REST API directly so the user sees WHY in the
            # terminal instead of having to dig through the dashboard.
            try:
                detail = requests.get(
                    f"{API_BASE}/v1/calls/{call_id}",
                    headers=HEADERS,
                    timeout=5,
                )
                if detail.ok:
                    info = detail.json() or {}
                    server_status = info.get("status", "?")
                    server_reason = info.get("end_reason") or info.get("status_reason") or ""
                    ended_at = info.get("ended_at") or ""
                    print(
                        f"  [agentcall status] status={server_status} "
                        f"end_reason={server_reason!r} ended_at={ended_at}"
                    )
                    end_reason = server_reason or end_reason
                else:
                    print(f"  [agentcall status] HTTP {detail.status_code}: {detail.text[:200]}")
            except Exception as exc:
                print(f"  [agentcall status] lookup failed: {exc}")

    # Mirror end_reason onto state for main() to pick up. We don't run
    # save_call_log or post to Slack here anymore - that's main()'s job,
    # inside a try/finally so Ctrl+C can't skip it. (See finalize_call()
    # below.)
    state["end_reason"] = end_reason


def finalize_call(
    call_id: str,
    state: dict,
    output_file: Optional[str],
    slack_channel: Optional[str],
) -> None:
    """
    Run the post-call cleanup: transcript markdown + Slack post.

    Pulled out of run_moderator() so main() can call it from a try/finally
    block. The KeyboardInterrupt that fires on Ctrl+C propagates out of
    asyncio.run() and would otherwise skip any code that lived after the
    WebSocket loop inside run_moderator(). Living in main() means this
    fires no matter how the call ended: normal exit, ws_closed, or Ctrl+C.

    Idempotent for accidental double-calls (e.g. if the state dict somehow
    got passed in twice) - we look for a sentinel key and bail.

    state contract (all optional, finalize_call fills in safe defaults):
      transcript    list[dict]  - rolling transcript
      chat                       - Gemini chat session
      end_reason    str          - "call.ended" reason / "ws_closed" / "interrupted"
      started_at    datetime     - when the call was created (for duration)
      bot_name      str          - display name (for the Slack header)
      voice         str          - TTS voice (for the Slack header)
    """
    if state.get("_finalized"):
        return
    state["_finalized"] = True

    transcript = state.get("transcript") or []
    chat = state.get("chat")
    end_reason = state.get("end_reason", "unknown")
    started_at = state.get("started_at") or datetime.now()
    ended_at = datetime.now()
    bot_name = state.get("bot_name", "Moderator")
    voice = state.get("voice", "unknown")

    # Normalize captured action-item owners against the real participant
    # list before anything else looks at state["action_items"]. STT
    # mishears names ("Aman" instead of "Yaman") and the model's
    # speaker-inference can also produce phantom names; this step
    # collapses them onto actual people in the call so the Slack post
    # never shows an unknown owner section.
    participants = [p for p in state.get("speaker_turns", {}).keys() if p != bot_name]
    if state.get("action_items") and participants:
        state["action_items"] = normalize_action_item_owners(
            state["action_items"], participants
        )
    # Bug 4 fix: "You" is the dry-run fallback speaker when no name
    # prefix is supplied. In live mode STT always provides a real
    # name, so "You" never appears. Filter it out of the Slack
    # participants display so accidental no-prefix dry-run lines
    # don't leak into the meeting summary.
    display_participants = [p for p in participants if p != "You"]

    save_call_log(call_id, transcript, end_reason, output_file)

    if not slack_channel:
        return
    if not SLACK_BOT_TOKEN:
        print("  --slack-channel was passed but SLACK_BOT_TOKEN is not set. Skipping Slack post.")
        return
    if not transcript:
        # Still send a heads-up - it's useful to know a call started and
        # ended even if nothing was said.
        body = build_slack_summary(
            started_at, ended_at, dict(_EMPTY_SUMMARY),
            participants=display_participants,
            jira_actions_taken=state.get("jira_actions_taken") or None,
            jira_base_url=ATLASSIAN_BASE_URL,
        )
        post_action_items(SLACK_BOT_TOKEN, slack_channel, body)
        return

    # Block 2 onward: extract_meeting_summary prefers state-tracked
    # agenda + action items (no hallucination risk) and only falls back
    # to a one-shot Gemini call for outcome verdict + anything state
    # didn't capture. No `chat` dependency - the chat is committed to
    # the moderator schema and can't dual-purpose for summary output.
    summary = extract_meeting_summary(transcript, state=state)
    body = build_slack_summary(
        started_at, ended_at, summary, participants=display_participants,
        jira_actions_taken=state.get("jira_actions_taken") or None,
        jira_base_url=ATLASSIAN_BASE_URL,
    )
    post_action_items(SLACK_BOT_TOKEN, slack_channel, body)


def run_dry(
    bot_name: str,
    slack_channel: Optional[str] = None,
    jira_project: Optional[str] = None,
    jira_issue_type: str = "Task",
    jira_client=None,
    jira_roster: Optional[dict] = None,
    jira_sprints: Optional[List[dict]] = None,
    jira_sprints_by_team: Optional[Dict[str, List[dict]]] = None,
    jira_components: Optional[List[str]] = None,
) -> None:
    """
    Interactive rehearsal mode - no AgentCall call, no WebSocket, no TTS.

    Reads your typed lines from stdin and runs each one through the SAME
    structured moderator pipeline the live bot uses. Prints the bot's
    decision per turn so you can see when it intervenes, when it stays
    silent, and what state it has accumulated.

    Free to run: the Gemini Flash free tier is 1,500 requests/day, and no
    AgentCall credits are spent.

    Suggested test scenarios to validate the prompt:
      - agenda-setting: first turn says "today we want to plan the Q3
        launch." -> bot should NOT respond verbally, but agenda_items
        should populate.
      - action-item capture: "Alice will own the pricing audit by
        Friday." -> bot stays silent, new_action_items populates.
      - off-topic redirect: type 2 unrelated lines back-to-back ("how
        was your weekend?" / "did you watch the game?") -> bot redirects
        on the second turn.
      - direct address: "Juno, summarize what we have so far." -> bot
        speaks.

    At exit (empty line or Ctrl+C), the same Slack post the live path
    would produce is built and (if --slack-channel was passed) posted.
    """
    print(f"[dry-run] {bot_name} is ready. Type a line and press Enter.")
    print("[dry-run] Empty line or Ctrl+C to exit.\n")
    print("[dry-run] Tip: prepend a name + colon to speak as someone, e.g. 'Bob: hi all.'")
    print("[dry-run] No name prefix = you speak as the resolved host (see line after the agenda greeting).")
    print("[dry-run] To simulate someone joining the room SILENTLY (for rule 3 testing),")
    print("[dry-run] type '/join Alice' — they'll appear at 0 turns in the bot's context.\n")

    chat = build_chat_session()

    # Mirror the live state shape so the per-turn prompt enrichment and
    # the end-of-call summary work identically.
    transcript: List[dict] = []
    state: dict = {
        "transcript": transcript,
        "agenda": [],
        "action_items": [],
        "speaker_turns": {},
        # See run_moderator for joined_at_turn semantics — same dict,
        # populated by /join in this path.
        "joined_at_turn": {},
        "off_topic_streak": 0,
        "off_topic_last_speaker": None,
        "nudged_set": set(),
        "follow_up": None,
        "pending_intervention": None,
        # Block 4 (mid-meeting Jira intent): same shape as run_moderator.
        "pending_jira_action": None,
        "last_jira_key": "",
        "jira_actions_taken": [],
        # Capability-question counter (bumped each time the user
        # asks "what can you do?"; per-turn prompt uses this to
        # vary the answer wording on repeat asks).
        "capability_questions_asked": 0,
        "chat": chat,
    }
    started_at = datetime.now()
    # Expose started_at through state so the per-turn prompt builder
    # (which only sees `state`) can compute meeting-elapsed time for
    # time-relative questions like "what did we discuss in the last
    # 5 minutes?".
    state["started_at"] = started_at

    # Opening turn: same agenda-soliciting prompt the live bot uses.
    opening_prompt = (
        "Meeting just started. A participant just joined. No agenda is "
        "set yet. Open the meeting with a brief agenda-soliciting line."
    )
    opening_turn = ask_moderator_structured(chat, opening_prompt)
    if opening_turn["speak"]:
        print(f"[bot] {opening_turn['speak']}\n")
        transcript.append({"role": "moderator", "text": opening_turn["speak"]})
    merge_turn_into_state(state, opening_turn)

    # Resolve the host's display name from ATLASSIAN_EMAIL + roster
    # so no-prefix dry-run lines get attributed to the actual human
    # (e.g. "Daniyal") rather than a generic "You" placeholder.
    # Falls back to "You" when the roster doesn't resolve.
    host_name = _resolve_host_display_name(ATLASSIAN_EMAIL, jira_roster or {})
    if host_name != "You":
        print(f"[dry-run] No-prefix lines will be attributed to '{host_name}' (resolved from ATLASSIAN_EMAIL).\n")

    def _parse_speaker(line: str) -> tuple:
        """Split 'Bob: hello' -> ('Bob', 'hello'), else (<host_name>, line)."""
        if ":" in line:
            head, _, rest = line.partition(":")
            head = head.strip()
            rest = rest.strip()
            if head and rest and len(head.split()) <= 3:
                return head, rest
        return host_name, line

    try:
        while True:
            try:
                user_text = input("> ").strip()
            except EOFError:
                break
            if not user_text:
                break

            # /join <Name> simulates someone joining the room silently - the
            # live path gets this for free via participant.joined events from
            # AgentCall, but dry-run has no event stream so we expose it as a
            # slash command. Critical for rule 3 testing (silent-participant
            # nudge): without it, the bot has no way to know a quiet
            # participant exists.
            if user_text.startswith("/join "):
                joinee = user_text[len("/join "):].strip()
                if joinee:
                    state["speaker_turns"].setdefault(joinee, 0)
                    # Mirror the live participant.joined seed: capture
                    # the total-turn count at join time so
                    # _silent_nudge_candidates can give this person
                    # their own SILENT_NUDGE_GRACE buffer instead of
                    # nudging on the very next turn.
                    state["joined_at_turn"].setdefault(
                        joinee, sum(state["speaker_turns"].values())
                    )
                    print(f"[state] +participant {joinee} (silent, 0 turns)\n")
                else:
                    print("[dry-run] usage: /join <Name>\n")
                continue

            speaker, text = _parse_speaker(user_text)
            transcript.append({"role": "participant", "speaker": speaker, "text": text})
            # Defensive joined_at_turn seed for speakers who weren't
            # explicitly /join'd before they spoke. See the matching
            # block in run_moderator for full reasoning.
            if speaker not in state["speaker_turns"]:
                state["joined_at_turn"].setdefault(
                    speaker, sum(state["speaker_turns"].values())
                )
            state["speaker_turns"][speaker] = state["speaker_turns"].get(speaker, 0) + 1

            # Block 4 (mid-meeting Jira intent): same hook as
            # run_moderator. Detect deterministically; if it fires,
            # simulate the bot speech + Meet chat post inline and
            # skip the Gemini moderator turn for this utterance.
            if jira_project and jira_client is not None:
                intent = detect_jira_intent(
                    text,
                    speaker,
                    bot_name,
                    state.get("pending_jira_action"),
                    project_key=jira_project,
                    last_key=state.get("last_jira_key", ""),
                )
                if intent is not None:
                    speak_text, chat_text = handle_jira_action(
                        intent,
                        state,
                        jira_client=jira_client,
                        project_key=jira_project,
                        issue_type=jira_issue_type,
                        base_url=ATLASSIAN_BASE_URL,
                        host_email=ATLASSIAN_EMAIL,
                        roster=jira_roster or {},
                        speaker=speaker,
                        sprints=jira_sprints or [],
                        sprints_by_team=jira_sprints_by_team or {},
                        components_list=jira_components or [],
                    )
                    # Bug 1 fix: emit a compact state line so the user
                    # can see speaker_turns ticking up during Jira-only
                    # exchanges. Without this, count appears to "jump"
                    # between the previous and the next non-Jira turn.
                    turn_counts_line = ", ".join(
                        f"{n}={c}" for n, c in sorted(
                            state["speaker_turns"].items(),
                            key=lambda kv: -kv[1],
                        )
                    ) or "(none yet)"
                    print(
                        f"[state] turns({turn_counts_line}) | jira={intent['action']}"
                    )
                    if speak_text:
                        print(f"[bot:jira/{intent['action']}] {speak_text}\n")
                        transcript.append({"role": "moderator", "text": speak_text})
                    if chat_text:
                        # In dry-run there's no Meet to post to, so
                        # surface the chat message in the terminal
                        # under a clear label so the user can spot it.
                        print(f"[meet-chat]\n{chat_text}\n")
                    continue

            agenda_now = state.get("agenda") or []
            agenda_line = ", ".join(agenda_now) if agenda_now else "NOT YET SET"
            participants_line = ", ".join(sorted(state["speaker_turns"].keys())) or "(none yet)"
            turn_counts_line = ", ".join(
                f"{n}={c}" for n, c in sorted(state["speaker_turns"].items(), key=lambda kv: -kv[1])
            ) or "(none yet)"
            total_turns = sum(state["speaker_turns"].values())
            silent_list = [n for n, c in state["speaker_turns"].items() if c == 0]
            silent_line = ", ".join(sorted(silent_list)) or "(none)"
            # See run_moderator for why we surface this separately —
            # the model should nudge ONLY from the per-participant
            # grace-filtered list, not the broader silent list.
            eligible_nudge_list = _silent_nudge_candidates(state)
            eligible_nudge_line = (
                ", ".join(sorted(eligible_nudge_list))
                or "(none — either no one silent, all in grace, or all already nudged)"
            )
            streak_line = (
                f"{state['off_topic_streak']} consecutive off-topic turn(s) by "
                f"{state['off_topic_last_speaker']}"
                if state.get("off_topic_streak") and state.get("off_topic_last_speaker")
                else "0 (no current streak)"
            )
            nudged_line = ", ".join(sorted(state["nudged_set"])) or "(none yet)"
            pending_line = _format_pending_for_prompt(state.get("pending_intervention"))
            existing_items_block = _format_existing_action_items(
                state.get("action_items") or []
            )
            # Mirror of the live path: bump the capability-question
            # counter and inject a variation directive when the user
            # asks more than once.
            if _is_capability_question(text, bot_name):
                state["capability_questions_asked"] = (
                    state.get("capability_questions_asked", 0) + 1
                )
            cap_directive = _format_capability_variation_directive(
                state.get("capability_questions_asked", 0)
            )
            prompt = (
                # See live-path prompt for why we ship today's date and
                # meeting-elapsed time.
                f"Today's date: {datetime.now():%Y-%m-%d (%A)}\n"
                f"Meeting elapsed: {_format_elapsed(state['started_at'])}\n"
                f"Agenda: {agenda_line}\n"
                f"Participants in this call: {participants_line}\n"
                f"Speaker turn counts so far: {turn_counts_line}\n"
                f"Total participant turns: {total_turns}\n"
                f"Silent participants (0 turns): {silent_line}\n"
                f"Silent participants eligible for nudge: {eligible_nudge_line}\n"
                f"Off-topic streak: {streak_line}\n"
                f"Already nudged (do NOT re-nudge): {nudged_line}\n"
                f"Pending intervention: {pending_line}\n"
                + cap_directive
                + existing_items_block +
                f"Current speaker: {speaker}\n"
                f"They just said: {text}"
            )

            turn = ask_moderator_structured(chat, prompt)

            # Group-wide streak: any off-topic turn from anyone grows it.
            # See the matching block in run_moderator for full reasoning.
            if turn["off_topic"]:
                state["off_topic_streak"] = (state["off_topic_streak"] or 0) + 1
                state["off_topic_last_speaker"] = speaker
            else:
                state["off_topic_streak"] = 0
                state["off_topic_last_speaker"] = None

            merge_turn_into_state(state, turn)

            # Set / clear the raise-hand pending state based on intent.
            update_pending_intervention(state, turn)

            # If the model just nudged a silent participant, mark them
            # as nudged. Only fires on intent="nudge" (the actual
            # delivery phase) - intent="raise_hand" doesn't name anyone
            # yet, so it'd be a false positive.
            if turn.get("intent") == "nudge" and turn.get("speak"):
                for participant_name in list(state["speaker_turns"].keys()):
                    first_name = participant_name.split()[0]
                    if (
                        state["speaker_turns"].get(participant_name, 0) <= 1
                        and first_name.lower() in turn["speak"].lower()
                    ):
                        state["nudged_set"].add(participant_name)

            # Per-turn diagnostics - identical formatter the live path uses.
            print(_format_state_diagnostics(state, turn).lstrip())

            # Raise-hand: silent in dry-run since there's no Meet to
            # send meeting.raise_hand to. Still surface it in the
            # terminal so the user can see the bot took the action.
            if turn.get("intent") == "raise_hand":
                print("[bot:raise_hand] ✋ (would raise hand in Meet — silent)\n")
                transcript.append({"role": "moderator", "text": "✋ [hand raised]"})
                continue

            if not turn["speak"]:
                print("[bot] (silent)\n")
                continue

            intent_tag = turn.get("intent") or "speak"
            print(f"[bot:{intent_tag}] {turn['speak']}\n")
            transcript.append({"role": "moderator", "text": turn["speak"]})
    except KeyboardInterrupt:
        print()  # newline after ^C

    # Slack-on-dry-run lets us validate the full delivery path - now
    # exercising the same rich-summary builder the live path uses.
    if slack_channel:
        if not SLACK_BOT_TOKEN:
            print("[dry-run] --slack-channel was passed but SLACK_BOT_TOKEN is not set. Skipping Slack post.")
        elif not transcript:
            print("[dry-run] Nothing to post to Slack (empty session).")
        else:
            # Normalize captured owners against the real (dry-run)
            # participant list before formatting, same as the live
            # finalize_call path.
            participants = [p for p in state.get("speaker_turns", {}).keys() if p != bot_name]
            if state.get("action_items") and participants:
                state["action_items"] = normalize_action_item_owners(
                    state["action_items"], participants
                )
            # Bug 4 fix: filter the dry-run "You" fallback speaker
            # out of the displayed participant list. Same reasoning
            # as the finalize_call path.
            display_participants = [p for p in participants if p != "You"]
            summary = extract_meeting_summary(transcript, state=state)
            body = build_slack_summary(
                started_at=started_at,
                ended_at=datetime.now(),
                summary=summary,
                participants=display_participants,
                dry_run=True,
                jira_actions_taken=state.get("jira_actions_taken") or None,
                jira_base_url=ATLASSIAN_BASE_URL,
            )
            post_action_items(SLACK_BOT_TOKEN, slack_channel, body)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AgentCall Meeting Moderator - Gemini STT->LLM->TTS round-trip",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples/meeting-moderator/moderator.py "https://meet.google.com/abc-def-ghi"
  python examples/meeting-moderator/moderator.py "https://meet.google.com/abc" --name "Mod" --voice bm_george
  python examples/meeting-moderator/moderator.py "https://meet.google.com/abc" --output run.md
  python examples/meeting-moderator/moderator.py --dry-run                # prompt-tuning, no credits
        """,
    )
    # meet_url is positional but optional - --dry-run doesn't need a Meet.
    # We validate the "must be set unless dry-run" rule manually after parsing.
    parser.add_argument("meet_url", nargs="?", default=None,
                        help="Meeting URL (Google Meet, Zoom, or Teams). Required unless --dry-run.")
    parser.add_argument("--name", default="Mod", help="Bot display name (default: Mod)")
    parser.add_argument(
        "--voice",
        default="bm_george",
        help="TTS voice ID (default: bm_george - British male). Always sent per-tts.speak; the startup flag alone is unreliable.",
    )
    parser.add_argument("--output", default=None, help="Transcript output filename")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Skip AgentCall entirely. Run an interactive Gemini chat session "
            "from the terminal using the same system prompt the real bot uses. "
            "Free; useful for prompt-tuning and edge-case testing before "
            "spending credits on a live Meet call."
        ),
    )
    parser.add_argument(
        "--avatar",
        nargs="?",
        const="avatar",
        default=None,
        choices=["avatar", "orb", "ring", "voice-agent", "dashboard", "blank"],
        help=(
            "Show a visual avatar in the bot's video tile (uses webpage-av "
            "mode, ~2-3x the per-minute cost of audio-only). Pass --avatar "
            "with no value to use the default 'avatar' template, or pick: "
            "avatar | orb | ring | voice-agent | dashboard | blank. Omit "
            "the flag entirely to stay in cheap audio-only mode."
        ),
    )
    parser.add_argument(
        "--slack-channel",
        default=None,
        help=(
            "Slack channel ID (e.g. C0123ABCD) or name (e.g. "
            "#moderator-action-items) to post the action-item summary to "
            "after the call. Requires SLACK_BOT_TOKEN in env / .env. Omit "
            "to skip Slack entirely."
        ),
    )
    parser.add_argument(
        "--show-jira-ping",
        action="store_true",
        help=(
            "Run a one-shot Jira auth check and exit. Calls /rest/api/3/myself "
            "with the ATLASSIAN_EMAIL + ATLASSIAN_API_TOKEN credentials from "
            ".env, prints the authenticated user's display name, and exits. "
            "Does NOT create a call or use any AgentCall credits. Use this "
            "to confirm the three ATLASSIAN_* env vars are correct before "
            "running with --jira-project."
        ),
    )
    parser.add_argument(
        "--jira-project",
        default=None,
        help=(
            "Jira project key (e.g. SHP) to create tickets in at end-of-call. "
            "One Task is created per captured action item; assignee resolved "
            "from team_roster.yaml (first-name lookup, case-insensitive) with "
            "fallback to the call host (= ATLASSIAN_EMAIL). Requires all three "
            "ATLASSIAN_* env vars to be set. Omit to skip Jira entirely."
        ),
    )
    parser.add_argument(
        "--jira-issue-type",
        default="Task",
        help=(
            "Jira issue type to create (default: Task). Most projects accept "
            "Task; some teams prefer Story or Sub-task. If the target project "
            "doesn't have this type, the API returns a 400 and the ticket is "
            "skipped — other tickets continue."
        ),
    )
    parser.add_argument(
        "--jira-roster",
        default=os.path.join(
            os.path.dirname(__file__) or ".", "team_roster.yaml"
        ),
        help=(
            "Path to the team-roster YAML mapping first-name -> work email. "
            "Defaults to examples/meeting-moderator-v2/team_roster.yaml. A "
            "missing file is fine — every owner just falls back to the call "
            "host email."
        ),
    )
    parser.add_argument(
        "--jira-board-id",
        type=int,
        default=41,
        help=(
            "Jira board ID for sprint lookups. Defaults to 41 (Shypple SHP "
            "board). At startup we cache the list of sprints so 'current "
            "sprint' / 'sprint 92' / 'sprint by name' references resolve "
            "without an extra API roundtrip per request. Pass 0 to skip "
            "sprint caching (sprint commands will then never resolve)."
        ),
    )
    args = parser.parse_args()

    # --show-jira-ping is a standalone sanity check. We handle it BEFORE
    # touching AgentCall or Gemini so a credential typo can't get masked
    # by a downstream error from a different subsystem.
    if args.show_jira_ping:
        missing = [
            n for n, v in (
                ("ATLASSIAN_EMAIL", ATLASSIAN_EMAIL),
                ("ATLASSIAN_API_TOKEN", ATLASSIAN_API_TOKEN),
                ("ATLASSIAN_BASE_URL", ATLASSIAN_BASE_URL),
            ) if not v
        ]
        if missing:
            print(f"Error: --show-jira-ping requires these env vars: {', '.join(missing)}")
            print("       Add them to .env at the repo root. See:")
            print("       https://id.atlassian.com/manage-profile/security/api-tokens")
            sys.exit(1)
        try:
            client = make_jira_client(
                ATLASSIAN_EMAIL, ATLASSIAN_API_TOKEN, ATLASSIAN_BASE_URL
            )
            me = jira_ping(client)
        except JiraClientError as exc:
            print(f"Error: {exc}")
            sys.exit(1)
        # /myself returns: accountId, displayName, emailAddress, active, ...
        # Print a compact summary so the user can confirm both auth and
        # that they're talking to the right Atlassian instance.
        print(
            f"OK: connected to {ATLASSIAN_BASE_URL} as "
            f"{me.get('displayName', '?')} <{me.get('emailAddress', '?')}> "
            f"(accountId={me.get('accountId', '?')})"
        )
        sys.exit(0)

    # Fail fast if Slack was requested but the token is missing. Catching
    # this here is much friendlier than letting the call run for 5 minutes,
    # paying for credits, and *then* discovering the post can't go through.
    if args.slack_channel and not SLACK_BOT_TOKEN:
        print("Error: --slack-channel was passed but SLACK_BOT_TOKEN is not set.")
        print("       Add SLACK_BOT_TOKEN=xoxb-... to .env at the repo root, or unset --slack-channel.")
        sys.exit(1)

    # Block 4b: same fail-fast pattern for Jira. We need all three
    # ATLASSIAN_* vars OR none — a partial config is always a user
    # error worth catching before we burn AgentCall credits.
    jira_client = None
    jira_roster: dict = {}
    if args.jira_project:
        missing = [
            n for n, v in (
                ("ATLASSIAN_EMAIL", ATLASSIAN_EMAIL),
                ("ATLASSIAN_API_TOKEN", ATLASSIAN_API_TOKEN),
                ("ATLASSIAN_BASE_URL", ATLASSIAN_BASE_URL),
            ) if not v
        ]
        if missing:
            print(f"Error: --jira-project requires these env vars: {', '.join(missing)}")
            print("       Add them to .env at the repo root, or run with --show-jira-ping first to validate.")
            sys.exit(1)
        try:
            jira_client = make_jira_client(
                ATLASSIAN_EMAIL, ATLASSIAN_API_TOKEN, ATLASSIAN_BASE_URL
            )
        except JiraClientError as exc:
            print(f"Error building Jira client: {exc}")
            sys.exit(1)
        jira_roster = load_team_roster(args.jira_roster)
        # Cache the sprint list once at startup. Used by the Jira
        # handler to resolve "current sprint" / "sprint 92" / sprint
        # names without a per-request roundtrip. Empty list is fine
        # — sprint commands just won't resolve.
        jira_sprints: List[dict] = []
        # Multi-team sprint cache: {team_name_lower: [sprint_dicts]}.
        # Built by listing the project's scrum boards and pulling
        # sprints from each.
        jira_sprints_by_team: Dict[str, List[dict]] = {}
        # Project component list — used to validate user-spoken
        # component names against what the project actually defines.
        jira_components: List[str] = []
        if args.jira_board_id:
            jira_sprints = jira_list_sprints(jira_client, args.jira_board_id)
            n_active = sum(1 for sp in jira_sprints if sp.get("state") == "active")
            n_future = sum(1 for sp in jira_sprints if sp.get("state") == "future")

            # Auto-discover other team boards on the SHP project so
            # "team core current sprint" / "team ux next sprint"
            # resolve against the right board.
            project_boards = jira_list_boards_for_project(jira_client, args.jira_project)
            team_board_map = derive_team_boards(project_boards)
            for team_name, board_id in team_board_map.items():
                if board_id == args.jira_board_id:
                    # Reuse the cache we already built for the
                    # default board so we don't double-fetch.
                    jira_sprints_by_team[team_name] = jira_sprints
                    continue
                team_sprints = jira_list_sprints(jira_client, board_id)
                if team_sprints:
                    jira_sprints_by_team[team_name] = team_sprints

            # Components: case-insensitive cache as a list of canonical
            # spellings (matching atop the project's defined set).
            comps_raw = jira_list_components(jira_client, args.jira_project)
            jira_components = [c.get("name") for c in comps_raw if c.get("name")]

            n_teams = len(jira_sprints_by_team)
            print(
                f"Jira:     project={args.jira_project} type={args.jira_issue_type} "
                f"roster={'(empty)' if not jira_roster else f'{len(jira_roster)} entries'} "
                f"board={args.jira_board_id} sprints={len(jira_sprints)} "
                f"(active={n_active} future={n_future}) "
                f"team_boards={n_teams} components={len(jira_components)}"
            )
            if jira_sprints_by_team:
                print(
                    f"          team sprint caches: "
                    + ", ".join(
                        f"{t}={len(s)}" for t, s in sorted(jira_sprints_by_team.items())
                    )
                )
        else:
            print(
                f"Jira:     project={args.jira_project} type={args.jira_issue_type} "
                f"roster={'(empty)' if not jira_roster else f'{len(jira_roster)} entries'} "
                f"sprints=(disabled, --jira-board-id=0)"
            )
    else:
        jira_sprints = []
        jira_sprints_by_team = {}
        jira_components = []

    # Dry-run branch: no AgentCall call, no WebSocket, no credits.
    if args.dry_run:
        run_dry(
            args.name,
            slack_channel=args.slack_channel,
            jira_project=args.jira_project,
            jira_issue_type=args.jira_issue_type,
            jira_client=jira_client,
            jira_roster=jira_roster,
            jira_sprints=jira_sprints,
            jira_sprints_by_team=jira_sprints_by_team,
            jira_components=jira_components,
        )
        return

    # Real run requires a meet URL.
    if not args.meet_url:
        parser.error("meet_url is required unless --dry-run is set")

    print(f"Creating moderator call for: {args.meet_url}")
    print(f"Bot name: {args.name}")
    print(f"Voice:    {args.voice}")
    print(f"Avatar:   {args.avatar or 'off (audio-only)'}")
    print(f"Model:    {MODEL}\n")

    # `state` is the bridge between run_moderator (which writes to it as the
    # call progresses) and finalize_call (which reads from it to save the
    # log + post to Slack). We own it here in main() so even a Ctrl+C-driven
    # KeyboardInterrupt can't cause us to lose the transcript or skip the
    # Slack post.
    state: dict = {
        "bot_name": args.name,
        "voice": args.voice,
        "started_at": datetime.now(),
    }

    # Avatar mode requires the template server + tunnel to be live BEFORE
    # FirstCall fetches the page. Both are async (aiohttp + websockets), so
    # for the avatar path we run setup + run_moderator inside a single
    # asyncio.run; the audio path keeps the simpler sync-then-async flow.
    call_id: str = ""
    try:
        if args.avatar:
            call_id = asyncio.run(_run_with_avatar(args, state, dict(
                jira_client=jira_client,
                jira_roster=jira_roster,
                jira_sprints=jira_sprints,
                jira_sprints_by_team=jira_sprints_by_team,
                jira_components=jira_components,
            )))
        else:
            call = create_call(args.meet_url, args.name, args.avatar)
            call_id = call["call_id"]
            print(f"Call created: {call_id}")
            print(f"Status:       {call['status']}\n")
            asyncio.run(run_moderator(
                call,
                args.voice,
                args.name,
                args.output,
                slack_channel=args.slack_channel,
                state=state,
                jira_project=args.jira_project,
                jira_issue_type=args.jira_issue_type,
                jira_client=jira_client,
                jira_roster=jira_roster,
                jira_sprints=jira_sprints,
                jira_sprints_by_team=jira_sprints_by_team,
                jira_components=jira_components,
            ))
    except KeyboardInterrupt:
        # Ctrl+C path. We still want save_call_log + Slack post to run, so
        # mark the reason and fall through into the finally block.
        print("\nInterrupted - cleaning up...")
        state["end_reason"] = "interrupted"
    finally:
        # ALWAYS runs: normal call.ended exit, ws_closed, Ctrl+C, or any
        # unhandled exception inside run_moderator. Order matters:
        #   1) finalize_call — transcript markdown + Slack post (uses chat
        #      session that's still alive at this point)
        #   2) end_call — REST DELETE so AgentCall stops billing
        # Doing Slack first lets us see any Slack failure before the bot
        # leaves the room.
        if call_id:
            finalize_call(call_id, state, args.output, args.slack_channel)
            end_call(call_id)


async def _run_with_avatar(args, state: dict, jira_ctx: dict) -> str:
    """
    Async wrapper for the --avatar path. Brings up the local template
    server, fires create_call so FirstCall starts loading the bot's video
    feed, brings up the tunnel client, and finally runs the moderator
    loop. Returns the call_id (used by main()'s finalize_call / end_call
    cleanup in the finally block).

    Failures during setup fall back to audio-only with a printed warning
    — the user still gets a working call, just without the avatar tile.
    """
    template_name = args.avatar  # e.g. "avatar" / "orb" / "ring"
    print(f"  [avatar] starting local template server for '{template_name}'...")
    try:
        runner, ui_port = await start_template_server(template_name)
    except RuntimeError as exc:
        print(f"  [avatar] {exc}")
        print("  [avatar] falling back to audio-only.")
        runner, ui_port = None, 0
    if not runner:
        # Template not found / aiohttp missing — degrade gracefully.
        call = create_call(args.meet_url, args.name, None)
    else:
        print(f"  [avatar] template server on port {ui_port}")
        call = create_call(args.meet_url, args.name, template_name, ui_port=ui_port)

    call_id = call["call_id"]
    print(f"Call created: {call_id}")
    print(f"Status:       {call['status']}\n")

    tunnel = None
    if runner and call.get("tunnel_id") and call.get("tunnel_access_key"):
        tunnel_ws_base = API_BASE.replace("https://", "wss://").replace("http://", "ws://")
        tunnel_ws_url = f"{tunnel_ws_base}/internal/tunnel/connect"
        tunnel = _BridgeTunnelClient(
            tunnel_ws_url=tunnel_ws_url,
            tunnel_id=call["tunnel_id"],
            access_key=call["tunnel_access_key"],
            ui_port=ui_port,
        )
        try:
            await tunnel.connect()
        except Exception as exc:
            print(f"  [avatar] tunnel connect failed: {exc}")
            print("  [avatar] continuing with audio-only fallback (bot will join, video tile may be blank).")
            tunnel = None

    try:
        await run_moderator(
            call,
            args.voice,
            args.name,
            args.output,
            slack_channel=args.slack_channel,
            state=state,
            jira_project=args.jira_project,
            jira_issue_type=args.jira_issue_type,
            **jira_ctx,
        )
    finally:
        if tunnel is not None:
            await tunnel.close()
        if runner is not None:
            await runner.cleanup()
    return call_id


if __name__ == "__main__":
    main()
