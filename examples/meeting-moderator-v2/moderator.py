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
import sys
from datetime import datetime
from typing import List, Optional

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

HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

# ----------------------------------------------------------------------------
# Gemini config
# ----------------------------------------------------------------------------

# gemini-2.5-flash is the right size for live voice: fast, cheap, multilingual.
# Don't swap to gemini-3-* preview models without checking - they're not stable.
MODEL = "gemini-2.5-flash"

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
   When: "Off-topic streak" reaches EIGHT cumulative turns across all
         participants. Single drifts and brief tangents are fine.

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
         for "0 turns AND at least 10 turns elapsed SINCE THEY
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
       to you (Rule 4), and the off-topic streak is < 8 (Rule 2
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
     5. Recapping action items and posting a meeting summary to
        Slack when we wrap up

   CRITICAL: When ASKED "what can you do?" / "what are you for?" /
   similar capability questions, state ONLY the POSITIVES from the
   list above. DO NOT append "I cannot..." or "but I can't..."
   disclaimers — those make the answer feel defensive and undersell
   the bot. Keep the spoken answer roughly 50 words, upbeat, and
   forward-looking. End on a positive note (e.g. "...when we wrap
   up.") — never with a limitation.

   The list below is for YOUR INTERNAL REFERENCE ONLY. Use it to
   recognize and politely decline out-of-scope requests (Rule 4
   OUT-OF-SCOPE handling). NEVER recite this list in response to
   a "what can you do?" question — only mention a limitation when
   the speaker has specifically asked for that exact out-of-scope
   thing:

   Internal reference — out-of-scope items (don't volunteer these):
     - Scheduling meetings, sending emails, controlling external apps
     - Looking up info outside this conversation (no web/document search)
     - Remembering anything from previous meetings
     - Speaking any language other than English

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
# moment they arrive into a room that's already past 10 total turns.
# Each participant gets their own 10-turn buffer measured from the
# total-turn count at the moment they joined. SILENT_NUDGE_GRACE is
# both the threshold (in update_pending_intervention) and the buffer
# size (in this picker) - keeping it in one named constant means the
# two stay in lockstep if we ever tune the value.
SILENT_NUDGE_GRACE = 10


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
        conditions are currently met. Off-topic streak >= 8 takes
        priority over silent-nudge if both happen to be true.
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
        # off-topic streak=4 instead of 8). If neither rule's hard
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
        if streak >= 8:
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
# AgentCall lifecycle
# ----------------------------------------------------------------------------

def create_call(meet_url: str, bot_name: str, avatar: Optional[str]) -> dict:
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
        # KNOWN BROKEN as of 2026-05-19. The AgentCall API does NOT accept
        # `ui_template` as a field on POST /v1/calls (see references/api.md).
        # webpage-av mode requires a `webpage_url` pointing at a publicly
        # reachable page that AgentCall can load. The built-in templates in
        # `ui-templates/` are HTML files YOU host locally + tunnel; see
        # scripts/python/bridge-visual.py for the full pattern. Passing
        # --avatar today causes the create-call POST to fail with HTTP 400.
        # For the 2026-05-20 demo, run without --avatar (audio-only mode).
        payload["mode"] = "webpage-av"
        payload["ui_template"] = avatar
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


def build_slack_summary(
    started_at: datetime,
    ended_at: datetime,
    summary: dict,
    participants: Optional[List[str]] = None,
    dry_run: bool = False,
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

    async with websockets.connect(ws_url) as ws:
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
                        + existing_items_block +
                        f"Current speaker: {speaker}\n"
                        f"They just said: {text}"
                    )

                    turn = ask_moderator_structured(chat, prompt)

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
        except websockets.ConnectionClosed:
            end_reason = "ws_closed"
            print("\nWebSocket closed.")

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
            participants=participants,
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
        started_at, ended_at, summary, participants=participants,
    )
    post_action_items(SLACK_BOT_TOKEN, slack_channel, body)


def run_dry(bot_name: str, slack_channel: Optional[str] = None) -> None:
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
    print("[dry-run] No name prefix = you speak as 'You'.")
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

    def _parse_speaker(line: str) -> tuple:
        """Split 'Bob: hello' -> ('Bob', 'hello'), else ('You', line)."""
        if ":" in line:
            head, _, rest = line.partition(":")
            head = head.strip()
            rest = rest.strip()
            if head and rest and len(head.split()) <= 3:
                return head, rest
        return "You", line

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
            summary = extract_meeting_summary(transcript, state=state)
            body = build_slack_summary(
                started_at=started_at,
                ended_at=datetime.now(),
                summary=summary,
                participants=participants,
                dry_run=True,
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
    args = parser.parse_args()

    # Fail fast if Slack was requested but the token is missing. Catching
    # this here is much friendlier than letting the call run for 5 minutes,
    # paying for credits, and *then* discovering the post can't go through.
    if args.slack_channel and not SLACK_BOT_TOKEN:
        print("Error: --slack-channel was passed but SLACK_BOT_TOKEN is not set.")
        print("       Add SLACK_BOT_TOKEN=xoxb-... to .env at the repo root, or unset --slack-channel.")
        sys.exit(1)

    # Dry-run branch: no AgentCall call, no WebSocket, no credits.
    if args.dry_run:
        run_dry(args.name, slack_channel=args.slack_channel)
        return

    # Real run requires a meet URL.
    if not args.meet_url:
        parser.error("meet_url is required unless --dry-run is set")

    print(f"Creating moderator call for: {args.meet_url}")
    print(f"Bot name: {args.name}")
    print(f"Voice:    {args.voice}")
    print(f"Avatar:   {args.avatar or 'off (audio-only)'}")
    print(f"Model:    {MODEL}\n")

    call = create_call(args.meet_url, args.name, args.avatar)
    call_id = call["call_id"]
    print(f"Call created: {call_id}")
    print(f"Status:       {call['status']}\n")

    # `state` is the bridge between run_moderator (which writes to it as the
    # call progresses) and finalize_call (which reads from it to save the
    # log + post to Slack). We own it here in main() so even a Ctrl+C-driven
    # KeyboardInterrupt can't cause us to lose the transcript or skip the
    # Slack post.
    #
    # bot_name / voice / started_at are seeded here so finalize_call can
    # build a complete Slack header even if run_moderator gets cancelled
    # before it ever connects to the WebSocket.
    state: dict = {
        "bot_name": args.name,
        "voice": args.voice,
        "started_at": datetime.now(),
    }
    try:
        asyncio.run(run_moderator(
            call,
            args.voice,
            args.name,
            args.output,
            slack_channel=args.slack_channel,
            state=state,
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
        finalize_call(call_id, state, args.output, args.slack_channel)
        end_call(call_id)


if __name__ == "__main__":
    main()
