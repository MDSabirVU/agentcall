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
Listening is your default. Most turns you should output speak: null.

Only set "speak" to a non-null value when ONE of these triggers fires:

  1. AGENDA-SETTING: the meeting just started and you don't know the
     agenda yet. Greet the room warmly with a brief "Hi everyone," (or
     "Hello!") and then ask what the agenda for the call is. Do this
     ONCE at the opening, then stay silent until someone answers. Once
     an agenda has been established, never ask again.

  2. OFF-TOPIC REDIRECT: the conversation has been off-topic from the
     stated agenda for EIGHT turns in a row, COUNTING ACROSS ALL
     PARTICIPANTS (look at the "Off-topic streak" number in the per-turn
     context - it tracks the whole group, not any one speaker). When
     the streak reaches 8, redirect POLITELY with one short sentence -
     use words like "please", "sorry to interrupt", "if I may" so it
     reads as a gentle nudge, not an order. You can address the
     most-recent drifter by name. Do NOT redirect earlier than eight
     consecutive off-topic turns - meetings naturally drift and an
     early intervention feels rude. Single drifts and brief tangents
     are fine.

     Example good redirects:
       - "Sorry to interrupt — could we please come back to <agenda>?"
       - "Bob, if I may, can we steer back toward <agenda>?"
       - "Just gently flagging - we're a bit off the agenda. Mind if we
         circle back?"

  3. SILENT-PARTICIPANT NUDGE: a participant has spoken ZERO (0)
     times AND the total turn count across the OTHER participants is
     10 or more (i.e. the meeting has had 10+ utterances without them).
     When both conditions hold, gently and politely invite that
     person to share their input. Use softening words like "please",
     "would love to hear", "we'd love your thoughts". Address them by
     first name. Do this AT MOST ONCE per participant per meeting -
     check the "Already nudged" list in the context and never re-nudge
     someone already in it. Don't nudge the bot itself.

     Example good nudges:
       - "Alice, we'd love to hear your thoughts on this — please
          jump in whenever."
       - "Bob, you've been quiet — anything you'd like to add?"

  4. DIRECT ADDRESS: someone calls you by name ("Juno, what do you
     think?" / "Mod, ..."). Answer briefly and helpfully.

  5. WRAP-UP RECAP: the meeting is winding down. Fire IMMEDIATELY -
     don't wait for confirmation - on ANY of these signals from any
     speaker:
       - "thanks all", "thanks everyone", "thank you all"
       - "that's everything", "that's it", "that's all"
       - "let's end here", "let's wrap up", "we're done"
       - "have a good one", "talk soon", "catch you later"
       - "bye", "goodbye", "see you"
     When firing: read back the captured action items in 1-3
     sentences. If there are NO action items in state, summarize the
     main discussion points (priorities, decisions, themes) instead -
     don't stay silent just because no formal action items existed.
     The recap is the meeting's closing handshake; the audience
     expects it.

When in doubt: stay silent. speak: null is the right answer for any
turn where you wouldn't naturally jump into a real meeting as a human
moderator.

YOU MUST RESPOND IN THIS EXACT JSON SHAPE ON EVERY TURN. No prose,
no markdown code fences, no preamble - just the raw JSON object:

{
  "speak": null OR "the words to say out loud (<= 25 words)",
  "agenda_items": [] OR ["item 1", "item 2"],
  "new_action_items": [] OR [
    {"owner": "<name>", "task": "<short description>",
     "due": null OR "YYYY-MM-DD"}
  ],
  "off_topic": false OR true,
  "follow_up_meeting": null OR {
    "topic": "<short description>",
    "attendees": ["<name>", ...]
  }
}

Field semantics:
- speak: the literal words AgentCall TTS will play. Set to null on any
  turn where no trigger fired. Keep under 25 words.
- agenda_items: populate ONLY on the turn where the agenda is being
  established (typically the first substantive turn after you asked).
  After that, leave it empty - the system already has it.
- new_action_items: capture every commitment, decision, priority,
  or concrete task mentioned in THIS turn. Interpret "action item"
  LIBERALLY - it's not just things explicitly framed as "Alice will
  do X by Friday". It also includes:
    * Priorities someone is committing the team to ("we should
      prioritize feature X first" → owner=speaker, task="prioritize
      feature X first").
    * Decisions made in the meeting ("we should retire feature Z"
      → owner=speaker, task="retire feature Z").
    * Aspirations someone owns by virtue of having proposed them
      ("I think a launch webinar is worth doing" → owner=speaker,
      task="organize launch webinar").
    * Explicit assignments ("Alice will own the pricing audit by
      Friday" → owner="Alice", task="pricing audit", due="<friday-iso>").
  When in doubt: CAPTURE IT. Over-capture is fine; under-capture
  hurts the meeting summary. Use the speaker's name as owner when
  no other name is given (treat "we should..." statements as the
  speaker owning the task by virtue of having raised it).
  Empty list ONLY if the turn was purely conversational ("hi",
  "good morning", "interesting") with zero forward-looking content.
  Use ISO YYYY-MM-DD for due dates when possible. CRITICAL: ALL due
  dates MUST be in the current year shown in "Today's date" in the
  per-turn context, OR later. NEVER emit a due date earlier than
  today. When the speaker says "by Friday", "next week", "end of
  Q3", "tomorrow" - resolve them using the current calendar year
  from the context, NOT from any year you remember from training. The per-turn
  context begins with "Today's date: YYYY-MM-DD" - resolve any
  relative due dates ("Friday", "end of Q3", "28th of May") against
  THAT date, not against your training cutoff.
- off_topic: true if the current utterance is off-topic vs the stated
  agenda. False (or the agenda is unset) otherwise.
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
            # Bumped from 400 (Block 1 single-sentence reply) to 800 to
            # fit the full JSON envelope: ~80-120 tokens of fixed fields
            # plus an arbitrary list of action items. We hit 400 once
            # already on the summary call - don't repeat that mistake.
            max_output_tokens=800,
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
    "agenda_items": [],
    "new_action_items": [],
    "off_topic": False,
    "follow_up_meeting": None,
}


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
            # Skip near-duplicates - same owner + same task text (case
            # insensitive). The model sometimes re-asserts an item it
            # already mentioned on a previous turn; dedupe so the Slack
            # post doesn't show the same line twice.
            dup = any(
                ex["owner"].lower() == item["owner"].lower()
                and ex["task"].lower() == item["task"].lower()
                for ex in existing
            )
            if not dup:
                existing.append(item)

    if turn.get("follow_up_meeting"):
        state["follow_up"] = turn["follow_up_meeting"]


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
    "agenda": "",
    "outcome_met": "unknown",
    "outcome_note": "",
    "action_items_md": "",
}


def _format_state_action_items_md(action_items: List[dict]) -> str:
    """
    Format the cumulative action items captured in state into Slack
    mrkdwn. Preferred over the post-hoc Gemini extraction because the
    items came directly from per-turn structured responses, so there's no
    hallucination risk.

    Slack mrkdwn: bold = *single asterisks*, NOT **double**.
    """
    lines: List[str] = []
    for item in action_items:
        owner = item.get("owner") or "Unknown"
        task = item.get("task") or ""
        if not task:
            continue
        due = item.get("due")
        if due:
            lines.append(f"- *{owner}*: {task} (due {due})")
        else:
            lines.append(f"- *{owner}*: {task}")
    return "\n".join(lines)


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

    # outcome_met / outcome_note still need an LLM verdict - we can't
    # derive "did we cover the agenda?" deterministically from state.
    # Skip the call only if there's literally nothing to summarize.
    if not transcript:
        return {
            "agenda": agenda_str,
            "outcome_met": "unknown",
            "outcome_note": "",
            "action_items_md": action_items_md,
        }

    body = _format_transcript_for_extraction(transcript)
    if not body:
        return {
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

    prompt = (
        "Below is the transcript of a meeting that just ended. Respond with "
        "a single JSON object containing EXACTLY these keys (no extras):\n"
        " - agenda: a short (max 20 words) description of what the meeting "
        "was about. Empty string if no clear agenda was discussed.\n"
        " - outcome_met: one of \"yes\", \"no\", \"partial\", or \"unknown\" "
        "indicating whether the agenda was achieved.\n"
        " - outcome_note: a single sentence (max 25 words) justifying the "
        "outcome_met verdict.\n"
        " - action_items_md: a Slack-mrkdwn bullet list of action items, one "
        "bullet per item in the EXACT format \"- *Owner*: task (due if "
        "mentioned)\". Empty string if no action items. Use *single asterisks* "
        "for bold, NOT **double**.\n\n"
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
            "agenda": agenda_str,
            "outcome_met": "unknown",
            "outcome_note": "",
            "action_items_md": action_items_md,
        }
    except Exception as exc:
        print(f"  [summary extract error] {exc}")
        return {
            "agenda": agenda_str,
            "outcome_met": "unknown",
            "outcome_note": "",
            "action_items_md": action_items_md,
        }

    # Prefer state-derived fields when we have them - they're more
    # trustworthy than re-extracted text.
    agenda_out = agenda_str or str(data.get("agenda", "")).strip()
    action_items_out = action_items_md or str(data.get("action_items_md", "")).strip()

    return {
        "agenda": agenda_out,
        "outcome_met": str(data.get("outcome_met", "unknown")).strip().lower(),
        "outcome_note": str(data.get("outcome_note", "")).strip(),
        "action_items_md": action_items_out,
    }


def build_slack_summary(
    call_id: str,
    started_at: datetime,
    ended_at: datetime,
    bot_name: str,
    voice: str,
    end_reason: str,
    summary: dict,
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
      Call ID / Duration / Bot / End reason
      Agenda           (omitted if empty)
      Action items     (always shown - "no items" placeholder if empty)
      Outcome reached  (omitted entirely if outcome is unknown AND no note)
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
        f"*Call ID:* {call_id}",
        f"*Duration:* {duration_str} ({start_hm} – {end_hm} local)",
        f"*Bot:* {bot_name} (voice: {voice})",
        f"*End reason:* {end_reason}",
        "",
    ]

    agenda = summary.get("agenda", "").strip()
    if agenda:
        lines.append(f"*Agenda:* {agenda}")
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
                    # historical nudges).
                    if leaver != bot_name:
                        speaker_turns.pop(leaver, None)

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
                    speaker_turns[speaker] = speaker_turns.get(speaker, 0) + 1

                    # Build the per-turn context the moderator sees. The
                    # idea is to give the model just enough state to make
                    # the five intervention decisions in SYSTEM_PROMPT
                    # WITHOUT re-sending the entire transcript (it's
                    # already in the chat's rolling history).
                    agenda_now = state.get("agenda") or []
                    agenda_line = ", ".join(agenda_now) if agenda_now else "NOT YET SET"
                    turn_counts_line = ", ".join(
                        f"{n}={c}" for n, c in sorted(speaker_turns.items(), key=lambda kv: -kv[1])
                    ) or "(none yet)"
                    total_turns = sum(speaker_turns.values())
                    silent_list = [n for n, c in speaker_turns.items() if c == 0]
                    silent_line = ", ".join(sorted(silent_list)) or "(none)"
                    streak_line = (
                        f"{state['off_topic_streak']} consecutive off-topic turn(s) by "
                        f"{state['off_topic_last_speaker']}"
                        if state.get("off_topic_streak") and state.get("off_topic_last_speaker")
                        else "0 (no current streak)"
                    )
                    nudged_line = ", ".join(sorted(state["nudged_set"])) or "(none yet)"

                    prompt = (
                        # Today's date is the anchor for resolving any
                        # relative due dates the speaker mentions ("by
                        # Friday", "end of Q3", "28th of May"). Without
                        # it, Gemini defaults to a year near its training
                        # cutoff (e.g. 2024) instead of the current year.
                        f"Today's date: {datetime.now():%Y-%m-%d (%A)}\n"
                        f"Agenda: {agenda_line}\n"
                        f"Speaker turn counts so far: {turn_counts_line}\n"
                        f"Total participant turns: {total_turns}\n"
                        f"Silent participants (0 turns): {silent_line}\n"
                        f"Off-topic streak: {streak_line}\n"
                        f"Already nudged (do NOT re-nudge): {nudged_line}\n"
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
                    for participant_name in list(speaker_turns.keys()):
                        first_name = participant_name.split()[0]
                        if (
                            speaker_turns.get(participant_name, 0) <= 1
                            and first_name.lower() in turn["speak"].lower()
                        ):
                            state["nudged_set"].add(participant_name)

                    print(f"  [bot] {turn['speak']}")
                    transcript.append({"role": "moderator", "text": turn["speak"]})
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
            call_id, started_at, ended_at, bot_name, voice, end_reason,
            dict(_EMPTY_SUMMARY),
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
        call_id, started_at, ended_at, bot_name, voice, end_reason, summary,
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
        "off_topic_streak": 0,
        "off_topic_last_speaker": None,
        "nudged_set": set(),
        "follow_up": None,
        "chat": chat,
    }
    started_at = datetime.now()

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
                    print(f"[state] +participant {joinee} (silent, 0 turns)\n")
                else:
                    print("[dry-run] usage: /join <Name>\n")
                continue

            speaker, text = _parse_speaker(user_text)
            transcript.append({"role": "participant", "speaker": speaker, "text": text})
            state["speaker_turns"][speaker] = state["speaker_turns"].get(speaker, 0) + 1

            agenda_now = state.get("agenda") or []
            agenda_line = ", ".join(agenda_now) if agenda_now else "NOT YET SET"
            turn_counts_line = ", ".join(
                f"{n}={c}" for n, c in sorted(state["speaker_turns"].items(), key=lambda kv: -kv[1])
            ) or "(none yet)"
            total_turns = sum(state["speaker_turns"].values())
            silent_list = [n for n, c in state["speaker_turns"].items() if c == 0]
            silent_line = ", ".join(sorted(silent_list)) or "(none)"
            streak_line = (
                f"{state['off_topic_streak']} consecutive off-topic turn(s) by "
                f"{state['off_topic_last_speaker']}"
                if state.get("off_topic_streak") and state.get("off_topic_last_speaker")
                else "0 (no current streak)"
            )
            nudged_line = ", ".join(sorted(state["nudged_set"])) or "(none yet)"
            prompt = (
                # See live-path prompt for why we ship today's date.
                f"Today's date: {datetime.now():%Y-%m-%d (%A)}\n"
                f"Agenda: {agenda_line}\n"
                f"Speaker turn counts so far: {turn_counts_line}\n"
                f"Total participant turns: {total_turns}\n"
                f"Silent participants (0 turns): {silent_line}\n"
                f"Off-topic streak: {streak_line}\n"
                f"Already nudged (do NOT re-nudge): {nudged_line}\n"
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

            # Per-turn diagnostics so we can see what the bot decided.
            decisions: List[str] = []
            if turn["agenda_items"]:
                decisions.append(f"agenda={turn['agenda_items']}")
            if turn["new_action_items"]:
                decisions.append(f"+{len(turn['new_action_items'])} action item(s)")
            if turn["off_topic"]:
                decisions.append("off_topic=true")
            if turn["follow_up_meeting"]:
                decisions.append("follow_up suggested")
            if decisions:
                print(f"[state] {' | '.join(decisions)}")

            if not turn["speak"]:
                print("[bot] (silent)\n")
                continue

            print(f"[bot] {turn['speak']}\n")
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
            summary = extract_meeting_summary(transcript, state=state)
            body = build_slack_summary(
                call_id="dry-run",
                started_at=started_at,
                ended_at=datetime.now(),
                bot_name=bot_name,
                voice="(dry-run, no TTS)",
                end_reason="dry-run",
                summary=summary,
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
