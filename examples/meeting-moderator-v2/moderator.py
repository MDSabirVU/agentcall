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
SYSTEM_PROMPT = """
You are Juno, a friendly conversational companion who has joined a Google
Meet call. Just chat naturally with whoever is talking - this is a casual
conversation, not a meeting you need to run.

Style rules:
- Keep replies SHORT - two spoken sentences min and max.
- Sound warm, curious, and natural.
- Address people by name when it feels natural.
- Don't refuse to chat about a topic, and don't try to "keep things on
  track" - you're not the moderator, you're a participant.
- Don't say "as an AI" or talk about being a language model.
""".strip()
# To make this bot a moderator instead, replace the prompt above with
# guidance about agenda-keeping / drawing out silent participants /
# reining in dominators / capturing action items. The plumbing in this
# file doesn't change - only the prompt does.


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
            # Headroom for ~3-4 spoken sentences. The prompt itself caps the
            # bot at two sentences; this budget exists purely so Gemini never
            # truncates mid-word. Bigger is cheap - flash tokens are fractions
            # of a cent and an unfinished sentence sounds much worse over TTS
            # than a slightly longer one.
            max_output_tokens=400,
            temperature=0.7,
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
    """
    try:
        response = chat.send_message(user_text)
        return (response.text or "").strip()
    except Exception as exc:
        # Don't crash the whole call if one Gemini request fails - log and
        # return empty so the loop continues listening.
        print(f"  [gemini error] {exc}")
        return ""


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


def extract_meeting_summary(chat, transcript: List[dict]) -> dict:
    """
    Ask Gemini, in ONE call, for all four post-call fields the Slack post
    needs: agenda summary, outcome verdict, outcome note, and action-item
    markdown.

    Returns a dict with these keys (always present, always strings):
      agenda          short description of what the meeting was about
                      ("" if no clear agenda was discussed).
      outcome_met     "yes" | "no" | "partial" | "unknown".
      outcome_note    one-sentence explanation of the outcome.
      action_items_md Slack-mrkdwn bullet list ("" if none).

    Why one call instead of four?
      - Cheaper and faster (one round-trip, shared context).
      - The model sees all four fields together, so an action item that
        directly drives the outcome verdict can be reasoned about
        consistently.
      - One JSON parse, one failure mode.

    Caveat for the 2026-05-21 demo: until Block 2 lands, the moderator
    doesn't actively *ask* for an agenda. The `agenda` field here is
    inferred from whatever participants happened to say ("today we want
    to discuss X"). When Block 2 lands, the bot will solicit the agenda
    explicitly and we'll pass it in here as ground truth instead of
    re-inferring.

    Failure mode: any exception OR a JSON parse failure -> return the
    _EMPTY_SUMMARY shape so callers can render a placeholder Slack post
    without crashing the cleanup chain.
    """
    if not transcript:
        return dict(_EMPTY_SUMMARY)
    body = _format_transcript_for_extraction(transcript)
    if not body:
        return dict(_EMPTY_SUMMARY)
    prompt = (
        "Below is the transcript of a meeting you just attended. Analyze it "
        "and respond with a single JSON object containing EXACTLY these keys "
        "(no extras, no nested wrapping):\n"
        " - agenda: a short (max 20 words) description of what the meeting "
        "was about. Empty string if no clear agenda was discussed.\n"
        " - outcome_met: one of \"yes\", \"no\", \"partial\", or \"unknown\" "
        "indicating whether the agenda was achieved.\n"
        " - outcome_note: a single sentence (max 25 words) justifying the "
        "outcome_met verdict.\n"
        " - action_items_md: a Slack-mrkdwn bullet list of action items in "
        "the EXACT format \"- *Owner*: task (due if mentioned)\" with one "
        "bullet per item. Empty string if no action items were discussed. "
        "Use *single asterisks* for bold, NOT **double**.\n\n"
        "Reply with ONLY the raw JSON object. No preamble, no closing "
        "remarks, no markdown code fences.\n\n"
        "Transcript:\n"
        f"{body}"
    )
    try:
        # Override the chat-level config for this one call. Two changes
        # matter here:
        #   1) response_mime_type="application/json" puts Gemini into
        #      JSON mode - it commits to emitting parseable JSON with no
        #      markdown code fences and no preamble text. This was the
        #      single biggest reliability fix; without it Gemini
        #      sometimes wrapped the object in ```json fences or
        #      prepended "Here is the summary:" prose.
        #   2) max_output_tokens=2000 (vs the chat's default 400) gives
        #      enough headroom for a multi-action-item meeting. The
        #      "unterminated string" error we hit on the first live run
        #      was a 400-token cap clipping mid-quote in the
        #      action_items_md field.
        # We keep the chat session's system_instruction (still cached
        # server-side) and rolling history (which has the meeting fresh
        # in context) - we only override these two fields.
        response = chat.send_message(
            prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                max_output_tokens=2000,
                # Low temperature for structured output - we want
                # deterministic JSON, not creative paraphrasing.
                temperature=0.2,
            ),
        )
        raw = (response.text or "").strip()
        if not raw:
            print("  [summary extract error] Gemini returned empty response.")
            return dict(_EMPTY_SUMMARY)
        data = json.loads(raw)
        return {
            "agenda": str(data.get("agenda", "")).strip(),
            "outcome_met": str(data.get("outcome_met", "unknown")).strip().lower(),
            "outcome_note": str(data.get("outcome_note", "")).strip(),
            "action_items_md": str(data.get("action_items_md", "")).strip(),
        }
    except json.JSONDecodeError as exc:
        # Save the raw output so we can diagnose what Gemini emitted.
        # response.text is only available inside the try, so we re-print
        # it here from the local `raw` we captured before parsing.
        print(f"  [summary parse error] {exc}")
        print(f"  [summary raw output] {raw[:300]!r}{'...' if len(raw) > 300 else ''}")
        return dict(_EMPTY_SUMMARY)
    except Exception as exc:
        print(f"  [summary extract error] {exc}")
        return dict(_EMPTY_SUMMARY)


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
                    if not greeted:
                        # First human in - generate a greeting via Gemini so
                        # the moderator's "voice" is consistent from turn one,
                        # rather than a hard-coded canned line.
                        greeted = True
                        greeting = ask_gemini(
                            chat,
                            f"{name} just joined the call. Say a short, friendly hello "
                            f"in one sentence. Use their first name.",
                        )
                        if greeting:
                            print(f"  [bot] {greeting}")
                            transcript.append({"role": "moderator", "text": greeting})
                            # NB: the WebSocket protocol uses "type", not
                            # "command" - the bridge scripts translate the
                            # stdin "command" form into "type" for the WS.
                            # Sending "command" here silently drops the
                            # message and no audio plays.
                            await ws.send(json.dumps({
                                "type": "tts.speak",
                                "text": greeting,
                                "voice": voice,
                            }))

                elif event_type == "participant.left":
                    print(f"  - {speaker_name(event.get('name') or event.get('participant'))} left")

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

                    # Hand the utterance to Gemini. The chat session keeps the
                    # rolling history; we just send the latest turn.
                    reply = ask_gemini(chat, f"{speaker}: {text}")

                    if not reply:
                        # Gemini returned empty (safety filter, etc.) -
                        # explicitly do NOT send a blank tts.speak.
                        print("  [bot] (empty reply, skipping tts)")
                        continue

                    print(f"  [bot] {reply}")
                    transcript.append({"role": "moderator", "text": reply})
                    await ws.send(json.dumps({
                        "type": "tts.speak",
                        "text": reply,
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
    if chat is None:
        # No chat session means run_moderator never got past startup -
        # nothing to extract from. Still post the header so the channel
        # sees the call happened.
        body = build_slack_summary(
            call_id, started_at, ended_at, bot_name, voice, end_reason,
            dict(_EMPTY_SUMMARY),
        )
        post_action_items(SLACK_BOT_TOKEN, slack_channel, body)
        return

    summary = extract_meeting_summary(chat, transcript)
    body = build_slack_summary(
        call_id, started_at, ended_at, bot_name, voice, end_reason, summary,
    )
    post_action_items(SLACK_BOT_TOKEN, slack_channel, body)


def run_dry(bot_name: str, slack_channel: Optional[str] = None) -> None:
    """
    Interactive rehearsal mode - no AgentCall call, no WebSocket, no TTS.

    Reads your typed sentences from stdin and prints Gemini's replies. Same
    SYSTEM_PROMPT, same model, same chat session as the real run - so this
    is a faithful test of the bot's *conversational behaviour* without
    spending any AgentCall credits. Useful for prompt-tuning and edge-case
    testing.

    The Gemini Flash free tier is 1,500 requests/day, so this is effectively
    free as well.
    """
    print(f"[dry-run] {bot_name} is ready. Type a line and press Enter.")
    print("[dry-run] Empty line or Ctrl+C to exit.\n")

    chat = build_chat_session()

    # Mirror the live transcript buffer so we can exercise the Slack
    # extraction path on dry-run too. Cheaper than burning AgentCall credits
    # every time we tweak the action-item prompt.
    transcript: List[dict] = []
    started_at = datetime.now()

    # Mirror the real flow's opening: the live bot greets on the first
    # participant.joined. Do the same here so prompt-tuning covers the
    # greeting path too.
    greeting = ask_gemini(
        chat,
        "You just joined the call. Say a short, friendly hello in one sentence.",
    )
    if greeting:
        print(f"[bot] {greeting}\n")
        transcript.append({"role": "moderator", "text": greeting})

    try:
        while True:
            try:
                user_text = input("> ").strip()
            except EOFError:
                break
            if not user_text:
                break
            transcript.append({"role": "participant", "speaker": "You", "text": user_text})
            reply = ask_gemini(chat, f"You: {user_text}")
            if not reply:
                print("[bot] (empty reply)\n")
                continue
            print(f"[bot] {reply}\n")
            transcript.append({"role": "moderator", "text": reply})
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
            summary = extract_meeting_summary(chat, transcript)
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
