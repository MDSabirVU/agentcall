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

# Load GEMINI_API_KEY from a .env file at the repo root (or the cwd).
# google-genai picks GEMINI_API_KEY up automatically when genai.Client() runs,
# so we just need it to be present in os.environ.
load_dotenv()

from google import genai
from google.genai import types

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

async def run_moderator(call: dict, voice: str, bot_name: str, output_file: Optional[str]) -> None:
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

    # Rolling transcript we'll dump to markdown at the end of the call.
    transcript: List[dict] = []
    # `is_speaking` defends against the bot's own TTS audio bleeding back
    # into STT. In direct mode, the bot's voice is injected into the Meet's
    # audio bus, so the STT engine can transcribe it. We skip transcript.final
    # events while we know we're speaking. (Same heuristic as support-agent.)
    is_speaking = False
    # We only greet once - on the first human that joins, not on every join.
    greeted = False
    end_reason = "unknown"

    # Build the Gemini chat session BEFORE connecting so the bot can greet
    # the moment a human joins without a cold-start delay.
    chat = build_chat_session()

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

    save_call_log(call_id, transcript, end_reason, output_file)


def run_dry(bot_name: str) -> None:
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

    # Mirror the real flow's opening: the live bot greets on the first
    # participant.joined. Do the same here so prompt-tuning covers the
    # greeting path too.
    greeting = ask_gemini(
        chat,
        "You just joined the call. Say a short, friendly hello in one sentence.",
    )
    if greeting:
        print(f"[bot] {greeting}\n")

    try:
        while True:
            try:
                user_text = input("> ").strip()
            except EOFError:
                break
            if not user_text:
                break
            reply = ask_gemini(chat, f"You: {user_text}")
            if not reply:
                print("[bot] (empty reply)\n")
                continue
            print(f"[bot] {reply}\n")
    except KeyboardInterrupt:
        print()  # newline after ^C


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
    args = parser.parse_args()

    # Dry-run branch: no AgentCall call, no WebSocket, no credits.
    if args.dry_run:
        run_dry(args.name)
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

    try:
        asyncio.run(run_moderator(call, args.voice, args.name, args.output))
    except KeyboardInterrupt:
        # Ctrl+C path. Without this, the bot stays in the Meet and keeps
        # burning AgentCall credits. End the call hard via the REST API.
        print("\nInterrupted - cleaning up...")
        end_call(call_id)
    else:
        # Normal exit path: call.ended already fired. Still call DELETE in
        # case anything is lingering server-side - it's idempotent.
        end_call(call_id)


if __name__ == "__main__":
    main()
