# Meeting Moderator (Gemini, 1-on-1 round-trip)

A minimal **STT → LLM → TTS** loop: the bot joins a Google Meet, AgentCall transcribes whoever's talking, the utterance is sent to **Google Gemini**, and Gemini's reply is spoken back through the bot using AgentCall's TTS.

This is the "prove the round-trip works" example. The bot replies to every utterance — no moderator-specific gating yet. Once this loop is solid, agenda-tracking, speaker-fairness, off-topic detection, etc. get layered on top of `moderator.py`.

## What It Does

1. Joins a meeting in **audio mode** with **direct** voice strategy (no GetSun — your script drives the bot).
2. Waits for the first human participant, then greets them via a Gemini-generated line.
3. Listens for `transcript.final` events.
4. For each finalized utterance, sends it to a **Gemini chat session** (history kept by the SDK).
5. Speaks Gemini's reply back via `tts.speak` over the WebSocket.
6. On `call.ended` or `Ctrl+C`, saves the transcript and force-ends the call so it stops billing.

## Architecture

```
You speak in Meet
    ↓
AgentCall STT → transcript.final event (over WebSocket)
    ↓
moderator.py → Gemini chat session (gemini-2.5-flash)
    ↓
Gemini reply text
    ↓
moderator.py → {"command": "tts.speak", ..., "voice": "bm_george"} → WebSocket
    ↓
AgentCall TTS plays through bot's mic in Meet
```

## Files

| File | Purpose |
|------|---------|
| `moderator.py` | The whole script — call lifecycle + Gemini chat loop |
| `README.md` | This file |

## Setup

```bash
cd ~/Repositories/agentcall
source .venv/Scripts/activate          # Git Bash on Windows
pip install google-genai python-dotenv # aiohttp/websockets/requests already installed

# AgentCall key (env var OR ~/.agentcall/config.json):
export AGENTCALL_API_KEY="ak_ac_your_key"

# Gemini key — put it in .env at the repo root (already in .gitignore):
echo 'GEMINI_API_KEY=your-key-from-aistudio.google.com/apikey' > .env
```

## Run

```bash
python examples/meeting-moderator/moderator.py "https://meet.google.com/abc-def-ghi"

# Or with custom name / voice / output:
python examples/meeting-moderator/moderator.py "https://meet.google.com/abc" \
    --name "Mod" --voice bm_george --output run.md
```

Default voice is `bm_george` (British male). The voice is sent on **every** `tts.speak` command, not just at startup — the `--voice` flag alone is unreliable in this AgentCall version.

## Step-by-step test plan

### One-time setup

1. Install the two new deps (see Setup above).
2. Get a Gemini API key: https://aistudio.google.com/apikey
3. Put `GEMINI_API_KEY=...` in `~/Repositories/agentcall/.env` (gitignored).
4. Confirm your AgentCall key is set (`echo $AGENTCALL_API_KEY` or check `~/.agentcall/config.json`).
5. Open https://app.agentcall.dev and confirm **no calls are listed as active** before you start.

### Happy-path test

1. Sign into your **personal** Google account in Chrome — work Workspaces typically block bots.
2. Visit https://meet.google.com/new — copy the URL from the address bar.
3. In Git Bash:
   ```bash
   cd ~/Repositories/agentcall
   source .venv/Scripts/activate
   python examples/meeting-moderator/moderator.py "<your-meet-url>" --name "Mod" --voice bm_george
   ```
4. Watch the terminal — you should see:
   ```
   Creating moderator call for: ...
   Call created: call-...
   Connecting to WebSocket: wss://...
   ```
5. Back in the Meet tab, **admit "Mod" from the lobby**.
6. Once admitted, the terminal prints `Bot is in the meeting.` followed by `+ <YourName> joined`, and you hear a short Gemini-generated greeting in a British male voice through Meet's audio.
7. Say out loud: **"Hello, can you hear me?"**
8. **Expected (within ~3–5 s):** terminal shows
   ```
   [you] (<YourName>) Hello, can you hear me?
   [bot] <Gemini reply>
   ```
   and the reply plays through Meet.
9. Have a short back-and-forth — 3–4 turns. Each utterance should produce a Gemini reply via TTS.

### Cleanup test (do not skip)

10. Press **Ctrl+C** in the terminal.
    Expected output: `Interrupted - cleaning up...` then `Call ended (cleanup).`
    The bot should leave the Meet within a few seconds.
11. Open https://app.agentcall.dev and confirm **no active calls**. If one is still listed, end it manually — orphaned calls keep burning credits.

### Negative tests

- **Empty / safety-filtered Gemini reply** — make a noise or speak gibberish. Terminal should log `[bot] (empty reply, skipping tts)` and not send a blank `tts.speak`.
- **Multi-participant** — invite a colleague to the same Meet. Both speakers should appear in `[you] (<name>) ...` lines; the bot replies to whichever utterance came last.
- **Bot kicked from Meet side** — click "Remove" on the Mod participant inside Meet instead of Ctrl+C-ing the script. Terminal should print `Call ended: <reason>`, save the transcript, then call cleanup. Verify the dashboard shows no active calls.

### After a successful run

You should see `meeting-moderator-log-YYYY-MM-DD-HHMM.md` in your current working directory with every transcribed turn and every moderator reply.

## Troubleshooting

| Symptom | Most likely cause |
|---------|-------------------|
| Bot joins but never speaks after you talk | `tts.speak` JSON not flushed, OR you're still in the waiting room, OR `voice` missing from the command (this script always sends it). |
| `Error: set GEMINI_API_KEY ...` on startup | `.env` missing or in the wrong directory — must be at the repo root, or set in your shell. |
| `Error: set AGENTCALL_API_KEY ...` on startup | No env var and no `~/.agentcall/config.json`. |
| Bot speaks once then goes silent | Bot's own TTS is being transcribed back — the script skips while `is_speaking` is True, but the heuristic relies on `tts.started` / `tts.done` firing. Check the `[event]` lines for those. |
| Active call still showing on dashboard after you exit | The REST `DELETE /v1/calls/{id}` fallback failed (network blip). End the call manually from the dashboard. |

## Billing

| Component | Charged? | Notes |
|-----------|----------|-------|
| Meeting bot (base) | Yes | Per minute of call |
| Speech-to-text | Yes | Required for us to hear the participants |
| Voice intelligence (GetSun) | **No** | Direct mode — bypassed |
| Text-to-speech | Yes | Per minute of generated audio |
| Gemini API | Yes | Billed by Google on your Gemini key |

⚠️ **Always end calls cleanly.** Every minute the bot sits in a Meet burns credits. Ctrl+C exits via the cleanup path; `call.ended` events do the same. After each test session, glance at https://app.agentcall.dev to confirm nothing is still running.

## What's intentionally out of scope (deferred to later iterations)

- Moderator-specific gating ("only speak when X happens")
- Per-speaker tracking (who's dominated, who hasn't spoken)
- Meeting phase state machine (agenda / discussion / wrap-up)
- Streaming the TTS sentence-by-sentence for lower latency
- Visual avatar (`bridge-visual.py` route)
- Slack action-item posting
- Persona prompt engineering — the system prompt is deliberately minimal

## Picking a different voice

The `--voice` argument accepts any ID from `GET /v1/tts/voices`. Common picks:

| Voice ID | Name | Gender | Language |
|----------|------|--------|----------|
| `bm_george` | George | Male | en-gb |
| `bf_emma` | Emma | Female | en-gb |
| `am_michael` | Michael | Male | en-us |
| `af_bella` | Bella | Female | en-us |
| `af_heart` | Heart | Female | en-us |

54 voices across 9 languages are available — see the AgentCall docs.
