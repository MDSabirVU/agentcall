"""
Slack delivery for the meeting moderator.

Single responsibility: post a markdown action-item summary to a configured
Slack channel after a call ends. We deliberately keep this tiny - no retry
loop, no template engine, no per-person routing - because for the
2026-05-21 innovation-day demo we agreed on "one channel, one post".

If we later move to per-person DMs, this is where conversations.open +
chat.postMessage(channel=user_id) would go. For now: channel-only.

Env contract:
    SLACK_BOT_TOKEN  Bot token (starts with xoxb-...). The Slack app
                     created for this project must have chat:write and
                     chat:write.public scopes, and be installed to the
                     workspace. chat:write.public lets us post to a
                     channel without first invoking conversations.join,
                     which keeps this module stateless.

Why not call the API directly with requests.post?
    slack_sdk.WebClient handles the form-encoded body quirks, gives us a
    typed exception with the real Slack error code in
    e.response["error"], and is what every Slack tutorial uses - so a
    future debugger has somewhere familiar to look.
"""

from __future__ import annotations

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


def post_action_items(token: str, channel: str, markdown: str) -> bool:
    """
    Post a single message containing the action-item markdown to a Slack
    channel.

    Args:
        token:     Bot OAuth token (xoxb-...).
        channel:   Either a channel ID like "C0123ABCD" or a name like
                   "#moderator-action-items". chat.postMessage accepts
                   both, though IDs are slightly more reliable across
                   renames.
        markdown:  The pre-formatted message body. Slack's mrkdwn dialect
                   is a SUPERSET of standard markdown - bold is *bold*
                   not **bold**, but bullets `- ` and headings work. The
                   moderator output happens to render fine because we
                   stick to bullets + bold.

    Returns:
        True on success, False on any Slack API failure. We deliberately
        do NOT raise - a Slack failure at end-of-call should not crash
        the cleanup path or block end_call() from running. The transcript
        is already on disk by the time we get here, so the worst case is
        the user re-runs the post manually.
    """
    if not token:
        print("Slack post skipped: SLACK_BOT_TOKEN is empty.")
        return False
    if not channel:
        print("Slack post skipped: no channel passed.")
        return False

    client = WebClient(token=token)
    try:
        resp = client.chat_postMessage(
            channel=channel,
            text=markdown,
            mrkdwn=True,
            # unfurl_links/media off keeps the post visually tight, which
            # matters when the demo audience is reading off a projector.
            unfurl_links=False,
            unfurl_media=False,
        )
        print(f"Slack post ok (channel={channel}, ts={resp.get('ts')}).")
        return True
    except SlackApiError as exc:
        # e.response["error"] is the *Slack* error code (e.g.
        # "channel_not_found", "not_in_channel", "invalid_auth"). That's
        # the thing a human needs to debug; the surrounding HTTP body is
        # noise.
        err = exc.response.get("error", "unknown") if exc.response else "unknown"
        print(f"Slack post failed: {err}")
        if err == "not_in_channel":
            print("  Hint: invite the bot to the channel, or grant chat:write.public scope.")
        elif err == "channel_not_found":
            print("  Hint: pass the channel ID (Cxxxx) instead of the #name, or check spelling.")
        elif err == "invalid_auth":
            print("  Hint: SLACK_BOT_TOKEN looks wrong or has been revoked. Reinstall the app.")
        return False
    except Exception as exc:
        # Network errors, DNS failures, etc. Still don't crash the caller.
        print(f"Slack post failed (network/unexpected): {exc}")
        return False
