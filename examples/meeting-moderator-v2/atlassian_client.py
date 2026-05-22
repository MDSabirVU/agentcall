"""
Atlassian (Jira) thin wrapper for the Meeting Moderator.

Mirror of slack_client.py's design philosophy:
  - One module, narrow tool surface.
  - Fail OPEN on errors — Atlassian hiccups should never crash a live
    call. Functions return None / False / empty rather than raising.
  - Hide SDK-specific details (the atlassian-python-api Jira class
    method names) behind small functions named after the meeting-
    moderator domain ("create issue from action item", "look up by
    key", etc.).

The library uses one `Jira(...)` client object that we pass around. It
holds the auth (Basic Auth with email + API token) and a connection
pool. Build it once at startup; reuse across the call.

References:
  - Atlassian Cloud API docs:
      https://developer.atlassian.com/cloud/jira/platform/rest/v3/
  - SDK docs:
      https://atlassian-python-api.readthedocs.io/
  - API token management:
      https://id.atlassian.com/manage-profile/security/api-tokens
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    # atlassian-python-api exposes both Jira() and Confluence() classes.
    # We import only Jira here; Confluence is deferred to a later block.
    from atlassian import Jira  # type: ignore
except ImportError:
    Jira = None  # type: ignore


# ----------------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------------


class JiraClientError(Exception):
    """Raised on hard auth / config failures. Soft failures fail open."""


# Story-points custom field ID. Hard-coded after a one-time discovery
# (2026-05-20) against SHP-17182: customfield_10020 holds the numeric
# story-points value on Shypple's Atlassian Cloud instance.
# Other Jira instances commonly use customfield_10016 ("Story Points
# Estimate") or customfield_10026 — if this code ever moves to a
# different org, re-run the discovery: `client.issue("KEY")` and grep
# the response for which customfield_NNNNN holds a numeric value.
STORY_POINTS_FIELD = "customfield_10020"

# Sprint custom field ID. Discovered (2026-05-20) by inspecting
# SHP-17182's customfield_10018, which held a list of sprint dicts.
# On Jira Cloud team-managed projects, this same field accepts a
# SINGLE sprint ID (int) on create and an int on update_issue_field,
# even though it round-trips back as a list of historical sprints.
SPRINT_FIELD = "customfield_10018"


# ----------------------------------------------------------------------------
# Client construction + sanity check
# ----------------------------------------------------------------------------


def make_jira_client(email: str, api_token: str, base_url: str):
    """
    Build a Jira client. Does NOT make a network call.

    cloud=True tells the SDK to use the Atlassian Cloud auth scheme
    (Basic Auth with email + API token). For server / data-center
    installs you'd use cloud=False with a personal access token, but
    Shypple is on Cloud (shypple.atlassian.net) so we hardcode True.

    Raises JiraClientError if the SDK isn't installed or any required
    field is missing — better to fail loudly here than to silently
    return a half-configured client that explodes on first use.
    """
    if Jira is None:
        raise JiraClientError(
            "atlassian-python-api is not installed. Run:\n"
            "  pip install -r examples/meeting-moderator-v2/requirements.txt"
        )
    if not email or not api_token or not base_url:
        raise JiraClientError(
            "make_jira_client requires email, api_token, and base_url "
            "(all three must be non-empty)."
        )
    # username + password is the Basic Auth pair in the SDK; for Cloud
    # this is (email, API token), NOT (email, account password).
    return Jira(
        url=base_url.rstrip("/"),
        username=email,
        password=api_token,
        cloud=True,
    )


def jira_ping(client) -> Dict[str, Any]:
    """
    Confirm the client is authenticated by calling /rest/api/3/myself.

    Returns the authenticated user's profile dict on success:
        {"accountId": "...", "displayName": "Daniyal Sabir",
         "emailAddress": "daniyal.sabir@shypple.com", ...}

    Raises JiraClientError with a helpful message on failure — most
    common causes:
      - 401: token wrong or expired
      - 403: account exists but lacks API access
      - DNS / TLS: base_url typo (e.g. missing the .atlassian.net)
    """
    try:
        me = client.myself()
    except Exception as exc:  # broad on purpose — SDK can raise many types
        raise JiraClientError(
            f"Jira /myself call failed: {exc}\n"
            "  Common causes: wrong API token, wrong base_url, or "
            "account lacks API access. Re-check the three ATLASSIAN_* "
            "values in .env."
        ) from exc
    if not isinstance(me, dict) or "accountId" not in me:
        raise JiraClientError(
            f"Jira /myself returned unexpected shape: {me!r}"
        )
    return me


# ----------------------------------------------------------------------------
# User resolution (email -> accountId)
# ----------------------------------------------------------------------------


def jira_resolve_account_id(client, email: str) -> Optional[str]:
    """
    Look up a Jira accountId from an email address.

    Jira Cloud's REST API requires `accountId`, not email, for the
    `assignee` field on issue create / update. So we have to translate
    here. Returns None on any failure — caller decides whether to
    create unassigned or fall back to a default.

    Strategy:
      1. Search via the SDK's user_find_by_user_string (wraps
         /rest/api/3/user/search?query=<email>).
      2. Prefer an exact emailAddress match.
      3. Fall back to the first result if no exact match (Jira
         sometimes hides email for GDPR reasons; first result is
         usually right).
      4. None on any exception — fail open.
    """
    if not email:
        return None
    try:
        users = client.user_find_by_user_string(query=email, start=0, limit=5)
    except Exception:
        return None
    if not users or not isinstance(users, list):
        return None
    # Exact match first
    for u in users:
        if isinstance(u, dict) and u.get("emailAddress") == email:
            return u.get("accountId")
    # Fall back to first
    first = users[0]
    if isinstance(first, dict):
        return first.get("accountId")
    return None


# ----------------------------------------------------------------------------
# Issue CRUD — placeholders for Block 4b-4e. Implemented now so the
# module is feature-complete and 4b can call straight in.
# ----------------------------------------------------------------------------


def jira_list_components(client, project_key: str) -> List[Dict[str, Any]]:
    """
    Return the components defined on a Jira project.

    Used at moderator startup to cache the valid set so user-spoken
    component names ("with component STACK") can be validated and
    normalised to the exact spelling Jira expects. Returns [] on
    error — fail open.
    """
    if not project_key:
        return []
    try:
        comps = client.get_project_components(project_key) or []
        return [c for c in comps if isinstance(c, dict)]
    except Exception as exc:
        print(f"  [jira components list error] {exc}")
        return []


def jira_list_boards_for_project(client, project_key: str) -> List[Dict[str, Any]]:
    """
    Return every agile board attached to a project.

    Used to auto-discover the per-team scrum boards (Team STACK, Team
    CORE, etc.) so multi-team sprint references work without
    hard-coding board IDs in CLI flags. Returns [] on error.
    """
    if not project_key:
        return []
    try:
        # The agile API supports filtering by projectKeyOrId.
        page = client.get_all_agile_boards(limit=100)
        boards = page.get("values", []) if isinstance(page, dict) else []
        # Filter Python-side to be tolerant of API quirks. Match on
        # the board's location.projectKey.
        out = []
        for b in boards:
            loc = b.get("location", {}) if isinstance(b, dict) else {}
            if loc.get("projectKey") == project_key:
                out.append(b)
        return out
    except Exception as exc:
        print(f"  [jira boards list error] {exc}")
        return []


def jira_add_comment(client, issue_key: str, body: str) -> bool:
    """
    Add a comment to a Jira issue. Returns True on success.

    The SDK's issue_add_comment is a thin wrapper around POST
    /rest/api/3/issue/{key}/comment. Body is plain string; the SDK
    handles ADF conversion for Cloud. Fails OPEN.
    """
    if not issue_key or not body:
        return False
    try:
        client.issue_add_comment(issue_key, body)
        return True
    except Exception as exc:
        print(f"  [jira comment error] {exc}")
        return False


def jira_list_sprints(client, board_id: int) -> List[Dict[str, Any]]:
    """
    Return every sprint on a board (paginated under the hood).

    Used at moderator startup to cache the sprint list so we can
    resolve user-spoken sprint names/numbers to IDs without burning
    a roundtrip per lookup. Returns [] on any error — fail open.

    Each entry has keys: id, state ("active"|"closed"|"future"), name,
    boardId, plus optional startDate/endDate/goal.
    """
    if not board_id:
        return []
    all_sprints: List[Dict[str, Any]] = []
    start = 0
    try:
        while True:
            page = client.get_all_sprints_from_board(board_id, start=start, limit=50)
            vals = page.get("values", []) if isinstance(page, dict) else []
            if not vals:
                break
            all_sprints.extend(vals)
            if page.get("isLast") or len(vals) < 50:
                break
            start += 50
    except Exception as exc:
        print(f"  [jira sprint list error] {exc}")
        return []
    return all_sprints


def jira_add_to_sprint(client, sprint_id: int, issue_key: str) -> bool:
    """
    Move a single issue into a sprint. Returns True on success.

    This is the canonical agile-API call and is preferred over a
    raw customfield_10018 PUT because Jira maintains internal
    sprint membership state that direct field writes can desync.
    Fails OPEN: prints the error and returns False so the caller
    can speak a graceful "couldn't move it" line.
    """
    if not sprint_id or not issue_key:
        return False
    try:
        client.add_issues_to_sprint(sprint_id, [issue_key])
        return True
    except Exception as exc:
        print(f"  [jira sprint add error] {exc}")
        return False


def jira_create_issue(
    client,
    project_key: str,
    summary: str,
    description: str = "",
    assignee_email: Optional[str] = None,
    fallback_assignee_email: Optional[str] = None,
    due_date: Optional[str] = None,
    issue_type: str = "Task",
    priority: Optional[str] = None,
    story_points: Optional[float] = None,
    epic_key: Optional[str] = None,
    sprint_id: Optional[int] = None,
    components: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    reporter_email: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Create a Jira issue. Returns the created issue dict (containing at
    minimum `key` and `id`) on success, or None on failure (logged to
    stdout).

    Assignee resolution order:
      1. assignee_email → resolve to accountId via /user/search
      2. fallback_assignee_email (the call host, typically) → same
      3. Unassigned (no assignee field set) if both fail
    This matches the "fall back to call host" decision from the
    Block 4 setup conversation.

    due_date must be ISO YYYY-MM-DD or None. The SDK accepts this
    directly in the `duedate` field.

    issue_type defaults to "Task" — the safest cross-project default.
    If the target project doesn't have a Task type, the API will
    return 400 with a clear message; we log and return None.
    """
    fields: Dict[str, Any] = {
        "project": {"key": project_key},
        "summary": summary[:255],  # Jira hard-caps summary at 255 chars
        "issuetype": {"name": issue_type},
    }
    if description:
        fields["description"] = description

    # Assignee — try the captured owner's email first, then the host.
    account_id = None
    if assignee_email:
        account_id = jira_resolve_account_id(client, assignee_email)
    if not account_id and fallback_assignee_email:
        account_id = jira_resolve_account_id(client, fallback_assignee_email)
    if account_id:
        fields["assignee"] = {"accountId": account_id}

    if due_date:
        fields["duedate"] = due_date

    # Priority: Jira Cloud defaults are Highest/High/Medium/Low/Lowest.
    # We capitalize the first letter so callers can pass either "high"
    # or "High" and the API still accepts it.
    if priority:
        fields["priority"] = {"name": priority.strip().capitalize()}

    # Story points: discovered (2026-05-20 inspection of SHP-17182)
    # to live at customfield_10020 on Shypple's instance. Pass as a
    # number (float or int) — Jira accepts both.
    if story_points is not None:
        fields[STORY_POINTS_FIELD] = float(story_points)

    # Epic / parent: Shypple's SHP is a team-managed project, so the
    # epic link is the `parent` shape, not the classic
    # customfield_10014. Verified by inspecting SHP-17182's parent
    # (SHP-15162). For other Jira instances, this would need a
    # configuration-time check.
    if epic_key:
        fields["parent"] = {"key": epic_key.strip().upper()}

    # Sprint: customfield_10018 on SHP. Accepts a single int on
    # create even though it reads back as a list.
    if sprint_id:
        fields[SPRINT_FIELD] = int(sprint_id)

    # Components: array of dicts with `name`. Names must match the
    # project's defined components (case-sensitive on the Jira side
    # — the caller is responsible for normalising to the exact
    # spelling via the cached component list).
    if components:
        fields["components"] = [{"name": c} for c in components if c]

    # Labels: free-text array. Jira accepts any string; convention
    # is no spaces (use hyphens / underscores).
    if labels:
        fields["labels"] = [str(l) for l in labels if l]

    # Reporter: same accountId shape as assignee. Resolve from email.
    if reporter_email:
        reporter_account_id = jira_resolve_account_id(client, reporter_email)
        if reporter_account_id:
            fields["reporter"] = {"accountId": reporter_account_id}

    try:
        issue = client.issue_create(fields=fields)
    except Exception as exc:
        print(f"  [jira create error] {exc}")
        return None
    if not isinstance(issue, dict) or "key" not in issue:
        print(f"  [jira create] unexpected response: {issue!r}")
        return None
    return issue


def jira_get_issue(client, key: str) -> Optional[Dict[str, Any]]:
    """
    Fetch an issue by key (e.g. "SHP-17182"). Returns the issue dict
    on success, None if not found or on any error. Used by the
    mid-meeting lookup flow (Block 4c).
    """
    if not key:
        return None
    try:
        issue = client.issue(key)
    except Exception as exc:
        print(f"  [jira get error] {exc}")
        return None
    if not isinstance(issue, dict):
        return None
    return issue


def jira_search_jql(
    client,
    jql: str,
    fields: Optional[List[str]] = None,
    max_results: int = 50,
) -> List[Dict[str, Any]]:
    """
    Run a JQL query and return the matching issues as a list of dicts.
    Returns [] on any error or empty result.

    Used by the mid-meeting "how many open tickets are there for team
    X" capability — the caller builds a JQL string and asks for a
    bounded count. `max_results` is capped at 100 so we never accidentally
    page through huge backlogs during a live call (Jira REST defaults
    to 50; the SDK respects this kwarg).

    Field projection is opt-in: pass `fields=["summary","status","assignee"]`
    when you only need a few — keeps payload small for boards with
    thousands of tickets.
    """
    if not jql or not jql.strip():
        return []
    capped = max(1, min(int(max_results), 100))
    try:
        # The SDK exposes .jql() which is the JQL search endpoint
        # (POST /rest/api/3/search). Returns {"issues": [...], "total": N}.
        result = client.jql(
            jql,
            limit=capped,
            fields=",".join(fields) if fields else "*all",
        )
    except Exception as exc:
        print(f"  [jira jql error] {exc}")
        return []
    if not isinstance(result, dict):
        return []
    issues = result.get("issues") or []
    if not isinstance(issues, list):
        return []
    return issues


def jira_search_jql_total(client, jql: str) -> int:
    """
    Run a JQL query and return ONLY the total count (no issue payload).
    Returns -1 on any error so the caller can distinguish "no matches"
    (0) from "query failed" (-1).

    Implementation note: Atlassian Cloud rejects maxResults < 1
    ("The max results parameter has to be between 1 and 5,000"), so
    we ask for the minimum (1 issue) and only read the "total" key.
    The one issue payload returned is the cheapest we can do; we
    discard it.
    """
    if not jql or not jql.strip():
        return -1
    try:
        result = client.jql(jql, limit=1, fields="summary")
    except Exception as exc:
        print(f"  [jira jql count error] {exc}")
        return -1
    if not isinstance(result, dict):
        return -1
    total = result.get("total")
    if not isinstance(total, int):
        return -1
    return total


def jira_edit_issue(client, key: str, fields: Dict[str, Any]) -> bool:
    """
    Update fields on an existing issue. Returns True on success.

    `fields` is a partial fields dict matching the Jira REST API
    shape, e.g.:
      {"priority": {"name": "High"}}
      {"assignee": {"accountId": "5b10..."}}
      {"customfield_10016": 5}  # story points
      {"parent": {"key": "SHP-100"}}  # epic link (Cloud team-managed)

    Used by Block 4d (mid-meeting edit). Failures log and return
    False — caller decides whether to retry or surface the error.
    """
    if not key or not fields:
        return False
    try:
        client.update_issue_field(key, fields)
    except Exception as exc:
        print(f"  [jira edit error] {exc}")
        return False
    return True


# ----------------------------------------------------------------------------
# Formatting helpers — turn a Jira issue dict into Slack / Meet-chat text
# ----------------------------------------------------------------------------


def format_issue_url(base_url: str, key: str) -> str:
    """Build the Jira browse URL for a given issue key."""
    return f"{base_url.rstrip('/')}/browse/{key}"


def format_issue_for_chat(base_url: str, issue: Dict[str, Any]) -> str:
    """
    Single-line-ish Meet chat representation of a Jira issue.

    Used for both newly-created tickets (Block 4b) and mid-meeting
    lookups (Block 4c). Plain text — Google Meet chat doesn't render
    markdown, so we lean on emoji + line breaks for visual structure.
    """
    key = issue.get("key", "?")
    fields = issue.get("fields") or {}
    summary = fields.get("summary") or "(no summary)"
    status_obj = fields.get("status") or {}
    status = status_obj.get("name") if isinstance(status_obj, dict) else "Unknown"
    assignee_obj = fields.get("assignee") or {}
    assignee = (
        assignee_obj.get("displayName")
        if isinstance(assignee_obj, dict) and assignee_obj
        else "Unassigned"
    )
    priority_obj = fields.get("priority") or {}
    priority = (
        priority_obj.get("name")
        if isinstance(priority_obj, dict) and priority_obj
        else "—"
    )

    # Story points live in a custom field whose ID varies by Jira
    # configuration. SHP uses customfield_10020 (per the one-time
    # discovery hard-coded in STORY_POINTS_FIELD). Older Cloud
    # instances commonly use customfield_10016 (Story Points
    # Estimate) or customfield_10026 (Story Points). Try the
    # SHP-specific field first, then fall back to the older
    # common IDs. Show "—" if none resolves.
    story_points = None
    for fld in (STORY_POINTS_FIELD, "customfield_10016", "customfield_10026"):
        if fields.get(fld) is not None:
            story_points = fields[fld]
            break
    # Display ints cleanly (5.0 -> 5) while keeping fractional points
    # readable (2.5 stays 2.5).
    if story_points is None or story_points == "":
        sp_str = "—"
    elif isinstance(story_points, (int, float)) and story_points == int(story_points):
        sp_str = str(int(story_points))
    else:
        sp_str = str(story_points)

    url = format_issue_url(base_url, key)
    return (
        f"📋 {key} — {summary}\n"
        f"   Status: {status}  •  Assignee: {assignee}  •  "
        f"Priority: {priority}  •  Story Points: {sp_str}\n"
        f"   {url}"
    )


def format_created_issue_for_chat(base_url: str, issue: Dict[str, Any]) -> str:
    """
    Compact one-line confirmation for a freshly-created ticket. Used
    when posting the post-call summary to Meet chat (Block 4b).
    """
    key = issue.get("key", "?")
    fields = issue.get("fields") or {}
    summary = fields.get("summary") or "(no summary)"
    url = format_issue_url(base_url, key)
    return f"✅ [{key}] {summary} → {url}"


def format_issue_for_slack(base_url: str, issue: Dict[str, Any]) -> str:
    """
    Slack mrkdwn formatting for a newly-created ticket. Used to
    append to the existing post-call Slack summary (Block 4b).

    Slack supports `<url|text>` for hyperlinks and `*bold*`.
    """
    key = issue.get("key", "?")
    fields = issue.get("fields") or {}
    summary = fields.get("summary") or "(no summary)"
    url = format_issue_url(base_url, key)
    return f"• <{url}|*{key}*> — {summary}"
