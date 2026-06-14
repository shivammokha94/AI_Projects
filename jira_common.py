"""Shared Jira Cloud REST API helpers: auth, retries, project lookup, transitions."""

import os
import re
import sys
import time

try:
    import requests
except ImportError:
    print("The 'requests' package is required. Install it with: pip install requests")
    sys.exit(1)


DEFAULT_TIMEOUT = (10, 30)
MAX_RETRIES = 4
SITE_PATTERN = re.compile(r'^[A-Za-z0-9][A-Za-z0-9.-]*\.atlassian\.net$')


def get_env_or_arg(env_name, arg_value, required=False):
    value = arg_value if arg_value is not None else os.environ.get(env_name)
    if required and not value:
        print(f"Error: {env_name} is required (set env var or pass via CLI flag)")
        sys.exit(1)
    return value


def validate_site(site):
    if not SITE_PATTERN.match(site or ''):
        print(f"Error: --site '{site}' is not a valid Atlassian Cloud host (expected something.atlassian.net)")
        sys.exit(1)
    return site


def build_session():
    """Session with a default timeout injected into every request."""
    session = requests.Session()
    original_request = session.request

    def request_with_timeout(method, url, **kwargs):
        kwargs.setdefault('timeout', DEFAULT_TIMEOUT)
        return original_request(method, url, **kwargs)

    session.request = request_with_timeout
    return session


def request_with_retry(session, method, url, **kwargs):
    """Wraps session.request to handle 429 with Retry-After backoff."""
    last_response = None
    for _ in range(MAX_RETRIES):
        response = session.request(method, url, **kwargs)
        if response.status_code == 429:
            wait = int(response.headers.get('Retry-After', 5))
            print(f"  rate limited; sleeping {wait}s before retry")
            time.sleep(wait)
            last_response = response
            continue
        return response
    return last_response


def truncate(text, limit=200):
    if not text:
        return ''
    return text[:limit] + ('...' if len(text) > limit else '')


def confirm_or_exit(prompt):
    try:
        answer = input(f"{prompt} [y/N]: ").strip().lower()
    except EOFError:
        answer = ''
    if answer not in ('y', 'yes'):
        print("Aborted.")
        sys.exit(0)


def resolve_project_key(session, site, auth, project_input):
    url = f"https://{site}/rest/api/3/project/search"
    response = request_with_retry(
        session, 'GET', url,
        auth=auth,
        headers={'Accept': 'application/json'},
        params={'query': project_input},
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"Project search failed for '{project_input}': "
            f"{response.status_code} {truncate(response.text)}"
        )

    values = response.json().get('values', [])
    target = project_input.strip().lower()
    for project in values:
        if project.get('key', '').lower() == target:
            return project['key']
    for project in values:
        if project.get('name', '').lower() == target:
            return project['key']
    if len(values) == 1:
        return values[0]['key']
    if values:
        options = ', '.join(f"{p.get('name')} ({p.get('key')})" for p in values)
        raise RuntimeError(f"Ambiguous project '{project_input}'. Candidates: {options}")
    raise RuntimeError(f"No Jira project found matching '{project_input}'.")


def get_transitions(session, site, auth, issue_key):
    url = f"https://{site}/rest/api/3/issue/{issue_key}/transitions"
    response = request_with_retry(
        session, 'GET', url,
        auth=auth,
        headers={'Accept': 'application/json'},
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to fetch transitions for {issue_key}: "
            f"{response.status_code} {truncate(response.text)}"
        )
    return response.json().get('transitions', [])


def transition_issue(session, site, auth, issue_key, target_status_name):
    """Match by destination status name first, then transition name."""
    transitions = get_transitions(session, site, auth, issue_key)
    target = target_status_name.strip().lower()

    chosen = None
    for t in transitions:
        if t.get('to', {}).get('name', '').lower() == target:
            chosen = t
            break
    if not chosen:
        for t in transitions:
            if t.get('name', '').lower() == target:
                chosen = t
                break
    if not chosen:
        available = ', '.join(f"{t.get('name')}->{t.get('to', {}).get('name')}" for t in transitions)
        raise RuntimeError(f"No transition to '{target_status_name}' for {issue_key}. Available: {available}")

    url = f"https://{site}/rest/api/3/issue/{issue_key}/transitions"
    response = request_with_retry(
        session, 'POST', url,
        auth=auth,
        headers={'Content-Type': 'application/json'},
        json={'transition': {'id': chosen['id']}},
    )
    if response.status_code not in (200, 204):
        raise RuntimeError(
            f"Failed to transition {issue_key} to '{target_status_name}': "
            f"{response.status_code} {truncate(response.text)}"
        )
