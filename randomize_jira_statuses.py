#!/usr/bin/env python3
"""Randomize statuses of EXISTING Jira issues across one or more projects.

Moves 20% of all issues to 'In Progress' and 15% to 'Done' (random, disjoint sets).
The remaining ~65% are left untouched.

Usage:
  export JIRA_API_TOKEN='...'
  python3 randomize_jira_statuses.py \
    --site shivammokha94.atlassian.net \
    --user your-email@example.com

By default operates on: Delivery, Feature Tracking, Change Request, Product Delivery.
Override with --projects (names or keys).
"""

import argparse
import os
import random
import sys

try:
    import requests
except ImportError:
    print("The 'requests' package is required. Install it with: pip install requests")
    sys.exit(1)

DEFAULT_PROJECTS = ['Delivery', 'Feature Tracking', 'Change Request', 'Product Delivery']


def get_env_or_arg(env_name, arg_value, required=False):
    value = arg_value if arg_value else os.environ.get(env_name)
    if required and not value:
        print(f"Error: {env_name} is required (or pass via CLI)")
        sys.exit(1)
    return value


def resolve_project_key(session, site, auth, project_input):
    """Return the Jira project key matching a name or key."""
    url = f"https://{site}/rest/api/3/project/search"
    response = session.get(
        url, auth=auth,
        headers={'Accept': 'application/json'},
        params={'query': project_input},
    )
    if response.status_code != 200:
        raise RuntimeError(f"Project search failed for '{project_input}': {response.status_code} {response.text}")

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


def fetch_all_issues(session, site, auth, project_key):
    """Yield (issue_key, status_name) for every issue in a project via the enhanced JQL search API."""
    url = f"https://{site}/rest/api/3/search/jql"
    next_page_token = None
    while True:
        payload = {
            'jql': f'project = "{project_key}" ORDER BY created ASC',
            'maxResults': 100,
            'fields': ['status'],
        }
        if next_page_token:
            payload['nextPageToken'] = next_page_token

        response = session.post(
            url, auth=auth,
            headers={'Content-Type': 'application/json', 'Accept': 'application/json'},
            json=payload,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Search failed for {project_key}: {response.status_code} {response.text}")

        data = response.json()
        for issue in data.get('issues', []):
            status_name = issue.get('fields', {}).get('status', {}).get('name', '')
            yield issue['key'], status_name

        next_page_token = data.get('nextPageToken')
        if data.get('isLast') or not next_page_token:
            break


def get_transitions(session, site, auth, issue_key):
    url = f"https://{site}/rest/api/3/issue/{issue_key}/transitions"
    response = session.get(url, auth=auth, headers={'Accept': 'application/json'})
    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch transitions for {issue_key}: {response.status_code} {response.text}")
    return response.json().get('transitions', [])


def transition_issue(session, site, auth, issue_key, target_status_name):
    """Match by destination status first, then transition name (case-insensitive)."""
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
    response = session.post(
        url, auth=auth,
        headers={'Content-Type': 'application/json'},
        json={'transition': {'id': chosen['id']}},
    )
    if response.status_code not in (200, 204):
        raise RuntimeError(
            f"Failed to transition {issue_key} to '{target_status_name}': "
            f"{response.status_code} {response.text}"
        )


def main():
    parser = argparse.ArgumentParser(description='Randomize statuses for existing Jira issues.')
    parser.add_argument('--site', help='Jira site hostname, e.g. shivammokha94.atlassian.net')
    parser.add_argument('--user', help='Atlassian account email')
    parser.add_argument('--token', help='Atlassian API token')
    parser.add_argument(
        '--projects', nargs='+', default=DEFAULT_PROJECTS,
        help='Project names or keys. Defaults to the 4 standard test projects.',
    )
    parser.add_argument('--in-progress-pct', type=float, default=0.20,
                        help='Fraction of all issues to move to In Progress')
    parser.add_argument('--done-pct', type=float, default=0.15,
                        help='Fraction of all issues to move to Done')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--dry-run', action='store_true', help='Show targets without applying transitions')
    args = parser.parse_args()

    site = get_env_or_arg('JIRA_SITE', args.site, required=True)
    user = get_env_or_arg('JIRA_USER', args.user, required=True)
    token = get_env_or_arg('JIRA_API_TOKEN', args.token, required=True)

    session = requests.Session()
    auth = (user, token)

    project_keys = []
    for project_input in args.projects:
        try:
            key = resolve_project_key(session, site, auth, project_input)
            project_keys.append(key)
            print(f"Resolved '{project_input}' -> {key}")
        except Exception as exc:
            print(f"Skipping '{project_input}': {exc}")
    if not project_keys:
        print("No projects resolved. Exiting.")
        sys.exit(1)

    all_issues = []
    for pkey in project_keys:
        print(f"\nFetching issues in {pkey}...")
        count = 0
        for key, status in fetch_all_issues(session, site, auth, pkey):
            all_issues.append((key, status))
            count += 1
        print(f"  {count} issues")

    if not all_issues:
        print("No issues found. Exiting.")
        return

    n = len(all_issues)
    n_in_progress = int(round(n * args.in_progress_pct))
    n_done = int(round(n * args.done_pct))
    print(f"\nTotal issues across all projects: {n}")
    print(f"  -> {n_in_progress} to In Progress ({args.in_progress_pct:.0%})")
    print(f"  -> {n_done} to Done ({args.done_pct:.0%})")

    rng = random.Random(args.seed)
    shuffled = all_issues[:]
    rng.shuffle(shuffled)
    in_progress_targets = shuffled[:n_in_progress]
    done_targets = shuffled[n_in_progress:n_in_progress + n_done]

    if args.dry_run:
        print("\n[dry-run] Would transition to In Progress:")
        for key, status in in_progress_targets:
            print(f"  {key} (currently {status})")
        print("\n[dry-run] Would transition to Done:")
        for key, status in done_targets:
            print(f"  {key} (currently {status})")
        return

    print("\nTransitioning to In Progress...")
    in_progress_ok = 0
    for key, status in in_progress_targets:
        if status.lower() == 'in progress':
            in_progress_ok += 1
            print(f"  {key} already In Progress, skipping")
            continue
        try:
            transition_issue(session, site, auth, key, 'In Progress')
            in_progress_ok += 1
            print(f"  {key} -> In Progress (was {status})")
        except Exception as exc:
            print(f"  {key} error: {exc}")

    print("\nTransitioning to Done...")
    done_ok = 0
    for key, status in done_targets:
        if status.lower() == 'done':
            done_ok += 1
            print(f"  {key} already Done, skipping")
            continue
        try:
            transition_issue(session, site, auth, key, 'Done')
            done_ok += 1
            print(f"  {key} -> Done (was {status})")
        except Exception:
            try:
                transition_issue(session, site, auth, key, 'In Progress')
                transition_issue(session, site, auth, key, 'Done')
                done_ok += 1
                print(f"  {key} -> In Progress -> Done (was {status})")
            except Exception as exc2:
                print(f"  {key} error: {exc2}")

    print(f"\nSummary: {in_progress_ok}/{n_in_progress} In Progress, {done_ok}/{n_done} Done")


if __name__ == '__main__':
    main()
