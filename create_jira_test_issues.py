#!/usr/bin/env python3
"""Create test Jira issues across one or more projects via Atlassian Cloud REST API.

Usage:
  python create_jira_test_issues.py \
    --site shivammokha94.atlassian.net \
    --user your-email@example.com \
    --token YOUR_API_TOKEN \
    --count 50

By default this creates --count issues in each of:
  Delivery, Feature Tracking, Change Request, Product Delivery

Override with --projects (names or keys, space-separated):
  --projects DEL "Feature Tracking" CR PD

Environment variables (used if matching CLI flag is omitted):
  JIRA_SITE, JIRA_USER, JIRA_API_TOKEN, ISSUE_COUNT
"""

import argparse
import json
import os
import random
import sys

try:
    import requests
except ImportError:
    print("The 'requests' package is required. Install it with: pip install requests")
    sys.exit(1)

DEFAULT_PROJECTS = ['Delivery', 'Feature Tracking', 'Change Request', 'Product Delivery']
ISSUE_TYPES = ['Task', 'Story', 'Bug', 'Improvement', 'Epic', 'Spike']
PRIORITIES = ['Highest', 'High', 'Medium', 'Low', 'Lowest']


def get_env_or_arg(env_name, arg_value, required=False):
    value = arg_value if arg_value else os.environ.get(env_name)
    if required and not value:
        print(f"Error: {env_name} is required (or pass via CLI)")
        sys.exit(1)
    return value


def resolve_project_key(session, site, auth, project_input):
    """Return a Jira project key for the given input (name or key)."""
    url = f"https://{site}/rest/api/3/project/search"
    params = {'query': project_input}
    response = session.get(url, auth=auth, headers={'Accept': 'application/json'}, params=params)
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


def build_issue_payload(project_key, summary, description, issue_type, priority=None, labels=None):
    fields = {
        'project': {'key': project_key},
        'summary': summary,
        'description': {
            'type': 'doc',
            'version': 1,
            'content': [
                {
                    'type': 'paragraph',
                    'content': [
                        {'type': 'text', 'text': description}
                    ]
                }
            ]
        },
        'issuetype': {'name': issue_type}
    }

    if priority:
        fields['priority'] = {'name': priority}

    if labels:
        fields['labels'] = labels

    return {'fields': fields}


def create_issue(session, site, auth, payload):
    url = f"https://{site}/rest/api/3/issue"
    response = session.post(url, auth=auth, headers={'Content-Type': 'application/json'}, json=payload)
    if response.status_code not in (200, 201):
        raise RuntimeError(f"Failed to create issue: {response.status_code} {response.text}")
    return response.json()


def create_issue_link(session, site, auth, link_type, inward_key, outward_key):
    """Link two issues. For 'Blocks', outward = blocker, inward = blocked."""
    url = f"https://{site}/rest/api/3/issueLink"
    payload = {
        'type': {'name': link_type},
        'inwardIssue': {'key': inward_key},
        'outwardIssue': {'key': outward_key},
    }
    response = session.post(url, auth=auth, headers={'Content-Type': 'application/json'}, json=payload)
    if response.status_code not in (200, 201):
        raise RuntimeError(
            f"Failed to link {outward_key} -> {inward_key} ({link_type}): "
            f"{response.status_code} {response.text}"
        )


def get_transitions(session, site, auth, issue_key):
    url = f"https://{site}/rest/api/3/issue/{issue_key}/transitions"
    response = session.get(url, auth=auth, headers={'Accept': 'application/json'})
    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch transitions for {issue_key}: {response.status_code} {response.text}")
    return response.json().get('transitions', [])


def transition_issue(session, site, auth, issue_key, target_status_name):
    """Move issue to the given target status (case-insensitive match on destination status, then transition name)."""
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
        url,
        auth=auth,
        headers={'Content-Type': 'application/json'},
        json={'transition': {'id': chosen['id']}},
    )
    if response.status_code not in (200, 204):
        raise RuntimeError(
            f"Failed to transition {issue_key} to '{target_status_name}': "
            f"{response.status_code} {response.text}"
        )


def randomize_statuses(session, site, auth, totals, dry_run, seed=None,
                       in_progress_pct=0.20, done_pct=0.15):
    if dry_run:
        print("\nSkipping status transitions (dry-run).")
        return

    all_keys = [k for keys in totals.values() for k in keys if k]
    if not all_keys:
        return

    rng = random.Random(seed)
    shuffled = all_keys[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_in_progress = int(round(n * in_progress_pct))
    n_done = int(round(n * done_pct))
    in_progress_keys = shuffled[:n_in_progress]
    done_keys = shuffled[n_in_progress:n_in_progress + n_done]

    print(f"\nTransitioning {len(in_progress_keys)} issues ({in_progress_pct:.0%}) to 'In Progress'...")
    for key in in_progress_keys:
        try:
            transition_issue(session, site, auth, key, 'In Progress')
            print(f"  {key} -> In Progress")
        except Exception as exc:
            print(f"  transition error: {exc}")

    print(f"\nTransitioning {len(done_keys)} issues ({done_pct:.0%}) to 'Done'...")
    for key in done_keys:
        try:
            transition_issue(session, site, auth, key, 'Done')
            print(f"  {key} -> Done")
        except Exception:
            try:
                transition_issue(session, site, auth, key, 'In Progress')
                transition_issue(session, site, auth, key, 'Done')
                print(f"  {key} -> In Progress -> Done")
            except Exception as exc2:
                print(f"  transition error: {exc2}")


def create_relationships(session, site, auth, totals, dry_run, seed=None):
    """Create one-to-many, many-to-many, and many-to-one issue links across the created issues."""
    if dry_run:
        print("\nSkipping issue links (dry-run).")
        return

    all_keys = [k for keys in totals.values() for k in keys if k]
    if len(all_keys) < 4:
        print("\nNot enough issues to create relationships.")
        return

    rng = random.Random(seed)
    links_made = 0

    print("\nCreating one-to-many links (parent 'Blocks' 5 children, per project)...")
    for keys in totals.values():
        if len(keys) < 6:
            continue
        parent = keys[0]
        for child in keys[1:6]:
            try:
                create_issue_link(session, site, auth, 'Blocks', inward_key=child, outward_key=parent)
                links_made += 1
                print(f"  {parent} blocks {child}")
            except Exception as exc:
                print(f"  link error: {exc}")

    print("\nCreating many-to-many links (20 random cross-project 'Relates' pairs)...")
    pairs_target = min(20, len(all_keys) // 2)
    seen_pairs = set()
    attempts = 0
    while len(seen_pairs) < pairs_target and attempts < pairs_target * 5:
        attempts += 1
        a, b = rng.sample(all_keys, 2)
        pair = tuple(sorted((a, b)))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        try:
            create_issue_link(session, site, auth, 'Relates', inward_key=b, outward_key=a)
            links_made += 1
            print(f"  {a} relates to {b}")
        except Exception as exc:
            print(f"  link error: {exc}")

    print("\nCreating many-to-one links (5 issues 'Blocks' one hub, per project)...")
    for keys in totals.values():
        if len(keys) < 12:
            continue
        hub = keys[-1]
        for src in keys[6:11]:
            try:
                create_issue_link(session, site, auth, 'Blocks', inward_key=hub, outward_key=src)
                links_made += 1
                print(f"  {src} blocks {hub}")
            except Exception as exc:
                print(f"  link error: {exc}")

    print(f"\nCreated {links_made} issue links.")


def create_issues_for_project(session, site, auth, project_key, count, issue_type, dry_run):
    created = []
    for i in range(1, count + 1):
        summary = f"[{project_key}] Auto-generated test issue #{i}"
        description = (
            "This is a test issue created by create_jira_test_issues.py for dashboard validation. "
            "Use this issue to verify that Jira project page data is visible in your Forge app."
        )
        payload = build_issue_payload(
            project_key=project_key,
            summary=summary,
            description=description,
            issue_type=issue_type if issue_type else random.choice(ISSUE_TYPES),
            priority=random.choice(PRIORITIES),
            labels=['auto-test', 'dashboard-sample']
        )

        if dry_run:
            print(json.dumps(payload, indent=2))
            continue

        try:
            result = create_issue(session, site, auth, payload)
            issue_key = result.get('key')
            created.append(issue_key)
            print(f"  Created {issue_key}")
        except Exception as exc:
            print(f"  Error creating issue #{i} in {project_key}: {exc}")
            break

    return created


def main():
    parser = argparse.ArgumentParser(description='Create test Jira issues for dashboard testing.')
    parser.add_argument('--site', help='Jira site hostname, e.g. shivammokha94.atlassian.net')
    parser.add_argument('--user', help='Atlassian account email')
    parser.add_argument('--token', help='Atlassian API token')
    parser.add_argument(
        '--projects',
        nargs='+',
        default=DEFAULT_PROJECTS,
        help='Jira project names or keys (space-separated). Defaults to the 4 standard test projects.',
    )
    parser.add_argument('--count', type=int, default=50, help='Number of issues to create per project')
    parser.add_argument('--issuetype', default='Task', help='Jira issue type to use')
    parser.add_argument('--dry-run', action='store_true', help='Show issue payloads without creating them')
    parser.add_argument('--no-links', action='store_true', help='Skip creating issue link relationships')
    parser.add_argument('--no-transitions', action='store_true', help='Skip randomized status transitions')
    parser.add_argument('--in-progress-pct', type=float, default=0.20, help='Fraction of issues to move to In Progress')
    parser.add_argument('--done-pct', type=float, default=0.15, help='Fraction of issues to move to Done')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducible link pairs / status picks')
    args = parser.parse_args()

    site = get_env_or_arg('JIRA_SITE', args.site, required=True)
    user = get_env_or_arg('JIRA_USER', args.user, required=True)
    token = get_env_or_arg('JIRA_API_TOKEN', args.token, required=True)
    count_env = os.environ.get('ISSUE_COUNT')
    count = int(count_env) if count_env else args.count
    issue_type = args.issuetype

    if issue_type not in ISSUE_TYPES:
        print(f"Warning: '{issue_type}' is not in the standard issue type list. Proceeding anyway.")

    session = requests.Session()
    auth = (user, token)

    resolved = []
    for project_input in args.projects:
        try:
            key = resolve_project_key(session, site, auth, project_input)
            resolved.append((project_input, key))
            print(f"Resolved '{project_input}' -> {key}")
        except Exception as exc:
            print(f"Skipping '{project_input}': {exc}")

    if not resolved:
        print("No projects could be resolved. Exiting.")
        sys.exit(1)

    totals = {}
    for project_input, key in resolved:
        print(f"\nCreating {count} issues in {project_input} ({key}) on {site}")
        created = create_issues_for_project(session, site, auth, key, count, issue_type, args.dry_run)
        totals[key] = created

    if not args.no_links:
        create_relationships(session, site, auth, totals, args.dry_run, seed=args.seed)

    if not args.no_transitions:
        randomize_statuses(
            session, site, auth, totals, args.dry_run,
            seed=args.seed,
            in_progress_pct=args.in_progress_pct,
            done_pct=args.done_pct,
        )

    if not args.dry_run:
        print("\nSummary:")
        for key, created in totals.items():
            print(f"  {key}: {len(created)} issues")
            if created:
                print(f"    First: {', '.join(created[:5])}")


if __name__ == '__main__':
    main()
