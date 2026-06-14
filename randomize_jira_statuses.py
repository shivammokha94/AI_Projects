#!/usr/bin/env python3
"""Randomize statuses of EXISTING Jira issues across one or more projects.

Auth (env var only, kept out of shell history and ps output):
  export JIRA_API_TOKEN='...'
  python3 randomize_jira_statuses.py \
    --site shivammokha94.atlassian.net \
    --user you@example.com

Moves 20% of all issues to 'In Progress' and 15% to 'Done' (random, disjoint sets).
The remaining ~65% are left untouched.
"""

import argparse
import random
import sys

from jira_common import (
    build_session,
    confirm_or_exit,
    get_env_or_arg,
    request_with_retry,
    resolve_project_key,
    transition_issue,
    truncate,
    validate_site,
)


DEFAULT_PROJECTS = ['Delivery', 'Feature Tracking', 'Change Request', 'Product Delivery']


def fetch_all_issues(session, site, auth, project_key):
    """Yield (issue_key, status_name) for every issue in a project via enhanced JQL search."""
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

        response = request_with_retry(
            session, 'POST', url,
            auth=auth,
            headers={'Content-Type': 'application/json', 'Accept': 'application/json'},
            json=payload,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Search failed for {project_key}: {response.status_code} {truncate(response.text)}"
            )

        data = response.json()
        for issue in data.get('issues', []):
            key = issue.get('key')
            if not key:
                continue
            status_name = issue.get('fields', {}).get('status', {}).get('name', '')
            yield key, status_name

        next_page_token = data.get('nextPageToken')
        if data.get('isLast') or not next_page_token:
            break


def main():
    parser = argparse.ArgumentParser(description='Randomize statuses for existing Jira issues.')
    parser.add_argument('--site', help='Jira site hostname, e.g. shivammokha94.atlassian.net')
    parser.add_argument('--user', help='Atlassian account email')
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
    parser.add_argument('--yes', action='store_true', help='Skip the confirmation prompt')
    args = parser.parse_args()

    if not (0.0 <= args.in_progress_pct <= 1.0 and 0.0 <= args.done_pct <= 1.0):
        parser.error("--in-progress-pct and --done-pct must be between 0 and 1")
    if args.in_progress_pct + args.done_pct > 1.0:
        parser.error("--in-progress-pct + --done-pct must sum to <= 1.0")

    site = validate_site(get_env_or_arg('JIRA_SITE', args.site, required=True))
    user = get_env_or_arg('JIRA_USER', args.user, required=True)
    token = get_env_or_arg('JIRA_API_TOKEN', None, required=True)

    session = build_session()
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

    if not args.yes:
        confirm_or_exit("This will modify the issues listed above. Proceed?")

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
        except Exception as first:
            try:
                transition_issue(session, site, auth, key, 'In Progress')
                transition_issue(session, site, auth, key, 'Done')
                done_ok += 1
                print(f"  {key} -> In Progress -> Done (was {status})")
            except Exception as second:
                print(f"  {key} error: {second} (direct: {first})")

    print(f"\nSummary: {in_progress_ok}/{n_in_progress} In Progress, {done_ok}/{n_done} Done")


if __name__ == '__main__':
    main()
