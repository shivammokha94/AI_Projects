#!/usr/bin/env python3
"""Create test Jira issues across one or more projects with varied generated data.

Auth (token MUST come from the env var so it stays out of shell history and ps output):
  export JIRA_API_TOKEN='...'
  python3 create_jira_test_issues.py \
    --site shivammokha94.atlassian.net \
    --user you@example.com

By default creates --count issues in each of:
  Delivery, Feature Tracking, Change Request, Product Delivery

Each issue gets a randomly composed summary, a multi-paragraph ADF description,
a random priority, a random set of labels (always including 'auto-test'), and
optionally a random issue type. After creation the script adds 1:N / N:N / N:1
issue links and randomly transitions a configurable fraction to In Progress / Done.
"""

import argparse
import json
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
SAFE_ISSUE_TYPES = ['Task', 'Story', 'Bug']
PRIORITIES = ['Highest', 'High', 'Medium', 'Low', 'Lowest']

ACTION_VERBS = [
    'Implement', 'Refactor', 'Investigate', 'Document', 'Optimize',
    'Migrate', 'Add', 'Remove', 'Update', 'Fix', 'Review', 'Profile',
    'Audit', 'Replace', 'Consolidate', 'Extract', 'Harden', 'Re-test',
    'Roll out', 'Deprecate',
]
COMPONENTS = [
    'authentication flow', 'payment gateway', 'dashboard widget',
    'user onboarding', 'search index', 'notification service',
    'reporting pipeline', 'admin panel', 'API rate limiter',
    'session store', 'background job runner', 'email templates',
    'webhook handler', 'CSV export', 'audit log', 'feature flag system',
    'caching layer', 'permission model', 'analytics tracker',
    'invoice generator', 'image upload service', 'OAuth integration',
    'metrics exporter', 'release pipeline',
]
QUALIFIERS = [
    'for the mobile clients', 'on the staging cluster',
    'before the next release', 'as part of Q3 cleanup',
    'per the latest RFC', '(blocking release)', 'for accessibility audit',
    'after the recent migration', 'flagged by oncall',
]
DESCRIPTION_TEMPLATES = [
    "Customer reports indicate that {component} occasionally times out under sustained load. "
    "Reproduce with the load test harness and capture flamegraph data.",
    "Stakeholders requested {component} be aligned with the new design system. "
    "Update tokens, spacing, and component variants. No behavioural changes.",
    "Tech-debt cleanup: {component} contains a deprecated helper that was supposed "
    "to be removed last quarter. Replace call sites and delete.",
    "Discovery: investigate whether {component} can be replaced with a managed "
    "vendor solution. Produce a tradeoff doc and a rough cost estimate.",
    "Observed in production: {component} is logging at DEBUG level by default. "
    "Confirm the root cause, lower the verbosity, and add a regression test.",
    "Compliance follow-up: {component} needs to support the new data-retention "
    "policy. Add the cleanup job and verify with the audit team.",
    "Pager fired last night on {component}. Suspected race condition on startup. "
    "Capture the dump, add metrics, and write a postmortem entry.",
    "Customer success requested an enhancement to {component} so account admins "
    "can self-serve the most common configuration changes.",
]
ACCEPTANCE_CRITERIA = [
    "Unit tests cover the new behaviour and existing regressions still pass.",
    "Documentation is updated in the runbook.",
    "A feature flag gates the rollout, defaulting to off.",
    "Monitoring alerts are added before merging to main.",
    "Reviewed and signed off by the security guild.",
    "Performance budget unchanged on the homepage critical path.",
    "Backfill script tested against the staging snapshot.",
]
EXTRA_LABEL_POOL = [
    'frontend', 'backend', 'infra', 'security', 'perf',
    'tech-debt', 'ux', 'data', 'platform', 'compliance', 'mobile',
]


def generate_summary(rng):
    parts = [rng.choice(ACTION_VERBS), rng.choice(COMPONENTS)]
    if rng.random() < 0.4:
        parts.append(rng.choice(QUALIFIERS))
    return ' '.join(parts)


def generate_description(rng):
    template = rng.choice(DESCRIPTION_TEMPLATES)
    paragraph = template.format(component=rng.choice(COMPONENTS))
    criteria = rng.sample(ACCEPTANCE_CRITERIA, k=rng.randint(1, 3))
    return paragraph, criteria


def generate_labels(rng):
    base = ['auto-test', 'dashboard-sample']
    extras = rng.sample(EXTRA_LABEL_POOL, k=rng.randint(0, 2))
    return base + extras


def build_adf_description(paragraph, criteria):
    content = [
        {
            'type': 'paragraph',
            'content': [{'type': 'text', 'text': paragraph}],
        }
    ]
    if criteria:
        content.append({
            'type': 'paragraph',
            'content': [{
                'type': 'text',
                'text': 'Acceptance criteria:',
                'marks': [{'type': 'strong'}],
            }],
        })
        content.append({
            'type': 'bulletList',
            'content': [
                {
                    'type': 'listItem',
                    'content': [
                        {'type': 'paragraph', 'content': [{'type': 'text', 'text': line}]}
                    ],
                }
                for line in criteria
            ],
        })
    return {'type': 'doc', 'version': 1, 'content': content}


def build_issue_payload(project_key, summary, adf_description, issue_type, priority=None, labels=None):
    fields = {
        'project': {'key': project_key},
        'summary': summary,
        'description': adf_description,
        'issuetype': {'name': issue_type},
    }
    if priority:
        fields['priority'] = {'name': priority}
    if labels:
        fields['labels'] = labels
    return {'fields': fields}


def create_issue(session, site, auth, payload):
    url = f"https://{site}/rest/api/3/issue"
    response = request_with_retry(
        session, 'POST', url,
        auth=auth,
        headers={'Content-Type': 'application/json'},
        json=payload,
    )
    if response.status_code not in (200, 201):
        raise RuntimeError(f"Failed to create issue: {response.status_code} {truncate(response.text)}")
    return response.json()


def create_issue_link(session, site, auth, link_type, inward_key, outward_key):
    url = f"https://{site}/rest/api/3/issueLink"
    payload = {
        'type': {'name': link_type},
        'inwardIssue': {'key': inward_key},
        'outwardIssue': {'key': outward_key},
    }
    response = request_with_retry(
        session, 'POST', url,
        auth=auth,
        headers={'Content-Type': 'application/json'},
        json=payload,
    )
    if response.status_code not in (200, 201):
        raise RuntimeError(
            f"Failed to link {outward_key} -> {inward_key} ({link_type}): "
            f"{response.status_code} {truncate(response.text)}"
        )


def create_issues_for_project(session, site, auth, project_key, count, issue_type_arg, dry_run, rng):
    created = []
    for i in range(1, count + 1):
        summary = generate_summary(rng)
        paragraph, criteria = generate_description(rng)
        if issue_type_arg.lower() == 'random':
            issue_type = rng.choice(SAFE_ISSUE_TYPES)
        else:
            issue_type = issue_type_arg

        payload = build_issue_payload(
            project_key=project_key,
            summary=f"[{project_key}] {summary}",
            adf_description=build_adf_description(paragraph, criteria),
            issue_type=issue_type,
            priority=rng.choice(PRIORITIES),
            labels=generate_labels(rng),
        )

        if dry_run:
            print(json.dumps(payload, indent=2))
            continue

        try:
            result = create_issue(session, site, auth, payload)
            issue_key = result.get('key')
            created.append(issue_key)
            print(f"  Created {issue_key}: {summary}")
        except Exception as exc:
            print(f"  Error creating issue #{i} in {project_key}: {exc}")
            continue

    return created


def create_relationships(session, site, auth, totals, dry_run, rng):
    if dry_run:
        print("\nSkipping issue links (dry-run).")
        return

    all_keys = [k for keys in totals.values() for k in keys if k]
    if len(all_keys) < 4:
        print("\nNot enough issues to create relationships.")
        return

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

    print("\nCreating many-to-many links (20 random 'Relates' pairs)...")
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


def randomize_statuses(session, site, auth, totals, dry_run, rng, in_progress_pct, done_pct):
    if dry_run:
        print("\nSkipping status transitions (dry-run).")
        return

    all_keys = [k for keys in totals.values() for k in keys if k]
    if not all_keys:
        return

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
        except Exception as first:
            try:
                transition_issue(session, site, auth, key, 'In Progress')
                transition_issue(session, site, auth, key, 'Done')
                print(f"  {key} -> In Progress -> Done")
            except Exception as second:
                print(f"  transition error: {second} (direct: {first})")


def main():
    parser = argparse.ArgumentParser(description='Create realistic test Jira issues with links and statuses.')
    parser.add_argument('--site', help='Jira site hostname, e.g. shivammokha94.atlassian.net')
    parser.add_argument('--user', help='Atlassian account email')
    parser.add_argument(
        '--projects', nargs='+', default=DEFAULT_PROJECTS,
        help='Project names or keys. Defaults to the 4 standard test projects.',
    )
    parser.add_argument('--count', type=int, default=50, help='Issues to create per project')
    parser.add_argument('--issuetype', default='Task',
                        help='Issue type name, or "random" to randomize among Task/Story/Bug')
    parser.add_argument('--no-links', action='store_true', help='Skip issue link creation')
    parser.add_argument('--no-transitions', action='store_true', help='Skip randomized status transitions')
    parser.add_argument('--in-progress-pct', type=float, default=0.20,
                        help='Fraction of created issues to move to In Progress')
    parser.add_argument('--done-pct', type=float, default=0.15,
                        help='Fraction of created issues to move to Done')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--dry-run', action='store_true', help='Show payloads without creating anything')
    parser.add_argument('--yes', action='store_true', help='Skip the confirmation prompt')
    args = parser.parse_args()

    if not (0.0 <= args.in_progress_pct <= 1.0 and 0.0 <= args.done_pct <= 1.0):
        parser.error("--in-progress-pct and --done-pct must be between 0 and 1")
    if args.in_progress_pct + args.done_pct > 1.0:
        parser.error("--in-progress-pct + --done-pct must sum to <= 1.0")

    site = validate_site(get_env_or_arg('JIRA_SITE', args.site, required=True))
    user = get_env_or_arg('JIRA_USER', args.user, required=True)
    token = get_env_or_arg('JIRA_API_TOKEN', None, required=True)

    count_env = get_env_or_arg('ISSUE_COUNT', None)
    count = int(count_env) if count_env else args.count

    session = build_session()
    auth = (user, token)
    rng = random.Random(args.seed)

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

    print(f"\nPlanned: create {count} issues in each of {len(resolved)} project(s) on {site}.")
    if not args.dry_run and not args.yes:
        confirm_or_exit("Proceed?")

    totals = {}
    for project_input, key in resolved:
        print(f"\nCreating {count} issues in {project_input} ({key})")
        created = create_issues_for_project(
            session, site, auth, key, count,
            args.issuetype, args.dry_run, rng,
        )
        totals[key] = created

    if not args.no_links:
        create_relationships(session, site, auth, totals, args.dry_run, rng)

    if not args.no_transitions:
        randomize_statuses(
            session, site, auth, totals, args.dry_run, rng,
            args.in_progress_pct, args.done_pct,
        )

    if not args.dry_run:
        print("\nSummary:")
        for key, created in totals.items():
            print(f"  {key}: {len(created)} issues")
            if created:
                print(f"    First: {', '.join(created[:5])}")


if __name__ == '__main__':
    main()
