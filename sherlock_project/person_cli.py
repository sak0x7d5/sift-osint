"""`sherlock-rm person ...` -- group usernames that belong to one person.

A person is a named set of stored usernames with one profile built across all
of them. Every subcommand here either only reads (`list`, `show`, `suggest`)
or makes exactly the one change its name says. Nothing links usernames on its
own: `suggest` prints what the evidence connects, and a link exists only once
someone runs `link`.

`delete` removes the person and its profile and keeps every username and scan
result. That is the opposite of deleting a username, so it asks first unless
`--yes` is given.
"""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from collections.abc import Sequence
from typing import Any

from sherlock_project.database import (
    PersonError,
    SherlockDB,
    default_database_path,
)
from sherlock_project.investigation_context import parse_inline_anchor
from sherlock_project.notify import TerminalReporter
from sherlock_project.person_links import LinkSuggestion, load_link_suggestions
from sherlock_project.show import _load_profile


def build_person_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="sherlock-rm person",
        description=(
            "Group usernames that belong to one person, and build one profile "
            "across them."
        ),
    )
    # On every subcommand, so it can go after the command as well as before:
    # `person show X --no-color` is how the scan's own flags are written.
    common = ArgumentParser(add_help=False)
    common.add_argument(
        "--no-color", action="store_true", default=False, help="Don't color terminal output."
    )
    parser.add_argument("--no-color", action="store_true", default=False, dest="no_color_top")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def add(name: str, **kwargs: Any) -> ArgumentParser:
        return commands.add_parser(name, parents=[common], **kwargs)

    add("list", help="List every person and the usernames linked to it.")

    create = add("create", help="Create a person, optionally linking usernames.")
    create.add_argument("name")
    create.add_argument("usernames", nargs="*", metavar="USERNAME")

    link = add("link", help="Link stored usernames to an existing person.")
    link.add_argument("name")
    link.add_argument("usernames", nargs="+", metavar="USERNAME")

    unlink = add("unlink", help="Remove usernames from a person.")
    unlink.add_argument("name")
    unlink.add_argument("usernames", nargs="+", metavar="USERNAME")

    rename = add("rename", help="Rename a person.")
    rename.add_argument("old")
    rename.add_argument("new")

    delete = add(
        "delete",
        help="Remove a person and its profile. Usernames and results are kept.",
    )
    delete.add_argument("name")
    delete.add_argument("--yes", action="store_true", help="Do not ask first.")

    add(
        "suggest",
        help=(
            "List usernames the stored evidence connects (shared handles, links "
            "between profiles, the same email). Never writes."
        ),
    )

    show = add("show", help="Show a person's stored profile. Never writes.")
    show.add_argument("name")
    show.add_argument("--sources", action="store_true", help="Show every source URL.")
    show.add_argument("--verbose", "-v", action="store_true", help="Show build notes.")
    show.add_argument("--json", action="store_true", dest="as_json", help="Emit JSON.")

    build = add(
        "build",
        help="Build the person's profile from every linked username's stored pages.",
    )
    build.add_argument("name")
    build.add_argument(
        "--anchor",
        action="append",
        default=[],
        type=parse_inline_anchor,
        metavar="[TRUST:]FIELD=VALUE",
        help="An identity anchor, as for --ai-synthesize-only. Repeatable.",
    )
    build.add_argument(
        "--force", action="store_true", help="Rebuild even if the stored profile is current."
    )
    build.add_argument(
        "--analyse",
        action="store_true",
        help="Analyse stored pages the model has not read yet before merging.",
    )
    return parser


def describe_suggestion(suggestion: LinkSuggestion) -> str:
    return f"{suggestion.first} ⇄ {suggestion.second}"


async def _list(db: SherlockDB, reporter: TerminalReporter) -> int:
    people = await db.list_people()
    if not people:
        reporter.info(
            "No people yet. Create one with: sherlock-rm person create NAME USERNAME..."
        )
        return 0
    for person in people:
        names = ", ".join(person.usernames) if person.usernames else "no usernames yet"
        profile = "profile built" if person.has_profile else "no profile"
        reporter.info(
            person.name,
            detail=f"{names} · {person.claimed_sites} found · {profile}",
        )
    return 0


async def _link_all(
    db: SherlockDB, reporter: TerminalReporter, name: str, usernames: Sequence[str]
) -> int:
    status = 0
    for username in usernames:
        try:
            await db.link_username(name, username)
        except PersonError as error:
            reporter.failure(str(error))
            status = 1
        else:
            reporter.success(f"Linked {username} to {name}")
    return status


async def _suggest(db: SherlockDB, reporter: TerminalReporter) -> int:
    suggestions = await load_link_suggestions(db)
    if not suggestions:
        reporter.info("No links suggested by the stored evidence.")
        return 0
    for suggestion in suggestions:
        if suggestion.conflict is not None:
            detail = (
                f"already in different people: {suggestion.conflict[0]} and "
                f"{suggestion.conflict[1]}"
            )
            reporter.warning(describe_suggestion(suggestion), detail=detail)
        else:
            reporter.info(describe_suggestion(suggestion))
        for reason in suggestion.reasons:
            reporter.hint(reason.text)
    reporter.hint("Nothing was linked. Accept one with: sherlock-rm person link NAME USERNAME")
    return 0


async def _show(db: SherlockDB, reporter: TerminalReporter, args: Any) -> int:
    usernames = await db.get_person_usernames(args.name)
    if usernames is None:
        reporter.failure(f"No person called {args.name!r}")
        return 1
    cache = await db.get_person_profile_cache(args.name)
    raw = cache.profile_summary if cache is not None else None
    profile = _load_profile(raw)
    if args.as_json:
        print(
            json.dumps(
                {
                    "person": args.name,
                    "usernames": usernames,
                    "profile_updated_at": cache.updated_at if cache else None,
                    "profile": profile.model_dump(mode="json") if profile else None,
                },
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    reporter.info(
        f"Person {args.name!r}",
        detail=(", ".join(usernames) if usernames else "no usernames linked")
        + " · nothing written",
    )
    if profile is None:
        reporter.info(
            f"No profile stored for {args.name!r}. Build one with: "
            f"sherlock-rm person build {_quoted(args.name)}"
        )
        return 0
    reporter.info(
        f"Profile for {args.name!r}",
        detail=f"built {cache.updated_at if cache else 'at an unknown time'}",
    )
    reporter.render_profile(profile, show_sources=args.sources)
    return 0


async def _build(reporter: TerminalReporter, args: Any) -> int:
    # Imported here: the scan module is heavy, and every other subcommand is
    # a few SQLite statements.
    from sherlock_project.sherlock import run_synthesis_only

    failures = await run_synthesis_only(
        usernames=[],
        people=[args.name],
        force=args.force,
        inline_anchors=args.anchor,
        reporter=reporter,
        analyse_pending=args.analyse,
    )
    return 1 if failures else 0


def _quoted(name: str) -> str:
    return f'"{name}"' if " " in name else name


async def run_person(argv: Sequence[str]) -> int:
    args = build_person_parser().parse_args(list(argv))
    reporter = TerminalReporter(
        no_color=args.no_color or args.no_color_top,
        verbose=getattr(args, "verbose", False),
    )
    if args.command == "build":
        db = await SherlockDB.create(str(default_database_path()))
        try:
            if await db.get_person_usernames(args.name) is None:
                reporter.failure(f"No person called {args.name!r}")
                return 1
        finally:
            await db.close()
        return await _build(reporter, args)

    db = await SherlockDB.create(str(default_database_path()))
    try:
        if args.command == "list":
            return await _list(db, reporter)
        if args.command == "suggest":
            return await _suggest(db, reporter)
        if args.command == "show":
            return await _show(db, reporter, args)
        try:
            if args.command == "create":
                await db.create_person(args.name)
                reporter.success(f"Created {args.name}")
                return await _link_all(db, reporter, args.name, args.usernames)
            if args.command == "link":
                return await _link_all(db, reporter, args.name, args.usernames)
            if args.command == "unlink":
                status = 0
                for username in args.usernames:
                    if await db.unlink_username(args.name, username):
                        reporter.success(f"Removed {username} from {args.name}")
                    else:
                        reporter.failure(f"{username} is not linked to {args.name}")
                        status = 1
                return status
            if args.command == "rename":
                await db.rename_person(args.old, args.new)
                reporter.success(f"Renamed {args.old} to {args.new}")
                return 0
            if args.command == "delete":
                usernames = await db.get_person_usernames(args.name)
                if usernames is None:
                    reporter.failure(f"No person called {args.name!r}")
                    return 1
                if not args.yes and not _confirm(args.name, usernames):
                    reporter.info("Nothing deleted.")
                    return 1
                await db.delete_person(args.name)
                reporter.success(
                    f"Removed {args.name} and its profile",
                    detail=(
                        f"kept {', '.join(usernames)} and their results"
                        if usernames
                        else None
                    ),
                )
                return 0
        except PersonError as error:
            reporter.failure(str(error))
            return 1
    finally:
        await db.close()
    return 2


def _confirm(name: str, usernames: Sequence[str]) -> bool:
    kept = (
        f" The {len(usernames)} username(s) ({', '.join(usernames)}) and their "
        "scan results are kept."
        if usernames
        else ""
    )
    if not sys.stdin.isatty():
        print(f"Refusing to delete {name!r} without --yes (no terminal to ask on).")
        return False
    answer = input(f"Remove the person {name!r} and its profile?{kept} [y/N] ")
    return answer.strip().casefold() in {"y", "yes"}
