"""`sherlock-rm person ...` against a real (temporary) database.

`list`, `show` and `suggest` must never write; every other subcommand makes the
one change its name says, and `delete` keeps the usernames.
"""

from __future__ import annotations

import json

import pytest

from sherlock_project.ai_engine import pass_one_contract_hash
from sherlock_project.database import SherlockDB, default_database_path
from sherlock_project.person_cli import run_person
from sherlock_project.result import QueryStatus


async def _seed() -> None:
    db = await SherlockDB.create(str(default_database_path()))
    try:
        for username, site, facts in (
            ("alice", "GitHub", {"full_name": ["Alice Smith"], "email": ["a@x.io"]}),
            ("alice_dev", "GitLab", {"roles": ["Engineer"], "email": ["a@x.io"]}),
        ):
            site_id = await db.save_result(
                username=username,
                site_name=site,
                site_url=f"https://{site.lower()}.com/{username}",
                status=str(QueryStatus.CLAIMED),
                response_text="<html>page</html>",
            )
            await db.update_result_ai_extraction(
                site_id,
                json.dumps(facts),
                contract_hash=pass_one_contract_hash(),
                model_key="m",
            )
    finally:
        await db.close()


async def _people() -> dict[str, list[str]]:
    db = await SherlockDB.create(str(default_database_path()))
    try:
        return {p.name: list(p.usernames) for p in await db.list_people()}
    finally:
        await db.close()


async def test_create_link_show_and_build(capsys):
    await _seed()
    assert await run_person(["create", "Alice Smith", "alice", "--no-color"]) == 0
    assert await run_person(["link", "Alice Smith", "alice_dev"]) == 0
    assert await _people() == {"Alice Smith": ["alice", "alice_dev"]}

    assert await run_person(["show", "Alice Smith", "--no-color"]) == 0
    assert "No profile stored" in capsys.readouterr().out

    assert await run_person(["build", "Alice Smith", "--no-color"]) == 0
    capsys.readouterr()
    assert await run_person(["show", "Alice Smith", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["usernames"] == ["alice", "alice_dev"]
    assert payload["profile"]["person"] == "Alice Smith"
    assert payload["profile"]["strong_profile"]["roles"] == ["Engineer"]


async def test_suggest_lists_evidence_and_writes_nothing(capsys):
    await _seed()
    assert await run_person(["suggest", "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "alice ⇄ alice_dev" in out
    assert "both show a@x.io" in out
    assert await _people() == {}


async def test_linking_an_unknown_username_fails_and_says_why(capsys):
    await _seed()
    await run_person(["create", "A"])
    assert await run_person(["link", "A", "ghost", "--no-color"]) == 1
    assert "nothing is stored for 'ghost'" in capsys.readouterr().out


async def test_delete_needs_yes_without_a_terminal_and_keeps_usernames(monkeypatch):
    await _seed()
    await run_person(["create", "A", "alice"])
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert await run_person(["delete", "A"]) == 1
    assert await _people() == {"A": ["alice"]}
    assert await run_person(["delete", "A", "--yes"]) == 0
    assert await _people() == {}
    db = await SherlockDB.create(str(default_database_path()))
    try:
        assert {u.username for u in await db.list_usernames()} == {"alice", "alice_dev"}
    finally:
        await db.close()


async def test_rename_and_unlink():
    await _seed()
    await run_person(["create", "A", "alice", "alice_dev"])
    assert await run_person(["rename", "A", "Alice"]) == 0
    assert await run_person(["unlink", "Alice", "alice_dev"]) == 0
    assert await _people() == {"Alice": ["alice"]}
    assert await run_person(["unlink", "Alice", "alice_dev"]) == 1


async def test_build_of_an_unknown_person_fails():
    assert await run_person(["build", "Nobody"]) == 1


def test_person_is_dispatched_before_the_scan_parser(monkeypatch):
    import asyncio

    from sherlock_project import sherlock as sherlock_module

    seen = {}

    async def fake(argv):
        seen["argv"] = argv
        return 0

    monkeypatch.setattr("sherlock_project.person_cli.run_person", fake)
    monkeypatch.setattr("sys.argv", ["sherlock-rm", "person", "list"])
    assert asyncio.run(sherlock_module.main()) == 0
    assert seen["argv"] == ["list"]


@pytest.mark.parametrize("command", [["list"], ["suggest"]])
async def test_read_only_commands_work_on_an_empty_database(command):
    assert await run_person(command) == 0
