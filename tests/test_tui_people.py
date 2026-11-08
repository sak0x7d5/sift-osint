"""People in the results pane: the nested list, a person's record, and the dialogs.

Driven through the real app, against a seeded database, with the same settle
helpers the rest of the TUI suite uses so the database reads are waited on
rather than raced.
"""

from __future__ import annotations

import json

from test_tui import _open_results, _settle
from textual.widgets import DataTable, Static

from sherlock_project.ai_engine import pass_one_contract_hash
from sherlock_project.database import SherlockDB, default_database_path
from sherlock_project.result import QueryStatus
from sherlock_project.tui.app import SherlockUI
from sherlock_project.tui.person_screen import (
    ChoiceScreen,
    LinkSuggestionsScreen,
    NameScreen,
)
from sherlock_project.tui.results_pane import (
    SEC_SITES,
    SUGGESTIONS_KEY,
    ResultsPane,
    person_key,
)


async def _seed(rows, people=None) -> None:
    """rows: (username, site, url, facts). people: {name: [usernames]}."""
    db = await SherlockDB.create(str(default_database_path()))
    try:
        for username, site, url, facts in rows:
            site_id = await db.save_result(
                username=username,
                site_name=site,
                site_url=url,
                status=str(QueryStatus.CLAIMED),
                status_code=200,
                query_time_ms=1.0,
                response_text="<html>page</html>",
                transport="browser",
            )
            if facts is not None:
                await db.update_result_ai_extraction(
                    site_id,
                    json.dumps(facts),
                    contract_hash=pass_one_contract_hash(),
                    model_key="test/model",
                )
        for name, usernames in (people or {}).items():
            await db.create_person(name)
            for username in usernames:
                await db.link_username(name, username)
    finally:
        await db.close()


ROWS = [
    ("alice", "GitHub", "https://github.com/alice", {"full_name": ["Alice Smith"]}),
    (
        "alice_dev",
        "GitHub",
        "https://github.com/alice_dev",
        {"other_usernames": ["alice"]},
    ),
    ("bob", "GitHub", "https://github.com/bob", {"full_name": ["Bob"]}),
]


def _list_keys(app) -> list[str]:
    table = app.query_one("#username-list", DataTable)
    return [str(row.key.value) for row in table.ordered_rows]


def _list_labels(app) -> list[str]:
    table = app.query_one("#username-list", DataTable)
    return [str(table.get_row_at(index)[0]) for index in range(table.row_count)]


def _text(app, selector: str, *, screen: bool = False) -> str:
    """What a Static shows, as plain text, whatever renderable it holds."""
    from io import StringIO

    from rich.console import Console

    root = app.screen if screen else app
    content = root.query_one(selector, Static).content
    buffer = StringIO()
    Console(file=buffer, width=200, color_system=None).print(content)
    return buffer.getvalue()


def _pane(app) -> ResultsPane:
    return app.query_one(ResultsPane)


async def test_a_person_heads_its_usernames_in_the_list():
    await _seed(ROWS, people={"Alice Smith": ["alice", "alice_dev"]})
    app = SherlockUI()
    async with app.run_test(size=(130, 40)) as pilot:
        await _open_results(app, pilot)
        await _settle(app, pilot, lambda: len(_list_keys(app)) >= 4)

        keys = _list_keys(app)
        # Person first, its usernames under it, then the loose one.
        assert keys[:4] == [person_key("Alice Smith"), "alice", "alice_dev", "bob"]
        labels = _list_labels(app)
        assert labels[0] == "◆ Alice Smith"
        assert labels[1] == "├ alice"
        assert labels[2] == "└ alice_dev"
        assert labels[3] == "bob"
        assert _text(app, ".pane-title").strip() == "PEOPLE & USERNAMES"


async def test_opening_a_person_shows_every_linked_account_labelled():
    await _seed(ROWS, people={"Alice Smith": ["alice", "alice_dev"]})
    app = SherlockUI()
    async with app.run_test(size=(130, 40)) as pilot:
        await _open_results(app, pilot)
        pane = _pane(app)
        pane.select_key(person_key("Alice Smith"))
        await _settle(
            app,
            pilot,
            lambda: (pane._record or {}).get("person") == "Alice Smith",
        )
        sites = app.query_one(f"#{SEC_SITES}", DataTable)
        names = {str(sites.get_row_at(i)[1]) for i in range(sites.row_count)}
        assert names == {"GitHub (alice)", "GitHub (alice_dev)"}
        header = _text(app, "#detail-header")
        assert "◆ Alice Smith" in header
        assert "2 usernames: alice, alice_dev" in header


async def test_a_linked_username_names_its_person_and_p_opens_it():
    await _seed(ROWS, people={"Alice Smith": ["alice"]})
    app = SherlockUI()
    async with app.run_test(size=(130, 40)) as pilot:
        await _open_results(app, pilot)
        pane = _pane(app)
        pane.select_key("alice")
        await _settle(app, pilot, lambda: (pane._record or {}).get("username") == "alice")
        assert "part of ◆ Alice Smith" in _text(app, "#detail-header")
        pane.action_open_person()
        await _settle(
            app, pilot, lambda: (pane._record or {}).get("person") == "Alice Smith"
        )
        assert pane._selected == person_key("Alice Smith")


async def test_adding_a_username_to_a_new_person_creates_and_links_it():
    await _seed(ROWS)
    app = SherlockUI()
    async with app.run_test(size=(130, 40)) as pilot:
        await _open_results(app, pilot)
        pane = _pane(app)
        pane.select_key("bob")
        await _settle(app, pilot, lambda: pane._selected == "bob")

        pane.action_add_to_person()
        await _settle(app, pilot, lambda: isinstance(app.screen, ChoiceScreen))
        await pilot.press("enter")  # "+ New person…" is first and highlighted
        await _settle(app, pilot, lambda: isinstance(app.screen, NameScreen))
        app.screen.query_one("#name-input").value = "Robert"
        await pilot.press("enter")
        await _settle(app, pilot, lambda: person_key("Robert") in _list_keys(app))

    db = await SherlockDB.create(str(default_database_path()))
    try:
        assert await db.get_person_usernames("Robert") == ["bob"]
    finally:
        await db.close()


async def test_suggested_links_are_offered_and_accepting_one_links_the_pair():
    await _seed(ROWS)
    app = SherlockUI()
    async with app.run_test(size=(130, 40)) as pilot:
        await _open_results(app, pilot)
        await _settle(app, pilot, lambda: SUGGESTIONS_KEY in _list_keys(app))
        assert _list_labels(app)[-1] == "+ 1 suggested"

        _pane(app).action_link_suggestions()
        await _settle(app, pilot, lambda: isinstance(app.screen, LinkSuggestionsScreen))
        await pilot.press("enter")
        await _settle(app, pilot, lambda: isinstance(app.screen, NameScreen))
        # Named after the first of the pair by default.
        assert app.screen.query_one("#name-input").value == "alice"
        await pilot.press("enter")
        await _settle(app, pilot, lambda: person_key("alice") in _list_keys(app))
        # Linked, so no longer suggested.
        assert SUGGESTIONS_KEY not in _list_keys(app)

    db = await SherlockDB.create(str(default_database_path()))
    try:
        assert sorted(await db.get_person_usernames("alice")) == ["alice", "alice_dev"]
    finally:
        await db.close()


async def test_dismissing_a_suggestion_writes_nothing():
    await _seed(ROWS)
    app = SherlockUI()
    async with app.run_test(size=(130, 40)) as pilot:
        await _open_results(app, pilot)
        await _settle(app, pilot, lambda: SUGGESTIONS_KEY in _list_keys(app))
        _pane(app).action_link_suggestions()
        await _settle(app, pilot, lambda: isinstance(app.screen, LinkSuggestionsScreen))
        await pilot.press("d")
        await _settle(app, pilot, lambda: SUGGESTIONS_KEY not in _list_keys(app))

    db = await SherlockDB.create(str(default_database_path()))
    try:
        assert await db.list_people() == []
    finally:
        await db.close()


async def test_removing_a_person_keeps_its_usernames():
    await _seed(ROWS, people={"Alice Smith": ["alice", "alice_dev"]})
    app = SherlockUI()
    async with app.run_test(size=(130, 40)) as pilot:
        await _open_results(app, pilot)
        pane = _pane(app)
        pane.select_key(person_key("Alice Smith"))
        await _settle(
            app, pilot, lambda: (pane._record or {}).get("person") == "Alice Smith"
        )
        pane.action_delete_username()
        await _settle(app, pilot, lambda: bool(app.screen.query("#confirm-yes")))
        detail = _text(app, "#confirm-detail", screen=True)
        assert "Kept:" in detail and "alice, alice_dev" in detail
        await pilot.click("#confirm-yes")
        await _settle(
            app, pilot, lambda: person_key("Alice Smith") not in _list_keys(app)
        )
        assert {"alice", "alice_dev", "bob"} <= set(_list_keys(app))


async def test_building_a_person_profile_merges_both_usernames():
    await _seed(ROWS, people={"Alice Smith": ["alice", "alice_dev"]})
    app = SherlockUI()
    async with app.run_test(size=(130, 40)) as pilot:
        await _open_results(app, pilot)
        pane = _pane(app)
        pane.select_key(person_key("Alice Smith"))
        await _settle(
            app, pilot, lambda: (pane._record or {}).get("person") == "Alice Smith"
        )
        pane._build_profile()
        await _settle(
            app,
            pilot,
            lambda: (pane._record or {}).get("profile") is not None,
            tries=60,
        )
        profile = pane._record["profile"]
        assert profile.person == "Alice Smith"
        assert profile.usernames == ["alice", "alice_dev"]
        assert {d.username for d in profile.source_decisions} == {"alice", "alice_dev"}
        # alice_dev's page lists "alice": kept, because it is not alice_dev's
        # own handle, and the person's name was never used as a filter.
        assert profile.strong_profile["full_name"] == ["Alice Smith"]
        assert profile.strong_profile["other_usernames"] == ["alice"]
