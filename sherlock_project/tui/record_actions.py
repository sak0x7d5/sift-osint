"""The menu behind `⋯ Actions`: everything that can be done to one stored record.

It replaced a permanent `✕ Delete this username` button in the record header.
That button was the most saturated thing on the results screen, sat on every
section including the two where nobody is thinking about deleting anything, and
was the first Tab stop after the username list -- so the one irreversible action
in the app was also the easiest control to reach by accident.

Here Delete is still one step away for anyone looking for it, but it is last,
set apart by a rule, drawn red, and ends in an ellipsis because it opens a
confirmation rather than acting. Everything above it is safe.

**Each row names its key.** The menu is how the shortcuts are learned, not a
substitute for them: someone who opens it twice to export will see `^e` beside
the word and stop opening it.
"""

from __future__ import annotations

from typing import ClassVar

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, OptionList
from textual.widgets.option_list import Option

from sherlock_project.tui.theme import count_of, tone

# Width of the label column, so the keys line up down the right edge.
LABEL_WIDTH = 30


def _row(label: str, key: str = "", *, style: str = "") -> Text:
    line = Text(f"{label:<{LABEL_WIDTH}}", style=style)
    line.append(f"{key:>4}", style="dim" if not style else style)
    return line


class RecordActionsScreen(ModalScreen[str | None]):
    """Pick one action for a record. Dismisses with its id, or None."""

    BINDINGS: ClassVar = [Binding("escape", "close", "close")]

    def __init__(
        self,
        username: str,
        *,
        links: int = 0,
        person: bool = False,
        belongs_to: str | None = None,
    ) -> None:
        super().__init__()
        self._username = username
        self._links = links
        # `username` is a person's name when `person` is set. A person has no
        # scan of its own, so it has no re-scan or copy rows; it has rows for
        # its usernames instead, and its Delete removes the grouping only.
        self._person = person
        # The person a USERNAME belongs to, if any.
        self._belongs_to = belongs_to

    def _options(self) -> list[Option | None]:
        if self._person:
            return [
                Option(_row("Export JSON…", "^e"), id="export"),
                Option(_row("Show the combined profile"), id="profile"),
                Option(_row("Add a username…"), id="add_username"),
                Option(_row("Rename…"), id="rename_person"),
                None,
                Option(
                    _row(
                        f"Remove {self._username}…", "del", style=tone("bold error")
                    ),
                    id="delete",
                ),
            ]
        people: list[Option | None] = []
        if self._belongs_to is None:
            people.append(Option(_row("Add to a person…", "n"), id="add_to_person"))
        else:
            people.extend(
                [
                    Option(_row(f"Open ◆ {self._belongs_to}", "p"), id="open_person"),
                    Option(_row("Move to another person…", "n"), id="add_to_person"),
                    Option(
                        _row(f"Remove from {self._belongs_to}"),
                        id="remove_from_person",
                    ),
                ]
            )
        return [
            Option(_row("Export JSON…", "^e"), id="export"),
            Option(_row("Scan this username again…"), id="rescan"),
            Option(
                _row(
                    f"Copy {count_of(self._links, 'found link')}"
                    if self._links
                    else "Copy found links"
                ),
                id="copy",
                # Nothing to copy is a disabled row rather than a missing
                # one, so the menu keeps the same shape for every record.
                disabled=not self._links,
            ),
            Option(_row("Show the AI profile"), id="profile"),
            # People next, after the safe reads and before the divider: they
            # change only how usernames are grouped, never what is stored for
            # them.
            *people,
            None,
            Option(
                _row(f"Delete {self._username}…", "del", style=tone("bold error")),
                id="delete",
            ),
        ]

    def compose(self) -> ComposeResult:
        with Vertical(id="actions-menu"):
            # Text, not markup: the username is user data.
            title = Text()
            if self._person:
                title.append("◆ ", style=tone("accent"))
            title.append(self._username, style="bold")
            yield Label(title, id="actions-title")
            yield OptionList(*self._options(), id="actions-list")

    def on_mount(self) -> None:
        # The first row, which is safe. A menu that opened on its last row
        # would put Delete under Enter.
        self.query_one(OptionList).highlighted = 0

    @on(OptionList.OptionSelected)
    def _chosen(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id)

    def on_click(self, event) -> None:
        # A click outside the menu closes it, like any popup menu.
        if event.widget is self:
            self.dismiss(None)

    def action_close(self) -> None:
        self.dismiss(None)
