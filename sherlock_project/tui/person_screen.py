"""Dialogs for people: pick one, name one, and review suggested links.

A person is a named group of stored usernames that one human is believed to
use, with one profile built across all of them. These dialogs only ask; the
results pane does every write, so "viewing never writes" stays a rule of one
module rather than of four.

**Linking is always a choice someone makes.** The suggestions dialog lists
what the stored evidence connects and says why, line by line, so a link is
accepted on evidence someone has read -- never applied because a list said so.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, OptionList, Static
from textual.widgets.option_list import Option

from sherlock_project.person_links import LinkSuggestion
from sherlock_project.tui.theme import tone

# The id of the "make a new one" row in a person picker. Not a valid person
# name (names are trimmed, and this is all whitespace-free control text), so it
# cannot collide with a real person.
NEW_PERSON = "\x00new"

# Person glyph. One character, used wherever a row or a title is a PERSON
# rather than a username, so the two can be told apart at a glance.
PERSON_MARK = "◆"


class ChoiceScreen(ModalScreen[str | None]):
    """Pick one of a list. Dismisses with the chosen id, or None."""

    BINDINGS: ClassVar = [Binding("escape", "close", "close")]

    def __init__(
        self,
        title: str,
        blurb: str | Text,
        options: Sequence[tuple[str, str | Text]],
    ) -> None:
        super().__init__()
        self._title = title
        self._blurb = Text(blurb) if isinstance(blurb, str) else blurb
        self._options = list(options)

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static(Text(self._title), classes="dialog-title")
            yield Static(self._blurb, id="choice-blurb")
            yield OptionList(
                *(
                    Option(Text(label) if isinstance(label, str) else label, id=key)
                    for key, label in self._options
                ),
                id="choice-list",
            )
            yield Label("enter picks · esc cancels", classes="dim")

    def on_mount(self) -> None:
        listing = self.query_one(OptionList)
        if self._options:
            listing.highlighted = 0
        listing.focus()

    @on(OptionList.OptionSelected)
    def _chosen(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id)

    def action_close(self) -> None:
        self.dismiss(None)


def person_picker(username: str, people: Sequence[str]) -> ChoiceScreen:
    """"Add <username> to which person?" -- a new one first, then the rest.

    New first because the common case is the first link: a username that
    belongs to nobody yet, being given a person.
    """
    options: list[tuple[str, str | Text]] = [(NEW_PERSON, "+ New person…")]
    for name in people:
        label = Text()
        label.append(f"{PERSON_MARK} ", style=tone("accent"))
        label.append(name)
        options.append((name, label))
    return ChoiceScreen(
        f"Add {username} to a person",
        "Its pages will count toward that person's profile. Nothing is "
        "rebuilt until you press Build there.",
        options,
    )


class NameScreen(ModalScreen[str | None]):
    """Type a name. Dismisses with the trimmed name, or None on cancel."""

    BINDINGS: ClassVar = [Binding("escape", "close", "close")]

    def __init__(
        self,
        title: str,
        blurb: str,
        *,
        initial: str = "",
        confirm_label: str = "Save",
        taken: Sequence[str] = (),
    ) -> None:
        super().__init__()
        self._title = title
        self._blurb = blurb
        self._initial = initial
        self._confirm_label = confirm_label
        self._taken = {name.casefold() for name in taken}

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static(Text(self._title), classes="dialog-title")
            yield Static(Text(self._blurb), id="name-blurb")
            yield Input(value=self._initial, placeholder="e.g. Alice Smith", id="name-input")
            yield Static(id="name-status")
            with Horizontal(id="confirm-buttons"):
                yield Button("Cancel", id="name-cancel")
                yield Button(self._confirm_label, variant="primary", id="name-ok")
            yield Label("enter saves · esc cancels", classes="dim")

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    def _value(self) -> str | None:
        name = " ".join(self.query_one(Input).value.split())
        status = self.query_one("#name-status", Static)
        if not name:
            status.update(Text("A person needs a name.", style=tone("warning")))
            return None
        if len(name) > 80:
            status.update(Text("At most 80 characters.", style=tone("warning")))
            return None
        if name.casefold() in self._taken and name != self._initial:
            status.update(
                Text(f"There is already a person called {name}.", style=tone("warning"))
            )
            return None
        return name

    @on(Input.Submitted)
    @on(Button.Pressed, "#name-ok")
    def _save(self) -> None:
        name = self._value()
        if name is not None:
            self.dismiss(name)

    @on(Button.Pressed, "#name-cancel")
    def action_close(self) -> None:
        self.dismiss(None)


# What the suggestions dialog hands back: what to do, and to which pair.
SuggestionChoice = tuple[str, LinkSuggestion]


class LinkSuggestionsScreen(ModalScreen[SuggestionChoice | None]):
    """Usernames the stored evidence connects, each with the reasons why.

    Enter links the highlighted pair; `d` dismisses it for this session.
    A pair whose usernames already belong to two different people is shown
    but cannot be linked from here: that is a merge of two people, and it
    deserves more than one keypress.
    """

    BINDINGS: ClassVar = [
        Binding("escape", "close", "close"),
        Binding("d", "dismiss_one", "dismiss"),
    ]

    def __init__(self, suggestions: Sequence[LinkSuggestion]) -> None:
        super().__init__()
        self._suggestions = list(suggestions)

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static(
                Text(f"Suggested links ({len(self._suggestions)})"),
                classes="dialog-title",
            )
            yield Static(
                Text(
                    "Pairs of usernames whose stored pages point at each other: "
                    "one lists the other as a handle, links to its profile, or "
                    "shows the same email. Read the reasons before linking -- "
                    "a link merges both into one person's profile.",
                    style="dim",
                ),
                id="choice-blurb",
            )
            yield OptionList(
                *(
                    Option(
                        _suggestion_label(suggestion),
                        id=str(index),
                        disabled=suggestion.conflict is not None,
                    )
                    for index, suggestion in enumerate(self._suggestions)
                ),
                id="suggestion-list",
            )
            with Horizontal(id="confirm-buttons"):
                yield Button("Dismiss", id="suggestion-dismiss")
                yield Button("Link", variant="primary", id="suggestion-link")
            yield Label("enter links · d dismisses · esc closes", classes="dim")

    def on_mount(self) -> None:
        listing = self.query_one(OptionList)
        first = next(
            (
                index
                for index, suggestion in enumerate(self._suggestions)
                if suggestion.conflict is None
            ),
            None,
        )
        if first is not None:
            listing.highlighted = first
        listing.focus()

    def _current(self) -> LinkSuggestion | None:
        index = self.query_one(OptionList).highlighted
        if index is None or not 0 <= index < len(self._suggestions):
            return None
        return self._suggestions[index]

    @on(OptionList.OptionSelected)
    def _selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(("link", self._suggestions[int(event.option.id or 0)]))

    @on(Button.Pressed, "#suggestion-link")
    def _link(self) -> None:
        current = self._current()
        if current is not None and current.conflict is None:
            self.dismiss(("link", current))

    @on(Button.Pressed, "#suggestion-dismiss")
    def action_dismiss_one(self) -> None:
        current = self._current()
        if current is not None:
            self.dismiss(("dismiss", current))

    def action_close(self) -> None:
        self.dismiss(None)


def _suggestion_label(suggestion: LinkSuggestion) -> Text:
    """`alice ⇄ alice_dev`, then one dim line per reason."""
    label = Text()
    label.append(suggestion.first, style="bold")
    label.append("  ⇄  ", style=tone("accent"))
    label.append(suggestion.second, style="bold")
    if suggestion.conflict is not None:
        label.append(
            f"\n  already in two people: {suggestion.conflict[0]} and "
            f"{suggestion.conflict[1]}",
            style=tone("warning"),
        )
    for reason in suggestion.reasons:
        label.append(f"\n  {reason.text}", style="dim")
    return label
