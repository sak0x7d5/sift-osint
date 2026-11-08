"""Suggested links between stored usernames that may belong to one person.

Pure apart from `load_link_suggestions` at the bottom, which only reads:
no model, no network. It is handed what is already stored
and says which pairs of usernames the evidence itself connects, and why.

**Only signals that name the other account count.** A wrong link is worse than
a missed one -- it merges two people's facts into one profile -- so the rules
are the ones where a page points at the other username directly:

1. A page lists the other username as one of its owner's handles
   (`usernames` / `aliases`).
2. A page links to a profile URL the scan found for the other username.
3. Both usernames' pages state the same email address.

A shared display name, location or employer is NOT a suggestion. "Alex" in
"Berlin" describes thousands of people, and a list full of those would teach
someone to click Link without reading.

Nothing here writes. A suggestion becomes a link only when someone accepts it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sherlock_project.global_synthesis import link_targets
from sherlock_project.profile_synthesis import (
    canonical_field,
    is_username_variant,
    normalize_value,
)

if TYPE_CHECKING:
    from sherlock_project.database import SherlockDB

_HANDLE_FIELDS = frozenset({"usernames", "aliases"})
_EMAIL_FIELD = "email"


@dataclass(frozen=True, slots=True)
class AccountEvidence:
    """One found account: where it is, and what pass one read off it."""

    site_name: str
    site_url: str | None
    facts: Mapping[str, Sequence[str]]


@dataclass(frozen=True, slots=True)
class LinkReason:
    """One piece of evidence for a suggestion, in words a person can check."""

    # The username whose page says it, and the page.
    username: str
    site_name: str
    text: str


@dataclass(frozen=True, slots=True)
class LinkSuggestion:
    """Two usernames the evidence connects. `first` sorts before `second`."""

    first: str
    second: str
    reasons: tuple[LinkReason, ...]
    # Set when the two already belong to DIFFERENT people. Shown, so the
    # connection is not hidden, but it cannot be accepted as a plain link:
    # merging two people is a bigger decision than this list makes.
    conflict: tuple[str, str] | None = None

    @property
    def pair(self) -> tuple[str, str]:
        return (self.first, self.second)


def suggest_links(
    accounts: Mapping[str, Sequence[AccountEvidence]],
    *,
    person_of: Mapping[str, str] | None = None,
    dismissed: frozenset[tuple[str, str]] = frozenset(),
) -> list[LinkSuggestion]:
    """Every pair of stored usernames the evidence connects.

    `accounts` maps each stored username to its found accounts. `person_of`
    maps a username to the person it is linked to, if any: pairs already in the
    same person are skipped, pairs in two different people are returned with
    `conflict` set. `dismissed` pairs (in sorted order) are skipped.
    """
    person_of = person_of or {}
    usernames = sorted(accounts)
    found: dict[tuple[str, str], list[LinkReason]] = {}

    def add(a: str, b: str, reason: LinkReason) -> None:
        pair = (a, b) if a < b else (b, a)
        reasons = found.setdefault(pair, [])
        if reason not in reasons:
            reasons.append(reason)

    # Normalised profile URL -> the username it was found for, and the URL as
    # stored. Matched on the normalised form, SHOWN as stored: the normalised
    # one drops the scheme and reads like a typo.
    url_owner: dict[str, tuple[str, str]] = {}
    for username in usernames:
        for account in accounts[username]:
            if account.site_url:
                url_owner.setdefault(
                    normalize_value("url", account.site_url),
                    (username, account.site_url),
                )

    emails: dict[str, list[tuple[str, str]]] = {}
    for username in usernames:
        for account in accounts[username]:
            for key, values in account.facts.items():
                field_name = canonical_field(key)
                for value in values:
                    if not isinstance(value, str) or not value.strip():
                        continue
                    if field_name in _HANDLE_FIELDS:
                        for other in usernames:
                            if other != username and is_username_variant(value, other):
                                add(
                                    username,
                                    other,
                                    LinkReason(
                                        username,
                                        account.site_name,
                                        f"{account.site_name} ({username}) lists "
                                        f"{value.strip()} as another handle",
                                    ),
                                )
                    elif field_name == _EMAIL_FIELD:
                        normalised = normalize_value(_EMAIL_FIELD, value)
                        emails.setdefault(normalised, []).append(
                            (username, account.site_name)
                        )
            for target in link_targets(account.facts):
                owner = url_owner.get(target)
                if owner is None or owner[0] == username:
                    continue
                other, shown = owner
                add(
                    username,
                    other,
                    LinkReason(
                        username,
                        account.site_name,
                        f"{account.site_name} ({username}) links to {shown}, "
                        f"found for {other}",
                    ),
                )

    for address, seen in sorted(emails.items()):
        owners = sorted({username for username, _ in seen})
        for index, a in enumerate(owners):
            for b in owners[index + 1 :]:
                site_a = next(site for user, site in seen if user == a)
                site_b = next(site for user, site in seen if user == b)
                add(
                    a,
                    b,
                    LinkReason(
                        a,
                        site_a,
                        f"{site_a} ({a}) and {site_b} ({b}) both show {address}",
                    ),
                )

    suggestions = []
    for pair in sorted(found):
        if pair in dismissed:
            continue
        a, b = pair
        person_a, person_b = person_of.get(a), person_of.get(b)
        if person_a is not None and person_a == person_b:
            continue
        conflict = (
            (person_a, person_b)
            if person_a is not None and person_b is not None
            else None
        )
        suggestions.append(
            LinkSuggestion(a, b, tuple(found[pair]), conflict=conflict)
        )
    return suggestions


async def load_link_suggestions(
    db: SherlockDB,
    *,
    dismissed: frozenset[tuple[str, str]] = frozenset(),
) -> list[LinkSuggestion]:
    """Read what is stored and suggest links from it. Never writes."""
    records = await db.get_all_site_extractions()
    accounts = {
        username: [
            AccountEvidence(
                site_name=record.site_name,
                site_url=record.site_url,
                facts=record.facts,
            )
            for record in rows
        ]
        for username, rows in records.items()
    }
    return suggest_links(
        accounts, person_of=await db.get_person_map(), dismissed=dismissed
    )
