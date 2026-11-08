"""People: the storage, the per-page username filter, the cache, and suggestions.

A person is a named group of stored usernames with one profile built across
all of them. The properties pinned here are the ones a wrong answer would hide:
a username belongs to at most one person, deleting a person keeps every
username, a person's NAME is never used to filter facts, and username profiles
hash exactly as they did before people existed.
"""

from __future__ import annotations

import json

import pytest

from sherlock_project.ai_engine import AIService
from sherlock_project.database import PersonError, SherlockDB
from sherlock_project.person_links import AccountEvidence, suggest_links
from sherlock_project.profile_synthesis import (
    InvestigationContext,
    ProfileSynthesis,
    SiteExtraction,
    SourceDecision,
    SynthesisEvidence,
    aggregate_synthesis,
    compute_synthesis_input_hash,
)
from sherlock_project.result import QueryStatus
from sherlock_project.synthesis_pipeline import (
    synthesize_person_profile,
    synthesize_username_profile,
)

CONTRACT = "pass-one-contract-v2"


async def _claimed(
    db: SherlockDB, username: str, site: str, facts: dict | None = None
) -> int:
    site_id = await db.save_result(
        username=username,
        site_name=site,
        site_url=f"https://{site.lower()}.example/{username}",
        status=str(QueryStatus.CLAIMED),
        response_text="<html>page</html>",
    )
    if facts is not None:
        await db.update_result_ai_extraction(
            site_id, json.dumps(facts), contract_hash=CONTRACT, model_key="m"
        )
    return site_id


async def _hash_of(db: SherlockDB, name: str) -> str | None:
    cache = await db.get_person_profile_cache(name)
    return cache.input_hash if cache else None


# -- storage --------------------------------------------------------------


async def test_create_link_and_list_a_person(db: SherlockDB):
    await _claimed(db, "alice", "GitHub")
    await _claimed(db, "alice_dev", "GitLab")
    await db.create_person("Alice Smith")
    await db.link_username("Alice Smith", "alice")
    await db.link_username("Alice Smith", "alice_dev")

    [person] = await db.list_people()
    assert person.name == "Alice Smith"
    assert person.usernames == ("alice", "alice_dev")
    assert person.claimed_sites == 2
    assert person.has_profile is False
    assert await db.get_person_of("alice") == "Alice Smith"
    assert await db.get_person_map() == {
        "alice": "Alice Smith",
        "alice_dev": "Alice Smith",
    }


async def test_names_are_trimmed_and_must_be_unique(db: SherlockDB):
    await db.create_person("  Alice   Smith ")
    assert [p.name for p in await db.list_people()] == ["Alice Smith"]
    with pytest.raises(PersonError):
        await db.create_person("Alice Smith")
    with pytest.raises(PersonError):
        await db.create_person("   ")


async def test_a_username_belongs_to_at_most_one_person(db: SherlockDB):
    await _claimed(db, "alice", "GitHub")
    await db.create_person("A")
    await db.create_person("B")
    await db.link_username("A", "alice")
    # Linking again to the same person is a no-op, not an error.
    await db.link_username("A", "alice")
    with pytest.raises(PersonError, match="already belongs to 'A'"):
        await db.link_username("B", "alice")


async def test_linking_needs_a_stored_username_and_a_person(db: SherlockDB):
    await db.create_person("A")
    with pytest.raises(PersonError, match="nothing is stored"):
        await db.link_username("A", "ghost")
    await _claimed(db, "alice", "GitHub")
    with pytest.raises(PersonError, match="no person"):
        await db.link_username("Nobody", "alice")


async def test_deleting_a_person_keeps_every_username_and_result(db: SherlockDB):
    await _claimed(db, "alice", "GitHub")
    await db.create_person("A")
    await db.link_username("A", "alice")
    await db.update_person_profile_summary("A", "{}", "hash")

    assert await db.delete_person("A") == ["alice"]
    assert await db.list_people() == []
    assert [item.username for item in await db.list_usernames()] == ["alice"]
    assert await db.get_person_of("alice") is None


async def test_deleting_a_username_removes_its_link(db: SherlockDB):
    await _claimed(db, "alice", "GitHub")
    await _claimed(db, "alice_dev", "GitHub")
    await db.create_person("A")
    await db.link_username("A", "alice")
    await db.link_username("A", "alice_dev")
    await db.update_person_profile_summary("A", "{}", "hash")

    await db.delete_username("alice")
    assert await db.get_person_usernames("A") == ["alice_dev"]
    # The profile counted alice's pages, so it is out of date now.
    assert await _hash_of(db, "A") is None


async def test_an_extraction_write_invalidates_the_persons_profile(db: SherlockDB):
    site_id = await _claimed(db, "alice", "GitHub", {"roles": ["x"]})
    await db.create_person("A")
    await db.link_username("A", "alice")
    await db.update_person_profile_summary("A", "{}", "hash")

    await db.update_result_ai_extraction(
        site_id, json.dumps({"roles": ["y"]}), contract_hash=CONTRACT, model_key="m"
    )
    assert await _hash_of(db, "A") is None


async def test_link_unlink_and_rename_each_invalidate(db: SherlockDB):
    await _claimed(db, "alice", "GitHub")
    await db.create_person("A")
    for change in (
        lambda: db.link_username("A", "alice"),
        lambda: db.unlink_username("A", "alice"),
    ):
        await db.update_person_profile_summary("A", "{}", "hash")
        await change()
        assert await _hash_of(db, "A") is None
    await db.update_person_profile_summary("A", "{}", "hash")
    await db.rename_person("A", "Alice")
    assert await _hash_of(db, "Alice") is None
    assert await db.unlink_username("Alice", "alice") is False


async def test_person_evidence_carries_each_rows_username(db: SherlockDB):
    await _claimed(db, "alice", "GitHub", {"roles": ["x"]})
    await _claimed(db, "alice_dev", "GitHub", {"roles": ["y"]})
    await _claimed(db, "bob", "GitHub", {"roles": ["z"]})
    await db.create_person("A")
    await db.link_username("A", "alice")
    await db.link_username("A", "alice_dev")

    records = await db.get_person_profile_evidence("A")
    assert [(r.site_name, r.username) for r in records] == [
        ("GitHub", "alice"),
        ("GitHub", "alice_dev"),
    ]
    # A username's own evidence is unchanged: no username attached.
    assert all(r.username is None for r in await db.get_ai_profile_evidence("alice"))


# -- synthesis --------------------------------------------------------------


def _page(site_id: int, facts: dict, username: str | None = None) -> SiteExtraction:
    return SiteExtraction(
        site_id=site_id,
        site_name=f"site-{site_id}",
        extraction=facts,
        username=username,
    )


def test_each_page_filters_its_own_username_not_the_persons_name():
    profile = aggregate_synthesis(
        username="Alice Smith",
        input_hash="h",
        context=InvestigationContext(),
        evidence=SynthesisEvidence(
            extractions=[
                _page(1, {"full_name": ["Alice Smith"], "usernames": ["alice"]}, "alice"),
                _page(2, {"usernames": ["alice", "alice_dev"]}, "alice_dev"),
            ]
        ),
    )
    # The person's name is a fact, not a handle to drop.
    assert profile.strong_profile["full_name"] == ["Alice Smith"]
    # Page 1 drops "alice" (its own handle); page 2 keeps it as evidence and
    # drops "alice_dev" (its own).
    assert profile.strong_profile["usernames"] == ["alice"]


def test_username_profiles_hash_and_serialise_as_before():
    """Adding people must not invalidate any username profile already stored."""
    evidence = SynthesisEvidence(extractions=[_page(1, {"roles": ["x"]})])
    arguments = {
        "username": "alice",
        "model_key": "m",
        "context": InvestigationContext(),
        "prompt_fingerprints": {},
    }
    # The value this exact input hashed to before people existed (computed on
    # the commit before them). If this changes, every username profile on
    # disk is silently rebuilt on its next build.
    assert compute_synthesis_input_hash(evidence=evidence, **arguments) == (
        "f17f64213c27e4b74a7366e40c23649616eb932189f1974a100662bb74998967"
    )
    tagged = SynthesisEvidence(extractions=[_page(1, {"roles": ["x"]}, "alice")])
    assert compute_synthesis_input_hash(
        evidence=tagged, **arguments
    ) != compute_synthesis_input_hash(evidence=evidence, **arguments)

    profile = aggregate_synthesis(
        username="alice",
        input_hash="h",
        context=InvestigationContext(),
        evidence=evidence,
    )
    dumped = profile.model_dump(mode="json")
    assert "person" not in dumped and "usernames" not in dumped
    assert all("username" not in decision for decision in dumped["source_decisions"])


def test_person_fields_round_trip_when_set():
    profile = ProfileSynthesis(
        username="A",
        input_hash="h",
        mode="aggregate",
        resolution_status="aggregated",
        completeness="complete",
        person="A",
        usernames=["alice"],
        source_decisions=[
            SourceDecision(
                site_id=1, site_name="GitHub", disposition="aggregated", username="alice"
            )
        ],
    )
    again = ProfileSynthesis.model_validate_json(profile.model_dump_json())
    assert again.person == "A"
    assert again.usernames == ["alice"]
    assert again.source_decisions[0].username == "alice"


class _Service:
    """A real AIService for the deterministic path, with a fixed contract."""

    def __init__(self) -> None:
        self._inner = AIService()
        self.model_key = "m"
        self.pass_one_contract_hash = CONTRACT
        self.synthesis_prompt_fingerprints: dict[str, str] = {}
        self.calls = 0

    async def synthesize(self, **kwargs):
        self.calls += 1
        return await self._inner.synthesize(**kwargs)


async def test_person_profile_is_built_cached_and_stored_separately(db: SherlockDB):
    await _claimed(db, "alice", "GitHub", {"full_name": ["Alice Smith"]})
    await _claimed(db, "alice_dev", "GitLab", {"roles": ["Engineer"]})
    await db.create_person("Alice Smith")
    await db.link_username("Alice Smith", "alice")
    await db.link_username("Alice Smith", "alice_dev")
    service = _Service()

    first = await synthesize_person_profile(
        db=db, ai_service=service, person="Alice Smith", context=InvestigationContext()
    )
    assert first.cache_hit is False
    assert first.profile.person == "Alice Smith"
    assert first.profile.usernames == ["alice", "alice_dev"]
    assert first.profile.strong_profile == {
        "full_name": ["Alice Smith"],
        "roles": ["Engineer"],
    }
    assert {d.username for d in first.profile.source_decisions} == {"alice", "alice_dev"}

    second = await synthesize_person_profile(
        db=db, ai_service=service, person="Alice Smith", context=InvestigationContext()
    )
    assert second.cache_hit is True
    assert service.calls == 1

    # The usernames' own profiles were not touched.
    assert (await db.get_profile_summary_cache("alice")).profile_summary is None

    # Unlinking changes what the person describes: no cache hit.
    await db.unlink_username("Alice Smith", "alice_dev")
    third = await synthesize_person_profile(
        db=db, ai_service=service, person="Alice Smith", context=InvestigationContext()
    )
    assert third.cache_hit is False
    assert third.profile.strong_profile == {"full_name": ["Alice Smith"]}


async def test_a_person_and_a_username_of_the_same_name_hash_apart(db: SherlockDB):
    await _claimed(db, "alice", "GitHub", {"roles": ["x"]})
    await db.create_person("alice")
    service = _Service()
    username = await synthesize_username_profile(
        db=db, ai_service=service, username="alice", context=InvestigationContext()
    )
    person = await synthesize_person_profile(
        db=db, ai_service=service, person="alice", context=InvestigationContext()
    )
    assert username.profile.input_hash != person.profile.input_hash


async def test_building_an_unknown_person_says_so(db: SherlockDB):
    with pytest.raises(PersonError):
        await synthesize_person_profile(
            db=db, ai_service=_Service(), person="Nobody", context=InvestigationContext()
        )


# -- suggestions ------------------------------------------------------------


def _account(site: str, profile_url: str | None, **facts) -> AccountEvidence:
    return AccountEvidence(site_name=site, site_url=profile_url, facts=facts)


def test_a_listed_handle_suggests_a_link():
    [suggestion] = suggest_links(
        {
            "alice": [_account("GitHub", "https://github.com/alice")],
            "alice_dev": [
                _account("Twitter", "https://x.com/alice_dev", other_usernames=["@Alice"])
            ],
        }
    )
    assert suggestion.pair == ("alice", "alice_dev")
    assert "lists @Alice as another handle" in suggestion.reasons[0].text


def test_a_link_to_another_usernames_profile_suggests_a_link():
    [suggestion] = suggest_links(
        {
            "alice": [
                _account(
                    "GitHub", "https://github.com/alice", url=["https://x.com/alice_dev/"]
                )
            ],
            "alice_dev": [_account("X", "https://x.com/alice_dev")],
        }
    )
    # Shown as stored, not in its normalised form.
    assert "links to https://x.com/alice_dev, found for alice_dev" in (
        suggestion.reasons[0].text
    )


def test_a_shared_email_suggests_a_link():
    [suggestion] = suggest_links(
        {
            "a": [_account("GitLab", None, email=["Me@Example.com"])],
            "b": [_account("Reddit", None, email=["me@example.com"])],
        }
    )
    assert suggestion.pair == ("a", "b")
    assert "both show me@example.com" in suggestion.reasons[0].text


def test_shared_names_and_places_suggest_nothing():
    """Thousands of people are called Alex and live in Berlin."""
    assert (
        suggest_links(
            {
                "a": [_account("GitHub", None, full_name=["Alex"], location=["Berlin"])],
                "b": [_account("GitLab", None, full_name=["Alex"], location=["Berlin"])],
            }
        )
        == []
    )


def test_pairs_in_one_person_are_skipped_and_two_people_are_flagged():
    accounts = {
        "a": [_account("GitLab", None, email=["x@y.z"])],
        "b": [_account("Reddit", None, email=["x@y.z"])],
    }
    assert suggest_links(accounts, person_of={"a": "P", "b": "P"}) == []
    [conflict] = suggest_links(accounts, person_of={"a": "P", "b": "Q"})
    assert conflict.conflict == ("P", "Q")
    assert suggest_links(accounts, dismissed=frozenset({("a", "b")})) == []


def test_an_anchor_equal_to_the_persons_name_still_matches():
    """The deterministic round must not treat the person's name as a handle."""
    from sherlock_project.global_synthesis import deterministic_round, site_refs
    from sherlock_project.profile_synthesis import IdentityAnchor

    sites = site_refs(
        [
            _page(1, {"full_name": ["Alice Smith"]}, "alice"),
            _page(2, {"roles": ["Engineer"]}, "alice_dev"),
        ]
    )
    verdicts: dict = {}
    deterministic_round(
        username="Alice Smith",
        context=InvestigationContext(
            anchors=[IdentityAnchor(field="full_name", value="Alice Smith")]
        ),
        sites=sites,
        verdicts=verdicts,
    )
    matched = {sites[ref].site_id for ref, v in verdicts.items() if v.status == "strong_match"}
    assert matched == {1}


def test_an_anchor_equal_to_a_linked_handle_is_not_evidence():
    """A page found UNDER the anchored handle trivially carries it; no match."""
    from sherlock_project.global_synthesis import deterministic_round, site_refs
    from sherlock_project.profile_synthesis import IdentityAnchor

    sites = site_refs(
        [
            _page(1, {"usernames": ["alice"]}, "alice"),
            _page(2, {"roles": ["Engineer"]}, "alice_dev"),
        ]
    )
    verdicts: dict = {}
    deterministic_round(
        username="Alice Smith",
        context=InvestigationContext(
            anchors=[IdentityAnchor(field="usernames", value="alice")]
        ),
        sites=sites,
        verdicts=verdicts,
    )
    assert verdicts == {}
