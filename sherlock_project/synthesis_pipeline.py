from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from hashlib import sha256

from pydantic import ValidationError

from sherlock_project.ai_engine import (
    AIService,
    validate_pass_one_extraction_payload,
)
from sherlock_project.database import (
    AIProfileEvidenceRecord,
    PersonError,
    ProfileSummaryCache,
    SherlockDB,
)
from sherlock_project.profile_synthesis import (
    InvalidExtraction,
    InvestigationContext,
    ProfileSynthesis,
    SiteExtraction,
    SynthesisEvidence,
    compute_synthesis_input_hash,
)


@dataclass(frozen=True, slots=True)
class SynthesisRunResult:
    profile: ProfileSynthesis
    cache_hit: bool


async def synthesize_username_profile(
    *,
    db: SherlockDB,
    ai_service: AIService,
    username: str,
    context: InvestigationContext,
    force: bool = False,
) -> SynthesisRunResult:
    return await _synthesize(
        ai_service=ai_service,
        label=username,
        records=await db.get_ai_profile_evidence(username),
        context=context,
        force=force,
        load_cache=lambda: db.get_profile_summary_cache(username),
        invalidate=lambda: db.invalidate_username_profile_summary(username),
        store=lambda summary, input_hash: db.update_username_profile_summary(
            username=username,
            profile_summary=summary,
            input_hash=input_hash,
        ),
    )


async def synthesize_person_profile(
    *,
    db: SherlockDB,
    ai_service: AIService,
    person: str,
    context: InvestigationContext,
    force: bool = False,
) -> SynthesisRunResult:
    """One profile across every username linked to `person`.

    The same pass two as a username profile, over the union of the linked
    usernames' pages, with each page tagged by the username it was found
    under. Raises `PersonError` when there is no such person.
    """
    usernames = await db.get_person_usernames(person)
    if usernames is None:
        raise PersonError(f"no person called {person!r}")
    result = await _synthesize(
        ai_service=ai_service,
        label=person,
        records=await db.get_person_profile_evidence(person),
        context=context,
        force=force,
        load_cache=lambda: db.get_person_profile_cache(person),
        invalidate=lambda: db.invalidate_person_profile_summary(person),
        store=lambda summary, input_hash: db.update_person_profile_summary(
            person, summary, input_hash
        ),
        person=person,
        usernames=usernames,
    )
    return result


async def _synthesize(
    *,
    ai_service: AIService,
    label: str,
    records: list[AIProfileEvidenceRecord],
    context: InvestigationContext,
    force: bool,
    load_cache: Callable[[], Awaitable[ProfileSummaryCache | None]],
    invalidate: Callable[[], Awaitable[None]],
    store: Callable[[str, str], Awaitable[None]],
    person: str | None = None,
    usernames: list[str] | None = None,
) -> SynthesisRunResult:
    """Evidence -> input hash -> cache gate -> pass two -> store.

    One body for username and person profiles, so the cache rules (a cached
    profile with a failed decision is never reused) cannot drift between them.
    """
    current_contract_hash = ai_service.pass_one_contract_hash
    extractions: list[SiteExtraction] = []
    pending_site_ids: list[int] = []
    invalid_extractions: list[InvalidExtraction] = []
    owners: dict[int, str] = {}

    for record in records:
        if record.username is not None:
            owners[record.site_id] = record.username
        if (
            record.ai_extraction is None
            or record.ai_extraction_contract_hash != current_contract_hash
        ):
            pending_site_ids.append(record.site_id)
            continue
        try:
            parsed = json.loads(record.ai_extraction)
            validated = validate_pass_one_extraction_payload(parsed)
        except (json.JSONDecodeError, ValidationError, ValueError) as error:
            invalid_extractions.append(
                InvalidExtraction(
                    site_id=record.site_id,
                    site_name=record.site_name,
                    payload_hash=sha256(
                        record.ai_extraction.encode("utf-8")
                    ).hexdigest(),
                    error=str(error),
                )
            )
            continue

        extractions.append(
            SiteExtraction(
                site_id=record.site_id,
                site_name=record.site_name,
                site_url=record.site_url,
                scanned_at=record.scanned_at,
                extraction=validated,
                username=record.username,
            )
        )

    evidence = SynthesisEvidence(
        extractions=extractions,
        pending_site_ids=sorted(pending_site_ids),
        invalid_extractions=invalid_extractions,
    )
    input_hash = compute_synthesis_input_hash(
        username=label,
        model_key=ai_service.model_key,
        context=context,
        evidence=evidence,
        prompt_fingerprints=ai_service.synthesis_prompt_fingerprints,
    )
    if person is not None:
        # A person with no analysed pages yet would otherwise hash exactly like
        # a USERNAME of the same name, and the linked set is part of what the
        # profile describes even before any of them has evidence.
        input_hash = sha256(
            json.dumps(
                {"person": person, "usernames": sorted(usernames or []), "base": input_hash},
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    if not force:
        cached = await load_cache()
        if (
            cached is not None
            and cached.input_hash == input_hash
            and cached.profile_summary is not None
        ):
            try:
                profile = ProfileSynthesis.model_validate_json(cached.profile_summary)
            except ValidationError:
                pass
            else:
                has_failed_decisions = any(
                    decision.disposition == "failed"
                    for decision in profile.source_decisions
                )
                if profile.input_hash == input_hash and not has_failed_decisions:
                    return SynthesisRunResult(profile=profile, cache_hit=True)

    if force:
        await invalidate()

    profile = await ai_service.synthesize(
        username=label,
        extractions=extractions,
        context=context,
        input_hash=input_hash,
        pending_site_ids=pending_site_ids,
        invalid_extractions=invalid_extractions,
    )
    if person is not None:
        profile.person = person
        profile.usernames = list(usernames or [])
        for decision in profile.source_decisions:
            decision.username = owners.get(decision.site_id)
    serialized = json.dumps(
        profile.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
    )
    await store(serialized, input_hash)
    return SynthesisRunResult(profile=profile, cache_hit=False)
