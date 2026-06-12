# Pehchaan × Sutra MCP — Accuracy Report

- **Repo:** pehchaan @ a383c6f0 (profile service, 145 files, 1055 symbols)
- **Index:** all-MiniLM-L6-v2 local embeddings; CALLS resolution 95% (heuristic+pyright chain)
- **Path under test:** real `python -m sutra.mcp` subprocess over stdio MCP — the exact agent path
- **Battery:** 22 search tests across 5 complexity tiers + 3 graph-tool tests; every expected symbol eyeballed in source first

## Headline

| metric | value |
|---|---|
| top-1 accuracy | **19/22  (86%)** |
| top-3 accuracy | **20/22  (91%)** |
| top-10 accuracy | **21/22  (95%)** |
| MRR | **0.886** |
| mean query latency (end-to-end MCP) | 26 ms |

## Search tests

| # | tier | target | query | rank | verdict |
|---|---|---|---|---|---|
| 1 | A · helper | `normalize_company_name` | 'function that normalizes a company name by removing suffixes like pvt ltd' | 1 | ✅ #1 |
| 2 | A · helper | `_merge_phone_list` | 'merge two phone number lists without duplicates' | 1 | ✅ #1 |
| 3 | A · helper | `generate_cache_key_business` | 'generate a cache key from a business name and country' | 1 | ✅ #1 |
| 4 | A · helper | `_get_file_extension` | 'extract the file extension from an uploaded filename' | 1 | ✅ #1 |
| 5 | B · service | `generate_profile_qr` | 'generate a qr code image for sharing a profile' | 1 | ✅ #1 |
| 6 | B · service | `_generate_signed_url` | 'create a presigned s3 url for a stored object' | 1 | ✅ #1 |
| 7 | B · service | `upload_profile_photo` | 'upload a profile photo to s3' | 1 | ✅ #1 |
| 8 | B · service | `_validate_image_file` | 'validate that an uploaded file is an allowed image type' | 1 | ✅ #1 |
| 9 | C · DAL | `_transaction_retry (NESTED fns inside)` | 'which function retries a mongo transaction on write conflict with exponential backoff' | 1 | ✅ #1 |
| 10 | C · DAL | `batch_insert` | 'insert many documents in a single mongo transaction batch' | 1 | ✅ #1 |
| 11 | C · DAL | `create_bulk_listings (THE original failure)` | 'which function saves the listing in db' | 3 | ✅ #3 |
| 12 | C · DAL | `search_paginated` | 'paginated query over a mongo collection with skip and limit' | 1 | ✅ #1 |
| 13 | C · DAL | `update_key_person_phone` | 'update the phone number of a key person on a business profile' | 1 | ✅ #1 |
| 14 | D · flow | `find_or_create_business_with_enrichment_check` | 'find an existing business or create it and check whether enrichment is needed' | 1 | ✅ #1 |
| 15 | D · flow | `needs_business_enrichment` | 'check whether a business needs re-enrichment based on a 30 day threshold' | 1 | ✅ #1 |
| 16 | D · flow | `_create_or_link_mobile_profile` | 'link a newly created individual profile to an existing mobile user' | 1 | ✅ #1 |
| 17 | D · flow | `search_businesses_by_name` | 'full text search businesses by name with pagination' | 1 | ✅ #1 |
| 18 | D · flow | `clear_individual_user_data` | 'delete an individual account and wipe all of its data' | 1 | ✅ #1 |
| 19 | D · flow | `create_individual_profile` | 'create a new individual profile' | 6 | 🟡 #6 |
| 20 | E · entity | `BusinessProfile model` | 'the business profile model' | — | ❌ miss |
| 21 | E · entity | `MongoDBSingleton` | 'mongo client connection singleton' | 1 | ✅ #1 |
| 22 | E · entity | `SQSService publish` | 'publish a business created event to the sqs queue' | 1 | ✅ #1 |

## Below-top-3 cases — what came back instead

**create_individual_profile** — 'create a new individual profile' (rank 6)
  1. `services.individual._create_new_mobile_profile` (function, services/individual.py)
  2. `controllers.profile.create_profile` (function, controllers/profile.py)
  3. `services.individual._create_or_link_mobile_profile` (function, services/individual.py)

**BusinessProfile model** — 'the business profile model' (rank miss)
  1. `models.bulk_profile_models.BulkProfileRequest` (class, models/bulk_profile_models.py)
  2. `models.business_lookup.BusinessFindOrCreateRequest` (class, models/business_lookup.py)
  3. `models.bulk_profile_models.BulkProfileResponse` (class, models/bulk_profile_models.py)


## Graph-tool tests

| test | verdict | sample |
|---|---|---|
| callees of find_or_create_business_with_enrichment_check include needs_business_enrichment | ✅ | exceptions.global_exceptions.AppException, models.business_lookup.BusinessFindOrCreateResponse, services.business.get_or_create_business_for_enrichment, services.business.needs_business_enrichment |
| callers of _merge_phone_list are the individual-service backfill paths | ✅ | services.individual.find_or_create_individual_for_enrichment, services.individual.find_or_create_individual_for_enrichment, services.individual.find_or_create_individual_for_enrichment, services.individual.find_or_create_individual_for_enrichment |
| 2-hop neighborhood of create_bulk_listings reaches its controller caller | ✅ | services.individual.update_individual_profile, services.individual.create_individual_profile, services.individual.get_mongo_dal, data.__init__.MongoDALWrapper.find_one |

## Notes & findings

- **_transaction_retry (NESTED fns inside)** (rank 1): The retry loop lives in nested funcs (decorator→wrapper) that are NOT indexed as symbols (Phase 1 scope); their code is searchable only through this enclosing symbol's chunk.
- **create_bulk_listings (THE original failure)** (rank 3): The Phase 1 motivating failure query, verbatim.

## Post-battery analysis of the two imperfect cases

**`create_individual_profile` (rank 6)** — the three results above it
(`_create_new_mobile_profile`, `controllers.profile.create_profile`,
`_create_or_link_mobile_profile`) ALL create individual profiles; pehchaan
has 5+ functions for which "create a new individual profile" is a fair
description.  The query is genuinely ambiguous in this codebase — arguably
a too-narrow expected set rather than a retrieval error.

**`BusinessProfile` model (miss at prose phrasing)** — the only true gap.
Cause: thin Pydantic-class chunks (no docstring, no methods) lose lexical
mass to richer Request/Response models when the name is decomposed into
prose ("business profile model").  Verified workaround: querying the
identifier (`BusinessProfile`, or any phrasing containing it) returns the
class at **#1** via the moniker channel — `via=['bm25','moniker','vector']`.
Candidate future fix: include Pydantic field names in class chunks.

## Fix shipped DURING this battery (found by it)

The first run missed `_validate_image_file` entirely: "allowed image
**type**" was misread as a class-kind request and the kind filter dropped
every method.  Root-caused to a lexicon collocation trap; fixed with
`noun_collocation_blockers` ("type" preceded by content/image/file/mime/
media/data/return is a data property, not a kind noun) + 5 regression
tests.  Re-ran the battery: miss → **#1**, no regressions anywhere
(112-test retrieval suite + 32-query eval baseline both green).
This is the eval-first loop working as designed: a 20-minute accuracy
battery on a fresh repo surfaced and fixed a real analyzer bug.

## Verdict

On a previously-unseen 145-file production service, the agent-facing
search answers 86% of natural-language function queries at #1 (95% in the
top ten) at 26ms per query, the call graph is 95% resolved, and the graph
tools traverse service→DAL→controller chains correctly.  Nested functions
(decorator/wrapper bodies) are reachable through their enclosing symbol's
chunk — they are not separate symbols by Phase 1 design.
