# Coding Standard — Agent Edition

Version: 1.2 | 2026-11-08 | Canonical wording: Coding-standard.md

Standalone compact policy; same rule IDs/levels/scope as human edition. M=MUST when applicable; S=SHOULD (document concrete deviations); O=MAY. M exceptions require rationale, compensating controls, owner, review date. Higher-priority execution instructions apply. Explicit safety/correctness/domain/adopted-boundary rules outrank simplification examples. Conditional topics do not require unused infrastructure. Update both editions together.

Read WORK+CODE for every task; select additional themes below. Headings and rule prefixes support targeted search; no need to load human examples/bibliography for routine execution. Human wording resolves uncertainty. Merge into existing repository instructions without deleting unrelated policy.

## ROUTE

- WORK: every task; ambiguity, planning, scope, verification, delivery
- CODE: names, functions, parameters, types, control flow, classes, duplication
- ARCH: system design, modules, dependencies, services, plugins, migration
- SCALE: growth/capacity targets; codebase/team growth, load balancing, autoscaling, replication/sharding, distributed cache, multi-node CPU/GPU, load tests
- PIPE: core-step files, H1/H2/H3, pipeline, stage contracts, scientific runs
- DATA: persistence, schemas, transactions, concurrency, caches, idempotency, events
- FAIL: exceptions, timeouts, retries, rate limits, resource use, overload, recovery
- TEST: unit/integration/contract/E2E, cases, doubles, determinism, regression, CI
- INPUT: HTTP/CLI/files/config/queues/providers/model output, schemas, parsing, Unicode
- AUTH: authentication, authorization, tenant/object/field scope, sessions, CSRF
- SINK: SQL, HTML/JS/templates, shell, filters, paths, uploads, SSRF
- PRIV: collection, purposes, defaults, sharing, processors, retention, deletion, backups
- KEY: secrets, TLS, encryption, passwords, dependencies, builds, vulnerability response
- LOG: errors, logs, metrics, traces, debugging, audits, security events
- AI: LLMs, agents, prompts, retrieval, tools, model loading, evaluation artifacts
- OPT: performance targets/changes; profiling, loading, cache, CPU threads/SIMD/NUMA, GPU batching/transfers/memory, numerical accuracy
- DOC: ADRs, diagrams, review, versioning, standard maintenance

## WORK

- WORK1 M Define observable success, constraints, material assumptions, and the smallest convincing checks before consequential work.
- WORK2 M Clarify ambiguity changing behavior, APIs, ownership, persistence, security, or irreversible choices; otherwise state a conventional assumption and proceed.
- WORK3 M Keep edits request-related; match local style. Remove only change-created orphans; leave unrelated cleanup and dead code alone.
- WORK4 S Use the smallest readable design preserving real boundaries; justify abstractions by responsibility, repeated knowledge, volatility, or invariants.
- WORK5 M Run applicable existing checks; establish baseline when useful, fix failures, rerun affected checks. Report actual results and unverified behavior honestly.
- WORK6 M Review final diff for excess code, unrelated edits, boundary violations, leaks, and residual risk; treat AI-written code as an untrusted draft.

## CODE

- CODE1 S Use consistent domain vocabulary and intent-revealing names; verbs for operations, predicates for conditions; match name detail to scope.
- CODE2 S Keep functions/modules cohesive and abstraction levels coherent; make side effects explicit; separate queries from mutations.
- CODE3 S Use few meaningful parameters; group cohesive data only; prefer explicit modes over ambiguous behavioral booleans.
- CODE4 S Prefer clear flow and useful guards; reduce nesting/flags; localize repeated dispatch without forcing polymorphism.
- CODE5 S Use behavioral objects for invariants, records for data; avoid mixed ownership and long internal access chains. Deduplicate knowledge, not coincidental text.
- CODE6 S Annotate public parameters/returns and important data; use configured static checking. Annotations are not runtime checks. Boundary validation: INPUT; mutable invariants: DATA/AUTH.
- CODE7 S Follow project/language conventions; name meaningful constants; comment rationale, constraints, or surprises rather than obvious execution.

## ARCH

- ARCH1 M Define capabilities, workload, state, reliability, change drivers, security, ownership, and measurable success before major design; record trade-offs and simpler alternatives.
- ARCH2 S Prioritize 3–5 quality scenarios; prefer a cohesive/modular monolith. Choose deployment, dependency shape, module boundaries, execution, and state independently.
- ARCH3 M Expose deliberate module APIs; keep private methods/tables/caches private; prevent dependency cycles and unauthorized cross-module state mutation.
- ARCH4 S Organize by capability/reason to change; share stable primitives narrowly; use one architecture vocabulary without duplicate conceptual layers.
- ARCH5 S Construct substantial configured infrastructure at the composition root; pass concrete/callable dependencies first; add ports/DI only for meaningful volatility or substitution.
- ARCH6 M Justify services/queues/plugins with present operational needs; specify extension lifecycle/contracts and isolate untrusted plugins. Treat remote calls as failure boundaries.
- ARCH7 S Migrate incrementally through stable seams and behavior checks; retain rollback, retire proven replacements, and avoid combining migrations with unrelated rewrites.
- ARCH8 M Honor adopted dependency boundaries; in clean/hexagonal designs source dependencies point toward stable policy, independent of volatile concrete infrastructure.

## SCALE


### Capacity and project growth

- SCALE1 M For scaling decisions define demand/data growth envelope, SLOs, time horizon, bottlenecks, resource/cost limits, and measured capacity/headroom including required failure scenarios.
- SCALE2 S Scale project delivery through owned cohesive modules, stable contracts, dependency-aware CI/builds and bounded change scope; split services/repos only for demonstrated ownership/deployment/operational benefit.

### Horizontal scaling and overload

- SCALE3 M For horizontal replicas define state/session/job ownership and routing; balance to ready capacity, drain removal, preserve in-flight effects; identify shared bottlenecks and required failure-domain independence.
- SCALE4 M When autoscaling, define workload signals, min/max/cost bounds, warmup/readiness, stabilization and safe scale-in; cap aggregate pools/connections/fan-out against downstream capacity.
- SCALE5 M Apply FAIL/DATA limits across tiers and tenants; define backlog age/recovery, admission/fairness, retry amplification and optional-feature degradation without weakening auth or data invariants.

### Databases and distributed caches

- SCALE6 S Before database distribution, examine query plans/indexes, N+1 access, bounded pagination/batching, pool size, retention and I/O; add replicas/partitions only for measured bottlenecks.
- SCALE7 M For replicas/regions define per-operation consistency, lag/freshness, read-after-write and failover/recovery behavior; route decisions needing current authority to a source that guarantees it.
- SCALE8 M If partitioning/sharding, choose keys from cardinality/access/skew/locality; define hot-key relief, routing/rebalance, cross-shard queries/transactions/uniqueness, migration and recovery.
- SCALE9 M For distributed caches define source of truth, freshness/invalidation, expiry/eviction, concurrent fills and miss/outage load; preserve tenant scope/deletion and protect origin from refill storms.

### Distributed CPU/GPU work

- SCALE10 S For multi-node CPU/GPU work measure useful scaling versus network/collectives/I/O; partition inputs once, bound per-rank CPU/RAM/VRAM, preserve collective/seed/global-batch/checkpoint semantics.

### Verification and safe rollout

- SCALE11 M Validate scale claims with representative ramp/burst/soak, cold-cache and relevant failure/scale-in tests; record offered/admitted/completed work, tails/errors/backlog, bottlenecks, recovery and cost; check generator capacity.
- SCALE12 M For scaling rollouts/resharding define mixed-version schema/event compatibility, bounded backfill/dual-write reconciliation, staged exposure, stop/rollback criteria and post-change capacity verification.

## PIPE

- PIPE1 M For project core workflows: one core step/file, exactly one public `run_<step>()` H1/step, and a separate final orchestrator composing step H1s only.
- PIPE2 M Never import/call another step's private helpers or depend on its private storage; pass cross-step state/contracts explicitly.
- PIPE3 S H2/H3 are optional; H1 may call direct code/libraries/helpers. Order imports/types/constants, needed helpers, then H1 preferably last.
- PIPE4 M Specify stage input/output, failure, artifact ownership, and numerical semantics: shape, units, ordering, dtype/device, score direction where relevant.
- PIPE5 M For scientific reproducibility record code/model/input/config revisions, split, seed, numerical semantics/tolerance; honor privacy limits. Seed alone does not guarantee reproducibility.

## DATA

- DATA1 M Own each entity/artifact and transaction boundary; define persistent/API/event schema evolution and producer–consumer compatibility.
- DATA2 M Enforce mutable invariants atomically using authoritative state, transactions/constraints/conditional writes; prevent check/use races and unintended partial state.
- DATA3 M For consequential retries/delivery, scope durable idempotency/deduplication to actor+operation with a retention window; reject conflicting reuse; prevent duplicate effects.
- DATA4 S Publish artifacts atomically; prefer immutable/versioned results and safe checkpoints; scope caches by permissions plus relevant input/config/model/schema revisions.
- DATA5 M For async jobs/events define identity/version, producer, states, timestamps, ordering/delivery, retention, cancellation, result access/replay; failed/cancelled work cannot expose partial results as success.
- DATA6 S Use outbox/equivalent for atomic DB+publish needs; track lag/dead letters/replay; avoid distributed transactions without genuine distributed ownership needs.
- DATA7 M For cross-service effects define compensable/retryable/irreversible steps and recovery when compensation fails; sagas do not provide global rollback.

## FAIL

- FAIL1 S Propagate errors to a handler able to act; avoid blanket catches, redundant translation, speculative recovery, and success-shaped fallbacks.
- FAIL2 M Set finite remote timeouts/retries and cancellation behavior; retry only safely with bounded backoff/jitter; remote timeout can mean completion is unknown.
- FAIL3 M Bound exposed work before expense: bytes/expansion/depth/items, CPU/time, queries/results, concurrency/queue, disk/output, fan-out/model cost.
- FAIL4 M Use appropriate admission/rate/quota controls, including pre-auth work; reject overload predictably, cap retry amplification, release cancelled work/resources.
- FAIL5 S Measure capacity and recovery needs; distinguish availability from backup/recovery. Add circuit breakers, bulkheads, or tracing only for demonstrated benefit.

## TEST

- TEST1 M Test behavior/invariants at lowest reliable level; unit-test nontrivial domain logic where practical. No tests solely for annotations, trivial forwarding, private call order, or trusted libraries.
- TEST2 M Verify SQL/mappings/constraints/transactions on real DB (prefer production engine family), serialization/API/producer–consumer contracts at real boundaries; never mock integration under test.
- TEST3 M Cover relevant valid/invalid/boundary cases, state transitions, rejection/failure invariants, permission denials, and duplicate effects; use controlled concurrency for race claims.
- TEST4 S Choose partitions, both sides of boundaries, decision/state tables; parameterize equivalent cases. Use properties/fuzzing for clear invariants and broad input spaces.
- TEST5 M Use behavior names and clear arrange/act/assert; derive expectations independently; assert observable state or contractually meaningful interactions.
- TEST6 M Isolate data/resources/accounts; control relevant time/randomness/side effects; no order dependence, arbitrary sleeps, or production credentials/data/services.
- TEST7 S Use real fast collaborators; stub costly/nondeterministic boundaries. Keep fixtures/scenario values explicit; avoid mock chains, artificial interfaces, and unnecessary harnesses.
- TEST8 S For confirmed bugs, reproduce intended failure before fixing; retain regression cases. Characterize affected behavior before substantial insufficiently tested refactors.
- TEST9 M Keep E2E on critical workflows; use stable semantic selectors/readiness, isolated accounts, privacy-safe diagnostics; automate applicable critical workflows.
- TEST10 M Document test commands; run relevant PR/integration/contract gates; fail on failures. Never weaken assertions, silence tests, discard fixed-bug regressions, or retry indefinitely.
- TEST11 S Use configured coverage diagnostically, mutation testing selectively; do not chase percentages. Schedule costly performance/resilience checks separately when appropriate.
- TEST12 M Never mock the subject under test; keep fake behavior consistent with production contracts; verify important integration contracts against real implementations.
- TEST13 M Review uncovered changed critical branches; investigate reproducible flakes. Temporary quarantine requires a tracked issue and explicit resolution criteria.

## INPUT

- INPUT1 M Validate receiving trust-boundary data; trust established internal types. Reassess after new boundaries, meaning-changing transforms, or mutable security-state changes.
- INPUT2 M Specify media/encoding/version, shape/required/null/unknown fields, coercion, ranges/units/cross-field rules, duplicates, cost limits, rejection, and writable fields.
- INPUT3 M Bound bytes during receipt before buffering/parsing; bound decompression and parser work/depth/items separately; Content-Length or post-parse schemas alone are insufficient.
- INPUT4 M Use maintained restricted parsers; no eval/exec, untrusted pickle/executable artifacts, unsafe YAML constructors, external entities, or unnecessary active format features.
- INPUT5 M Define duplicate/repeated-parameter semantics; reject security-relevant ambiguity; decode expected encodings before checks, prevent later reinterpretation, align all parsers/proxies.
- INPUT6 M Enforce finite numeric domain/precision/overflow rules and strict coercion policy; use bounded whole-value checks and safe regex work limits.
- INPUT7 S Preserve legitimate Unicode/punctuation; normalize only by field policy before comparison/uniqueness; reject malformed encoding/invalid controls. No universal sanitizer or client-only enforcement.

## AUTH

- AUTH1 M Deny unless explicit policy permits; cover every protected path and actor/tenant/object/action/field scope; missing/failed policy cannot grant access.
- AUTH2 M Derive principal/tenant membership from verified context; authorize selectors; scope caches/search/jobs/storage/downloads and handle revocation/stale permissions.
- AUTH3 M Verify token signature/authenticity, allowed algorithms/trusted keys, issuer, audience, expiry, and token type with maintained identity implementations.
- AUTH4 M Protect session credentials; Secure/HttpOnly/appropriate SameSite/narrow cookies; rotate on login/privilege changes, expire/invalidate; risk-based step-up/reauthentication.
- AUTH5 M Use appropriate framework CSRF defenses for ambient browser credentials; no GET mutations. CORS is not authorization; preserve TLS, auth, escaping, and CSRF defaults.

## SINK

- SINK1 M Bind SQL values; choose identifiers/sort/query structure from fixed mappings. Build explicit permitted filters; never pass raw request objects as ORM writes/queries.
- SINK2 M Use context-aware escaping/text APIs; keep templates controlled; sanitize allowed rich HTML with maintained tools; restrict active URL schemes. CSP is supplemental.
- SINK3 M Prefer libraries over OS commands; otherwise fixed executable+argv, validate options, tool-supported --, bounded environment/workdir/runtime/output and least privilege.
- SINK4 M Use authorized server-generated storage IDs; contain paths including symlinks/archive links/platform forms. No string-prefix checks; prevent check/open races; use secure temp facilities.
- SINK5 M Validate needed upload formats with intended parsers; distrust MIME/extensions/signatures alone. Bound expansion/entries/dimensions/work; private quarantine, isolated parsing, deliberate safe serving.
- SINK6 M Prefer mapped IDs. For user-influenced URLs enforce scheme/port/destination on actual IPv4/IPv6 connections incl DNS/retries/redirects; no forbidden networks/ambient credentials; bound responses, egress controls as appropriate.

## PRIV

- PRIV1 M Before personal-data flows define purpose/minimum fields+precision, recipients/access, defaults/choices, copies/derivatives, retention/deletion, owner, verification; communicate collection/recipients/controls at use.
- PRIV2 M Default optional tracking/sharing/public exposure to privacy-protective settings; enforce choices server-side and downstream; assess processors/SDK telemetry and new purposes.
- PRIV3 S Prefer noncollection, coarse data, local processing, separated identity, or aggregation when sufficient; anonymity claims are constrained by PRIV7.
- PRIV4 M Implement owned expiry/deletion/correction across applicable originals, derivatives, processors; track partial failures; prevent delayed jobs/replay from recreating deleted data.
- PRIV5 M Define backup exceptions/expiry/access; reapply deletion before restored availability. Authorize private exports/deletion, limit export lifetime, explicitly select response fields.
- PRIV6 M Do not retain raw personal data indefinitely for debugging/reproducibility; prefer synthetic/minimal evidence. Match privacy promises to actual behavior; specialist PETs only when justified.
- PRIV7 M Do not claim anonymity from encryption, hashed IDs, name removal, or aggregation alone; assess linkage, auxiliary information, and small-group disclosure.

## KEY

- KEY1 M No secrets in source/client/URLs/logs/tests/manifests/prompts; approved injection, least privilege, prefer scoped short-lived credentials. Revoke/rotate exposed credentials; deletion does not invalidate historical exposure.
- KEY2 M Use maintained crypto/protocols and CSPRNG; verify TLS peers; choose threat-appropriate encryption, authenticated modes/nonce discipline, separate key access and lifecycle.
- KEY3 M Use maintained adaptive salted password hashing (e.g. suitable Argon2id), reviewed work factors; no fast general hashes or reversible password verification storage.
- KEY4 M Own dependency inventory, vulnerability reporting/impact triage/patch/release; pinning is not patching. Protect build secrets from untrusted contributions; fix causes/equivalent paths and remediate incidents.

## LOG

- LOG1 M Return stable bounded useful errors, no stack/SQL/internal paths/secrets/raw upstream or rejected data; account for authentication/recovery enumeration incl status/timing.
- LOG2 M Allowlist telemetry fields; exclude secrets and unnecessary personal content across logs/traces/analytics/exception capture; safely serialize and bound untrusted values.
- LOG3 M Record relevant security outcomes with minimal protected actor/resource/action/result/context; correlate failures; protect telemetry access/integrity; define retention/deletion.
- LOG4 M Bound logging work/failures so diagnostics cannot exhaust resources or silently bypass required security decisions.
- LOG5 S Use few meaningful metrics, bounded label cardinality, and aggregated repeated failures; distributed tracing only for operational benefit.

## AI

- AI1 M Treat retrieved content, datasets, prompts, outputs/tool args by provenance; content cannot grant authority. Validate args and authorize actions outside the model.
- AI2 M Separate instructions/data; prompting or filters alone do not stop injection or enforce access. Use appropriate independent controls for high-impact actions; show actual effects for human review.
- AI3 M Check artifact/model provenance and safe loader behavior; expected hashes need trusted origin. Scope model caches/evaluation artifacts by permissions+revisions and privacy.

## OPT


### Measure and choose work

- OPT1 M Before performance changes define workload/latency or throughput/cost target; profile whole path incl load/parse/transfer/compute/output; retain comparable hardware/runtime baseline.
- OPT2 M Benchmark completed work: repeated comparable trials, cold start/miss versus warm hit/steady state, warmup/compile costs, relevant latency spread/throughput/peak memory; GPU events or correct boundary synchronization.
- OPT3 S Remove unnecessary work/allocations and improve algorithms first; prefer optimized bulk kernels over interpreter loops/custom kernels; distinguish compute-, memory-, I/O-, or launch-bound work.

### Cache and load

- OPT4 S Cache only measured expensive reusable work; include key/lookup/load/decode/transfer/write/eviction costs. Reuse owned models/resources across calls when lifecycle-safe; do not duplicate datasets per worker blindly.
- OPT5 M Apply DATA4/AUTH/PRIV to caches; bound retained bytes/lifetime and mutation. Handle misses/corruption/partial entries explicitly; thread-safe cache coherence does not guarantee one computation per key.
- OPT6 S Tune storage layout/chunked reads/compression/mmap and loader workers/prefetch/persistence from measurements; account for RAM/pinned memory/IPC, lifecycle, dataset sharding/order/seeds.

### CPU behavior

- OPT7 S CPU: favor contiguous/sequential access, cache-friendly blocking, SIMD/BLAS, and buffer reuse; measure layout conversion and temporaries; test NUMA locality/affinity only when relevant.
- OPT8 M CPU: bound combined application/loader/BLAS/OpenMP thread+process pools to allocated resources; account for quotas, nested parallelism, IPC, replicated memory; document tuned settings.
- OPT9 S CPU: choose threads for I/O or native work releasing GIL; for GIL-bound Python consider coarse processes/native kernels. Check actual interpreter build, sharing races, and process-start overhead.

### GPU behavior

- OPT10 S GPU: offload only when end-to-end wins; batch/vectorize within latency+VRAM budgets, keep reusable data resident, avoid tiny launches/round trips; benchmark fusion/compile incl cold/recompile costs.
- OPT11 M GPU: budget live tensors/activations/workspaces/allocator-reserved memory and concurrent work; monitor allocated vs reserved peaks. Releasing allocator cache does not free live tensors.
- OPT12 S GPU: reduce/batch transfers first; try bounded pinned host buffers+nonblocking copies/stream overlap only when supported and measured. Avoid routine item()/CPU reads or allocator-cache flushing in hot loops.
- OPT13 M GPU: preserve stream/event dependencies and buffer lifetime; no read/reuse/mutation before async completion. Use runtime-supported multiprocessing; never fork an initialized accelerator runtime.

### Inference and numerical correctness

- OPT14 S CPU/GPU ML: use eval plus appropriate no-grad/inference mode only when gradients are unnecessary; benchmark autocast/precision/layout/compile options on supported hardware, not blanket half conversion.
- OPT15 M Validate optimizations against reference outputs/domain tolerances, finite/edge cases, gradient/training metrics as applicable; report numerical/determinism changes and actual end-to-end gains, memory/latency regressions.

## DOC

- DOC1 S Record consequential decisions with context/options/choice/consequences/status/revisit trigger; supersede history. Diagrams only when useful, current, and clear about calls/data/dependencies/trust.
- DOC2 S Automate enduring high-risk boundaries with existing checks when worthwhile; review simple conventions directly; no mandatory custom AST/fitness framework.
- DOC3 M Maintain matching human/agent version, IDs, levels, conditions/exceptions; update both for policy changes. Examples/menus never override numbered rules.

