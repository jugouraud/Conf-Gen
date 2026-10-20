# Clean Code — Project Guidelines

> Condensed from the original clean-code guide without removing distinct guidelines. Inspired by Robert C. Martin's *Clean Code* and [Codacy's guide](https://blog.codacy.com/what-is-clean-code). **Section 2 contains mandatory project architecture rules**; the general principles are design heuristics, not rigid metrics.

## 1. Core Clean-Code Principles

### Naming and readability

- **Reveal intent.** Name variables, functions, classes, modules, and files for their domain purpose, not an incidental implementation detail. Prefer `pending_orders`, `billing_address`, and `calculate_invoice_total()` over `data2`, `account_info`, or `do_stuff()`; distinctions between names must mean something.
- **Use a consistent vocabulary.** One concept should have one term throughout the codebase (e.g., use `fetch` consistently for the same kind of retrieval). Prefer domain language (`invoice`, `shipment`) where it fits; use technical names (`http_client`, `retry_policy`) when the technical mechanism matters.
- **Name functions as actions.** Prefer descriptive verbs and objects such as `calculate_tax()`, `load_customer()`, `send_receipt()`, and predicates such as `is_payment_valid()`. A reader should infer a function's purpose without opening its body; avoid `process()`, `handle_data()`, `helper()`, and `_do_work()`.
- **Match name detail to scope.** Public, widely understood operations can use concise conventional names (`queue.push()`); narrowly scoped private helpers can have longer, precise names (`calculate_tax_for_cross_border_business_customer()`). More local visibility permits more specific names.
- **Prefer readable flow over clever compactness.** Give meaningful decisions names instead of compressing several rules into dense expressions. A top-level operation should read like a summary, with helpers exposing detail as needed (the *step-down rule*).

### Functions, responsibilities, and interfaces

- **Keep functions small and cohesive, not arbitrarily short.** Each should have one meaningful responsibility and one reason to change. Parsing, business validation, persistence, notifications, and response formatting should not be intermingled in one implementation. Small functions improve testing, reuse, debugging, and safe changes; a five-line function can still be complex and a twenty-line function can still be cohesive.
- **“Do one thing” means one level of responsibility, not one statement.** An orchestrator such as `publish_article()` may validate, save, and notify as a single coherent use case—but should delegate SQL, escaping, SMTP configuration, and retry logic. If describing a function requires unrelated “and/then/also” clauses, consider splitting it.
- **Keep one abstraction level per function.** Do not mix high-level calls (e.g., `validate_order()`) with cursor management or SQL in the same body. Extract lower-level details into appropriately named helpers. Read the call chain from high-level intent to increasingly concrete operations.
- **Use recognizable function roles:**
  - **Query:** returns information without externally visible mutation (`has_available_credit()`).
  - **Command:** performs an action or changes state (`reserve_inventory()`).
  - **Transformation:** converts representations (`parse_address()`).
  - **Predicate:** answers a focused yes/no question (`is_eligible()`).
  - **Orchestrator:** composes focused operations into a use case (`fulfill_order()`).
  A function should not switch roles unexpectedly.
- **Separate commands from queries.** Prefer explicit `activate_user(user)` and `is_user_active(user)` to a misleading `set_user_active()` used both to mutate and answer a question. Make side effects clear from names/contracts; avoid validation or getter functions that secretly update state, send messages, or create records. Hidden mutations create temporal coupling between calls.
- **Keep parameters few, meaningful, and unambiguous.** Large positional lists obscure order and valid combinations. Group truly cohesive data in a domain parameter object (e.g., `UserRegistration`), but do not manufacture objects for unrelated arguments.
- **Be cautious with boolean mode flags.** `render_page(page, True)` hides intent and may combine separate behaviors. Prefer `render_preview()`/`render_published_page()` or an explicit `mode=RenderMode.PREVIEW` where one abstraction genuinely supports multiple modes. A boolean that represents actual data is not inherently wrong.

### Complexity and control flow

- **Reduce cognitive complexity, not just line count.** Watch for deep nesting, many branches, mixed abstraction levels, flags, multiple duties, hidden mutation, excessive arguments, repeated rules, exception-driven flow, and call-order dependencies.
- **Make the happy path obvious.** Use guard clauses for prerequisites, or extract decisions into named predicates such as `can_ship(order)`. Simplify nested `if`/`else` structures when this clarifies intent; do not mechanically ban all nesting.
- **Treat repeated `switch`/`match`/`if` chains as a design signal.** If behavior repeatedly branches on type or role, localize that variation behind a meaningful interface or polymorphic operation; keep subtype dispatch near a boundary. A simple switch may still be clearer than introducing unnecessary inheritance.
- **Handle errors at a level that can act on them.** Keep the normal success path legible; use meaningful, domain-specific exceptions when preferable to repeatedly passing error codes through business logic. Do not use exceptions indiscriminately or hide failures behind excessive exception-driven control flow.

### Data, classes, and dependencies

- **Choose objects versus data structures deliberately.** Behavioral objects should protect invariants and hide representation behind operations (`account.withdraw()`); simple transfer records can expose fields explicitly (`Coordinate`). Avoid hybrids that reveal all internals yet require callers to enforce hidden rules. Data-oriented design is appropriate for data transformations.
- **Keep classes and modules cohesive.** Their methods and fields should belong to the same responsibility and evolve for related reasons. Split oversized `Manager`, `Service`, or `Utils` classes when they collect independently changing concerns—not just to create more classes.
- **Aim for high cohesion and low coupling.** Keep related behavior together; expose narrow APIs, not long chains of another object's internal fields (`order.customer.payment_profile.provider.token`). Depend on a suitable operation such as `payment_service.charge(order)` and keep infrastructure dependencies explicit.
- **Apply DRY to repeated knowledge, not matching text.** Centralize the *same business rule* when every copy would need to change for the same reason. Similar-looking code serving different concepts need not be unified; premature abstractions can be worse than duplication.

### Code clarity, conventions, and maintenance

- **Replace meaningful magic numbers with named constants** (e.g., `DISCOUNT_RATE` instead of an unexplained `0.10`), making their purpose and point of change explicit.
- **Write comments for what code cannot communicate clearly:** rationale, non-obvious constraints, interoperability quirks, legal/protocol requirements, and surprising behavior. Do not narrate obvious statements. First consider clearer names, helper functions, or types; keep necessary warnings.
- **Follow language and project conventions.** Use established standards (e.g., Python PEP 8, Java and JavaScript/TypeScript conventions), project-specific naming/folder rules, formatters, linters, and consistent layout. Consistency reduces review friction and onboarding cost.
- **Refactor continuously.** Rename unclear entities, extract responsibilities, remove duplicated logic and dead code, simplify branches and interfaces, and improve test clarity. Aim to leave code cleaner than you found it, without mistaking passing tests for good design.
- **Use version control (e.g., Git)** to track history, collaborate and review changes, experiment in branches, and revert mistakes—especially when refactoring.

## 2. Mandatory Project Workflow Architecture

**One core processing step per file**, plus **one separate final pipeline orchestrator**. The architecture must mirror the conceptual flow: **pipeline → core step (H1) → substep (H2) → atomic action (H3)**. A reader should understand the pipeline from the final orchestrator and any step from its H1 function without inspecting low-level code.

| Level | Purpose | Required name | Allowed calls |
|---|---|---|---|
| **H1** | Exactly one public entry point per core-step file; orchestrates the full step | `run_<core_step>()`, descriptive step name | **Only same-file H2** functions |
| **H2** | Coordinates one coherent subworkflow at a uniform abstraction level | `<verb>_<object>()`, **no leading `_`** | **Only same-file H3** actions |
| **H3** | Performs one precisely named, concrete action | `_<verb>_<object>()`, **exactly one leading `_`** | Low-level libraries, APIs, and shared primitives |

**Required boundaries:**

- **Strict adjacent-level calls (no hierarchy skipping):** A function may directly call only functions in the **immediately lower workflow hierarchy**: final pipeline → H1, H1 → H2, and H2 → H3. **An H1 orchestrator must never call an H3 atomic function directly**, even if both are in the same file. In general, a level `n` function must not call level `n−2` functions; **all `n−2` calls must be inside an `n−1` function** that level `n` calls. This rule applies to every code path, including branches and error handling.
- Name each step file for its core action (e.g., `step_02_extract_vector_field.py`). The H1 body is only a clear sequence of H2 calls; **no low-level tensor work, parsing, file I/O, plotting, metric calculations, or other implementation details** directly in H1.
- H2 functions coordinate related H3 actions and remain in the same core-step file. They may perform several operations only when these form one coherent substep; keep levels of abstraction consistent.
- H3 functions implement individual actions, not hidden subworkflows. If one needs to coordinate distinct actions, split it or move orchestration to H2.
- **Naming is part of the contract:** H1 starts with `run_`; H2 uses an unprefixed action verb/object; H3 starts with **exactly one** `_`. The leading underscore is reserved for H3 atomic functions *within core-step files*; do not prefix H2 merely because it is internal. Avoid generic helper names.
- The final pipeline orchestrator imports and invokes **H1 functions only**. Core-step files must **never import or call another step's H2/H3**. Put truly reusable low-level domain primitives in narrow shared modules; they must not conceal cross-step orchestration.
- Pass cross-step state **explicitly** via arguments, return values, or cohesive data objects. Avoid mutable global state and implicit coupling.

### Required source-file order (strict)

Every core-step file is arranged **top to bottom by dependency level**, not by runtime call order:

1. **Library imports** — Python standard-library and third-party imports, grouped to project standards.
2. **Macro imports** — project-local shared types, configuration, constants, and reusable low-level primitives; no cross-step H2/H3 imports.
3. **H3 atomic actions** — underscore-prefixed concrete implementations.
4. **H2 subworkflow coordinators** — unprefixed functions calling same-file H3 actions.
5. **H1 file orchestrator** — exactly one `run_<core_step>()`, **defined last** and calling same-file H2 functions only.

**Important distinction:** the *logical* workflow reads from high-level intent down to detail, but this project's **physical source-file order is deliberately bottom-up (H3 → H2 → H1)**. Keep each hierarchy grouped, not interleaved by call sequence. The final pipeline is a separate file.

### Example: one core-step file

```python
# step_02_extract_vector_field.py

# 1. Library imports
import numpy as np

# 2. Macro imports (shared project dependencies)
from project.types import ExtractionResult

# 3. H3: single actions
def _normalize_prompts(prompts):
    return [prompt.strip() for prompt in prompts]


def _evaluate_vector_field(model, prompts):
    return model.evaluate(prompts)


def _package_result(values):
    return ExtractionResult(values=values)


# 4. H2: coherent subworkflows
def prepare_extraction_inputs(prompts):
    return _normalize_prompts(prompts)


def compute_field_evaluations(model, prompts):
    return _evaluate_vector_field(model, prompts)


def package_vector_field_representation(values):
    return _package_result(values)


# 5. H1: only the step orchestration; ALWAYS LAST
def run_vector_field_extraction(model, prompts):
    prepared = prepare_extraction_inputs(prompts)
    evaluations = compute_field_evaluations(model, prepared)
    return package_vector_field_representation(evaluations)
```

### Example: separate final orchestrator

```python
# run_pipeline.py
from step_01_load_inputs import run_input_loading
from step_02_extract_vector_field import run_vector_field_extraction
from step_03_build_safe_reference import run_safe_reference_construction
from step_04_evaluate_prompts import run_prompt_evaluation


def run_pipeline(config):
    inputs = run_input_loading(config)
    fields = run_vector_field_extraction(inputs.model, inputs.prompts)
    reference = run_safe_reference_construction(config, fields)
    return run_prompt_evaluation(config, reference, inputs)
```

## 3. Refactoring, Reviews, and Quality Gates

### Practical refactoring sequence

1. Make the behavior work; **add or preserve tests** before changing structure.
2. Identify mixed responsibilities, abstraction levels, and surprising side effects.
3. Extract cohesive operations; choose names that expose purpose and function role.
4. Centralize duplicated *knowledge* without merging unrelated concepts.
5. Reduce nesting, unwieldy parameters, unclear branches, and needless coupling.
6. Remove dead code; improve interfaces, tests, and source organization while respecting the mandatory **H3 → H2 → H1** file order.
7. Re-run tests and analysis; review the result for clarity. Passing tests enable safer refactoring but do not prove maintainability.

### Complete review checklist

- [ ] **Names:** Intent-revealing, meaningful distinctions, consistent domain/technical vocabulary; suitable verb/predicate names and name detail for scope.
- [ ] **Functions:** One coherent responsibility and abstraction level; appropriate command/query/transformation/predicate/orchestrator role; purpose clear from the name; no hidden side effects.
- [ ] **Interfaces:** Few understandable arguments; cohesive parameter objects when warranted; no confusing boolean mode flags or call-order dependencies.
- [ ] **Control flow:** Readable happy path; manageable nesting/branching; large type switches or repeated decisions localized; errors handled without burying the normal path.
- [ ] **Design:** Cohesive classes/modules, deliberate object-versus-data choices, hidden invariants where appropriate, narrow coupling, no unnecessary abstractions or repeated business rules.
- [ ] **Clarity:** Constants named; comments explain reasons/constraints, not obvious statements; code readable rather than clever; no dead code.
- [ ] **Workflow files:** One named file per core step; exactly one H1 defined **last**; imports ordered **libraries → macros**, then **H3 → H2 → H1**, grouped by level.
- [ ] **Workflow names/dependencies:** Strict adjacent-level calls only (pipeline → H1 → H2 → H3); no skipped levels or direct H1 → H3 calls—all H3 calls required by H1 are inside H2; H1 `run_...` calls same-file H2 only; H2 unprefixed verbs call same-file H3 only; H3 exactly `_verb_object` performs one action; no cross-step private-helper imports.
- [ ] **Pipeline/data:** Separate final orchestrator imports H1 only; overall flow readable there; shared primitives kept low-level; explicit data passing without mutable globals.
- [ ] **Delivery:** Conventions and formatting followed; tests cover important behavior; Git history/review available; security, secrets, dependencies, and static-analysis findings checked.

## 4. AI-Assisted Development and Automation

- **Treat AI-generated code as a draft, never a free pass.** Check behavior, design fit, function length, duplication, naming, nesting, and unnecessary abstractions. Look for embedded credentials, environment-specific values, insecure/outdated dependencies, and other security flaws.
- **Review before merging:** verify correctness, conform to the mandatory architecture, refactor for readability, run tests and static analysis, scan dependencies/security/secrets, and normalize naming and formatting.
- **Automate repeatable checks:** formatters, linters, static analysis/type checks, test suites, coverage tools, duplication/complexity analysis, dependency and secret scanners, security checks, and automated pull-request review. Automation supports—but does not replace—human judgment about intent and maintainability.

## References

- Robert C. Martin, *Clean Code: A Handbook of Agile Software Craftsmanship* (names, functions, comments, objects/data structures, error handling, classes, and code smells).
- [Codacy — What Is Clean Code? A Guide to Principles and Best Practices](https://blog.codacy.com/what-is-clean-code)
- [Robert C. Martin — Clean Code course outline](https://cleancoder.com/files/cleanCodeCourse.md)
- [Robert C. Martin — The Inverse Scope Law of Function Names](https://www.informit.com/articles/article.aspx?p=1323426)
- [InformIT — Clean Code table of contents](https://www.informit.com/store/clean-code-a-handbook-of-agile-software-craftsmanship-9780132350884)
