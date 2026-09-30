# Clean Code — Principles and Best Practices

> **Source:** Codacy, “What Is Clean Code? A Guide to Principles and Best Practices”  
> https://blog.codacy.com/what-is-clean-code  
> **Note:** This is an original summary for project use, not a verbatim copy of the article.

## What Is Clean Code?

Clean code is code designed to be easy for people to read, understand, modify, test, and maintain. The idea is commonly associated with Robert C. Martin’s *Clean Code*, and emphasizes writing software that remains understandable throughout its lifecycle—not merely code that executes correctly.

Good clean-code practices help make intent obvious, reduce unnecessary complexity, and make future changes safer.

## Why Clean Code Matters

Clean code improves a software project in several ways:

- **Readability and maintainability:** Developers can understand existing behavior and make changes more quickly.
- **Team collaboration:** Consistent conventions make it easier for multiple contributors to work in the same codebase.
- **Debugging:** Clear structure, names, and responsibilities make defects easier to locate and fix.
- **Quality and reliability:** Simpler, more consistent code reduces opportunities for accidental errors.
- **Onboarding:** New contributors can become productive faster when the code communicates its intent clearly.

## Core Clean-Code Principles

### 1. Avoid Magic Numbers

Do not scatter unexplained numeric values throughout the code. Give important values meaningful constant names.

Instead of:

```python
discount = price * 0.1
```

Prefer:

```python
DISCOUNT_RATE = 0.10
discount = price * DISCOUNT_RATE
```

This makes the purpose of the value obvious and centralizes future changes.

### 2. Use Meaningful Names

Variable, function, class, and module names should communicate purpose.

Prefer names such as:

```python
product_price
discount_amount
calculate_order_total()
```

over vague names such as:

```python
x
value
do_it()
```

Names should reduce the need for explanatory comments.

### 3. Use Comments Sparingly

Comments are most useful when they explain **why** something exists, document constraints, warn about surprising behavior, or clarify decisions that cannot be expressed directly in code.

Avoid comments that merely repeat what the code already says.

### 4. Keep Functions Small and Focused

A function should ideally have one clear responsibility. If it validates input, performs calculations, formats output, and writes data all at once, it is usually doing too much.

Breaking large functions into focused pieces makes them easier to:

- understand,
- test,
- reuse,
- debug,
- and change safely.

### 5. Follow DRY — Don't Repeat Yourself

Repeated logic creates multiple places that must remain synchronized.

Extract duplicated behavior into reusable functions, classes, modules, or other suitable abstractions. The goal is not to eliminate every repeated line, but to avoid maintaining the same rule or business logic in multiple places.

### 6. Follow Established Coding Standards

Use the conventions of your language and project. Consistency makes a codebase easier to scan and reduces unnecessary stylistic debates.

Examples include:

- PEP 8 for Python,
- established Java conventions,
- established JavaScript/TypeScript style guides,
- project-specific naming and folder conventions.

Automated formatters and linters can enforce many of these rules.

### 7. Simplify Nested Conditionals

Deeply nested `if`/`else` logic is hard to follow.

Consider extracting complicated decisions into well-named helper functions:

```python
def calculate_discounted_price(product_price):
    rate = get_discount_rate(product_price)
    return product_price * (1 - rate)
```

The helper name can communicate the intent of the decision without forcing readers to understand every branch immediately.

### 8. Refactor Continuously

Clean code is not a one-time activity.

Review and improve code as the system evolves. Refactoring can include:

- renaming unclear variables,
- splitting large functions,
- reducing duplication,
- simplifying conditionals,
- removing dead code,
- improving interfaces,
- and making tests easier to understand.

A useful habit is to leave the code slightly cleaner than you found it.

### 9. Use Version Control

Version-control systems such as Git provide a safety net for changes.

They allow teams to:

- track modifications,
- review history,
- collaborate safely,
- experiment in branches,
- and revert problematic changes.

Version control makes continuous refactoring much less risky.


## Robert C. Martin's Principles in More Detail

The following expands the ideas most closely associated with Robert C. Martin's *Clean Code*. They are best treated as design heuristics rather than absolute laws: the goal is to make intent easy to see and change inexpensive.

### Names Should Reveal Intent

A good name should answer the reader's first question: **what role does this thing play?**

Names should describe purpose rather than implementation accidents.

Instead of:

```python
d = 7
items2 = []
do_stuff()
```

prefer names that expose the domain meaning:

```python
trial_period_days = 7
eligible_orders = []
calculate_invoice_total()
```

For functions, Martin's guidance strongly favors **verb-like names** that describe an action:

```text
calculate_tax()
load_customer()
is_payment_valid()
send_receipt()
```

The reader should be able to predict what a function does from its name without opening its body.

#### Make distinctions meaningful

Avoid pairs such as:

```text
data / data2
account / account_info
product / product_data
```

unless the distinction is genuinely meaningful.

Prefer names that encode the actual difference:

```text
pending_orders / fulfilled_orders
billing_address / shipping_address
gross_total / net_total
```

#### Use one vocabulary consistently

If the codebase uses `fetch` for retrieving remote data, do not randomly alternate between:

```text
fetch_user()
get_customer()
retrieve_account()
load_member()
```

for the same conceptual operation.

Different words should generally indicate different concepts. Consistent vocabulary makes an API easier to learn.

#### Use domain language where appropriate

Names should come from the language of the problem when possible:

```text
invoice
shipment
reservation
subscription
settlement
```

Implementation-oriented names are still useful when they describe a technical mechanism that genuinely matters:

```text
http_client
token_parser
retry_policy
```

The important point is that a name should orient the reader immediately.

### Function Names and Scope

Martin has also argued that function names and visibility are related.

A widely used public operation may have a concise, conventional name:

```python
queue.push(item)
file.close()
stream.read()
```

A small private helper called from one nearby place can afford to be much more descriptive:

```python
def calculate_tax_for_cross_border_business_customer(...):
    ...
```

The local helper's long name acts almost like a sentence in the calling function.

For example:

```python
def checkout(order):
    validate_order_is_ready_for_checkout(order)
    reserve_inventory_for_order_items(order.items)
    charge_customer_for_order_total(order)
    send_order_confirmation(order)
```

The top-level function reads like a summary of the workflow. The lower-level functions contain the implementation details.

This creates a useful hierarchy:

```text
checkout
  -> validate order
  -> reserve inventory
  -> charge customer
  -> send confirmation
```

The code becomes readable from high-level intent down into increasingly detailed steps.

### Functions Should Be Small

Martin's strongest function-level heuristic is simple: **keep functions small**.

Smallness is not valuable because of an arbitrary line limit. It is valuable because small functions force responsibilities to become explicit.

A function that:

- parses input,
- validates business rules,
- queries a database,
- calculates prices,
- sends email,
- and formats an HTTP response

contains several reasons to change.

A cleaner design separates those responsibilities.

Instead of:

```python
def create_order(request):
    # parse request
    # validate customer
    # calculate price
    # save order
    # send notification
    # construct HTTP response
    ...
```

prefer an orchestration function whose body stays at one conceptual level:

```python
def create_order(request):
    order_request = parse_order_request(request)
    validate_order_request(order_request)
    order = place_order(order_request)
    notify_customer(order)
    return build_created_response(order)
```

The exact number of extracted functions is context-dependent. The important outcome is that each function has a clear purpose.

### "Do One Thing" Means One Level of Responsibility

"One thing" can be misunderstood as "one statement" or "one operation."

A function may perform several steps and still have one responsibility if those steps all belong to one coherent abstraction.

For example:

```python
def publish_article(article):
    validate_article(article)
    save_article(article)
    notify_subscribers(article)
```

At the level of the caller, those operations together may represent the single use-case concept **publish an article**.

But if `publish_article()` also contains low-level SQL construction, HTML escaping, SMTP configuration, and retry timing calculations, then it mixes several abstraction levels and responsibilities.

A practical test is:

> Can you describe the function accurately with one short verb phrase without using "and", "or", "then", or "also"?

If the description becomes:

> "validate the payment **and** calculate tax **and** update the database **and** email the customer"

the function is probably carrying several roles.

### Keep One Level of Abstraction per Function

A function is easier to read when its statements operate at roughly the same conceptual level.

This is mixed abstraction:

```python
def process_order(order):
    validate_order(order)

    connection = database.connect()
    cursor = connection.cursor()
    cursor.execute(
        "INSERT INTO orders(customer_id, total) VALUES (?, ?)",
        (order.customer_id, order.total),
    )

    send_confirmation(order)
```

`validate_order()` and `send_confirmation()` are high-level operations, while cursor management and SQL are much lower-level details.

A cleaner version hides those details:

```python
def process_order(order):
    validate_order(order)
    save_order(order)
    send_confirmation(order)
```

Then `save_order()` can contain the persistence-specific work.

This is closely related to Martin's **step-down rule**: a reader should be able to move from a high-level function into its helpers and encounter progressively more detailed explanations of the same operation.

### Think of Functions as Having Clear Roles

A useful way to apply Martin's principles is to assign each function a clear role.

These are not rigid categories, but they help expose mixed responsibilities.

#### Query

Returns information without changing externally visible state.

```python
def has_available_credit(customer) -> bool:
    ...
```

#### Command

Changes state or performs an action.

```python
def reserve_inventory(order) -> None:
    ...
```

#### Transformation

Converts one representation into another.

```python
def parse_address(payload) -> Address:
    ...
```

#### Predicate

Answers a focused yes/no question.

```python
def is_eligible_for_discount(customer) -> bool:
    ...
```

#### Orchestrator

Coordinates a use case by delegating work to more focused functions.

```python
def fulfill_order(order):
    reserve_inventory(order)
    charge_customer(order)
    create_shipment(order)
```

Problems often appear when one function silently changes roles—for example, a function named `get_customer()` that also creates a missing customer, writes an audit record, and sends an email.

### Separate Commands from Queries

Martin emphasizes **command-query separation**.

A function should normally either:

- **do something**, or
- **answer something**,

rather than surprise the caller by doing both.

Potentially confusing:

```python
if set_user_active(user):
    ...
```

Does `set_user_active()` change state? Return the previous state? Return whether the write succeeded? All three are plausible.

Clearer APIs separate the intentions:

```python
activate_user(user)

if is_user_active(user):
    ...
```

This is especially valuable when reading code during debugging because mutation is easy to identify.

### Avoid Hidden Side Effects

A function's behavior should match what its name and signature imply.

Consider:

```python
def validate_password(user, password):
    if password_matches(user, password):
        reset_login_attempts(user)
        update_last_login(user)
        return True
    return False
```

A reader may reasonably expect `validate_password()` only to validate. Resetting counters and updating timestamps are additional effects that are not obvious from the name.

Better options include:

- rename the operation so the side effects are part of its contract,
- move the mutations into an explicit command,
- or have a higher-level function coordinate validation and mutation.

Hidden side effects create temporal coupling: callers must know that invoking one function changes what later functions are allowed to do.

### Prefer Few Function Arguments

Long parameter lists increase cognitive load because the reader must understand:

- the meaning of every argument,
- their order,
- whether arguments are related,
- which combinations are valid,
- and whether the function is doing too much.

Hard to scan:

```python
create_user(
    name,
    email,
    street,
    city,
    postcode,
    country,
    send_email,
    make_admin,
    trial_days,
)
```

A cohesive parameter object can make the relationship clearer:

```python
registration = UserRegistration(
    name=name,
    email=email,
    address=address,
    trial_days=trial_days,
)

create_user(registration)
```

Parameter objects are not automatically cleaner. They are useful when the grouped values form a meaningful concept.

#### Be suspicious of boolean flag arguments

A call like this is difficult to understand:

```python
render_page(page, True)
```

The boolean often means the function contains two behaviors.

More explicit alternatives are:

```python
render_preview(page)
render_published_page(page)
```

or:

```python
render_page(page, mode=RenderMode.PREVIEW)
```

The right choice depends on whether the behaviors genuinely belong to one abstraction.

### Complexity Is More Than Line Count

A short function can still be difficult to understand.

Sources of complexity include:

- deep nesting,
- many branches,
- mixed abstraction levels,
- boolean flags,
- multiple responsibilities,
- hidden mutation,
- many arguments,
- duplicated rules,
- exception-heavy control flow,
- and temporal coupling between calls.

For example:

```python
def calculate_price(order):
    if order.customer:
        if order.customer.active:
            if order.items:
                if order.region == "EU":
                    ...
```

The difficulty comes from the number of conditions a reader must hold in working memory.

A decomposition might expose the decision structure:

```python
def calculate_price(order):
    ensure_order_can_be_priced(order)
    subtotal = calculate_subtotal(order.items)
    discount = calculate_discount(order.customer, subtotal)
    tax = calculate_tax(order.region, subtotal - discount)
    return subtotal - discount + tax
```

The lower-level functions may still contain branching, but each branch is now attached to a named concept.

### Reduce Nesting

Nesting often indicates that several decisions are being expressed at once.

Guard clauses can make prerequisites explicit:

```python
def ship_order(order):
    if order.cancelled:
        return
    if not order.paid:
        return
    if not order.items:
        return

    create_shipment(order)
```

This is frequently easier to scan than wrapping the main behavior inside multiple nested `if` blocks.

Extraction is another option:

```python
def ship_order(order):
    if not can_ship(order):
        return

    create_shipment(order)
```

The goal is not to mechanically eliminate every nested block. It is to keep the main path obvious.

### Treat Switches and Large Conditionals as Design Signals

A large `switch`, `match`, or `if/elif` chain is not automatically wrong, but it can signal that behavior varies by type or role.

For example:

```python
if employee.type == "hourly":
    ...
elif employee.type == "salaried":
    ...
elif employee.type == "contractor":
    ...
```

If the same branching recurs throughout the system, behavior may belong behind a polymorphic interface:

```python
employee.calculate_pay()
```

The branching can then be localized near object creation or another boundary rather than repeated throughout business logic.

The important principle is **localize variation**. Do not make every caller understand every subtype.

### Objects and Data Structures Have Different Roles

Martin draws an important distinction between objects and data structures.

An **object** hides its internal representation and exposes behavior:

```python
account.withdraw(amount)
account.available_credit()
```

A **data structure** primarily exposes data and carries little behavior:

```python
@dataclass
class Coordinate:
    latitude: float
    longitude: float
```

Both styles are useful.

Problems arise when code mixes them carelessly—for example, a class that exposes all internal fields publicly while also requiring callers to respect complex behavioral rules.

Ask what role a type is intended to play:

- If it models behavior and invariants, hide representation behind meaningful operations.
- If it is simply transferring structured data, keep it simple and explicit.

### Classes Should Be Cohesive

At class level, the same responsibility principle applies.

A cohesive class has fields and methods that belong together.

Warning signs include:

- methods using completely different subsets of fields,
- unrelated business rules grouped in one utility class,
- a class changing for many independent reasons,
- or a generic manager/service class accumulating every new feature.

For example:

```text
UserManager
  create_user()
  send_marketing_email()
  generate_invoice()
  resize_avatar()
  export_csv()
```

The name is broad because the responsibilities are broad.

More cohesive roles might be:

```text
UserRegistrationService
MarketingNotifier
InvoiceService
AvatarProcessor
UserExporter
```

Splitting is useful when responsibilities actually evolve independently—not merely to maximize the number of classes.

### Cohesion and Coupling

Clean design aims for:

- **high cohesion** inside a module or class,
- **low coupling** between modules or classes.

High cohesion means related behavior lives together.

Low coupling means one component knows as little as practical about the internals of another.

Instead of:

```python
order.customer.payment_profile.provider.token
```

prefer an operation that belongs to the relevant abstraction:

```python
payment_service.charge(order)
```

Long chains of internal navigation often expose representation details and make changes ripple through the system.

### Prefer Exceptions to Error-Code Plumbing

Martin favors keeping error handling separate from the normal success path.

Error-code style:

```python
result = save_order(order)

if result == ERROR_DATABASE:
    ...
elif result == ERROR_DUPLICATE:
    ...
```

Exception-based style can keep the primary logic clearer:

```python
try:
    save_order(order)
except DuplicateOrderError:
    ...
except DatabaseUnavailableError:
    ...
```

The goal is not "exceptions everywhere." It is to avoid forcing normal business logic to carry low-level error-state bookkeeping through every call.

Use domain-specific exceptions where they make failure meaning clearer, and handle errors at a level that can make a useful decision.

### DRY Is About Repeated Knowledge

"Don't Repeat Yourself" is most valuable when applied to **knowledge and rules**, not merely identical text.

If a tax rule appears in four services, then changing the tax rule requires finding four implementations. That is dangerous duplication.

Repeated syntax is not always the same thing.

Two similar-looking functions may represent different concepts that happen to be implemented the same way today. Merging them prematurely can create the wrong abstraction.

A good question is:

> If this rule changes, should all copies change for the same reason?

If yes, they probably represent duplicated knowledge.

### Comments Should Explain Context, Not Compensate for Code

A comment is useful when the reason cannot be expressed clearly in code.

Useful:

```python
# The provider may repeat webhook deliveries for up to 24 hours,
# so event IDs must remain idempotent across retries.
```

Less useful:

```python
# Increment retry count
retry_count += 1
```

Before adding a comment that explains *what* a block does, consider whether:

- a better function name,
- a better variable name,
- an extracted helper,
- or a clearer type

could make the comment unnecessary.

Comments are still appropriate for:

- rationale,
- non-obvious constraints,
- interoperability quirks,
- legal or protocol requirements,
- and warnings about surprising behavior.

### Prefer Readable Flow Over Cleverness

Clean code should minimize the amount of mental simulation required.

Compact code is not automatically clean:

```python
return [x for x in xs if p(x) and not q(x) and (r(x) or s(x))]
```

An explicit version may communicate the business rules more clearly:

```python
def eligible_items(items):
    return [
        item
        for item in items
        if is_eligible(item)
    ]
```

with:

```python
def is_eligible(item):
    if not is_active(item):
        return False
    if is_blocked(item):
        return False
    return meets_primary_rule(item) or meets_exception_rule(item)
```

The cleaner version gives important concepts names.

### Organize Code From General to Specific

Martin often describes source code as something that should read top-to-bottom like a well-structured article.

A file can start with the high-level operation:

```python
def fulfill_order(order):
    validate(order)
    reserve(order)
    charge(order)
    dispatch(order)
```

and place increasingly detailed helpers below it.

This structure helps readers choose how far down they need to go.

A developer investigating the workflow may only need the first function. A developer debugging payment can continue into `charge()`.

### Refactoring Is How Clean Functions Are Usually Produced

Clean functions often do not emerge perfectly in the first draft.

A practical workflow is:

1. Make the behavior work.
2. Add or preserve tests.
3. Identify responsibilities hidden inside the code.
4. Extract those responsibilities into named functions.
5. Rename until intent is obvious.
6. Remove duplication.
7. Reduce nesting and argument complexity.
8. Reorder the code so it reads from intent to detail.

The important discipline is to separate "working" from "finished." Passing tests are a prerequisite for safe refactoring, not proof that the design is already clear.

### A Function Review Checklist

When reviewing a function, ask:

- [ ] Does the name describe the function's purpose rather than its implementation?
- [ ] Can I understand the purpose without reading the body?
- [ ] Does it have one coherent responsibility?
- [ ] Are its statements at roughly one level of abstraction?
- [ ] Does it behave as a command, a query, a transformation, a predicate, or a clear orchestrator rather than mixing roles unexpectedly?
- [ ] Are side effects explicit?
- [ ] Is the argument list small and understandable?
- [ ] Are boolean flags hiding multiple behaviors?
- [ ] Is the happy path easy to find?
- [ ] Can nesting or branching be simplified?
- [ ] Is repeated business knowledge centralized?
- [ ] Are errors handled without obscuring the normal flow?
- [ ] Would extracting a helper give an important concept a useful name?

### A Class and Module Review Checklist

For larger units, ask:

- [ ] Does the name describe a coherent role?
- [ ] Do the methods and data belong together?
- [ ] Does the unit have a focused reason to change?
- [ ] Are unrelated concerns being accumulated in a generic `Manager`, `Helper`, `Utils`, or `Service`?
- [ ] Is internal representation hidden when invariants matter?
- [ ] Are simple data-transfer types kept simple?
- [ ] Is variation localized rather than duplicated across many callers?
- [ ] Is coupling to infrastructure or neighboring modules kept explicit and narrow?
- [ ] Can the public API be understood without knowing private implementation details?

### Interpreting the Principles Pragmatically

These principles are heuristics, not mechanical scoring rules.

For example:

- A five-line function can still be too complex.
- A twenty-line function can be perfectly coherent.
- A boolean argument can be appropriate when it represents genuine data rather than a mode switch.
- A `switch` can be clearer than unnecessary inheritance.
- Duplication can be safer than forcing unrelated concepts into one abstraction.
- A data-oriented structure can be better than an object if the code is fundamentally transforming data.

The deeper goal is consistent throughout Martin's work: **make the structure of the code communicate the structure of the problem**.


## Clean Code in AI-Assisted Development

AI coding tools can generate useful code quickly, but generated code still requires the same engineering review as human-written code.

Potential problems include:

- overly long functions,
- duplicated logic,
- inconsistent naming or structure,
- unnecessary abstractions,
- excessive nesting,
- hard-coded credentials or secrets,
- and outdated or insecure dependencies.

AI-generated code may be functionally correct while still being difficult to maintain or unsafe in a larger codebase.

Treat generated code as a draft:

1. Review it for correctness.
2. Check whether it fits the existing architecture.
3. Refactor for readability.
4. Run tests and static analysis.
5. Check dependencies and security issues.
6. Remove secrets and environment-specific values.
7. Make naming and formatting consistent with the project.

## Practical Clean-Code Checklist

Before merging code, ask:

- [ ] Are names clear and specific?
- [ ] Are unexplained constants replaced with named values?
- [ ] Does each function have a focused responsibility?
- [ ] Is duplicated business logic minimized?
- [ ] Are comments explaining useful context rather than obvious code?
- [ ] Are complex conditionals simplified or extracted?
- [ ] Does the code follow project formatting and naming conventions?
- [ ] Are tests present for important behavior?
- [ ] Has dead or unnecessary code been removed?
- [ ] Have security issues, secrets, and dependencies been checked?
- [ ] Would another developer understand the intent without extra explanation?

## Tooling and Automation

Many clean-code practices can be supported automatically with:

- formatters,
- linters,
- static-analysis tools,
- test suites,
- code-coverage tools,
- dependency scanners,
- secret scanners,
- security analysis,
- and automated pull-request review.

The Codacy article presents its platform as one option for automating code-quality and security checks, including duplication, complexity, test coverage, static analysis, dependency/security checks, and AI-assisted pull-request review.

## Key Takeaway

Clean code is code optimized for future readers and maintainers as well as for the computer executing it.

The most useful habits are simple: choose clear names, keep responsibilities focused, avoid unnecessary duplication and complexity, follow consistent standards, refactor regularly, review AI-generated code carefully, and automate quality checks where possible.

## Further Reading

- Robert C. Martin, *Clean Code: A Handbook of Agile Software Craftsmanship* — especially the chapters on meaningful names, functions, comments, objects/data structures, error handling, classes, and code smells.
- Robert C. Martin, Clean Code course outline: https://cleancoder.com/files/cleanCodeCourse.md
- Robert C. Martin, “The Inverse Scope Law of Function Names”: https://www.informit.com/articles/article.aspx?p=1323426
- InformIT, *Clean Code* table of contents: https://www.informit.com/store/clean-code-a-handbook-of-agile-software-craftsmanship-9780132350884
