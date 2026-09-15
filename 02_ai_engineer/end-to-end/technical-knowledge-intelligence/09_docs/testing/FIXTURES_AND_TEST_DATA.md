# Fixtures and Test Data

## Principle

Normal tests must not depend on the user's private technical books.

User-library documents are:
- large;
- copyrighted/private;
- slow to parse;
- unsuitable for deterministic CI;
- not available to other GitHub users.

Therefore the test suite uses **small synthetic fixtures** and the public demo corpus.

## Shared factories

Location:

```text
06_tests/helpers/factories.py
```

Factories create:
- `Chunk`
- `SearchHit`
- `SourceCitation`
- `DocumentRecord`

Example:

```python
chunk = make_chunk(
    chunk_id="docker-networking",
    text="Docker bridge networking connects containers...",
)
```

The purpose is to keep tests focused on behavior instead of long Pydantic constructor boilerplate.

## Isolated config fixture

`isolated_config` is defined in:

```text
06_tests/conftest.py
```

It changes runtime paths to pytest's temporary directory and forces:

```text
embedding provider = local_hashing
vector store = numpy
user-library documents = empty temp directory
logs = temp directory
indexes = temp directory
```

This guarantees that tests do not:
- overwrite the user's real index;
- parse user-library documents accidentally;
- consume Gemini embedding quota;
- write test telemetry into production telemetry.

## Creating document fixtures

For file parsers, generate the smallest possible valid file in `tmp_path`.

Example DOCX test:

```text
heading
paragraph
2x2 table
```

This is much safer than depending on a 300-page real ebook.

## Evaluation fixtures

Metric tests should prefer hand-calculable ranking lists:

```text
results:  [irrelevant, relevant, other]
expected: {relevant}
```

Expected:

```text
MRR = 0.5
Recall@1 = 0
Recall@3 = 1
Precision@3 = 1/3
```

This makes failures easy to understand and audit.
