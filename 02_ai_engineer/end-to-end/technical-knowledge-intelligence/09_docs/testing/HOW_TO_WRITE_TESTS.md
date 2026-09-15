# How to Write Tests in This Repository

## Naming

Prefer behavior-oriented names:

```python
def test_bm25_prefers_chunk_with_matching_technical_terms():
```

Avoid vague names:

```python
def test_bm25():
```

## Arrange → Act → Assert

For non-trivial tests:

```python
def test_example():
    # Arrange
    input_data = ...

    # Act
    result = function_under_test(input_data)

    # Assert
    assert result == expected
```

## One principal behavior per test

A test may use multiple assertions if they describe one contract.

Bad:

```text
test_everything()
```

Good:

```text
test_citation_is_rejected_when_chunk_is_not_in_selected_context()
test_registered_tool_executes_with_valid_arguments()
test_unregistered_tool_name_is_rejected()
```

## Prefer deterministic data

Good:

```text
3 ranked IDs + 1 known relevant ID
```

Avoid depending on:
- current internet content;
- random LLM prose;
- private 300-page books;
- free-tier quota.

## Use `pytest.approx` for floats

```python
assert metric == pytest.approx(1 / 3)
```

Never rely on exact floating-point representation when unnecessary.

## Use parametrization for equivalent cases

```python
@pytest.mark.parametrize(
    ("question", "expected"),
    [...],
)
def test_intent(question, expected):
    ...
```

## External APIs

External API calls belong only in explicitly marked live tests.

Normal tests should mock failures/success contracts where possible.

## Performance tests

Use generous thresholds and explain their purpose.

Do not write a flaky assertion such as:

```python
assert latency_ms < 5
```

unless the execution environment is tightly controlled.
