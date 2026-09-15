# Python Data Model and Iteration

Python iteration is based on iterable objects that produce iterators. An iterator exposes a next-step operation and signals exhaustion. Generator functions provide a compact way to implement lazy iteration. Context managers separate resource acquisition from cleanup and are commonly used with files.

## Example
```python
def batches(items, size):
    for i in range(0, len(items), size):
        yield items[i:i+size]
```
