# PyTorch Dataset and DataLoader

A Dataset represents addressable samples while a DataLoader manages batching, ordering, worker processes and collation. A custom Dataset typically implements length and indexed item access. Separating data access from batching makes training code reusable.

## Custom Dataset example
```python
from torch.utils.data import Dataset
class Rows(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]
```
