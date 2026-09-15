# Test Matrix

This matrix connects system components to the test layer that validates them.

| Component | Unit | Integration | Evaluation | Regression | Robustness | Performance | Smoke | Live Gemini |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Config / presets | ✓ |  |  | ✓ |  |  |  |  |
| PDF/DOCX parsing | ✓ | ✓ |  |  | malformed cases planned |  |  |  |
| Chunking | ✓ | ✓ |  |  |  |  |  |  |
| Embedding cache |  | ✓ |  | ✓ |  |  |  |  |
| BM25 | ✓ | ✓ | metric benchmark |  |  | ✓ |  |  |
| Dense retrieval |  | ✓ | benchmark |  |  |  |  |  |
| Hybrid / RRF | ✓ | ✓ | Recall/MRR/nDCG | regression dataset |  |  |  |  |
| Reranking |  | ✓ | benchmark | regression dataset |  |  |  |  |
| Query analysis | ✓ |  | tool/topic scoring | ✓ | multilingual planned |  |  |  |
| Prompt engineering | ✓ |  | prompt benchmark | prompt regression planned | injection |  |  | optional |
| Tool registry | ✓ | ✓ | tool metrics |  | allowlist |  |  | optional |
| Citation validation | ✓ | ✓ | citation metrics | golden cases planned | false citation |  |  | optional |
| FastAPI |  |  |  |  |  |  | ✓ |  |
| Gemini SDK integration | mocked |  |  |  |  |  |  | ✓ |
| Monitoring / telemetry | unit planned | integration planned |  |  |  |  |  |  |

## Interpretation

A check mark does not mean the component is "perfect". It means at least one explicit contract exists in that test layer.

The matrix is intended to make test gaps visible rather than hide them.
