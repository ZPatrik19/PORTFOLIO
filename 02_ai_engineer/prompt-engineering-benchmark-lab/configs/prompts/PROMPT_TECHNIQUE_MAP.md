# Prompt Technique Map

| ID | Technique | What changes? | Main metric to watch |
|---|---|---|---|
| P0 | Zero-shot | Minimal task instruction | Macro F1 |
| P1 | Definitions | Adds label semantics/context | Macro F1 / confusion pairs |
| P2 | Role/System | Adds system-level role | F1 / adherence |
| P3 | Few-shot | Adds demonstrations | F1 / token overhead |
| P4 | Constraints | Adds explicit do/don't rules | Invalid-output rate |
| P5 | Decision policy | Adds deterministic precedence rules | Macro F1 |
| P6 | Prompt-only JSON | Requests JSON in natural language | Invalid JSON rate |
| P7 | Structured Output | Provider schema enforcement | JSON validity |
| P8 | Persona | Strong domain persona | F1 |
| P9 | Instruction + Context | Separates task/context/reference/data | F1 / robustness |
| P10 | Format + Audience + Tone | Optimizes downstream communication contract | Format validity |
| P11 | Delimited Data | Isolates untrusted ticket data | Robustness |
| P12 | Contrastive Few-shot | Adds confusing near-boundary examples | Per-class F1 |
| P13 | Reasoning-model mode | Requests reasoning effort without exposing CoT | Hard-case accuracy / latency |
| P14 | Branch + Vote | 3 independent expert branches + majority vote | F1 / cost / latency |
| P15 | Grammar/Schema constrained | Constrained machine-readable output | Output validity |
| P16 | Full Advanced Template | Persona+instruction+context+audience+tone+data+format+examples | Overall trade-off |

P14 is **Tree-of-Thought-inspired**, not hidden-chain-of-thought extraction. The benchmark observes only final branch labels and the majority vote.
