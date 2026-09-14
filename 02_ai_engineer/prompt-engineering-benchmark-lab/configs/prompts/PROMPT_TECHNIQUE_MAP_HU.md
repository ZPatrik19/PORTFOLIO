# Prompttechnikák – magyar térkép

| ID | Technika | Mit változtatunk? | Mit mérünk? |
|---|---|---|---|
| P0 | Zero-shot baseline | Minimális instrukció | baseline Accuracy/Macro F1 |
| P1 | Label definitions | Kategóriák jelentése | ambiguity reduction |
| P2 | Role/System | Rendszer-szintű szerep | instruction adherence |
| P3 | Few-shot | Példák | quality vs token overhead |
| P4 | Constraints | Explicit szabályok | invalid-output rate |
| P5 | Decision policy | Döntési prioritás | multi-intent/boundary accuracy |
| P6 | Prompt-only JSON | JSON kimeneti kérés | invalid JSON rate |
| P7 | Structured Output | Provider schema | syntactic reliability |
| P8 | Persona | Domain persona | domain-sensitive quality |
| P9 | Instruction + Context | Tagolt prompt | long-context robustness |
| P10 | Format + Audience + Tone | Downstream cél és stílus | format compliance |
| P11 | Delimited data | Untrusted input izolálása | prompt-injection robustness |
| P12 | Contrastive few-shot | Határeset példák | confusing-class performance |
| P13 | Reasoning mode | Provider reasoning effort | hard-case quality vs tokens/latency |
| P14 | Branch + Vote | Több független ág | robustness vs 3× request overhead |
| P15 | Grammar/schema constraint | Constrained generation | output validity |
| P16 | Full advanced template | Persona+instruction+context+examples+constraints+format | overall trade-off |

A `configs/prompts/hu/` mappában mind a P0–P16 technikához magyar tükörtemplate található.
A `configs/prompts/custom/` mappába a UI-ból létrehozott saját prompt presetek kerülnek.
