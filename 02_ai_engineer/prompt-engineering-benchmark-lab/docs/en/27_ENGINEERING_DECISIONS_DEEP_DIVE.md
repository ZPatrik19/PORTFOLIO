# Engineering Decisions — Deep Dive

1. **Classification task:** objective labels make prompt differences measurable.
2. **Development + holdout:** reduces prompt-overfitting risk.
3. **Balanced core benchmark:** prevents dominant classes from hiding failures.
4. **Macro F1 primary:** every class matters equally.
5. **Accuracy not enough:** class-wise metrics and confusion are required.
6. **Output validity separate:** parseable machine contracts matter in production.
7. **P6 vs P7:** prompt-only JSON and constrained generation are different mechanisms.
8. **P95/P99 latency:** tail latency matters for user/SLA experience.
9. **Capability matrix:** do not send unsupported parameters for UI symmetry.
10. **No hidden CoT storage:** reasoning is evaluated through final decisions and external branch aggregation.
11. **Raw predictions persisted:** aggregate metrics are insufficient for debugging.
12. **Resumable benchmark:** API failures/quota should not force re-paying completed requests.
13. **Dataset-aware cache:** sample IDs alone are not enough identity.
14. **Custom prompts:** the platform must evaluate user hypotheses, not only built-in recipes.
15. **Fine-tuning export, not auto-training:** avoids surprise cost and provider lock-in.
16. **Multi-objective production decision:** quality, reliability, tokens, latency, cost, and complexity all matter.
