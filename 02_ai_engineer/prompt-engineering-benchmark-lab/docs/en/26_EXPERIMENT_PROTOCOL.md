# Experiment Protocol

## Controlled prompt benchmark
Keep dataset/order, provider/model, decoding settings, max output tokens, parser, metric code, retry/cache rules fixed. Change only the prompt strategy.

## Experiment families
1. Prompt strategy benchmark.
2. Decoding sweep with fixed prompt.
3. Prompt ablation.
4. Provider/model comparison with fixed prompt.
5. Base vs fine-tuned comparison on the same holdout.

## Leakage policy
Few-shot examples and fine-tuning training/validation rows cannot originate from the final holdout.

## Run identity
Record provider, model, strategy, sampling settings, dataset/sample identity, prompt version, pricing configuration, timestamps, and raw predictions. Cache reuse is allowed only when identity-compatible.

## Reporting
Report both aggregate and sliced metrics, uncertainty, invalid output, tokens, latency, and cost. State whether a run is mock simulation or real provider evidence.
