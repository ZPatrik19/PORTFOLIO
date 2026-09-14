# Experiment Protocol

## Research question

How much do specific prompt-engineering interventions change LLM support-ticket intent classification quality, format reliability, token usage, latency and estimated cost when the model and evaluation data are held constant?

## Independent variable

Prompt strategy (P0–P16), analyzed in benchmark families so prompt design, decoding parameters and advanced execution are not conflated.

## Controlled variables

- LLM model and provider
- holdout examples
- example order
- API configuration
- max output tokens
- parsing rules
- pricing assumptions
- metric definitions
- retry behavior

## Dependent variables

- Accuracy
- Macro Precision / Recall / F1
- Weighted F1
- per-class Precision / Recall / F1
- invalid output rate
- invalid JSON rate
- input/output/total tokens
- mean/median/P95 latency
- estimated API cost
- bootstrap 95% CI of Macro F1

## Data protocol

1. Load and clean the source dataset.
2. Remove blank/unknown-label records and exact duplicates.
3. For each class, select disjoint benchmark, development and few-shot examples using a fixed seed.
4. Verify zero text overlap between development and benchmark.
5. Freeze the final benchmark.

## Prompt development protocol

1. Develop P0–P16 against the task definition and development data only.
2. Use ablation studies on development data.
3. Freeze prompt versions.
4. Run the holdout benchmark.
5. Do not modify prompts based on final holdout errors.

## Execution protocol

1. Run `--dry-run` and inspect prompt payloads.
2. Run a 10-request API smoke test.
3. Confirm parsing/token/latency logging.
4. Run every P0–P16 strategy on the same holdout. Keep decoding settings fixed for the prompt-technique comparison; report reasoning/branching strategies as a separate advanced-execution family.
5. Generate reports.
6. Save exact model and pricing configuration with the run.

## Interpretation protocol

A strategy is not considered better solely because its F1 is numerically higher. Review:

- absolute F1 gain over P0
- confidence interval
- per-class gains/losses
- fixed examples vs regressions
- invalid-output change
- input-token overhead
- cost per 1K requests
- P95 latency
- implementation complexity

## Production decision

Choose the simplest strategy on or near the quality/cost/latency frontier that satisfies output-validity requirements. If schema validity is operationally mandatory, Structured Outputs can be preferred even when its F1 is similar to prompt-only JSON.

## Provider protocol

The repository supports `mock`, `ollama`, `groq`, `gemini`, `openrouter`, and `openai`, but a single P0–P16 benchmark family must use exactly one fixed provider/model combination.

Provider comparison is a separate experiment. Do not combine prompt-strategy rows from different providers into one causal conclusion about prompt quality.

Recommended free workflow:

1. `mock` to verify the repository.
2. `ollama`, `groq`, or `gemini` for a real-LLM pilot.
3. `--source huggingface` for the final public-data benchmark.
4. Increase `--limit` progressively when using quota-limited cloud free tiers.
5. Generate a provider-specific report only after the selected P0–P16 runs are complete.
