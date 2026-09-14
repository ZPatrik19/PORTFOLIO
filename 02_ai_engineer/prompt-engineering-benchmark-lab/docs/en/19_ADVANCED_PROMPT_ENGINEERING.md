# Advanced Prompt Engineering — Deep Dive

## Prompt anatomy
P16 separates persona, instruction, context, audience, tone, reference data, examples, constraints, untrusted input, and output format. These blocks are hypotheses, not dogma: the benchmark measures whether each component adds quality relative to token/latency overhead.

## P0–P16 interpretation
P0 is the control. P1 adds label semantics; P2 system role; P3 few-shot; P4 constraints; P5 decision policy; P6 prompt-only JSON; P7 provider-enforced structured output; P8 persona; P9 explicit instruction/context blocks; P10 machine audience/tone/format; P11 delimited untrusted data; P12 contrastive examples; P13 reasoning-mode configuration; P14 independent branch-and-vote; P15 grammar/schema constrained generation; P16 combines the advanced architecture.

## Reasoning safety
The project does not request or persist private chain-of-thought. P13 uses provider-exposed reasoning controls where supported. P14 uses multiple independent final-label branches and deterministic voting; only final decisions are stored.

## What to measure
Quality, output validity, tokens, latency, cost, uncertainty, scenario performance, fixed/regressed examples, and hard-case behavior. The longest prompt is not assumed to be the best production prompt.

## Anti-patterns
Do not tune on the final holdout, present mock scores as real LLM evidence, change model and prompt simultaneously in a prompt-only experiment, or mix decoding changes into the same causal comparison without control.
