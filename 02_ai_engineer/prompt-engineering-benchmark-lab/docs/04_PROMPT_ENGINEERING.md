# Prompt Engineering Design

The project treats prompt design as an experiment variable, not as a collection of examples.

P0–P16 cover zero-shot, label definitions, system/role prompts, few-shot, constraints, decision policies, JSON/structured output, persona, instruction/context separation, audience/tone, delimited data, contrastive examples, provider reasoning mode, branch-and-vote, grammar/schema constraints and a full advanced template.

For every strategy the important engineering questions are:

1. What changed relative to the baseline?
2. Which failure mode is it expected to address?
3. Does quality improve on the same examples?
4. Does output validity improve?
5. How much token/latency/cost overhead is introduced?
6. Does the strategy fix errors or create regressions?
7. Is the added complexity justified for production?

P14 does not ask the model to reveal hidden chain-of-thought. It executes independent branches that emit final labels and aggregates them deterministically.

Custom prompts use the same benchmark interface as built-in strategies, so they can be compared rather than merely tested interactively.
