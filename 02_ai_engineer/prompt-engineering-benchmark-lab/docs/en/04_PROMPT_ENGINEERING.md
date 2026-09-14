# Prompt Engineering

## Experimental principle
Prompt design is an independent variable. The benchmark keeps dataset, model/provider, parser, metric implementation, and evaluation protocol fixed while changing prompt strategy.

## Strategy ladder
- P0: naive zero-shot baseline.
- P1: explicit label definitions.
- P2: role/system instruction.
- P3: few-shot demonstrations.
- P4: explicit constraints.
- P5: decision policy / structured reasoning without requesting private chain-of-thought.
- P6: prompt-only JSON output.
- P7: schema-constrained structured output.
- P8–P12: persona, context/instruction separation, audience/tone, delimited user data, contrastive examples.
- P13: provider-supported reasoning configuration.
- P14: Tree-of-Thought-inspired independent branches + vote; only final labels are stored.
- P15: grammar/schema-constrained generation.
- P16: full advanced prompt template.

## Custom prompts
Users can define system and user templates, configure structured output/reasoning, save presets, run them in the Playground, and benchmark them against P0–P16.

## Fair-comparison rule
A strategy comparison is meaningful only when the same samples and model/provider settings are used. Sampling-parameter sweeps are therefore separated from prompt-strategy benchmarks.
