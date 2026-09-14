# Benchmark Datasets

## Challenge Set
The built-in synthetic Challenge Set is designed for controlled stress testing, not as a claim of natural production-distribution realism. It contains 10,800 raw rows, a 6,000-row holdout, 3,000-row development set, six labels, and 18 scenario families.

## Scenario design
Cases include clear intent, implicit request, ambiguity, multi-intent priority, typos/noise, long context, prompt injection, resolved history, negation/correction, quoted thread, multilingual mix, telegraphic text, primary-first/last, distractors, code/log noise, label-word attacks, and double negation.

## External data
The Hugging Face support-ticket source is useful as an additional generalization set but is itself synthetic; it should not be described as production ground truth.

## Strong portfolio evidence
The strongest evidence is a locked evaluation protocol across: a challenge suite, an independent external set, and ideally a human-reviewed/private holdout, all run with a real provider/model.
