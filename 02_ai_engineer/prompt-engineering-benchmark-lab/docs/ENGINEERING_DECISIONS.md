# Engineering Decisions — Why the Project Is Built This Way

This file is the decision log for the project. It is intentionally explicit: in a portfolio review, the important question is not only *what* was implemented, but *why* it was implemented that way and what trade-offs were accepted.

## D01 — Why support-ticket intent classification?

**Decision:** use a six-class SaaS customer-support routing problem (`api`, `billing`, `cancellation`, `complaint`, `technical`, `upgrade`).

**Why:** prompt engineering is easiest to evaluate when the output has an objective ground truth. A free-form chatbot answer is much harder to score reliably because multiple answers can be acceptable. Classification gives an exact label and enables Accuracy, Precision, Recall, Macro F1 and confusion matrices.

**Alternative considered:** open-ended response generation with an LLM-as-a-judge.

**Why not selected:** it would introduce another model into the evaluation loop, making the benchmark less controlled and less interpretable.

## D02 — Why use a public labelled dataset?

**Decision:** use `cngchis/Support-Ticket-Router-12K-Cleaned`.

**Why:** a public source makes the experiment reproducible for a recruiter or reviewer. The labels already match a realistic routing problem, so no unverifiable hand-labelling is required for the primary benchmark.

**Risk:** the dataset may contain synthetic or simplified examples and therefore does not perfectly represent a production inbox.

**Mitigation:** state this explicitly in Limitations and avoid claiming that benchmark scores equal production performance.

## D03 — Why a balanced final benchmark?

**Decision:** sample the same number of examples from each class.

**Why:** the goal is to compare prompt strategies, not to reproduce the exact class frequency of one particular company. Equal representation makes per-class changes visible and prevents a majority class from dominating Accuracy.

**Alternative:** natural class distribution.

**Trade-off:** natural distribution is better when estimating real traffic accuracy, while balanced evaluation is better for controlled prompt comparison. A production project should ideally report both.

## D04 — Why separate development and holdout sets?

**Decision:** use `development.csv` for prompt iteration/ablation and `benchmark.csv` as a frozen final holdout.

**Why:** repeatedly inspecting final test errors and rewriting prompts creates **prompt overfitting**, the prompt-engineering equivalent of tuning a machine-learning model on the test set.

**Rule:** do not modify P0–P16 after inspecting final holdout results.

## D05 — Why must few-shot examples be disjoint from both evaluation sets?

**Decision:** few-shot examples come from rows used by neither development nor benchmark evaluation.

**Why:** placing the exact evaluation example inside the prompt is data leakage. The model could appear accurate because the expected mapping is already present in context.

## D06 — Why keep the model fixed?

**Decision:** use the same model/provider and fixed decoding configuration across the prompt-technique benchmark. P13 and P14 are reported separately as advanced execution strategies because they intentionally change reasoning/execution behavior.

**Why:** a controlled experiment changes one major variable at a time. If both the model and prompt change, the performance delta cannot be attributed to prompt engineering.

## D07 — Why use a deterministic/low-variance configuration?

**Decision:** use the most deterministic configuration supported by the selected model; temperature is left configurable because model support differs.

**Why:** stochastic generation adds noise to prompt comparisons. We want differences caused mainly by prompt design.

**Important:** even low-temperature API inference can still be nondeterministic. This is why raw outputs and model name are logged.

## D08 — Why Macro F1 is the primary quality metric?

**Decision:** rank quality primarily by Macro F1, while also reporting Accuracy and Weighted F1.

**Why:** Macro F1 calculates F1 separately for every class and then averages them equally. A strategy cannot obtain a strong primary score merely by performing well on one dominant/easy category.

**Why not Accuracy alone:** Accuracy hides whether one class has very poor recall.

## D09 — Why report per-class Precision/Recall/F1?

**Decision:** save one per-class report per strategy.

**Why:** aggregate metrics say *whether* performance changed, but not *where*. For example, label definitions might specifically improve `api` vs `technical` separation while leaving other categories unchanged.

## D10 — Why use confusion matrices?

**Decision:** plot baseline and best confusion matrices, and retain per-class error views.

**Why:** a confusion matrix exposes systematic routing mistakes. `billing → cancellation` and `technical → api` have different business implications even if they contribute equally to the error count.

## D11 — Why benchmark a progression of prompts instead of unrelated prompts?

**Decision:** P0–P7 form the core interpretable ladder: baseline → definitions → role → examples → constraints → decision policy → JSON → schema enforcement. P8–P16 extend this with persona, explicit instruction/context, audience/tone/format, delimiters, contrastive examples, reasoning-model controls, external branch-and-vote, grammar-constrained output and a full reusable template.

**Why:** each stage introduces a recognizable engineering intervention. This makes causal interpretation stronger than comparing eight unrelated prompt texts.

## D12 — Why include an explicit decision policy but not Chain-of-Thought collection?

**Decision:** provide routing rules and request only the final classification.

**Why:** the task benefits from explicit decision boundaries, but the application does not need verbose internal reasoning. Short output reduces tokens, latency and parsing complexity.

## D13 — Why compare P6 JSON prompting with P7 Structured Outputs?

**Decision:** keep both.

**Why:** they answer different engineering questions. P6 asks the model in natural language to produce JSON. P7 constrains the response at the API/schema layer. Comparing them quantifies how much format reliability comes from prompting versus constrained generation.

## D14 — Why log every raw prediction?

**Decision:** store row-level predictions, raw response, tokens, latency, validity and error data.

**Why:** aggregate metrics are not auditable. Error analysis, regression analysis and reproducibility require the original row-level evidence.

## D15 — Why checkpoint after every request?

**Decision:** write results incrementally.

**Why:** an API benchmark can fail after hundreds of paid requests because of networking, rate limits or local interruption. Resumability prevents unnecessary repeated cost.

## D16 — Why add caching and `--force`?

**Decision:** completed `sample_id` values are skipped unless explicitly forced.

**Why:** repeated identical API calls cost money and can introduce unnecessary stochastic differences. `--force` remains available for intentional reruns.

## D17 — Why measure token usage?

**Decision:** record input, output and total tokens for every request.

**Why:** few-shot and policy-heavy prompts may improve F1 while substantially increasing context size. Prompt quality must be evaluated together with inference efficiency.

## D18 — Why measure mean, median and P95 latency?

**Decision:** report distribution-sensitive latency statistics instead of mean only.

**Why:** production users experience tail latency. A strategy with a good average but poor P95 may still produce an unacceptable UX or routing delay.

## D19 — Why externalize pricing in YAML?

**Decision:** API token prices live in `configs/pricing.yaml`.

**Why:** provider prices change. Hard-coding them in evaluation logic would silently make cost results stale and reduce reproducibility.

## D20 — Why include cost per 1,000 requests?

**Decision:** normalize cost to 1K requests in addition to total benchmark cost.

**Why:** normalized cost is easier to compare and can be scaled to expected production volume.

## D21 — Why use bootstrap confidence intervals?

**Decision:** calculate a 95% bootstrap CI for Macro F1.

**Why:** a point estimate does not show uncertainty. On a finite benchmark, a small F1 difference can be sampling noise. Confidence intervals discourage overclaiming tiny improvements.

**Caution:** overlapping/non-overlapping intervals alone are not a complete paired significance test. The project does not claim formal significance from CI overlap alone.

## D22 — Why perform error regression analysis?

**Decision:** count not only samples fixed by a stronger prompt but also previously correct samples that become wrong.

**Why:** optimization can introduce regressions. A serious engineering review must show both gains and losses.

## D23 — Why perform ablation on development data?

**Decision:** remove definitions/examples/constraints/policy one component at a time on the development set.

**Why:** prompt length itself is not evidence of value. Ablation tests whether each component contributes measurable quality or reliability.

## D24 — Why not automatically select the highest-F1 prompt for production?

**Decision:** final selection considers quality, validity, latency, cost and complexity.

**Why:** production systems are multi-objective. A +0.002 F1 improvement may not justify 3× token cost or worse P95 latency.

## D25 — Why provide mock mode?

**Decision:** include an offline keyword-based `MockLLMClient`.

**Why:** reviewers should be able to verify that data loading, prompt rendering, logging, metrics, plots and tests work without purchasing API usage.

**Rule:** mock metrics are engineering smoke-test results only and must never be presented as LLM benchmark evidence.

## D26 — Why no Docker/Kubernetes/FastAPI in this repository?

**Decision:** keep deployment infrastructure out of this project.

**Why:** the portfolio goal is to demonstrate prompt experimentation and LLM evaluation. Adding unrelated infrastructure would increase repository size without strengthening the central claim. Deployment can be demonstrated in a separate portfolio project.

## Final engineering principle

A prompt is treated here as a versioned component of an AI system, not as a piece of prose. It is designed against a development set, frozen, evaluated on a holdout, measured for quality/reliability/cost/latency, inspected for regressions, and selected using explicit trade-offs.

## D27 — Why separate mock data from the mock model?

**Decision:** treat `data source` and `LLM provider` as two independent experiment axes.

**Why:** synthetic data can still be classified by a real LLM, while real public data can be passed through a mock provider for pipeline testing. Conflating the two would make it easy to misinterpret what was actually evaluated.

**Engineering consequence:** the final portfolio benchmark should use `--source huggingface` plus a real provider (`ollama`, `groq`, `gemini`, or optional `openai`).

## D28 — Why add Ollama?

**Decision:** support a local Ollama provider in addition to cloud APIs.

**Why:** the strongest interpretation of “free” is local inference: no API key, no per-token invoice and no cloud quota. It also demonstrates that the benchmark logic is provider-agnostic.

**Trade-off:** local models may be smaller/slower than hosted models and consume local RAM/VRAM and electricity.

## D29 — Why add Groq and Gemini free-tier providers?

**Decision:** support two cloud providers with free-tier access instead of making the portfolio depend on a paid OpenAI key.

**Why:** reviewers and learners can reproduce real LLM experiments without mandatory spend. Groq is useful for fast inference and provider-level structured output; Gemini provides another independently hosted model family and structured-output implementation.

**Risk:** free-tier quotas and model availability can change.

**Mitigation:** provider/model IDs are environment-configurable and every request is checkpointed.

## D30 — Why keep raw results in provider-specific folders?

**Decision:** write `07_outputs/results/raw/<provider>/<strategy>.csv` rather than one global strategy file.

**Why:** the central prompt benchmark must keep provider/model fixed. Mixing `groq:p0` with `gemini:p7` would confound model capability with prompt design.

**Engineering consequence:** each provider gets its own leaderboard and figures.

## D31 — Why is the free demo intentionally small?

**Decision:** `run_free_demo.py` defaults to 20 rows per prompt strategy.

**Why:** a full 300-row holdout across eight strategies requires 2,400 model requests before ablations. A small first pass is safer for quota-limited free tiers and catches configuration mistakes early.

**Important:** `--limit` works with resumable caching. Increasing from 20 to 50 does not need to repeat the first 20 completed requests.

## D32 — Why provider-specific generation settings are allowed?

**Decision:** keep settings fixed *within* a provider benchmark, but do not force one universal temperature/thinking configuration onto every model family.

**Why:** model providers publish different recommended inference settings and supported parameter ranges. A fair prompt-technique experiment requires the comparable prompt strategies to use the same settings on the chosen model; it does not require intentionally misconfiguring another provider to match those settings.

**Engineering consequence:** provider comparisons are secondary experiments. Prompt-strategy comparisons remain the primary controlled experiment.
