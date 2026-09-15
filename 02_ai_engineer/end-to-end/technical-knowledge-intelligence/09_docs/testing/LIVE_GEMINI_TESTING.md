# Live Gemini Testing

## Important

Live Gemini tests are **not part of the default test suite**.

They make real API requests and can consume:
- request quota;
- token quota;
- paid billing.

They can also fail because of external conditions unrelated to your code.

## Current quota-related failure example

A response such as:

```text
429 RESOURCE_EXHAUSTED
GenerateRequestsPerDayPerProjectPerModel-FreeTier
```

means the API key may be valid, but the project/model quota is exhausted.

That is not the same as a unit/integration failure.

## Running the live test

1. Put the key in local `.env` or environment variable:

```env
GEMINI_API_KEY=...
```

2. Run:

```bat
run_project.bat test live
```

The runner sets:

```text
TKI_RUN_LIVE_GEMINI=1
```

so the test knows you explicitly opted in.

## What the live suite currently proves

The minimal live smoke test proves:
- `google-genai` can initialize;
- the configured key can authenticate;
- the configured Gemini model can accept a request at that moment.

It does not benchmark answer quality.

## Why it is minimal

A test suite should not burn 5–10 Gemini calls just to prove connectivity.

Quality benchmarking belongs in the explicit Benchmarking/Playground workflow, where request count and cost are visible.

## If quota is exhausted

Use:

```bat
run_project.bat test offline
```

All deterministic project validation still works.
