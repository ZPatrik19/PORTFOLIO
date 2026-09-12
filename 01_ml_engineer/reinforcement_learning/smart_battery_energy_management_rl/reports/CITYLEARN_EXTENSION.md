# Optional CityLearn extension

## Why CityLearn is the second stage, not the first

The custom `BatteryEnvironment` is deliberately small enough that every transition can be inspected manually. The goal is to understand what the agent observes, what action it requests, what action the battery can physically execute, how SOC changes, and how the reward is constructed.

CityLearn becomes useful after those fundamentals are clear because it provides a more standardized building/district energy-management benchmark with richer exogenous signals and storage control tasks.

```text
Custom educational BatteryEnvironment
        ↓
State / action / transition / reward are transparent
        ↓
Baselines and RL evaluation are validated
        ↓
CityLearn
        ↓
More realistic standardized benchmark
```

## Conceptual mapping

| Custom project | CityLearn-level concept |
|---|---|
| hourly synthetic demand | building electrical load |
| renewable generation | on-site solar / renewable generation |
| battery SOC | electrical storage state |
| charge / idle / discharge | storage control action |
| electricity price | pricing signal |
| grid import | net electricity consumption |
| custom reward | cost / peak / carbon / district objectives |
| one battery | one or multiple buildings / district |

## Recommended benchmark experiment

1. Install CityLearn as an optional dependency in a separate environment or with the project optional dependency.
2. Select a CityLearn schema that exposes electrical storage and pricing.
3. Start with a single building so the comparison remains interpretable.
4. Reuse the same evaluation philosophy from this repository:
   - electricity cost;
   - peak demand;
   - storage throughput;
   - equivalent cycles or storage-use proxy;
   - constraint validity;
   - episode return.
5. Train at least one standard policy, for example PPO, with multiple random seeds.
6. Compare the CityLearn policy against CityLearn-compatible rule-based or no-storage baselines.
7. Only after the single-building benchmark is understood, move to multi-building coordination.

## Important methodological point

Do not compare raw reward values between the custom environment and CityLearn as though they were the same objective. Compare interpretable KPIs or normalize objectives first. The environments can have different reward definitions, scales, constraints and exogenous data distributions.

## Portfolio story

A strong interview narrative is:

> I first created a transparent Gymnasium-compatible battery environment and validated the physics, reward shaping and baseline logic. After I understood the RL mechanics end-to-end, I moved the same evaluation methodology to CityLearn to test whether the conclusions held in a richer standardized energy-management benchmark.
