# Smart Battery Energy Management — Reinforcement Learning Decision System

A portfolio-grade reinforcement learning project that combines **Electrical Engineering + Machine Learning + Optimization + Reinforcement Learning**. The goal is to learn when an energy-storage system should charge, remain idle or discharge while electricity prices, demand, renewable generation and battery state evolve over time.

> **Main portfolio message:** I did not run an RL tutorial. I designed a physically interpretable decision environment, reward function, battery constraints, baseline controllers and multiple RL policies, then evaluated them with business and energy KPIs rather than cumulative reward alone.

## System view

```text
             Environment
                  │
          state / observation
                  ▼
             RL Agent
                  │
               action
                  ▼
        Battery Controller
                  │
       charge / idle / discharge
                  │
                  ▼
         Energy Environment
                  │
     cost + degradation + peak
                  │
                  ▼
                reward
                  │
                  └──────────► Agent
```

## Why Reinforcement Learning?

This is not a standard supervised-learning problem. In supervised learning we normally learn a mapping from an input to a known target label. Here there is no single correct action label for every hour. The value of an action depends on its long-term consequences:

```text
current action
      ↓
changes battery SOC
      ↓
changes future feasible actions
      ↓
changes future energy cost / peak / degradation
      ↓
long-term return matters
```

Charging now can be locally expensive yet globally optimal if it preserves energy for a future high-price period. Conversely, aggressive discharge can reduce today's bill while leaving the battery empty when it is more valuable later.

## RL concepts mapped to the battery problem

- **Agent:** the decision maker choosing battery actions.
- **Environment:** the battery + price + demand + renewable system.
- **State / observation:** information available before an action: SOC, current price, demand, renewable generation and hour of day; optionally forecast features.
- **Action:** charge, idle or discharge in the discrete environment; continuous battery power in PPO.
- **Transition:** the physics that converts action + current state into next SOC and grid balance.
- **Reward:** numerical objective combining energy cost, peak demand, degradation and attempted constraint violation.
- **Policy:** the rule learned by the agent that maps observations to actions.
- **Episode:** one 24-hour control horizon by default; change `episode_hours` to 168 for a week.
- **Return:** the sum of rewards over an episode.
- **Discount factor (`gamma`):** controls how much future rewards matter relative to immediate rewards.
- **Exploration:** trying uncertain actions to learn their value.
- **Exploitation:** choosing the action currently believed to be best.
- **Value function:** expected future return from a state.
- **Q-function `Q(s,a)`:** expected future return after taking action `a` in state `s` and following the policy afterwards.
- **Bellman equation:** expresses current action value using immediate reward plus discounted future value.
- **Experience Replay:** DQN stores past transitions and trains on shuffled minibatches, breaking temporal correlation.
- **Target Network:** DQN uses a delayed copy of the Q-network to stabilize Bellman targets.

## State space

Minimum normalized observation vector:

```python
[
    battery_state_of_charge,
    electricity_price,
    electricity_demand,
    renewable_generation,
    hour_of_day,
]
```

Why each feature is present:

| Variable | Why the agent needs it |
|---|---|
| SOC | Determines how much energy can still be charged/discharged and affects future flexibility. |
| Electricity price | Directly changes the economic value of import and discharge. |
| Demand | Determines local load and potential peak demand. |
| Renewable generation | Indicates how much local generation can cover load or charge the battery. |
| Hour of day | Encodes recurring daily structure without exposing the future. |
| Forecasts (optional) | A realistic controller may have decision-time demand/renewable forecasts. They are simulated forecasts, not future actual values. |

The project uses a chronological train/validation/test split. Future test observations are not used for training.

## Action space

Discrete environment:

```text
0 = charge
1 = idle
2 = discharge
```

Continuous environment for PPO:

```text
[-1, 1]
-1 = maximum discharge
 0 = idle
+1 = maximum charge
```

The environment records both `requested_battery_power_kw` and `executed_battery_power_kw`. Requested power is clipped by charge/discharge power limits and SOC bounds before it affects the system.

## Battery physics

Parameters in `config.yaml`:

```text
capacity_kwh
initial_soc
min_soc
max_soc
max_charge_kw
max_discharge_kw
charging_efficiency
discharging_efficiency
feed_in_tariff_eur_per_kwh
```

Charging:

```text
ΔE_battery = charge_power × charging_efficiency × Δt
```

Discharging:

```text
ΔE_battery = -discharge_power / discharging_efficiency × Δt
```

Energy balance:

```text
grid_power = demand - renewable_generation + battery_charging - battery_discharging
```

Positive grid power means import. Negative grid power means export.

## Reward function

The reward is deliberately decomposed:

```text
reward =
    - electricity_cost
    - peak_demand_penalty
    - battery_degradation_penalty
    - constraint_violation_penalty
```

Every component is logged per timestep and visualized. This prevents the common mistake of treating RL reward as an opaque score.

A deliberately bad reward is also included in `src/battery_rl/rewards.py`. It rewards discharge directly, which can encourage wasteful battery cycling. The key lesson is:

> The agent optimizes the reward we implement, not the intent we had in mind.

## Baselines

RL is never evaluated alone:

1. **No Battery** — always idle.
2. **Random Valid Policy** — samples a physically admissible discrete action.
3. **Rule-Based Controller** — charges below a low price threshold, discharges above a high threshold.
4. **Perfect-Information Heuristic** — sees the full episode price profile and therefore acts as an oracle-style offline reference, not a fair online controller.
5. **Tabular Q-Learning** — own implementation with discretized state.
6. **DQN** — explicit educational PyTorch implementation with Q-network, replay buffer, target network and epsilon-greedy exploration.
7. **PPO** — explicit educational PyTorch actor-critic implementation with squashed Gaussian continuous actions, GAE and clipped surrogate objective.

Stable-Baselines3 remains an optional reference dependency rather than the only way the project can run.

## Evaluation KPIs

The project reports:

- total electricity cost;
- cost saving % vs No Battery;
- peak grid demand;
- peak reduction %;
- renewable self-consumption;
- battery throughput;
- equivalent battery cycles;
- constraint-attempt count;
- average SOC;
- cumulative reward / episode return;
- mean ± standard deviation over repeated episodes and, for full DQN/PPO runs, multiple training seeds.

## How to interpret model ranking

This project does **not** assume that reinforcement learning must automatically beat a strong heuristic. A tabular Q-Learning policy can underperform the rule-based controller because discretization throws away information and the finite Q-table suffers from the curse of dimensionality. That is a valid engineering result. DQN/PPO are included precisely to test whether function approximation or continuous control improves the trade-off. The correct conclusion comes from the KPI table, not from the algorithm name.

The peak component is a **soft per-timestep threshold penalty**, not a literal monthly utility demand-charge model. It is intentionally simple and transparent for the educational environment; replacing it with tariff-specific demand charges is a natural production extension.

## Stress tests

The same learned policy is evaluated on:

```text
normal prices
high-price scenario
volatile-price scenario
high renewable scenario
low renewable scenario
```

This tests whether a policy is robust beyond the exact training distribution.

## Ablation study

Two examples are automated in Step 08:

```text
reward without degradation penalty
vs
reward with degradation penalty
```

and

```text
state without forecast
vs
state with forecast
```

The forecast experiment explicitly uses simulated decision-time forecasts rather than future realized values.

## Project structure

```text
smart_battery_energy_management_rl/
│
├── 01_data/
│   ├── raw/
│   └── processed/
│
├── workflow/
│   ├── step01_problem_and_data.ipynb
│   ├── step01_problem_and_data.py
│   ├── step02_eda.ipynb
│   ├── step02_eda.py
│   ├── step03_environment_design.ipynb
│   ├── step03_environment_design.py
│   ├── step04_baseline_policies.ipynb
│   ├── step04_baseline_policies.py
│   ├── step05_q_learning.ipynb
│   ├── step05_q_learning.py
│   ├── step06_dqn.ipynb
│   ├── step06_dqn.py
│   ├── step07_ppo.ipynb
│   ├── step07_ppo.py
│   ├── step08_evaluation_stress_ablation.ipynb
│   ├── step08_evaluation_stress_ablation.py
│   ├── step09_policy_analysis.ipynb
│   ├── step09_policy_analysis.py
│   └── shared/
│       ├── utils.py
│       └── visualization.py
│
├── src/battery_rl/
│   ├── environment/battery_env.py
│   ├── agents/baselines.py
│   ├── agents/q_learning.py
│   ├── agents/dqn_torch.py
│   ├── agents/ppo_torch.py
│   ├── data.py
│   ├── rewards.py
│   ├── evaluation.py
│   ├── sb3_utils.py              # optional SB3 reference helper
│   └── citylearn_extension.py
│
├── 03_tests/
├── 04_results/
│   ├── figures/
│   ├── metrics/
│   ├── models/
│   └── predictions/
│
├── 05_graphviz/
├── reports/
├── 00_setup_project.py
├── switch_language.py
├── tools/
│   └── set_notebook_language.py
├── run_project.py
├── config.yaml
├── requirements.txt
├── pyproject.toml
└── README.md
```

### Why this structure?

`workflow/` answers **"what do I run, and in what order?"**. The paired `.py` files are compact automation/orchestration entry points. The `.ipynb` files are deliberately more verbose: they expose intermediate calculations, tables, training diagnostics and plotting code cell by cell for learning and portfolio review. Both reuse the same `src/battery_rl/` domain/RL classes, so the physics and algorithms are not reimplemented inconsistently. `04_results/` contains generated outputs and exported figures.

## STEP 00 — One-command environment setup

Recommended Python: 3.10–3.12. Instead of manually creating the virtual environment and installing packages one by one, run:

```bash
python 00_setup_project.py
```

The bootstrap script performs the complete local setup:

```text
validate Python version
        ↓
create/reuse .venv
        ↓
upgrade pip + setuptools + wheel
        ↓
install project + dev dependencies
        ↓
prepare/generate/download configured data if missing
        ↓
register Jupyter kernel
        ↓
run import + BatteryEnvironment smoke test
```

Useful setup options:

```bash
# completely recreate the virtual environment
python 00_setup_project.py --recreate

# regenerate/redownload the configured dataset
python 00_setup_project.py --force-data

# install optional reference/extension dependencies too
python 00_setup_project.py --with-sb3
python 00_setup_project.py --with-citylearn
python 00_setup_project.py --with-sb3 --with-citylearn
```

The default `data.source: synthetic` configuration does not require internet access; it generates the educational hourly dataset locally. If the project is later switched to `data.source: url`, the same setup script downloads the configured CSV only when it is missing.

A Python process cannot permanently activate a virtual environment in its parent terminal, so the script prints the exact activation command at the end. On Windows it is typically:

```powershell
.venv\Scripts\Activate.ps1
```

and on Linux/macOS:

```bash
source .venv/bin/activate
```

Manual installation is still supported if required:

```bash
python -m venv .venv
pip install -e ".[dev]"
```

## Hungarian / English notebook language switch

The workflow notebooks contain both Hungarian and English narrative variants in cell metadata. There is **one canonical notebook set**; the code, execution outputs, charts, and model results are not duplicated. Only the active notebook markdown changes.

Switch all workflow notebooks to Hungarian:

```bash
python tools/set_notebook_language.py --language hu
# or the root convenience wrapper
python switch_language.py --language hu
```

Switch all workflow notebooks to English:

```bash
python tools/set_notebook_language.py --language en
# or
python switch_language.py --language en
```

Check current status:

```bash
python tools/set_notebook_language.py --status
```

Optionally export both rendered-source versions without changing the canonical notebooks:

```bash
python tools/set_notebook_language.py --export-both notebook_exports
```

The switcher preserves execution counts and embedded PNG outputs. The current code cells intentionally use English identifiers; they contain almost no natural-language `#` comments. The same metadata mechanism also supports language-specific code-cell/comment variants if such comments are added later.

## How to run it step by step

From the project root:

```bash
python workflow/step01_problem_and_data.py
python workflow/step02_eda.py
python workflow/step03_environment_design.py
python workflow/step04_baseline_policies.py
python workflow/step05_q_learning.py
python workflow/step06_dqn.py
python workflow/step07_ppo.py
python workflow/step08_evaluation_stress_ablation.py
python workflow/step09_policy_analysis.py
```

Or open the matching notebooks in exactly the same order.

## How to run the complete workflow with one command

Fast smoke run:

```bash
python run_project.py --quick
```

Full configured run:

```bash
python run_project.py
```

Selected steps:

```bash
python run_project.py --steps 1-5 --quick
python run_project.py --steps 8-9
python run_project.py --steps 3,4,5
```

## Tests

```bash
pytest
```

The tests validate battery SOC limits, energy balance, efficiency direction, action clipping and Q-learning discretization.

## Notebook-first workflow and embedded outputs

The notebooks are intentionally **not thin wrappers around the `.py` files**. They are portfolio/tutorial artifacts with visible calculations and rendered outputs. The current executed notebook set contains:

| Notebook | Total cells | Code cells | Embedded PNG outputs |
|---|---:|---:|---:|
| STEP 01 — Problem & data | 22 | 10 | 4 |
| STEP 02 — EDA | 26 | 12 | 9 |
| STEP 03 — Environment | 37 | 17 | 6 |
| STEP 04 — Baselines | 23 | 11 | 8 |
| STEP 05 — Q-Learning | 25 | 12 | 5 |
| STEP 06 — DQN | 25 | 12 | 5 |
| STEP 07 — PPO | 27 | 12 | 7 |
| STEP 08 — Evaluation/Ablation | 22 | 12 | 6 |
| STEP 09 — Policy analysis | 22 | 11 | 7 |

All nine notebooks have been executed and saved with outputs. `reports/notebook_execution_report.json` records the validation summary.

## Expected visual outputs

The repository now contains **50+ notebook-generated figures** in addition to the compact workflow figures. These include:

- price, demand, renewable and net-load time series;
- train/validation/test timeline;
- forecast-vs-actual leakage demonstration;
- price/demand/renewable distributions and hourly profiles;
- feature correlation and price-vs-net-load scatter;
- agent-environment control-loop diagram;
- requested-vs-executed battery power under SOC constraints;
- episode SOC, charge/discharge and grid-import dynamics;
- reward decomposition and cumulative reward;
- baseline cost, peak, cycles, renewable self-consumption and trajectory comparisons;
- epsilon decay, multi-seed Q-Learning curves and state-space coverage;
- learned Q-policy heatmaps;
- DQN return, loss, epsilon, mean-Q and representative policy episode;
- PPO continuous-action mapping, clipped-objective curve, losses, entropy/KL and action distribution;
- stress-test comparisons and reward/state ablations;
- SOC/action distributions, price-by-action, price×SOC decisions and action-transition matrix;
- CityLearn extension path.


## What each notebook is responsible for

- `step01_problem_and_data.ipynb`: problem framing, RL terminology, schema, temporal split, leakage and episode construction.
- `step02_eda.ipynb`: data quality, distributions, daily structure, net load, correlation, peaks and RL-relevant EDA interpretation.
- `step03_environment_design.ipynb`: battery physics, observation/action spaces, SOC transition, clipping, energy balance, reward decomposition and physical assertions.
- `step04_baseline_policies.ipynb`: all baseline controllers, common evaluation and trajectory/KPI comparison.
- `step05_q_learning.ipynb`: discretization, Bellman update, epsilon schedule, multi-seed training, Q-table coverage and learned policy heatmap.
- `step06_dqn.ipynb`: PyTorch Q-network, replay buffer, Bellman target, target network, training diagnostics and test rollout.
- `step07_ppo.ipynb`: continuous control, actor-critic, squashed Gaussian policy, clipped objective, GAE, optimization diagnostics and rollout.
- `step08_evaluation_stress_ablation.ipynb`: KPI comparison, stress scenarios, reward ablation and state/forecast ablation.
- `step09_policy_analysis.ipynb`: SOC/action behavior, price-conditioned decisions, transition matrix, policy lookup and CityLearn path.

## CityLearn extension

The project deliberately follows this progression:

```text
Custom educational BatteryEnvironment
        ↓
understand state/action/transition/reward
        ↓
validate controllers and RL evaluation
        ↓
standardized realistic environment
        ↓
CityLearn benchmark
```

`src/battery_rl/citylearn_extension.py` documents the conceptual mapping. CityLearn remains optional so the core project stays understandable and runnable without hiding battery physics behind a larger framework. A more detailed migration/benchmark plan is in `reports/CITYLEARN_EXTENSION.md`.

## Interview discussion points

A strong walkthrough should explain:

- why chronological splitting matters even in RL scenario generation;
- why reward design is an engineering specification, not just a scalar;
- why SOC clipping alone is not enough and requested vs executed action should be logged;
- why `reward = -cost` may underrepresent degradation and peak-demand objectives;
- why a perfect-information baseline is useful but not a fair online competitor;
- why DQN fits a discrete action space while PPO naturally extends to continuous battery power;
- why cumulative reward is insufficient for energy-system evaluation;
- how equivalent cycles approximate wear and why a better electrochemical degradation model would be a natural future extension.
