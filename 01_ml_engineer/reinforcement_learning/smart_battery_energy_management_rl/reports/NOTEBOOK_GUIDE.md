# Notebook Guide — Smart Battery Energy Management RL

The `workflow/` notebooks are the pedagogical and portfolio-facing layer of the project. They are intentionally more verbose than the paired `.py` steps. Each notebook contains visible Python code, intermediate tables, inline plots and engineering interpretation.

## Execution order

1. `step01_problem_and_data.ipynb` — formulate the sequential decision problem, inspect the dataset, temporal split, leakage and 24-hour episode windows.
2. `step02_eda.ipynb` — data quality, distributions, daily profiles, net load, correlation and peak periods.
3. `step03_environment_design.ipynb` — state/action definition, battery physics, SOC transition, requested vs executed power, energy balance, reward decomposition and physical assertions.
4. `step04_baseline_policies.ipynb` — No Battery, Random, Rule-Based and Perfect-Information heuristic baselines with KPI and trajectory comparison.
5. `step05_q_learning.ipynb` — discretization, Bellman update, epsilon-greedy exploration, multi-seed training, state-space coverage and policy heatmap.
6. `step06_dqn.ipynb` — PyTorch Q-network, experience replay, target network, Bellman target, DQN training diagnostics and test rollout.
7. `step07_ppo.ipynb` — continuous control, actor-critic, squashed Gaussian policy, PPO clipping, GAE, optimization diagnostics and evaluation.
8. `step08_evaluation_stress_ablation.ipynb` — policy comparison, business/energy KPIs, stress scenarios, reward ablation and forecast-state ablation.
9. `step09_policy_analysis.ipynb` — SOC/action distributions, price-conditioned decisions, action transitions, learned heatmap and CityLearn extension path.

## Notebook quality checks

The notebooks were executed after generation and saved with outputs. The execution report is stored in `reports/notebook_execution_report.json`.

| Notebook | Cells | Code | Markdown | Inline PNG | Errors |
|---|---:|---:|---:|---:|---:|
| STEP 01 | 22 | 10 | 12 | 4 | 0 |
| STEP 02 | 26 | 12 | 14 | 9 | 0 |
| STEP 03 | 37 | 17 | 20 | 6 | 0 |
| STEP 04 | 23 | 11 | 12 | 8 | 0 |
| STEP 05 | 25 | 12 | 13 | 5 | 0 |
| STEP 06 | 25 | 12 | 13 | 5 | 0 |
| STEP 07 | 27 | 12 | 15 | 7 | 0 |
| STEP 08 | 22 | 12 | 10 | 6 | 0 |
| STEP 09 | 22 | 11 | 11 | 7 | 0 |

A separate PNG file is also written for every notebook visualization into `04_results/figures/`, so the same chart can be embedded in the README, reports or presentation material later.

## `.ipynb` versus `.py`

The two formats have different responsibilities:

- `.ipynb`: learning, inspection, visible intermediate calculations, plots and interpretation.
- `.py`: reproducible automation, one-command workflow and compact orchestration.
- `src/battery_rl/`: the reusable physics, agent and evaluation implementation used by both layers.

This avoids two bad extremes: notebooks that contain only `main()` calls, and notebooks that duplicate an entirely separate hidden production implementation.

## Quick and full modes

The notebooks default to `QUICK_MODE = True` so a recruiter or reviewer can rerun them without waiting for a long RL experiment. Change it to `False` for the larger configured training budgets.

The command-line equivalent is:

```bash
python run_project.py --quick
python run_project.py
```

## STEP 00 before opening the notebooks

Run from the repository root:

```bash
python 00_setup_project.py
```

This creates the local `.venv`, installs the project dependencies, prepares the configured data, and registers the `Python (battery-energy-rl)` kernel expected by the notebook metadata.

## Hungarian / English narrative

The notebooks are bilingual without duplicating the codebase. Every markdown cell stores both language variants in `cell.metadata.i18n`.

```bash
python switch_language.py --language hu
python switch_language.py --language en
python switch_language.py --status
```

The current ZIP defaults to Hungarian. Switching language preserves code cells, execution counts, tables and all embedded PNG outputs.
