# Engineering notes

## Core physical convention

`executed_battery_power_kw > 0` means charging from the AC bus.  
`executed_battery_power_kw < 0` means discharging to the AC bus.

Charging changes stored energy by

`ΔE = P_charge × η_charge × Δt`

while discharging changes stored energy by

`ΔE = -P_discharge / η_discharge × Δt`.

This avoids the common modelling error of applying efficiency in the wrong direction.

## Energy balance

`grid_power = demand - renewable + charge - discharge`

Positive grid power is import; negative grid power is export. Imports are paid at the time-varying electricity price, while exports receive a lower feed-in tariff.

## Why requested vs executed action matters

An agent may ask for 30 kW charging while the battery is already near `max_soc`. The environment clips this to the physically feasible power. We store both values so we can measure how often the policy tries to violate constraints without ever allowing an impossible battery state.

## Reward design warning

The main reward combines electricity cost, peak-demand penalty, degradation proxy and constraint-attempt penalty. Each component is logged separately. This is important because an RL agent optimizes the numerical reward literally; a badly shaped reward can create undesirable cycling, peak shifting or constraint-hitting behaviour.

## Leakage policy

Train/validation/test are chronological. Forecast features are synthetic *decision-time forecasts*, generated from calendar expectations plus forecast noise. They do not directly expose the future realized observation. The project therefore distinguishes a forecast feature from perfect-information baselines that intentionally use future prices as an offline upper-reference.
