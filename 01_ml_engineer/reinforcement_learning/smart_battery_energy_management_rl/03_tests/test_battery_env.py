import pandas as pd
import numpy as np
from battery_rl.data import generate_synthetic_energy_data
from battery_rl.environment import BatteryEnvironment

BATTERY={"capacity_kwh":100.0,"initial_soc":0.5,"min_soc":0.1,"max_soc":0.9,"max_charge_kw":25.0,"max_discharge_kw":25.0,"charging_efficiency":0.95,"discharging_efficiency":0.95,"timestep_hours":1.0,"feed_in_tariff_eur_per_kwh":0.04}
REWARD={"cost_weight":1.0,"peak_penalty_eur_per_kw2":0.01,"degradation_eur_per_kwh_throughput":0.01,"constraint_penalty_eur_per_kwh":0.2}

def make_env():
    return BatteryEnvironment(generate_synthetic_energy_data(days=3),BATTERY,REWARD,episode_hours=24,random_start=False,normalize_observation=True)

def test_soc_never_exceeds_bounds_when_repeatedly_charging():
    env=make_env(); env.reset(seed=1)
    for _ in range(24):
        _,_,t,tr,_=env.step(0)
        assert BATTERY["min_soc"]-1e-9 <= env.soc <= BATTERY["max_soc"]+1e-9
        if t or tr: break

def test_requested_action_can_differ_from_executed_action():
    env=make_env(); env.reset(seed=1,options={"initial_soc":0.899})
    *_,info=env.step(0)
    assert info["requested_battery_power_kw"] > info["executed_battery_power_kw"]
    assert info["constraint_violation"] == 1

def test_continuous_action_sign_convention():
    env=BatteryEnvironment(generate_synthetic_energy_data(days=3),BATTERY,REWARD,episode_hours=24,continuous_action=True,random_start=False)
    env.reset(seed=1)
    *_,info=env.step(np.array([1.0],dtype=np.float32))
    assert info["executed_battery_power_kw"] > 0
