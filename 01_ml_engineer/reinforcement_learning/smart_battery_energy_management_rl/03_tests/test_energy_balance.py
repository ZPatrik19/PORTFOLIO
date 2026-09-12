import numpy as np
from battery_rl.data import generate_synthetic_energy_data
from battery_rl.environment import BatteryEnvironment

BATTERY={"capacity_kwh":100.0,"initial_soc":0.5,"min_soc":0.1,"max_soc":0.9,"max_charge_kw":20.0,"max_discharge_kw":20.0,"charging_efficiency":0.9,"discharging_efficiency":0.8,"timestep_hours":1.0,"feed_in_tariff_eur_per_kwh":0.04}
REWARD={"cost_weight":1.0,"peak_penalty_eur_per_kw2":0.0,"degradation_eur_per_kwh_throughput":0.0,"constraint_penalty_eur_per_kwh":0.0}

def test_energy_balance_identity():
    env=BatteryEnvironment(generate_synthetic_energy_data(days=2),BATTERY,REWARD,episode_hours=24,random_start=False)
    env.reset(seed=0); *_,info=env.step(0)
    expected=info["demand_kw"]-info["renewable_kw"]+info["charge_kw"]-info["discharge_kw"]
    assert np.isclose(info["grid_power_kw"],expected)

def test_efficiency_reduces_stored_energy_gain():
    env=BatteryEnvironment(generate_synthetic_energy_data(days=2),BATTERY,REWARD,episode_hours=24,random_start=False)
    env.reset(seed=0); soc0=env.soc; *_,info=env.step(0)
    stored_gain=(env.soc-soc0)*BATTERY["capacity_kwh"]
    assert np.isclose(stored_gain, info["charge_kw"]*BATTERY["charging_efficiency"], atol=1e-6)
