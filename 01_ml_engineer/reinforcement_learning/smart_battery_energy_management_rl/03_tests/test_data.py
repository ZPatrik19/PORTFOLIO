from battery_rl.data import generate_synthetic_energy_data, temporal_split

def test_data_has_required_columns():
    df=generate_synthetic_energy_data(days=3,seed=1)
    required={"electricity_price_eur_per_kwh","electricity_demand_kw","renewable_generation_kw","demand_forecast_1h_kw","renewable_forecast_1h_kw"}
    assert required.issubset(df.columns)
    assert len(df)==72

def test_temporal_split_preserves_order():
    df=generate_synthetic_energy_data(days=10,seed=1)
    tr,va,te=temporal_split(df,.7,.15)
    assert tr.timestamp.max() < va.timestamp.min() < te.timestamp.min()
