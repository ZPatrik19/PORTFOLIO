from travel_agent.quality import audit_all
from travel_agent.training import model_status


def test_data_quality_gates_pass():
    report = audit_all(write_outputs=False)
    assert all(report["quality_gates"].values())
    assert report["split_leakage"]["test__train"]["shared_patterns"] == 0


def test_router_model_matches_current_dataset():
    status = model_status()
    assert status["exists"]
    assert not status["stale"]
