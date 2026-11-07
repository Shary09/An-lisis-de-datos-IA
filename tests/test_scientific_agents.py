"""Agent separation, scientific criteria, API sanitization and fallback tests."""
from types import SimpleNamespace
import pytest
from framework_ia.agents.agent_orchestrator import AgentOrchestrator
from framework_ia.agents.regression_agent import RegressionAgent
from framework_ia.agents.classification_agent import ClassificationAgent
from framework_ia.services.openai_interpretation_service import OpenAIInterpretationService


class FakeResponses:
    def __init__(self):
        self.calls = []
    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text="Selección: modelo respaldado por métricas.")


def test_regression_selection_prefers_cv_not_test():
    evidence = {"models": [
        {"model": "A", "cv_rmse_mean": 24, "RMSE": 1},
        {"model": "B", "cv_rmse_mean": 20, "RMSE": 100},
    ]}
    result = RegressionAgent().interpret(evidence)
    assert result["selection"]["candidate"] == "B"
    assert result["selection"]["scope"] == "cross_validation"
    assert result["mode"] == "deterministic"


def test_regression_test_only_is_not_definitive_selection():
    output = RegressionAgent().interpret({"models": [{"model":"A", "RMSE":4.0}]})
    assert output["selection"]["scope"] == "test_exploratory_only"
    assert "no usar" in output["selection"]["warning"].lower()


def test_classification_selection_uses_cv():
    output = ClassificationAgent().interpret({"models": [
        {"model":"A", "f1_macro_cv":.8, "test_f1_macro":.9},
        {"model":"B", "f1_macro_cv":.85, "test_f1_macro":.6},
    ]})
    assert output["selection"]["candidate"] == "B"


def test_all_agents_router_exposes_scientific_roles():
    o = AgentOrchestrator()
    assert o.run_one("eda", {"shape":{"rows":3,"columns":2}})["agent"] == "eda_agent"
    assert o.run_one("clustering", {"cluster_sizes":{"0":2,"1":1}})["agent"] == "clustering_agent"
    with pytest.raises(ValueError): o.run_one("unknown", {})


def test_api_uses_responses_and_bounded_aggregate_payload():
    responses = FakeResponses()
    fake_client = SimpleNamespace(responses=responses)
    service = OpenAIInterpretationService(client=fake_client, model="test-model")
    result = service.generate("regression_agent", {"models":[{"model":"A", "RMSE":1.25}]},
                              {"candidate":"A"})
    assert "Selección" in result
    call = responses.calls[0]
    assert call["store"] is False
    assert "1.25" in call["input"]
    assert "predicciones" in call["instructions"]
    assert call["model"] == "test-model"


def test_api_failures_fall_back_to_deterministic():
    class Broken:
        def generate(self, *args):
            from framework_ia.services.openai_interpretation_service import OpenAIInterpretationError
            raise OpenAIInterpretationError("API unavailable")
    output = RegressionAgent(service=Broken()).interpret({"models":[{"model":"A","RMSE":2}]})
    assert output["mode"] == "deterministic"
    assert output["api_warning"] == "API unavailable"


def test_no_api_key_has_friendly_error():
    from framework_ia.services.openai_interpretation_service import OpenAIInterpretationError
    service = OpenAIInterpretationService(api_key=None)
    with pytest.raises(OpenAIInterpretationError):
        service.generate("eda", {"shape":{"rows":2}}, {})
