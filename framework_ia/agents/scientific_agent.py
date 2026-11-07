"""Data scientist reasoning agents. These modules never fit models or predict."""
from framework_ia.agents.base_agent import BaseAgent
from framework_ia.services.openai_interpretation_service import OpenAIInterpretationService, OpenAIInterpretationError


class ScientificAgent(BaseAgent):
    name = "scientific_agent"
    def __init__(self, service=None):
        self.service = service

    def select(self, evidence):
        """Deterministic candidate selection; no model execution."""
        return {"criterion": "descriptive evidence only", "candidate": None}

    def interpret(self, results):
        choice = self.select(results)
        notes = self._fallback(results, choice)
        output = {"agent": self.name, "mode": "deterministic", "selection": choice,
                  "findings": notes, "recommendations": ["Revisar las limitaciones antes de concluir."],
                  "limitations": ["La interpretación descriptiva no prueba causalidad ni superioridad estadística."]}
        if self.service is not None:
            try:
                output["scientific_report"] = self.service.generate(self.name, results, choice)
                output["mode"] = "openai"
            except OpenAIInterpretationError as exc:
                output["api_warning"] = str(exc)
        return output

    def _fallback(self, evidence, selection):
        return ["No hay interpretación específica disponible para los resultados recibidos."]


class RegressionScientificAgent(ScientificAgent):
    name = "regression_agent"
    def select(self, evidence):
        rows = [r for r in evidence.get("models", []) if isinstance(r, dict)]
        metric = "cv_rmse_mean" if any(_numeric(r.get("cv_rmse_mean")) is not None for r in rows) else None
        if metric is None:
            metric = "RMSE" if any(_numeric(r.get("RMSE")) is not None for r in rows) else None
        if not metric:
            return {"candidate": None, "criterion": "insufficient metrics", "warning": "No evaluable RMSE"}
        valid = [r for r in rows if _numeric(r.get(metric)) is not None]
        best = min(valid, key=lambda r: float(r[metric]))
        return {"candidate": str(best.get("model", "unknown")), "criterion": metric,
                "value": float(best[metric]),
                "scope": "cross_validation" if metric == "cv_rmse_mean" else "test_exploratory_only",
                "warning": None if metric == "cv_rmse_mean" else
                "Solo hay métricas de prueba: no usar este ranking para selección definitiva sin CV."}

    def _fallback(self, evidence, selection):
        if not selection["candidate"]:
            return ["No hay métricas RMSE válidas para comparar modelos."]
        return [f"Menor {selection['criterion']}: {selection['candidate']} ({selection['value']:.4f}).",
                selection.get("warning") or "Candidato seleccionado en validación cruzada; revisar prueba y residuos."]


class ClassificationScientificAgent(ScientificAgent):
    name = "classification_agent"
    def select(self, evidence):
        rows = [r for r in evidence.get("models", []) if isinstance(r, dict)]
        # Prefer the experiment's predefined validation metric, never test F1 to select.
        metric = next((k for k in ("f1_macro_cv", "F1 macro CV") if any(_numeric(r.get(k)) is not None for r in rows)), None)
        if not metric:
            return {"candidate": None, "criterion": "insufficient cross validation data"}
        valid = [r for r in rows if _numeric(r.get(metric)) is not None]
        best = max(valid, key=lambda r: float(r[metric]))
        return {"candidate": str(best.get("model", "unknown")), "criterion": metric,
                "value": float(best[metric]), "scope": "cross_validation"}

    def _fallback(self, evidence, selection):
        if not selection["candidate"]:
            return ["No se proporcionó F1 macro de validación cruzada para seleccionar un modelo."]
        return [f"Mayor {selection['criterion']}: {selection['candidate']} ({selection['value']:.4f}).",
                "Contrastar precisión, recall y matrices de confusión en prueba."]


class EDAScientificAgent(ScientificAgent):
    name = "eda_agent"
    def _fallback(self, evidence, selection):
        shape = evidence.get("shape", {})
        return [f"Dataset: {shape.get('rows', '?')} filas y {shape.get('columns', '?')} columnas.",
                "Analizar faltantes, duplicados y distribuciones antes de modelar."]


class ClusteringScientificAgent(ScientificAgent):
    name = "clustering_agent"
    def _fallback(self, evidence, selection):
        return ["Interpretar métricas de cohesión y separación, tamaños de grupos y sensibilidad a parámetros.",
                "Sin métricas de validación no puede justificarse un agrupamiento óptimo."]


def _numeric(v):
    try:
        import math
        x = float(v)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None
