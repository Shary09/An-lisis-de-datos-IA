"""Thin coordinator for specialized agents; no predictive algorithms."""
from framework_ia.agents.eda_agent import EDAAgent
from framework_ia.agents.regression_agent import RegressionAgent
from framework_ia.agents.classification_agent import ClassificationAgent
from framework_ia.agents.clustering_agent import ClusteringAgent

AGENT_TYPES = {"eda": EDAAgent, "regression": RegressionAgent,
               "classification": ClassificationAgent, "clustering": ClusteringAgent}

class AgentOrchestrator:
    def __init__(self, agents=None):
        self.agents = list(agents or [])

    def run(self, results):
        return {agent.name: agent.interpret(results) for agent in self.agents}

    def run_one(self, kind, evidence, service=None):
        if kind not in AGENT_TYPES:
            raise ValueError(f"Unsupported analysis type: {kind}")
        return AGENT_TYPES[kind](service=service).interpret(evidence)
