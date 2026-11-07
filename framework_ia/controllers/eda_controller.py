"""Controller for EDA requests."""
from framework_ia.models.eda.eda_analysis import analyze_dataframe
from framework_ia.agents.eda_agent import EDAAgent


class EDAController:
    def __init__(self, agent=None):
        self.agent = agent or EDAAgent()

    def run(self, dataframe):
        analysis = analyze_dataframe(dataframe)
        interpretation = self.agent.interpret(analysis)
        return {"analysis": analysis, "interpretation": interpretation}
