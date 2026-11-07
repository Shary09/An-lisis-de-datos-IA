"""Controller that exposes agent analysis to the presentation layer."""
class AgentController:
    def __init__(self, agent):
        self.agent = agent

    def interpret(self, results: dict) -> dict:
        return self.agent.interpret(results)
