"""Base contract for analysis agents; independent of any LLM provider."""
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    name = "base_agent"

    @abstractmethod
    def interpret(self, results: dict) -> dict:
        """Interpret structured analysis results and return evidence-based findings."""
        raise NotImplementedError
