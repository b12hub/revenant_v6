from abc import ABC, abstractmethod
from typing import Any, Dict


class RevenantAgentBase(ABC):
    """Base class for all Revenant agents. Provides common interface and error handling."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.metadata: Dict[str, Any] = {}

    @abstractmethod
    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's logic.

        Args:
            input_data: Dictionary containing input parameters

        Returns:
            Dictionary with keys: agent, status, summary, data
        """
        pass

    async def setup(self):
        """Optional setup method called during agent initialization."""
        pass

    async def on_error(self, error: Exception) -> Dict[str, Any]:
        """
        Handle errors that occur during agent execution.

        Args:
            error: The exception that was raised

        Returns:
            Dictionary with error information
        """
        return {
            "agent": self.name,
            "status": "error",
            "summary": f"Error in {self.name}: {str(error)}",
            "data": {
                "error_type": type(error).__name__,
                "error_message": str(error)
            }
        }

    def info(self) -> Dict[str, Any]:
        """Get agent information including metadata."""
        return {
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata
        }