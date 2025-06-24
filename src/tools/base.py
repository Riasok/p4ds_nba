import json
from typing import Any, Dict
from abc import ABC, abstractmethod

class BaseTool(ABC):
    """Base class for all tools"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def run(self, query: str) -> str:
        """Execute the tool with given query"""
        pass
    
    def _format_response(self, data: Dict[str, Any]) -> str:
        """Format response data as JSON string"""
        return json.dumps(data, indent=2)
    
    def _handle_error(self, error: Exception) -> str:
        """Handle and format errors"""
        return f"{self.name} error: {str(error)}"
