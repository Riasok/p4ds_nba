
import json
import numpy as np
from typing import Dict, Any
from langchain.tools import Tool
from .base import BaseTool

class CasterDBTool(BaseTool):
    """Tool for accessing basketball caster speaking patterns"""
    
    def __init__(self, db_path: str = "data/castors.json"):
        super().__init__(
            name="caster_db",
            description="Accesses basketball caster speaking patterns to generate authentic commentary style for responses."
        )
        self.db_path = db_path
        self._load_data()
    
    def _load_data(self) -> None:
        """Load caster database"""
        try:
            with open(self.db_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        except FileNotFoundError:
            self.data = {
                "speaking_patterns": {
                    "excitement": ["Absolutely!", "That's a great point!", "Here's what's fascinating:"],
                    "analysis": ["The model flagged", "Here's the specific data point", "Think of it as"],
                    "transitions": ["However,", "But here's the thing:", "That gets right to the heart of"]
                },
                "phrases": {
                    "probability_intro": ["Win Probability:", "Here's what the numbers say:"],
                    "factor_intro": ["Key factors:", "The model is weighing:"]
                }
            }
    
    def run(self, query: str) -> str:
        """Execute caster tool"""
        try:
            if "prediction" in query.lower() or "probability" in query.lower():
                intro = np.random.choice(self.data["phrases"]["probability_intro"])
            else:
                intro = np.random.choice(self.data["speaking_patterns"]["excitement"])
            
            return f"{intro} Based on the caster patterns, here is a framework for the analysis."
        except Exception as e:
            return self._handle_error(e)

def create_caster_db_tool(db_path: str = "data/castors.json") -> Tool:
    """Create LangChain tool wrapper for caster database"""
    caster_tool = CasterDBTool(db_path)
    return Tool(
        name=caster_tool.name,
        description=caster_tool.description,
        func=caster_tool.run
    )
