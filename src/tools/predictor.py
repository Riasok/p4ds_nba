import json
import numpy as np
from typing import Dict, Any, List
from langchain.tools import Tool
from .base import BaseTool

class MockPredictorTool(BaseTool):
    """Mock prediction tool for demonstration purposes"""
    
    def __init__(self):
        super().__init__(
            name="mock_predictor_model",
            description="Generates a mock NBA game prediction for demonstration. Provides simulated win probabilities and key factors. Use when a user asks for a prediction."
        )
    
    def run(self, query: str) -> str:
        """Execute mock prediction"""
        try:
            from nba_api.stats.static import teams
            
            nba_teams = teams.get_teams()
            query_lower = query.lower()
            found_teams = []

            for team in nba_teams:
                if team['full_name'].lower() in query_lower or team['nickname'].lower() in query_lower:
                    if team['full_name'] not in found_teams:
                        found_teams.append(team['full_name'])
            
            if len(found_teams) < 2:
                return "Please specify at least two teams for a prediction (e.g., 'Predict Celtics vs Mavericks')."
            
            home_prob = np.random.uniform(0.40, 0.70)
            away_prob = 1 - home_prob
            
            prediction_result = {
                "prediction_type": "Mock Demonstration",
                "home_team": found_teams[0],
                "away_team": found_teams[1],
                "home_win_probability": round(home_prob, 3),
                "away_win_probability": round(away_prob, 3),
                "key_factors": [
                    "Simulated home court advantage",
                    "Randomized defensive efficiency", 
                    "Simulated shooting variance"
                ],
                "prediction_confidence": round(np.random.uniform(0.65, 0.90), 2),
                "model_version": "Mock_Predictor_v1.0"
            }
            
            return self._format_response(prediction_result)
            
        except Exception as e:
            return self._handle_error(e)

def create_mock_predictor_model_tool() -> Tool:
    """Create LangChain tool wrapper for mock predictor"""
    predictor_tool = MockPredictorTool()
    return Tool(
        name=predictor_tool.name,
        description=predictor_tool.description,
        func=predictor_tool.run
    )