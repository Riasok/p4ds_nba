import os
import json
import yaml
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import requests
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from pydantic import BaseModel, Field

# LangChain imports
from langchain.tools import BaseTool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish

# =====================================
# SERVER DEPLOYMENT CONFIGURATION
# =====================================
# Run command: streamlit run nba_chat_langchain.py --server.address=0.0.0.0 --server.port=8501
# 
# Server setup commands:
# sudo ufw allow 8501          # Open port 8501
# sudo ufw delete allow 8501   # Close port 8501 (if needed)
# 
# Access URL: http://YOUR_SERVER_IP:8501
# Example: http://147.47.236.52:8501
#
# Alternative ports: 8502, 8503, etc. (update commands accordingly)
# =====================================

# Environment Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "your-openai-api-key-here"
NBA_API_KEY = os.getenv("NBA_API_KEY") or "your-nba-api-key-here"  # Optional
SERVER_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))
SERVER_ADDRESS = os.getenv("STREAMLIT_ADDRESS", "0.0.0.0")

# Tool Models
class CasterResponse(BaseModel):
    """Response model for caster speaking style"""
    response: str = Field(description="Caster-style response to the query")
    confidence: float = Field(description="Confidence in the response style")

class NBAGameData(BaseModel):
    """NBA game data model"""
    game_id: str
    home_team: str
    away_team: str
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    game_date: str
    status: str
    stats: Optional[Dict[str, Any]] = None

class PredictionResult(BaseModel):
    """Prediction model result"""
    home_win_probability: float = Field(description="Home team win probability")
    away_win_probability: float = Field(description="Away team win probability")
    key_factors: List[str] = Field(description="Key factors influencing prediction")
    prediction_confidence: float = Field(description="Model confidence score")

# Custom Tools
class CasterDBTool(BaseTool):
    """Tool for accessing caster speaking patterns from JSON database"""
    
    name = "caster_db"
    description = "Access basketball caster speaking patterns and examples for generating authentic commentary style responses"
    
    def __init__(self, db_path: str = "caster_db.json"):
        super().__init__()
        self.db_path = db_path
        self.caster_data = self._load_caster_db()
    
    def _load_caster_db(self) -> Dict[str, Any]:
        """Load caster database from JSON file"""
        try:
            with open(self.db_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            # Fallback sample data
            return {
                "speaking_patterns": {
                    "excitement": ["Absolutely!", "That's a great point!", "Here's what's fascinating:"],
                    "analysis": ["The model flagged", "Here's the specific data point", "Think of it as"],
                    "transitions": ["However,", "But here's the thing:", "That gets right to the heart of"],
                    "explanations": ["It's not just about", "Essentially,", "Here's why that matters:"]
                },
                "phrases": {
                    "probability_intro": ["Win Probability:", "Here's what the numbers say:"],
                    "factor_intro": ["Key factors:", "The model is weighing:"],
                    "statistical_emphasis": ["The data shows", "Historically", "In this matchup"]
                }
            }
    
    def _run(self, query: str) -> str:
        """Generate caster-style response based on query"""
        # Simple pattern matching for demo - you can enhance this
        response_parts = []
        
        if "prediction" in query.lower() or "probability" in query.lower():
            response_parts.extend(self.caster_data["phrases"]["probability_intro"])
        
        if "why" in query.lower() or "how" in query.lower():
            response_parts.extend(self.caster_data["speaking_patterns"]["explanations"])
        
        # Build response with caster style
        base_response = f"{np.random.choice(self.caster_data['speaking_patterns']['excitement'])} "
        
        return base_response + "Based on the caster patterns, here's the analysis you're looking for."

class NBADBTool(BaseTool):
    """Tool for accessing NBA API data"""
    
    name = "nba_db"
    description = "Access live NBA game data, statistics, and team information through API calls"
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self.api_key = api_key or os.getenv("NBA_API_KEY")
        self.base_url = "https://api.sportsdata.io/v3/nba"  # Example API
    
    def _get_headers(self) -> Dict[str, str]:
        """Get API headers"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Ocp-Apim-Subscription-Key"] = self.api_key
        return headers
    
    def _run(self, query: str) -> str:
        """Fetch NBA data based on query"""
        try:
            # Parse query to determine what data to fetch
            if "game" in query.lower() and "today" in query.lower():
                return self._get_todays_games()
            elif "team stats" in query.lower():
                return self._get_team_stats(query)
            elif any(team in query.lower() for team in ["pacers", "thunder", "okc", "ind"]):
                return self._get_matchup_data(query)
            else:
                return "Please specify what NBA data you need (games, team stats, matchup info)"
        except Exception as e:
            return f"Error fetching NBA data: {str(e)}"
    
    def _get_todays_games(self) -> str:
        """Get today's games - mock implementation"""
        # Mock data for demo
        games_data = {
            "games": [
                {
                    "game_id": "OKC_vs_IND_G2",
                    "home_team": "Oklahoma City Thunder",
                    "away_team": "Indiana Pacers",
                    "status": "scheduled",
                    "game_time": "8:00 PM ET"
                }
            ]
        }
        return json.dumps(games_data, indent=2)
    
    def _get_team_stats(self, query: str) -> str:
        """Get team statistics"""
        # Mock implementation
        stats = {
            "OKC": {
                "defensive_rebound_rate": 0.742,
                "fast_break_points_allowed": 12.4,
                "clutch_offensive_efficiency": 1.18
            },
            "IND": {
                "fast_break_points": 18.6,
                "three_point_percentage": 0.381,
                "offensive_rebound_rate": 0.285
            }
        }
        return json.dumps(stats, indent=2)
    
    def _get_matchup_data(self, query: str) -> str:
        """Get matchup-specific data"""
        matchup_data = {
            "matchup": "OKC vs IND",
            "game_1_stats": {
                "IND_fast_break_points": 22,
                "IND_three_point_pct": 0.462,
                "OKC_defensive_rebounds": 34,
                "IND_offensive_rebounds": 12
            },
            "historical_h2h": {
                "games_played": 5,
                "okc_wins": 3,
                "ind_wins": 2
            }
        }
        return json.dumps(matchup_data, indent=2)

class PredictorModelTool(BaseTool):
    """Tool for XGBoost-based NBA game predictions"""
    
    name = "predictor_model"
    description = "Generate NBA game predictions using XGBoost model with win probabilities and key factors"
    
    def __init__(self, model_path: str = "nba_model.pkl"):
        super().__init__()
        self.model_path = model_path
        self.model = self._load_model()
        self.feature_names = self._get_feature_names()
    
    def _load_model(self):
        """Load the XGBoost model"""
        try:
            return joblib.load(self.model_path)
        except FileNotFoundError:
            # Mock model for demo
            return None
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names for the model"""
        return [
            "home_defensive_rebound_rate",
            "away_fast_break_points",
            "home_clutch_efficiency",
            "away_three_point_pct",
            "home_recent_form",
            "away_recent_form",
            "pace_differential",
            "defensive_rating_diff"
        ]
    
    def _run(self, query: str) -> str:
        """Generate prediction based on query"""
        try:
            # Extract teams from query
            teams = self._extract_teams(query)
            if not teams:
                return "Please specify the teams for prediction (e.g., 'OKC vs IND')"
            
            # Get prediction
            prediction_result = self._predict_game(teams)
            return json.dumps(prediction_result, indent=2)
            
        except Exception as e:
            return f"Error generating prediction: {str(e)}"
    
    def _extract_teams(self, query: str) -> Optional[Dict[str, str]]:
        """Extract team names from query"""
        team_mappings = {
            "okc": "Oklahoma City Thunder",
            "thunder": "Oklahoma City Thunder",
            "ind": "Indiana Pacers",
            "pacers": "Indiana Pacers"
        }
        
        found_teams = []
        for key, full_name in team_mappings.items():
            if key in query.lower():
                found_teams.append(full_name)
        
        if len(found_teams) >= 2:
            return {"home": found_teams[0], "away": found_teams[1]}
        return None
    
    def _predict_game(self, teams: Dict[str, str]) -> Dict[str, Any]:
        """Generate mock prediction"""
        # Mock prediction logic
        if self.model is None:
            # Generate mock prediction
            home_prob = np.random.uniform(0.45, 0.65)
            away_prob = 1 - home_prob
            
            return {
                "home_team": teams["home"],
                "away_team": teams["away"],
                "home_win_probability": round(home_prob, 3),
                "away_win_probability": round(away_prob, 3),
                "key_factors": [
                    "Home court advantage",
                    "Defensive rebounding efficiency",
                    "Three-point shooting variance",
                    "Pace of play differential"
                ],
                "prediction_confidence": 0.78,
                "model_version": "XGBoost_v2.1"
            }
        
        # If real model exists, use it here
        # features = self._prepare_features(teams)
        # prediction = self.model.predict_proba(features)
        # return self._format_prediction(prediction, teams)

# Streamlit Callback Handler
class StreamlitCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for Streamlit integration"""
    
    def __init__(self):
        self.steps = []
    
    def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        """Called when agent takes an action"""
        self.steps.append({
            "type": "action",
            "tool": action.tool,
            "input": action.tool_input,
            "log": action.log
        })
    
    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
        """Called when agent finishes"""
        self.steps.append({
            "type": "finish",
            "output": finish.return_values
        })

# Main Application Class
class NBAAnalysisAgent:
    """Main NBA Analysis Agent with LangChain integration"""
    
    def __init__(self, openai_api_key: str, prompts_config: Dict[str, str]):
        self.llm = ChatOpenAI(
            temperature=0.7,
            model="gpt-4o-mini",
            openai_api_key=openai_api_key
        )
        
        self.prompts = prompts_config
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize tools
        self.tools = [
            CasterDBTool(),
            NBADBTool(),
            PredictorModelTool()
        ]
        
        # Initialize agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )
        
        # Set system prompt
        self._set_system_prompt()
    
    def _set_system_prompt(self):
        """Set the system prompt for the agent"""
        system_message = SystemMessage(content=self.prompts.get("system_prompt", ""))
        self.memory.chat_memory.add_message(system_message)
    
    def process_query(self, query: str, callback_handler: Optional[StreamlitCallbackHandler] = None) -> Dict[str, Any]:
        """Process user query and return response"""
        try:
            callbacks = [callback_handler] if callback_handler else []
            
            # Run the agent
            response = self.agent.run(input=query, callbacks=callbacks)
            
            return {
                "response": response,
                "success": True,
                "steps": callback_handler.steps if callback_handler else []
            }
        
        except Exception as e:
            return {
                "response": f"I encountered an error: {str(e)}",
                "success": False,
                "error": str(e)
            }

# Streamlit UI Functions
def load_prompts() -> Dict[str, str]:
    """Load prompts from YAML file"""
    try:
        with open('prompts.yaml', 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # Fallback prompts
        return {
            "system_prompt": """You are GPT CourtVision, an expert NBA analyst and caster. You have access to three tools:

1. caster_db - For authentic basketball caster speaking patterns
2. nba_db - For live NBA data and statistics  
3. predictor_model - For XGBoost-based game predictions

When responding:
- Use the caster_db to maintain authentic commentary style
- Pull relevant stats from nba_db to support your analysis
- Generate predictions using the predictor_model when asked
- Be conversational and engaging like a real NBA analyst
- Focus on key factors that influence game outcomes
- Explain complex concepts in an accessible way

Always cite your sources and explain your reasoning."""
        }

def render_message(role: str, content: str):
    """Render chat message with styling"""
    if role == "user":
        st.markdown(f"""
        <div style="
            background-color:#0B93F6; color:white; padding:12px 18px; 
            border-radius:20px 20px 0 20px; max-width:70%; margin-left:auto; margin-bottom:10px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            box-shadow: 1px 1px 3px rgba(0,0,0,0.2);
            ">
            {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="
            background-color:#E5E5EA; color:#000; padding:12px 18px; 
            border-radius:20px 20px 20px 0; max-width:70%; margin-right:auto; margin-bottom:10px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            box-shadow: 1px 1px 3px rgba(0,0,0,0.1);
            ">
            {content}
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="GPT CourtVision: NBA Analysis Agent", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Server info display (only show in development)
    if os.getenv("STREAMLIT_ENV") == "development":
        st.info(f"🚀 Server running on: http://{SERVER_ADDRESS}:{SERVER_PORT}")
    
    # Sidebar configuration
    with st.sidebar:
        st.title("🏀 Configuration")
        
        # Server info section
        with st.expander("🖥️ Server Info"):
            st.code(f"""
# Current Configuration
Port: {SERVER_PORT}
Address: {SERVER_ADDRESS}

# Run Command:
streamlit run nba_chat_langchain.py --server.address={SERVER_ADDRESS} --server.port={SERVER_PORT}

# Server Commands:
sudo ufw allow {SERVER_PORT}
sudo ufw delete allow {SERVER_PORT}
            """)
        
        # API Key input
        openai_key = st.text_input(
            "OpenAI API Key", 
            type="password", 
            help="Enter your OpenAI API key",
            value=OPENAI_API_KEY if OPENAI_API_KEY != "your-openai-api-key-here" else ""
        )
        
        nba_key = st.text_input(
            "NBA API Key (Optional)", 
            type="password", 
            help="Enter your NBA API key for live data"
        )
        
        if not openai_key:
            openai_key = OPENAI_API_KEY
        
        # Model settings
        st.subheader("Model Settings")
        show_tool_calls = st.checkbox("Show Tool Calls", value=False)
        show_reasoning = st.checkbox("Show Agent Reasoning", value=False)
    
    # Main interface
    st.title("🏀 GPT CourtVision: NBA Analysis Agent")
    st.caption("Powered by LangChain Tools: CasterDB + NBA API + XGBoost Predictions")
    
    if not openai_key:
        st.warning("Please provide your OpenAI API key in the sidebar or set the OPENAI_API_KEY environment variable.")
        return
    
    # Initialize session state
    if "agent" not in st.session_state:
        prompts = load_prompts()
        st.session_state.agent = NBAAnalysisAgent(openai_key, prompts)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    st.subheader("Conversation")
    for message in st.session_state.messages:
        render_message(message["role"], message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about NBA games, predictions, or analysis..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        render_message("user", prompt)
        
        # Process with agent
        with st.spinner("Analyzing..."):
            callback_handler = StreamlitCallbackHandler()
            result = st.session_state.agent.process_query(prompt, callback_handler)
            
            if result["success"]:
                # Add assistant response
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": result["response"]
                })
                render_message("assistant", result["response"])
                
                # Show tool calls if enabled
                if show_tool_calls and result["steps"]:
                    with st.expander("🔧 Tool Calls"):
                        for step in result["steps"]:
                            if step["type"] == "action":
                                st.code(f"Tool: {step['tool']}\nInput: {step['input']}")
                
                # Show reasoning if enabled
                if show_reasoning and result["steps"]:
                    with st.expander("🧠 Agent Reasoning"):
                        for step in result["steps"]:
                            if "log" in step:
                                st.text(step["log"])
            else:
                st.error(f"Error: {result['error']}")
        
        st.experimental_rerun()

if __name__ == "__main__":
    # Environment setup
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
    if NBA_API_KEY != "your-nba-api-key-here":
        os.environ['NBA_API_KEY'] = NBA_API_KEY
    
    # Set development flag for debugging
    if not os.getenv("STREAMLIT_ENV"):
        os.environ["STREAMLIT_ENV"] = "development"
    
    # Port configuration for server deployment
    print(f"""
    =====================================
    🏀 NBA Analysis Agent Starting...
    =====================================
    
    🚀 Server Configuration:
    - Address: {SERVER_ADDRESS}
    - Port: {SERVER_PORT}
    
    📡 Run Command:
    streamlit run nba_chat_langchain.py --server.address={SERVER_ADDRESS} --server.port={SERVER_PORT}
    
    🔧 Server Commands:
    - Open port:  sudo ufw allow {SERVER_PORT}
    - Close port: sudo ufw delete allow {SERVER_PORT}
    
    🌐 Access URL:
    - Local: http://localhost:{SERVER_PORT}
    - Server: http://YOUR_SERVER_IP:{SERVER_PORT}
    
    =====================================
    """)
    
    main()