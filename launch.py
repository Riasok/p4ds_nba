import os
import json
import yaml
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import requests
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

try:
    from pydantic.v1 import BaseModel, Field
except ImportError:
    from pydantic import BaseModel, Field

# For OpenAI v0.28 compatibility
import openai

try:
    # Try modern imports first
    try:
        from langchain.tools import Tool
        from langchain.agents import initialize_agent, AgentType
        from langchain.memory import ConversationBufferMemory
        from langchain.schema import HumanMessage, SystemMessage, AgentAction, AgentFinish
        from langchain_openai import ChatOpenAI
        from langchain.callbacks.base import BaseCallbackHandler
        LANGCHAIN_VERSION = "modern"
    except ImportError:
        # Fallback to older imports
        from langchain.tools import Tool
        from langchain.agents import initialize_agent, AgentType
        from langchain.memory import ConversationBufferMemory
        from langchain.schema import HumanMessage, SystemMessage, AgentAction, AgentFinish
        from langchain.chat_models import ChatOpenAI
        from langchain.callbacks.base import BaseCallbackHandler
        LANGCHAIN_VERSION = "legacy"
        
except ImportError as e:
    st.error(f"LangChain import error: {e}")
    st.info("Installing compatible versions: pip install langchain==0.1.0 langchain-openai==0.0.5 openai==0.28")
    st.stop()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def create_caster_db_tool(db_path: str = "data/castors.json"):
    
    def load_caster_db() -> Dict[str, Any]:
        try:
            with open(db_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
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
    
    caster_data = load_caster_db()
    
    def run_caster_tool(query: str) -> str:
        try:
            if "prediction" in query.lower() or "probability" in query.lower():
                intro = np.random.choice(caster_data["phrases"]["probability_intro"])
            else:
                intro = np.random.choice(caster_data["speaking_patterns"]["excitement"])
            return f"{intro} Based on the caster patterns, here is a framework for the analysis."
        except Exception as e:
            return f"Caster DB tool error: {str(e)}"
    
    return Tool(
        name="caster_db",
        description="Accesses basketball caster speaking patterns to generate authentic commentary style for responses.",
        func=run_caster_tool
    )

def create_nba_db_tool():
    
    def run_nba_tool(query: str) -> str:
        try:
            from nba_api.live.nba.endpoints import scoreboard
            from nba_api.stats.endpoints import teamgamelog, playercareerstats, leaguegamefinder
            from nba_api.stats.static import players, teams
            
            query_lower = query.lower()
            
            if "today" in query_lower and ("games" in query_lower or "scoreboard" in query_lower):
                return _get_todays_games()
            elif "team stats" in query_lower or "team statistics" in query_lower:
                return _get_team_stats(query)
            elif "player" in query_lower:
                return _get_player_stats(query)
            elif "matchup" in query_lower or "vs" in query_lower:
                return _get_matchup_data(query)
            else:
                return _get_general_nba_info(query)
                
        except ImportError:
            return "nba_api package not installed. Please install with: pip install nba_api"
        except Exception as e:
            return f"Error fetching NBA data: {str(e)}"
    
    def _get_todays_games() -> str:
        from nba_api.live.nba.endpoints import scoreboard
        games = scoreboard.ScoreBoard()
        games_data = games.get_dict()
        game_info = []
        if 'scoreboard' in games_data and 'games' in games_data['scoreboard']:
            for game in games_data['scoreboard']['games']:
                game_summary = {
                    "game_id": game.get('gameId', ''),
                    "home_team": game.get('homeTeam', {}).get('teamName', ''),
                    "away_team": game.get('awayTeam', {}).get('teamName', ''),
                    "home_score": game.get('homeTeam', {}).get('score', 0),
                    "away_score": game.get('awayTeam', {}).get('score', 0),
                    "game_status": game.get('gameStatusText', ''),
                }
                game_info.append(game_summary)
        return json.dumps({"todays_games": game_info, "total_games": len(game_info)}, indent=2)
    
    def _get_team_stats(query: str) -> str:
        from nba_api.stats.static import teams
        from nba_api.stats.endpoints import teamgamelog
        nba_teams = teams.get_teams()
        team_id = None
        team_name = None
        query_lower = query.lower()
        for team in nba_teams:
            if (team['full_name'].lower() in query_lower or 
                team['abbreviation'].lower() in query_lower or
                team['nickname'].lower() in query_lower):
                team_name = team['full_name']
                team_id = team['id']
                break
        
        if team_id:
            team_games = teamgamelog.TeamGameLog(team_id=team_id, season='2023-24')
            team_data = team_games.get_data_frames()[0]
            if team_data.empty:
                return json.dumps({"team_name": team_name, "error": "No game data found for the 2023-24 season."})
            
            recent_games = team_data.head(10)
            stats_summary = {
                "team_name": team_name,
                "team_id": team_id,
                "recent_games_count": len(recent_games),
                "avg_points": round(recent_games['PTS'].mean(), 1),
                "avg_rebounds": round(recent_games['REB'].mean(), 1),
                "avg_assists": round(recent_games['AST'].mean(), 1),
                "wins_last_10": len(recent_games[recent_games['WL'] == 'W']),
                "losses_last_10": len(recent_games[recent_games['WL'] == 'L']),
            }
            return json.dumps(stats_summary, indent=2)
        else:
            return "Team not found. Please specify a valid NBA team."
                
    def _get_player_stats(query: str) -> str:
        from nba_api.stats.static import players
        from nba_api.stats.endpoints import playercareerstats
        nba_players = players.get_players()
        query_lower = query.lower()
        player_id = None
        target_player = None

        for player in nba_players:
            if player['full_name'].lower() in query_lower:
                player_id = player['id']
                target_player = player['full_name']
                break
        
        if player_id:
            career = playercareerstats.PlayerCareerStats(player_id=player_id)
            career_data = career.get_data_frames()[0]
            if career_data.empty:
                return json.dumps({"player_name": target_player, "error": "No career data found."})

            recent_season = career_data.iloc[-1]
            player_summary = {
                "player_name": target_player,
                "player_id": player_id,
                "most_recent_season": recent_season['SEASON_ID'],
                "team": recent_season['TEAM_ABBREVIATION'],
                "games_played": int(recent_season['GP']),
                "points_per_game": float(recent_season['PTS']) / int(recent_season['GP']) if recent_season['GP'] > 0 else 0,
                "rebounds_per_game": float(recent_season['REB']) / int(recent_season['GP']) if recent_season['GP'] > 0 else 0,
                "assists_per_game": float(recent_season['AST']) / int(recent_season['GP']) if recent_season['GP'] > 0 else 0,
            }
            return json.dumps(player_summary, indent=2)
        else:
            return "Player not found. Please use the full name of an active player."

    def _get_matchup_data(query: str) -> str:
        matchup_info = {
            "feature": "Team Matchup Analysis",
            "description": "Provides historical head-to-head data between teams.",
            "note": "Specify teams like 'Lakers vs Warriors' for detailed matchup data."
        }
        return json.dumps(matchup_info, indent=2)
            
    def _get_general_nba_info(query: str) -> str:
        from nba_api.stats.static import teams
        nba_teams = teams.get_teams()
        general_info = {
            "nba_api_status": "Connected",
            "total_teams": len(nba_teams),
            "sample_queries": [
                "today's games",
                "Lakers team stats",
                "LeBron James player stats",
                "Thunder vs Pacers matchup"
            ]
        }
        return json.dumps(general_info, indent=2)
    
    return Tool(
        name="nba_db", 
        description="Accesses live NBA game data, team statistics, player stats, and matchup information using the nba_api package.",
        func=run_nba_tool
    )

def create_mock_predictor_model_tool():
    
    def run_mock_predictor_tool(query: str) -> str:
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
            return json.dumps(prediction_result, indent=2)
            
        except Exception as e:
            return f"Error generating mock prediction: {str(e)}"
    
    return Tool(
        name="mock_predictor_model",
        description="Generates a mock NBA game prediction for demonstration. Provides simulated win probabilities and key factors. Use when a user asks for a prediction.",
        func=run_mock_predictor_tool
    )

class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.steps = []
    
    def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        self.steps.append({
            "type": "action",
            "tool": action.tool,
            "input": action.tool_input,
            "log": action.log
        })
    
    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
        self.steps.append({
            "type": "finish",
            "output": finish.return_values
        })

class NBAAnalysisAgent:
    def __init__(self, openai_api_key: str, prompts_config: Dict[str, str], model_name: str = "gpt-4"):
        # Set the API key for OpenAI v0.28
        openai.api_key = openai_api_key
        
        # Create LLM based on LangChain version
        if LANGCHAIN_VERSION == "modern":
            self.llm = ChatOpenAI(
                temperature=0.5,
                model=model_name,
                openai_api_key=openai_api_key,
                max_tokens=1000,
                timeout=60
            )
        else:
            self.llm = ChatOpenAI(
                temperature=0.5,
                model_name=model_name,
                openai_api_key=openai_api_key,
                max_tokens=1000,
                request_timeout=60
            )
        
        self.prompts = prompts_config
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.tools = [
            create_caster_db_tool(),
            create_nba_db_tool(),
            create_mock_predictor_model_tool()
        ]
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
        )
        self._set_system_prompt()
    
    def _set_system_prompt(self):
        system_message = SystemMessage(content=self.prompts.get("system_prompt", "You are a helpful NBA assistant."))
        if not any(isinstance(m, SystemMessage) for m in self.memory.chat_memory.messages):
            self.memory.chat_memory.add_message(system_message)
    
    def process_query(self, query: str, callback_handler: Optional[StreamlitCallbackHandler] = None) -> Dict[str, Any]:
        try:
            callbacks = [callback_handler] if callback_handler else []
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

def load_prompts() -> Dict[str, str]:
    try:
        with open('data/prompts.yaml', 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {"system_prompt": "You are GPT CourtVision, an expert NBA analyst. Your tools include a caster database for authentic commentary, an NBA database for live stats, and a mock prediction model for demonstrations. Your goal is to provide engaging and insightful analysis."}

def render_message(role: str, content: str):
    if role == "user":
        st.markdown(f'<div style="background-color:#0B93F6; color:white; padding:12px 18px; border-radius:20px 20px 0 20px; max-width:70%; margin-left:auto; margin-bottom:10px; font-family: sans-serif; box-shadow: 1px 1px 3px rgba(0,0,0,0.2);">{content}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="background-color:#E5E5EA; color:#000; padding:12px 18px; border-radius:20px 20px 20px 0; max-width:70%; margin-right:auto; margin-bottom:10px; font-family: sans-serif; box-shadow: 1px 1px 3px rgba(0,0,0,0.1);">{content}</div>', unsafe_allow_html=True)

def check_versions():
    """Check versions of key libraries"""
    versions = {}
    try:
        import openai
        versions['openai'] = openai.__version__
    except:
        versions['openai'] = "Not installed"
    
    try:
        import langchain
        versions['langchain'] = langchain.__version__
    except:
        versions['langchain'] = "Not installed"
        
    return versions

def main():
    st.set_page_config(
        page_title="GPT CourtVision", 
        layout="wide"
    )
    
    with st.sidebar:
        st.title("Configuration")
        
        # Show library versions for debugging
        with st.expander("Library Versions (OpenAI v0.28 Compatible)"):
            versions = check_versions()
            for lib, version in versions.items():
                if version != "Not installed":
                    if lib == "openai" and version.startswith("0.28"):
                        st.success(f"{lib}: {version} ✓")
                    else:
                        st.text(f"{lib}: {version}")
                else:
                    st.error(f"{lib}: Not installed")
        
        # Debug: Show what's being read from environment
        if OPENAI_API_KEY:
            st.success(f"API Key found: {OPENAI_API_KEY[:10]}...")
        else:
            st.warning("No API key found in environment")
            
        openai_key = st.text_input(
            "OpenAI API Key", 
            type="password", 
            help="Enter your OpenAI API key to begin.",
            value=OPENAI_API_KEY or ""
        )
        
        try:
            import nba_api
            st.success("NBA API Connected")
        except ImportError:
            st.error("NBA API Not Installed")
            st.caption("Run: pip install nba_api")
        
        st.subheader("Agent Settings")
        show_tool_calls = st.checkbox("Show Tool Calls", value=False)
        show_reasoning = st.checkbox("Show Agent Reasoning", value=False)
        
        # Model selection for v0.28
        model_choice = st.selectbox(
            "GPT Model",
            ["gpt-4o-mini"],
            index=0,
            help="Choose the OpenAI model to use"
        )
    
    st.title("GPT CourtVision: NBA Analysis Agent")
    st.caption("Powered by LangChain, OpenAI v0.28, and the NBA API")
    
    if not openai_key:
        st.warning("Please provide your OpenAI API key in the sidebar to start the agent.")
        st.stop()
    
    # Initialize agent with error handling and model selection
    if "agent" not in st.session_state or st.session_state.get("current_model") != model_choice:
        try:
            prompts = load_prompts()
            # Update the agent initialization to use selected model
            agent_config = prompts.copy()
            st.session_state.agent = NBAAnalysisAgent(openai_key, agent_config, model_choice)
            st.session_state.current_model = model_choice
            
            st.success(f"Agent initialized successfully with {model_choice}!")
        except Exception as e:
            st.error(f"Failed to initialize agent: {str(e)}")
            if "api_key" in str(e).lower():
                st.info("Check your OpenAI API key")
            elif "model" in str(e).lower():
                st.info("Try switching to gpt-3.5-turbo model")
            elif "langchain" in str(e).lower():
                st.info("Try: pip uninstall langchain langchain-community langchain-openai && pip install langchain==0.1.0")
            st.stop()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        render_message(message["role"], message["content"])
    
    if prompt := st.chat_input("Ask about NBA games, predictions, or stats..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        render_message("user", prompt)
        
        with st.spinner("Analyzing..."):
            callback_handler = StreamlitCallbackHandler()
            result = st.session_state.agent.process_query(prompt, callback_handler)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": result["response"]
            })
            render_message("assistant", result["response"])
            
            if result["success"]:
                if show_tool_calls and result["steps"]:
                    with st.expander("Tool Calls"):
                        for step in result["steps"]:
                            if step["type"] == "action":
                                st.code(f"Tool: {step['tool']}\nInput: {step['input']}")
                
                if show_reasoning and result["steps"]:
                    with st.expander("Agent Reasoning"):
                        for step in result["steps"]:
                            if "log" in step and step["type"] == "action":
                                st.text(step["log"].strip())
            else:
                st.error(f"Error: {result['error']}")
        
        st.rerun()

if __name__ == "__main__":
    main()