import streamlit as st
from typing import Dict, Any, Optional
from ..config.settings import AppConfig, check_library_versions

def render_message(role: str, content: str) -> None:
    """Render a chat message with appropriate styling"""
    if role == "user":
        st.markdown(
            f'<div style="background-color:#0B93F6; color:white; padding:12px 18px; '
            f'border-radius:20px 20px 0 20px; max-width:70%; margin-left:auto; '
            f'margin-bottom:10px; font-family: sans-serif; '
            f'box-shadow: 1px 1px 3px rgba(0,0,0,0.2);">{content}</div>', 
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div style="background-color:#E5E5EA; color:#000; padding:12px 18px; '
            f'border-radius:20px 20px 20px 0; max-width:70%; margin-right:auto; '
            f'margin-bottom:10px; font-family: sans-serif; '
            f'box-shadow: 1px 1px 3px rgba(0,0,0,0.1);">{content}</div>', 
            unsafe_allow_html=True
        )

def create_sidebar() -> tuple[Optional[str], str, bool, bool]:
    """Create and return sidebar configuration"""
    with st.sidebar:
        st.title("Configuration")
        
        # Show library versions
        show_debug_info()
        
        # API Key input
        openai_key = st.text_input(
            "OpenAI API Key", 
            type="password", 
            help="Enter your OpenAI API key to begin.",
            value=st.session_state.get("openai_key", "")
        )
        
        # NBA API status
        try:
            import nba_api
            st.success("NBA API Connected")
        except ImportError:
            st.error("NBA API Not Installed")
            st.caption("Run: pip install nba_api")
        
        st.subheader("Agent Settings")
        
        # Debug options
        show_tool_calls = st.checkbox("Show Tool Calls", value=False)
        show_reasoning = st.checkbox("Show Agent Reasoning", value=False)
        
        # Model selection
        model_choice = st.selectbox(
            "GPT Model",
            ["gpt-4o-mini", "gpt-3.5-turbo"],
            index=0,
            help="Choose the OpenAI model to use"
        )
        
        # Clear conversation button
        if st.button("Clear Conversation"):
            if "messages" in st.session_state:
                st.session_state.messages = []
            if "agent" in st.session_state:
                st.session_state.agent.clear_memory()
            st.rerun()
    
    return openai_key, model_choice, show_tool_calls, show_reasoning

def show_debug_info() -> None:
    """Show debug information in expandable section"""
    with st.expander("Library Versions (OpenAI v0.28 Compatible)"):
        versions = check_library_versions()
        for lib, version in versions.items():
            if version != "Not installed":
                if lib == "openai" and version.startswith("0.28"):
                    st.success(f"{lib}: {version} ✓")
                else:
                    st.text(f"{lib}: {version}")
            else:
                st.error(f"{lib}: Not installed")

def show_tool_calls(steps: list) -> None:
    """Display tool calls in expandable section"""
    if not steps:
        return
        
    with st.expander("Tool Calls"):
        for step in steps:
            if step["type"] == "action":
                st.code(f"Tool: {step['tool']}\nInput: {step['input']}")

def show_reasoning(steps: list) -> None:
    """Display agent reasoning in expandable section"""
    if not steps:
        return
        
    with st.expander("Agent Reasoning"):
        for step in steps:
            if "log" in step and step["type"] == "action":
                st.text(step["log"].strip())

def display_welcome_message() -> None:
    """Display welcome message and instructions"""
    st.markdown("### Welcome to GPT CourtVision! 🏀")
    st.markdown("""
    I'm your NBA analysis assistant powered by AI. I can help you with:
    
    - **Live Game Data**: Get today's scores and schedules
    - **Team Statistics**: Deep dive into team performance metrics  
    - **Player Analysis**: Career stats and current season performance
    - **Game Predictions**: Mock predictions for upcoming matchups
    - **Commentary Style**: Authentic basketball commentary and analysis
    
    Try asking me something like:
    - "What games are on today?"
    - "Show me Lakers team stats"
    - "Predict Celtics vs Warriors"
    - "Tell me about LeBron James stats"
    """)