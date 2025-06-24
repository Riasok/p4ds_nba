import streamlit as st
from typing import Dict, Any, Optional

from ..config.settings import AppConfig, load_prompts
from ..agents import NBAAnalysisAgent, StreamlitCallbackHandler
from .components import (
    render_message, 
    create_sidebar, 
    show_tool_calls, 
    show_reasoning,
    display_welcome_message
)

def initialize_agent(config: AppConfig, prompts: Dict[str, str]) -> Optional[NBAAnalysisAgent]:
    """Initialize the NBA analysis agent"""
    try:
        agent = NBAAnalysisAgent(config, prompts)
        st.success(f"Agent initialized successfully with {config.model_name}!")
        return agent
    except Exception as e:
        st.error(f"Failed to initialize agent: {str(e)}")
        
        # Provide helpful error messages
        error_str = str(e).lower()
        if "api_key" in error_str:
            st.info("Check your OpenAI API key")
        elif "model" in error_str:
            st.info("Try switching to gpt-3.5-turbo model")
        elif "langchain" in error_str:
            st.info("Try: pip uninstall langchain langchain-community langchain-openai && pip install langchain==0.1.0")
        
        return None

def handle_chat_input(prompt: str, agent: NBAAnalysisAgent, show_tool_calls_flag: bool, show_reasoning_flag: bool) -> None:
    """Handle user chat input and generate response"""
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    render_message("user", prompt)
    
    # Generate response
    with st.spinner("Analyzing..."):
        callback_handler = StreamlitCallbackHandler()
        result = agent.process_query(prompt, callback_handler)
        
        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant", 
            "content": result["response"]
        })
        render_message("assistant", result["response"])
        
        # Show debug information if requested
        if result["success"]:
            if show_tool_calls_flag:
                show_tool_calls(result["steps"])
            if show_reasoning_flag:
                show_reasoning(result["steps"])
        else:
            st.error(f"Error: {result['error']}")

def main():
    """Main application entry point"""
    # Configure Streamlit page
    st.set_page_config(
        page_title="GPT CourtVision", 
        page_icon="🏀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Main title
    st.title("GPT CourtVision: NBA Analysis Agent")
    st.caption("Powered by LangChain, OpenAI v0.28, and the NBA API")
    
    # Create sidebar and get configuration
    openai_key, model_choice, show_tool_calls_flag, show_reasoning_flag = create_sidebar()
    
    # Check for API key
    if not openai_key:
        st.warning("Please provide your OpenAI API key in the sidebar to start the agent.")
        display_welcome_message()
        st.stop()
    
    # Store API key in session state
    if "openai_key" not in st.session_state or st.session_state.openai_key != openai_key:
        st.session_state.openai_key = openai_key
    
    # Initialize agent if needed or model changed
    if ("agent" not in st.session_state or 
        st.session_state.get("current_model") != model_choice or
        st.session_state.get("current_api_key") != openai_key):
        
        # Create configuration
        config = AppConfig.from_env(openai_key)
        config.model_name = model_choice
        
        # Load prompts
        prompts = load_prompts()
        
        # Initialize agent
        agent = initialize_agent(config, prompts)
        if agent is None:
            st.stop()
        
        # Store in session state
        st.session_state.agent = agent
        st.session_state.current_model = model_choice
        st.session_state.current_api_key = openai_key
    
    # Initialize messages if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display conversation history
    if not st.session_state.messages:
        display_welcome_message()
    else:
        for message in st.session_state.messages:
            render_message(message["role"], message["content"])
    
    # Handle chat input
    if prompt := st.chat_input("Ask about NBA games, predictions, or stats..."):
        handle_chat_input(
            prompt, 
            st.session_state.agent, 
            show_tool_calls_flag, 
            show_reasoning_flag
        )
        st.rerun()

if __name__ == "__main__":
    main()