
import openai
from typing import Dict, Any, Optional, List
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage

from ..config.settings import AppConfig
from ..tools import create_caster_db_tool, create_nba_db_tool, create_mock_predictor_model_tool
from .callbacks import StreamlitCallbackHandler

# Handle LangChain version compatibility
try:
    try:
        from langchain_openai import ChatOpenAI
        LANGCHAIN_VERSION = "modern"
    except ImportError:
        from langchain.chat_models import ChatOpenAI
        LANGCHAIN_VERSION = "legacy"
except ImportError as e:
    raise ImportError(f"LangChain import error: {e}. "
                     "Try: pip install langchain==0.1.0 langchain-openai==0.0.5")

class NBAAnalysisAgent:
    """Main agent for NBA analysis and predictions"""
    
    def __init__(self, config: AppConfig, prompts: Dict[str, str]):
        self.config = config
        self.prompts = prompts
        
        # Set OpenAI API key for v0.28 compatibility
        openai.api_key = config.openai_api_key
        
        # Initialize LLM based on LangChain version
        self.llm = self._create_llm()
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize tools
        self.tools = self._create_tools()
        
        # Initialize agent
        self.agent = self._create_agent()
        
        # Set system prompt
        self._set_system_prompt()
    
    def _create_llm(self):
        """Create LLM instance based on LangChain version"""
        llm_params = {
            "temperature": self.config.temperature,
            "openai_api_key": self.config.openai_api_key,
            "max_tokens": self.config.max_tokens,
        }
        
        if LANGCHAIN_VERSION == "modern":
            llm_params.update({
                "model": self.config.model_name,
                "timeout": self.config.timeout
            })
        else:
            llm_params.update({
                "model_name": self.config.model_name,
                "request_timeout": self.config.timeout
            })
        
        return ChatOpenAI(**llm_params)
    
    def _create_tools(self) -> List[Tool]:
        """Create and return list of tools"""
        return [
            create_caster_db_tool(),
            create_nba_db_tool(),
            create_mock_predictor_model_tool()
        ]
    
    def _create_agent(self):
        """Create and return agent instance"""
        return initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=self.config.max_iterations,
        )
    
    def _set_system_prompt(self) -> None:
        """Set system prompt in memory"""
        system_message = SystemMessage(
            content=self.prompts.get(
                "system_prompt", 
                "You are a helpful NBA assistant."
            )
        )
        
        # Only add if not already present
        if not any(isinstance(m, SystemMessage) for m in self.memory.chat_memory.messages):
            self.memory.chat_memory.add_message(system_message)
    
    def process_query(self, query: str, callback_handler: Optional[StreamlitCallbackHandler] = None) -> Dict[str, Any]:
        """Process user query and return response"""
        try:
            callbacks = [callback_handler] if callback_handler else []
            response = self.agent.run(input=query, callbacks=callbacks)
            
            return {
                "response": response,
                "success": True,
                "steps": callback_handler.steps if callback_handler else [],
                "error": None
            }
        except Exception as e:
            return {
                "response": f"I encountered an error: {str(e)}",
                "success": False,
                "steps": callback_handler.steps if callback_handler else [],
                "error": str(e)
            }
    
    def clear_memory(self) -> None:
        """Clear conversation memory"""
        self.memory.clear()
        self._set_system_prompt()
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get formatted conversation history"""
        history = []
        for message in self.memory.chat_memory.messages:
            if hasattr(message, 'content'):
                message_type = "system" if isinstance(message, SystemMessage) else "user"
                history.append({
                    "role": message_type,
                    "content": message.content
                })
        return history
