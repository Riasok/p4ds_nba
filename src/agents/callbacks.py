from typing import Any, Dict, List
from langchain.schema import AgentAction, AgentFinish
from langchain.callbacks.base import BaseCallbackHandler

class StreamlitCallbackHandler(BaseCallbackHandler):
    """Callback handler for Streamlit integration"""
    
    def __init__(self):
        self.steps: List[Dict[str, Any]] = []
    
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
    
    def clear_steps(self) -> None:
        """Clear recorded steps"""
        self.steps = []
