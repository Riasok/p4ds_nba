import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class AppConfig:
    """Application configuration"""
    openai_api_key: str
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.5
    max_tokens: int = 1000
    timeout: int = 60
    max_iterations: int = 5
    
    @classmethod
    def from_env(cls, openai_key: Optional[str] = None) -> 'AppConfig':
        """Create config from environment variables"""
        api_key = openai_key or os.getenv("OPENAI_API_KEY", "")
        return cls(
            openai_api_key=api_key,
            model_name=os.getenv("MODEL_NAME", "gpt-4o-mini"),
            temperature=float(os.getenv("TEMPERATURE", "0.5")),
            max_tokens=int(os.getenv("MAX_TOKENS", "1000")),
            timeout=int(os.getenv("TIMEOUT", "60")),
            max_iterations=int(os.getenv("MAX_ITERATIONS", "5"))
        )

def load_prompts(path: str = "data/prompts.yaml") -> Dict[str, str]:
    """Load prompts from YAML file"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {
            "system_prompt": "You are GPT CourtVision, an expert NBA analyst. "
                           "Your tools include a caster database for authentic commentary, "
                           "an NBA database for live stats, and a mock prediction model "
                           "for demonstrations. Your goal is to provide engaging and insightful analysis."
        }

def check_library_versions() -> Dict[str, str]:
    """Check versions of key libraries"""
    versions = {}
    libraries = ['openai', 'langchain', 'nba_api', 'streamlit']
    
    for lib in libraries:
        try:
            module = __import__(lib)
            versions[lib] = getattr(module, '__version__', 'Unknown')
        except ImportError:
            versions[lib] = "Not installed"
    
    return versions
