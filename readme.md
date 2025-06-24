## NBA Predictor Team 1

Set your OpenAI API key in `.env` or through the Streamlit sidebar. The application supports:

- Multiple OpenAI models (gpt-4o-mini, gpt-3.5-turbo)
- Adjustable temperature and token limits
- Debug modes for tool calls and agent reasoning
- 
## Tools Available

1. **Caster DB**: Authentic basketball commentary patterns
2. **NBA DB**: Live game data, team stats, player information
3. **Predictor**: Demonstration predictions with simulated probabilities

## Requirements

- Python 3.8+
- OpenAI API key
- Internet connection for NBA API data

## Architecture

The application follows a modular architecture with clear separation of concerns:

- **Config Layer**: Environment and settings management
- **Tools Layer**: MCP tools for external data sources
- **Agent Layer**: LangChain agent orchestration
- **UI Layer**: Streamlit interface components
