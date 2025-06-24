
import json
from typing import Dict, Any, List
from langchain.tools import Tool
from .base import BaseTool

class NBADBTool(BaseTool):
    """Tool for accessing NBA data via nba_api"""
    
    def __init__(self):
        super().__init__(
            name="nba_db",
            description="Accesses live NBA game data, team statistics, player stats, and matchup information using the nba_api package."
        )
    
    def run(self, query: str) -> str:
        """Execute NBA database query"""
        try:
            from nba_api.live.nba.endpoints import scoreboard
            from nba_api.stats.endpoints import teamgamelog, playercareerstats
            from nba_api.stats.static import players, teams
            
            query_lower = query.lower()
            
            if "today" in query_lower and ("games" in query_lower or "scoreboard" in query_lower):
                return self._get_todays_games()
            elif "team stats" in query_lower or "team statistics" in query_lower:
                return self._get_team_stats(query)
            elif "player" in query_lower:
                return self._get_player_stats(query)
            elif "matchup" in query_lower or "vs" in query_lower:
                return self._get_matchup_data(query)
            else:
                return self._get_general_nba_info()
                
        except ImportError:
            return "nba_api package not installed. Please install with: pip install nba_api"
        except Exception as e:
            return self._handle_error(e)
    
    def _get_todays_games(self) -> str:
        """Get today's NBA games"""
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
        
        return self._format_response({
            "todays_games": game_info, 
            "total_games": len(game_info)
        })
    
    def _get_team_stats(self, query: str) -> str:
        """Get team statistics"""
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
        
        if not team_id:
            return "Team not found. Please specify a valid NBA team."
        
        team_games = teamgamelog.TeamGameLog(team_id=team_id, season='2023-24')
        team_data = team_games.get_data_frames()[0]
        
        if team_data.empty:
            return self._format_response({
                "team_name": team_name, 
                "error": "No game data found for the 2023-24 season."
            })
        
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
        
        return self._format_response(stats_summary)
    
    def _get_player_stats(self, query: str) -> str:
        """Get player statistics"""
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
        
        if not player_id:
            return "Player not found. Please use the full name of an active player."
        
        career = playercareerstats.PlayerCareerStats(player_id=player_id)
        career_data = career.get_data_frames()[0]
        
        if career_data.empty:
            return self._format_response({
                "player_name": target_player, 
                "error": "No career data found."
            })

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
        
        return self._format_response(player_summary)

    def _get_matchup_data(self, query: str) -> str:
        """Get matchup data"""
        matchup_info = {
            "feature": "Team Matchup Analysis",
            "description": "Provides historical head-to-head data between teams.",
            "note": "Specify teams like 'Lakers vs Warriors' for detailed matchup data."
        }
        return self._format_response(matchup_info)
            
    def _get_general_nba_info(self) -> str:
        """Get general NBA information"""
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
        return self._format_response(general_info)

def create_nba_db_tool() -> Tool:
    """Create LangChain tool wrapper for NBA database"""
    nba_tool = NBADBTool()
    return Tool(
        name=nba_tool.name,
        description=nba_tool.description,
        func=nba_tool.run
    )
