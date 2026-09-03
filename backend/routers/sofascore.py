from curl_cffi import AsyncSession
from fastapi import APIRouter, Request
from backend.clientHelper import api_get_sofascore
import os
from dotenv import load_dotenv

load_dotenv()
TMCL: str = os.getenv("TOURNAMENT_CALENDAR")
T_ID: str = os.getenv("TOURNAMENT_ID")
S_ID: str = os.getenv("SEASON_ID")

router = APIRouter(
    prefix="/sofascore",
    tags=["sofascore"],
)

@router.get("/serie-a/top-players")
async def get_top_players(
    request: Request,
):
    return await get_helper_season(request, endpoint="top-players/overall")

@router.get("/serie-a/top-players-per-game")
async def get_top_players_per_game(
    request: Request,
):
    return await get_helper_season(request, endpoint="top-players-per-game/all/overall")

@router.get("/serie-a/top-teams")
async def get_top_teams(
    request: Request,
):
    return await get_helper_season(request, endpoint="top-teams/overall")

@router.get("/serie-a/player-of-the-season-race")
async def get_player_of_the_season_race(
    request: Request,
):
    return await get_helper_season(request, endpoint="player-of-the-season-race")

@router.get("/serie-a/team-of-the-period")
async def get_team_of_the_period(
    request: Request,
):
    client: AsyncSession = request.app.state.http_client

    response = await api_get_sofascore(
        client,
        f"/team-of-the-period/28788",
    )

    return response.json()

@router.get("/serie-a/results")
async def get_results(request: Request, round: int = 1):
    return await get_helper_season(request, endpoint=f"events/round/{round}")

# get della classifica finale
@router.get("/serie-a/standings/total")
async def get_standings_total(request: Request):
    return await get_helper_season(request, endpoint=f"standings/total")

@router.get("/serie-a/average_positions")
async def get_average_positions(
    request: Request,
    game_id: int
):
    return await get_helper_event(request, endpoint="average-positions", game_id=game_id)

@router.get("/serie-a/lineups")
async def get_lineups(
    request: Request,
    game_id: int
):
    return await get_helper_event(request, endpoint="lineups", game_id=game_id)

@router.get("/serie-a/best-players/summary")
async def get_best_players(
        request: Request,
        game_id: int
):
    return await get_helper_event(request, endpoint="best-players/summary", game_id=game_id)

@router.get("/serie-a/incidents")
async def get_incidents(
    request: Request,
    game_id: int
):
    return await get_helper_event(request, endpoint="incidents", game_id=game_id)

@router.get("/serie-a/shotmap")
async def get_shotmap(
    request: Request,
    game_id: int
):
    return await get_helper_event(request, endpoint="shotmap", game_id=game_id)

@router.get("/serie-a/team-streaks")
async def get_team_streaks(
    request: Request,
    game_id: int
):
    return await get_helper_event(request, endpoint="team-streaks", game_id=game_id)

@router.get("/serie-a/statistics")
async def get_statistics(
    request: Request,
    game_id: int
):
    return await get_helper_event(request, endpoint="statistics", game_id=game_id)

@router.get("/serie-a/h2h")
async def get_h2h(
    request: Request,
    game_id: int
):
    return await get_helper_event(request, endpoint="h2h", game_id=game_id)

@router.get("/serie-a/goal-distributions")
async def get_goal_distributions(request: Request, team_id: int):
    return await get_helper_season(request, endpoint=f"goal-distributions", pre_endpoint=f"/team/{team_id}")

@router.get("/serie-a/statistics/overall")
async def get_statistics_overall(request: Request, team_id: int):
    return await get_helper_season(request, endpoint=f"statistics/overall", pre_endpoint=f"/team/{team_id}")

@router.get("/serie-a/featured-players")
async def get_featured_players(request: Request, team_id: int):
    return await get_helper_team(request, endpoint=f"featured-players", team_id=team_id)

@router.get("/serie-a/team-statistics/seasons")
async def get_team_statistics(request: Request, team_id: int):
     return await get_helper_team(request, endpoint=f"team-statistics/seasons", team_id=team_id)

async def get_helper_team(request: Request, endpoint: str, team_id: int, params: dict | None = None):
    client: AsyncSession = request.app.state.http_client

    response = await api_get_sofascore(
        client,
        f"/team/{team_id}/{endpoint}",
        params=params,
    )

    return response.json()

async def get_helper_event(request: Request, endpoint: str, game_id: int, params: dict | None = None):
    client: AsyncSession = request.app.state.http_client

    response = await api_get_sofascore(
        client,
        f"/event/{game_id}/{endpoint}",
        params=params,
    )

    return response.json()

async def get_helper_season(request: Request, endpoint: str, pre_endpoint: str = "", params: dict | None = None):
    client: AsyncSession = request.app.state.http_client

    response = await api_get_sofascore(
        client,
        f"{pre_endpoint}/unique-tournament/{T_ID}/season/{S_ID}/{endpoint}",
        params=params,
    )

    return response.json()