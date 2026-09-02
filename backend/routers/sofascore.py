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

async def get_helper_season(request: Request,  endpoint: str, params: dict | None = None):
    client: AsyncSession = request.app.state.http_client

    response = await api_get_sofascore(
        client,
        f"/unique-tournament/{T_ID}/season/{S_ID}/{endpoint}",
        params=params,
    )

    return response.json()