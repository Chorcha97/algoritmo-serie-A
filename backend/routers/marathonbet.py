from curl_cffi import AsyncSession
from fastapi import APIRouter, Request
from backend.clientHelper import api_get_sofascore, api_get_marathonbet
import os
from dotenv import load_dotenv

load_dotenv()
TMCL: str = os.getenv("TOURNAMENT_CALENDAR")
T_ID: str = os.getenv("TOURNAMENT_ID")
S_ID: str = os.getenv("SEASON_ID")

router = APIRouter(
    prefix="/marathonbet",
    tags=["marathonbet"],
)

@router.get("/serie-a-bet/login")
async def get_top_players(
    request: Request,
):
    return await get_helper(request, endpoint="loginContogioco", params={"username": "Chorcha97", "password": "Beastocco07!"})

async def get_helper(request: Request,  endpoint: str, params: dict | None = None):
    client: AsyncSession = request.app.state.http_client

    response = await api_get_marathonbet(
        client,
        f"/{endpoint}",
        params=params,
    )

    return response.json()