from curl_cffi import AsyncSession
from fastapi import APIRouter, Request
from backend.clientHelper import api_get, api_get_dataviz
import os
from dotenv import load_dotenv

load_dotenv()
TMCL: str = os.getenv("TOURNAMENT_CALENDAR")

router = APIRouter(
    prefix="/theanalyst",
    tags=["theanalyst"],
)

@router.get("/serie-a/standings")
async def get_standings(
    request: Request,
):
    return await get_helper(request, endpoint="standings")

@router.get("/serie-a/seasonpowerrankings")
async def get_seasonpowerrankings(
    request: Request,
):
    return await get_helper(request, endpoint="seasonpowerrankings")

@router.get("/serie-a/tournamentstats")
async def get_tournamentstats(
    request: Request,
):
    return await get_helper(request, endpoint="tournamentstats")

@router.get("/serie-a/expectedpoints")
async def get_expectedpoints(
    request: Request,
):
    client: AsyncSession = request.app.state.http_client

    response = await api_get_dataviz(
        client,
        f"/project-data/soccer/{TMCL}/expected-points.json",
    )

    return response.json()

async def get_helper(request: Request,  endpoint: str):
    client: AsyncSession = request.app.state.http_client

    response = await api_get(
        client,
        f"/wp-json/sdapi/v1/soccerdata/{endpoint}",
        params={
            "tmcl": TMCL,
        },
    )

    return response.json()