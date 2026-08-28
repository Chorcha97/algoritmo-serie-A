
from fastapi import APIRouter, Request
import httpx
from backend.clientHelper import api_get
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
    client: httpx.AsyncClient = request.app.state.http_client

    response = await api_get(
        client,
        "/wp-json/sdapi/v1/soccerdata/standings",
        params={
            "tmcl": TMCL,
        },
    )

    return response.json()