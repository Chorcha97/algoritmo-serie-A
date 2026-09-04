from curl_cffi import AsyncSession
from fastapi import APIRouter, Request, Query
from backend.clientHelper import api_get_marathonbet
import os
from dotenv import load_dotenv
from backend.models.IdAggregataEnum import IdAggregata
from backend.routers.authmanager import auth_manager

load_dotenv()

router = APIRouter(
    prefix="/marathonbet",
    tags=["marathonbet"],
)

@router.get("/serie-a-bet/pre-match/eventi")
async def get_bet_eventi(
    request: Request,
    id_aggregata: IdAggregata= Query(...)
):
    return await get_helper(request, endpoint=f"XSportDatastore/getTorneoCentrale?systemCode=MARATHONBET&lingua=IT&hash=&sportId=1&categoryId=31&tournamentId=33&idAggregata={id_aggregata.value}")

async def get_helper(request: Request, endpoint: str, params: dict | None = None, headers: dict | None = None):
    client: AsyncSession = request.app.state.http_client
    #token = await auth_manager.get_token(request)
    response = await api_get_marathonbet(
        client,
        f"/{endpoint}",
        params=params,
        headers=headers
        #{"Authorization": f"Bearer {token}"},
    )

    return response.json()