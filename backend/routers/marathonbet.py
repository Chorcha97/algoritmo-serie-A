from curl_cffi import AsyncSession
from fastapi import APIRouter, Request, Query, HTTPException
from backend.clientHelper import api_get_marathonbet
import os
from dotenv import load_dotenv
import json
from typing import Any, List
from backend.routers.authmanager import auth_manager
from pydantic import BaseModel

class MenuItem(BaseModel):
    id: int
    name: str
    submenu: List[SubMenuItem] | None

class SubMenuItem(BaseModel):
    id: int
    name: str

load_dotenv()

CALCIO_SPORT_ID = "1"
router = APIRouter(
    prefix="/marathonbet",
    tags=["marathonbet"],
)

@router.get("/serie-a-bet/pre-match/eventi")
async def get_bet_eventi(
    request: Request,
    id_aggregata: int= Query(...)
):
    return await get_helper(request, endpoint=f"XSportDatastore/getTorneoCentrale?systemCode=MARATHONBET&lingua=IT&hash=&sportId=1&categoryId=31&tournamentId=33&idAggregata={id_aggregata}")

@router.get("/serie-a-bet/menu")
async def get_menu(request: Request):
    response = await get_helper(
        request,
        f"XSportDatastore/getStaticData?systemCode=MARATHONBET&lingua=IT&hash=&signatureAggregate=&signatureMacrogruppi=&signatureConfiguration=&signatureLabels=&betTemplatesSignature=&localStorageVersion=&isMobile=false"
    )

    try:
        return extract_calcio_menu(response)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

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


def _load_json_field(raw: dict[str, Any], field: str) -> dict[str, Any]:
    """I campi 'macro' e 'aggr' nella response esterna sono stringhe JSON
    annidate dentro il JSON principale: vanno deserializzate a parte."""
    value = raw.get(field)
    if value is None:
        raise ValueError(f"Campo '{field}' mancante nella response esterna")
    if isinstance(value, str):
        return json.loads(value)
    return value  # nel caso arrivi già come dict


def extract_calcio_menu(
        raw: dict[str, Any],
        sport_id: str = CALCIO_SPORT_ID,
) -> list[MenuItem]:
    macro = _load_json_field(raw, "macro")
    aggr = _load_json_field(raw, "aggr")

    try:
        top_level_items = macro["mcs"][sport_id]["pr"]
    except KeyError as exc:
        raise ValueError(
            f"Sport id '{sport_id}' non trovato in macro.mcs"
        ) from exc

    try:
        submenu_source = aggr["ags"][sport_id]["pr"]
    except KeyError as exc:
        raise ValueError(
            f"Sport id '{sport_id}' non trovato in aggr.ags"
        ) from exc

    # indice id -> nome per un lookup rapido delle sottovoci
    submenu_by_id: dict[int, str] = {
        item["id"]: item["ds"] for item in submenu_source
    }

    menu: list[MenuItem] = []
    for item in sorted(top_level_items, key=lambda x: x.get("rnk", 0)):
        item_id = item["id"]
        if item_id < 0:
            # voci "virtuali" come "Tutte" (id -1): niente sottomenu reale, si scartano
            continue

        submenu_ids = item.get("ags", [])
        submenu = [
            SubMenuItem(id=sub_id, name=submenu_by_id[sub_id])
            for sub_id in submenu_ids
            if sub_id in submenu_by_id
        ]

        menu.append(
            MenuItem(
                id=item_id,
                name=item["d"].strip(),
                submenu=submenu,
            )
        )

    return menu