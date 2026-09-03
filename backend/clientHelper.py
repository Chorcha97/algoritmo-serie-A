from curl_cffi import AsyncSession
import os
from dotenv import load_dotenv
from enum import Enum

class RefreshUrl(Enum):
    BASE = "BASE_URL"
    MARATHONBET = "MARATHONBET_URL"

load_dotenv()
BASE_URL: str = os.getenv("BASE_URL")
DATAVIZ_URL: str = os.getenv("DATAVIZ_URL")
SOFASCORE_URL: str = os.getenv("SOFASCORE_URL")
MARATHONBET_URL: str = os.getenv("MARATHONBET_URL")

async def refresh_session(client: AsyncSession, refresh_type: RefreshUrl = RefreshUrl.BASE):
    session_url = f"{BASE_URL}/wp-json/sdapi/v1/session" if refresh_type == RefreshUrl.BASE else f"{MARATHONBET_URL}/"
    response = await client.get(
        session_url
    )

    if response.status_code not in (200, 204):
        raise RuntimeError(
            f"Session initialization failed: {response.status_code}"
        )

async def api_get(
    client: AsyncSession,
    path: str,
    params: dict | None = None,
):
    url = f"{BASE_URL}{path}"

    return await api_get_base(client, url, params)

async def api_get_dataviz(
    client: AsyncSession,
    path: str,
    params: dict | None = None,
):
    url = f"{DATAVIZ_URL}{path}"

    return await api_get_base(client, url, params)

async def api_get_sofascore(
    client: AsyncSession,
    path: str,
    params: dict | None = None,
):
    url = f"{SOFASCORE_URL}{path}"

    return await api_get_base(client, url, params)

async def api_get_marathonbet(
    client: AsyncSession,
    path: str,
    params: dict | None = None,
):
    url = f"{MARATHONBET_URL}{path}"

    return await api_get_base(client, url, params, RefreshUrl.MARATHONBET)

async def api_get_base(
    client: AsyncSession,
    url: str,
    params: dict | None = None,
    refresh_type: RefreshUrl = RefreshUrl.BASE,
):
    response = await client.get(
        url,
        params=params,
        allow_redirects=True,
    )

    # Sessione scaduta
    if response.status_code in (401, 403):

        await refresh_session(client, refresh_type)

        # Retry UNA SOLA VOLTA
        response = await client.get(
            url,
            params=params,
            allow_redirects=True,
        )

    response.raise_for_status()

    return response
