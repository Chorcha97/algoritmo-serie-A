from curl_cffi import AsyncSession

import os
from dotenv import load_dotenv

load_dotenv()
BASE_URL: str = os.getenv("BASE_URL")

async def refresh_session(client: AsyncSession):
    response = await client.get(
        f"{BASE_URL}/wp-json/sdapi/v1/session"
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

    response = await client.get(
        url,
        params=params,
    )

    # Sessione scaduta
    if response.status_code in (401, 403):

        await refresh_session(client)

        # Retry UNA SOLA VOLTA
        response = await client.get(
            url,
            params=params,
        )

    response.raise_for_status()

    return response