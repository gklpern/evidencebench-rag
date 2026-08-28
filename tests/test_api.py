import httpx
import pytest

from evidencebench.api import app


@pytest.mark.asyncio
async def test_liveness_does_not_require_dependencies() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["x-request-id"]
