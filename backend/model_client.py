import httpx


class ModelServiceUnavailable(Exception):
    """Model service unreachable or too slow."""


class ModelServiceError(Exception):
    """Model service answered, but with an error."""


class ModelClient:
    def __init__(self, base_url: str, timeout: float):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def predict(self, features: dict) -> dict:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(f"{self.base_url}/predict", json={"features": features})
        except httpx.TransportError as exc:  # connect error, timeout, network error
            raise ModelServiceUnavailable(str(exc)) from exc
        if resp.status_code != 200:
            raise ModelServiceError(f"model service returned {resp.status_code}")
        try:
            return resp.json()
        except ValueError as exc:
            raise ModelServiceError("model service returned invalid JSON") from exc

    async def is_healthy(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(f"{self.base_url}/health")
            return resp.status_code == 200
        except httpx.TransportError:
            return False
