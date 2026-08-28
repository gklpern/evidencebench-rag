import json
from hmac import compare_digest
from uuid import UUID

from pydantic import TypeAdapter


class TenantAuthenticator:
    def __init__(self, api_keys_json: str) -> None:
        parsed = TypeAdapter(dict[str, str]).validate_python(json.loads(api_keys_json or "{}"))
        self._keys = [(key, str(UUID(tenant_id))) for key, tenant_id in parsed.items()]

    def authenticate(self, authorization: str | None) -> str | None:
        if not authorization or not authorization.startswith("Bearer "):
            return None
        presented = authorization.removeprefix("Bearer ")
        matched_tenant = None
        for key, tenant_id in self._keys:
            if compare_digest(presented, key):
                matched_tenant = tenant_id
        return matched_tenant
