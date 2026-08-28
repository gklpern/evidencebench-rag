from evidencebench.auth import TenantAuthenticator


def test_api_key_resolves_tenant() -> None:
    tenant = "12345678-1234-5678-1234-567812345678"
    authenticator = TenantAuthenticator(f'{{"secret":"{tenant}"}}')
    assert authenticator.authenticate("Bearer secret") == tenant
    assert authenticator.authenticate("Bearer wrong") is None
    assert authenticator.authenticate(None) is None
