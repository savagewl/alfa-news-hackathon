from datetime import UTC, datetime, timedelta
from uuid import uuid4

import jwt
import pytest

from app.core.config import Settings
from app.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)

SETTINGS = Settings(
    _env_file=None, jwt_secret="test-secret-with-at-least-48-characters-for-algorithm-tests"
)


def test_password_is_argon2id_and_verifies():
    encoded = hash_password("my-test-password")
    assert encoded.startswith("$argon2id$")
    assert encoded != hash_password("my-test-password")
    assert verify_password("my-test-password", encoded)
    assert not verify_password("wrong-password", encoded)
    assert not verify_password("my-test-password", "broken hash")


def test_token_subject_version_and_24_hour_expiry():
    user_id = uuid4()
    token = create_access_token(user_id, 2, SETTINGS)
    assert decode_access_token(token, SETTINGS) == (user_id, 2)
    claims = jwt.decode(token, SETTINGS.jwt_secret.get_secret_value(), algorithms=["HS256"])
    assert claims["exp"] - claims["iat"] == 86400


@pytest.mark.parametrize("case", ["expired", "missing_exp", "wrong_algorithm", "wrong_key"])
def test_rejects_invalid_tokens(case):
    now = datetime.now(UTC)
    payload = {"sub": str(uuid4()), "iat": now, "exp": now + timedelta(hours=1), "ver": 0}
    if case == "expired":
        payload["exp"] = now - timedelta(seconds=1)
    if case == "missing_exp":
        del payload["exp"]
    key = (
        "different-secret-with-32-characters"
        if case == "wrong_key"
        else SETTINGS.jwt_secret.get_secret_value()
    )
    token = jwt.encode(payload, key, algorithm="HS384" if case == "wrong_algorithm" else "HS256")
    with pytest.raises(jwt.InvalidTokenError):
        decode_access_token(token, SETTINGS)
