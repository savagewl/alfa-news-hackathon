from datetime import UTC, datetime, timedelta
from uuid import UUID

import jwt
from argon2 import PasswordHasher, Type
from argon2.exceptions import InvalidHashError, VerificationError
from fastapi import HTTPException

from app.core.config import Settings

password_hasher = PasswordHasher(type=Type.ID)
# Use the same expensive verification path when an email is not registered.
dummy_hash = password_hasher.hash("unused-dummy-password")


def hash_password(password: str) -> str:
    return password_hasher.hash(password)


def verify_password(password: str, encoded_hash: str) -> bool:
    try:
        return password_hasher.verify(encoded_hash, password)
    except (VerificationError, InvalidHashError):
        return False


def jwt_key(settings: Settings) -> str:
    if settings.jwt_secret is None:
        raise HTTPException(status_code=503, detail="JWT secret is not configured")
    return settings.jwt_secret.get_secret_value()


def create_access_token(user_id: UUID, auth_version: int, settings: Settings) -> str:
    now = datetime.now(UTC)
    return jwt.encode(
        {
            "sub": str(user_id),
            "iat": now,
            "exp": now + timedelta(minutes=settings.jwt_expire_minutes),
            "ver": auth_version,
        },
        jwt_key(settings),
        algorithm="HS256",
    )


def decode_access_token(token: str, settings: Settings) -> tuple[UUID, int]:
    payload = jwt.decode(
        token,
        jwt_key(settings),
        algorithms=["HS256"],
        options={"require": ["sub", "exp", "iat", "ver"]},
    )
    version = payload["ver"]
    if type(version) is not int or version < 0:
        raise jwt.InvalidTokenError("Invalid token version")
    return UUID(payload["sub"]), version
