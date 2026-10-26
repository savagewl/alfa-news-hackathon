from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import Settings
from app.core.security import decode_access_token
from app.db.session import SessionDep
from app.models import User

bearer = HTTPBearer(auto_error=False, description="JWT из POST /user/login")


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


SettingsDep = Annotated[Settings, Depends(get_app_settings)]


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    session: SessionDep,
    settings: SettingsDep,
) -> User:
    unauthorized = HTTPException(
        status_code=401, detail="Invalid or expired token", headers={"WWW-Authenticate": "Bearer"}
    )
    if credentials is None:
        raise unauthorized
    try:
        user_id, version = decode_access_token(credentials.credentials, settings)
    except (jwt.InvalidTokenError, ValueError, TypeError, AttributeError):
        raise unauthorized from None
    user = await session.get(User, user_id)
    if user is None or user.auth_version != version:
        raise unauthorized
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]
