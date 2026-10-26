from uuid import UUID

from fastapi import APIRouter, HTTPException, Response, status
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from starlette.concurrency import run_in_threadpool

from app.api.dependencies import CurrentUser, SettingsDep
from app.core.security import create_access_token, dummy_hash, hash_password, verify_password
from app.db.session import SessionDep
from app.models import User
from app.schemas.user import (
    LoginRequest,
    RegisterRequest,
    ResetPasswordRequest,
    TokenResponse,
    UserResponse,
)

router = APIRouter(prefix="/user", tags=["User"])


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=201,
    summary="Зарегистрировать пользователя",
    description="Email используется как логин без учёта регистра. Пароль: 8–128 символов.",
    responses={409: {"description": "Email уже зарегистрирован"}},
)
async def register(body: RegisterRequest, session: SessionDep) -> User:
    user = User(
        email=body.login,
        password_hash=await run_in_threadpool(hash_password, body.password.get_secret_value()),
        full_name=body.full_name,
    )
    session.add(user)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(status_code=409, detail="Email already registered") from None
    return user


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Войти по email и паролю",
    description="Возвращает JWT на 24 часа. Вставьте access_token в кнопку Authorize.",
    responses={401: {"description": "Неверный логин или пароль"}},
)
async def login(body: LoginRequest, session: SessionDep, settings: SettingsDep) -> TokenResponse:
    user = await session.scalar(select(User).where(User.email == body.login))
    valid = await run_in_threadpool(
        verify_password,
        body.password.get_secret_value(),
        user.password_hash if user else dummy_hash,
    )
    if user is None or not valid:
        raise HTTPException(
            status_code=401,
            detail="Invalid login or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return TokenResponse(
        access_token=create_access_token(user.user_id, user.auth_version, settings),
        expires_in=settings.jwt_expire_minutes * 60,
    )


@router.post(
    "/reset-password",
    status_code=204,
    summary="Изменить свой пароль",
    description=(
        "Требует Bearer JWT с sub = userId и старый пароль. "
        "После смены все прежние JWT пользователя отзываются: войдите заново."
    ),
    responses={
        400: {"description": "Неверный старый пароль"},
        401: {"description": "Требуется действующий JWT"},
        403: {"description": "Нельзя изменить пароль другого пользователя"},
    },
)
async def reset_password(
    body: ResetPasswordRequest, user: CurrentUser, session: SessionDep
) -> Response:
    if body.user_id != user.user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    if not await run_in_threadpool(
        verify_password, body.old_password.get_secret_value(), user.password_hash
    ):
        raise HTTPException(status_code=400, detail="Invalid old password")
    encoded = await run_in_threadpool(hash_password, body.new_password.get_secret_value())
    result = await session.execute(
        update(User)
        .where(User.user_id == user.user_id, User.auth_version == user.auth_version)
        .values(password_hash=encoded, auth_version=User.auth_version + 1)
        .execution_options(synchronize_session=False)
    )
    if result.rowcount != 1:
        await session.rollback()
        raise HTTPException(status_code=409, detail="Credentials changed; log in again")
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/{userId}",
    response_model=UserResponse,
    summary="Получить свой профиль",
    description="Возвращает профиль только если JWT.sub совпадает с userId.",
    responses={
        401: {"description": "Требуется действующий JWT"},
        403: {"description": "Доступ к чужому профилю запрещён"},
    },
)
async def get_user(userId: UUID, user: CurrentUser) -> User:
    if userId != user.user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    return user
