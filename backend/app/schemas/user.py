from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field, SecretStr, field_validator

Password = Annotated[SecretStr, Field(min_length=8, max_length=128)]


class LoginRequest(BaseModel):
    login: EmailStr = Field(description="Email пользователя", examples=["user@example.com"])
    password: SecretStr = Field(min_length=1, max_length=128)

    @field_validator("login")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        return value.lower()


class RegisterRequest(LoginRequest):
    password: Password
    full_name: str | None = Field(default=None, alias="fullName", max_length=255)

    @field_validator("full_name")
    @classmethod
    def normalize_full_name(cls, value: str | None) -> str | None:
        return value.strip() or None if value is not None else None


class ResetPasswordRequest(BaseModel):
    user_id: UUID = Field(alias="userId")
    old_password: SecretStr = Field(alias="oldPassword", min_length=1, max_length=128)
    new_password: Password = Field(alias="newPassword")


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    user_id: UUID = Field(alias="userId")
    login: EmailStr = Field(validation_alias="email")
    full_name: str | None = Field(alias="fullName")


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int = Field(description="Срок жизни токена в секундах")
