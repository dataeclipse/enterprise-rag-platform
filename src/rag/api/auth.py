from datetime import UTC, datetime, timedelta
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from rag.config import AuthConfig
from rag.exceptions import AuthError

_bearer = HTTPBearer(auto_error=False)


def create_access_token(subject: str, config: AuthConfig) -> str:
    now = datetime.now(UTC)
    payload = {
        "sub": subject,
        "iat": now,
        "exp": now + timedelta(seconds=config.token_ttl_seconds),
    }
    return jwt.encode(payload, config.secret_key.get_secret_value(), algorithm=config.algorithm)


def decode_token(token: str, config: AuthConfig) -> str:
    try:
        payload = jwt.decode(
            token,
            config.secret_key.get_secret_value(),
            algorithms=[config.algorithm],
            options={"require": ["sub", "exp"]},
        )
    except jwt.PyJWTError as exc:
        raise AuthError("invalid or expired token") from exc
    return str(payload["sub"])


async def get_current_subject(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)] = None,
) -> str:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    config: AuthConfig = request.app.state.container.settings.auth
    try:
        return decode_token(credentials.credentials, config)
    except AuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
