"""Service helpers for user management within the API layer."""

from __future__ import annotations

from collections.abc import Mapping

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from apps.api.db import models, repositories


def ensure_owner(db_session: Session, user_profile: Mapping[str, object]) -> models.User:
    """Return the persistent :class:`~apps.api.db.models.User` for a request."""

    email = user_profile.get("email")
    if not isinstance(email, str) or not email:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Invalid user profile")

    normalized_email = email.strip().lower()
    repo = repositories.UserRepository(db_session)
    user = repo.get_by_email(normalized_email)
    if user is None:
        name_value = user_profile.get("name")
        user = repo.create(email=normalized_email, name=str(name_value) if name_value else None)
    else:
        name_value = user_profile.get("name")
        if isinstance(name_value, str) and name_value and name_value != (user.name or ""):
            user.name = name_value
            db_session.add(user)
    return user


__all__ = ["ensure_owner"]
