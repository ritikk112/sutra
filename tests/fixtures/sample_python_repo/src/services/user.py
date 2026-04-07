"""User service module."""

from pathlib import Path
import os

MAX_RETRIES: int = 3


class Base:
    pass


class UserService(Base):
    """Manages user operations."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def create_user(self, name: str) -> str:
        return _generate_id(name)


def _generate_id(name: str) -> str:
    return name


def bootstrap(path: str) -> "UserService":
    svc = UserService(path)
    svc.create_user("admin")
    return svc
