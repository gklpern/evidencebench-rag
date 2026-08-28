from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


@dataclass(frozen=True)
class Database:
    engine: AsyncEngine
    sessions: async_sessionmaker[AsyncSession]


def create_database(database_url: str) -> Database:
    engine = create_async_engine(
        database_url,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
        pool_recycle=1800,
    )
    return Database(engine, async_sessionmaker(engine, expire_on_commit=False))


def create_session_factory(database_url: str) -> async_sessionmaker[AsyncSession]:
    return create_database(database_url).sessions


@asynccontextmanager
async def tenant_transaction(
    sessions: async_sessionmaker[AsyncSession], tenant_id: str
) -> AsyncIterator[AsyncSession]:
    canonical_tenant = str(UUID(tenant_id))
    async with sessions() as session, session.begin():
        await session.execute(
            text("SELECT set_config('app.tenant_id', :tenant_id, true)"),
            {"tenant_id": canonical_tenant},
        )
        yield session
