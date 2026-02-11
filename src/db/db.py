from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.config import get_variables


engine = create_async_engine(
    url=get_variables().DATABASE_URL_asyncpg,
    echo=True)

session_fabric = async_sessionmaker(engine)


class Base(DeclarativeBase):
    pass

