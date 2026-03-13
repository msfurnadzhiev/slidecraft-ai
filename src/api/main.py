"""SlideCraft API application entrypoint."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routers import document, context
from src.bootstrap import startup

_API_PREFIX = "/api/v1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Async context manager for application lifespan."""
    startup()
    yield


app = FastAPI(title="SlideCraft API", version="0.1", lifespan=lifespan)

app.include_router(document.router, prefix=_API_PREFIX)
app.include_router(context.router, prefix=_API_PREFIX)
