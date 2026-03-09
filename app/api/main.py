"""SlideCraft API application entrypoint."""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routers import document, context
from app.bootstrap import startup


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Async context manager for application lifespan."""
    startup()
    yield


app = FastAPI(title="SlideCraft API", version="0.1", lifespan=lifespan)

app.include_router(document.router, prefix="/api/v1")
app.include_router(context.router, prefix="/api/v1")
