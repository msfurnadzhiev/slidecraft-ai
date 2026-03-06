from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routers import document
from app.bootstrap import startup


@asynccontextmanager
async def lifespan(app: FastAPI):
    startup()
    yield


app = FastAPI(title="SlideCraft API", version="0.1", lifespan=lifespan)

app.include_router(document.router, prefix="/api/v1")