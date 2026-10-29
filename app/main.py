from contextlib import asynccontextmanager
from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool
from app.routers.videoupload import router as video_router
from app.services.video_processor import shutdown_workers


@asynccontextmanager
async def lifespan(app):
    try:
        yield
    finally:
        await run_in_threadpool(shutdown_workers)


app = FastAPI(
    lifespan=lifespan,
    title="Football Kick Analyser",
    description="Upload a football match video and get kick detection results with extracted clips.",
    version="1.0.0",
)

app.include_router(video_router, prefix="/video")


@app.get("/")
def read_root():
    return {
        "message": "Football Kick Analyser API is running.",
        "docs": "/docs",
    }
