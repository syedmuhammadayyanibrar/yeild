from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from loop.api.routes import router as api_router
from loop.config import ensure_directories

ensure_directories()

app = FastAPI(title="Yeild Learning Engine", version="0.1.0")

app.include_router(api_router, prefix="/api")

web_dir = Path(__file__).resolve().parent.parent / "web"
static_dir = web_dir / "static"

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
def serve_index() -> FileResponse:
    index_file = web_dir / "index.html"
    return FileResponse(str(index_file))

@app.get("/{full_path:path}")
def serve_spa(full_path: str) -> FileResponse:
    index_file = web_dir / "index.html"
    return FileResponse(str(index_file))
