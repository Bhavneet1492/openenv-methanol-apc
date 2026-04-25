"""Local preview server for the MS-AOS dashboard.

Run this to see the UI locally before pushing anywhere:
    python methanol_apc_env/server/preview_ui.py

Opens at http://localhost:7860
"""

import os
import sys
from pathlib import Path

# Add parent to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI()

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    print(f"Starting local preview at http://localhost:7860")
    print(f"Serving static files from: {STATIC_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=7860)
