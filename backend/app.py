"""
Main FastAPI entrypoint for HACKNATION Super Resolution Mapping (SRM) System.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pathlib import Path
from backend.api.routes import router as api_router
from backend.config import BASE_DIR


class CharsetUtf8Middleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        ct = response.headers.get("content-type", "")
        needs_charset = (
            ct.startswith("text/")
            or ct.startswith("application/json")
            or ct.startswith("application/geo+json")
            or ct.startswith("application/javascript")
        )
        if needs_charset and "charset" not in ct.lower():
            sep = "; " if ct else ""
            response.headers["content-type"] = f"{ct}{sep}charset=utf-8"
        return response


app = FastAPI(
    title="NETRA - Super Resolution Mapping (SRM)",
    description="Sentinel-2 10m to <4m Super Resolution with Hallucination-Aware Uncertainty USP",
    version="1.0.0"
)

# Enable CORS for local/remote development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(CharsetUtf8Middleware)

# Register API endpoints
app.include_router(api_router)

# Mount frontend static directory
frontend_dir = BASE_DIR / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    ico_path = frontend_dir / "favicon.ico"
    if ico_path.exists():
        return FileResponse(str(ico_path), media_type="image/x-icon")
    logo_path = frontend_dir / "assets" / "logo.png"
    if logo_path.exists():
        return FileResponse(str(logo_path), media_type="image/png")
    return Response(status_code=204)

@app.get("/")
async def serve_index():
    index_path = frontend_dir / "index.html"
    if index_path.exists():
        return FileResponse(
            str(index_path),
            media_type="text/html; charset=utf-8",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    return {"message": "NETRA SRM Backend Active. Frontend index.html not found."}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "HACKNATION Super Resolution Mapping",
        "usp": "Hallucination-Aware Uncertainty Mapping (MC-Dropout + opensr-test)",
        "resolution": "10m -> 2.5m (4x upsampling)"
    }
