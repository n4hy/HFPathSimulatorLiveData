"""FastAPI application for HF Path Simulator REST API."""

import time
from contextlib import asynccontextmanager
from typing import Optional
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .models import HealthResponse, GPUInfoResponse, ErrorResponse
from .routes import channel_router, processing_router, streaming_router
from ..engine import SimulationEngine, EngineConfig
from ..engine.session import get_session_manager, shutdown_session_manager

# Module version
__version__ = "0.1.0"

# Global engine for session-less access
_global_engine: Optional[SimulationEngine] = None
_start_time: float = 0.0

logger = logging.getLogger(__name__)


def get_global_engine() -> SimulationEngine:
    """Get or create the global simulation engine."""
    global _global_engine
    if _global_engine is None:
        _global_engine = SimulationEngine(
            config=EngineConfig(use_gpu=True)
        )
    return _global_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global _start_time

    # Startup
    _start_time = time.time()
    logger.info("HF Path Simulator API starting up")

    # Initialize global engine
    get_global_engine()

    # Initialize session manager
    get_session_manager()

    yield

    # Shutdown
    logger.info("HF Path Simulator API shutting down")
    shutdown_session_manager()


def create_app(
    title: str = "HF Path Simulator API",
    description: str = "REST API for HF ionospheric channel simulation",
    cors_origins: Optional[list[str]] = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        title: API title
        description: API description
        cors_origins: Allowed CORS origins (default allows all)

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title=title,
        description=description,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request timing middleware
    @app.middleware("http")
    async def add_timing_header(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        response.headers["X-Process-Time"] = f"{elapsed:.4f}"
        return response

    # Exception handlers
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc)},
        )

    # Health and info endpoints
    @app.get(
        "/api/v1/health",
        response_model=HealthResponse,
        tags=["system"],
    )
    async def health_check():
        """Health check endpoint.

        Returns service status and uptime.
        """
        return HealthResponse(
            status="ok",
            version=__version__,
            uptime_seconds=time.time() - _start_time,
        )

    @app.get(
        "/api/v1/gpu",
        response_model=GPUInfoResponse,
        tags=["system"],
    )
    async def gpu_info():
        """Get GPU information.

        Returns GPU availability, name, memory, and backend.
        """
        engine = get_global_engine()
        info = engine.get_gpu_info()

        return GPUInfoResponse(
            available=info.get("available", True) if "name" in info else False,
            name=info.get("name"),
            compute_capability=info.get("compute_capability"),
            total_memory_gb=info.get("total_memory_gb"),
            multiprocessors=info.get("multiprocessors"),
            backend=info.get("backend"),
        )

    # Include routers
    app.include_router(channel_router, prefix="/api/v1")
    app.include_router(processing_router, prefix="/api/v1")
    app.include_router(streaming_router, prefix="/api/v1")

    return app


# Default application instance
app = create_app()


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    workers: int = 1,
    log_level: str = "info",
):
    """Run the API server.

    Args:
        host: Host to bind to
        port: Port to listen on
        reload: Enable auto-reload for development
        workers: Number of worker processes
        log_level: Logging level
    """
    import uvicorn

    uvicorn.run(
        "hfpathsim.api.app:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level=log_level,
    )


def main():
    """Entry point for hfpathsim-server command."""
    import argparse

    parser = argparse.ArgumentParser(
        description="HF Path Simulator REST API Server"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level",
    )

    args = parser.parse_args()

    run_server(
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
