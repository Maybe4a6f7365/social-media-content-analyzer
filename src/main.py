import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .config import settings
from .models.api_models import (
    SocialMediaPostRequest,
    AdvancedEvaluationResponse,
    HealthResponse
)
from .services.evaluation_service import AdvancedEvaluationService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Version constant
VERSION = "2.0.0"

app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_state["evaluation_service"] = AdvancedEvaluationService()
    logger.info("Social Media Context Analyzer started")
    yield
    app_state.clear()
    logger.info("Social Media Context Analyzer stopped")


def get_evaluation_service() -> AdvancedEvaluationService:
    return app_state["evaluation_service"]


def create_app() -> FastAPI:
    return FastAPI(
        title="Social Media Context Analyzer",
        description="Multi-stage analysis service for evaluating social media content using LLM-powered analysis",
        version=VERSION,
        lifespan=lifespan
    )


app = create_app()


@app.get("/", response_model=HealthResponse, tags=["Health"])
async def root() -> HealthResponse:
    service = get_evaluation_service()
    test_result = await service.perform_full_analysis("health", "test", "en")

    return HealthResponse(
        status="operational",
        timestamp=test_result["analysis_timestamp"],
        version=VERSION
    )


@app.post("/evaluate", response_model=AdvancedEvaluationResponse, tags=["Analysis"])
async def evaluate_content(request: SocialMediaPostRequest) -> AdvancedEvaluationResponse:
    service = get_evaluation_service()
    result = await service.perform_full_analysis(
        post_id=request.post_id,
        post_text=request.post_text,
        language=request.language
    )

    return AdvancedEvaluationResponse(**result)


@app.get("/config", tags=["Configuration"])
def get_config() -> dict:
    return {
        "enable_advanced_analysis": settings.ENABLE_ADVANCED_ANALYSIS,
        "enable_veracity_check": settings.ENABLE_VERACITY_CHECK,
        "enable_nuance_analysis": settings.ENABLE_NUANCE_ANALYSIS,
        "claude_model": settings.CLAUDE_MODEL
    }
