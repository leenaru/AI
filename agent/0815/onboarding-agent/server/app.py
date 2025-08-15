from fastapi import FastAPI
from server.core.config import settings
from server.api.routes import router as api_router
from server.ops.health import router as health_router
from server.slack.routes import router as slack_router

app = FastAPI(title="Onboarding Agent API")
app.include_router(health_router, prefix="/")
app.include_router(api_router, prefix="/")
app.include_router(slack_router, prefix="/")
