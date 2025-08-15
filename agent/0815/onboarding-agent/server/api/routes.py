from fastapi import APIRouter, UploadFile, File, Body
from typing import Any, Dict
from server.orchestrator.graph import run_conversation
from server.vision.service import analyze_image

router = APIRouter()

@router.post("/chat")
async def chat_endpoint(payload: Dict[str, Any]):
    return await run_conversation(payload)

@router.post("/vision/analyze")
async def vision_endpoint(file: UploadFile = File(...), tasks: str = Body("ocr,detect")):
    image_bytes = await file.read()
    return analyze_image(image_bytes, tasks.split(","))

@router.post("/actions/result")
async def actions_result(payload: Dict[str, Any]):
    return {"status": "ok"}
