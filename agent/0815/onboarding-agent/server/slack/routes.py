from fastapi import APIRouter
from typing import Dict, Any
from server.rag.pipeline import run_pipeline

router = APIRouter()

@router.post("/kb/propose")
async def kb_propose(payload: Dict[str, Any]):
    return {"ok": True, "message": "Proposed", "payload": payload}

@router.post("/kb/approve")
async def kb_approve(payload: Dict[str, Any]):
    docs = [(payload.get("doc_id", "ops"), payload.get("answer", ""))]
    run_pipeline(docs)
    return {"ok": True, "message": "Indexed"}

@router.post("/kb/reject")
async def kb_reject(payload: Dict[str, Any]):
    return {"ok": True, "message": "Rejected", "reason": payload.get("reason")}
