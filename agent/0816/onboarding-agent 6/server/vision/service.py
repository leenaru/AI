from typing import List, Dict
from .storage import persist_temp_artifacts

def analyze_image(image_bytes: bytes, tasks: List[str]) -> Dict:
    ref = persist_temp_artifacts(image_bytes, meta={"tasks": tasks}); return {"tasks": tasks, "bytes": len(image_bytes), "artifact_ref": ref}
