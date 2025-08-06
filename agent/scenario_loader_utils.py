# scenario_loader_utils.py

import yaml
import json
import os
from pathlib import Path
from typing import Dict
from datetime import datetime
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from langgraph.graph import StateGraph, END

BASE_DIR = Path(__file__).resolve().parent.parent
SCENARIO_DIR = BASE_DIR / "scenarios"
TRANSLATION_DIR = BASE_DIR / "translations"
RULES_DIR = BASE_DIR / "rules"
VERSION_DIR = BASE_DIR / "scenario_versions"

# -------------------------------
# 🧩 1. 시나리오 상속/오버레이 로더 + 버전 관리 + 직렬화
# -------------------------------
def load_yaml(file_path: Path) -> dict:
    with open(file_path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_json(data: dict, file_path: Path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def merge_dicts(base: dict, overlay: dict) -> dict:
    for key, value in overlay.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            base[key] = merge_dicts(base[key], value)
        else:
            base[key] = value
    return base

def load_scenario(scenario_name: str, lang: str = "base", versioned: bool = False) -> dict:
    base_file = SCENARIO_DIR / "base" / f"{scenario_name}.yaml"
    overlay_file = SCENARIO_DIR / lang / f"{scenario_name}.{lang}.yaml"

    base_scenario = load_yaml(base_file)

    if overlay_file.exists():
        overlay = load_yaml(overlay_file)
        if overlay.get("extends") != scenario_name:
            raise ValueError("Overlay 시나리오의 extends 값이 잘못되었습니다.")
        base_scenario = merge_dicts(base_scenario, overlay.get("overrides", {}))

    # 직렬화 및 버전 저장
    if versioned:
        VERSION_DIR.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_path = VERSION_DIR / f"{scenario_name}_{lang}_{timestamp}.json"
        save_json(base_scenario, output_path)

    return base_scenario

# -------------------------------
# 🌍 2. 번역 매니저
# -------------------------------
class TranslationManager:
    def __init__(self, lang: str = "ko"):
        self.lang = lang
        self.translations = self.load_translations()

    def load_translations(self) -> Dict[str, str]:
        path = TRANSLATION_DIR / f"{self.lang}.yaml"
        return load_yaml(path) if path.exists() else {}

    def get(self, key: str, default: str = "") -> str:
        return self.translations.get(key, default)

    def format(self, key: str, **kwargs) -> str:
        template = self.get(key)
        return template.format(**kwargs)

# -------------------------------
# 🔧 3. 규칙 확장 파서
# -------------------------------
def load_rules(scenario_name: str, lang: str = "base") -> dict:
    path = RULES_DIR / lang / f"{scenario_name}.yaml"
    fallback_path = RULES_DIR / "base_rules.yaml"

    if path.exists():
        return load_yaml(path)
    elif fallback_path.exists():
        return load_yaml(fallback_path)
    else:
        return {}

# -------------------------------
# 🧩 4. GUI 편집기 연동용 입출력
# -------------------------------
def export_scenario_to_gui_format(scenario: dict) -> dict:
    nodes = []
    edges = []
    for state_name, state_data in scenario.get("states", {}).items():
        nodes.append({
            "id": state_name,
            "type": state_data.get("type"),
            "data": state_data,
        })
        next_state = state_data.get("next")
        if isinstance(next_state, str):
            edges.append({"source": state_name, "target": next_state})
        elif isinstance(next_state, dict):
            for _, target in next_state.items():
                edges.append({"source": state_name, "target": target})
    return {"nodes": nodes, "edges": edges}

def import_scenario_from_gui_format(gui_data: dict) -> dict:
    scenario = {"states": {}}
    for node in gui_data.get("nodes", []):
        scenario["states"][node["id"]] = node.get("data", {})
    return scenario

# -------------------------------
# ▶️ 5. LangGraph 실행 통합
# -------------------------------
def build_langgraph_from_scenario(scenario_name: str, lang: str = "base"):
    from app.handlers import function_map  # 실제 구현 함수가 정의된 모듈에서 가져옴

    scenario = load_scenario(scenario_name, lang)
    states = scenario.get("states", {})

    graph = StateGraph()

    for state_name, state_data in states.items():
                fn = function_map.get(state_name, lambda x: x)
        graph.add_node(state_name, fn)

    for state_name, state_data in states.items():
        next_info = state_data.get("next")
        if isinstance(next_info, str):
            graph.add_edge(state_name, next_info)
        elif isinstance(next_info, dict):
            graph.add_conditional_edges(state_name, lambda x: x.get("__key__"), next_info)

    graph.set_entry_point(scenario.get("start", "start"))
    return graph.compile()

# -------------------------------
# 🖥️ 6. FastAPI GUI API 라우터
# -------------------------------
router = APIRouter()

@router.get("/scenario/load")
def api_load_scenario(name: str, lang: str = "base"):
    try:
        scenario = load_scenario(name, lang)
        gui_format = export_scenario_to_gui_format(scenario)
        return JSONResponse(content=gui_format)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/scenario/save")
def api_save_scenario(name: str, lang: str = "base", gui_data: dict = {}):
    try:
        scenario = import_scenario_from_gui_format(gui_data)
        path = SCENARIO_DIR / lang / f"{name}.{lang}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump({"extends": name, "overrides": scenario["states"]}, f, allow_unicode=True)
        return {"status": "saved", "path": str(path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
