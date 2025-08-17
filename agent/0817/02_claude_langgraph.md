이 LangGraph 워크플로우를 분석하여 mermaid 다이어그램으로 시각화해드리겠습니다.이 mermaid 다이어그램은 코드에서 구현된 LangGraph 워크플로우를 시각화한 것입니다. 주요 구성 요소는 다음과 같습니다:

## 🔄 **메인 플로우**
1. **사용자 요청** → **의도 분류** → **워크플로우 선택**

## 📱 **Device Registration Workflow (기기 등록)**
- `validate_prerequisites` → `scan_environment` → `check_compatibility`
- `provide_setup_guidance` → `verify_connection` → `finalize_registration`
- 에러 발생 시 `handle_setup_errors`로 분기

## 🔧 **Troubleshooting Workflow (문제 해결)**
- `analyze_symptoms` → `run_diagnostics` → `rank_solutions`
- `provide_step_by_step_fix` → `verify_resolution`
- 해결 안 될 시 `escalate_support`로 에스컬레이션

## 📖 **Manual Guidance Workflow (매뉴얼 가이드)**
- `understand_query` → `search_manual` → `provide_guidance`
- 복잡한 경우 `check_understanding` → `provide_additional_help`

## 🛒 **Purchase Guide Workflow (구매 가이드)**
- `assess_needs` → `build_compatibility_matrix` → `generate_recommendations`
- `provide_comparison` → `assist_purchase`

각 워크플로우는 조건부 분기를 통해 사용자 상황에 맞는 최적의 경로를 선택하며, 에러 처리 및 재시도 로직이 포함되어 있습니다.

```
graph TD
    %% 메인 오케스트레이터 플로우
    START([사용자 요청]) --> INTENT[의도 분류 및 워크플로우 선택]
    
    INTENT --> |device registration| DR_WORKFLOW[Device Registration Workflow]
    INTENT --> |troubleshooting| TR_WORKFLOW[Troubleshooting Workflow] 
    INTENT --> |manual guidance| MG_WORKFLOW[Manual Guidance Workflow]
    INTENT --> |purchase guide| PG_WORKFLOW[Purchase Guide Workflow]

    %% Device Registration Workflow
    DR_WORKFLOW --> DR_VALIDATE[validate_prerequisites]
    DR_VALIDATE --> |continue| DR_SCAN[scan_environment]
    DR_VALIDATE --> |missing_info| DR_GUIDANCE[provide_setup_guidance]
    DR_VALIDATE --> |error| DR_ERROR[handle_setup_errors]
    
    DR_SCAN --> DR_COMPAT[check_compatibility]
    DR_COMPAT --> |compatible| DR_GUIDANCE
    DR_COMPAT --> |incompatible| DR_ERROR
    DR_COMPAT --> |need_more_info| DR_GUIDANCE
    
    DR_GUIDANCE --> DR_VERIFY[verify_connection]
    DR_VERIFY --> |success| DR_FINAL[finalize_registration]
    DR_VERIFY --> |failed| DR_ERROR
    DR_VERIFY --> |retry| DR_GUIDANCE
    
    DR_ERROR --> |retry| DR_GUIDANCE
    DR_ERROR --> |escalate| END_DR[END]
    DR_ERROR --> |complete| DR_FINAL
    
    DR_FINAL --> END_DR

    %% Troubleshooting Workflow
    TR_WORKFLOW --> TR_ANALYZE[analyze_symptoms]
    TR_ANALYZE --> TR_DIAG[run_diagnostics]
    TR_DIAG --> TR_RANK[rank_solutions]
    TR_RANK --> TR_FIX[provide_step_by_step_fix]
    
    TR_FIX --> |verify| TR_VERIFY[verify_resolution]
    TR_FIX --> |continue| TR_FIX
    
    TR_VERIFY --> |resolved| END_TR[END]
    TR_VERIFY --> |retry| TR_FIX
    TR_VERIFY --> |escalate| TR_ESCALATE[escalate_support]
    
    TR_ESCALATE --> END_TR

    %% Manual Guidance Workflow
    MG_WORKFLOW --> MG_UNDERSTAND[understand_query]
    MG_UNDERSTAND --> MG_SEARCH[search_manual]
    MG_SEARCH --> MG_PROVIDE[provide_guidance]
    
    MG_PROVIDE --> |check| MG_CHECK[check_understanding]
    MG_PROVIDE --> |complete| END_MG[END]
    
    MG_CHECK --> |additional_help| MG_HELP[provide_additional_help]
    MG_CHECK --> |complete| END_MG
    
    MG_HELP --> END_MG

    %% Purchase Guide Workflow
    PG_WORKFLOW --> PG_ASSESS[assess_needs]
    PG_ASSESS --> PG_MATRIX[build_compatibility_matrix]
    PG_MATRIX --> PG_RECOMMEND[generate_recommendations]
    PG_RECOMMEND --> PG_COMPARE[provide_comparison]
    PG_COMPARE --> PG_ASSIST[assist_purchase]
    PG_ASSIST --> END_PG[END]

    %% 스타일링
    classDef startEnd fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef workflow fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef process fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef decision fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef error fill:#ffebee,stroke:#b71c1c,stroke-width:2px

    class START,END_DR,END_TR,END_MG,END_PG startEnd
    class DR_WORKFLOW,TR_WORKFLOW,MG_WORKFLOW,PG_WORKFLOW workflow
    class DR_VALIDATE,DR_SCAN,DR_COMPAT,DR_GUIDANCE,DR_VERIFY,DR_FINAL process
    class TR_ANALYZE,TR_DIAG,TR_RANK,TR_FIX,TR_VERIFY process
    class MG_UNDERSTAND,MG_SEARCH,MG_PROVIDE,MG_CHECK,MG_HELP process
    class PG_ASSESS,PG_MATRIX,PG_RECOMMEND,PG_COMPARE,PG_ASSIST process
    class INTENT decision
    class DR_ERROR,TR_ESCALATE error
```
