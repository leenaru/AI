# GraphRAG 완전 가이드: 설치부터 질의응답까지

GraphRAG(Graph + RAG)는 Microsoft에서 개발한 혁신적인 검색 증강 생성 기술로, 기존의 벡터 기반 RAG와 달리 **지식 그래프**를 활용하여 복잡한 질문에 대한 정교한 답변을 제공합니다. 이 가이드는 GraphRAG를 처음부터 설치하고 구성하여 실제 질의응답까지 수행하는 완전한 워크플로우를 제공합니다.[1][2]
## 주요 특징과 작동원리

GraphRAG는 다음과 같은 혁신적인 특징들을 가지고 있습니다:

**지식 그래프 기반 구조화**: 텍스트에서 엔티티와 관계를 추출하여 구조적 지식 그래프를 구축합니다. 이를 통해 텍스트 간의 복잡한 관계를 이해하고 활용할 수 있습니다.[3]

**계층적 클러스터링**: Leiden 커뮤니티 탐지 알고리즘을 사용하여 밀접하게 연결된 데이터를 그룹으로 식별하고 분할합니다. 각 커뮤니티에 대한 요약을 생성하여 데이터셋에 대한 전체적인 이해를 돕습니다.[3]

**이중 검색 모드**: 
- **전역 검색(Global Search)**: 전체 데이터셋을 대상으로 하는 고차원적 질문 처리[4]
- **지역 검색(Local Search)**: 특정 엔티티 중심의 상세한 질문 처리[4]

## 환경 설정 및 설치

### 1. Python 환경 구성

```bash
# Conda 환경 생성 (Python 3.10 또는 3.11 권장)
conda create --name graphrag python=3.11 -y
conda activate graphrag
```

Python 3.10-3.11 버전을 사용하는 것이 중요합니다. 이는 CUDA 및 PyTorch와의 호환성을 위한 것입니다.[5]

### 2. GraphRAG 설치

```bash
# 기본 설치
pip install graphrag

# 특정 버전 설치 (로컬 설정 최적화)
pip install graphrag==0.1.1 ollama
```

GraphRAG 0.1.1 버전은 로컬 설정에 특히 최적화되어 있습니다.[5]

### 3. Ollama 설치 (로컬 LLM 사용 시)

```bash
# Linux/Mac용 Ollama 설치
curl -fsSL https://ollama.com/install.sh | sh

# 필요한 모델 다운로드
ollama pull qwen2              # LLM 모델
ollama pull nomic-embed-text   # 임베딩 모델
```

Ollama는 로컬에서 LLM을 실행할 수 있게 해주는 도구로, 비용 효율적인 GraphRAG 구현을 가능하게 합니다.[6][5]

## 프로젝트 초기화와 설정

### 1. 작업 디렉토리 생성

```bash
# 작업 폴더 생성
mkdir ragtest
cd ragtest

# 입력 데이터 폴더 생성
mkdir input
```

### 2. GraphRAG 프로젝트 초기화

```bash
# GraphRAG 프로젝트 초기화
graphrag init --root ./
```

이 명령어는 다음 핵심 파일들을 생성합니다:[7][1]
- `.env`: 환경 변수 설정 파일
- `settings.yaml`: GraphRAG 설정 파일  
- `prompts/`: LLM 프롬프트 템플릿 폴더

### 3. 설정 파일 구성

#### .env 파일 설정:
```bash
# OpenAI 사용 시
GRAPHRAG_API_KEY=your_openai_api_key_here

# Ollama 사용 시 (더미 키)
GRAPHRAG_API_KEY=ollama
```

#### settings.yaml 파일 설정 예시:

**OpenAI 사용 시:**
```yaml
llm:
  api_key: ${GRAPHRAG_API_KEY}
  type: openai_chat
  model: gpt-4o-mini
  model_supports_json: true
  max_tokens: 4000
  request_timeout: 180.0

embeddings:
  llm:
    api_key: ${GRAPHRAG_API_KEY}
    type: openai_embedding
    model: text-embedding-ada-002

parallelization:
  stagger: 0.3
  num_threads: 50
  async_mode: threaded

encoding_model: cl100k_base
skip_workflows: []
```

**Ollama 사용 시:**
```yaml
llm:
  api_key: ollama
  type: openai_chat
  api_base: http://localhost:11434/v1
  model: qwen2
  model_supports_json: true
  max_tokens: 4000
  request_timeout: 180.0

embeddings:
  llm:
    api_key: ollama
    type: openai_embedding
    api_base: http://localhost:11434/v1
    model: nomic-embed-text

parallelization:
  stagger: 0.3
  num_threads: 10
  async_mode: threaded
```

이 설정에서 `api_base`를 `http://localhost:11434/v1`로 설정하는 것이 중요합니다. 이는 Ollama가 기본적으로 11434 포트에서 실행되며 OpenAI 호환 API 엔드포인트를 제공하기 때문입니다.[8]

## 데이터 준비와 인덱싱

### 1. 입력 데이터 준비

```bash
# input 폴더에 텍스트 파일 배치
cp your_documents.txt ./input/
```

GraphRAG는 `.txt`, `.md`, `.docx`, `.pdf` 등 다양한 파일 형식을 지원합니다.[1]

**샘플 데이터 준비 예시:**
```python
# sample_data.py
content = """
GraphRAG는 Microsoft에서 개발한 혁신적인 RAG 기술입니다.
이 기술은 지식 그래프를 활용하여 복잡한 질문에 정교한 답변을 제공합니다.
전통적인 RAG와 달리, GraphRAG는 엔티티 간의 관계를 파악하여 
더 나은 컨텍스트를 제공합니다.
"""

with open('./input/sample.txt', 'w', encoding='utf-8') as f:
    f.write(content)
```

### 2. 인덱싱 실행

```bash
# GraphRAG 인덱싱 시작
graphrag index --root ./
```

인덱싱 프로세스는 다음 단계를 거칩니다:[2][9]

1. **문서 분할**: 입력 문서를 분석 가능한 TextUnit으로 분할
2. **엔티티 추출**: LLM을 사용하여 모든 엔티티, 관계, 주요 주장을 추출  
3. **그래프 구축**: 추출된 정보로 지식 그래프 생성
4. **계층적 클러스터링**: Leiden 알고리즘으로 그래프의 계층적 클러스터링 수행
5. **요약 생성**: 각 커뮤니티에 대한 상향식(bottom-up) 요약 생성

### 3. 출력 파일 확인

인덱싱이 완료되면 다음과 같은 파케이 파일들이 생성됩니다:[10][11]

```
output/
├── communities.parquet      # 커뮤니티 정보
├── entities.parquet         # 엔티티 정보  
├── relationships.parquet    # 관계 정보
├── text_units.parquet       # 텍스트 청크 정보
├── community_reports.parquet # 커뮤니티 보고서
└── artifacts/               # 기타 산출물
```

이러한 parquet 파일들은 빠른 검색과 질의 응답에 사용됩니다.[12]

## 질의응답 실행

GraphRAG는 두 가지 주요 검색 모드를 제공합니다:

### 1. 전역 검색 (Global Search)

전체 데이터셋을 대상으로 하는 추상적이고 광범위한 질문에 사용됩니다:[13][4]

```bash
# 커맨드라인을 통한 전역 검색
graphrag query \
    --root ./ \
    --method global \
    "데이터에서 나타나는 주요 트렌드는 무엇인가요?"
```

**Python 코드 예시:**
```python
import asyncio
from graphrag.query.structured_search.global_search import GlobalSearch
from graphrag.llm.openai import OpenAIChat

async def global_search_example():
    # LLM 초기화
    llm = OpenAIChat(api_key="your_api_key")
    
    # 전역 검색 초기화 및 실행
    global_search = GlobalSearch(
        llm=llm,
        context_builder=global_context,
        token_encoder=token_encoder
    )
    
    result = await global_search.asearch("주요 테마는 무엇인가요?")
    print(result.response)

asyncio.run(global_search_example())
```

### 2. 지역 검색 (Local Search)

특정 엔티티나 개념에 집중된 질문에 사용됩니다:[14][13]

```bash
# 커맨드라인을 통한 지역 검색
graphrag query \
    --root ./ \
    --method local \
    "GraphRAG의 주요 특징은 무엇인가요?"
```

**Python 코드 예시:**
```python
from graphrag.query.structured_search.local_search import LocalSearch

async def local_search_example():
    local_search = LocalSearch(
        llm=llm,
        context_builder=local_context,
        token_encoder=token_encoder
    )
    
    result = await local_search.asearch("GraphRAG의 주요 기능")
    print(result.response)
```

### 3. 고급 질의 옵션

```bash
# 상세한 질의 옵션 사용
graphrag query \
    --root ./ \
    --method global \
    --community_level 2 \
    --response_type "Multiple Paragraphs" \
    --streaming \
    "GraphRAG가 기존 RAG와 어떻게 다른가요?"
```

주요 CLI 옵션들:[15]
- `--community_level`: 커뮤니티 계층 구조에서 사용할 레벨 (기본값: 2)
- `--response_type`: 응답 형식 ("Multiple Paragraphs", "Single Sentence" 등)
- `--streaming`: LLM 응답을 스트리밍 방식으로 반환

## 고급 구현 예시

### 완전한 GraphRAG 질의 엔진 클래스

```python
import os
import asyncio
import logging
from typing import Optional
from graphrag.llm.openai import OpenAIChat
from graphrag.query.structured_search.global_search import GlobalSearch
from graphrag.query.structured_search.local_search import LocalSearch

class GraphRAGQueryEngine:
    def __init__(self, data_dir: str, config_path: str):
        self.data_dir = data_dir
        self.config_path = config_path
        self.llm = OpenAIChat(api_key=os.getenv("GRAPHRAG_API_KEY"))
        
        # 컨텍스트 빌더 초기화
        self._init_context_builders()
        
    def _init_context_builders(self):
        """컨텍스트 빌더들을 초기화합니다."""
        # 실제 구현에서는 데이터 로더를 통해 parquet 파일들을 로드
        from graphrag.query.context_builder.entity import EntityContextBuilder
        from graphrag.query.context_builder.community import CommunityContextBuilder
        
        self.entity_context = EntityContextBuilder()
        self.community_context = CommunityContextBuilder()
        
    async def global_search(self, query: str) -> str:
        """전역 검색을 수행합니다."""
        global_search = GlobalSearch(
            llm=self.llm,
            context_builder=self.community_context,
            token_encoder=self._get_token_encoder()
        )
        
        result = await global_search.asearch(query)
        return result.response
        
    async def local_search(self, query: str) -> str:
        """지역 검색을 수행합니다."""
        local_search = LocalSearch(
            llm=self.llm,
            context_builder=self.entity_context,
            token_encoder=self._get_token_encoder()
        )
        
        result = await local_search.asearch(query)
        return result.response
        
    def _get_token_encoder(self):
        """토큰 인코더를 반환합니다."""
        import tiktoken
        return tiktoken.encoding_for_model("gpt-4")

# 사용 예시
async def main():
    engine = GraphRAGQueryEngine("./output", "./settings.yaml")
    
    # 전역 검색
    global_result = await engine.global_search(
        "데이터에서 가장 중요한 트렌드는 무엇인가요?"
    )
    print("전역 검색 결과:")
    print(global_result)
    
    # 지역 검색  
    local_result = await engine.local_search(
        "GraphRAG의 구체적인 작동 원리는?"
    )
    print("\n지역 검색 결과:")
    print(local_result)

if __name__ == "__main__":
    asyncio.run(main())
```

### 배치 처리 시스템

```python
from typing import List, Dict

class BatchGraphRAGProcessor:
    def __init__(self, query_engine: GraphRAGQueryEngine):
        self.query_engine = query_engine
        
    async def process_batch_queries(self, queries: List[Dict]) -> List[Dict]:
        """여러 질의를 배치로 처리합니다."""
        results = []
        
        for query_item in queries:
            query_text = query_item["query"]
            method = query_item.get("method", "global")
            
            try:
                if method == "global":
                    result = await self.query_engine.global_search(query_text)
                else:
                    result = await self.query_engine.local_search(query_text)
                    
                results.append({
                    "query": query_text,
                    "method": method,
                    "result": result,
                    "status": "success"
                })
            except Exception as e:
                results.append({
                    "query": query_text,
                    "method": method,
                    "error": str(e),
                    "status": "error"
                })
                
        return results
```

## 성능 최적화 및 모니터링

### 설정 최적화 예시

**고성능 설정:**
```yaml
parallelization:
  stagger: 0.1        # 낮은 지연시간
  num_threads: 100    # 높은 병렬성

chunk:
  size: 1200          # 최적 청크 크기
  overlap: 100        # 적절한 중복

llm:
  max_tokens: 16000   # 큰 컨텍스트 윈도우
  temperature: 0      # 일관된 결과
```

**비용 최적화 설정:**
```yaml
llm:
  model: gpt-4o-mini  # 더 저렴한 모델
  max_tokens: 4000    # 제한된 토큰 수

parallelization:
  num_threads: 10     # 적은 병렬 요청

embeddings:
  batch_size: 128     # 배치 처리로 효율성 증대
```

### 성능 모니터링 코드

```python
import time
import psutil
from contextlib import contextmanager

@contextmanager
def performance_monitor():
    """성능 모니터링 컨텍스트 매니저"""
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    try:
        yield
    finally:
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        print(f"실행 시간: {end_time - start_time:.2f}초")
        print(f"메모리 사용량: {end_memory - start_memory:.2f}MB")

# 사용 예시
async def monitored_query():
    engine = GraphRAGQueryEngine("./output", "./settings.yaml")
    
    with performance_monitor():
        result = await engine.global_search("복잡한 질의 예시")
        print(result)
```

## 트러블슈팅 및 최적화 팁

**하드웨어 요구사항**: NVIDIA RTX 3060 이상의 GPU, 16GB 이상의 RAM, SSD 스토리지를 권장합니다. SSD는 인덱싱 지연시간을 현저히 줄여줍니다.[5]

**네트워크 설정**: 로컬 설정의 경우 loopback 주소(127.0.0.1)를 사용하고 불필요한 방화벽을 비활성화하여 지연시간을 최소화하세요.[5]

**메모리 최적화**: 대용량 데이터셋의 경우 청크 크기를 적절히 조정하고 배치 처리를 활용하여 메모리 사용량을 관리하세요.

이 완전한 가이드를 따라하면 GraphRAG를 성공적으로 설치, 구성하고 실제 질의응답 시스템으로 활용할 수 있습니다. GraphRAG의 강력한 지식 그래프 기반 검색 능력을 통해 기존 RAG 시스템보다 훨씬 정교하고 맥락적인 답변을 얻을 수 있을 것입니다.

[1] https://www.chitika.com/graphrag-local-install-setup-using-vllm-and-ollama/
[2] https://gist.github.com/rohit-lakhanpal/1cea160ffe0de4cbcb52f2046ebdfd00
[3] https://neo4j.com/blog/developer/microsoft-graphrag-neo4j/
[4] https://microsoft.github.io/graphrag/get_started/
[5] https://chaechaecaley.tistory.com/18
[6] https://fornewchallenge.tistory.com/entry/%F0%9F%93%8AGraphRAG-%EB%A7%88%EC%9D%B4%ED%81%AC%EB%A1%9C%EC%86%8C%ED%94%84%ED%8A%B8%EC%9D%98-%EA%B7%B8%EB%9E%98%ED%94%84%EA%B8%B0%EB%B0%98-RAG-%EC%A0%91%EA%B7%BC%EB%B2%95feat-Ollama
[7] https://www.youtube.com/watch?v=6Yu6JpLMWVo
[8] https://chaechaecaley.tistory.com/20
[9] https://memgraph.com/blog/how-microsoft-graphrag-works-with-graph-databases
[10] https://microsoft.github.io/graphrag/
[11] https://learnopencv.com/graphrag-explained-knowledge-graphs-medical/
[12] https://huggingface.co/spaces/Raj95/RAG-App/blob/c3d0aba123ce067039679a9f43116e50513e643c/settings.yaml.example
[13] https://microsoft.github.io/graphrag/index/byog/
[14] https://github.com/springtiger/graphrag-ollama-azure
[15] https://microsoft.github.io/graphrag/config/init/
[16] https://microsoft.github.io/graphrag/config/yaml/
[17] https://neo4j.com/blog/developer/neo4j-graphrag-workflow-langchain-langgraph/
[18] https://github.com/TheAiSingularity/graphrag-local-ollama
[19] https://github.com/microsoft/graphrag/discussions/1181
[20] https://velog.io/@jiyoon_sw524/MS%EC%82%AC%EC%9D%98-graphRAG-%ED%86%BA%EC%95%84%EB%B3%B4%EA%B8%B0
[21] https://www.microsoft.com/en-us/research/blog/graphrag-improving-global-search-via-dynamic-community-selection/
[22] https://neo4j.com/blog/genai/what-is-graphrag/
[23] https://uoahvu.tistory.com/entry/GraphRAG-Neo4j-%EC%83%9D%EC%84%B1%ED%98%95-AI-%ED%8C%A8%ED%82%A4%EC%A7%80Neo4j-GenAI%EB%A1%9C-%EC%98%81%ED%99%94-%EC%A0%95%EB%B3%B4-%EC%A7%88%EC%9D%98%EC%9D%91%EB%8B%B5-%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B0-Cypher-%EC%BF%BC%EB%A6%AC-%EC%9E%90%EB%8F%99-%EC%83%9D%EC%84%B1
[24] https://www.microsoft.com/en-us/research/blog/introducing-drift-search-combining-global-and-local-search-methods-to-improve-quality-and-efficiency/
[25] https://docs.llamaindex.ai/en/stable/examples/query_engine/knowledge_graph_rag_query_engine/
[26] https://uoahvu.tistory.com/entry/GraphRAG-Neo4j-DB%EC%99%80-LangChain-%EA%B2%B0%ED%95%A9%EC%9D%84-%ED%86%B5%ED%95%9C-%EC%A7%88%EC%9D%98%EC%9D%91%EB%8B%B5-%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B0-Kaggle-CSV-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%A0%81%EC%9A%A9%ED%95%98%EA%B8%B0
[27] https://microsoft.github.io/graphrag/query/global_search/
[28] https://graphrag.com/concepts/intro-to-graphrag/
[29] https://kr.linkedin.com/posts/parknayeon_db-%EC%A1%B0%ED%9A%8C-%EA%B2%B0%EA%B3%BC%EC%97%90-%EB%94%B0%EB%9D%BC-%EC%8A%A4%EC%8A%A4%EB%A1%9C-%EC%BF%BC%EB%A6%AC%EB%AC%B8%EC%9D%84-%EC%88%98%EC%A0%95%ED%95%98%EB%8A%94-graphrag-agent-activity-7301043432590872576-9Fa0
[30] https://chaechaecaley.tistory.com/23
[31] https://github.com/stephenc222/example-graphrag
[32] https://machinelearningmastery.com/building-graph-rag-system-step-by-step-approach/
[33] https://www.datastax.com/ko/blog/graphrag-by-example
[34] https://velog.io/@jiyoon_sw524/Neo4j%EC%97%90-GraphRAG-%EA%B2%B0%EA%B3%BC-Parquet-%ED%8C%8C%EC%9D%BC-%EA%B0%80%EC%A0%B8%EC%98%A4%EA%B8%B0
[35] https://www.youtube.com/watch?v=Ia18KPkMDPk
[36] https://neo4j.com/blog/news/graphrag-python-package/
[37] https://github.com/microsoft/graphrag/discussions/328
[38] https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/your-first-graphrag-demo---a-video-walkthrough/4410246
[39] https://neo4j.com/blog/developer/get-started-graphrag-python-package/
[40] https://microsoft.github.io/graphrag/index/outputs/
[41] https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/36aaf6d245caed1e62efd2e171f0cd1a/f6809385-af4f-4115-b8f2-f924c2d4cd29/ac45a72e.md
