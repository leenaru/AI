# GraphRAG 완전 가이드: 설치부터 질의응답까지

## 목차
1. [GraphRAG 개요](#graphrag-개요)
2. [환경 설정 및 설치](#환경-설정-및-설치)
3. [프로젝트 초기화](#프로젝트-초기화)
4. [데이터 준비](#데이터-준비)
5. [설정 파일 구성](#설정-파일-구성)
6. [인덱싱 실행](#인덱싱-실행)
7. [질의응답 실행](#질의응답-실행)
8. [고급 사용법](#고급-사용법)

## GraphRAG 개요

GraphRAG(Graph + RAG)는 Microsoft에서 개발한 혁신적인 검색 증강 생성(RAG) 기술로, 기존의 벡터 기반 유사성 검색과 달리 **지식 그래프**를 활용하여 복잡한 질문에 대한 정교한 답변을 제공합니다.

### 주요 특징
- **지식 그래프 기반**: 텍스트에서 엔티티와 관계를 추출하여 구조적 지식 그래프 구축
- **계층적 클러스터링**: Leiden 알고리즘을 사용한 커뮤니티 탐지 및 계층적 요약
- **전역/지역 검색**: 전체 데이터셋 대상의 전역 검색과 특정 엔티티 중심의 지역 검색
- **고도화된 질의 처리**: 복잡한 정보 간의 연결성을 이해한 정교한 답변 생성

## 환경 설정 및 설치

### 1. Python 환경 구성
```bash
# Conda 환경 생성 (Python 3.10 또는 3.11 권장)
conda create --name graphrag python=3.11 -y
conda activate graphrag
```

### 2. GraphRAG 설치
```bash
# GraphRAG 패키지 설치
pip install graphrag

# 또는 최신 개발 버전 설치
pip install graphrag==0.1.1
```

### 3. Ollama 설치 (로컬 LLM 사용 시)
```bash
# Linux/Mac용 Ollama 설치
curl -fsSL https://ollama.com/install.sh | sh

# 필요한 모델 다운로드
ollama pull qwen2          # LLM 모델
ollama pull nomic-embed-text  # 임베딩 모델
```

## 프로젝트 초기화

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

이 명령어는 다음 파일들을 생성합니다:
- `.env`: 환경 변수 설정 파일
- `settings.yaml`: GraphRAG 설정 파일
- `prompts/`: LLM 프롬프트 템플릿 폴더

## 데이터 준비

### 1. 입력 데이터 배치
```bash
# input 폴더에 텍스트 파일 복사
cp your_documents.txt ./input/
```

**지원하는 파일 형식**: `.txt`, `.md`, `.docx`, `.pdf` 등

### 2. 샘플 데이터 준비 예시
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

## 설정 파일 구성

### 1. .env 파일 설정
```bash
# OpenAI 사용 시
GRAPHRAG_API_KEY=your_openai_api_key_here

# Azure OpenAI 사용 시
GRAPHRAG_API_KEY=your_azure_openai_key_here
```

### 2. settings.yaml 파일 설정

#### OpenAI 사용 시:
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

#### Ollama 사용 시:
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

encoding_model: cl100k_base
skip_workflows: []
```

#### Azure OpenAI 사용 시:
```yaml
llm:
  api_key: ${GRAPHRAG_API_KEY}
  type: azure_openai_chat
  api_base: https://your-instance.openai.azure.com
  api_version: 2024-02-15-preview
  deployment_name: your-gpt-deployment-name
  model: gpt-4o-mini
  model_supports_json: true

embeddings:
  llm:
    api_key: ${GRAPHRAG_API_KEY}
    type: azure_openai_embedding
    api_base: https://your-instance.openai.azure.com
    api_version: 2024-02-15-preview
    deployment_name: your-embedding-deployment-name
    model: text-embedding-ada-002
```

## 인덱싱 실행

### 1. 인덱싱 프로세스 시작
```bash
# GraphRAG 인덱싱 실행
graphrag index --root ./
```

### 2. 인덱싱 단계별 과정
GraphRAG 인덱싱은 다음 단계들을 거칩니다:

1. **문서 분할**: 입력 문서를 TextUnit으로 분할
2. **엔티티 추출**: LLM을 사용하여 엔티티와 관계 추출
3. **그래프 구축**: 추출된 정보로 지식 그래프 생성
4. **커뮤니티 탐지**: Leiden 알고리즘으로 계층적 클러스터링
5. **요약 생성**: 각 커뮤니티에 대한 요약문 생성

### 3. 인덱싱 진행 상황 모니터링
```bash
# 로그 확인
tail -f logs/indexing-engine.log

# 출력 파일 확인
ls -la output/
```

### 4. 생성되는 주요 출력 파일들
```
output/
├── communities.parquet      # 커뮤니티 정보
├── entities.parquet         # 엔티티 정보
├── relationships.parquet    # 관계 정보
├── text_units.parquet       # 텍스트 청크 정보
├── community_reports.parquet # 커뮤니티 보고서
└── artifacts/               # 기타 산출물
```

## 질의응답 실행

### 1. 전역 검색 (Global Search)
전체 데이터셋을 대상으로 하는 광범위한 질문에 사용:

```bash
# 전역 검색 실행
graphrag query \
    --root ./ \
    --method global \
    "데이터에서 나타나는 주요 트렌드는 무엇인가요?"
```

```python
# Python 코드로 전역 검색
import asyncio
from graphrag.query.structured_search.global_search import GlobalSearch
from graphrag.llm.openai import OpenAIChat
from graphrag.config import GraphRagConfig

async def global_search_example():
    config = GraphRagConfig.from_file("./settings.yaml")
    llm = OpenAIChat(api_key="your_api_key")
    
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
특정 엔티티나 개념에 집중된 질문에 사용:

```bash
# 지역 검색 실행
graphrag query \
    --root ./ \
    --method local \
    "GraphRAG의 주요 특징은 무엇인가요?"
```

```python
# Python 코드로 지역 검색
import asyncio
from graphrag.query.structured_search.local_search import LocalSearch

async def local_search_example():
    local_search = LocalSearch(
        llm=llm,
        context_builder=local_context,
        token_encoder=token_encoder
    )
    
    result = await local_search.asearch("GraphRAG의 주요 기능")
    print(result.response)

asyncio.run(local_search_example())
```

### 3. 고급 질의 옵션
```bash
# 상세한 질의 옵션
graphrag query \
    --root ./ \
    --method global \
    --community_level 2 \
    --response_type "Multiple Paragraphs" \
    --streaming \
    "GraphRAG가 기존 RAG와 어떻게 다른가요?"
```

### 4. Python API를 통한 질의응답
```python
# complete_rag_example.py
import os
import asyncio
from graphrag.llm.openai import OpenAIChat
from graphrag.query.structured_search.global_search import GlobalSearch
from graphrag.query.structured_search.local_search import LocalSearch
from graphrag.query.context_builder.entity import EntityContextBuilder
from graphrag.query.context_builder.community import CommunityContextBuilder

class GraphRAGQueryEngine:
    def __init__(self, data_dir: str, config_path: str):
        self.data_dir = data_dir
        self.config_path = config_path
        self.llm = OpenAIChat(api_key=os.getenv("GRAPHRAG_API_KEY"))
        
        # 컨텍스트 빌더 초기화
        self._init_context_builders()
        
    def _init_context_builders(self):
        """컨텍스트 빌더들을 초기화합니다."""
        # 실제 구현에서는 데이터 로더를 통해 파케이 파일들을 로드
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

## 고급 사용법

### 1. 커스텀 프롬프트 튜닝
```bash
# 프롬프트 자동 튜닝
graphrag prompt-tune \
    --root ./ \
    --domain "your_domain" \
    --method random \
    --limit 5

# 프롬프트 수동 편집
nano prompts/entity_extraction.txt
```

### 2. 배치 처리
```python
# batch_processing.py
import asyncio
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

# 사용 예시
async def batch_example():
    engine = GraphRAGQueryEngine("./output", "./settings.yaml")
    processor = BatchGraphRAGProcessor(engine)
    
    queries = [
        {"query": "주요 트렌드는?", "method": "global"},
        {"query": "GraphRAG의 특징은?", "method": "local"},
        {"query": "데이터의 핵심 요약", "method": "global"}
    ]
    
    results = await processor.process_batch_queries(queries)
    
    for result in results:
        print(f"Query: {result['query']}")
        print(f"Result: {result.get('result', result.get('error'))}")
        print("-" * 50)

asyncio.run(batch_example())
```

### 3. 성능 모니터링
```python
# monitoring.py
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

### 4. 에러 처리 및 로깅
```python
# error_handling.py
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustGraphRAGEngine(GraphRAGQueryEngine):
    def __init__(self, data_dir: str, config_path: str, max_retries: int = 3):
        super().__init__(data_dir, config_path)
        self.max_retries = max_retries
        
    async def safe_search(self, query: str, method: str = "global") -> Optional[str]:
        """안전한 검색 실행 (재시도 로직 포함)"""
        for attempt in range(self.max_retries):
            try:
                logger.info(f"검색 시도 {attempt + 1}/{self.max_retries}: {query[:50]}...")
                
                if method == "global":
                    result = await self.global_search(query)
                else:
                    result = await self.local_search(query)
                    
                logger.info("검색 성공")
                return result
                
            except Exception as e:
                logger.error(f"검색 실패 (시도 {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    logger.error("모든 재시도 실패")
                    return None
                    
                # 재시도 전 잠시 대기
                await asyncio.sleep(2 ** attempt)
                
        return None
```

### 5. 설정 최적화 팁

#### 성능 최적화:
```yaml
# 고성능 설정
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

#### 비용 최적화:
```yaml
# 저비용 설정  
llm:
  model: gpt-4o-mini  # 더 저렴한 모델
  max_tokens: 4000    # 제한된 토큰 수

parallelization:
  num_threads: 10     # 적은 병렬 요청

embeddings:
  batch_size: 128     # 배치 처리로 효율성 증대
```

이 가이드를 따라하면 GraphRAG를 성공적으로 설치하고 구성하여 복잡한 질의응답 시스템을 구축할 수 있습니다. 추가적인 문제가 발생하면 공식 문서나 GitHub Issues를 참고하시기 바랍니다.