# 요약 (TL;DR)

Weaviate는 **오브젝트(메타데이터) + 벡터**를 함께 저장하고, **벡터 검색·키워드(BM25F) 검색·하이브리드 검색·재랭킹·생성형 RAG**를 한 번의 쿼리로 조합할 수 있는 **AI-네이티브 벡터 데이터베이스**입니다. HNSW 기반 ANN 인덱스, 컬렉션 단위 스키마, 다중 벡터( Named Vectors ), 멀티 테넌시, 튜너블 컨시스턴시(복제), RBAC 보안, 백업(S3/GCS/Azure/FS) 등을 제공합니다. Python v4/JS/Go/Java 클라이언트가 있으며 v4는 gRPC로 빠르고 타입이 엄격합니다. ([docs.weaviate.io][1], [weaviate.io][2])

**바로 시작**: (1) Docker로 로컬 실행 → (2) Python v4 클라이언트 연결 → (3) 컬렉션(스키마) 생성 → (4) 배치 적재(내장 벡터라이저 또는 BYO 임베딩) → (5) `hybrid()/bm25()/near_*()` 검색 + 필터/재랭크 → (6) 필요 시 RBAC·백업·복제 설정. 아래 ‘빠른 시작’과 ‘실전 RAG 미니 서비스’ 예제 코드를 따라 하시면 됩니다. 문서 링크는 모두 클릭해 확인하실 수 있도록 첨부했습니다.

---

# 1) Weaviate 한눈에 보기

* **핵심 개념**:

  * *컬렉션(Collection)* = 스키마·인덱스 단위 테이블.
  * *벡터 인덱스* = HNSW/Flat. *역인덱스* = BM25F/필터용. 두 인덱스를 함께 써서 **하이브리드 검색** 가능. ([docs.weaviate.io][3], [weaviate.io][2])
  * *Named Vectors* = 한 오브젝트에 여러 임베딩(예: `title`, `body`, `image`)을 저장·검색. 멀티모달/다국어에 유리. ([docs.weaviate.io][4])
  * *하이브리드 검색* = BM25F(희소) + 벡터(밀집)를 가중 결합(`alpha`)하거나 상대 점수 융합. Named Vectors 사용 시 `target_vector` 지정. ([docs.weaviate.io][5])
  * *재랭커/생성형* = Cohere/Jina/OpenAI 등 모듈로 재랭크·요약·답변 생성 연계. ([docs.weaviate.io][6])
  * *복제 & 컨시스턴시* = Dynamo 스타일 리더리스 데이터 복제(튜너블 읽기/쓰기 일관성), 메타데이터는 Raft. ([docs.weaviate.io][7])
  * *보안* = API Key/OIDC + **RBAC**(v1.29+ GA)로 역할/권한 관리. ([docs.weaviate.io][8])
  * *백업* = S3/GCS/Azure/로컬 파일시스템에 단일 API로 백업/복원. ([docs.weaviate.io][9])

---

# 2) 빠른 시작 (로컬 Docker → Python v4)

## 2.1 로컬 실행

가장 쉬운 방법은 Docker입니다. 공식 가이드는 다음 순서로 안내합니다: 설치 → 컨테이너 실행 → 헬스체크. ([docs.weaviate.io][10])

```bash
# 1) Docker 데몬 실행
# 2) Weaviate 컨테이너 기동 (예시)
docker run -d --name weaviate \
  -p 8080:8080 -p 50051:50051 \
  -e QUERY_DEFAULTS_LIMIT=25 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH="/var/lib/weaviate" \
  cr.weaviate.io/semitechnologies/weaviate:1.32.6
```

> 참고: 모듈(예: `text2vec-openai`, `reranker-cohere`, `backup-s3`)을 쓰려면 `ENABLE_MODULES` 등 환경변수를 추가합니다. 구성 옵션은 **Configuration**/**Modules** 문서를 확인하세요. ([docs.weaviate.io][11])

## 2.2 Python v4 클라이언트 연결

v4는 gRPC를 사용하여 더 빠르고 타입이 엄격한 API를 제공합니다. 로컬 기본 연결은 다음과 같습니다. ([docs.weaviate.io][10], [weaviate.io][12])

```python
import weaviate
client = weaviate.connect_to_local()  # 8080/50051
print(client.is_ready())   # True 기대
# ... 사용 후
client.close()
```

* WCS(클라우드)나 OIDC/API Key 인증, 컨텍스트 매니저 예시는 문서/레퍼런스 참고. ([weaviate-python-client.readthedocs.io][13], [docs.weaviate.io][14])

---

# 3) 스키마(컬렉션) 설계

## 3.1 텍스트 전용(단일 벡터) 컬렉션

```python
import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Property, DataType, Configure, Tokenization

client = weaviate.connect_to_local()

# 예: 문서(title, body)를 단일 벡터 공간으로 색인
docs = client.collections.create(
    name="Document",
    properties=[
        Property(name="title", data_type=DataType.TEXT,
                 tokenization=Tokenization.WORD, index_searchable=True),
        Property(name="body", data_type=DataType.TEXT,
                 tokenization=Tokenization.WORD, index_searchable=True),
        Property(name="url", data_type=DataType.TEXT, index_filterable=True),
        Property(name="lang", data_type=DataType.TEXT, index_filterable=True),
        Property(name="created_at", data_type=DataType.DATE, index_range_filters=True),
    ],
    # 역인덱스(BM25F/필터) 파라미터
    inverted_index_config=Configure.inverted_index(
        bm25_k1=1.2, bm25_b=0.75  # 기본값 예시
    ),
    # 벡터라이저: 로컬 임베딩 사용(BYO)일 경우 none처럼 설정(예시)
    vectorizer_config=Configure.Vectorizer.none(),  # BYO 임베딩
)
```

* 역인덱스 구성(검색·필터·범위), 토크나이제이션 옵션(한국어는 형태소/공백/단어 등 시나리오별 선택), BM25 파라미터는 컬렉션·프로퍼티 레벨에서 설정합니다. ([docs.weaviate.io][15])

## 3.2 Named Vectors (다중 벡터)

타이틀/본문/이미지 등 **여러 임베딩**을 같은 오브젝트에 저장·검색하려면 Named Vectors를 사용합니다. ([docs.weaviate.io][4])

```python
mv = client.collections.create(
    name="ArticleNV",
    properties=[
        Property(name="title", data_type=DataType.TEXT, index_searchable=True),
        Property(name="body", data_type=DataType.TEXT, index_searchable=True),
        Property(name="image_url", data_type=DataType.TEXT, index_filterable=True),
    ],
    vectorizer_config=Configure.NamedVectors.multi(
        {
          "title_vec": Configure.Vectorizer.none(),   # BYO
          "body_vec":  Configure.Vectorizer.none(),   # BYO
        }
    ),
)
```

> 하이브리드 검색 시 `target_vector`를 지정해야 합니다. 멀티-타겟 벡터 동시 검색(여러 벡터 공간을 합성)도 지원합니다. ([docs.weaviate.io][5])

---

# 4) 데이터 적재 (싱글/배치 & BYO 임베딩)

```python
from uuid import uuid4
import numpy as np

docs = client.collections.get("Document")

# 4.1 단건 추가(벡터 없이 = 외부 벡터라이저 사용 예정)
doc_id = docs.data.insert(
    properties={"title": "하이브리드 검색 가이드", "body": "...", "lang": "ko"},
)

# 4.2 BYO 임베딩(예: sentence-transformers)
vec = np.random.rand(384).astype(np.float32)  # 예시: 실제론 모델로 생성
docs.data.update(
    uuid=doc_id,
    properties=None,
    vector=vec
)

# 4.3 배치 적재
with docs.batch.dynamic() as batch:  # 자동 배치/재시도
    for i in range(1000):
        batch.add_object(
          properties={"title": f"title-{i}", "body": "..."},
          vector=np.random.rand(384).astype(np.float32)
        )
```

* v3 시절의 배치 사이즈 추천치 로직과 달리 v4는 단순화되었지만, **batch.dynamic/fixed\_size**가 제공됩니다. 대량 적재 시 병렬/재시도/백오프를 조절하세요. (v3 배치 가이드의 개념은 참고용) ([weaviate-python-client.readthedocs.io][16])

---

# 5) 검색·필터·재랭킹·생성형

## 5.1 키워드(BM25F)

```python
from weaviate.classes.query import BM25Operator

docs = client.collections.get("Document")
res = docs.query.bm25(
  query="하이브리드 검색 성능",
  operator=BM25Operator.and_(),  # and/or/not 조합
  limit=5,
)
for o in res.objects:
    print(o.properties["title"])
```

* BM25는 토크나이제이션 설정에 영향을 받습니다(단어/필드/공백 등). ([weaviate.io][17], [docs.weaviate.io][18])

## 5.2 벡터 검색 (near\_\*)

```python
# near_vector
qv = np.random.rand(384).astype(np.float32)
res = docs.query.near_vector(near_vector=qv, limit=5)
```

예시는 v4 아카데미/쿼리 예제를 참고하십시오. ([weaviate.io][19])

## 5.3 하이브리드 검색 (BM25F + 벡터)

```python
# 간단 하이브리드: 질의문만으로 (내부에서 벡터화 or target_vector 지정)
res = docs.query.hybrid(query="저지연 한국어 RAG", alpha=0.5, limit=5)

# Named Vectors가 있다면:
# reviews.query.hybrid(query="French Riesling", target_vector="title_country", limit=3)
```

* `alpha`로 희소/밀집 가중치를 조정합니다. Named Vectors 컬렉션에서는 `target_vector` 지정이 필요합니다. ([docs.weaviate.io][5])

## 5.4 필터(where)·정렬·집계(aggregate)

* 필터는 프로퍼티 타입별 연산(=, >, <, contains, isNull 등)을 지원합니다.
* **Aggregate**는 통계/그룹별 메타를 반환하며 `groupBy`로 그룹화가 가능합니다(제약 확인). ([docs.weaviate.io][20])

## 5.5 재랭킹·생성형(RAG)

* Cohere/Jina/OpenAI 등의 **재랭커·생성형 모듈**을 켜면 `query.rerank()`나 `generate.*` 네임스페이스로 한 번에 파이프라인을 구성할 수 있습니다. ([docs.weaviate.io][6])
* Python v4 아카데미의 *Single prompt generation* 예시를 참고하세요. ([docs.weaviate.io][21])

---

# 6) 관계 모델링 (Cross-references)

오브젝트 간 \*\*참조(그래프 엣지)\*\*를 만들어 관계형 질의를 할 수 있습니다. (예: `Document` ↔ `Chunk`) ([docs.weaviate.io][22])

```python
from weaviate.classes.config import ReferenceProperty, Property, DataType

# 1) 참조 프로퍼티 추가
client.collections.get("Document").config.add_reference(
    ReferenceProperty(name="hasChunks", target_collection="Chunk")
)

# 2) 참조 추가
doc = client.collections.get("Document")
doc.data.reference_add(
    from_uuid=document_id,
    from_property="hasChunks",
    to=chunk_id
)
```

> 크로스 레퍼런스는 단방향입니다. 양방향이 필요하면 각 컬렉션에 참조 프로퍼티를 각각 추가하세요. ([docs.weaviate.io][23])

---

# 7) 운영 기능 (보안/RBAC·복제/컨시스턴시·백업)

## 7.1 인증/인가

* **API Key/OIDC**, **RBAC(v1.29+ GA)** 지원. 역할(읽기/쓰기/스키마/백업 등)과 사용자/토큰을 매핑합니다. 클라우드에서는 v1.30+ 신규 클러스터에 RBAC가 기본 활성화됩니다. ([docs.weaviate.io][8])

## 7.2 복제·일관성

* 데이터 복제는 리더리스(가용성 우선) 구조이며 \*\*읽기/쓰기 컨시스턴시 레벨(ONE/QUORUM/ALL)\*\*을 튜닝할 수 있습니다. 쿼리 시 컬렉션 핸들에 `with_consistency_level()`로 지정 가능합니다. 메타데이터는 Raft로 강한 일관성을 보장합니다. ([docs.weaviate.io][24], [weaviate-python-client.readthedocs.io][25])

## 7.3 백업/복원

* **S3/GCS/Azure/Filesystem** 백업 백엔드를 모듈로 활성화하고 단일 API로 백업/복원이 가능합니다. (복원 시 버전/클래스 충돌 제약 참고) ([docs.weaviate.io][9], [Weaviate Community Forum][26])

---

# 8) 실전: **문서 RAG 미니 서비스** (로컬·무상 임베딩 예시)

아래 예제는 **내장 벡터라이저 없이(BYO)** Sentence-Transformers로 임베딩을 만들어 Weaviate에 적재하고, 하이브리드 검색 + 간단 생성형(외부 LLM 연계는 선택)을 보여줍니다.

## 8.1 의존성

```bash
pip install weaviate-client==4.* sentence-transformers fastapi uvicorn pydantic
```

## 8.2 스키마 & 적재 스크립트(`ingest.py`)

```python
import weaviate, weaviate.classes as wvc
from weaviate.classes.config import Property, DataType, Configure, Tokenization
from sentence_transformers import SentenceTransformer
import glob, json, numpy as np

client = weaviate.connect_to_local()

# 1) 컬렉션 생성(없으면)
name = "DocsRAG"
if not client.collections.exists(name):
    client.collections.create(
        name=name,
        properties=[
            Property(name="title", data_type=DataType.TEXT, tokenization=Tokenization.WORD, index_searchable=True),
            Property(name="body", data_type=DataType.TEXT,  tokenization=Tokenization.WORD, index_searchable=True),
            Property(name="source", data_type=DataType.TEXT, index_filterable=True),
        ],
        inverted_index_config=Configure.inverted_index(),  # 기본 BM25F
        vectorizer_config=Configure.Vectorizer.none(),     # BYO 벡터
    )

col = client.collections.get(name)

# 2) 임베딩 모델
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 3) 배치 적재
files = glob.glob("./data/*.json")
with col.batch.dynamic() as batch:
    for f in files:
        obj = json.load(open(f, "r", encoding="utf-8"))
        vec = model.encode(obj["body"]).astype(np.float32)
        batch.add_object(properties=obj, vector=vec)

client.close()
```

## 8.3 검색/질의 API (FastAPI) — `app.py`

```python
from fastapi import FastAPI, Query
import weaviate, numpy as np
from pydantic import BaseModel

app = FastAPI()
client = weaviate.connect_to_local()
col = client.collections.get("DocsRAG")

class SearchResult(BaseModel):
    title: str
    body: str
    source: str
    score: float

@app.get("/search", response_model=list[SearchResult])
def search(q: str = Query(..., description="질의문"),
           k: int = 5, alpha: float = 0.5):
    # 하이브리드 검색
    res = col.query.hybrid(query=q, alpha=alpha, limit=k)
    out = []
    for o in res.objects:
        p = o.properties
        out.append(SearchResult(title=p["title"], body=p["body"],
                                source=p.get("source",""), score=o.metadata.score))
    return out

@app.on_event("shutdown")
def shutdown():
    client.close()
```

실행: `uvicorn app:app --reload --port 8000`

> 하이브리드 파라미터/필터/재랭킹/Named Vectors 확장은 위 절의 API를 그대로 조합하면 됩니다. 쿼리 문법과 예제는 문서를 참고하세요. ([weaviate.io][19], [docs.weaviate.io][5])

---

# 9) 성능·스케일·운영 체크리스트

* **인덱스 튜닝**

  * *HNSW*의 `efConstruction / maxConnections / ef`(쿼리 시) 조절로 정확도·지연을 균형화합니다. 벡터 인덱스 설정은 스키마의 `vectorIndexConfig`에서 관리합니다. ([weaviate.io][2])
  * **역인덱스**는 필요한 필드만 `indexSearchable/indexFilterable/indexRangeFilters`를 켜서 저장공간과 적재 시간을 절약합니다. ([docs.weaviate.io][27])
* **다국어**

  * 한국어 텍스트는 토크나이제이션/스톱워드 구성을 데이터 특성에 맞춰 조정하세요(예: 조사/어미 처리, 정확한 필드 검색이 필요한 경우 `FIELD` 토큰화 등). ([docs.weaviate.io][28])
* **하이브리드 & 재랭크**

  * 쿼리 길이·오타·용어중심 질의가 많다면 `alpha`를 키우고, 의미 유사성이 중요하면 줄이는 식으로 조정합니다. 재랭커로 상위 n개를 정밀 정렬. ([docs.weaviate.io][5])
* **Named Vectors / 멀티타겟**

  * 다국어(ko/en), 필드별(title/body), 멀티모달(text/image) 임베딩을 함께 저장해 상황별 검색 품질을 끌어올립니다. 멀티타겟 조인 전략(최소/합/평균/가중)을 실험하세요. ([docs.weaviate.io][29])
* **복제·컨시스턴시**

  * 읽기 QPS 확장 목적이면 `read=ONE`, 데이터 신뢰성이 더 중요하면 `QUORUM/ALL`로 높입니다(쓰기 일관성도 동일). ([docs.weaviate.io][30])
* **보안/RBAC**

  * 운영 환경에서는 익명 접근을 끄고(OIDC/API Key), 역할(데이터 쓰기/읽기/스키마/백업)을 명시적으로 분리하세요. ([docs.weaviate.io][31])
* **백업/복원**

  * 정기 스냅샷 + 버전 호환성 체크(상위 버전 백업을 하위 버전에서 복원 불가 등). 클라우드 사용 시 관리형 버킷 제공 여부를 확인하세요. ([docs.weaviate.io][9], [Weaviate Community Forum][32])

---

# 10) 자주 쓰는 코드 스니펫 모음

* **클라이언트 연결(로컬/클라우드/임베디드/커스텀) 예시**: v4 helper 함수 레퍼런스. ([weaviate-python-client.readthedocs.io][13])
* **컬렉션 생성**: Python v4 예제(속성·데이터타입·토크나이제이션). ([docs.weaviate.io][33])
* **BM25 연산자(and/or/not)**: Python v4 예제. ([weaviate.io][17])
* **하이브리드 with Named Vectors**: `target_vector` 예제. ([docs.weaviate.io][5])
* **멀티타겟 벡터 검색**: 조인 전략(최소/합/평균/가중/상대가중). ([docs.weaviate.io][29])
* **Cross-references**: 정의·추가·양방향 구성 가이드. ([docs.weaviate.io][22])
* **집계(Aggregate/groupBy)**: 개념·제약. ([docs.weaviate.io][20])

---

# 11) 응용 서비스 아이디어

1. **제품 검색**: 속성 필터(브랜드/가격) + 하이브리드 질의 + 재랭크, 추천은 cross-ref로 “함께 본 상품” 모델링.
2. **지식기반 RAG**: 문서/청크를 cross-ref로 연결(문서↔청크), Named Vectors(제목/본문) + 하이브리드 + 생성형 요약. ([docs.weaviate.io][34])
3. **이미지/텍스트 멀티모달 검색**: `title_vec`(텍스트), `image_vec`(CLIP) 동시 운용, 멀티타겟 조인 전략으로 정확도 향상. ([docs.weaviate.io][35])

---

# 12) 마이그레이션·버전 호환

* Python v4는 Weaviate ≥ **1.23.7** 필요. v3→v4 마이그레이션 가이드를 참고하세요. ([docs.weaviate.io][36])

---

# 13) 추가 참고 링크(클릭)

* **개요/콘셉트**: 벡터·인덱싱·하이브리드 검색, 문서 메인. ([docs.weaviate.io][1])
* **Python v4 시작**: 연결/컬렉션/쿼리 아카데미. ([docs.weaviate.io][10], [weaviate.io][19])
* **HNSW/인덱스 튜닝**: 벡터 인덱스 설정. ([weaviate.io][2])
* **Named Vectors/멀티타겟**: 설계·검색 가이드. ([docs.weaviate.io][4])
* **RBAC/인증**: OIDC/API Key, 역할 관리. ([docs.weaviate.io][8])
* **복제/컨시스턴시**: 철학·클러스터 구조·튜너블 일관성. ([docs.weaviate.io][24])
* **백업**: S3/GCS/Azure/FS 가이드. ([docs.weaviate.io][9])

---

## 마무리

요청 주신 대로 **처음 사용하는 분·응용 서비스 제작자** 관점에서 Weaviate의 개념 → 배포 → 스키마 → 적재 → 검색/RAG → 운영(보안/복제/백업)까지 한 번에 따라 할 수 있는 매뉴얼을 정리해 드렸습니다.
원하시면 \*\*귀하의 실제 도메인(예: IoT/스마트홈 트러블슈팅, 다국어 RAG, 온프레미스/온디바이스 제약)\*\*에 맞춰 **스키마 설계·토크나이제이션·하이브리드 튜닝(alpha/재랭크)·RBAC 정책**을 바로 적용한 “맞춤형 보일러플레이트”도 만들어 드리겠습니다.

[1]: https://docs.weaviate.io/academy/py/vector_index/hnsw?utm_source=chatgpt.com "HNSW index in depth - Weaviate Documentation"
[2]: https://weaviate.io/blog/hybrid-search-explained?utm_source=chatgpt.com "Hybrid Search Explained"
[3]: https://docs.weaviate.io/academy/py/zero_to_mvp/schema_and_imports/data_structure?utm_source=chatgpt.com "Data structure in Weaviate"
[4]: https://docs.weaviate.io/weaviate/config-refs/collections?utm_source=chatgpt.com "Collection definition"
[5]: https://docs.weaviate.io/weaviate/search/hybrid?utm_source=chatgpt.com "Hybrid search | Weaviate Documentation"
[6]: https://docs.weaviate.io/weaviate/modules?utm_source=chatgpt.com "Reference - Modules | Weaviate Documentation"
[7]: https://docs.weaviate.io/weaviate/concepts/replication-architecture/cluster-architecture?utm_source=chatgpt.com "Cluster Architecture"
[8]: https://docs.weaviate.io/weaviate/configuration/authz-authn?utm_source=chatgpt.com "Authentication and authorization"
[9]: https://docs.weaviate.io/deploy/configuration/backups?utm_source=chatgpt.com "Backups"
[10]: https://docs.weaviate.io/weaviate/quickstart/local?utm_source=chatgpt.com "Quickstart: Locally hosted"
[11]: https://docs.weaviate.io/weaviate/configuration?utm_source=chatgpt.com "How to configure Weaviate"
[12]: https://weaviate.io/blog/py-client-v4-release?utm_source=chatgpt.com "Weaviate Python client (v4) goes GA"
[13]: https://weaviate-python-client.readthedocs.io/en/v4.4.0/_modules/weaviate/connect/helpers.html?utm_source=chatgpt.com "Source code for weaviate.connect.helpers"
[14]: https://docs.weaviate.io/weaviate/client-libraries/python/notes-best-practices?utm_source=chatgpt.com "Notes and best practices"
[15]: https://docs.weaviate.io/weaviate/config-refs/indexing/inverted-index?utm_source=chatgpt.com "Inverted index | Weaviate Documentation"
[16]: https://weaviate-python-client.readthedocs.io/en/v4.16.8/weaviate.collections.html?utm_source=chatgpt.com "weaviate.collections"
[17]: https://weaviate.io/developers/weaviate/search/bm25?utm_source=chatgpt.com "Keyword search | Weaviate Documentation"
[18]: https://docs.weaviate.io/academy/py/zero_to_mvp/queries_2/bm25?utm_source=chatgpt.com "BM25 (Keyword) searches | Weaviate Documentation"
[19]: https://weaviate.io/?utm_source=chatgpt.com "Weaviate: The AI-native database developers love"
[20]: https://docs.weaviate.io/weaviate/search/aggregate?utm_source=chatgpt.com "Aggregate data"
[21]: https://docs.weaviate.io/academy/py/starter_text_data/text_rag/single_prompt?utm_source=chatgpt.com "'Single prompt' generation"
[22]: https://docs.weaviate.io/weaviate/manage-collections/cross-references?utm_source=chatgpt.com "Cross-references"
[23]: https://docs.weaviate.io/weaviate/tutorials/cross-references?utm_source=chatgpt.com "Manage relationships with cross-references"
[24]: https://docs.weaviate.io/weaviate/concepts/replication-architecture/philosophy?utm_source=chatgpt.com "Philosophy | Weaviate Documentation"
[25]: https://weaviate-python-client.readthedocs.io/en/v4.4.3/weaviate.collections.html?utm_source=chatgpt.com "weaviate.collections package"
[26]: https://forum.weaviate.io/t/how-can-we-extract-aws-backup-restore-parameter-information-from-the-weaviate-client-in-python/1241?utm_source=chatgpt.com "How can we extract AWS backup-restore parameter information from ..."
[27]: https://docs.weaviate.io/weaviate/concepts/indexing?utm_source=chatgpt.com "Indexing | Weaviate Documentation"
[28]: https://docs.weaviate.io/weaviate/concepts/search/keyword-search?utm_source=chatgpt.com "Keyword Search (BM25) | Weaviate Documentation"
[29]: https://docs.weaviate.io/weaviate/search/multi-vector?utm_source=chatgpt.com "Multiple target vectors"
[30]: https://docs.weaviate.io/weaviate/concepts/replication-architecture/motivation?utm_source=chatgpt.com "Use Cases (Motivation) | Weaviate Documentation"
[31]: https://docs.weaviate.io/deploy/configuration/authorization?utm_source=chatgpt.com "Authorization"
[32]: https://forum.weaviate.io/t/problems-restoring-from-weaviate-backup/2489?utm_source=chatgpt.com "Problems restoring from weaviate backup - Support"
[33]: https://docs.weaviate.io/weaviate/config-refs/datatypes?utm_source=chatgpt.com "Property data types"
[34]: https://docs.weaviate.io/weaviate/starter-guides/managing-collections?utm_source=chatgpt.com "Collection definitions (schemas)"
[35]: https://docs.weaviate.io/academy/py/named_vectors?utm_source=chatgpt.com "220 Named vectors"
[36]: https://docs.weaviate.io/weaviate/client-libraries/python?utm_source=chatgpt.com "Python | Weaviate Documentation"


---
---
---

아래에 **요약본 → 상세 설명** 순으로 정리했습니다. 필요한 코드와 설계안까지 한 번에 보실 수 있습니다.

---

## 요약 (TL;DR)

1. **Weaviate vs FAISS**

* **FAISS**는 *라이브러리*입니다. 애플리케이션 프로세스 안에서 벡터 인덱스를 직접 다루며 GPU 가속( CUDA )과 다양한 인덱스(Flat/IVF/PQ/HNSW 등)를 제공합니다. 하지만 **스키마/메타데이터, BM25/필터, 복제·백업, RBAC, 클러스터 운영** 같은 DB 기능은 스스로 구현해야 합니다. ([Faiss][1], [GitHub][2])
* **Weaviate**는 \*AI-네이티브 벡터 데이터베이스(서버)\*입니다. HNSW 기반 ANN + **BM25 하이브리드 검색**, **Named Vectors**, 필터/집계, **복제·일관성**, **RBAC**, **백업(S3/GCS/Azure/FS)**, **PQ/SQ 등 벡터 압축**을 내장합니다. 운영·보안·확장성이 필요하면 Weaviate가 유리합니다. ([Weaviate Documentation][3], [Weaviate Python Client][4])

2. **Python 3 지원 여부**

* Weaviate **Python v4 클라이언트는 Python 3.9+에서 테스트**되었고, 서버는 Weaviate \*\*v1.23.7+\*\*와 호환됩니다. 즉, Python 3 환경(권장 3.9 이상)에서 정상 사용 가능합니다. ([GitHub][5], [Weaviate Documentation][6])

3. **귀하의 프로젝트 적용 설계(요약)**

* 온디바이스 **Gemma-3n**(의도/슬롯·프롬프트 전처리) + 온프레미스 **Weaviate**(하이브리드 검색, 다국어/멀티모달 Named Vectors, 재랭킹, RBAC, 백업) + 서버측 **GraphRAG/HQ-RAG**(FAISS/BM25/RRF 조합 유지 가능) 구조를 제안드립니다.
* 스키마: `DeviceManual`, `ErrorCode`, `FixStep`, `FAQ`, `VideoCue` 등 컬렉션과 교차참조로 **문서↔청크↔오류코드**를 그래프처럼 연결. 다국어(`ko`, `en`)·필드별(`title`, `body`, `image`) **Named Vectors**로 검색 품질을 튜닝합니다. ([Weaviate Documentation][7])
* 운영: **복제/일관성(ONE/QUORUM/ALL)**, **RBAC**, **S3/GCS/Azure 백업**, **PQ/SQ 압축**으로 비용·성능·보안을 균형화합니다. ([Weaviate Documentation][8])

---

## 1) Weaviate ↔ FAISS 비교 분석 (상세)

### 쓰임새/역할

* **FAISS**: 밀집 벡터 **유사도 검색/클러스터링 라이브러리**. C++/Python 바인딩, **GPU 가속**, 다양한 인덱스(Flat, IVF, HNSW, PQ, 조합형 IndexHNSWPQ 등). 인덱스 선택·튜닝은 앱에서 직접 담당. ([Faiss][1], [GitHub][9])
* **Weaviate**: **서버형 벡터 DB**. 데이터 오브젝트(메타데이터) + 벡터를 함께 저장하고 **BM25 키워드 + 벡터 하이브리드** 검색, **필터/집계**, **스키마·크로스레퍼런스**, **RBAC/복제/백업**을 제공. 기본 인덱스는 **HNSW**(옵션으로 PQ/SQ 등 압축). ([Weaviate Documentation][3])

### 검색 기능

* **FAISS**: 벡터 유사도(브루트포스/근사) + 군집화. **BM25나 필터**는 별도 구현 또는 다른 엔진과 결합 필요. ([Faiss][1])
* **Weaviate**: **하이브리드(BM25+벡터, `alpha` 가중)**, 벡터 전용(`near_vector`), 키워드 전용(`bm25`), **Named Vectors**(한 오브젝트에 다중 임베딩) + **멀티타겟 조합** 등 고급 검색 제공. ([Weaviate Documentation][10])

### 스키마/관계

* **FAISS**: 스칼라 메타데이터/필터/조인은 애플리케이션 DB(예: PostgreSQL/Elastic)와 **직접 통합**해야 함.
* **Weaviate**: 컬렉션 스키마, **속성 인덱스(BM25F/필터/범위)**, **크로스레퍼런스**로 문서↔청크↔개체 관계 조회 가능. ([Weaviate Documentation][11])

### 운영·보안·확장

* **FAISS**: 인덱스 파일 관리, 샤딩/복제/업그레이드/롤백/백업·복원, 접근제어를 **직접 설계**해야 함. GPU/멀티GPU 활용은 가능. ([GitHub][9], [bge-model.com][12])
* **Weaviate**: **복제/튜너블 일관성(ONE/QUORUM/ALL)**, **RBAC**, **백업 모듈(S3/GCS/Azure/FS)**, **Zero-downtime 업그레이드**, **리소스 압축(PQ/SQ/BQ/RQ)** 등 운영 기능 내장. ([Weaviate Documentation][8])

### 성능/튜닝 관점

* **FAISS**: **GPU 가속**과 다양한 조합 인덱스(IVF+PQ, IVF\_HNSW 등)로 초대규모/저지연을 공들여 튜닝 가능. 다만 운영 스택을 스스로 갖춰야 함. ([GitHub][13], [Faiss][14])
* **Weaviate**: **HNSW 파라미터(ef/efConstruction/maxConnections)**, **동적 ef**, **PQ/SQ 압축**으로 메모리/지연/정확도 트레이드오프를 조절. 벤치·사이징 가이드 제공. ([Weaviate Documentation][15])

> **결론**:
>
> * **로컬 임베디드·경량·초저지연**(특히 GPU 적극 활용) + 자체 백엔드를 이미 갖고 있다면 **FAISS**.
> * **서비스형 검색/RAG 백엔드**로 **스키마·하이브리드·보안·복제·백업까지 포함**한 일체형 솔루션이 필요하면 **Weaviate**가 개발·운영 총비용(TCO)을 크게 줄입니다. (필요 시 “온디바이스 FAISS + 서버 Weaviate” 혼합도 가능)

---

## 2) “Python3 에서는 안되는거야?” — 지원 범위

* **가능합니다.** Weaviate **Python v4 클라이언트는 Python 3.9+에서 테스트**되었고, 서버는 \*\*Weaviate 1.23.7+\*\*와 호환됩니다. (일반적으로 Python 3.10\~3.12 환경에서 널리 사용) ([GitHub][5], [Weaviate Documentation][6])

> 설치 예:
> `pip install weaviate-client==4.*` (서버는 1.23.7+ 권장)

---

## 3) 귀하의 프로젝트에의 적용 설계(상세)

귀하는 **온디바이스 Gemma-3n**(의도 인식/슬롯 채우기/카메라 기반 가이드) + **온프레미스·하이브리드 서버(LLMaaS, GraphRAG/HQ-RAG)** 아키텍처를 지향하고 계십니다. 여기에 Weaviate를 다음처럼 배치하는 구성을 제안드립니다.

### 3.1 전체 아키텍처(개요)

* **클라이언트(모바일)**:

  * Gemma-3n 온디바이스 NLU(빠른 의도/슬롯), 카메라 스트림 전처리(오류코드/LED 패턴 인식), 간단 FAQ 캐시(선택 시 FAISS 로컬 소형 인덱스).
* **API 게이트웨이/오케스트레이터(LangGraph)**:

  * 의도 → 태스크 매핑 → **Weaviate 질의(하이브리드)** → 재랭크 → 생성(요약/답변) → 툴 실행(연결 가이드, 체크리스트).
* **지식 저장소(Weaviate)**:

  * **컬렉션 스키마 + Named Vectors**(ko/en·title/body·image), **BM25+벡터 하이브리드**, **크로스레퍼런스**로 문맥 연결.
  * **복제·일관성**(읽기 ONE/쓰기 QUORUM 권장) + **RBAC**(운영/편집/읽기 분리) + **백업(S3/GCS/Azure/FS)**. ([Weaviate Documentation][10])
* **분석/옵스**:

  * 검색 로그 → 재랭크/하이브리드 `alpha` 최적화, **PQ/SQ** 적용 여부 실험, 멀티테넌시(브랜드/리전) 분리.

### 3.2 핵심 스키마(예시)

* `DeviceManual`(title, body, device\_model, locale, url, created\_at; **NV: title\_vec, body\_vec**)
* `ErrorCode`(code, device\_model, symptoms, cause; **NV: code\_vec, cause\_vec**)
* `FixStep`(step\_title, step\_body, image\_url; **NV: step\_vec**)
* `FAQ`(q, a, locale; **NV: q\_vec, a\_vec**)
* 참조:

  * `DeviceManual` —hasChunks→ `Chunk`
  * `ErrorCode` —resolvedBy→ `FixStep`
  * `FAQ` —references→ `DeviceManual`/`FixStep`

**설계 이유**

* **Named Vectors**로 한 오브젝트에 다국어/필드별 임베딩을 *동시에* 저장 후, 쿼리 맥락에 맞는 `target_vector`를 선택해 검색 품질을 높입니다(예: “E18 에러” → `code_vec` 중심, “실외기 소음 해결” → `body_vec`/`step_vec`). ([Weaviate Documentation][11])
* **하이브리드 검색**(BM25+벡터)로 “정확 키워드 매칭(모델명/코드)”과 “의미 유사성(증상 설명)”을 함께 반영합니다. ([Weaviate Documentation][10])

### 3.3 색인·검색 파이프라인

1. **적재(ingest)**

* 문서 파서(매뉴얼 PDF/웹) → 섹션/청크 분할 → ko/en 번역 동기화(선택) → 임베딩(ko/en 전용 또는 멀티링구얼) → **Weaviate 배치 적재**(BYO 임베딩).
* 대량 데이터는 **배치 API**로 재시도/백오프, 샤드/레플리카 균형화를 고려.

2. **검색(query)**

* 기본: `hybrid(query=q, alpha=0.35~0.65)` + 필터(`device_model`, `locale`) + 상위 N **재랭킹**(Cross-encoder). ([Weaviate Documentation][16])
* 코드성 질의: `bm25()` 우선 + 벡터 보조.
* 증상 설명 질의: `near_text/near_vector` 비중 확대.
* **Named Vector 타겟 선택**: 예) 코드 질의 → `target_vector="code_vec"`, 증상 질의 → `"body_vec"`.

3. **후처리(생성/RAG)**

* 상위 k 문서 요약/답변, 단계 지시(“앱에서 메뉴 → 진단 시작” 등), 링크·이미지 **출처 표시**.

### 3.4 운영/보안/비용 전략

* **복제/일관성**: 읽기 많은 서비스면 `read=ONE`, 쓰기는 `QUORUM`으로 균형. 리전 DR은 비동기 백업+주기적 스냅샷. ([Weaviate Documentation][8])
* **RBAC**: 운영팀(읽기/집계), 콘텐츠팀(쓰기/스키마 제한), 배치봇(쓰기 한정) 역할 분리. ([Weaviate Documentation][17])
* **백업**: 일일 증분 + 주간 전체 백업(S3/GCS/Azure/FS). 복원 리허설을 분기마다 수행. ([Weaviate Documentation][18])
* **압축(PQ/SQ)**: 메모리 압박 시 적용. **HNSW+PQ**로 메모리 1/2\~1/4 수준 절감 사례(정확도/지연 트레이드오프). 필수 쿼리는 re-score로 보정. ([Weaviate][19])
* **튜닝**: HNSW의 `ef`/`efConstruction` 및 동적 `ef` 활용, 하이브리드 `alpha`·재랭커 컷오프 최적화. ([Weaviate Documentation][15])

### 3.5 구현 스니펫 (Python 3.10+ 가정)

```python
# 연결
import weaviate
from weaviate.classes.config import Property, DataType, Configure, Tokenization

client = weaviate.connect_to_local()

# Named Vectors 컬렉션
manuals = client.collections.create(
    name="DeviceManual",
    properties=[
        Property(name="title", data_type=DataType.TEXT, tokenization=Tokenization.WORD, index_searchable=True),
        Property(name="body",  data_type=DataType.TEXT, tokenization=Tokenization.WORD, index_searchable=True),
        Property(name="device_model", data_type=DataType.TEXT, index_filterable=True),
        Property(name="locale", data_type=DataType.TEXT, index_filterable=True),
        Property(name="url", data_type=DataType.TEXT, index_filterable=True),
    ],
    vectorizer_config=Configure.NamedVectors.multi({
        "title_vec": Configure.Vectorizer.none(),
        "body_vec":  Configure.Vectorizer.none(),
        "ko_vec":    Configure.Vectorizer.none(),
        "en_vec":    Configure.Vectorizer.none(),
    }),
    inverted_index_config=Configure.inverted_index(),  # BM25F/필터
)

# 배치 적재 (BYO 임베딩)
from sentence_transformers import SentenceTransformer
import numpy as np

model_ko = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
col = client.collections.get("DeviceManual")

data = [{
  "title": "실외기 소음 해결 가이드",
  "body": "전원 차단 후 팬 블레이드 이물질 제거...",
  "device_model": "AC-X100",
  "locale": "ko",
  "url": "https://…"
}]

with col.batch.dynamic() as batch:
    for d in data:
        vecs = {
            "title_vec": model_ko.encode(d["title"]).astype(np.float32),
            "body_vec":  model_ko.encode(d["body"]).astype(np.float32),
            "ko_vec":    model_ko.encode(d["body"]).astype(np.float32),
        }
        batch.add_object(properties=d, vectors=vecs)

# 하이브리드 검색 + Named Vector 타겟
res = col.query.hybrid(query="AC-X100 E18 에러 복구", target_vector="title_vec", alpha=0.5, limit=5)
for o in res.objects:
    print(o.properties["title"], o.metadata.score)

client.close()
```

> 하이브리드/Named Vectors/인덱스·압축에 관한 공식 문서는 아래에서 바로 확인하실 수 있습니다. ([Weaviate Documentation][10])

### 3.6 FAISS와의 **혼합 운용**(권장 시나리오)

* **모바일 오프라인/초저지연**: 소형 **FAISS** 인덱스를 앱에 동봉(최근 오류코드·FAQ 3\~5천 건).
* **서버 온라인**: **Weaviate**가 전체 지식(대규모)과 하이브리드 검색/필터/권한을 담당.
* 오케스트레이터가 **네트워크/쿼리 유형/개인정보 정책**을 기준으로 로컬/서버 경로를 선택. (예: 에러코드 단문 질의는 로컬, 서술형 증상 질의는 서버)

---

## 마무리 & 다음 단계 제안

* 이미 보유하신 **LangGraph + HQ-RAG** 파이프라인에 Weaviate를 연결하면, **하이브리드 검색·Named Vectors·운영 기능**을 바로 활용하면서도, 온디바이스에서는 **FAISS 소형 인덱스**로 초저지연 경험을 유지할 수 있습니다.
* 원하시면 \*\*초기 스키마 및 튜닝값(HNSW `ef`, 하이브리드 `alpha`, 재랭커 컷오프)\*\*을 귀하의 실제 데이터 샘플(ko/en 혼합, 기기군/오류코드 분포)에 맞춰 **벤치마크 스크립트**와 함께 드리겠습니다.

**참고 링크(클릭 가능)**

* Weaviate 하이브리드 검색, 개념/예제: ([Weaviate Documentation][10])
* Weaviate HNSW·인덱스 튜닝: ([Weaviate Documentation][3])
* Named Vectors: ([Weaviate Documentation][11])
* 복제/일관성, RBAC, 백업: ([Weaviate Documentation][8])
* 벡터 압축(PQ/SQ 등): ([Weaviate Documentation][20])
* Python 클라이언트 버전/요건: ([Weaviate Documentation][6], [GitHub][5])
* FAISS 개요/인덱스/GPU: ([Faiss][1], [GitHub][2])

필요하시면 위 설계를 \*\*귀하의 레포 구조(모노레포)\*\*와 **배포 스크립트(Docker Compose/K8s Helm)**, **샘플 데이터**까지 포함한 “부트스트랩 패키지”로 정리해 드리겠습니다.

[1]: https://faiss.ai/index.html?utm_source=chatgpt.com "Welcome to Faiss Documentation — Faiss documentation"
[2]: https://github.com/facebookresearch/faiss/wiki/Faiss-indexes?utm_source=chatgpt.com "Faiss indexes · facebookresearch/faiss Wiki"
[3]: https://docs.weaviate.io/academy/py/vector_index/hnsw?utm_source=chatgpt.com "HNSW index in depth"
[4]: https://weaviate-python-client.readthedocs.io/en/v4.16.1/weaviate.backup.html?utm_source=chatgpt.com "weaviate.backup"
[5]: https://github.com/weaviate/weaviate-python-client?utm_source=chatgpt.com "A python native client for easy interaction with a Weaviate ..."
[6]: https://docs.weaviate.io/weaviate/client-libraries/python?utm_source=chatgpt.com "Python | Weaviate Documentation"
[7]: https://docs.weaviate.io/academy/py/named_vectors?utm_source=chatgpt.com "220 Named vectors"
[8]: https://docs.weaviate.io/weaviate/concepts/replication-architecture/consistency?utm_source=chatgpt.com "Consistency"
[9]: https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU?utm_source=chatgpt.com "Faiss on the GPU · facebookresearch/faiss Wiki - GitHub"
[10]: https://docs.weaviate.io/weaviate/search/hybrid?utm_source=chatgpt.com "Hybrid search - Weaviate Documentation"
[11]: https://docs.weaviate.io/weaviate/config-refs/collections?utm_source=chatgpt.com "Collection definition"
[12]: https://bge-model.com/tutorial/3_Indexing/3.1.2.html?utm_source=chatgpt.com "Faiss GPU — BGE documentation"
[13]: https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index?utm_source=chatgpt.com "Guidelines to choose an index · facebookresearch/faiss Wiki"
[14]: https://faiss.ai/cpp_api/class/classfaiss_1_1gpu_1_1GpuIndexFlat.html?utm_source=chatgpt.com "Class faiss::gpu::GpuIndexFlat"
[15]: https://docs.weaviate.io/weaviate/config-refs/indexing/vector-index?utm_source=chatgpt.com "Vector index | Weaviate Documentation"
[16]: https://docs.weaviate.io/academy/py/zero_to_mvp/queries_2/hybrid?utm_source=chatgpt.com "Hybrid searches - Weaviate Documentation"
[17]: https://docs.weaviate.io/deploy/configuration/configuring-rbac?utm_source=chatgpt.com "Enable and configure RBAC"
[18]: https://docs.weaviate.io/deploy/configuration/backups?utm_source=chatgpt.com "Backups"
[19]: https://weaviate.io/blog/ann-algorithms-hnsw-pq?utm_source=chatgpt.com "HNSW+PQ - Exploring ANN algorithms Part 2.1"
[20]: https://docs.weaviate.io/weaviate/concepts/vector-quantization?utm_source=chatgpt.com "Compression (Vector Quantization)"

---
---
---

# 요약 (TL;DR)

요청하신 **모두 포함형 부트스트랩 패키지 v0.1**를 지금 바로 다운로드하실 수 있도록 준비했습니다.

* **구성**: 모노레포 구조, **Docker Compose**(Weaviate + FastAPI), **스키마 생성/적재 스크립트**, **Named Vectors 하이브리드 검색 API**, **샘플 데이터**, **Helm 차트(K8s)**, **튜너 스켈레톤**까지 포함.
* **오프라인 친화**: Sentence-Transformers가 없으면 자동으로 **해시기반 대체 임베딩**으로 동작 → 바로 데모 가능.
* **다운로드**: [weaviate\_bootstrap\_v0\_1.zip](sandbox:/mnt/data/weaviate_bootstrap_v0_1.zip)

---

아래부터 **사용 가이드**입니다. (필요하시면 영어 README도 추가해 드리겠습니다.)

## 1) 받기 & 빠른 실행

1. 위 링크에서 압축을 받으신 뒤 풀어주세요.
2. 루트에서 Docker Compose 실행:

   ```bash
   docker compose up -d
   ```
3. 스키마 생성:

   ```bash
   docker compose exec api python /app/scripts/init_schema.py
   ```
4. 샘플 데이터 적재:

   ```bash
   docker compose exec api python /app/scripts/ingest.py
   ```
5. 헬스체크 & 검색:

   ```bash
   curl "http://localhost:8000/health"
   curl "http://localhost:8000/search?q=AC-X100%20E18%20%EC%97%90%EB%9F%AC%20%EB%B3%B5%EA%B5%AC&k=5&alpha=0.5"
   ```

> 참고: 루트에 `Makefile`도 제공되어 `make up`, `make schema`, `make ingest`로 간편 실행이 가능합니다.

---

## 2) 포함 내용 요약

* **모노레포 구조**

  ```
  weaviate_bootstrap_v0_1/
  ├─ README_ko.md
  ├─ docker-compose.yml
  ├─ .env.example
  ├─ Makefile
  ├─ services/
  │  ├─ api/                 # FastAPI 서비스(검색 API)
  │  │  ├─ main.py           # /search /bm25 /near /health
  │  │  ├─ utils/embedder.py # SBERT(있으면) + 해시 임베더(없으면)
  │  │  ├─ requirements.txt
  │  │  └─ Dockerfile
  │  └─ scripts/             # 스키마/적재/튜닝
  │     ├─ init_schema.py    # Named Vectors 스키마 + 참조 생성
  │     ├─ ingest.py         # 샘플 데이터 BYO 임베딩 적재
  │     ├─ utils/embedder.py # 동일 임베더 재사용
  │     ├─ tune_alpha.py     # 하이브리드 alpha 튜너 스켈레톤
  │     └─ requirements.txt
  ├─ data/
  │  └─ sample/              # 샘플 JSON 3개
  └─ k8s/
     └─ helm/                # 배포 스켈레톤 (API, Weaviate)
        ├─ Chart.yaml
        ├─ values.yaml
        └─ templates/*.yaml
  ```

* **핵심 포인트**

  * **Named Vectors**: `title_vec`, `body_vec`, `ko_vec`, `en_vec`를 동일 오브젝트에 저장 → 질의 맥락별 `target_vector`로 품질 향상.
  * **하이브리드 검색**: `/search`에서 `alpha`(BM25↔벡터 가중), `device_model`, `locale` 필터 지원.
  * **오프라인 대체 임베딩**: Sentence-Transformers 미설치 시에도 **해시 임베더**가 즉시 동작(데모용).
  * **튜닝**: `scripts/tune_alpha.py`로 기본 스윕(0.1\~0.9).

---

## 3) 환경 변수

`.env.example` 참고 (Compose가 기본값으로 이미 주입합니다)

* `WEAVIATE_SCHEME`=`http`, `WEAVIATE_HOST`=`weaviate`, `WEAVIATE_HTTP_PORT`=`8080`, `WEAVIATE_GRPC_PORT`=`50051`
* `EMBED_MODEL_NAME`=`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
* `EMBED_LOCAL_PATH` (로컬 모델 경로, 선택)
* `EMBED_DIM`=384

> 실제 운영에서는 **익명 접근 비활성화**, **RBAC 활성화**, **백업 모듈 설정**, **복제/일관성** 등을 반드시 반영하십시오.

---

## 4) 스키마 설계(초기값)

* 컬렉션

  * `DeviceManual(title, body, device_model, locale, url, created_at; NV: title_vec, body_vec, ko_vec, en_vec)`
  * `Chunk(body, sequence, device_model, locale)`
  * `ErrorCode(code, device_model, symptoms, cause; NV: code_vec, cause_vec)`
  * `FixStep(step_title, step_body, image_url)`
  * `FAQ(q, a, locale; NV: q_vec, a_vec)`
* 참조

  * `DeviceManual` —hasChunks→ `Chunk`
  * `ErrorCode` —resolvedBy→ `FixStep`
  * `FAQ` —referencesManual→ `DeviceManual`

---

## 5) API 사용 예시

* 하이브리드 검색:

  ```
  GET /search?q=AC-X100%20E18%20%EB%B3%B5%EA%B5%AC&k=5&alpha=0.5&target_vector=title_vec&device_model=AC-X100&locale=ko
  ```
* BM25만:

  ```
  GET /bm25?q=실외기%20소음&k=5
  ```
* 벡터 직접 질의:

  ```
  POST /near
  {
    "vector": [0.12, -0.08, ... , 0.01],
    "k": 5,
    "collection": "DeviceManual"
  }
  ```
* 임베딩 확인(디버그):

  ```
  GET /embed?q=실외기%20소음%20해결
  ```

---

## 6) Helm 차트(스켈레톤)

`k8s/helm/values.yaml` 에서 이미지/포트 조정 후 배포:

```bash
helm install weaviate-bootstrap k8s/helm
```

> 실제 운영용으로는 **스토리지 클래스/리소스**, **인증/인가**, **백업 모듈**, **복제/일관성** 설정을 추가하세요.

---

## 7) FAISS와의 하이브리드 운용(선택)

* 모바일/온디바이스: 최근 FAQ/오류코드 수천 건 정도는 **FAISS 소형 인덱스**로 초저지연 대응
* 서버: 대규모 본지식은 **Weaviate**가 담당 (하이브리드 검색, 필터, 권한, 백업 등)
* 오케스트레이터(LangGraph/서버)가 **네트워크/쿼리 유형/개인정보 정책** 기준으로 경로 스위칭

---

## 8) 다음에 해드릴 수 있는 것

* 실제 데이터 스키마/토크나이제이션·`alpha` 튜닝, 재랭커(cross-encoder) 통합, RBAC/백업/복제 설정
* CI/CD, Prometheus/Grafana 대시보드, Artifacts(모델 캐시) 볼륨화
* 몬테카를로/그리드/Hyperband 기반 **hybrid 가중치 튜너** 고도화, 쿼리 로그 기반 최적화

---

필요하시면 이 패키지를 **귀하의 기존 레포 구조(모노레포)와 네이밍 컨벤션**에 맞춰 재정렬하고, **LangGraph 오케스트레이터와 직접 연결 예제**까지 포함해 드리겠습니다.
추가 요구사항을 알려주시면 바로 반영하겠습니다.

---
---
---

# 요약 (TL;DR)

요청하신 대로 **기존 모노레포 스타일 + 네이밍 컨벤션**에 맞춰 디렉터리 재정렬을 하고, **LangGraph 오케스트레이터**를 **Weaviate**와 직접 연결한 예제까지 포함한 새 패키지를 준비했습니다.

* **다운로드**: [monorepo\_langgraph\_weaviate\_v0\_2.zip](sandbox:/mnt/data/monorepo_langgraph_weaviate_v0_2.zip)
* **구성**: `apps/search-api`(Weaviate 하이브리드 검색 API), `apps/orchestrator-graph`(LangGraph + FastAPI), `packages/*`(LLM·Weaviate 클라이언트·모델 어댑터·스키마·RAG 유틸), `infra`(Compose/Helm), `data/sample`
* **LLM 연계**: 기본은 **Ollama** 연동(옵션), 미설정 시 **요약 에코** 폴백으로 바로 동작
* **Named Vectors + 하이브리드 검색 + LangGraph 파이프라인**이 한 번에 구동됩니다.

---

## 어떻게 쓰나요? (빠른 시작)

1. 압축 해제 후 루트에서 컨테이너 기동

```bash
docker compose up -d
```

2. 스키마 생성 & 샘플 적재

```bash
docker compose exec search-api python /workspace/apps/search-api/scripts/init_schema.py
docker compose exec search-api python /workspace/apps/search-api/scripts/ingest.py
```

3. 헬스체크 & 검색 API 테스트

```bash
curl "http://localhost:8001/health"
curl "http://localhost:8001/search?q=AC-X100%20E18%20%EC%97%90%EB%9F%AC%20%EB%B3%B5%EA%B5%AC&k=5&alpha=0.5"
```

4. 오케스트레이터(LangGraph) 테스트

```bash
curl -X POST "http://localhost:8010/chat" -H "Content-Type: application/json" \
  -d '{"user_input":"AC-X100 E18 오류 해결해줘","device_model":"AC-X100","locale":"ko"}'
```

> **Ollama 연동(선택):** `.env.example`에서 `OLLAMA_BASE_URL`과 `OLLAMA_MODEL`을 지정하면 생성 응답이 LLM 결과로 대체됩니다. (미지정 시 컨텍스트 기반 요약/에코 폴백)

---

## 포함된 것 (모노레포 정렬)

```
monorepo_langgraph_weaviate_v0_2/
├─ apps/
│  ├─ search-api/                # Weaviate 하이브리드 검색 API (FastAPI)
│  │  ├─ main.py                 # /health /search /bm25 /near
│  │  ├─ utils/embedder.py       # SBERT > 해시 임베더(폴백)
│  │  ├─ scripts/
│  │  │  ├─ init_schema.py       # Named Vectors + Cross-Refs 스키마
│  │  │  └─ ingest.py            # 샘플 데이터 BYO 임베딩 적재
│  │  ├─ requirements.txt
│  │  └─ Dockerfile
│  └─ orchestrator-graph/        # LangGraph + FastAPI
│     ├─ main.py                 # intent → retrieve → answer 그래프
│     ├─ requirements.txt
│     └─ Dockerfile
├─ packages/
│  ├─ clients-weaviate/          # weaviate-client v4 래퍼
│  │  └─ weaviate_client.py
│  ├─ llm-system-service/        # LLM 어댑터 (Ollama/폴백)
│  │  └─ adapter.py
│  ├─ model-adapter/             # 경량 Intent/Slot (규칙 기반)
│  │  └─ adapter.py
│  ├─ rag-pipeline/
│  │  └─ utils.py                # 정규화/출처 포맷팅
│  └─ schemas/
│     └─ models.py               # 공용 Pydantic 스키마
├─ data/
│  └─ sample/                    # 샘플 JSON 3개
├─ infra/
│  └─ k8s/helm/                  # 배포 스켈레톤 (필요시 확장)
├─ docker-compose.yml            # weaviate + search-api + orchestrator
├─ .env.example
├─ Makefile
└─ README_ko.md
```

---

## LangGraph 오케스트레이터 흐름

1. **intent 노드**: `packages/model-adapter`의 규칙 기반 NLU로 `retrieval / troubleshoot / explain` 등 의도 및 간단 슬롯(모델명/언어) 추출
2. **retrieve 노드**: `packages/clients-weaviate` 래퍼로 **하이브리드 검색**(BM25+벡터, 필터·Named Vector 타겟 지원) 수행
3. **answer 노드**: `packages/llm-system-service` 어댑터로 **Ollama** 또는 **폴백 요약**으로 응답 생성. 소스는 번호 매겨 첨부

> 생성 모델 미사용 모드에서도 **검색/출처 포함 응답**이 즉시 반환되므로 개발 환경에서 가볍게 확인 가능합니다.

---

## 운영 팁 (간단 체크리스트)

* **RBAC/인증**: 프로덕션에서는 Weaviate 익명 접근 비활성화, API Key/OIDC 및 역할 분리
* **백업/복제/일관성**: 모듈 활성화(backup-s3 등), 읽기/쓰기 consistency 튜닝(ONE/QUORUM/ALL)
* **Named Vectors**: `title_vec/body_vec/ko_vec/en_vec` 외에도 `code_vec/step_vec` 등 도메인별 추가 권장
* **튜닝**: 하이브리드 `alpha`/재랭커(선택)를 실험 로그 기반으로 고도화

---

## 다음 커스터마이징도 바로 가능해요

* 귀하 레포의 **실제 네임스페이스/모듈 네이밍** 반영
* **LangGraph 에이전트 분기 확장**(FAQ 전용 경로, 멀티타겟 벡터 조인, 재랭커 추가)
* **CI/CD**(빌드/배포/마이그레이션), **모니터링**(Prometheus/Grafana), **RBAC/백업** 템플릿화

필요하신 변경 사항(예: 조직 표준 디렉터리/패키지 접두사, 컨테이너 네이밍 규칙, K8s 네임스페이스/리소스쿼터)을 알려주시면 여기 패키지를 그 규칙에 맞게 더 깔끔하게 다듬어 드리겠습니다.


