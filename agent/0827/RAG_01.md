> RAG 와 LLM에 모든 내용을 다 넣고 질문하는것과 어떤게 더 정확할까? 두 가지를 다각도로 비교해봐

# 요약(한 줄 결론)

* \*\*지식이 크고 자주 바뀌는 실제 서비스(IT/IoT 헬프·온보딩·트러블슈팅)\*\*라면 **RAG**가 보통 **더 정확하고 신뢰 가능한 답**(근거/출처 제공, 최신성 유지)을 냅니다. 반대로 **질문에 필요한 정보가 소수의 문서에 모여 있고 변화가 적으며** 길게 붙여 넣어도 모델이 핵심을 잘 찾을 수 있다면 \*\*장문 컨텍스트(“모든 내용을 프롬프트에 넣기”)\*\*가 간단하고 빠를 수 있습니다. 다만 장문 컨텍스트는 **중간에 있는 근거를 놓치는 위치 바이어스**가 여전히 보고되고 있고, 연구별로 **RAG가 우세** 혹은 **장문 컨텍스트가 우세**라는 **상반 결과**가 공존하니 **업무 데이터로 A/B 평가**가 필수입니다. ([arXiv][1], [ACL Anthology][2], [OpenReview][3])

---

## 핵심 요약

* **정확도(정답률/팩트성)**

  * 여러 연구에서 **RAG가 장문 컨텍스트보다 정확**하다는 보고가 있고(특히 대화형/일반 질의), 반대로 **백과사전형 QA**에선 **장문 컨텍스트가 우세**하다는 재평가도 있습니다 → **도메인과 태스크에 따라 승부가 갈립니다.** ([OpenReview][3], [arXiv][4])
  * 장문 컨텍스트는 “**중간을 잃는 현상(Lost in the Middle)**”으로 중요한 근거를 놓칠 수 있습니다(최근 보완 연구도 있으나 완전 해결은 아님). ([arXiv][1], [ACL Anthology][2])
  * **자기반성형 RAG(Self-RAG)** 같은 하이브리드는 팩트성과 인용 정확도를 꾸준히 끌어올렸습니다. ([arXiv][5], [OpenReview][6], [selfrag.github.io][7])

* **최신성·지식 업데이트**

  * RAG는 **지식 베이스만 교체/증분 인덱싱**하면 최신 상태 유지가 쉽고, **출처 링크**를 제시하기 좋습니다. (RAG 원전) ([arXiv][8])
  * 장문 컨텍스트는 매 질의 때 **관련 없는 대량 토큰을 함께 전송**하기 쉬워 **비용과 지연**이 커지고, **출처 관리**가 체계적이지 않으면 검증이 어렵습니다.

* **성능·비용(토큰/지연)**

  * 트랜스포머 주의(attention)는 \*\*시퀀스 길이 $n$\*\*에 대해 \*\*계산량이 최소 $O(n^2)$\*\*로 증가합니다(FlashAttention은 메모리/IO를 줄여도 **복잡도 자체는 유지**). → **긴 프롬프트는 비용·지연이 급증**합니다. ([arXiv][9], [NeurIPS Proceedings][10])
  * RAG는 \*\*작은 질의 + 소수 문서(k개)\*\*만 모델에 투입 → **토큰·지연·비용을 상한**할 수 있습니다.

* **다국어(한국어+영어 혼재)**

  * **다국어 임베딩**(예: **BGE-M3**, **jina-embeddings-v3**)을 쓰면 한국어/영어가 섞인 KB에서도 **검색-정확도**가 높습니다. ([arXiv][11], [ACL Anthology][12])

* **추천**

  * 귀하의 **IT/IoT 온보딩·트러블슈팅**에는 \*\*RAG(+요청별 동적 검색)\*\*이 기본값, **작은 고정 문답/오프라인**은 장문 컨텍스트 보조.
  * **하이브리드**(“필요 시에만 검색”) 게이트를 두면 \*\*정확도↑ 비용↓\*\*를 동시에 달성할 수 있습니다. ([arXiv][5])

---

# 상세 비교

### 1) 무엇이 “더 정확한가?” — **케이스별 승부**

| 상황                                   | 더 유리한 쪽               | 이유                                                    |
| ------------------------------------ | --------------------- | ----------------------------------------------------- |
| **대규모·이질적·자주 갱신** KB (매뉴얼, 포럼, 이슈로그) | **RAG**               | 최신성·출처·스케일. 장문 컨텍스트는 불필요 토큰 증가·근거 누락 위험. ([arXiv][8]) |
| **백과사전형/단일 코퍼스 QA**                  | **장문 컨텍스트가 우세할 수 있음** | 최근 재평가에서 위키 기반 QA에선 장문 컨텍스트가 유리하다는 결과. ([arXiv][4])   |
| **대화형/일반 질의**                        | **RAG 우세** 경향 보고      | 다양한 태스크에서 RAG가 답변 정확도에서 우위 보고. ([OpenReview][3])      |
| **복잡 장문 문서 내 다단계 근거 결합**             | **RAG(계층형/요약형 리트리벌)** | RAPTOR 같은 트리형 요약·재귀 검색이 장문 합성에 강점. ([arXiv][13])      |
| **인터넷 차단/오프라인, 소형 KB**               | **장문 컨텍스트**           | 간단·구축 난이도↓. KB가 작고 정적이면 충분.                           |

> 참고: “**Lost in the Middle**”는 장문 입력에서 **초중반 근거를 놓치는 위치 바이어스**를 보여줍니다. 보정 연구가 있으나 **완전 해결은 아님**—따라서 장문 컨텍스트 전략이라도 **문서 순서·요약·헤더** 설계를 잘 해야 합니다. ([arXiv][1], [ACL Anthology][2])

---

### 2) 최신성·근거 제시·운영 관점

* **RAG**: 문서 교체·증분 인덱싱으로 **지식 업데이트가 쉬움**, **URL/문서ID를 함께 제시**해 **검증가능성**이 높음(원전 취지). ([arXiv][8])
* **장문 컨텍스트**: 새 지식을 쓰려면 **매번 프롬프트에 넣거나 재학습**. 출처가 섞이면 **근거 추적**이 어렵습니다.

---

### 3) 성능·비용·레이턴시

* **장문 컨텍스트**: 입력 길이에 따라 attention 연산이 **급격히 증가**(최소 $O(n^2)$); FlashAttention은 **메모리/IO는 줄여도** 계산 복잡도는 유지 → **롱 컨텍스트는 곧 비용**입니다. ([arXiv][9], [NeurIPS Proceedings][10])
* **RAG**: 검색(인덱스 탐색) + 짧은 컨텍스트 → **토큰 예산 고정화**가 쉬워 **지연·비용 예측성**이 좋습니다.

---

### 4) 다국어(KO+EN 혼재) 정확도

* **멀티링구얼 임베딩**을 쓰면 **교차언어 매칭**(한국어 질문 ↔ 영어 문서)이 향상됩니다.

  * 권장: **BGE-M3**(다국어·다기능·다입도, 8K 토큰), **jina-embeddings-v3**(다국어 SOTA 보고). ([arXiv][11], [ACL Anthology][12])

---

### 5) 실무 설계 팁(정확도 최적화)

1. **“검색 필요 여부” 게이팅**:

* *규칙*: “최근/고유명사/버전/링크 요구/수치” 포함 시 **검색**.
* *모델*: **Self-RAG**처럼 **자기반성 토큰**으로 “검색할지/몇 개 가져올지/재검색할지”를 스스로 결정. ([arXiv][5])

2. **리트리벌 품질**:

* **하이브리드**(BM25+Dense) + **RRF 재랭크**로 recall/precision 균형.
* \*\*계층형 리트리벌(RAPTOR)\*\*로 장문·매뉴얼 통합 질의에 강함. ([arXiv][13])

3. **컨텍스트 구성**:

* \*\*문단 요약 + 헤더(제목/버전/언어 태그)\*\*를 함께 주입 → “중간 손실” 완화. (Lost-in-the-Middle 보정에 도움) ([arXiv][1])

4. **다국어 인덱스**:

* 문서에 `lang: ko|en` 메타·번역 제목 키워드 추가, 임베딩은 **bge-m3/jina-v3**. ([arXiv][11])

5. **평가·관측**:

* **RAGAS**로 *Answer Correctness, Faithfulness, Context Recall/Precision* 측정,
* **Needle-in-a-Haystack**로 장문 컨텍스트 사용성 테스트. ([ragas.io][14], [Medium][15], [GitHub][16])
* RAG 전용 **서베이/튜토리얼·서베이** 참고. ([arXiv][17])

---

### 6) 선택 가이드(의사결정 트리)

1. **KB 규모가 크고 자주 바뀌나?** → **예**: RAG 기본.
2. **답변에 출처(링크) 필요? 규제/감사 요구?** → **예**: RAG. ([arXiv][8])
3. **오프라인/소형 KB/고정 FAQ?** → 장문 컨텍스트(간단·배포 용이).
4. **문서가 길고 단계적 결합이 필요한가?** → RAG(+RAPTOR). ([arXiv][13])
5. **백과사전형 질의·단일 코퍼스?** → 장문 컨텍스트도 강함(실험 보고). ([arXiv][4])

---

## 귀하(한국/다국어, IT/IoT 온보딩·트러블슈팅)에게의 실전 권장안

1. **기본**: **RAG**(BM25+Dense 하이브리드, RRF 재랭크).

* **임베딩**: **BGE-M3** 또는 **jina-embeddings-v3** (한국어/영어 혼합 최적화). ([arXiv][11])
* **계층형 검색**: 매뉴얼·포럼 같은 **장문**엔 **RAPTOR**를 병행. ([arXiv][13])
* **게이팅**: Self-RAG식 “검색 필요” 판단으로 **불필요 검색 억제**. ([arXiv][5])

2. **보조**: **롱-컨텍스트 버퍼**

* 자주 묻는 소수 문서(“빠른 시작/모델별 스펙표”)는 **질문 세션 버퍼**에 **압축 요약**으로 유지 → **지연 최소화**(필요 시만 RAG).

3. **평가/관측 루프**

* **RAGAS**로 주/월간 정확도·근거충실도 트래킹,
* **Needle 테스트**로 장문 컨텍스트 품질 점검(“중간 근거” 탐색률). ([ragas.io][14], [GitHub][16])

---

## 보너스: “왜 장문 컨텍스트가 비싼가요?”

* 트랜스포머는 입력 길이 $n$에 대해 \*\*주의 계산량이 최소 $O(n^2)$\*\*로 늘어납니다. **FlashAttention**은 **메모리/IO 최적화**로 속도·메모리를 줄이지만, **이론적 연산 차수는 그대로**이므로 프롬프트가 길어질수록 비용·지연은 계속 커집니다. ([arXiv][9], [NeurIPS Proceedings][10])

---

## 연구 근거(클릭해 확인)

* **장문 컨텍스트 한계/보정**: *Lost in the Middle* & TACL 버전, *Found in the Middle*. ([arXiv][1], [ACL Anthology][2])
* **RAG 원전**: *Retrieval-Augmented Generation for Knowledge-Intensive NLP*. ([arXiv][8])
* **RAG vs 장문 컨텍스트 비교 연구(상반 결과 공존)**: OpenReview 2024(“RAG 우세”), arXiv 2024/2025(“장문 우세 사례”). ([OpenReview][3], [arXiv][4])
* **자기반성형 RAG(Self-RAG)**: 논문/프로젝트 페이지. ([arXiv][5], [selfrag.github.io][7])
* **롱컨텍스트 벤치**: Needle-in-a-Haystack(텍스트/멀티모달). ([GitHub][16], [arXiv][18], [ACL Anthology][19])
* **RAG 평가 서베이 & 도구**: RAG 평가 서베이, RAGAS. ([arXiv][17], [ragas.io][14])
* **다국어 임베딩**: BGE-M3, jina-embeddings-v3. ([arXiv][11], [ACL Anthology][12])
* **긴 프롬프트의 계산 복잡도**: Transformer/FlashAttention. ([arXiv][9])

---

## 마무리 제안(실행 순서)

1. 현재 KB(한국어+영어)로 **작은 골든셋**을 만들고,
2. **(A) RAG 파이프라인** vs **(B) 장문 컨텍스트 버퍼** vs **(C) 하이브리드(Self-RAG 게이팅)** 를 **동일 질의**로 **RAGAS** 평가,
3. **정확도/근거충실도/지연/토큰비용**을 표준화해 \*\*정책(게이트 임계값·top-k·컨텍스트 길이)\*\*를 확정하십시오. ([ragas.io][14])

필요하시면, 위 평가를 바로 돌릴 수 있는 **샘플 스크립트/파이프라인**(BGE-M3 + BM25 + RRF + RAPTOR 옵션 + RAGAS 평가)을 만들어 드리겠습니다.

[1]: https://arxiv.org/abs/2307.03172?utm_source=chatgpt.com "Lost in the Middle: How Language Models Use Long Contexts - arXiv"
[2]: https://aclanthology.org/2024.tacl-1.9/?utm_source=chatgpt.com "Lost in the Middle: How Language Models Use Long Contexts"
[3]: https://openreview.net/forum?id=k3ibnAmzKG&utm_source=chatgpt.com "RAG vs. Long Context: Examining Frontier Large ..."
[4]: https://arxiv.org/abs/2501.01880?utm_source=chatgpt.com "Long Context vs. RAG for LLMs: An Evaluation and Revisits"
[5]: https://arxiv.org/abs/2310.11511?utm_source=chatgpt.com "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"
[6]: https://openreview.net/pdf?id=hSyW5go0v8&utm_source=chatgpt.com "SELF-RAG: LEARNING TO RETRIEVE, GENERATE, AND ..."
[7]: https://selfrag.github.io/?utm_source=chatgpt.com "Self-RAG: Learning to Retrieve, Generate and Critique ..."
[8]: https://arxiv.org/abs/2005.11401?utm_source=chatgpt.com "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
[9]: https://arxiv.org/abs/1706.03762?utm_source=chatgpt.com "[1706.03762] Attention Is All You Need"
[10]: https://proceedings.neurips.cc/paper_files/paper/2022/file/67d57c32e20fd0a7a302cb81d36e40d5-Supplemental-Conference.pdf?utm_source=chatgpt.com "Fast and Memory-Efficient Exact Attention with IO-Awareness"
[11]: https://arxiv.org/abs/2402.03216?utm_source=chatgpt.com "BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation"
[12]: https://aclanthology.org/2024.findings-acl.137/?utm_source=chatgpt.com "M3-Embedding: Multi-Linguality, Multi-Functionality, ..."
[13]: https://arxiv.org/abs/2401.18059?utm_source=chatgpt.com "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval"
[14]: https://www.ragas.io/?utm_source=chatgpt.com "Ragas"
[15]: https://medium.com/data-science/evaluating-rag-applications-with-ragas-81d67b0ee31a?utm_source=chatgpt.com "Evaluating RAG Applications with RAGAs"
[16]: https://github.com/gkamradt/LLMTest_NeedleInAHaystack?utm_source=chatgpt.com "gkamradt/LLMTest_NeedleInAHaystack"
[17]: https://arxiv.org/html/2405.07437v2?utm_source=chatgpt.com "Evaluation of Retrieval-Augmented Generation: A Survey"
[18]: https://arxiv.org/abs/2406.11230?utm_source=chatgpt.com "Multimodal Needle in a Haystack: Benchmarking Long ..."
[19]: https://aclanthology.org/2025.naacl-long.166.pdf?utm_source=chatgpt.com "Multimodal Needle in a Haystack: Benchmarking Long- ..."

---

> 계층형 retrieval 에 대해 자세히 알려줘. 예시를 보여주고 코드도 제시해줘

# 요약

* \*\*계층형 리트리벌(Hierarchical Retrieval)\*\*은 문서 → 섹션(또는 클러스터/요약) → 문단/문장 순으로 “굵게 고르고, 점점 촘촘히 고르는” **다단계 검색 전략**입니다. 긴 매뉴얼·다국어 KB 같은 **장문·대규모 지식**에서 \*\*정확도(정답률/근거충실도)\*\*와 \*\*효율(토큰/지연/비용)\*\*을 동시에 개선합니다. 특히 “**중간 근거를 놓치는**” 롱컨텍스트 한계를 줄이는 데 유리합니다.
* 대표 방식은 **RAPTOR(요약 트리 기반)**, **섹션→문단 2단계 검색**, **증분/보충 질의 + RRF 융합** 등이 있으며, 다국어 환경에서는 **BGE-M3**나 **jina-embeddings-v3** 같은 멀티링구얼 임베딩과 결합하는 것이 효과적입니다.
* 아래에 **실전 설계 패턴**과 **파이썬 예제 코드**(FAISS + BM25 + RRF + 선택적 크로스-리랭커)를 제공합니다. 그대로 실행하면 \*\*섹션-우선(상위 수준)\*\*으로 좁히고, 이어서 **문단-정밀(하위 수준)** 검색 및 **융합/리랭크**까지 한 번에 돌아갑니다. RAG 파이프라인에 바로 넣기 좋습니다.

---

## 1) 개념: 왜 “계층형”인가?

1. **장문·대규모 KB에서의 난점**

   * 한 번에 전부 임베딩/검색하면 **노이즈↑·비용↑**.
   * LLM은 긴 입력에서 **중간 위치 근거**를 자주 놓칩니다(“Lost in the Middle”). 계층형으로 **상위 요약/섹션에서 후보를 좁힌 뒤**, 하위에서 정밀히 재검색·재랭크하면 이 문제를 줄일 수 있습니다.

2. **핵심 아이디어**

   * **상위 레벨**(문서·섹션·클러스터·요약)에서 **조-대역(coarse)** 후보를 뽑고,
   * **하위 레벨**(문단·문장·슬라이드 조각)에서 **세밀(fine)** 재검색/리랭크.
   * 필요 시 \*\*질의 보정/보강(HyDE 등)\*\*과 \*\*순위 융합(RRF)\*\*으로 강건화.

3. **대표 연구/패턴**

   * **RAPTOR**: 문서들을 **요약 트리**로 구성(문단→섹션 요약→상위 요약…)하여 **다층에서 검색**하고 근거를 모읍니다. 장문 합성·다단계 근거에 강함.
   * **LongRefiner/최근 HR 패턴들**: 상위 레벨 선별 → 하위 레벨 정밀 추출로 **지연/비용을 10배 절감**하면서 정확도 유지/상향을 보고.
   * **Evidence Curation형 HR**: 표준화 문서(금융 등)에서 중복/군더더기를 제거하고 **핵심 증거를 선별**해 질의에 맞춘 보충 질의까지 수행.

---

## 2) 설계 패턴(실무 체크리스트)

* **인덱스 구성**

  * (A) **섹션/요약 레벨 임베딩 인덱스**(FAISS 등)
  * (B) **문단/문장 레벨 세부 인덱스**(BM25 + Dense 하이브리드 권장)
  * (옵션) **요약 트리(RAPTOR)**: LLM/룰 기반으로 상향식 요약 노드 생성 후 각 노드 임베딩.

* **질의 처리 흐름**

  1. **Coarse Retrieval(상위)**: 섹션/요약 인덱스에서 Top-S 추출
  2. **Candidate Expansion**: 선택 섹션 하위 문단/문장 풀로 확장
  3. **Fine Retrieval(하위)**: BM25(정확한 키워드 일치) + Dense(의미 유사) 검색
  4. **순위 융합**: **RRF**로 BM25와 Dense 결과를 간단·강건하게 결합
  5. **(선택) 크로스-리랭크**: CrossEncoder로 상위 N개 정밀 재랭크
  6. **컨텍스트 구성**: 서로 **보완적인 근거**가 나오도록 중복/중언 제거
  7. **(옵션) HyDE/다중 질의**: 부족하면 가설 문서를 생성해 재검색, 또는 질의 변형을 다중 생성 후 **RRF 융합**.

* **다국어 최적화**

  * 임베딩은 **BGE-M3** 또는 **jina-embeddings-v3** 추천(한국어/영어 혼재에 강함). 섹션 메타에 `lang: ko|en` 태깅·이중 제목(ko/en) 키워드 부여.

---

## 3) 미니 예시

**상황**: 스마트홈 제품(허브/에어컨/세탁기) 온보딩·오류코드 매뉴얼(한국어/영어 혼재).
**질의**: “**E23 오류로 Wi-Fi 페어링이 중단될 때 재시도 절차**”
**계층형 흐름**:

1. 섹션 인덱스에서 “*네트워크 설정/에러 코드 표*” 섹션이 Top-S로 선택.
2. 그 섹션의 문단 후보에서 BM25(“E23”, “페어링”, “재시도”)와 Dense(의미유사)로 상위 문단 추출.
3. RRF로 융합 → “E23=AP 격리/2.4GHz 전용, 재부팅→WPS 오프→재시도 60초” 문단이 1위.
4. (선택) Cross-Encoder로 최종 5개 문단 정밀 재랭크.
   → 이렇게 하면 긴 매뉴얼 전체를 직접 LLM에 넣을 때보다 **근거 집중도↑, 토큰↓**.

---

## 4) 파이썬 예제 코드 (FAISS + BM25 + RRF + 선택적 Cross-Encoder)

> 요구 라이브러리
> `pip install sentence-transformers faiss-cpu rank_bm25 numpy scikit-learn`

```python
# -*- coding: utf-8 -*-
import re, math, json, numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from rank_bm25 import BM25Okapi

# ---------- 0) 샘플 코퍼스(섹션/문단 구조) ----------
DOCS = [
    {
        "doc_id": "hub_guide_ko",
        "title": "스마트허브 설치 및 네트워크 가이드",
        "sections": [
            {
                "sec_id": "net_basics",
                "title": "네트워크 기본 (2.4GHz 권장, AP 격리 해제)",
                "paragraphs": [
                    "스마트허브는 2.4GHz 대역에서 가장 안정적으로 동작합니다.",
                    "공유기의 AP 격리가 활성화되면 기기 간 통신이 차단되어 페어링이 실패할 수 있습니다."
                ]
            },
            {
                "sec_id": "error_codes",
                "title": "에러 코드 표 (E2x/E3x)",
                "paragraphs": [
                    "E21: 라우터 인증 실패. 비밀번호를 확인 후 재시도하세요.",
                    "E23: AP 격리 또는 임시 네트워크 충돌로 페어링 중단. 공유기 재부팅 후 WPS 기능을 끄고 60초 뒤 다시 시도하세요.",
                    "E31: 인터넷 연결 없음. ISP 모뎀 상태를 확인하세요."
                ]
            }
        ]
    },
    {
        "doc_id": "ac_manual_en",
        "title": "Air Conditioner Wi-Fi Setup & Troubleshooting",
        "sections": [
            {
                "sec_id": "pairing_steps",
                "title": "Pairing Steps (LED Blink Codes)",
                "paragraphs": [
                    "Use the app to start pairing. Ensure the device is in AP mode.",
                    "If pairing fails, reset the device and retry within 60 seconds."
                ]
            },
            {
                "sec_id": "error_table",
                "title": "Error Codes (E2x/E3x)",
                "paragraphs": [
                    "E23 indicates AP isolation or temporary interference. Reboot router, disable WPS, then retry.",
                    "E24 refers to DNS misconfiguration."
                ]
            }
        ]
    }
]

# ---------- 1) 유틸 ----------
def flatten_sections(docs):
    """섹션/문단을 평탄화해서 섹션/문단 레벨 텍스트 리스트 생성"""
    section_texts, section_meta = [], []
    para_texts, para_meta = [], []
    for d in docs:
        for s in d["sections"]:
            # 섹션 대표 텍스트: 섹션 제목 + 앞 문단 1개를 간단 요약처럼 사용
            sec_text = f"{d['title']} > {s['title']}\n" + (s["paragraphs"][0] if s["paragraphs"] else "")
            section_texts.append(sec_text)
            section_meta.append({"doc_id": d["doc_id"], "sec_id": s["sec_id"], "title": s["title"]})

            for i, p in enumerate(s["paragraphs"]):
                para_texts.append(p)
                para_meta.append({
                    "doc_id": d["doc_id"], "sec_id": s["sec_id"],
                    "para_idx": i, "path": f"{d['title']} > {s['title']} > P{i+1}"
                })
    return section_texts, section_meta, para_texts, para_meta

# RRF(Reciprocal Rank Fusion)
def rrf_fuse(rank_lists: List[List[int]], k: int = 60) -> Dict[int, float]:
    """
    rank_lists: 각 랭킹에서의 '문서 인덱스' 리스트(0이 1등)
    반환: doc_idx -> fused_score
    """
    scores = {}
    for ranks in rank_lists:
        for r, doc_idx in enumerate(ranks):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + r + 1)
    return scores

# ---------- 2) 모델/인덱스 준비 ----------
# 멀티링구얼 권장: 'BAAI/bge-m3' (대형) 또는 'paraphrase-multilingual-MiniLM-L12-v2'(경량)
EMB_MODEL_NAME = "BAAI/bge-m3"
emb_model = SentenceTransformer(EMB_MODEL_NAME)

section_texts, section_meta, para_texts, para_meta = flatten_sections(DOCS)

# 임베딩 (코사인 유사도 사용 위해 정규화)
def embed(texts: List[str]) -> np.ndarray:
    X = emb_model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return np.array(X, dtype="float32")

E_sec = embed(section_texts)
E_para = embed(para_texts)

# FAISS 인덱스 (cosine = inner product with normalized vectors)
d = E_sec.shape[1]
sec_index = faiss.IndexFlatIP(d); sec_index.add(E_sec)
d2 = E_para.shape[1]
para_index = faiss.IndexFlatIP(d2); para_index.add(E_para)

# BM25 (문단 레벨)
tokenized_corpus = [re.findall(r"\w+", t.lower()) for t in para_texts]
bm25 = BM25Okapi(tokenized_corpus)

# (선택) 크로스-리랭커 (문맥상 정확도 향상용)
# 필요 시 주석 해제
# cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ---------- 3) 계층형 검색 ----------
@dataclass
class Retrieved:
    text: str
    score: float
    meta: dict

def hierarchical_retrieve(query: str, top_sections=3, top_paras=5, use_cross=False) -> List[Retrieved]:
    # 1) 상위(섹션/요약) 검색
    q_vec = embed([query])
    D_sec, I_sec = sec_index.search(q_vec, top_sections)  # inner product == cosine (normalized)
    candidate_sec_ids = I_sec[0].tolist()

    # 2) 후보 섹션의 문단들을 모아 '세밀 검색'
    candidate_para_idxs = []
    for sec_id in candidate_sec_ids:
        sec = section_meta[sec_id]
        # 동일 섹션의 문단들 인덱스 수집
        for i, m in enumerate(para_meta):
            if m["sec_id"] == sec["sec_id"] and m["doc_id"] == sec["doc_id"]:
                candidate_para_idxs.append(i)

    candidate_para_idxs = list(dict.fromkeys(candidate_para_idxs))  # 중복 제거 유지

    # 2-a) Dense(FAISS) 점수
    E_q = q_vec
    D_p_all, I_p_all = para_index.search(E_q, len(para_texts))
    # 후보만 필터
    dense_rank = [idx for idx in I_p_all[0].tolist() if idx in candidate_para_idxs][:max(top_paras*5, 20)]

    # 2-b) BM25 점수
    tokenized_query = re.findall(r"\w+", query.lower())
    bm25_scores = bm25.get_scores(tokenized_query)
    # 후보만 필터한 BM25 랭크
    bm25_pairs = [(i, bm25_scores[i]) for i in candidate_para_idxs]
    bm25_rank = [i for i, _ in sorted(bm25_pairs, key=lambda x: x[1], reverse=True)][:max(top_paras*5, 20)]

    # 3) RRF 융합
    fused = rrf_fuse([dense_rank, bm25_rank], k=60)
    fused_rank = [i for i, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)][:max(top_paras, 8)]

    results = [Retrieved(para_texts[i], fused[i], para_meta[i]) for i in fused_rank]

    # 4) (선택) 크로스-리랭크로 최종 정밀 정렬
    if use_cross:
        pairs = [[query, r.text] for r in results]
        ce_scores = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2").predict(pairs)
        for r, s in zip(results, ce_scores):
            r.score = float(s)
        results = sorted(results, key=lambda x: x.score, reverse=True)

    return results

# ---------- 4) 동작 예시 ----------
if __name__ == "__main__":
    query = "E23 오류로 Wi-Fi 페어링이 중단될 때 어떻게 재시도하나요?"
    hits = hierarchical_retrieve(query, top_sections=2, top_paras=5, use_cross=False)
    print(f"[Query] {query}\n")
    for rank, h in enumerate(hits, 1):
        print(f"#{rank} score={h.score:.4f} | {h.meta['path']}")
        print("  ", h.text)
```

### 코드 설명 요점

* **상위 단계**: 섹션(제목 + 첫 문단 요약 대용) 임베딩으로 **FAISS 코사인 검색**.
* **하위 단계**: 상위에서 고른 섹션 안의 문단만 대상으로 **Dense + BM25** 이중 검색.
* **RRF 융합**: 간단하고 강건한 랭킹 융합으로 **두 세계의 장점 결합**.
* **옵션**: Cross-Encoder로 상위 N개 **정밀 재랭크**(질의-문단 의미적 적합성 점수).
* **다국어**: `BAAI/bge-m3`는 한국어/영어 혼재에 강함(원 논문/Findings 보고).

---

## 5) 고급화 포인트

1. **RAPTOR식 요약 트리**

* 문단→섹션 요약→문서 요약으로 **트리 노드**를 만들고, **노드 임베딩**을 인덱싱 → **트리 위에서 내려오며** coarse→fine 검색.

2. **보충 질의/HyDE/다중질의 + 융합**

* 질의가 모호하면 **가설 문서(HyDE)** 생성 후 재검색, 또는 **질의 변형** 다중 생성 → **RRF**로 결과 융합.

3. **증거 큐레이션(Evidence Curation)**

* 표준 문서의 중복/군더더기 제거, **보완 정보 질의**까지 수행하여 **근거 충실도** 향상.

4. **평가/튜닝**

* \*\*top-S(섹션)\*\*과 \*\*top-k(문단)\*\*을 각각 튜닝.
* BM25의 `k1`, `b`, 문단 크기(예: 120~~200단어, stride 30~~50), 요약 길이, 언어 태그 등을 함께 최적화.
* 롱컨텍스트 문제는 여전히 존재하므로, **계층형으로 중간 근거를 앞쪽에 배치**해 투입하세요.

---

## 6) 참고 자료(클릭해 확인)

* **RAPTOR(요약 트리 기반 계층형 검색)**: arXiv/요약 페이지.
* **롱컨텍스트 한계(Lost in the Middle)**: 스탠퍼드/애리조나 등 연구.
* **RRF(순위 융합)**: SIGIR/ACM 및 원 논문 PDF.
* **BM25(확률적 관련도 프레임워크)**: 개관·서베이.
* **LongRefiner/최근 HR 동향**(롱텍스트 RAG 효율화).
* **다국어 임베딩**: BGE-M3, jina-embeddings-v3.

---

## 마무리

* 위 코드는 **현실적인 HR 파이프라인의 최소 골격**입니다.
* 운영에서는 여기에 **증분 인덱싱(섹션/문단에 버전 태그)**, **언어/제품별 오버레이**, **Self-RAG식 재검색 게이팅**, **크로스-리랭커 배치 최적화**를 얹으시면 **정확도·비용·지연**의 균형이 잡힙니다. 필요하시면 귀하의 IoT KB 스키마에 맞춰 **RAPTOR-스타일 요약 트리 생성 스크립트**도 바로 구성해 드리겠습니다.

---

> 그래 RAPTOR 스타일 요약 트리 생성 스크립트도 구성해줘.
> 그리고 self-RAG 재검색 게이팅과 크로스-리랭커 배치최적화도 추가해줘.

# 요약

* **RAPTOR 스타일 요약 트리 생성기 + Self-RAG 재검색 게이팅 + 크로스-리랭커 배치 최적화**를 한 번에 돌릴 수 있는 **실전용 미니 레포**를 준비해 드렸습니다.
* 포함 내용: 트리 빌더(문서→청크→요약 트리), 계층형 검색(Coarse→Fine), BM25+Dense **RRF 융합**, **Self-RAG 게이팅**(LLM/휴리스틱), **Cross-Encoder 리랭크**(배치 크기 자동 조절) 데모.
* 바로 내려받아 실행해 보시고, Ollama/vLLM 환경에 맞춰 요약기·게이팅 LLM만 연결하시면 됩니다.

[📦 raptor\_hr\_demo.zip 다운로드](sandbox:/mnt/data/raptor_hr_demo.zip)

---

아래는 **구성 설명과 사용 방법**, **핵심 코드 개요**, **튜닝 포인트**입니다.

---

## 1) 레포 구성

```
raptor_hr_demo/
  README_ko.md                 # 사용 방법
  requirements.txt
  sample_docs/
    hub_guide_ko.md
    ac_manual_en.md
    router_troubleshooting_ko.md
  raptor_tree.py               # RAPTOR 스타일 요약 트리 + 레벨별 FAISS 인덱스
  retrieval_pipeline.py        # 계층형 검색 + BM25+Dense RRF + (옵션) Cross-Encoder
  selfrag_gating.py            # Self-RAG 게이팅(LLM/휴리스틱) + 질의 재작성
  demo.py                      # 인덱스 생성 후 질의 실행 데모 엔트리
```

> 샘플 문서는 한국어/영어 혼재(E23 오류 등)로 구성되어 있습니다.

---

## 2) 설치 & 빠른 실행

```bash
# 가상환경
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 의존성
pip install -r requirements.txt
```

### 2-1) 요약 트리 + 인덱스 생성

```bash
python raptor_tree.py \
  --docs_dir ./sample_docs \
  --out_dir ./artifacts \
  --branch_factor 6 \
  --levels 3 \
  --chunk_words 120
```

* **Ollama로 실제 요약 사용**(권장):

```bash
python raptor_tree.py \
  --docs_dir ./sample_docs \
  --out_dir ./artifacts \
  --branch_factor 6 --levels 3 --chunk_words 120 \
  --summarizer ollama --ollama_model llama3:instruct --ollama_url http://127.0.0.1:11434
```

> 기본값은 **간이 요약(naive)** 이며, 프로덕션에서는 꼭 **로컬 LLM**으로 요약을 생성하세요.

### 2-2) Self-RAG 게이팅 + 계층형 검색 + Cross-Encoder 리랭크

```bash
python demo.py \
  --artifacts ./artifacts \
  --query "E23 오류로 Wi-Fi 페어링이 중단될 때 어떻게 재시도하나요?" \
  --use_selfrag \
  --use_cross \
  --cross_model cross-encoder/ms-marco-MiniLM-L-6-v2
```

* 콘솔 출력 예시(요약):

  * `[Self-RAG Gate] {"need_search": true, "k": 6, "rewrites": ["..."]}`
  * `=== 최종 후보 문단 ===`와 함께 RRF 점수, Cross-Encoder 점수로 정렬된 문단 목록

> `--use_selfrag` 시 현재는 **휴리스틱 게이트**(숫자/오류코드/버전/최신성 키워드 감지).
> `selfrag_gating.py`에서 `use_llm=True`로 바꾸면 **Ollama LLM JSON 게이트**를 사용합니다.

---

## 3) 핵심 구성 요소 요약

### 3-1) RAPTOR 스타일 요약 트리 (`raptor_tree.py`)

* **Leaf(0레벨)**: 문서 → 고정 길이 **청크**로 분할(`--chunk_words`, `--stride_words`).
* **상위 레벨**: **branch\_factor**만큼 하위 노드를 묶어 **요약 생성** → **부모 노드** 생성.
* **레벨별 FAISS 인덱스** 생성: `faiss_level{L}.index` + `emb_level{L}.npy` + `nodes_level{L}.jsonl`
* 요약기는 **`Summarizer`** 클래스로 추상화:

  * `mode="ollama"`: `http://127.0.0.1:11434/api/generate` 호출
  * `mode="naive"`: 간이 요약(첫 문장 기반)

> 트리 깊이(`--levels`)와 폭(`--branch_factor`)은 데이터 크기/성격에 맞춰 조정하십시오(긴 매뉴얼은 깊이 3~~4, branch 6~~12 권장).

### 3-2) 계층형 검색 + RRF 융합 (`retrieval_pipeline.py`)

* **Coarse 단계**: **상위 레벨**(root에 가까운 레벨)에서 **Top-S** 후보 요약 노드 검색 → 하위로 **자식 확장**.
* **Fine 단계**: 선택된 자식들(대개 leaf 후보)에 대해

  * **Dense**(SentenceTransformer 임베딩) + **BM25** 함께 수행
  * \*\*RRF(Reciprocal Rank Fusion)\*\*로 **강건 융합**
* 결과는 leaf 인덱스와 함께 반환 → 생성 단계에서 **인용·하이라이트** 구성에 사용

### 3-3) Self-RAG 재검색 게이팅 (`selfrag_gating.py`)

* **LLM 게이트(권장)**:

  * 시스템 프롬프트에 따라 **JSON만** 반환: `{"need_search": bool, "k": int, "rewrites": [...]}`
  * 최신성·검증 필요·고유명사/버전/코드·링크 요구 시 `need_search=true` 권장
* **휴리스틱 게이트(기본)**:

  * “오류/E23/버전/최신/URL/절차/숫자” 등 포함 시 검색 필요로 판단
  * **질의 재작성(rewrites)**: 키워드 중심 버전 추가 → **다중 질의 검색 + 융합**에 활용

### 3-4) 크로스-리랭커 배치 최적화 (`demo.py`)

* **CrossEncoder**(`ms-marco-MiniLM-L-6-v2`)로 상위 N개를 **정밀 재랭크**
* 입력 길이에 따른 **배치 크기 자동 조절**(간단 휴리스틱) → GPU/CPU 모두 안정적
* 프로덕션에서는 **로더를 전역 1회 초기화**하고, 큐를 써서 **배치 병합**(micro-batching)하면 Throughput↑

---

## 4) 실무 튜닝 가이드

1. **트리 파라미터**

* `chunk_words`: 100~~180 단어 권장(한국어는 문장 길이 감안), `stride_words`: 1/3~~1/2
* `branch_factor`: 6~~12, `levels`: 3~~4 (코퍼스 규모가 크고 문서가 길수록 상향)

2. **임베딩 모델**

* 다국어 KB(한/영 혼재)는 **BGE-M3** 또는 **jina-embeddings-v3** 권장.
* 도메인 특화 용어가 많다면 **도메인 튜닝**(CAUTION: 라이선스와 데이터 보호)

3. **리트리벌 파이프라인**

* **Top-S(상위)**, **Top-k(leaf)**, **RRF k값**, **BM25 k1/b**를 A/B로 최적화
* **다중 질의(질의 재작성/HyDE)** + **RRF 융합**으로 강건성↑
* **Cross-Encoder**는 상위 20\~50개만 리랭크 → **지연/비용** 제어

4. **Self-RAG 게이팅**

* “날짜/버전/코드/모델명/링크/법규/수치/오류코드/제품명” 포함 시 대부분 검색
* 게이트가 산출한 `k`를 **정책**(SLA·비용)과 연동(예: 모바일=작게, 서버=크게)
* LLM 게이트 출력이 불안정하면 **JSON 파서 + 백오프** 및 **휴리스틱 폴백** 유지

5. **운영·관측**

* **RAGAS** 등으로 *Answer Correctness, Faithfulness, Context Precision/Recall* 모니터링
* **증분 인덱싱**: 문서 버전·언어 태그(`lang: ko|en`)를 메타에 추가
* **근거 하이라이트**: leaf 문장 토큰화 후 스팬 매칭 또는 Cross-Attention 기반 하이라이트

---

## 5) 다음 단계

* 원하시면 **FastAPI + Web UI(하이라이트/출처 표시, 스트리밍)**,
  **Redis/PG Checkpoint**, **BM25+FAISS 하이브리드 가중치 튜너**를 위 레포에 그대로 **추가 패치**해 드리겠습니다.
* 또한 **Ollama/vLLM** 양쪽 모두에서 동작하는 **모델 어댑터 레이어**를 얹어, 요약/게이팅/생성 모두를 **교체 가능**하게 일반화할 수 있습니다.

필요하신 세부 조정(브랜치/레벨/임베딩 교체/HyDE·다중질의·세션 캐시/리랭크 정책) 말씀 주시면 즉시 반영하여 **패치 버전**으로 재구성해 드리겠습니다.

---

> 계층형 retrieval 에 대해 자세히 알려줘. 예시를 보여주고 코드도 제시해줘.
> RAPTOR 스타일 요약 트리 생성 스크립트도 구성해줘. 그리고 self-RAG 재검색 게이팅과 크로스-리랭커 배치최적화도 추가해줘.
> Ollama/vLLM 양쪽 모두에서 동작하는 모델 어댑터 레이어를 얹어, 요약/게이팅/생성 모두를 교체 가능하게 일반화해줘

# 요약

* \*\*계층형 Retrieval(HR)\*\*의 개념과 장점, 실제 **RAPTOR 스타일 요약 트리** 구축 절차와 **Self-RAG 재검색 게이팅**·**크로스-리랭커 배치 최적화**를 한 번에 구현한 **실전용 레포**를 제공해 드렸습니다.
* **Ollama / vLLM(OpenAI-호환)** 양쪽을 지원하는 **모델 어댑터 레이어**를 포함하여, **요약(트리 생성) / 게이팅 / 생성(Answer)** 을 모두 **교체 가능한 구조**로 일반화했습니다.
* 바로 실행 가능한 샘플 문서와 스크립트를 포함합니다.

[📦 hr\_raptor\_adapter.zip 다운로드](sandbox:/mnt/data/hr_raptor_adapter.zip)

---

## 1) 계층형 Retrieval 개념 정리 (실무 관점)

1. **왜 계층형인가?**

* 긴 매뉴얼/규격서/포럼 같은 대규모 KB에서 한 번에 전 범위를 검색하면 \*\*노이즈↑·비용↑\*\*입니다.
* HR은 **상위(요약/섹션) → 하위(문단/문장)** 단계로 후보를 좁혀 **정확도(근거충실도)**·\*\*효율(토큰/지연/비용)\*\*을 동시에 잡습니다.

2. **흐름**

* (Coarse) **상위 레벨** 요약/섹션 인덱스에서 Top-S →
* (Expand) 해당 섹션의 하위 청크(문단/문장)로 확장 →
* (Fine) **Dense + BM25** 하이브리드 검색 → **RRF 융합** →
* (옵션) **Cross-Encoder**로 상위 N개 정밀 리랭크 → **컨텍스트 구성**.

3. **RAPTOR 스타일**

* 문단(leaf)들을 **branch\_factor**로 묶어 **그룹 요약**을 만들고, 그 요약들끼리 다시 묶어 상위 요약…
* 이렇게 **요약 트리**를 만들면, 상위 노드(요약)에서 빠르게 Coarse 검색 후 자식 방향으로 Fine하게 내려갈 수 있습니다.

4. **Self-RAG 게이팅**

* “검색이 필요한가?”, “Top-k는 몇 개?”, “질의 재작성(rewrites)이 필요한가?”를 **LLM/룰**로 결정 → **불필요한 검색을 억제**하면서도 **정확도** 유지.

5. **크로스-리랭커 배치 최적화**

* 문장쌍 평가(CrossEncoder)로 정밀 리랭크 시 **입력 길이 기반 배치 크기 자동 조절**(간단 휴리스틱) → GPU/CPU 모두 안정적 처리량 확보.

---

## 2) 레포 구성 및 설치

```
hr_raptor_adapter/
  README_ko.md
  requirements.txt
  sample_docs/
    hub_guide_ko.md
    ac_manual_en.md
    router_troubleshooting_ko.md
  adapters/
    model_adapter.py          # Ollama / OpenAI-호환(vLLM) 어댑터
  raptor_tree.py              # RAPTOR 트리 생성(+요약기 플러그)
  retrieval_pipeline.py       # HR + BM25+Dense RRF + (옵션) Cross-리랭크
  selfrag_gating.py           # Self-RAG 게이트(LLM/휴리스틱)
  demo.py                     # 게이트→검색→리랭크→(생성) 엔드-투-엔드 데모
```

설치:

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 3) RAPTOR 스타일 요약 트리 생성

### 3-1) 간이 요약(LLM 미사용)

```bash
python raptor_tree.py \
  --docs_dir ./sample_docs \
  --out_dir ./artifacts \
  --branch_factor 6 --levels 3 --chunk_words 120 \
  --llm_backend none
```

### 3-2) Ollama 요약 사용

```bash
python raptor_tree.py \
  --docs_dir ./sample_docs \
  --out_dir ./artifacts \
  --branch_factor 6 --levels 3 --chunk_words 120 \
  --llm_backend ollama \
  --llm_model llama3:instruct \
  --llm_base_url http://127.0.0.1:11434
```

### 3-3) vLLM(OpenAI-호환) 요약 사용

```bash
python raptor_tree.py \
  --docs_dir ./sample_docs \
  --out_dir ./artifacts \
  --branch_factor 6 --levels 3 --chunk_words 120 \
  --llm_backend openai \
  --llm_model qwen2.5-7b-instruct \
  --llm_base_url http://127.0.0.1:8000/v1 \
  --llm_api_key sk-xxx
```

> 생성 산출물: `artifacts/nodes_level{0..L}.jsonl`, `emb_level{L}.npy`, `faiss_level{L}.index`, `tree_meta.json`

---

## 4) Self-RAG 게이팅 + 계층형 검색 + Cross-리랭크

### 4-1) 기본 실행(휴리스틱 게이트)

```bash
python demo.py \
  --artifacts ./artifacts \
  --query "E23 오류로 Wi-Fi 페어링이 중단될 때 어떻게 재시도하나요?" \
  --use_selfrag --selfrag_backend heuristic \
  --use_cross --cross_model cross-encoder/ms-marco-MiniLM-L-6-v2
```

### 4-2) LLM 기반 Self-RAG 게이트 (Ollama)

```bash
python demo.py \
  --artifacts ./artifacts \
  --query "..." \
  --use_selfrag --selfrag_backend ollama \
  --llm_model llama3:instruct \
  --llm_base_url http://127.0.0.1:11434 \
  --use_cross --cross_model cross-encoder/ms-marco-MiniLM-L-6-v2
```

### 4-3) LLM 기반 Self-RAG 게이트 (vLLM/OpenAI-호환)

```bash
python demo.py \
  --artifacts ./artifacts \
  --query "..." \
  --use_selfrag --selfrag_backend openai \
  --llm_model qwen2.5-7b-instruct \
  --llm_base_url http://127.0.0.1:8000/v1 --llm_api_key sk-xxx \
  --use_cross --cross_model cross-encoder/ms-marco-MiniLM-L-6-v2
```

### 4-4) (옵션) 생성 단계까지 실행

```bash
python demo.py \
  --artifacts ./artifacts \
  --query "E23 오류..." \
  --use_selfrag --selfrag_backend heuristic \
  --use_cross \
  --do_generate --gen_backend ollama \
  --llm_model llama3:instruct --llm_base_url http://127.0.0.1:11434
```

* 선택된 후보 문단들을 bullet로 묶어 **근거/출처 포함 답변**을 생성합니다.
* vLLM(OpenAI-호환)로도 동일하게 `--gen_backend openai`로 사용 가능합니다.

---

## 5) 핵심 코드 스니펫

### 5-1) 모델 어댑터 레이어 (Ollama / OpenAI-호환)

```python
# adapters/model_adapter.py (발췌)
class BaseLLM: ...
class OllamaClient(BaseLLM):
    def generate(self, system, user, model, **kw):
        prompt = (f"[System]\n{system}\n\n" if system else "") + f"[User]\n{user}"
        body = {"model": model, "prompt": prompt, "stream": False, ...}
        return requests.post(f"{base}/api/generate", json=body).json()["response"]
class OpenAICompatClient(BaseLLM):
    def generate(self, system, user, model, **kw):
        messages = [{"role":"system","content":system},{"role":"user","content":user}]
        body = {"model": model, "messages": messages, "temperature": 0.2, ...}
        return requests.post(f"{base}/chat/completions", json=body, headers=...).json()["choices"][0]["message"]["content"]
```

### 5-2) RAPTOR 요약 트리 빌더

```python
# raptor_tree.py (발췌)
for lvl in range(1, levels):
    prev = levels_nodes[-1]
    groups = [prev[i:i+branch] for i in range(0, len(prev), branch)]
    cur = []
    for g in groups:
        summary = summarize_group([n.text for n in g], llm_backend, llm_base_url, llm_api_key, llm_model)
        parent = Node(level=lvl, text=summary, children=[n.node_id for n in g], ...)
        for n in g: n.parent = parent.node_id
        cur.append(parent)
    levels_nodes.append(cur)
# 각 레벨 텍스트 임베딩 → FAISS 인덱스 저장
```

### 5-3) HR + BM25+Dense RRF + Cross-리랭커

```python
# retrieval_pipeline.py (발췌)
# 상위 레벨 Top-S 검색 → 자식 leaf 후보 확장
D, I = level_idx[lvl].search(qv, top_sections)  # cosine(IP on normalized)
# leaf 후보에 대해 Dense(FAISS) + BM25
fused = rrf_fuse([dense_rank, bm25_rank], k=60)
# demo.py: CrossEncoder로 상위 N개 정밀 재랭크 + 배치 크기 자동 조절
```

### 5-4) Self-RAG 게이팅(JSON)

```python
# selfrag_gating.py (발췌)
SYS = "Return STRICT JSON: {need_search: bool, k: int, rewrites: [str,...]}"
resp = client.generate(SYS, user=f"Query: {query}\nReturn JSON ONLY.", model=llm_model, response_format="json_object")
data = try_parse_json(resp.text) or heuristic_gate(query)
```

---

## 6) 튜닝 가이드

* **트리 파라미터**: `chunk_words(100~180)`, `stride(1/3~1/2)`, `branch(6~12)`, `levels(3~4)`
* **하이브리드 검색**: Top-S/Top-k, RRF의 `k`(기본 60), BM25 `k1/b`를 A/B로 최적화
* **Cross-리랭크**: 상위 20\~50개만 리랭크, 배치 자동 조절 + 마이크로배칭으로 처리량↑
* **Self-RAG**: 날짜/버전/코드/링크/오류 코드/제품명·규격 등은 검색 필요로 판정, `k`는 SLA/비용에 연동
* **다국어**: 문서 메타에 `lang: ko|en`, 제목/키워드 이중 표기, 임베딩은 BGE-M3 권장
* **운영/관측**: RAGAS 등으로 *Correctness/Faithfulness/Context Precision-Recall* 상시 모니터링, 증분 인덱싱/버저닝

---

## 7) 다음 단계 제안

* **FastAPI + Web UI**(하이라이트/출처 표시, 스트리밍) 통합
* **BM25+FAISS 하이브리드 가중치 튜너**(Grid/Hyperband) 추가
* **세션 캐시/재검색 정책(게이트 임계값)**, **HyDE/다중 질의 RRF 융합** 확장
* **모델 어댑터 레이어**에 **함수 호출/툴 사용 프롬프트**·**안전 라우팅(국가별 정책 오버레이)** 연계

원하시면 위 레포에 바로 **FastAPI + UI + Adapter 라우팅**을 패치하여, 귀하의 온프레미스 환경(Ollama/vLLM)에서 곧장 운영 가능한 형태로 확장해 드리겠습니다.


---
---
---


> 아래 내용 추가해줘
> 
> - BM25+FAISS 하이브리드 가중치 튜너(Grid/Hyperband) 추가
> - 세션 캐시/재검색 정책(게이트 임계값), HyDE/다중 질의 RRF 융합 확장
> - 모델 어댑터 레이어에 함수 호출/툴 사용 프롬프트·안전 라우팅(국가별 정책 오버레이) 연계

# 요약

요청하신 세 항목을 **완전 반영**해 새 버전으로 정리했습니다.

* **BM25+FAISS 하이브리드 가중치 튜너(Grid/Hyperband)**: `tuner.py` (자동으로 `rrf/weighted` 융합과 `dense_w/bm25_w` 최적화)
* **세션 캐시/재검색 정책(게이트 임계값), HyDE/다중 질의 RRF 융합**: `demo.py` 플래그 및 `session_cache.py`, `hyde_multiquery.py`
* **모델 어댑터 레이어에 함수 호출/툴 프롬프트·안전 라우팅(국가별 정책 오버레이)**: `adapters/model_adapter.py` + `policies/policy_overlays.yaml`

  * OpenAI-호환(vLLM) 경로에 **tools/tool\_choice**가 실제로 전달되도록 **완전 연결**해두었습니다.

[📦 hr\_raptor\_adapter\_v3.zip 다운로드](sandbox:/mnt/data/hr_raptor_adapter_v3.zip)

---

## 자세한 설명

### 1) BM25+FAISS 하이브리드 가중치 튜너

* 파일: `tuner.py`
* 방식:

  1. **Grid**로 `alpha=dense_w ∈ {0.0, 0.25, 0.5, 0.75, 1.0}`와 `method ∈ {rrf, weighted}` 전역 탐색
  2. 최적 부근을 **미세 탐색**(간단 Hyperband 스타일)
* 메트릭(예시): top-1 후보 문단에 `keywords` 중 하나 이상 포함 여부(0/1). 실전에서는 **EM/F1**, **RAGAS**, 혹은 휴먼 라벨로 교체 권장.
* 사용:

```bash
python tuner.py --artifacts ./artifacts --eval_file ./eval_set.jsonl --out_file tuner_results.json
```

* `demo.py` 실행 시 `tuner_results.json`이 존재하면 **자동 반영**(융합 방식/가중치).

### 2) 세션 캐시 / 게이트 임계값 / HyDE·다중 질의 RRF 확장

* **세션 캐시**: `session_cache.py` (TTL·간단 LRU)

  * `demo.py`의 `--use_cache --cache_ttl 600`로 활성화
  * key는 질의 재작성 조합 기준으로 구성
* **재검색 정책(게이트 임계값)**:

  * Self-RAG 게이트가 반환한 `k` 기반으로 `--gate_threshold` 이상이면 검색 수행
  * 예) `--gate_threshold 5` → 게이트가 `k=4`면 검색 생략
* **HyDE/다중 질의**: `hyde_multiquery.py`

  * `--use_hyde --multiquery_n 2` 로 활성화 → LLM(또는 휴리스틱)로 가설 문단/질의 생성 → 기존 rewrites와 함께 **RRF/가중 융합**
  * 비용/지연 제어를 위해 `multiquery_n`과 질의 길이 제한을 조절
* 실행 예:

```bash
python demo.py --artifacts ./artifacts \
  --query "E23 오류로 Wi-Fi 페어링이 중단될 때 어떻게 재시도하나요?" \
  --use_selfrag --selfrag_backend heuristic \
  --use_hyde --multiquery_n 2 \
  --use_cache --cache_ttl 600 \
  --fusion weighted --dense_w 0.6 --bm25_w 0.4 \
  --use_cross --cross_model cross-encoder/ms-marco-MiniLM-L-6-v2
```

### 3) 어댑터 레이어: 함수 호출/툴 프롬프트 + 안전 라우팅(국가별 정책)

* 파일: `adapters/model_adapter.py`

  * **Ollama**: `/api/generate`
  * **OpenAI-호환(vLLM 등)**: `/v1/chat/completions`
  * `OpenAICompatClient.generate(...)`가 `tools`/`tool_choice` 파라미터를 **실제로 body에 포함**하도록 구현
* **안전 라우팅**: `SafetyRouter`

  * `policies/policy_overlays.yaml`의 국가 코드(KR/US 등)별 **system prefix**, **blocked\_tools** 사용
  * `demo.py` 생성 단계에서 overlay 적용 후, **차단된 툴 자동 필터링**
* 생성 단계 예:

```bash
python demo.py --artifacts ./artifacts \
  --query "E23 오류..." \
  --use_selfrag --selfrag_backend openai \
  --llm_model qwen2.5-7b-instruct --llm_base_url http://127.0.0.1:8000/v1 --llm_api_key sk-xxx \
  --use_hyde --multiquery_n 2 --use_cache \
  --do_generate --gen_backend openai \
  --policy_overlays ./policies/policy_overlays.yaml --country_code KR
```

---

## 폴더 구조 요약

```
hr_raptor_adapter_v3/
  README_ko.md
  requirements.txt
  sample_docs/
  policies/
    policy_overlays.yaml             # 국가별 정책 오버레이
  adapters/
    model_adapter.py                 # Ollama / OpenAI-호환 클라이언트 + SafetyRouter + tools 전달
  raptor_tree.py                     # RAPTOR 요약 트리 빌더
  retrieval_pipeline.py              # HR + RRF/가중 융합
  selfrag_gating.py                  # Self-RAG 게이트(LLM/휴리스틱)
  hyde_multiquery.py                 # HyDE/다중 질의 생성
  session_cache.py                   # TTL 캐시
  tuner.py                           # Grid/Hyperband 튜너
  eval_set.jsonl                     # 예시 평가셋
```

---

## 실무 팁

* **튜너 메트릭**을 반드시 도메인용으로 교체(정답 라벨/루브릭).
* **HyDE**는 품질 편차가 큽니다. 다중/가설 질의는 1\~3개 정도로 제한하고, Cross-리랭크로 **정밀 필터링**하세요.
* **정책 오버레이**는 국가·제품군·규제 레벨별로 세분화 가능(예: `KR:home_appliance`, `KR:medical`, …).
* **툴 사용**은 OpenAI-호환 백엔드에 최적화되어 있으니, 툴 결과 핸들러(예: `open_url`) 연결만 추가하면 **함수 호출 루프**로 확장할 수 있습니다.

---

필요하시면 **FastAPI + Web UI(출처 하이라이트/툴 버튼/스트리밍)**, **증분 인덱싱 파이프라인**, **다중 백엔드 비용/정책 기반 라우팅**까지 바로 확장해 드릴게요.

