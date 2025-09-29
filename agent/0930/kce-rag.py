# server/kce.py
import numpy as np
from typing import List, Dict, Optional, Any
import faiss
from rank_bm25 import BM25Okapi
import networkx as nx
from dataclasses import dataclass
import pickle
import json
import asyncio
from pathlib import Path

@dataclass
class Document:
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    
class KnowledgeContextEngine:
    def __init__(self, settings):
        self.settings = settings
        self.documents: List[Document] = []
        self.bm25 = None
        self.faiss_index = None
        self.graph = nx.DiGraph()
        self.error_db = {}  # 에러 코드 DB
        self.initialize_indices()
    
    def initialize_indices(self):
        """인덱스 초기화"""
        # FAISS 인덱스 초기화
        self.embedding_dim = self.settings.embedding_dim
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product for cosine similarity
        
        # 샘플 문서 로드 (실제로는 DB나 파일에서)
        self._load_sample_documents()
        
        # BM25 초기화
        self._initialize_bm25()
        
        # GraphRAG 초기화
        self._build_knowledge_graph()
        
        # 에러 코드 DB 초기화
        self._load_error_codes()
    
    def _load_sample_documents(self):
        """샘플 문서 로드"""
        sample_docs = [
            {
                "id": "doc_001",
                "content": "스마트홈 허브 설정 방법: 1. 전원을 연결합니다. 2. WiFi 설정 버튼을 3초간 누릅니다. 3. 모바일 앱에서 기기를 검색합니다.",
                "metadata": {"type": "manual", "device": "smart_hub", "category": "setup"}
            },
            {
                "id": "doc_002",
                "content": "WiFi 연결 실패 시: 라우터를 재시작하고, 2.4GHz 네트워크를 사용하는지 확인하세요. WPA2 보안 설정을 권장합니다.",
                "metadata": {"type": "troubleshooting", "device": "general", "category": "network"}
            },
            {
                "id": "doc_003",
                "content": "LED 깜빡임 패턴: 빨간색 점멸 - 네트워크 오류, 파란색 점멸 - 페어링 모드, 녹색 고정 - 정상 작동",
                "metadata": {"type": "manual", "device": "general", "category": "led_status"}
            },
            {
                "id": "doc_004",
                "content": "에러 코드 E001: 전원 공급 문제입니다. 어댑터를 확인하고 콘센트를 변경해보세요.",
                "metadata": {"type": "error_code", "code": "E001", "category": "power"}
            },
            {
                "id": "doc_005",
                "content": "스마트 조명 설치: 기존 전구를 제거하고 스마트 전구를 설치합니다. 전원을 켜고 5초 내에 3번 껐다 켜면 페어링 모드가 활성화됩니다.",
                "metadata": {"type": "manual", "device": "smart_light", "category": "installation"}
            }
        ]
        
        for doc_data in sample_docs:
            doc = Document(
                id=doc_data["id"],
                content=doc_data["content"],
                metadata=doc_data["metadata"]
            )
            self.documents.append(doc)
    
    def _initialize_bm25(self):
        """BM25 인덱스 초기화"""
        # 문서를 토큰화
        tokenized_docs = [doc.content.split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)
    
    def _build_knowledge_graph(self):
        """지식 그래프 구축"""
        # 문서 간 관계 구축
        for doc in self.documents:
            self.graph.add_node(doc.id, data=doc)
            
            # 같은 디바이스 관련 문서 연결
            device = doc.metadata.get("device")
            if device:
                for other_doc in self.documents:
                    if other_doc.id != doc.id and other_doc.metadata.get("device") == device:
                        self.graph.add_edge(doc.id, other_doc.id, weight=0.8)
            
            # 같은 카테고리 문서 연결
            category = doc.metadata.get("category")
            if category:
                for other_doc in self.documents:
                    if other_doc.id != doc.id and other_doc.metadata.get("category") == category:
                        self.graph.add_edge(doc.id, other_doc.id, weight=0.5)
    
    def _load_error_codes(self):
        """에러 코드 DB 로드"""
        self.error_db = {
            "E001": "전원 공급 문제: 어댑터와 콘센트를 확인하세요.",
            "E002": "네트워크 연결 실패: WiFi 설정을 확인하세요.",
            "E003": "펌웨어 업데이트 필요: 앱에서 업데이트를 진행하세요.",
            "E004": "센서 오류: 센서를 청소하거나 교체가 필요합니다.",
            "E005": "과열 감지: 기기를 끄고 충분히 식힌 후 재시작하세요.",
            "E101": "페어링 실패: 기기를 초기화하고 다시 시도하세요.",
            "E102": "권한 오류: 앱 권한 설정을 확인하세요.",
            "E103": "서버 연결 실패: 인터넷 연결을 확인하세요."
        }
    
    async def search(self, query: str, intent: Optional[str] = None, top_k: int = 5) -> List[Dict]:
        """하이브리드 검색 (BM25 + Vector + Graph)"""
        results = []
        
        # 1. BM25 검색
        bm25_scores = self.bm25.get_scores(query.split())
        bm25_top_indices = np.argsort(bm25_scores)[-top_k:][::-1]
        
        # 2. Vector 검색 (임베딩이 있다면)
        vector_scores = []
        if self.faiss_index.ntotal > 0:
            # 여기서는 모의 임베딩 사용 (실제로는 model_adapter.embed 사용)
            query_embedding = np.random.rand(self.embedding_dim).astype('float32')
            distances, indices = self.faiss_index.search(
                query_embedding.reshape(1, -1), 
                min(top_k, self.faiss_index.ntotal)
            )
            if len(indices[0]) > 0:
                vector_scores = [(idx, score) for idx, score in zip(indices[0], distances[0])]
        
        # 3. Graph 기반 확장 (intent가 있으면)
        graph_docs = []
        if intent:
            # Intent와 관련된 문서 찾기
            for doc in self.documents:
                if doc.metadata.get("type") == intent:
                    # 연결된 문서들도 포함
                    if doc.id in self.graph:
                        neighbors = list(self.graph.neighbors(doc.id))
                        graph_docs.extend([doc.id] + neighbors[:2])
        
        # 4. RRF (Reciprocal Rank Fusion) 적용
        rrf_scores = {}
        k = 60  # RRF 상수
        
        # BM25 scores
        for rank, idx in enumerate(bm25_top_indices):
            doc_id = self.documents[idx].id
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)
        
        # Vector scores
        for rank, (idx, _) in enumerate(vector_scores):
            if idx < len(self.documents):
                doc_id = self.documents[idx].id
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)
        
        # Graph docs boost
        for doc_id in graph_docs:
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 0.2
        
        # 최종 순위 결정
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # 결과 포맷팅
        for doc_id, score in sorted_docs:
            for doc in self.documents:
                if doc.id == doc_id:
                    results.append({
                        "id": doc.id,
                        "content": doc.content,
                        "metadata": doc.metadata,
                        "score": score,
                        "type": doc.metadata.get("type", "general")
                    })
                    break
        
        return results
    
    async def get_error_solution(self, error_code: str) -> Optional[str]:
        """에러 코드 해결책 조회"""
        return self.error_db.get(error_code.upper())
    
    async def add_document(self, content: str, metadata: Dict[str, Any]) -> str:
        """새 문서 추가"""
        doc_id = f"doc_{len(self.documents) + 1:03d}"
        doc = Document(
            id=doc_id,
            content=content,
            metadata=metadata
        )
        
        self.documents.append(doc)
        
        # BM25 재초기화
        self._initialize_bm25()
        
        # Graph에 추가
        self.graph.add_node(doc_id, data=doc)
        
        return doc_id
    
    async def update_embeddings(self, model_adapter):
        """임베딩 업데이트"""
        embeddings = []
        
        for doc in self.documents:
            embedding = await model_adapter.embed(doc.content)
            doc.embedding = np.array(embedding, dtype='float32')
            embeddings.append(doc.embedding)
        
        # FAISS 인덱스 재구축
        if embeddings:
            embeddings_matrix = np.vstack(embeddings)
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
            self.faiss_index.add(embeddings_matrix)
    
    def save_index(self, path: str):
        """인덱스 저장"""
        save_data = {
            "documents": self.documents,
            "graph": nx.node_link_data(self.graph),
            "error_db": self.error_db
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        
        # FAISS 인덱스 저장
        if self.faiss_index:
            faiss.write_index(self.faiss_index, f"{path}.faiss")
    
    def load_index(self, path: str):
        """인덱스 로드"""
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.documents = save_data["documents"]
        self.graph = nx.node_link_graph(save_data["graph"])
        self.error_db = save_data["error_db"]
        
        # BM25 재초기화
        self._initialize_bm25()
        
        # FAISS 인덱스 로드
        faiss_path = f"{path}.faiss"
        if Path(faiss_path).exists():
            self.faiss_index = faiss.read_index(faiss_path)
    
    def health_check(self) -> Dict:
        """헬스 체크"""
        return {
            "status": "healthy",
            "document_count": len(self.documents),
            "graph_nodes": self.graph.number_of_nodes(),
            "graph_edges": self.graph.number_of_edges(),
            "error_codes": len(self.error_db),
            "faiss_vectors": self.faiss_index.ntotal if self.faiss_index else 0
        }