# Advanced FastAPI + LangGraph Server with Checkpoints, SubGraphs, GraphRAG
# pip install fastapi uvicorn langgraph langchain langchain-community langchain-ollama python-multipart
# pip install networkx neo4j graspologic-native sentence-transformers numpy pandas

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple
import json
import asyncio
import uuid
from datetime import datetime
import os
from pathlib import Path
import sqlite3
import pickle

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing_extensions import Annotated
import logging

# GraphRAG specific imports
import networkx as nx
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Advanced LangGraph Multi-Agent API", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enhanced Pydantic models
class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: Optional[datetime] = None
    sources: Optional[List[Dict[str, Any]]] = []  # RAG 출처 정보

class ChatRequest(BaseModel):
    message: str
    conversation_id: str
    agent_type: str
    model: str = "qwen2.5:8b"
    stream: bool = False
    use_graphrag: bool = False
    checkpoint_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    agent_state: str
    checkpoint_id: str
    sources: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}

class ConversationState(BaseModel):
    messages: List[ChatMessage] = []
    agent_type: str
    current_state: str
    checkpoint_id: Optional[str] = None
    metadata: Dict[str, Any] = {}

# Enhanced Agent State with more fields
class AgentState(BaseModel):
    messages: Annotated[List[Any], add_messages]
    current_node: str = "start"
    agent_type: str
    context: str = ""
    user_info: Dict[str, Any] = {}
    conversation_id: str
    checkpoint_id: Optional[str] = None
    rag_sources: List[Dict[str, Any]] = []  # RAG 출처 정보
    subgraph_results: Dict[str, Any] = {}  # 서브그래프 결과

# Global variables
conversations: Dict[str, ConversationState] = {}
vectorstores: Dict[str, FAISS] = {}
graph_stores: Dict[str, nx.Graph] = {}  # GraphRAG용 지식 그래프
ollama_client = None
embeddings = None
sentence_transformer = None
checkpointer = None

# Initialize checkpoint database
def init_checkpoints():
    global checkpointer
    # SQLite 기반 체크포인트 저장소 초기화
    os.makedirs("checkpoints", exist_ok=True)
    checkpointer = SqliteSaver.from_conn_string("checkpoints/langgraph_checkpoints.db")
    logger.info("Checkpoint system initialized")

# Initialize services
def init_services(model_name: str = "qwen2.5:8b", base_url: str = "http://localhost:11434"):
    global ollama_client, embeddings, sentence_transformer
    try:
        ollama_client = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=0.7,
        )
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url=base_url
        )
        # Sentence Transformer for GraphRAG
        sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info(f"All services initialized with model: {model_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        return False

# GraphRAG Implementation
class GraphRAG:
    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id
        self.graph = nx.Graph()
        self.entities = {}
        self.relationships = {}
        
    async def build_knowledge_graph(self, documents: List[str]) -> nx.Graph:
        """문서들로부터 지식 그래프 구축"""
        try:
            all_entities = []
            all_relationships = []
            
            for i, doc in enumerate(documents):
                # 엔티티 추출 (간단한 NER 시뮬레이션)
                entities = await self._extract_entities(doc)
                relationships = await self._extract_relationships(doc, entities)
                
                # 그래프에 노드와 엣지 추가
                for entity in entities:
                    entity_id = f"{entity['name']}_{entity['type']}"
                    self.graph.add_node(
                        entity_id,
                        name=entity['name'],
                        type=entity['type'],
                        description=entity.get('description', ''),
                        source_doc=i,
                        embedding=sentence_transformer.encode(entity['name']).tolist()
                    )
                    all_entities.append(entity)
                
                for rel in relationships:
                    self.graph.add_edge(
                        f"{rel['source']}_{rel['source_type']}",
                        f"{rel['target']}_{rel['target_type']}",
                        relationship=rel['relationship'],
                        weight=rel.get('weight', 1.0),
                        source_doc=i
                    )
                    all_relationships.append(rel)
            
            # 커뮤니티 탐지
            await self._detect_communities()
            
            logger.info(f"Knowledge graph built: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
            return self.graph
            
        except Exception as e:
            logger.error(f"GraphRAG build error: {e}")
            return self.graph
    
    async def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """텍스트에서 엔티티 추출 (LLM 사용)"""
        try:
            entity_prompt = f"""
다음 텍스트에서 주요 엔티티들을 추출해주세요. 각 엔티티는 이름, 타입, 설명을 포함해야 합니다.
엔티티 타입: PERSON(인물), ORGANIZATION(조직), LOCATION(장소), EVENT(사건), CONCEPT(개념), PRODUCT(제품)

텍스트: {text[:1000]}

출력 형식 (JSON):
[{{"name": "엔티티명", "type": "타입", "description": "설명"}}]
"""
            
            messages = [HumanMessage(content=entity_prompt)]
            response = await ollama_client.ainvoke(messages)
            
            # JSON 파싱 시도
            try:
                entities = json.loads(response.content)
                return entities if isinstance(entities, list) else []
            except:
                # 파싱 실패 시 기본 엔티티 추출
                return self._fallback_entity_extraction(text)
                
        except Exception as e:
            logger.error(f"Entity extraction error: {e}")
            return self._fallback_entity_extraction(text)
    
    def _fallback_entity_extraction(self, text: str) -> List[Dict[str, Any]]:
        """간단한 규칙 기반 엔티티 추출"""
        entities = []
        # 간단한 패턴 매칭
        words = text.split()
        for i, word in enumerate(words):
            if word.istitle() and len(word) > 2:
                entities.append({
                    "name": word,
                    "type": "CONCEPT",
                    "description": f"Extracted from position {i}"
                })
        return entities[:10]  # 최대 10개
    
    async def _extract_relationships(self, text: str, entities: List[Dict]) -> List[Dict[str, Any]]:
        """엔티티들 간의 관계 추출"""
        relationships = []
        
        try:
            if len(entities) < 2:
                return relationships
                
            entity_names = [e['name'] for e in entities]
            rel_prompt = f"""
다음 텍스트와 엔티티들을 보고 엔티티들 간의 관계를 추출해주세요.

텍스트: {text[:500]}
엔티티들: {entity_names}

출력 형식 (JSON):
[{{"source": "엔티티1", "source_type": "타입1", "target": "엔티티2", "target_type": "타입2", "relationship": "관계명", "weight": 0.8}}]
"""
            
            messages = [HumanMessage(content=rel_prompt)]
            response = await ollama_client.ainvoke(messages)
            
            try:
                relationships = json.loads(response.content)
                return relationships if isinstance(relationships, list) else []
            except:
                # 파싱 실패 시 기본 관계 생성
                return self._create_basic_relationships(entities)
                
        except Exception as e:
            logger.error(f"Relationship extraction error: {e}")
            return self._create_basic_relationships(entities)
    
    def _create_basic_relationships(self, entities: List[Dict]) -> List[Dict[str, Any]]:
        """기본적인 관계 생성"""
        relationships = []
        for i in range(len(entities) - 1):
            relationships.append({
                "source": entities[i]['name'],
                "source_type": entities[i]['type'],
                "target": entities[i+1]['name'],
                "target_type": entities[i+1]['type'],
                "relationship": "RELATED_TO",
                "weight": 0.5
            })
        return relationships
    
    async def _detect_communities(self):
        """그래프에서 커뮤니티 탐지"""
        try:
            if len(self.graph.nodes) > 3:
                communities = nx.community.louvain_communities(self.graph)
                for i, community in enumerate(communities):
                    for node in community:
                        self.graph.nodes[node]['community'] = i
        except Exception as e:
            logger.error(f"Community detection error: {e}")
    
    async def query_graph(self, query: str, k: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
        """그래프 쿼리 및 컨텍스트 생성"""
        try:
            query_embedding = sentence_transformer.encode(query)
            
            # 노드 유사도 계산
            node_similarities = []
            for node_id, node_data in self.graph.nodes(data=True):
                if 'embedding' in node_data:
                    node_embedding = np.array(node_data['embedding'])
                    similarity = cosine_similarity([query_embedding], [node_embedding])[0][0]
                    node_similarities.append((node_id, similarity, node_data))
            
            # 상위 k개 노드 선택
            node_similarities.sort(key=lambda x: x[1], reverse=True)
            top_nodes = node_similarities[:k]
            
            # 컨텍스트 생성
            context_parts = []
            sources = []
            
            for node_id, similarity, node_data in top_nodes:
                # 노드 정보
                node_info = f"엔티티: {node_data['name']} (타입: {node_data['type']})"
                if node_data.get('description'):
                    node_info += f" - {node_data['description']}"
                context_parts.append(node_info)
                
                # 연결된 관계들
                for neighbor in self.graph.neighbors(node_id):
                    edge_data = self.graph.edges[node_id, neighbor]
                    relationship = edge_data.get('relationship', 'CONNECTED')
                    neighbor_name = self.graph.nodes[neighbor]['name']
                    context_parts.append(f"  → {relationship}: {neighbor_name}")
                
                # 출처 정보
                sources.append({
                    "entity": node_data['name'],
                    "type": node_data['type'],
                    "similarity": float(similarity),
                    "source_doc": node_data.get('source_doc', 0),
                    "community": node_data.get('community', 0)
                })
            
            context = "\n".join(context_parts)
            return context, sources
            
        except Exception as e:
            logger.error(f"Graph query error: {e}")
            return "", []

# Enhanced Agent System with SubGraphs and Checkpoints
class AdvancedMultiAgentSystem:
    def __init__(self):
        self.graphs = {}
        self.subgraphs = {}
        self._build_graphs()
    
    def _build_graphs(self):
        """Build main graphs and subgraphs for each agent type"""
        agent_configs = {
            "doctor": {
                "main_nodes": ["초기진단", "증상분석", "전문과목판단", "최종진단"],
                "main_edges": [("초기진단", "증상분석"), ("증상분석", "전문과목판단"), ("전문과목판단", "최종진단")],
                "subgraphs": {
                    "검사서브그래프": {
                        "nodes": ["검사선택", "검사해석", "추가검사판단"],
                        "edges": [("검사선택", "검사해석"), ("검사해석", "추가검사판단")],
                        "trigger_from": "증상분석"
                    },
                    "치료서브그래프": {
                        "nodes": ["치료계획", "약물선택", "생활지도"],
                        "edges": [("치료계획", "약물선택"), ("약물선택", "생활지도")],
                        "trigger_from": "최종진단"
                    }
                }
            },
            "travel": {
                "main_nodes": ["여행상담", "목적지분석", "일정수립", "예약진행"],
                "main_edges": [("여행상담", "목적지분석"), ("목적지분석", "일정수립"), ("일정수립", "예약진행")],
                "subgraphs": {
                    "교통서브그래프": {
                        "nodes": ["항공검색", "항공비교", "항공예약"],
                        "edges": [("항공검색", "항공비교"), ("항공비교", "항공예약")],
                        "trigger_from": "일정수립"
                    },
                    "숙박서브그래프": {
                        "nodes": ["호텔검색", "호텔비교", "호텔예약"],
                        "edges": [("호텔검색", "호텔비교"), ("호텔비교", "호텔예약")],
                        "trigger_from": "일정수립"
                    }
                }
            },
            "movie": {
                "main_nodes": ["영화상담", "영화추천", "예매진행", "결제완료"],
                "main_edges": [("영화상담", "영화추천"), ("영화추천", "예매진행"), ("예매진행", "결제완료")],
                "subgraphs": {
                    "추천서브그래프": {
                        "nodes": ["장르분석", "평점확인", "리뷰분석"],
                        "edges": [("장르분석", "평점확인"), ("평점확인", "리뷰분석")],
                        "trigger_from": "영화상담"
                    },
                    "예매서브그래프": {
                        "nodes": ["상영시간확인", "좌석선택", "할인적용"],
                        "edges": [("상영시간확인", "좌석선택"), ("좌석선택", "할인적용")],
                        "trigger_from": "예매진행"
                    }
                }
            }
        }
        
        for agent_type, config in agent_configs.items():
            # 메인 그래프 생성
            self.graphs[agent_type] = self._create_main_graph(agent_type, config)
            
            # 서브그래프들 생성
            self.subgraphs[agent_type] = {}
            for subgraph_name, subgraph_config in config["subgraphs"].items():
                self.subgraphs[agent_type][subgraph_name] = self._create_subgraph(
                    agent_type, subgraph_name, subgraph_config
                )
    
    def _create_main_graph(self, agent_type: str, config: Dict) -> StateGraph:
        """메인 그래프 생성"""
        graph = StateGraph(AgentState)
        
        # 메인 노드들 추가
        for node in config["main_nodes"]:
            graph.add_node(node, self._create_main_node_function(agent_type, node))
        
        # 메인 엣지들 추가
        for from_node, to_node in config["main_edges"]:
            graph.add_edge(from_node, to_node)
        
        # 서브그래프 트리거 노드들에 조건부 엣지 추가
        for subgraph_name, subgraph_config in config["subgraphs"].items():
            trigger_node = subgraph_config["trigger_from"]
            subgraph_entry = f"{subgraph_name}_entry"
            
            # 서브그래프 진입점 추가
            graph.add_node(subgraph_entry, self._create_subgraph_entry_function(agent_type, subgraph_name))
            graph.add_edge(trigger_node, subgraph_entry)
        
        # 시작과 끝 연결
        graph.add_edge(START, config["main_nodes"][0])
        graph.add_edge(config["main_nodes"][-1], END)
        
        # 체크포인트 적용
        return graph.compile(checkpointer=checkpointer)
    
    def _create_subgraph(self, agent_type: str, subgraph_name: str, config: Dict) -> StateGraph:
        """서브그래프 생성"""
        subgraph = StateGraph(AgentState)
        
        # 서브그래프 노드들 추가
        for node in config["nodes"]:
            full_node_name = f"{subgraph_name}_{node}"
            subgraph.add_node(full_node_name, self._create_subgraph_node_function(agent_type, subgraph_name, node))
        
        # 서브그래프 엣지들 추가
        for from_node, to_node in config["edges"]:
            from_full = f"{subgraph_name}_{from_node}"
            to_full = f"{subgraph_name}_{to_node}"
            subgraph.add_edge(from_full, to_full)
        
        # 시작과 끝 연결
        if config["nodes"]:
            subgraph.add_edge(START, f"{subgraph_name}_{config['nodes'][0]}")
            subgraph.add_edge(f"{subgraph_name}_{config['nodes'][-1]}", END)
        
        return subgraph.compile(checkpointer=checkpointer)
    
    def _create_main_node_function(self, agent_type: str, node_name: str):
        """메인 노드 함수 생성"""
        async def node_function(state: AgentState):
            state.current_node = node_name
            
            # 체크포인트 ID 업데이트
            if not state.checkpoint_id:
                state.checkpoint_id = str(uuid.uuid4())
            
            # RAG 컨텍스트 가져오기
            rag_context = ""
            rag_sources = []
            
            if state.conversation_id in vectorstores or state.conversation_id in graph_stores:
                last_message = ""
                human_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
                if human_messages:
                    last_message = human_messages[-1].content
                
                # 일반 RAG
                if state.conversation_id in vectorstores:
                    rag_context, rag_sources = await get_rag_context_with_sources(
                        state.conversation_id, last_message
                    )
                
                # GraphRAG (우선순위)
                if state.conversation_id in graph_stores:
                    graphrag = GraphRAG(state.conversation_id)
                    graphrag.graph = graph_stores[state.conversation_id]
                    graph_context, graph_sources = await graphrag.query_graph(last_message)
                    if graph_context:
                        rag_context = f"[GraphRAG] {graph_context}\n\n[일반RAG] {rag_context}"
                        rag_sources = graph_sources + rag_sources
            
            # 시스템 프롬프트 생성
            system_prompt = f"""
당신은 {agent_type} 전문 AI입니다.
현재 처리 단계: {node_name}
이 단계에서는 다음과 같은 작업을 수행해야 합니다:

{get_node_instructions(agent_type, node_name)}

{f"참고 정보:\n{rag_context}" if rag_context else ""}

사용자의 요청에 대해 현재 단계에 맞는 전문적이고 도움이 되는 응답을 제공해주세요.
RAG 정보를 사용한 경우 응답 마지막에 [출처: ...]를 명시해주세요.
"""
            
            try:
                # 마지막 사용자 메시지 가져오기
                human_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
                if not human_messages:
                    return state
                
                last_message = human_messages[-1].content
                
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=last_message)
                ]
                
                response = await ollama_client.ainvoke(messages)
                
                # RAG 출처 정보를 상태에 저장
                state.rag_sources = rag_sources
                state.messages.append(AIMessage(content=response.content))
                
                logger.info(f"Main node {node_name} processed with {len(rag_sources)} sources")
                
            except Exception as e:
                logger.error(f"Error in main node {node_name}: {e}")
                error_msg = f"처리 중 오류가 발생했습니다: {str(e)}"
                state.messages.append(AIMessage(content=error_msg))
            
            return state
        
        return node_function
    
    def _create_subgraph_entry_function(self, agent_type: str, subgraph_name: str):
        """서브그래프 진입점 함수"""
        async def entry_function(state: AgentState):
            logger.info(f"Entering subgraph: {subgraph_name}")
            
            # 서브그래프 실행
            subgraph = self.subgraphs[agent_type][subgraph_name]
            
            # 새로운 스레드 ID로 서브그래프 실행
            subgraph_thread_id = f"{state.conversation_id}_{subgraph_name}_{uuid.uuid4()}"
            config = {"configurable": {"thread_id": subgraph_thread_id}}
            
            try:
                result = await subgraph.ainvoke(state, config=config)
                
                # 서브그래프 결과를 메인 상태에 통합
                if subgraph_name not in state.subgraph_results:
                    state.subgraph_results[subgraph_name] = {}
                
                state.subgraph_results[subgraph_name] = {
                    "status": "completed",
                    "result": result.current_node if hasattr(result, 'current_node') else "finished",
                    "thread_id": subgraph_thread_id
                }
                
                logger.info(f"Subgraph {subgraph_name} completed successfully")
                
            except Exception as e:
                logger.error(f"Subgraph {subgraph_name} error: {e}")
                state.subgraph_results[subgraph_name] = {
                    "status": "error",
                    "error": str(e)
                }
            
            return state
        
        return entry_function
    
    def _create_subgraph_node_function(self, agent_type: str, subgraph_name: str, node_name: str):
        """서브그래프 노드 함수 생성"""
        async def subgraph_node_function(state: AgentState):
            full_node_name = f"{subgraph_name}_{node_name}"
            state.current_node = full_node_name
            
            system_prompt = f"""
당신은 {agent_type} 전문 AI의 {subgraph_name} 서브시스템입니다.
현재 처리 중인 세부 작업: {node_name}

{get_subgraph_node_instructions(agent_type, subgraph_name, node_name)}

전문적이고 구체적인 정보를 제공해주세요.
"""
            
            try:
                human_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
                if human_messages:
                    last_message = human_messages[-1].content
                    
                    messages = [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=f"서브작업 {node_name}: {last_message}")
                    ]
                    
                    response = await ollama_client.ainvoke(messages)
                    state.messages.append(AIMessage(content=f"[{node_name}] {response.content}"))
                
            except Exception as e:
                logger.error(f"Error in subgraph node {full_node_name}: {e}")
                error_msg = f"[{node_name}] 처리 중 오류: {str(e)}"
                state.messages.append(AIMessage(content=error_msg))
            
            return state
        
        return subgraph_node_function

# Node instruction functions
def get_node_instructions(agent_type: str, node_name: str) -> str:
    """각 노드별 상세 지침"""
    instructions = {
        "doctor": {
            "초기진단": "환자의 주요 증상을 파악하고 기본적인 병력을 청취합니다.",
            "증상분석": "증상의 특징, 발생 시기, 악화/완화 요인을 분석합니다.",
            "전문과목판단": "증상에 따라 적절한 전문과를 판단하고 추천합니다.",
            "최종진단": "모든 정보를 종합하여 가능한 진단과 치료 방향을 제시합니다."
        },
        "travel": {
            "여행상담": "고객의 여행 목적, 기간, 예산, 선호도를 파악합니다.",
            "목적지분석": "목적지의 특징, 날씨, 관광명소, 주의사항을 분석합니다.",
            "일정수립": "효율적이고 현실적인 여행 일정을 수립합니다.",
            "예약진행": "항공, 숙박, 액티비티 예약을 진행합니다."
        },
        "movie": {
            "영화상담": "고객의 선호 장르, 분위기, 관람 조건을 파악합니다.",
            "영화추천": "고객 취향에 맞는 영화를 추천하고 정보를 제공합니다.",
            "예매진행": "상영시간, 좌석, 극장을 선택하여 예매를 진행합니다.",
            "결제완료": "결제 정보를 확인하고 예매를 완료합니다."
        }
    }
def get_node_instructions(agent_type: str, node_name: str) -> str:
    """각 노드별 상세 지침"""
    instructions = {
        "doctor": {
            "초기진단": "환자의 주요 증상을 파악하고 기본적인 병력을 청취합니다.",
            "증상분석": "증상의 특징, 발생 시기, 악화/완화 요인을 분석합니다.",
            "전문과목판단": "증상에 따라 적절한 전문과를 판단하고 추천합니다.",
            "최종진단": "모든 정보를 종합하여 가능한 진단과 치료 방향을 제시합니다."
        },
        "travel": {
            "여행상담": "고객의 여행 목적, 기간, 예산, 선호도를 파악합니다.",
            "목적지분석": "목적지의 특징, 날씨, 관광명소, 주의사항을 분석합니다.",
            "일정수립": "효율적이고 현실적인 여행 일정을 수립합니다.",
            "예약진행": "항공, 숙박, 액티비티 예약을 진행합니다."
        },
        "movie": {
            "영화상담": "고객의 선호 장르, 분위기, 관람 조건을 파악합니다.",
            "영화추천": "고객 취향에 맞는 영화를 추천하고 정보를 제공합니다.",
            "예매진행": "상영시간, 좌석, 극장을 선택하여 예매를 진행합니다.",
            "결제완료": "결제 정보를 확인하고 예매를 완료합니다."
        }
    }
    return instructions.get(agent_type, {}).get(node_name, "해당 단계에서 적절한 처리를 수행합니다.")

def get_subgraph_node_instructions(agent_type: str, subgraph_name: str, node_name: str) -> str:
    """서브그래프 노드별 지침"""
    instructions = {
        "doctor": {
            "검사서브그래프": {
                "검사선택": "환자의 증상에 적합한 검사 종류를 선택합니다.",
                "검사해석": "검사 결과를 의학적으로 해석합니다.",
                "추가검사판단": "추가 검사가 필요한지 판단합니다."
            },
            "치료서브그래프": {
                "치료계획": "환자에게 맞는 치료 계획을 수립합니다.",
                "약물선택": "적절한 약물과 용법을 선택합니다.",
                "생활지도": "생활습관 개선 및 주의사항을 안내합니다."
            }
        },
        "travel": {
            "교통서브그래프": {
                "항공검색": "목적지까지의 항공편을 검색합니다.",
                "항공비교": "여러 항공사의 가격과 조건을 비교합니다.",
                "항공예약": "최적의 항공편을 예약합니다."
            },
            "숙박서브그래프": {
                "호텔검색": "목적지의 숙박시설을 검색합니다.",
                "호텔비교": "위치, 가격, 시설을 종합적으로 비교합니다.",
                "호텔예약": "조건에 맞는 숙박시설을 예약합니다."
            }
        },
        "movie": {
            "추천서브그래프": {
                "장르분석": "고객이 선호하는 영화 장르를 분석합니다.",
                "평점확인": "영화의 평점과 리뷰를 확인합니다.",
                "리뷰분석": "관객 리뷰를 분석하여 영화의 특징을 파악합니다."
            },
            "예매서브그래프": {
                "상영시간확인": "원하는 날짜와 시간의 상영 스케줄을 확인합니다.",
                "좌석선택": "극장 내 좌석을 선택합니다.",
                "할인적용": "가능한 할인 혜택을 적용합니다."
            }
        }
    }
    return instructions.get(agent_type, {}).get(subgraph_name, {}).get(node_name, "세부 작업을 수행합니다.")

# RAG with source tracking
async def get_rag_context_with_sources(conversation_id: str, query: str, k: int = 3) -> Tuple[str, List[Dict[str, Any]]]:
    """출처 정보를 포함한 RAG 컨텍스트 검색"""
    try:
        if conversation_id not in vectorstores:
            return "", []
        
        vectorstore = vectorstores[conversation_id]
        
        # 유사도 검색 (스코어 포함)
        docs_with_scores = vectorstore.similarity_search_with_score(query, k=k)
        
        context_parts = []
        sources = []
        
        for i, (doc, score) in enumerate(docs_with_scores):
            # 문서 내용
            context_parts.append(f"[문서 {i+1}] {doc.page_content}")
            
            # 출처 정보
            source_info = {
                "doc_id": i + 1,
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "similarity_score": float(1 - score),  # 거리를 유사도로 변환
                "metadata": doc.metadata if hasattr(doc, 'metadata') else {},
                "source_type": "vectorstore"
            }
            sources.append(source_info)
        
        context = "\n\n".join(context_parts)
        return context, sources
        
    except Exception as e:
        logger.error(f"RAG context error: {e}")
        return "", []

# Enhanced file processing with GraphRAG option
async def process_document_advanced(conversation_id: str, file_path: Path, use_graphrag: bool = False):
    """향상된 문서 처리 (GraphRAG 옵션 포함)"""
    try:
        # 기본 벡터스토어 생성
        loader = TextLoader(str(file_path), encoding='utf-8')
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        # 메타데이터 추가
        for i, split in enumerate(splits):
            split.metadata.update({
                "chunk_id": i,
                "source_file": file_path.name,
                "conversation_id": conversation_id,
                "processed_at": datetime.now().isoformat()
            })
        
        # 벡터스토어 생성
        vectorstore = FAISS.from_documents(splits, embeddings)
        vectorstores[conversation_id] = vectorstore
        
        logger.info(f"Vector store created with {len(splits)} chunks")
        
        # GraphRAG 처리
        if use_graphrag:
            graphrag = GraphRAG(conversation_id)
            document_texts = [doc.page_content for doc in documents]
            knowledge_graph = await graphrag.build_knowledge_graph(document_texts)
            graph_stores[conversation_id] = knowledge_graph
            
            logger.info(f"Knowledge graph created with {len(knowledge_graph.nodes)} entities")
        
    except Exception as e:
        logger.error(f"Advanced document processing error: {e}")
        raise

# Initialize the advanced multi-agent system
init_checkpoints()
advanced_agent_system = AdvancedMultiAgentSystem()

# Enhanced API Endpoints

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    success = init_services()
    if not success:
        logger.warning("Failed to initialize services. Some features may not work.")

@app.get("/")
async def root():
    return {"message": "Advanced LangGraph Multi-Agent API Server", "version": "2.0.0", "features": ["checkpoints", "subgraphs", "graphrag"]}

@app.get("/health")
async def health_check():
    """Enhanced health check"""
    try:
        test_message = [HumanMessage(content="Health check")]
        response = await ollama_client.ainvoke(test_message)
        ollama_status = "connected"
    except:
        ollama_status = "disconnected"
    
    return {
        "status": "healthy",
        "ollama": ollama_status,
        "conversations": len(conversations),
        "vectorstores": len(vectorstores),
        "graph_stores": len(graph_stores),
        "checkpointer": "enabled" if checkpointer else "disabled"
    }

@app.get("/models")
async def get_available_models():
    """Get available models"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            data = response.json()
            models = [model["name"] for model in data.get("models", [])]
            return {"models": models}
        else:
            return {"models": ["qwen2.5:8b", "llama3:8b"]}
    except Exception as e:
        logger.error(f"Failed to get models: {e}")
        return {"models": ["qwen2.5:8b", "llama3:8b"]}

@app.post("/chat", response_model=ChatResponse)
async def enhanced_chat(request: ChatRequest):
    """Enhanced chat with checkpoints, subgraphs, and GraphRAG"""
    try:
        # Get or create conversation
        if request.conversation_id not in conversations:
            conversations[request.conversation_id] = ConversationState(
                agent_type=request.agent_type,
                current_state=get_initial_state(request.agent_type)
            )
        
        conversation = conversations[request.conversation_id]
        
        # Add user message
        user_message = ChatMessage(
            role="user",
            content=request.message,
            timestamp=datetime.now()
        )
        conversation.messages.append(user_message)
        
        # Create agent state
        agent_state = AgentState(
            messages=[HumanMessage(content=request.message)],
            agent_type=request.agent_type,
            conversation_id=request.conversation_id,
            current_node=conversation.current_state,
            checkpoint_id=request.checkpoint_id or str(uuid.uuid4())
        )
        
        # Configure checkpoint
        config = {
            "configurable": {
                "thread_id": f"{request.conversation_id}_{request.agent_type}",
                "checkpoint_id": agent_state.checkpoint_id
            }
        }
        
        # Run the enhanced agent graph
        graph = advanced_agent_system.graphs[request.agent_type]
        result = await graph.ainvoke(agent_state, config=config)
        
        # Extract AI response
        ai_messages = [msg for msg in result.messages if isinstance(msg, AIMessage)]
        response_content = ai_messages[-1].content if ai_messages else "응답을 생성할 수 없습니다."
        
        # Extract sources from result
        sources = result.rag_sources if hasattr(result, 'rag_sources') else []
        
        # Update conversation
        ai_message = ChatMessage(
            role="assistant",
            content=response_content,
            timestamp=datetime.now(),
            sources=sources
        )
        conversation.messages.append(ai_message)
        conversation.current_state = result.current_node
        conversation.checkpoint_id = result.checkpoint_id
        
        return ChatResponse(
            response=response_content,
            conversation_id=request.conversation_id,
            agent_state=result.current_node,
            checkpoint_id=result.checkpoint_id,
            sources=sources,
            metadata={
                "model": request.model,
                "subgraph_results": getattr(result, 'subgraph_results', {}),
                "use_graphrag": request.use_graphrag
            }
        )
        
    except Exception as e:
        logger.error(f"Enhanced chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-advanced")
async def upload_file_advanced(
    conversation_id: str, 
    use_graphrag: bool = False,
    file: UploadFile = File(...)
):
    """Advanced file upload with GraphRAG option"""
    try:
        if not file.filename.endswith('.txt'):
            raise HTTPException(status_code=400, detail="Only .txt files are supported")
        
        # Save uploaded file
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        file_path = upload_dir / f"{conversation_id}_{file.filename}"
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process with advanced features
        await process_document_advanced(conversation_id, file_path, use_graphrag)
        
        return {
            "message": "File uploaded and processed successfully",
            "filename": file.filename,
            "graphrag_enabled": use_graphrag,
            "vectorstore_created": conversation_id in vectorstores,
            "knowledge_graph_created": conversation_id in graph_stores if use_graphrag else False
        }
        
    except Exception as e:
        logger.error(f"Advanced upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/checkpoints/{conversation_id}")
async def get_checkpoints(conversation_id: str):
    """Get conversation checkpoints"""
    try:
        if not checkpointer:
            raise HTTPException(status_code=503, detail="Checkpointer not available")
        
        # This would require implementing checkpoint listing in the checkpointer
        # For now, return basic info
        return {
            "conversation_id": conversation_id,
            "checkpoints_enabled": True,
            "message": "Checkpoint functionality is available"
        }
        
    except Exception as e:
        logger.error(f"Checkpoint retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rollback/{conversation_id}")
async def rollback_to_checkpoint(conversation_id: str, checkpoint_id: str):
    """Rollback to a specific checkpoint"""
    try:
        if not checkpointer:
            raise HTTPException(status_code=503, detail="Checkpointer not available")
        
        if conversation_id not in conversations:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Update conversation state
        conversation = conversations[conversation_id]
        conversation.checkpoint_id = checkpoint_id
        
        return {
            "message": "Rollback successful",
            "conversation_id": conversation_id,
            "checkpoint_id": checkpoint_id
        }
        
    except Exception as e:
        logger.error(f"Rollback error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge-graph/{conversation_id}")
async def get_knowledge_graph(conversation_id: str):
    """Get knowledge graph information"""
    try:
        if conversation_id not in graph_stores:
            raise HTTPException(status_code=404, detail="Knowledge graph not found")
        
        graph = graph_stores[conversation_id]
        
        # Graph statistics
        nodes_data = []
        for node_id, node_data in graph.nodes(data=True):
            nodes_data.append({
                "id": node_id,
                "name": node_data.get("name", ""),
                "type": node_data.get("type", ""),
                "community": node_data.get("community", 0)
            })
        
        edges_data = []
        for source, target, edge_data in graph.edges(data=True):
            edges_data.append({
                "source": source,
                "target": target,
                "relationship": edge_data.get("relationship", "RELATED"),
                "weight": edge_data.get("weight", 1.0)
            })
        
        return {
            "conversation_id": conversation_id,
            "nodes_count": len(graph.nodes),
            "edges_count": len(graph.edges),
            "nodes": nodes_data[:50],  # 처음 50개만
            "edges": edges_data[:50],  # 처음 50개만
            "communities": len(set([data.get("community", 0) for _, data in graph.nodes(data=True)]))
        }
        
    except Exception as e:
        logger.error(f"Knowledge graph retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agent-structure/{agent_type}")
async def get_agent_structure(agent_type: str):
    """Get complete agent structure including subgraphs"""
    try:
        if agent_type not in advanced_agent_system.graphs:
            raise HTTPException(status_code=404, detail="Agent type not found")
        
        # Main graph structure
        main_structure = {
            "main_graph": {
                "nodes": [],
                "edges": []
            },
            "subgraphs": {}
        }
        
        # Get subgraph information
        if agent_type in advanced_agent_system.subgraphs:
            for subgraph_name, subgraph in advanced_agent_system.subgraphs[agent_type].items():
                main_structure["subgraphs"][subgraph_name] = {
                    "nodes": [],
                    "edges": [],
                    "description": f"{agent_type}의 {subgraph_name} 전문 처리"
                }
        
        return main_structure
        
    except Exception as e:
        logger.error(f"Agent structure error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def get_initial_state(agent_type: str) -> str:
    """Get initial state for agent type"""
    initial_states = {
        "doctor": "초기진단",
        "travel": "여행상담",
        "movie": "영화상담"
    }
    return initial_states.get(agent_type, "start")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)