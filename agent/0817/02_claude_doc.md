<img width="4096" height="2792" alt="image" src="https://github.com/user-attachments/assets/1500fa7b-1d9f-4bb2-b742-35a8061c9393" />

```plantUML
@startuml Improved_IoT_AI_Chatbot_Architecture

!define RECTANGLE class
!define COMPONENT component

skinparam componentStyle uml2
skinparam backgroundColor white
skinparam component {
    BackgroundColor lightblue
    BorderColor darkblue
}

title 개선된 IoT AI 챗봇 시스템 아키텍처 (Edge-Fog-Cloud)

package "Edge Tier (On-Device + Local)" as EdgeTier {
    
    package "System-Level LLM Service (LLMaaS)" as LLMaaS {
        [Shared Memory Pool\n(Gemma 3N)] as SharedMemory
        [Dynamic Model Loading] as DynamicLoading
        [Resource Arbiter] as ResourceArbiter
        [Quantization Controller] as QuantController
    }
    
    package "Mobile App (Thin Client)" as MobileApp {
        [Camera Preview + AR] as CameraAR
        [Conversational Voice UI] as ConversationalUI
        [Context-aware Panels] as ContextPanels
        [Real-time Dashboard] as Dashboard
    }
    
    package "Local Processing Coordinator" as LocalCoordinator {
        [STT Engine (Native+Whisper)] as STTEngine
        [TTS Engine (Neural)] as TTSEngine
        [LLMaaS Client] as LLMaaSClient
        [Image Pipeline] as ImagePipeline
    }
    
    package "State Management Layer" as StateManagement {
        [Event Sourcing Client] as EventSourcing
        [CRDT State Sync] as CRDTSync
        [Conflict Resolution] as ConflictResolution
        [Priority Queue Manager] as PriorityQueue
    }
    
    package "Local Knowledge Cache" as LocalCache {
        [FAQ Cache (Compressed)] as FAQCache
        [Device Context] as DeviceContext
        [User Profile] as UserProfile
        [Offline Buffer] as OfflineBuffer
    }
}

package "Fog Tier (Regional Edge)" as FogTier {
    
    package "Edge Orchestrator" as EdgeOrchestrator {
        [Request Load Balancer] as LoadBalancer
        [Model Weight Cache] as ModelWeightCache
        [Dynamic Scaling Controller] as ScalingController
        [Health Check & Failover] as HealthCheck
    }
    
    package "Distributed Inference Pool" as InferencePool {
        [GPU Inference Nodes\n(vLLM + TensorRT)] as GPUNodes
        [CPU Inference Nodes\n(Ollama optimized)] as CPUNodes
        [Specialized Vision Nodes] as VisionNodes
        [Memory-optimized Cache] as MemoryCache
    }
    
    package "Regional Knowledge Cache" as RegionalCache {
        [Hot Embeddings] as HotEmbeddings
        [Localized FAQ] as LocalizedFAQ
        [Device Compatibility Matrix] as CompatibilityMatrix
        [A/B Test Configs] as ABTestConfigs
    }
}

package "Cloud Tier (Global Backend)" as CloudTier {
    
    package "API Gateway & Traffic Management" as APIGateway {
        [Global Load Balancer] as GlobalLB
        [Authentication & Authorization] as AuthService
        [Rate Limiting & DDoS] as RateLimit
        [Request Routing] as RequestRouter
    }
    
    package "Microworkflow Orchestrator" as MicroworkflowOrch {
        [Workflow Registry] as WorkflowRegistry
        [DeviceRegistrationGraph] as DeviceRegGraph
        [TroubleshootingGraph] as TroubleshootGraph
        [ManualGuideGraph] as ManualGraph
        [PurchaseGuideGraph] as PurchaseGraph
        [Event Bus (Kafka)] as EventBus
        [Workflow Composer] as WorkflowComposer
        [State Coordination] as StateCoordination
    }
    
    package "Advanced Model Management" as AdvancedModelMgmt {
        [Adaptive Model Router] as AdaptiveRouter
        [Multi-modal Fusion Engine] as MultimodalFusion
        [Large Context Models] as LargeContextModels
        [Fine-tuning & Personalization] as Finetuning
        [Model A/B Testing] as ModelABTesting
    }
    
    package "Enterprise Tool Services" as EnterpriseTools {
        [Advanced Vision AI] as AdvancedVision
        [Complex RAG Pipeline] as ComplexRAG
        [Multi-language Translation] as MultiLangTranslation
        [External API Hub] as ExternalAPIHub
        [Workflow Analytics] as WorkflowAnalytics
    }
}

package "Knowledge Tier (Distributed)" as KnowledgeTier {
    
    package "Knowledge Ingestion Pipeline" as IngestionPipeline {
        [Multi-source Crawlers] as MultiCrawlers
        [Real-time Update Streaming] as RealtimeUpdates
        [Quality Assessment] as QualityAssessment
        [Version Control] as VersionControl
        [Automated Fact Checking] as FactChecking
    }
    
    package "Intelligent Processing" as IntelligentProcessing {
        [Advanced Chunking] as AdvancedChunking
        [Multi-level Embedding] as MultiLevelEmbedding
        [Graph Relationship Extraction] as GraphExtraction
        [Cross-reference Linking] as CrossReference
        [Contextual Metadata] as ContextualMetadata
    }
    
    package "Hybrid Storage & Indexing" as HybridStorage {
        [Hot: Redis Vector] as RedisVector
        [Warm: FAISS Cluster] as FAISSCluster
        [Cold: Milvus Distributed] as MilvusDistributed
        [Graph: Neo4j Cluster] as Neo4jCluster
        [Search: Elasticsearch] as Elasticsearch
        [Object: MinIO/S3] as ObjectStorage
    }
    
    package "Advanced Retrieval Engine" as AdvancedRetrieval {
        [Multi-stage Pipeline] as MultiStagePipeline
        [Query Understanding] as QueryUnderstanding
        [Adaptive Strategy] as AdaptiveStrategy
        [Source Attribution] as SourceAttribution
    }
}

package "Security & Privacy Layer" as SecurityLayer {
    [Multi-factor Authentication] as MFA
    [Device Certificate Mgmt] as DeviceCert
    [End-to-End Encryption] as E2EEncryption
    [Homomorphic Encryption] as HomomorphicEnc
    [Differential Privacy] as DifferentialPrivacy
    [Federated Learning] as FederatedLearning
}

package "Observability Platform" as ObservabilityPlatform {
    [Edge Monitoring] as EdgeMonitoring
    [Fog Monitoring] as FogMonitoring
    [Cloud Monitoring] as CloudMonitoring
    [Business Intelligence] as BusinessIntel
    [Prometheus Stack] as PrometheusStack
    [Distributed Tracing] as DistributedTracing
}

cloud "Internet & CDN" as Internet

' Edge Tier Internal Connections
ConversationalUI --> STTEngine
STTEngine --> LLMaaSClient
LLMaaSClient --> SharedMemory
SharedMemory --> ResourceArbiter
LLMaaSClient --> TTSEngine
CameraAR --> ImagePipeline
ImagePipeline --> EventSourcing
EventSourcing --> CRDTSync
CRDTSync --> ConflictResolution
LocalCache --> FAQCache
LocalCache --> OfflineBuffer

' Edge to Fog Communication
ConflictResolution --> Internet
Internet --> LoadBalancer

' Fog Tier Internal Connections
LoadBalancer --> ScalingController
ScalingController --> GPUNodes
ScalingController --> CPUNodes
ScalingController --> VisionNodes
ModelWeightCache --> GPUNodes
ModelWeightCache --> CPUNodes
RegionalCache --> HotEmbeddings
RegionalCache --> CompatibilityMatrix

' Fog to Cloud Communication
GPUNodes --> Internet
CPUNodes --> Internet
Internet --> GlobalLB

' Cloud Tier Internal Connections
GlobalLB --> AuthService
AuthService --> RateLimit
RateLimit --> RequestRouter
RequestRouter --> WorkflowRegistry

WorkflowRegistry --> DeviceRegGraph
WorkflowRegistry --> TroubleshootGraph
WorkflowRegistry --> ManualGraph
WorkflowRegistry --> PurchaseGraph

DeviceRegGraph --> EventBus
TroubleshootGraph --> EventBus
EventBus --> WorkflowComposer
WorkflowComposer --> StateCoordination

StateCoordination --> AdaptiveRouter
AdaptiveRouter --> MultimodalFusion
AdaptiveRouter --> LargeContextModels
AdaptiveRouter --> Finetuning

WorkflowComposer --> AdvancedVision
WorkflowComposer --> ComplexRAG
WorkflowComposer --> MultiLangTranslation
WorkflowComposer --> ExternalAPIHub

' Knowledge Tier Connections
MultiCrawlers --> RealtimeUpdates
RealtimeUpdates --> QualityAssessment
QualityAssessment --> AdvancedChunking
AdvancedChunking --> MultiLevelEmbedding
MultiLevelEmbedding --> RedisVector
MultiLevelEmbedding --> FAISSCluster
MultiLevelEmbedding --> MilvusDistributed

GraphExtraction --> Neo4jCluster
AdvancedChunking --> Elasticsearch
ContextualMetadata --> ObjectStorage

ComplexRAG --> MultiStagePipeline
MultiStagePipeline --> QueryUnderstanding
QueryUnderstanding --> AdaptiveStrategy
AdaptiveStrategy --> SourceAttribution

RedisVector --> MultiStagePipeline
FAISSCluster --> MultiStagePipeline
Neo4jCluster --> MultiStagePipeline

' Security Layer Connections
AuthService --> MFA
AuthService --> DeviceCert
EventSourcing --> E2EEncryption
ImagePipeline --> HomomorphicEnc
ComplexRAG --> DifferentialPrivacy
Finetuning --> FederatedLearning

' Observability Connections
LLMaaSClient --> EdgeMonitoring
GPUNodes --> FogMonitoring
WorkflowRegistry --> CloudMonitoring
WorkflowAnalytics --> BusinessIntel

EdgeMonitoring --> PrometheusStack
FogMonitoring --> PrometheusStack
CloudMonitoring --> PrometheusStack
PrometheusStack --> DistributedTracing

note top of EdgeTier : **Edge Tier 특징**\n- 시스템 레벨 LLM 공유 (40-60% 메모리 절약)\n- 즉시 응답 (<100ms)\n- 완전 오프라인 지원\n- 프라이버시 우선 처리

note top of FogTier : **Fog Tier 특징**\n- 지역별 분산 추론 (200-400ms)\n- 모델 웨이트 캐싱\n- 동적 스케일링\n- 네트워크 적응형 라우팅

note top of CloudTier : **Cloud Tier 특징**\n- 마이크로워크플로우 오케스트레이션\n- 복합 AI 추론 (800-1200ms)\n- 글로벌 학습 및 최적화\n- 엔터프라이즈 통합

note top of KnowledgeTier : **Knowledge Tier 특징**\n- 계층화된 지식 저장\n- 실시간 업데이트 스트리밍\n- 다단계 검색 파이프라인\n- 소스 추적 및 신뢰성

note left of SecurityLayer : **보안 강화**\n- 제로 트러스트 아키텍처\n- 동형 암호화\n- 연합 학습\n- 차분 프라이버시

note right of ObservabilityPlatform : **전계층 관측성**\n- 실시간 성능 모니터링\n- 비용 최적화 인사이트\n- 예측 기반 스케일링\n- 사용자 여정 분석

@enduml

```

---

<img width="2299" height="2031" alt="image" src="https://github.com/user-attachments/assets/1c8c80d4-d089-4460-8b80-1b2f0358330f" />

```plantUML
' 데이터 플로우 다이어그램 (개선)
@startuml Improved_Data_Flow

title 개선된 데이터 플로우 다이어그램 (적응형 처리)

actor User as user
box "Edge Tier" #LightBlue
participant "Mobile App" as mobile
participant "LLMaaS" as llmaas
participant "Event Sourcing" as events
participant "Local Cache" as cache
end box

box "Fog Tier" #LightGreen  
participant "Load Balancer" as foglb
participant "GPU Inference" as foggpu
participant "Regional Cache" as fogcache
end box

box "Cloud Tier" #LightYellow
participant "Global LB" as cloudlb
participant "Workflow Orchestrator" as orchestrator
participant "Model Router" as router
participant "Knowledge Engine" as knowledge
end box

user -> mobile : Voice Input (PTT)
mobile -> llmaas : Audio Stream

llmaas -> llmaas : System-level STT + Intent Classification
llmaas -> events : Log Interaction Event

alt Edge Processing (Simple FAQ)
    llmaas -> cache : Query Local Knowledge
    cache -> llmaas : Cached Response
    llmaas -> mobile : Immediate Response
    mobile -> user : Voice Output (<100ms)
    
else Fog Processing (Medium Complexity)
    llmaas -> events : Queue Request with Context
    events -> foglb : Encrypted Request
    foglb -> foggpu : Route to Optimal Node
    foggpu -> fogcache : Fetch Regional Knowledge
    fogcache -> foggpu : Hot Embeddings + FAQ
    foggpu -> foggpu : Regional Inference
    foggpu -> foglb : Generated Response
    foglb -> events : Response + Metadata
    events -> mobile : Update Local State
    mobile -> user : Voice Output (200-400ms)
    
else Cloud Processing (Complex Workflow)
    llmaas -> events : Queue Complex Request
    events -> cloudlb : Full Context Transfer
    cloudlb -> orchestrator : Route to Microworkflow
    
    orchestrator -> orchestrator : Select Workflow Graph
    orchestrator -> router : Adaptive Model Selection
    orchestrator -> knowledge : Multi-stage Retrieval
    
    par Parallel Processing
        router -> router : Multi-modal Fusion
        knowledge -> knowledge : Graph + Vector Search
        orchestrator -> orchestrator : State Coordination
    end
    
    knowledge -> orchestrator : Retrieved Context
    router -> orchestrator : Generated Response
    orchestrator -> cloudlb : Final Response + Actions
    
    cloudlb -> events : Structured Response
    events -> mobile : State Update + UI Actions
    mobile -> user : Voice Output + Visual Guide (800-1200ms)
    
    opt Follow-up Actions Required
        mobile -> user : Context-aware Action Panel
        user -> mobile : User Action/Confirmation
        mobile -> events : Action Result
        events -> cloudlb : Action Feedback
        orchestrator -> orchestrator : Update Workflow State
    end
end

alt Network Offline/Degraded
    events -> events : Queue All Requests
    llmaas -> cache : Extended Local Processing
    cache -> llmaas : Best-effort Response
    llmaas -> mobile : Limited Offline Response
    mobile -> user : "Limited info, will sync when online"
    
    loop Background Sync
        events -> events : Monitor Network Quality
        events -> foglb : Attempt Sync (when available)
        note over events : CRDT-based conflict resolution
    end
end

alt Error Handling & Recovery
    foggpu -> foggpu : Detect Processing Error
    foggpu -> cloudlb : Escalate to Cloud
    cloudlb -> orchestrator : Handle with Fallback Model
    orchestrator -> events : Error Context + Recovery Plan
end

@enduml
```

---

<img width="3893" height="1860" alt="image" src="https://github.com/user-attachments/assets/d2c6090a-9d0f-42f4-a3bf-68fe2bd51890" />

```plantUML
' 배포 다이어그램 (개선)
@startuml Improved_Deployment

!define NODE node

title 개선된 시스템 배포 다이어그램 (다계층 분산)

NODE "Edge Devices" as EdgeDevices {
    NODE "Android Devices" as AndroidDevices {
        [Android App + LLMaaS] as AndroidApp
    }
    NODE "iOS Devices" as iOSDevices {
        [iOS App + LLMaaS] as iOSApp
    }
    NODE "Local Edge Servers" as LocalEdge {
        [Edge Inference Cache] as EdgeCache
        [Local Model Repository] as LocalRepo
    }
}

cloud "5G/WiFi Network" as Network5G

NODE "Regional Fog Infrastructure" as FogInfra {
    NODE "Fog Cluster (Seoul)" as SeoulFog {
        [Load Balancer] as SeoulLB
        [GPU Nodes (4x T4)] as SeoulGPU
        [CPU Nodes (8x)] as SeoulCPU
        [Regional Cache] as SeoulCache
    }
    
    NODE "Fog Cluster (Tokyo)" as TokyoFog {
        [Load Balancer] as TokyoLB
        [GPU Nodes (4x T4)] as TokyoGPU
        [CPU Nodes (8x)] as TokyoCPU
        [Regional Cache] as TokyoCache
    }
    
    NODE "Fog Cluster (Singapore)" as SingaporeFog {
        [Load Balancer] as SingaporeLB
        [GPU Nodes (6x A100)] as SingaporeGPU
        [CPU Nodes (12x)] as SingaporeCPU
        [Regional Cache] as SingaporeCache
    }
}

cloud "Global CDN & Load Balancer" as GlobalCDN

NODE "Global Cloud Infrastructure" as CloudInfra {
    NODE "Kubernetes Cluster (Primary)" as K8sPrimary {
        NODE "API Tier" as APITier {
            [FastAPI Pod 1] as API1
            [FastAPI Pod 2] as API2
            [FastAPI Pod 3] as API3
        }
        
        NODE "Orchestration Tier" as OrchTier {
            [Workflow Engine 1] as WF1
            [Workflow Engine 2] as WF2
            [Event Bus (Kafka)] as Kafka
            [State Coordinator] as StateCoord
        }
        
        NODE "Model Tier" as ModelTier {
            [vLLM GPU Cluster] as vLLMCluster
            [Ollama CPU Cluster] as OllamaCluster
            [Specialized Vision Cluster] as VisionCluster
        }
    }
    
    NODE "Kubernetes Cluster (DR)" as K8sDR {
        [Standby Services] as StandbyServices
        [Data Replication] as DataReplication
    }
}

NODE "Data & Storage Layer" as DataLayer {
    database "PostgreSQL Cluster\n(Primary + 2 Replicas)" as PGCluster
    database "Redis Cluster\n(6 nodes)" as RedisCluster
    database "FAISS Distributed\n(Vector Index)" as FAISSDistributed
    database "Neo4j Cluster\n(Graph Knowledge)" as Neo4jCluster
    storage "MinIO Cluster\n(Distributed Object)" as MinIOCluster
    component "Elasticsearch Cluster\n(Full-text Search)" as ESCluster
}

cloud "External Services" as ExternalSvcs {
    [GPT-4 API] as GPT4API
    [Claude API] as ClaudeAPI
    [Google Translate] as GoogleTranslate
    [IoT Vendor APIs] as IoTAPIs
}

NODE "Monitoring & Security Stack" as MonitoringStack {
    [Prometheus + Grafana] as PromGrafana
    [Jaeger Tracing] as JaegerTracing
    [ELK Stack] as ELKStack
    [Vault (Secrets)] as VaultSecrets
    [Istio Service Mesh] as IstioMesh
}

' Edge to Fog connections
EdgeDevices --> Network5G
Network5G --> FogInfra : "Regional routing based on latency"

' Fog internal connections
SeoulLB --> SeoulGPU
SeoulLB --> SeoulCPU
TokyoLB --> TokyoGPU
TokyoLB --> TokyoCPU
SingaporeLB --> SingaporeGPU
SingaporeLB --> SingaporeCPU

' Fog to Cloud connections
FogInfra --> GlobalCDN : "Complex queries & model updates"
GlobalCDN --> CloudInfra : "Global load balancing"

' Cloud internal connections
APITier --> OrchTier : "Request processing"
OrchTier --> ModelTier : "Model inference"
OrchTier --> DataLayer : "State & knowledge access"

' Cross-region replication
K8sPrimary --> K8sDR : "Active-passive DR"
PGCluster --> K8sDR : "Data replication"

' External integrations
ModelTier --> ExternalSvcs : "Fallback & specialized APIs"
OrchTier --> IoTAPIs : "Device vendor integrations"

' Monitoring connections
EdgeDevices --> MonitoringStack : "Edge metrics"
FogInfra --> MonitoringStack : "Fog metrics"
CloudInfra --> MonitoringStack : "Cloud metrics"
DataLayer --> MonitoringStack : "Data metrics"

note bottom of EdgeDevices : **Edge Computing**\n- LLMaaS 시스템 서비스 공유\n- 메모리 사용량 60% 절감\n- 오프라인 지원 85%\n- 즉시 응답 <100ms

note bottom of FogInfra : **Regional Fog**\n- 지역별 최적화된 모델\n- GPU 공유 풀 운영\n- 동적 로드 밸런싱\n- 200-400ms 응답시간

note bottom of CloudInfra : **Global Cloud**\n- 마이크로워크플로우 오케스트레이션\n- 멀티모달 AI 처리\n- 글로벌 학습 & 최적화\n- 99.9% 가용성 보장

note bottom of DataLayer : **Hybrid Storage**\n- 계층화된 데이터 관리\n- 실시간 동기화\n- 자동 백업 & 복구\n- GDPR 컴플라이언스

note bottom of MonitoringStack : **전계층 관측성**\n- 실시간 성능 대시보드\n- 예측 기반 알림\n- 비용 최적화 인사이트\n- 보안 위협 탐지

@enduml
```
