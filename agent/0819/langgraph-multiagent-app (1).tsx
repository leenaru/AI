      {/* Enhanced Right Panel */}
      {currentConv && (
        <div className="w-96 bg-white border-l p-4 overflow-y-auto">
          {/* Graph Visualization */}
          <GraphVisualizer 
            agentType={currentConv.agent} 
            currentState={currentState}
            showSubgraphs={true}
            showGraphInfo={showGraphInfo}
            setShowGraphInfo={setShowGraphInfo}
          />
          
          {/* Current State Info */}
          <div className="mt-4 p-4 bg-gray-50 rounded-lg">
            <h3 className="text-sm font-semibold mb-2 flex items-center">
              <GitBranch className="w-4 h-4 mr-2" />
              현재 상태
            </h3>
            <div className="text-sm">
              <span className="font-medium text-blue-600">{currentState}</span>
            </div>
          </div>

          {/* Checkpoint History */}
          {checkpointHistory.length > 0 && (
            <div className="mt-4 p-4 bg-gray-50 rounded-lg">
              <h3 className="text-sm font-semibold mb-2 flex items-center">
                <RotateCcw className="w-4 h-4 mr-2" />
                체크포인트 히스토리
              </h3>
              <div className="space-y-2 max-h-32 overflow-y-auto">
                {checkpointHistory.slice(-5).map((checkpoint, index) => (
                  <div 
                    key={checkpoint.id} 
                    className="text-xs bg-white p-2 rounded border cursor-pointer hover:bg-blue-50"
                    onClick={() => rollbackToCheckpoint(checkpoint.id)}
                  >
                    <div className="font-medium">{checkpoint.state}</div>
                    <div className="text-gray-500">{checkpoint.message}</div>
                    <div className="text-gray-400">{checkpoint.timestamp.toLocaleTimeString()}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Knowledge Graph Info */}
          {useGraphRAG && knowledgeGraphData && (
            <div className="mt-4 p-4 bg-gray-50 rounded-lg">
              <h3 className="text-sm font-semibold mb-2 flex items-center">
                <Network className="w-4 h-4 mr-2" />
                지식 그래프
              </h3>
              <div className="text-xs space-y-1">
                <div>엔티티: {knowledgeGraphData.nodes_count}개</div>
                <div>관계: {knowledgeGraphData.edges_count}개</div>
                <div>커뮤니티: {knowledgeGraphData.communities}개</div>
              </div>
              
              {knowledgeGraphData.nodes && knowledgeGraphData.nodes.length > 0 && (
                <div className="mt-2">
                  <div className="text-xs font-medium mb-1">주요 엔티티:</div>
                  <div className="space-y-1 max-h-20 overflow-y-auto">
                    {knowledgeGraphData.nodes.slice(0, 5).map((node, index) => (
                      <div key={index} className="text-xs bg-white p-1 rounded">
                        <span className="font-medium">{node.name}</span>
                        <span className="text-gray-500 ml-1">({node.type})</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Agent Configuration */}
          <div className="mt-4 p-4 bg-gray-50 rounded-lg">
            <h3 className="text-sm font-semibold mb-2 flex items-center">
              <Database className="w-4 h-4 mr-2" />
              Agent 설정
            </h3>
            <div className="text-sm space-y-1">
              <div><strong>Type:</strong> {agentConfigs[currentConv.agent].name}</div>
              <div><strong>Model:</strong> {currentModel}</div>
              <div><strong>Files:</strong> {uploadedFiles.length}</div>
              <div><strong>GraphRAG:</strong> {useGraphRAG ? '활성화' : '비활성화'}</div>
              <div  // Enhanced API call with checkpoint and GraphRAG support
  const callChatAPI = async (message, conversationId, agentType, model, checkpointId = null) => {
    setIsLoading(true);
    
    try {
      const response = await fetch(`${apiConfig.baseUrl}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: message,
          conversation_id: conversationId,
          agent_type: agentType,
          model: model,
          stream: false,
          use_graphrag: useGraphRAG,
          checkpoint_id: checkpointId
        }),
        signal: AbortSignal.timeout(apiConfig.timeout)
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `API error: ${response.status}`);
      }

      const data = await response.json();
      return {
        response: data.response,
        agentState: data.agent_state,
        checkpointId: data.checkpoint_id,
        sources: data.sources || [],
        metadata: data.metadata || {}
      };
    } catch (error) {
      console.error('Chat API call failed:', error);
      
      if (error.name === 'AbortError' || error.message.includes('timeout')) {
        throw new Error('응답 시간이 초과되었습니다. 잠시 후 다시 시도해주세요.');
      } else if (error.message.includes('fetch')) {
        throw new Error('API 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요.');
      } else {
        throw new Error(`API 호출 실패: ${error.message}`);
      }
    } finally {
      setIsLoading(false);
    }
  };

  // Fetch knowledge graph data
  const fetchKnowledgeGraphData = async (conversationId) => {
    try {
      const response = await fetch(`${apiConfig.baseUrl}/knowledge-graph/${conversationId}`);
      if (response.ok) {
        const data = await response.json();
        setKnowledgeGraphData(data);
      }
    } catch (error) {
      console.error('Failed to fetch knowledge graph:', error);
    }
  };

  // Rollback to checkpoint
  const rollbackToCheckpoint = async (checkpointId) => {
    try {
      const response = await fetch(`${apiConfig.baseUrl}/rollback/${activeConversation}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ checkpoint_id: checkpointId }),
      });

      if (response.ok) {
        // Reload conversation
        window.location.reload(); // Simple approach - in production, you'd reload conversation data
      }
    } catch (error) {
      console.error('Rollback failed:', error);
      alert('체크포인트 롤백에 실패했습니다.');
    }
  };import React, { useState, useRef, useEffect } from 'react';
import { Send, Upload, Settings, Bot, Plane, Film, Stethoscope, MessageSquare, FileText, Wifi, WifiOff, Loader, GitBranch, Database, RotateCcw, Network, Info } from 'lucide-react';

// Enhanced Mock LangGraph node definitions with subgraphs
const agentGraphs = {
  doctor: {
    main_nodes: ['초기진단', '증상분석', '전문과목판단', '최종진단'],
    main_edges: [
      { from: '초기진단', to: '증상분석' },
      { from: '증상분석', to: '전문과목판단' },
      { from: '전문과목판단', to: '최종진단' }
    ],
    subgraphs: {
      '검사서브그래프': {
        nodes: ['검사선택', '검사해석', '추가검사판단'],
        edges: [
          { from: '검사선택', to: '검사해석' },
          { from: '검사해석', to: '추가검사판단' }
        ],
        trigger_from: '증상분석'
      },
      '치료서브그래프': {
        nodes: ['치료계획', '약물선택', '생활지도'],
        edges: [
          { from: '치료계획', to: '약물선택' },
          { from: '약물선택', to: '생활지도' }
        ],
        trigger_from: '최종진단'
      }
    }
  },
  travel: {
    main_nodes: ['여행상담', '목적지분석', '일정수립', '예약진행'],
    main_edges: [
      { from: '여행상담', to: '목적지분석' },
      { from: '목적지분석', to: '일정수립' },
      { from: '일정수립', to: '예약진행' }
    ],
    subgraphs: {
      '교통서브그래프': {
        nodes: ['항공검색', '항공비교', '항공예약'],
        edges: [
          { from: '항공검색', to: '항공비교' },
          { from: '항공비교', to: '항공예약' }
        ],
        trigger_from: '일정수립'
      },
      '숙박서브그래프': {
        nodes: ['호텔검색', '호텔비교', '호텔예약'],
        edges: [
          { from: '호텔검색', to: '호텔비교' },
          { from: '호텔비교', to: '호텔예약' }
        ],
        trigger_from: '일정수립'
      }
    }
  },
  movie: {
    main_nodes: ['영화상담', '영화추천', '예매진행', '결제완료'],
    main_edges: [
      { from: '영화상담', to: '영화추천' },
      { from: '영화추천', to: '예매진행' },
      { from: '예매진행', to: '결제완료' }
    ],
    subgraphs: {
      '추천서브그래프': {
        nodes: ['장르분석', '평점확인', '리뷰분석'],
        edges: [
          { from: '장르분석', to: '평점확인' },
          { from: '평점확인', to: '리뷰분석' }
        ],
        trigger_from: '영화상담'
      },
      '예매서브그래프': {
        nodes: ['상영시간확인', '좌석선택', '할인적용'],
        edges: [
          { from: '상영시간확인', to: '좌석선택' },
          { from: '좌석선택', to: '할인적용' }
        ],
        trigger_from: '예매진행'
      }
    }
  }
};

const agentConfigs = {
  doctor: {
    name: 'AI Doctor',
    icon: Stethoscope,
    color: 'bg-red-500',
    description: '의료 상담 및 진료 지원'
  },
  travel: {
    name: 'Travel Agent',
    icon: Plane,
    color: 'bg-blue-500',
    description: '여행 계획 및 예약 서비스'
  },
  movie: {
    name: 'Movie Booking',
    icon: Film,
    color: 'bg-purple-500',
    description: '영화 티켓 예약 서비스'
  }
};

// Enhanced GraphVisualizer component with subgraphs
const GraphVisualizer = ({ agentType, currentState, showSubgraphs = false, showGraphInfo, setShowGraphInfo }) => {
  const graph = agentGraphs[agentType];
  const svgRef = useRef(null);

  useEffect(() => {
    if (!graph || !svgRef.current) return;

    const svg = svgRef.current;
    const width = 400;
    const height = 500;
    
    // Clear previous content
    svg.innerHTML = '';
    
    // Main nodes
    const mainNodes = graph.main_nodes;
    const mainNodePositions = mainNodes.map((node, index) => ({
      name: node,
      x: 200 + 100 * Math.cos((index / mainNodes.length) * 2 * Math.PI - Math.PI / 2),
      y: 150 + 80 * Math.sin((index / mainNodes.length) * 2 * Math.PI - Math.PI / 2),
      isMain: true
    }));

    // Subgraph nodes (if enabled)
    let subgraphPositions = [];
    if (showSubgraphs && graph.subgraphs) {
      let subY = 350;
      Object.entries(graph.subgraphs).forEach(([subName, subGraph]) => {
        subGraph.nodes.forEach((node, index) => {
          subgraphPositions.push({
            name: `${subName}_${node}`,
            displayName: node,
            x: 100 + (index * 100),
            y: subY,
            isMain: false,
            subgraph: subName
          });
        });
        subY += 100;
      });
    }

    const allPositions = [...mainNodePositions, ...subgraphPositions];

    // Draw edges for main graph
    graph.main_edges.forEach(edge => {
      const fromPos = allPositions.find(n => n.name === edge.from);
      const toPos = allPositions.find(n => n.name === edge.to);
      
      if (fromPos && toPos) {
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', fromPos.x);
        line.setAttribute('y1', fromPos.y);
        line.setAttribute('x2', toPos.x);
        line.setAttribute('y2', toPos.y);
        line.setAttribute('stroke', '#3b82f6');
        line.setAttribute('stroke-width', '2');
        svg.appendChild(line);
      }
    });

    // Draw edges for subgraphs
    if (showSubgraphs && graph.subgraphs) {
      Object.entries(graph.subgraphs).forEach(([subName, subGraph]) => {
        subGraph.edges.forEach(edge => {
          const fromPos = allPositions.find(n => n.name === `${subName}_${edge.from}`);
          const toPos = allPositions.find(n => n.name === `${subName}_${edge.to}`);
          
          if (fromPos && toPos) {
            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', fromPos.x);
            line.setAttribute('y1', fromPos.y);
            line.setAttribute('x2', toPos.x);
            line.setAttribute('y2', toPos.y);
            line.setAttribute('stroke', '#f59e0b');
            line.setAttribute('stroke-width', '1.5');
            line.setAttribute('stroke-dasharray', '5,5');
            svg.appendChild(line);
          }
        });
      });
    }

    // Draw nodes
    allPositions.forEach(pos => {
      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circle.setAttribute('cx', pos.x);
      circle.setAttribute('cy', pos.y);
      circle.setAttribute('r', pos.isMain ? '25' : '20');
      
      const isActive = pos.name === currentState || pos.name.includes(currentState);
      const fillColor = isActive ? '#ef4444' : (pos.isMain ? '#3b82f6' : '#f59e0b');
      const strokeColor = isActive ? '#dc2626' : (pos.isMain ? '#1d4ed8' : '#d97706');
      
      circle.setAttribute('fill', fillColor);
      circle.setAttribute('stroke', strokeColor);
      circle.setAttribute('stroke-width', '2');
      svg.appendChild(circle);

      const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      text.setAttribute('x', pos.x);
      text.setAttribute('y', pos.y + (pos.isMain ? 35 : 30));
      text.setAttribute('text-anchor', 'middle');
      text.setAttribute('font-size', pos.isMain ? '10' : '8');
      text.setAttribute('fill', isActive ? '#ef4444' : '#374151');
      text.textContent = pos.displayName || pos.name;
      svg.appendChild(text);
    });

    // Add legend
    if (showSubgraphs) {
      const legend = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      
      // Main graph legend
      const mainCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      mainCircle.setAttribute('cx', '30');
      mainCircle.setAttribute('cy', '30');
      mainCircle.setAttribute('r', '8');
      mainCircle.setAttribute('fill', '#3b82f6');
      legend.appendChild(mainCircle);
      
      const mainText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      mainText.setAttribute('x', '45');
      mainText.setAttribute('y', '35');
      mainText.setAttribute('font-size', '12');
      mainText.setAttribute('fill', '#374151');
      mainText.textContent = '메인';
      legend.appendChild(mainText);
      
      // Subgraph legend
      const subCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      subCircle.setAttribute('cx', '100');
      subCircle.setAttribute('cy', '30');
      subCircle.setAttribute('r', '8');
      subCircle.setAttribute('fill', '#f59e0b');
      legend.appendChild(subCircle);
      
      const subText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      subText.setAttribute('x', '115');
      subText.setAttribute('y', '35');
      subText.setAttribute('font-size', '12');
      subText.setAttribute('fill', '#374151');
      subText.textContent = '서브';
      legend.appendChild(subText);
      
      svg.appendChild(legend);
    }
  }, [agentType, currentState, showSubgraphs]);

  return (
    <div className="bg-white p-4 rounded-lg border">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-semibold">Agent Graph</h3>
        <button
          onClick={() => setShowGraphInfo(!showGraphInfo)}
          className="p-1 hover:bg-gray-100 rounded"
        >
          <Info className="w-4 h-4" />
        </button>
      </div>
      <svg ref={svgRef} width="400" height="500" className="w-full h-auto" />
    </div>
  );
};

export default function LangGraphMultiAgentApp() {
  const [conversations, setConversations] = useState([
    { id: 1, title: '의료 상담', agent: 'doctor', messages: [] },
    { id: 2, title: '여행 계획', agent: 'travel', messages: [] },
    { id: 3, title: '영화 예약', agent: 'movie', messages: [] }
  ]);
  
  const [activeConversation, setActiveConversation] = useState(1);
  const [message, setMessage] = useState('');
  const [currentModel, setCurrentModel] = useState('qwen2.5:8b');
  const [availableModels, setAvailableModels] = useState(['qwen2.5:8b', 'llama3:8b', 'mistral:7b']);
  const [showSettings, setShowSettings] = useState(false);
  const [currentState, setCurrentState] = useState('초기진단');
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [useGraphRAG, setUseGraphRAG] = useState(false);
  const [checkpointHistory, setCheckpointHistory] = useState([]);
  const [showGraphInfo, setShowGraphInfo] = useState(false);
  const [knowledgeGraphData, setKnowledgeGraphData] = useState(null);
  const fileInputRef = useRef(null);

  const currentConv = conversations.find(c => c.id === activeConversation);

  // Mock function to simulate agent state progression
  const getNextState = (agent, currentState) => {
    const graph = agentGraphs[agent];
    const edges = graph.edges.filter(e => e.from === currentState);
    return edges.length > 0 ? edges[0].to : currentState;
  };

  // API configuration
  const [apiConfig, setApiConfig] = useState({
    baseUrl: 'http://localhost:8000',
    timeout: 30000
  });
  const [isLoading, setIsLoading] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('disconnected'); // connected, connecting, disconnected, error

  // Check API connection
  const checkApiConnection = async () => {
    try {
      setConnectionStatus('connecting');
      const response = await fetch(`${apiConfig.baseUrl}/health`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        signal: AbortSignal.timeout(5000)
      });
      
      if (response.ok) {
        const data = await response.json();
        setConnectionStatus(data.status === 'healthy' ? 'connected' : 'error');
        return data.status === 'healthy';
      } else {
        setConnectionStatus('error');
        return false;
      }
    } catch (error) {
      console.error('API connection error:', error);
      setConnectionStatus('error');
      return false;
    }
  };

  // Get available models from API
  const getAvailableModels = async () => {
    try {
      const response = await fetch(`${apiConfig.baseUrl}/models`);
      const data = await response.json();
      return data.models || [];
    } catch (error) {
      console.error('Failed to get models:', error);
      return [];
    }
  };

  // Real API call to FastAPI server
  const callChatAPI = async (message, conversationId, agentType, model) => {
    setIsLoading(true);
    
    try {
      const response = await fetch(`${apiConfig.baseUrl}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: message,
          conversation_id: conversationId,
          agent_type: agentType,
          model: model,
          stream: false
        }),
        signal: AbortSignal.timeout(apiConfig.timeout)
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `API error: ${response.status}`);
      }

      const data = await response.json();
      return {
        response: data.response,
        agentState: data.agent_state,
        metadata: data.metadata
      };
    } catch (error) {
      console.error('Chat API call failed:', error);
      
      if (error.name === 'AbortError' || error.message.includes('timeout')) {
        throw new Error('응답 시간이 초과되었습니다. 잠시 후 다시 시도해주세요.');
      } else if (error.message.includes('fetch')) {
        throw new Error('API 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요.');
      } else {
        throw new Error(`API 호출 실패: ${error.message}`);
      }
    } finally {
      setIsLoading(false);
    }
  };

  // Upload file to API
  const uploadFileToAPI = async (file, conversationId) => {
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await fetch(`${apiConfig.baseUrl}/upload?conversation_id=${conversationId}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'File upload failed');
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('File upload error:', error);
      throw error;
    }
  };

  // Initialize connection check
  useEffect(() => {
    checkApiConnection();
  }, [apiConfig.baseUrl]);

  const handleSendMessage = async () => {
    if (!message.trim() || isLoading) return;

    // Check connection before sending
    if (connectionStatus !== 'connected') {
      const connected = await checkOllamaConnection();
      if (!connected) {
        alert('Ollama 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요.');
        return;
      }
    }

    const userMessage = { role: 'user', content: message };
    
    // Add user message
    setConversations(prev => prev.map(conv => 
      conv.id === activeConversation 
        ? { ...conv, messages: [...conv.messages, userMessage] }
        : conv
    ));

    const userInput = message;
    setMessage('');

    // Get context from uploaded files
    const context = uploadedFiles.map(f => f.content).join('\n\n');

    try {
      // Call real Ollama API
      const response = await callOllama(userInput, currentModel, context);
      
      const assistantMessage = { role: 'assistant', content: response };
      
      // Add assistant response
      setConversations(prev => prev.map(conv => 
        conv.id === activeConversation 
          ? { ...conv, messages: [...conv.messages, assistantMessage] }
          : conv
      ));

      // Update agent state
      if (currentConv) {
        const nextState = getNextState(currentConv.agent, currentState);
        setCurrentState(nextState);
      }
    } catch (error) {
      console.error('Error calling Ollama:', error);
      const errorMessage = { role: 'assistant', content: `오류: ${error.message}` };
      setConversations(prev => prev.map(conv => 
        conv.id === activeConversation 
          ? { ...conv, messages: [...conv.messages, errorMessage] }
          : conv
      ));
    }
  };

  // Load available models on component mount
  useEffect(() => {
    const loadModels = async () => {
      if (connectionStatus === 'connected') {
        const models = await getAvailableModels();
        if (models.length > 0) {
          setAvailableModels(models.map(m => m.name));
        }
      }
    };
    loadModels();
  }, [connectionStatus]);

  const handleFileUpload = (event) => {
    const files = Array.from(event.target.files);
    files.forEach(file => {
      if (file.type === 'text/plain') {
        const reader = new FileReader();
        reader.onload = (e) => {
          const newFile = {
            id: Date.now() + Math.random(),
            name: file.name,
            content: e.target.result
          };
          setUploadedFiles(prev => [...prev, newFile]);
        };
        reader.readAsText(file);
      }
    });
  };

  const createNewConversation = (agentType) => {
    const newId = Math.max(...conversations.map(c => c.id)) + 1;
    const newConv = {
      id: newId,
      title: `새 ${agentConfigs[agentType].name} 상담`,
      agent: agentType,
      messages: []
    };
    setConversations(prev => [...prev, newConv]);
    setActiveConversation(newId);
    
    // Set initial state based on agent type
    const initialStates = {
      'doctor': '초기진단',
      'travel': '여행상담', 
      'movie': '영화상담'
    };
    setCurrentState(initialStates[agentType] || '초기진단');
    
    // Reset context for new conversation
    setCheckpointHistory([]);
    setKnowledgeGraphData(null);
    setUploadedFiles([]);
  };

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <div className="w-80 bg-white border-r flex flex-col">
        <div className="p-4 border-b">
          <div className="flex items-center justify-between mb-4">
            <h1 className="text-xl font-bold">AI Agents</h1>
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="p-2 hover:bg-gray-100 rounded-lg"
            >
              <Settings className="w-5 h-5" />
            </button>
          </div>
          
          {showSettings && (
            <div className="mb-4 p-3 bg-gray-50 rounded-lg space-y-3">
              <div>
                <label className="block text-sm font-medium mb-2">API 서버 URL</label>
                <input
                  type="text"
                  value={apiConfig.baseUrl}
                  onChange={(e) => setApiConfig(prev => ({...prev, baseUrl: e.target.value}))}
                  placeholder="http://localhost:8000"
                  className="w-full p-2 border rounded-md text-sm"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-2">모델 선택</label>
                <select
                  value={currentModel}
                  onChange={(e) => setCurrentModel(e.target.value)}
                  className="w-full p-2 border rounded-md text-sm"
                >
                  {availableModels.map(model => (
                    <option key={model} value={model}>{model}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">타임아웃 (ms)</label>
                <input
                  type="number"
                  value={apiConfig.timeout}
                  onChange={(e) => setApiConfig(prev => ({...prev, timeout: parseInt(e.target.value)}))}
                  min="5000"
                  max="300000"
                  step="5000"
                  className="w-full p-2 border rounded-md text-sm"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">GraphRAG 사용</label>
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={useGraphRAG}
                    onChange={(e) => setUseGraphRAG(e.target.checked)}
                    className="rounded border-gray-300"
                  />
                  <span className="text-sm">지식 그래프 기반 검색 사용</span>
                </div>
              </div>

              <button
                onClick={checkApiConnection}
                disabled={connectionStatus === 'connecting'}
                className="w-full p-2 bg-blue-500 text-white rounded-md text-sm hover:bg-blue-600 disabled:bg-gray-400 flex items-center justify-center space-x-2"
              >
                {connectionStatus === 'connecting' ? (
                  <><Loader className="w-4 h-4 animate-spin" /><span>연결 확인 중...</span></>
                ) : (
                  <><Wifi className="w-4 h-4" /><span>연결 테스트</span></>
                )}
              </button>

              <div className="flex items-center space-x-2 text-sm">
                {connectionStatus === 'connected' && (
                  <><Wifi className="w-4 h-4 text-green-500" /><span className="text-green-600">API 서버 연결됨</span></>
                )}
                {connectionStatus === 'error' && (
                  <><WifiOff className="w-4 h-4 text-red-500" /><span className="text-red-600">API 서버 연결 실패</span></>
                )}
                {connectionStatus === 'connecting' && (
                  <><Loader className="w-4 h-4 text-blue-500 animate-spin" /><span className="text-blue-600">연결 중...</span></>
                )}
                {connectionStatus === 'disconnected' && (
                  <><WifiOff className="w-4 h-4 text-gray-500" /><span className="text-gray-600">API 서버 연결 안됨</span></>
                )}
              </div>
            </div>
          )}

          <div className="space-y-2">
            {Object.entries(agentConfigs).map(([key, config]) => {
              const Icon = config.icon;
              return (
                <button
                  key={key}
                  onClick={() => createNewConversation(key)}
                  className="w-full flex items-center space-x-3 p-3 hover:bg-gray-50 rounded-lg text-left"
                >
                  <div className={`p-2 rounded-lg ${config.color} text-white`}>
                    <Icon className="w-4 h-4" />
                  </div>
                  <div>
                    <div className="font-medium text-sm">{config.name}</div>
                    <div className="text-xs text-gray-500">{config.description}</div>
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-4">
          <h2 className="font-semibold text-sm text-gray-700 mb-3">대화 목록</h2>
          <div className="space-y-2">
            {conversations.map(conv => {
              const config = agentConfigs[conv.agent];
              const Icon = config.icon;
              return (
                <button
                  key={conv.id}
                  onClick={() => {
                    setActiveConversation(conv.id);
                    // Set initial state based on agent type
                    const initialStates = {
                      'doctor': '초기진단',
                      'travel': '여행상담', 
                      'movie': '영화상담'
                    };
                    setCurrentState(initialStates[conv.agent] || '초기진단');
                  }}
                  className={`w-full flex items-center space-x-3 p-3 rounded-lg text-left transition-colors ${
                    activeConversation === conv.id ? 'bg-blue-50 border border-blue-200' : 'hover:bg-gray-50'
                  }`}
                >
                  <div className={`p-1.5 rounded ${config.color} text-white`}>
                    <Icon className="w-3 h-3" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="font-medium text-sm truncate">{conv.title}</div>
                    <div className="text-xs text-gray-500">{conv.messages.length} messages</div>
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {/* File Upload Section */}
        <div className="p-4 border-t">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-medium">RAG 문서</h3>
            <button
              onClick={() => fileInputRef.current?.click()}
              className="p-1.5 hover:bg-gray-100 rounded-lg"
            >
              <Upload className="w-4 h-4" />
            </button>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept=".txt"
            multiple
            onChange={handleFileUpload}
            className="hidden"
          />
          <div className="space-y-1">
            {uploadedFiles.map(file => (
              <div key={file.id} className="flex items-center justify-between text-xs text-gray-600 bg-gray-100 p-2 rounded">
                <div className="flex items-center space-x-2">
                  <FileText className="w-3 h-3" />
                  <span className="truncate">{file.name}</span>
                </div>
                {file.uploaded && (
                  <span className="text-green-600 text-xs">✓ 업로드됨</span>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {currentConv ? (
          <>
            {/* Header */}
            <div className="p-4 border-b bg-white flex items-center space-x-3">
              <div className={`p-2 rounded-lg ${agentConfigs[currentConv.agent].color} text-white`}>
                {React.createElement(agentConfigs[currentConv.agent].icon, { className: "w-5 h-5" })}
              </div>
              <div>
                <h2 className="font-semibold">{currentConv.title}</h2>
                <p className="text-sm text-gray-500">
                  {agentConfigs[currentConv.agent].description} • {currentModel}
                </p>
              </div>
            </div>

            {/* Enhanced Messages with Sources */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {currentConv.messages.map((msg, index) => (
                <div
                  key={index}
                  className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                      msg.role === 'user'
                        ? 'bg-blue-500 text-white'
                        : 'bg-white border border-gray-200'
                    }`}
                  >
                    <div>{msg.content}</div>
                    
                    {/* Sources display */}
                    {msg.sources && msg.sources.length > 0 && (
                      <div className="mt-2 text-xs">
                        <div className="font-semibold text-gray-600 mb-1">📚 출처:</div>
                        {msg.sources.map((source, sourceIndex) => (
                          <div key={sourceIndex} className="mb-1 p-2 bg-gray-50 rounded text-gray-700">
                            {source.source_type === 'vectorstore' ? (
                              <div>
                                <div className="font-medium">문서 {source.doc_id}</div>
                                <div className="text-xs">유사도: {(source.similarity_score * 100).toFixed(1)}%</div>
                                <div className="text-xs mt-1">{source.content}</div>
                              </div>
                            ) : (
                              <div>
                                <div className="font-medium">{source.entity} ({source.type})</div>
                                <div className="text-xs">유사도: {(source.similarity * 100).toFixed(1)}%</div>
                                <div className="text-xs">커뮤니티: {source.community}</div>
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                    
                    {msg.timestamp && (
                      <div className="text-xs opacity-70 mt-1">
                        {new Date(msg.timestamp).toLocaleTimeString()}
                      </div>
                    )}
                  </div>
                </div>
              ))}
              
              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-white border border-gray-200 px-4 py-2 rounded-lg flex items-center space-x-2">
                    <Loader className="w-4 h-4 animate-spin text-blue-500" />
                    <span className="text-gray-600">
                      {useGraphRAG ? 'GraphRAG로 응답을 생성하고 있습니다...' : '응답을 생성하고 있습니다...'}
                    </span>
                  </div>
                </div>
              )}
            </div>

            {/* Input */}
            <div className="p-4 border-t bg-white">
              <div className="flex space-x-2">
                <input
                  type="text"
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && !isLoading && handleSendMessage()}
                  placeholder={isLoading ? "응답을 기다리는 중..." : "메시지를 입력하세요..."}
                  disabled={isLoading || connectionStatus !== 'connected'}
                  className="flex-1 p-3 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100 disabled:text-gray-500"
                />
                <button
                  onClick={handleSendMessage}
                  disabled={isLoading || connectionStatus !== 'connected' || !message.trim()}
                  className="p-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-400 disabled:cursor-not-allowed"
                >
                  {isLoading ? <Loader className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5" />}
                </button>
              </div>
              
              {connectionStatus !== 'connected' && (
                <div className="mt-2 text-sm text-amber-600 bg-amber-50 p-2 rounded-md">
                  ⚠️ API 서버에 연결되지 않았습니다. 설정에서 연결을 확인해주세요.
                </div>
              )}
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center">
              <Bot className="w-16 h-16 mx-auto text-gray-400 mb-4" />
              <h2 className="text-xl font-semibold text-gray-600 mb-2">대화를 선택해주세요</h2>
              <p className="text-gray-500">왼쪽에서 대화를 선택하거나 새 대화를 시작해보세요.</p>
            </div>
          </div>
        )}
      </div>

      {/* Right Panel - Graph Visualization */}
      {currentConv && (
        <div className="w-80 bg-white border-l p-4">
          <GraphVisualizer 
            agentType={currentConv.agent} 
            currentState={currentState}
          />
          
          <div className="mt-4 p-4 bg-gray-50 rounded-lg">
            <h3 className="text-sm font-semibold mb-2">현재 상태</h3>
            <div className="text-sm">
              <span className="font-medium text-blue-600">{currentState}</span>
            </div>
          </div>

          <div className="mt-4 p-4 bg-gray-50 rounded-lg">
            <h3 className="text-sm font-semibold mb-2">Agent 정보</h3>
            <div className="text-sm space-y-1">
              <div><strong>Type:</strong> {agentConfigs[currentConv.agent].name}</div>
              <div><strong>Model:</strong> {currentModel}</div>
              <div><strong>Files:</strong> {uploadedFiles.length}</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}