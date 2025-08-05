import { useState, useCallback } from "react";
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  addEdge,
  useNodesState,
  useEdgesState,
  ReactFlowProvider,
  Handle,
  Position,
} from "reactflow";
import "reactflow/dist/style.css";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tooltip, TooltipTrigger, TooltipContent, TooltipProvider } from "@/components/ui/tooltip";

const translationMap = {
  ko: {
    start_engine: "시동을 걸어주세요.",
    seatbelt: "안전벨트를 착용하세요.",
    gear_shift: "기어를 P에서 D로 변경하세요.",
    condition_step_1: "조건 분기 예시입니다.",
    error_step_2: "예외 처리 예시입니다.",
  },
  en: {
    start_engine: "Please start the engine.",
    seatbelt: "Fasten your seatbelt.",
    gear_shift: "Shift the gear from P to D.",
    condition_step_1: "This is a conditional step.",
    error_step_2: "This is an error handler step.",
  },
};

let id = 0;
const getId = () => `step-${id++}`;

function CustomNode({ data }) {
  const lang = "ko";
  const translation = translationMap[lang]?.[data.promptKey] || "";
  const allTranslations = Object.entries(translationMap)
    .map(([lang, dict]) => `${lang}: ${dict[data.promptKey] || "(없음)"}`)
    .join("\n");

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <div
            className={`border rounded p-2 shadow w-60 cursor-pointer ${data.type === "condition" ? "bg-yellow-100" : data.type === "error" ? "bg-red-100" : "bg-white"}`}
          >
            <Handle type="target" position={Position.Top} />
            <div className="text-sm font-bold mb-1">{data.label} ({data.type || "normal"})</div>
            <Input
              className="text-xs mb-1"
              value={data.promptKey || ""}
              onChange={(e) => data.onPromptChange(data.id, e.target.value)}
              placeholder="prompt_key"
            />
            <div className="text-xs italic text-gray-500 min-h-[1.25rem]">
              🌐 {translation || "(번역 없음)"}
            </div>
            <Handle type="source" position={Position.Bottom} />
          </div>
        </TooltipTrigger>
        <TooltipContent side="right">
          <pre className="text-xs whitespace-pre-wrap">{allTranslations}</pre>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

const nodeTypes = {
  custom: CustomNode,
};

function FlowEditor() {
  const initialNodes = [
    {
      id: getId(),
      type: "custom",
      position: { x: 100, y: 100 },
      data: { label: "start_engine", promptKey: "start_engine", type: "normal", onPromptChange },
    },
  ];

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  const onConnect = useCallback(
    (params) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

  const addNode = (type = "normal") => {
    const newId = getId();
    const newNode = {
      id: newId,
      type: "custom",
      position: {
        x: Math.random() * 400 + 100,
        y: Math.random() * 400 + 100,
      },
      data: {
        id: newId,
        label: `${type}_step_${id}`,
        promptKey: `${type}_step_${id}`,
        type,
        onPromptChange,
      },
    };
    setNodes((nds) => [...nds, newNode]);
  };

  const onPromptChange = (id, newPrompt) => {
    setNodes((nds) =>
      nds.map((n) =>
        n.id === id ? { ...n, data: { ...n.data, promptKey: newPrompt } } : n
      )
    );
  };

  const exportFSM = () => {
    const nodeMap = Object.fromEntries(nodes.map((n) => [n.id, n.data.label]));
    const code = `from langgraph.graph import StateGraph, END\n\nclass AgentState(dict):\n    pass\n\ndef make_node(prompt):\n    def node_fn(state):\n        return {**state, \"prompt\": prompt, \"step_index\": state.get(\"step_index\", 0) + 1}\n    return node_fn\n\ngraph = StateGraph(AgentState)\n`;

    const nodeDefs = nodes
      .map(
        (n) => `graph.add_node(\"${n.data.label}\", make_node(\"${n.data.promptKey || n.data.label}\"))`
      )
      .join("\n");

    const edgeDefs = edges
      .map(
        (e) =>
          `graph.add_edge(\"${nodeMap[e.source]}\", \"${nodeMap[e.target]}\")`
      )
      .join("\n");

    const entry = nodes.length > 0 ? `graph.set_entry_point(\"${nodes[0].data.label}\")` : "";
    const compile = "\nfsm_app = graph.compile()";
    alert(`${code}\n${nodeDefs}\n${edgeDefs}\n${entry}${compile}`);
  };

  return (
    <div className="w-full h-[600px]">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        fitView
        nodeTypes={nodeTypes}
      >
        <Background />
        <Controls />
        <MiniMap />
      </ReactFlow>
      <div className="flex gap-2 mt-4 p-4">
        <Button onClick={() => addNode("normal")}>노드 추가</Button>
        <Button onClick={() => addNode("condition")}>조건 분기 노드 추가</Button>
        <Button onClick={() => addNode("error")}>예외 노드 추가</Button>
        <Button onClick={exportFSM}>LangGraph FSM 코드 생성</Button>
      </div>
    </div>
  );
}

export default function ScenarioBuilder() {
  return (
    <div className="p-4 max-w-full">
      <h1 className="text-xl font-bold mb-4">🧠 시나리오 그래프 빌더</h1>
      <ReactFlowProvider>
        <FlowEditor />
      </ReactFlowProvider>
    </div>
  );
}
