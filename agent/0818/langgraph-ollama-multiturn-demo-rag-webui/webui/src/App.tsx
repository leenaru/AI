import React, { useMemo, useRef, useState } from 'react'
import { ragChat } from './api'

type Msg = { role: 'user'|'assistant', text: string }

function uuid() {
  return Math.random().toString(36).slice(2) + Date.now().toString(36)
}

export default function App() {
  const [mode, setMode] = useState<'basic'|'hq'|'graph'>('basic')
  const [threadId, setThreadId] = useState<string>(uuid())
  const [input, setInput] = useState('시드니에서 아이와 가볼만한 곳 알려줘')
  const [msgs, setMsgs] = useState<Msg[]>([])
  const [loading, setLoading] = useState(false)
  const boxRef = useRef<HTMLDivElement>(null)

  const send = async () => {
    if (!input.trim()) return
    const u: Msg = { role:'user', text: input }
    setMsgs(m => [...m, u])
    setInput('')
    setLoading(true)
    try {
      const { reply } = await ragChat(mode, threadId, u.text)
      const a: Msg = { role:'assistant', text: reply }
      setMsgs(m => [...m, a])
      setTimeout(()=>{ boxRef.current?.scrollTo({ top: 1e6, behavior:'smooth' }) }, 50)
    } catch (e:any) {
      setMsgs(m => [...m, { role:'assistant', text: '에러: ' + e.message }])
    } finally {
      setLoading(false)
    }
  }

  const newChat = () => {
    setThreadId(uuid())
    setMsgs([])
  }

  return (
    <div className="min-h-screen flex flex-col items-center p-4">
      <div className="w-full max-w-3xl space-y-3">
        <header className="flex items-center justify-between">
          <h1 className="text-xl font-bold">RAG Chat Demo</h1>
          <div className="flex items-center gap-2">
            <select className="border rounded px-2 py-1" value={mode} onChange={e=>setMode(e.target.value as any)}>
              <option value="basic">Basic RAG</option>
              <option value="hq">HQ-RAG</option>
              <option value="graph">GraphRAG</option>
            </select>
            <button onClick={newChat} className="px-3 py-1 rounded bg-gray-800 text-white">New Chat</button>
          </div>
        </header>

        <div className="text-sm text-gray-500">thread_id: <code>{threadId}</code></div>

        <div ref={boxRef} className="h-[60vh] overflow-auto bg-white rounded-xl border p-3 space-y-3">
          {msgs.map((m,i)=>(
            <div key={i} className={"flex " + (m.role==='user'?'justify-end':'justify-start')}>
              <div className={(m.role==='user'?'bg-blue-600 text-white':'bg-gray-100') + " rounded-2xl px-3 py-2 max-w-[80%] whitespace-pre-wrap"}>
                {m.text}
              </div>
            </div>
          ))}
          {loading && <div className="text-gray-400 text-sm">생성 중...</div>}
        </div>

        <div className="flex gap-2">
          <input className="flex-1 border rounded px-3 py-2" value={input} onChange={e=>setInput(e.target.value)} placeholder="메시지를 입력하세요" onKeyDown={e=>{ if(e.key==='Enter'&&!e.shiftKey){ e.preventDefault(); send(); }}} />
          <button onClick={send} disabled={loading} className="px-4 py-2 rounded bg-black text-white disabled:opacity-50">Send</button>
        </div>
      </div>
    </div>
  )
}
