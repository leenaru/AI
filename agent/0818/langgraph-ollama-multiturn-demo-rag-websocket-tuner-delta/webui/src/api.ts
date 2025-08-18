export const API_BASE = (window as any).__API__ || '';

export async function ragChat(mode: 'basic'|'hq'|'graph', threadId: string, message: string) {
  const res = await fetch(`${API_BASE}/rag/chat/${mode}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ thread_id: threadId, message })
  });
  if (!res.ok) throw new Error(await res.text());
  return await res.json() as { reply: string };
}

export function sseRag(mode: 'basic'|'hybrid', threadId: string, q: string) {
  const url = `${API_BASE}/sse/rag?mode=${encodeURIComponent(mode)}&thread_id=${encodeURIComponent(threadId)}&q=${encodeURIComponent(q)}`;
  return new EventSource(url);
}

export async function uploadDoc(file: File) {
  const fd = new FormData();
  fd.append('file', file);
  const res = await fetch(`${API_BASE}/admin/upload`, { method: 'POST', body: fd });
  if (!res.ok) throw new Error(await res.text());
  return await res.json() as { saved: string };
}

export async function reindexAll() {
  const res = await fetch(`${API_BASE}/admin/reindex`, { method: 'POST' });
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

export function wsRag(mode: 'basic'|'hybrid', threadId: string, model?: string) {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  const url = `${proto}://${location.host}/ws/rag?mode=${encodeURIComponent(mode)}${model?`&model=${encodeURIComponent(model)}`:''}`
  return new WebSocket(url)
}

export async function tuneHybrid(config: { weight_vec: number; weight_bm25: number; fuse: 'max'|'sum'|'rrf'; rrf_k?: number }) {
  const res = await fetch(`${API_BASE}/admin/hybrid/tune`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config)
  })
  if (!res.ok) throw new Error(await res.text())
  return await res.json()
}

export async function deltaAddText(text: string, source?: string) {
  const res = await fetch(`${API_BASE}/admin/delta/add_text`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, source })
  })
  if (!res.ok) throw new Error(await res.text())
  return await res.json()
}

export async function deltaDelete(source: string) {
  const res = await fetch(`${API_BASE}/admin/delta/delete`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source })
  })
  if (!res.ok) throw new Error(await res.text())
  return await res.json()
}

export async function compactIndexes() {
  const res = await fetch(`${API_BASE}/admin/compact`, { method: 'POST' })
  if (!res.ok) throw new Error(await res.text())
  return await res.json()
}
