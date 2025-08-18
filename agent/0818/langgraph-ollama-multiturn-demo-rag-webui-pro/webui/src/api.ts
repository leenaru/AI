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
