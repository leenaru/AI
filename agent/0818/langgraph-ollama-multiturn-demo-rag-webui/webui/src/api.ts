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
