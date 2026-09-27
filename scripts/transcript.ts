import type { Message, ToolResult, ToolUse } from '../src/types.js';

type Block = {
  type?: string;
  text?: string;
  id?: string;
  name?: string;
  input?: Record<string, unknown>;
  tool_use_id?: string;
  content?: string | Block[];
  is_error?: boolean;
};

type Entry = {
  type?: string;
  subtype?: string;
  isSidechain?: boolean;
  isMeta?: boolean;
  message?: { id?: string; role?: string; content?: string | Block[] };
};

function blockText(content: string | Block[] | undefined): string {
  if (typeof content === 'string') return content;
  return (content ?? [])
    .filter((block) => block.type === 'text' && typeof block.text === 'string')
    .map((block) => block.text)
    .join('\n');
}

/**
 * Reads a Claude Code session transcript (`~/.claude/projects/<project>/<session>.jsonl`)
 * into the library's messages, as the engine would hand them to `session.compact`:
 * only the main conversation since the last compaction, one message per API
 * message (streamed assistant blocks merged), thinking dropped, and each
 * tool result's text mirrored onto its tool_use.
 */
export function parseTranscript(jsonl: string): Message[] {
  let messages: Message[] = [];
  let lastAssistantId: string | undefined;
  for (const line of jsonl.split('\n')) {
    if (!line.trim()) continue;
    let entry: Entry;
    try {
      entry = JSON.parse(line) as Entry;
    } catch {
      continue;
    }
    if (entry.type === 'system' && entry.subtype === 'compact_boundary') {
      messages = [];
      lastAssistantId = undefined;
      continue;
    }
    if ((entry.type !== 'user' && entry.type !== 'assistant') || entry.isSidechain || entry.isMeta) {
      continue;
    }
    const content = entry.message?.content;
    const blocks: Block[] = typeof content === 'string' ? [{ type: 'text', text: content }] : (content ?? []);
    const text = blockText(blocks);
    const toolUses: ToolUse[] = blocks
      .filter((block) => block.type === 'tool_use' && block.id)
      .map((block) => ({ tool_use_id: block.id!, tool: block.name ?? '', input: block.input ?? {} }));
    const toolResults: ToolResult[] = blocks
      .filter((block) => block.type === 'tool_result' && block.tool_use_id)
      .map((block) => ({
        tool_use_id: block.tool_use_id!,
        text: blockText(block.content),
        isError: block.is_error ?? false,
      }));
    if (!text.trim() && toolUses.length === 0 && toolResults.length === 0) continue;

    const previous = messages[messages.length - 1];
    const id = entry.message?.id;
    if (entry.type === 'assistant' && previous?.role === 'assistant' && id && id === lastAssistantId) {
      previous.text = [previous.text, text].filter(Boolean).join('\n');
      previous.toolUses.push(...toolUses);
      continue;
    }
    lastAssistantId = entry.type === 'assistant' ? id : undefined;
    const message: Message = { role: entry.type, text, toolUses };
    if (toolResults.length > 0) message.toolResults = toolResults;
    messages.push(message);
  }

  const results = new Map<string, ToolResult>();
  for (const message of messages) for (const result of message.toolResults ?? []) results.set(result.tool_use_id, result);
  for (const message of messages) {
    for (const tool of message.toolUses) {
      const result = results.get(tool.tool_use_id);
      if (!result) continue;
      tool.text = result.text;
      if (result.isError) tool.isError = true;
    }
  }
  return messages;
}
