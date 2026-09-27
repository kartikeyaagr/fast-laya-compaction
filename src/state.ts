import type { Message, ToolCall, ToolResult } from './types.js';

/** Cap on the serialised tool input in a one-line call. */
const INPUT_CHARS = 200;
const GOAL_CHARS = 300;
const INTENT_CHARS = 200;
const OUTPUT_HEAD_CHARS = 400;
const LATER_CALLS = 3;

/** Input keys that name what a call operates on; equal targets mean the same file, command or search. */
const TARGET_KEYS = ['file_path', 'notebook_path', 'path', 'url', 'command', 'pattern'] as const;

/** Tools that change their target. A change is never superseded: the record that it was made still matters. */
const MUTATING_TOOLS = new Set(['Edit', 'MultiEdit', 'Write', 'NotebookEdit']);

/** Fields shortened, then removed, in this order when a state exceeds its cap: least decisive first. */
const TRIM_ORDER = ['output_head', 'intent', 'goal', 'later_calls'] as const;

export function truncate(text: string, limit: number): string {
  return text.length <= limit ? text : `${text.slice(0, Math.max(0, limit - 1))}…`;
}

export function isPinned(
  index: number,
  total: number,
  preserveRecentMessages: number,
): boolean {
  return index === 0 || index >= total - preserveRecentMessages;
}

/**
 * Pairs every tool_use with its tool_result by `tool_use_id`. Calls without a
 * result are not candidates (there is nothing to drop yet).
 */
export function collectToolCalls(
  messages: readonly Message[],
  preserveRecentMessages: number,
): ToolCall[] {
  const results = new Map<string, { index: number; result: ToolResult }>();
  messages.forEach((message, index) => {
    for (const result of message.toolResults ?? []) {
      results.set(result.tool_use_id, { index, result });
    }
  });
  const calls: ToolCall[] = [];
  messages.forEach((message, callIndex) => {
    for (const tool of message.toolUses) {
      const found = results.get(tool.tool_use_id);
      if (!found) continue;
      calls.push({
        id: `t${calls.length + 1}`,
        tool_use_id: tool.tool_use_id,
        tool: tool.tool,
        input: tool.input,
        callIndex,
        resultIndex: found.index,
        resultChars: found.result.text.length,
        isError: found.result.isError ?? false,
        pinned:
          isPinned(callIndex, messages.length, preserveRecentMessages) ||
          isPinned(found.index, messages.length, preserveRecentMessages),
      });
    }
  });
  return calls;
}

function inputText(value: unknown, limit: number): string {
  let json = '';
  try {
    json = JSON.stringify(value) ?? String(value);
  } catch {
    json = '[unserializable input]';
  }
  return truncate(json, limit);
}

/** One call as a single line, target first, e.g. `Edit file_path=src/a.ts old_string=… replace_all=false`. */
export function callLine(call: Pick<ToolCall, 'tool' | 'input'>): string {
  const isTarget = (key: string) => (TARGET_KEYS as readonly string[]).includes(key);
  const entries = Object.entries(call.input);
  const input = [...entries.filter(([key]) => isTarget(key)), ...entries.filter(([key]) => !isTarget(key))]
    .map(([key, value]) => {
      const text = typeof value === 'string' ? value : inputText(value, INPUT_CHARS);
      return `${key}=${text.replace(/\s+/g, ' ')}`;
    })
    .join(' ');
  return truncate(`${call.tool} ${input}`.trim(), INPUT_CHARS);
}

/** What a call operates on (file, command, search), or undefined when its input names nothing. */
export function callTarget(call: Pick<ToolCall, 'input'>): string | undefined {
  const parts = TARGET_KEYS.flatMap((key) => {
    const value = call.input[key];
    return typeof value === 'string' && value.trim() ? [value.trim()] : [];
  });
  return parts.length > 0 ? parts.join(' ') : undefined;
}

/** The same tool with the same input, a free-text `description` aside. */
function isRepeat(a: ToolCall, b: ToolCall): boolean {
  const { description: _a, ...inputA } = a.input;
  const { description: _b, ...inputB } = b.input;
  return a.tool === b.tool && inputText(inputA, Infinity) === inputText(inputB, Infinity);
}

/**
 * The later call that makes a read-only call obsolete: an exact repeat (the
 * same read, search or command run again) or a change to the same target,
 * after which the earlier output is stale. Such a call can be removed with
 * its result without asking Laya.
 */
export function supersededBy(call: ToolCall, calls: readonly ToolCall[]): ToolCall | undefined {
  if (MUTATING_TOOLS.has(call.tool)) return undefined;
  const target = callTarget(call);
  if (!target) return undefined;
  return calls.find(
    (later) =>
      later.callIndex > call.callIndex &&
      callTarget(later) === target &&
      (MUTATING_TOOLS.has(later.tool) || isRepeat(later, call)),
  );
}

/** The latest user prompts (tool-result messages excluded), as the default `goal`. */
export function goalFromMessages(messages: readonly Message[], count = 1): string {
  return messages
    .filter(
      (message) =>
        message.role === 'user' &&
        message.text.trim().length > 0 &&
        (message.toolResults ?? []).length === 0,
    )
    .slice(-count)
    .map((message) => truncate(message.text, GOAL_CHARS))
    .join('\n');
}

/** The assistant text that led to a call: its own message's, else the nearest before it this turn. */
function intentOf(messages: readonly Message[], callIndex: number): string {
  for (let i = callIndex; i >= 0; i--) {
    const message = messages[i]!;
    if (message.role === 'user' && (message.toolResults ?? []).length === 0) break;
    if (message.role === 'assistant' && message.text.trim()) return message.text.trim();
  }
  return '';
}

function resultText(messages: readonly Message[], call: ToolCall): string {
  return (
    messages[call.resultIndex]?.toolResults?.find((r) => r.tool_use_id === call.tool_use_id)?.text ?? ''
  );
}

/** Later assistant text mentions the target (a file by its base name). */
function mentionedLater(messages: readonly Message[], call: ToolCall, target: string): boolean {
  const needle = target.includes('/') ? target.slice(target.lastIndexOf('/') + 1) : target;
  if (needle.length < 3) return false;
  return messages
    .slice(call.resultIndex + 1)
    .some((message) => message.role === 'assistant' && message.text.includes(needle));
}

function capState(state: Record<string, unknown>, maxChars: number): Record<string, unknown> {
  for (const key of TRIM_ORDER) {
    const over = JSON.stringify(state).length - maxChars;
    if (over <= 0) break;
    const value = state[key];
    if (typeof value === 'string' && value.length > over + 20) {
      state[key] = truncate(value, value.length - over);
    } else delete state[key];
  }
  return state;
}

/**
 * What Laya reads about one tool call. Laya keeps only the head of a long
 * state, so the most decisive facts come first: the call, its outcome, how
 * long ago it was, and whether later calls touched the same target.
 * Context and a glimpse of the output follow, and are the first to go under
 * `maxChars`.
 */
export function callState(
  call: ToolCall,
  messages: readonly Message[],
  calls: readonly ToolCall[],
  goal: string,
  maxChars: number,
): Record<string, unknown> {
  const target = callTarget(call);
  const later = target
    ? calls.filter((other) => other.callIndex > call.callIndex && callTarget(other) === target)
    : [];
  const state: Record<string, unknown> = {
    tool_call: callLine(call),
    outcome: `${call.isError ? 'error' : 'ok'}, ${call.resultChars} chars`,
    messages_since: messages.length - 1 - call.resultIndex,
    // Laya reads key names as words: renaming this `touched_later` cut the
    // measured reduction from 33% to 23% on one session. Keep it.
    superseded: later.length > 0,
  };
  if (later.length > 0) state.later_calls = later.slice(0, LATER_CALLS).map(callLine);
  if (target) state.mentioned_later = mentionedLater(messages, call, target);
  if (goal) state.goal = truncate(goal, GOAL_CHARS);
  const intent = intentOf(messages, call.callIndex);
  if (intent) state.intent = truncate(intent, INTENT_CHARS);
  const output = resultText(messages, call);
  if (output) state.output_head = truncate(output, OUTPUT_HEAD_CHARS);
  return capState(state, maxChars);
}
