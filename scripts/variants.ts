/**
 * Prompt variants for the scoreboard (`npm run agreement -- --variant <name>`).
 * Each is a question plus, optionally, a different per-call state. Add one
 * here, score it against Jev, and promote the winner into src/.
 */
import { CALL_QUESTION, callLine, callState, type ChoiceQuestion, type Message, type ToolCall } from '../src/index.js';

export interface Variant {
  question: ChoiceQuestion;
  /** Laya's state for one call; the plugin's `callState` when absent. */
  state?: (call: ToolCall, messages: readonly Message[], calls: readonly ToolCall[], goal: string) => Record<string, unknown>;
  /** Tokens for the question and its options; longer questions need more than the default 128. */
  headMaxLen?: number;
  /** Tokens per row, question included; the typed-decisions checkpoint takes up to 1024. Default 512. */
  maxLen?: number;
}

const MAX_CHARS = 1600;

/** fast-jev-compaction's two questions folded into the three answers, worded positively. */
const JEV_QUESTION: ChoiceQuestion = {
  type: 'choice',
  instructions:
    'A coding assistant conversation is being compacted to free context. Whatever is not kept is deleted permanently, but the assistant can always re-run a tool or re-read a file. What should happen to this tool call?',
  criteria: {
    keep: 'its full output should stay verbatim: the assistant still needs its contents and re-running the tool would not do',
    truncate: 'only knowing this call was made, with its input, still matters for what the assistant does next',
    drop: 'the call and its output are irrelevant to what the assistant does next',
  },
};

/** Jev never sees tool outputs, only a note like `ok, 4213 chars`. */
function withoutOutput(state: Record<string, unknown>): Record<string, unknown> {
  const { output_head: _, ...rest } = state;
  return rest;
}

/** What the assistant said right after the result: whether it acted on it. */
function withNext(call: ToolCall, messages: readonly Message[], state: Record<string, unknown>): Record<string, unknown> {
  const next = messages.slice(call.resultIndex + 1).find((m) => m.role === 'assistant' && m.text.trim());
  if (!next) return state;
  const { goal, intent, output_head, ...head } = state;
  return { ...head, next: next.text.trim().slice(0, 200), goal, intent, output_head };
}

/**
 * A Jev-like view of what happened after the call, one line per message, as
 * much as fits in `maxChars`. Laya cannot pick one call out of a whole
 * history (measured: every call, and a call that does not exist, got the same
 * answer), so the history stays centred on the call being asked about.
 */
function withTimeline(
  call: ToolCall,
  messages: readonly Message[],
  calls: readonly ToolCall[],
  state: Record<string, unknown>,
  maxChars = 900,
): Record<string, unknown> {
  const after: string[] = [];
  let used = 0;
  for (let i = call.resultIndex + 1; i < messages.length && used < maxChars; i++) {
    const m = messages[i]!;
    const text = m.text.trim().replace(/\s+/g, ' ').slice(0, 120);
    const lines = [
      ...(text && (m.role === 'assistant' || (m.toolResults ?? []).length === 0) ? [`${m.role}: ${text}`] : []),
      ...calls.filter((c) => c.callIndex === i).map((c) => `call: ${callLine(c).slice(0, 100)}`),
    ];
    for (const line of lines) {
      after.push(line);
      used += line.length;
    }
  }
  const { goal, intent, output_head, ...head } = state;
  return { ...head, after, goal, intent, output_head };
}

export const VARIANTS: Record<string, Variant> = {
  current: { question: CALL_QUESTION },
  'jev-question': { question: JEV_QUESTION, headMaxLen: 192 },
  'no-output': {
    question: CALL_QUESTION,
    state: (call, messages, calls, goal) => withoutOutput(callState(call, messages, calls, goal, MAX_CHARS)),
  },
  'jev-question-no-output': {
    question: JEV_QUESTION,
    headMaxLen: 192,
    state: (call, messages, calls, goal) => withoutOutput(callState(call, messages, calls, goal, MAX_CHARS)),
  },
  timeline: {
    question: CALL_QUESTION,
    maxLen: 1024,
    state: (call, messages, calls, goal) => withTimeline(call, messages, calls, callState(call, messages, calls, goal, MAX_CHARS)),
  },
  next: {
    question: CALL_QUESTION,
    state: (call, messages, calls, goal) => withNext(call, messages, callState(call, messages, calls, goal, MAX_CHARS)),
  },
};
