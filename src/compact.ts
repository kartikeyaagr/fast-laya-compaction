import { callState, collectToolCalls, goalFromMessages, supersededBy } from './state.js';
import type {
  CallDecision,
  CallProbabilities,
  ChoiceQuestion,
  CompactOptions,
  CompactResult,
  LayaScorer,
  LayaScores,
  Message,
  ResolvedCompactOptions,
  ToolCall,
  ToolUse,
} from './types.js';

export const DEFAULT_OPTIONS: ResolvedCompactOptions = {
  goal: '',
  keepThreshold: 0.5,
  preserveRecentMessages: 6,
  maxCallStateChars: 1600,
  maxScoredCalls: 80,
  truncateHeadChars: 300,
};

function finite(value: number | undefined, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

export function resolveOptions(options: CompactOptions = {}): ResolvedCompactOptions {
  return {
    goal: options.goal ?? DEFAULT_OPTIONS.goal,
    keepThreshold: finite(options.keepThreshold, DEFAULT_OPTIONS.keepThreshold),
    preserveRecentMessages: Math.max(
      0,
      Math.floor(
        finite(options.preserveRecentMessages, DEFAULT_OPTIONS.preserveRecentMessages),
      ),
    ),
    maxCallStateChars: Math.max(
      200,
      Math.floor(finite(options.maxCallStateChars, DEFAULT_OPTIONS.maxCallStateChars)),
    ),
    maxScoredCalls: Math.max(
      0,
      Math.floor(finite(options.maxScoredCalls, DEFAULT_OPTIONS.maxScoredCalls)),
    ),
    truncateHeadChars: Math.max(
      0,
      Math.floor(finite(options.truncateHeadChars, DEFAULT_OPTIONS.truncateHeadChars)),
    ),
  };
}

/**
 * The one question asked about every scored call, each with its own state.
 * The criteria are worded positively: Laya reads negations poorly. (A bare
 * yes/no question measured worse on real sessions: less reduction, a less
 * sensible ranking, and the same answers when asked the opposite way.)
 */
export const CALL_QUESTION: ChoiceQuestion = {
  type: 'choice',
  instructions:
    "A coding assistant's conversation is being compacted. The state describes one earlier tool call. What should happen to it?",
  criteria: {
    keep: 'the assistant still needs the full output verbatim',
    truncate: 'only the fact that this call was made still matters',
    drop: 'the call is obsolete: superseded, exploratory, or already acted on',
  },
};

/** Laya's answer for one call; throws when a probability is missing or out of range. */
export function callProbabilities(scores: LayaScores['scores'], id: string): CallProbabilities {
  const p = scores[id];
  const valid = (x: unknown): x is number => typeof x === 'number' && x >= 0 && x <= 1;
  if (!p || !valid(p.keep) || !valid(p.truncate) || !valid(p.drop)) {
    throw new Error(`Invalid Laya answer for ${id}`);
  }
  return { keep: p.keep, truncate: p.truncate, drop: p.drop };
}

/**
 * One call's fate. Pinned calls stay and superseded ones go with their
 * result. For a scored call: `P(keep) ≥ threshold` keeps it whole, else
 * `P(keep) + P(truncate) ≥ threshold` keeps the call with a truncated output,
 * else it goes. An unscored call stays.
 */
export function decideCall(
  call: Pick<ToolCall, 'id' | 'tool' | 'pinned'>,
  verdict: { superseded?: boolean; probabilities?: CallProbabilities },
  options: Pick<ResolvedCompactOptions, 'keepThreshold'>,
): CallDecision {
  const base = { id: call.id, tool: call.tool };
  if (call.pinned) return { ...base, action: 'keep', reason: 'pinned' };
  if (verdict.superseded) return { ...base, action: 'drop_call', reason: 'superseded' };
  const p = verdict.probabilities;
  if (!p) return { ...base, action: 'keep', reason: 'unscored' };
  if (p.keep >= options.keepThreshold) return { ...base, probabilities: p, action: 'keep', reason: 'kept' };
  if (p.keep + p.truncate >= options.keepThreshold) {
    return { ...base, probabilities: p, action: 'drop_result', reason: 'result_dropped' };
  }
  return { ...base, probabilities: p, action: 'drop_call', reason: 'call_dropped' };
}

function truncatedResultText(text: string, isError: boolean, headChars: number): string {
  if (text.length <= headChars + 120) return text;
  const head = headChars > 0 ? `${text.slice(0, headChars)}\n` : '';
  return `${head}[fast-laya-compaction truncated ${text.length - headChars} chars of this tool result${
    isError ? ' (error)' : ''
  }; re-run the tool if needed]`;
}

/**
 * Rebuilds the conversation from the decisions. A dropped call disappears
 * together with its result; a dropped result keeps a bounded head and note.
 * Messages that lose all their content are removed; untouched messages are
 * returned as the same objects they came in as.
 */
export function applyDecisions(
  messages: readonly Message[],
  decisions: readonly CallDecision[],
  calls: readonly ToolCall[],
  headChars: number,
): Message[] {
  const byId = new Map(calls.map((call) => [call.id, call]));
  const actions = new Map<string, CallDecision['action']>();
  for (const decision of decisions) {
    const call = byId.get(decision.id);
    if (call && decision.action !== 'keep') actions.set(call.tool_use_id, decision.action);
  }
  const kept: Message[] = [];
  for (const message of messages) {
    const touched =
      message.toolUses.some((tool) => actions.has(tool.tool_use_id)) ||
      (message.toolResults ?? []).some((result) => actions.has(result.tool_use_id));
    if (!touched) {
      kept.push(message);
      continue;
    }
    const toolUses = message.toolUses
      .filter((tool) => actions.get(tool.tool_use_id) !== 'drop_call')
      .map((tool) => {
        if (actions.get(tool.tool_use_id) !== 'drop_result') return tool;
        const text = truncatedResultText(
          tool.text ?? '',
          tool.isError ?? false,
          headChars,
        );
        if ((tool.text ?? '') === text) return tool;
        const copy: ToolUse = {
          tool_use_id: tool.tool_use_id,
          tool: tool.tool,
          input: tool.input,
          text,
        };
        if (tool.isError) copy.isError = true;
        return copy;
      });
    const toolResults = (message.toolResults ?? [])
      .filter((result) => actions.get(result.tool_use_id) !== 'drop_call')
      .map((result) => {
        if (actions.get(result.tool_use_id) !== 'drop_result') return result;
        const text = truncatedResultText(result.text, result.isError ?? false, headChars);
        return text === result.text
          ? result
          : {
              tool_use_id: result.tool_use_id,
              text,
              isError: result.isError,
            };
      });
    if (
      !message.toolUses.some(
        (tool) => actions.get(tool.tool_use_id) === 'drop_call',
      ) &&
      !(message.toolResults ?? []).some(
        (result) => actions.get(result.tool_use_id) === 'drop_call',
      ) &&
      toolUses.every((tool, index) => tool === message.toolUses[index]) &&
      toolResults.every(
        (result, index) => result === message.toolResults?.[index],
      )
    ) {
      kept.push(message);
      continue;
    }
    if (message.text.trim().length === 0 && toolUses.length === 0 && toolResults.length === 0) {
      continue;
    }
    const rebuilt: Message = { role: message.role, text: message.text, toolUses };
    if (toolResults.length > 0) rebuilt.toolResults = toolResults;
    kept.push(rebuilt);
  }
  return kept;
}

/** Characters of text, tool input and tool output a message holds. */
export function messageChars(message: Message): number {
  let total = message.text.length;
  for (const tool of message.toolUses) {
    try {
      total += JSON.stringify(tool.input).length;
    } catch {
      total += 20;
    }
  }
  for (const result of message.toolResults ?? []) total += result.text.length;
  return total;
}

export function reductionRatio(result: Pick<CompactResult, 'stats'>): number {
  const { charsBefore, charsAfter } = result.stats;
  return charsBefore === 0 ? 0 : (charsBefore - charsAfter) / charsBefore;
}

function count(decisions: readonly CallDecision[], reason: CallDecision['reason']): number {
  return decisions.filter((decision) => decision.reason === reason).length;
}

/**
 * Compacts a transcript. Outside the pinned first and newest messages, a
 * read-only call that a later call repeated or made stale is removed with
 * its result; of the rest, the oldest `maxScoredCalls` are each described by
 * their own small state (see `callState`) and Laya is asked, in one batched
 * request, whether each should be kept, truncated or dropped. Throws when
 * Laya fails; the caller decides whether to fall back.
 */
export async function compact(
  messages: readonly Message[],
  scorer: LayaScorer,
  options: CompactOptions = {},
): Promise<CompactResult> {
  const started = Date.now();
  const resolved = resolveOptions(options);
  const calls = collectToolCalls(messages, resolved.preserveRecentMessages);
  const superseded = new Set(
    calls.filter((call) => !call.pinned && supersededBy(call, calls)).map((call) => call.id),
  );
  // Oldest first: they are the likeliest to be stale, and scoring time grows with every call.
  const candidates = calls
    .filter((call) => !call.pinned && !superseded.has(call.id))
    .slice(0, resolved.maxScoredCalls);
  const charsBefore = messages.reduce((sum, message) => sum + messageChars(message), 0);

  let laya: Omit<LayaScores, 'scores'> = { model: '', device: '', loadMs: 0, inferMs: 0 };
  const answers = new Map<string, CallProbabilities>();
  if (candidates.length > 0) {
    const goal = resolved.goal || goalFromMessages(messages);
    const states = candidates.map((call) => ({
      id: call.id,
      state: callState(call, messages, calls, goal, resolved.maxCallStateChars),
    }));
    const { model, device, loadMs, inferMs, scores } = await scorer.score(states, CALL_QUESTION);
    laya = { model, device, loadMs, inferMs };
    for (const call of candidates) answers.set(call.id, callProbabilities(scores, call.id));
  }

  const decisions = calls.map((call) =>
    decideCall(call, { superseded: superseded.has(call.id), probabilities: answers.get(call.id) }, resolved),
  );
  const kept = applyDecisions(
    messages,
    decisions,
    calls,
    resolved.truncateHeadChars,
  );
  return {
    messages: kept,
    decisions,
    stats: {
      messagesBefore: messages.length,
      messagesAfter: kept.length,
      charsBefore,
      charsAfter: kept.reduce((sum, message) => sum + messageChars(message), 0),
      calls: calls.length,
      kept: count(decisions, 'kept'),
      resultsDropped: count(decisions, 'result_dropped'),
      callsDropped: count(decisions, 'call_dropped'),
      superseded: count(decisions, 'superseded'),
      pinned: count(decisions, 'pinned'),
      unscored: count(decisions, 'unscored'),
      ...laya,
      ms: Date.now() - started,
    },
  };
}
