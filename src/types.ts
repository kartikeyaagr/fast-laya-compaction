export type Role = 'user' | 'assistant';

/**
 * A tool_use block of an assistant message. `text` and `isError` mirror the
 * outcome once the transcript holds it (Claude Code attaches them).
 */
export interface ToolUse {
  tool_use_id: string;
  tool: string;
  input: Record<string, unknown>;
  text?: string;
  isError?: boolean;
}

/** A tool_result block of a user message. */
export interface ToolResult {
  tool_use_id: string;
  text: string;
  isError?: boolean;
}

/**
 * One transcript message. The shape is a subset of Claude Code's
 * `SessionMessage`, so a session transcript can be passed in as is.
 */
export interface Message {
  role: Role;
  text: string;
  toolUses: ToolUse[];
  toolResults?: ToolResult[];
}

/** A tool call paired with its result by `tool_use_id`. */
export interface ToolCall {
  /** Short id used in the Laya request and decision log (`t1`, `t2`, ...). */
  id: string;
  tool_use_id: string;
  tool: string;
  input: Record<string, unknown>;
  /** Index of the message holding the tool_use block. */
  callIndex: number;
  /** Index of the message holding the tool_result block. */
  resultIndex: number;
  resultChars: number;
  isError: boolean;
  /** In the first or the newest preserved messages; never a candidate. */
  pinned: boolean;
}

export type CallAction = 'keep' | 'drop_result' | 'drop_call';

/** Laya's answer for one call: keep it whole, truncate its output, or drop it. */
export interface CallProbabilities {
  keep: number;
  truncate: number;
  drop: number;
}

export interface CallDecision {
  id: string;
  tool: string;
  /** Laya's answer; only on scored calls. */
  probabilities?: CallProbabilities;
  action: CallAction;
  /**
   * `superseded`: a later call repeated it or changed its target, so it is
   * removed without asking Laya. `kept`/`result_dropped`/`call_dropped`: Laya's
   * answer. `trimmed`: Laya would keep it, but its output was truncated to reach
   * `targetReduction`.
   */
  reason: 'pinned' | 'superseded' | 'unscored' | 'kept' | 'result_dropped' | 'call_dropped' | 'trimmed';
}

export interface CompactOptions {
  /** Ongoing task description; defaults to the latest user prompt. */
  goal?: string;
  /** Minimum keep probability for a call or its full output to stay. Default 0.5. */
  keepThreshold?: number;
  /** Newest messages never touched (the first message is always kept). Default 6. */
  preserveRecentMessages?: number;
  /** Character cap on the state Laya reads for one call. Default 1600. */
  maxCallStateChars?: number;
  /** Most calls scored per compaction, oldest first; newer ones are kept. Bounds latency. Default 80. */
  maxScoredCalls?: number;
  /** Characters of a dropped tool result to retain. Default 300. */
  truncateHeadChars?: number;
  /**
   * Reduction to reach even past Laya's answers: outputs Laya would keep are
   * truncated, least likely to be needed first, until it is met. 0 (default) off.
   */
  targetReduction?: number;
}

export interface ResolvedCompactOptions {
  goal: string;
  keepThreshold: number;
  preserveRecentMessages: number;
  maxCallStateChars: number;
  maxScoredCalls: number;
  truncateHeadChars: number;
  targetReduction: number;
}

export interface CompactResult {
  /** The compacted transcript; untouched messages are the input objects. */
  messages: Message[];
  decisions: CallDecision[];
  stats: {
    messagesBefore: number;
    messagesAfter: number;
    charsBefore: number;
    charsAfter: number;
    calls: number;
    kept: number;
    resultsDropped: number;
    /** Calls Laya dropped with their results. */
    callsDropped: number;
    /** Calls removed with their results because a later call superseded them. */
    superseded: number;
    pinned: number;
    /** Kept without scoring: past `maxScoredCalls`. */
    unscored: number;
    /** Outputs Laya would keep, truncated to reach `targetReduction`. */
    trimmed: number;
    /** Laya checkpoint and torch device that scored the calls; '' when Laya was not called. */
    model: string;
    device: string;
    /** Checkpoint load and batched inference time inside the Laya process. */
    loadMs: number;
    inferMs: number;
    ms: number;
  };
}

/** A Laya `choice` question: one label per criterion, answered with a probability each. */
export interface ChoiceQuestion {
  type: 'choice';
  instructions: string;
  criteria: Record<string, string>;
}

/** What Laya reads about one tool call; key order matters, Laya cuts from the end. */
export interface CallState {
  id: string;
  state: Record<string, unknown>;
}

export interface LayaScores {
  model: string;
  device: string;
  loadMs: number;
  inferMs: number;
  /** Per call id, the probability of every criterion label. */
  scores: Record<string, Record<string, number>>;
}

/** Anything that answers one question over many call states: a Laya process, or a fake. */
export interface LayaScorer {
  score(states: readonly CallState[], question: ChoiceQuestion): Promise<LayaScores>;
}
