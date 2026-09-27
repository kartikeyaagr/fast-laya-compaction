import { NodeLayaScorer, type NodeLayaScorerOptions } from './client.js';
import { compact } from './compact.js';
import type { CompactOptions, CompactResult, Message } from './types.js';

export type CompactMessagesOptions = CompactOptions & NodeLayaScorerOptions;

/** `compact` with a `NodeLayaScorer` built from the options (a local Laya run via `uv`). */
export function compactMessages(
  messages: readonly Message[],
  options: CompactMessagesOptions = {},
): Promise<CompactResult> {
  return compact(messages, new NodeLayaScorer(options), options);
}
