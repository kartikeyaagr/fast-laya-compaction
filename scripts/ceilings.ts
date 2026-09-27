/**
 * Benchmark context for one or more sessions, without calling any model:
 * the most any keep/truncate/drop policy could remove (every scored call
 * truncated, every scored call dropped), and what the original
 * fast-jev-compaction would send to Jev for the same session (its fitted
 * whole-history state, request count and tokens).
 *
 *   npm run bench:ceilings -- ~/.claude/projects/<project>/<session>.jsonl …
 */
import { readFileSync } from 'node:fs';
import { batchCalls, estimateTokens, fitState, questionsFor, resolveOptions as jevOptions } from 'fast-jev-compaction';
import { collectToolCalls } from '../src/index.js';
import { reductionUnder } from './agreement.js';
import { parseTranscript } from './transcript.js';

const pct = (x: number) => `${Math.round(x * 100)}%`;
for (const file of process.argv.slice(2)) {
  const messages = parseTranscript(readFileSync(file, 'utf8'));
  const calls = collectToolCalls(messages, 6);
  const candidates = calls.filter((c) => !c.pinned);
  const all = (action: 'drop_result' | 'drop_call') => new Map(candidates.map((c) => [c.id, action]));

  let jev: string;
  try {
    const opts = jevOptions({});
    const fitted = fitState(messages, calls, opts);
    const batches = batchCalls(candidates, fitted.tokens, opts);
    const questions = candidates.reduce((sum, c) => sum + estimateTokens(JSON.stringify(questionsFor(c))), 0);
    jev = `~${fitted.tokens} state tokens (${fitted.stage}), ${batches.length} request(s), ~${batches.length * (fitted.tokens + 20) + questions} tokens sent`;
  } catch (error) {
    jev = `does not fit: ${error instanceof Error ? error.message : String(error)}`;
  }
  console.log(`${file}`);
  console.log(`  ${messages.length} messages, ${calls.length} calls, ${candidates.length} scoreable`);
  console.log(`  ceilings: truncate every call ${pct(reductionUnder(messages, calls, all('drop_result')))}, drop every call ${pct(reductionUnder(messages, calls, all('drop_call')))}`);
  console.log(`  jev: ${jev}`);
}
