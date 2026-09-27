/**
 * Runs the compaction over a real session transcript and prints what Laya
 * decided, without touching anything. The tool for tuning `keepThreshold`,
 * the question wording and the checkpoint before trusting the hook.
 *
 *   npm run dry-run -- ~/.claude/projects/<project>/<session>.jsonl \
 *     [--threshold 0.5] [--recent 6] [--max-scored 80] [--model typed-decisions] [--device mps] [--min-reduction 0.25]
 *     [--target 0.5]   # simulate auto-compaction: trim what Laya keeps until this reduction
 */
import { readFileSync } from 'node:fs';
import { parseArgs } from 'node:util';
import {
  callLine,
  collectToolCalls,
  compact,
  NodeLayaScorer,
  reductionRatio,
} from '../src/index.js';
import { parseTranscript } from './transcript.js';

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    threshold: { type: 'string', default: '0.5' },
    recent: { type: 'string', default: '6' },
    model: { type: 'string' },
    device: { type: 'string' },
    'min-reduction': { type: 'string', default: '0.25' },
    'max-scored': { type: 'string', default: '80' },
    target: { type: 'string', default: '0' },
  },
});
const file = positionals[0];
if (!file) {
  console.error('usage: npm run dry-run -- <session.jsonl> [--threshold 0.5] [--recent 6] [--model m] [--device d]');
  process.exit(2);
}

const messages = parseTranscript(readFileSync(file, 'utf8'));
const options = {
  keepThreshold: Number(values.threshold),
  preserveRecentMessages: Number(values.recent),
  maxScoredCalls: Number(values['max-scored']),
  targetReduction: Number(values.target),
};
const calls = collectToolCalls(messages, options.preserveRecentMessages);

const scorer = new NodeLayaScorer({ model: values.model, device: values.device });

const started = Date.now();
const result = await compact(messages, scorer, options);
const wall = Date.now() - started;

const { stats } = result;
console.log(
  `${file}\n${messages.length} messages, ${calls.length} tool calls, ${stats.superseded} superseded, ${stats.kept + stats.resultsDropped} scored\n`,
);
const fixed = (n: number | undefined) => (n === undefined ? '  -  ' : n.toFixed(2).padStart(5));
console.log(`${'id'.padEnd(5)} ${'keep'.padStart(5)} ${'trunc'.padStart(5)} ${'drop'.padStart(5)}  ${'action'.padEnd(11)} call`);
for (const decision of result.decisions) {
  const call = calls.find((c) => c.id === decision.id)!;
  const p = decision.probabilities;
  const action = decision.reason === 'trimmed' || !p ? decision.reason : decision.action;
  console.log(
    `${decision.id.padEnd(5)} ${fixed(p?.keep)} ${fixed(p?.truncate)} ${fixed(p?.drop)}  ${action.padEnd(11)} ${callLine(call).slice(0, 90)}`,
  );
}
const ratio = reductionRatio(result);
console.log(
  `\nreduction ${Math.round(ratio * 100)}% (${stats.charsBefore} -> ${stats.charsAfter} chars, ${stats.messagesBefore} -> ${stats.messagesAfter} messages)`,
);
console.log(
  `${stats.kept} kept, ${stats.resultsDropped} results truncated, ${stats.trimmed} trimmed for room, ${stats.callsDropped} calls dropped, ${stats.superseded} superseded calls removed, ${stats.pinned} pinned, ${stats.unscored} unscored`,
);
console.log(
  stats.model
    ? `laya ${stats.model} on ${stats.device}: load ${stats.loadMs} ms + infer ${stats.inferMs} ms; wall ${wall} ms`
    : `laya not called; wall ${wall} ms`,
);
console.log(
  ratio >= Number(values['min-reduction'])
    ? 'the hook would replace the history'
    : `the hook would fall back to the built-in summary (below ${values['min-reduction']})`,
);
