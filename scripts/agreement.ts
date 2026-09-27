/**
 * The scoreboard: how close Laya's decisions come to Jev's on the sessions
 * labelled by `npm run jev-labels`. Reports decision agreement, the
 * Jev-by-Laya confusion, rank correlation of the probabilities, how Jev
 * treats the calls our superseded rule removes, and each session's reduction
 * under both. Laya scores every labelled call in one run (no maxScoredCalls).
 *
 *   npm run agreement -- [--variant current] [--threshold 0.5] [--labels data] [--model typed-decisions]
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { pathToFileURL } from 'node:url';
import {
  applyDecisions,
  callState,
  collectToolCalls,
  decideCall,
  goalFromMessages,
  messageChars,
  NodeLayaScorer,
  type CallAction,
  type CallDecision,
  type Message,
  type ToolCall,
} from '../src/index.js';
import { readJsonl, type JevLabel } from './jev-labels.js';
import { parseTranscript } from './transcript.js';
import { VARIANTS, type Variant } from './variants.js';

const ACTIONS: CallAction[] = ['keep', 'drop_result', 'drop_call'];
const REASON: Record<CallAction, CallDecision['reason']> = {
  keep: 'kept',
  drop_result: 'result_dropped',
  drop_call: 'call_dropped',
};

/** Ranks with ties averaged, 1-based. */
function ranks(xs: readonly number[]): number[] {
  const order = xs.map((x, i) => [x, i] as const).sort((a, b) => a[0] - b[0]);
  const out = new Array<number>(xs.length);
  for (let i = 0; i < order.length; ) {
    let j = i;
    while (j + 1 < order.length && order[j + 1]![0] === order[i]![0]) j++;
    for (let k = i; k <= j; k++) out[order[k]![1]] = (i + j) / 2 + 1;
    i = j + 1;
  }
  return out;
}

/** Spearman's rank correlation (Pearson on tie-averaged ranks); NaN when either side is constant. */
export function spearman(a: readonly number[], b: readonly number[]): number {
  const ra = ranks(a);
  const rb = ranks(b);
  const mean = (xs: number[]) => xs.reduce((s, x) => s + x, 0) / xs.length;
  const ma = mean(ra);
  const mb = mean(rb);
  let cov = 0;
  let va = 0;
  let vb = 0;
  for (let i = 0; i < ra.length; i++) {
    cov += (ra[i]! - ma) * (rb[i]! - mb);
    va += (ra[i]! - ma) ** 2;
    vb += (rb[i]! - mb) ** 2;
  }
  return cov / Math.sqrt(va * vb);
}

/** Counts of [jev action][laya action]. */
export function confusion(pairs: readonly (readonly [CallAction, CallAction])[]): Record<CallAction, Record<CallAction, number>> {
  const table = Object.fromEntries(ACTIONS.map((a) => [a, Object.fromEntries(ACTIONS.map((b) => [b, 0]))])) as Record<
    CallAction,
    Record<CallAction, number>
  >;
  for (const [jev, laya] of pairs) table[jev][laya]++;
  return table;
}

/** Character reduction when `actions` (by call id) are applied; pinned or absent calls stay. */
export function reductionUnder(messages: readonly Message[], calls: readonly ToolCall[], actions: Map<string, CallAction>): number {
  const decisions = calls.map((call): CallDecision => {
    const action = call.pinned ? 'keep' : (actions.get(call.id) ?? 'keep');
    return { id: call.id, tool: call.tool, action, reason: call.pinned ? 'pinned' : REASON[action] };
  });
  const before = messages.reduce((sum, m) => sum + messageChars(m), 0);
  const after = applyDecisions(messages, decisions, calls, 300).reduce((sum, m) => sum + messageChars(m), 0);
  return before === 0 ? 0 : (before - after) / before;
}

export interface LabelledSession {
  session: string;
  labels: JevLabel[];
  messages: Message[];
  calls: ToolCall[];
  byId: Map<string, ToolCall>;
  goal: string;
}

/** Each labelled session rebuilt as it was when labelled, so a variant can build its own states. */
export function labelledSessions(labels: readonly JevLabel[]): LabelledSession[] {
  return [...new Set(labels.map((l) => l.session))].map((session) => {
    const own = labels.filter((l) => l.session === session);
    const messages = parseTranscript(readFileSync(session, 'utf8')).slice(0, own[0]!.messages);
    const calls = collectToolCalls(messages, own[0]!.recent);
    const byId = new Map(calls.map((c) => [c.id, c]));
    for (const label of own) {
      if (byId.get(label.id)?.tool !== label.tool) throw new Error(`${session} changed since it was labelled (${label.id})`);
    }
    return { session, labels: own, messages, calls, byId, goal: goalFromMessages(messages) };
  });
}

/** The state Laya reads for one labelled call under a variant. */
export function variantState(variant: Variant, s: LabelledSession, label: JevLabel): Record<string, unknown> {
  const call = s.byId.get(label.id)!;
  return variant.state?.(call, s.messages, s.calls, s.goal) ?? callState(call, s.messages, s.calls, s.goal, 1600);
}

const pct = (x: number) => `${Math.round(x * 100)}%`;

async function main(): Promise<void> {
  const { values } = parseArgs({
    options: {
      labels: { type: 'string', default: 'data' },
      variant: { type: 'string', default: 'current' },
      threshold: { type: 'string', default: '0.5' },
      model: { type: 'string' },
    },
  });
  const variant = VARIANTS[values.variant];
  if (!variant) {
    console.error(`unknown variant ${values.variant}; one of: ${Object.keys(VARIANTS).join(', ')}`);
    process.exit(2);
  }
  const keepThreshold = Number(values.threshold);
  const labels = readJsonl<JevLabel>(join(values.labels, 'jev-labels.jsonl'));
  if (labels.length === 0) {
    console.error(`no labels in ${values.labels}/jev-labels.jsonl; run npm run jev-labels first`);
    process.exit(2);
  }

  const sessions = labelledSessions(labels);
  const states = sessions.flatMap((s, i) =>
    s.labels.filter((l) => !l.superseded).map((l) => ({ id: `${i}#${l.id}`, state: variantState(variant, s, l) })),
  );
  console.log(`variant ${values.variant}: scoring ${states.length} calls from ${sessions.length} session(s)…`);
  const scored = await new NodeLayaScorer({
    model: values.model,
    headMaxLen: variant.headMaxLen,
    maxLen: variant.maxLen,
    timeoutMs: 60 * 60_000,
  }).score(states, variant.question);

  const pairs: [CallAction, CallAction][] = [];
  const keepPairs: [number, number][] = [];
  const stayPairs: [number, number][] = [];
  const ruleVsJev: CallAction[] = [];
  const rows: string[] = [];
  sessions.forEach((s, i) => {
    const jevActions = new Map<string, CallAction>();
    const layaActions = new Map<string, CallAction>();
    for (const label of s.labels) {
      jevActions.set(label.id, label.jev.action);
      const p = label.superseded ? undefined : scored.scores[`${i}#${label.id}`];
      const probabilities = p ? { keep: p.keep!, truncate: p.truncate!, drop: p.drop! } : undefined;
      const laya = decideCall({ id: label.id, tool: label.tool, pinned: false }, { superseded: label.superseded, probabilities }, { keepThreshold });
      layaActions.set(label.id, laya.action);
      pairs.push([label.jev.action, laya.action]);
      if (label.superseded) ruleVsJev.push(label.jev.action);
      if (probabilities) {
        keepPairs.push([probabilities.keep, label.jev.keepResult]);
        stayPairs.push([probabilities.keep + probabilities.truncate, label.jev.keepCall]);
      }
    }
    rows.push(
      `  ${pct(reductionUnder(s.messages, s.calls, jevActions)).padStart(4)} jev  ${pct(reductionUnder(s.messages, s.calls, layaActions)).padStart(4)} laya  ${s.labels.length} calls  ${s.session.split('/').slice(-2).join('/')}`,
    );
  });

  const agree = pairs.filter(([a, b]) => a === b).length;
  const table = confusion(pairs);
  console.log(`\nagreement ${pct(agree / pairs.length)} (${agree}/${pairs.length}) at threshold ${keepThreshold}`);
  console.log(`\n${'jev \\ laya'.padEnd(12)}${ACTIONS.map((a) => a.padStart(12)).join('')}`);
  for (const jev of ACTIONS) console.log(`${jev.padEnd(12)}${ACTIONS.map((l) => String(table[jev][l]).padStart(12)).join('')}`);
  console.log(
    `\nrank correlation: P(keep) vs Jev keepResult ${spearman(keepPairs.map((p) => p[0]), keepPairs.map((p) => p[1])).toFixed(2)}, ` +
      `P(keep)+P(truncate) vs Jev keepCall ${spearman(stayPairs.map((p) => p[0]), stayPairs.map((p) => p[1])).toFixed(2)}`,
  );
  if (ruleVsJev.length > 0) {
    const same = ruleVsJev.filter((a) => a === 'drop_call').length;
    console.log(`superseded rule removed ${ruleVsJev.length} calls; Jev also removed ${same}, truncated ${ruleVsJev.filter((a) => a === 'drop_result').length}`);
  }
  console.log(`\nreduction per session:\n${rows.join('\n')}`);
  console.log(`\nlaya ${scored.model} on ${scored.device}: load ${scored.loadMs} ms + infer ${scored.inferMs} ms`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) await main();
