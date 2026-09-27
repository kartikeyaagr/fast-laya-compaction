/**
 * Collects Jev's decisions on real sessions, as the teacher for Laya. Runs the
 * original fast-jev-compaction over each transcript (this sends the transcript
 * to TypeSafe) and stores, for every scored tool call, Jev's two probabilities
 * next to the exact state Laya reads for that call. Nothing is compacted.
 *
 *   echo 'TYPESAFE_API_KEY=…' > .env
 *   npm run jev-labels -- ~/.claude/projects/<project>/<session>.jsonl … [--out data] [--recent 6]
 *
 * Writes `<out>/jev-labels.jsonl` (one line per call) and `<out>/jev-sessions.jsonl`
 * (one line per session); sessions already labelled are skipped, so it resumes.
 */
import { appendFileSync, existsSync, mkdirSync, readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { pathToFileURL } from 'node:url';
import { compact as jevCompact, JevClient, reductionRatio as jevReduction, type JevAsker } from 'fast-jev-compaction';
import {
  callLine,
  callState,
  collectToolCalls,
  goalFromMessages,
  resolveOptions,
  supersededBy,
  type CallAction,
  type CallProbabilities,
  type Message,
} from '../src/index.js';
import { parseTranscript } from './transcript.js';

export interface JevLabel {
  session: string;
  /** Messages the session had when labelled; a transcript that grew since is cut back to this. */
  messages: number;
  recent: number;
  id: string;
  tool: string;
  call: string;
  /** Removed by our superseded rule, whatever a model says. */
  superseded: boolean;
  state: Record<string, unknown>;
  jev: { keepCall: number; keepResult: number; action: CallAction };
}

export interface JevSession {
  session: string;
  messages: number;
  calls: number;
  scored: number;
  /** Jev's own character reduction on this session. */
  reduction: number;
  requests: number;
  ms: number;
}

/**
 * Jev's two independent answers as the three-way target Laya is asked for.
 * `P(keep) = keepResult` and `P(keep) + P(truncate) = max(keepCall, keepResult)`,
 * so `decideCall` on the target reproduces Jev's decision at every threshold.
 */
export function jevTarget(jev: Pick<JevLabel['jev'], 'keepCall' | 'keepResult'>): CallProbabilities {
  const keep = jev.keepResult;
  const stay = Math.max(jev.keepCall, jev.keepResult);
  return { keep, truncate: stay - keep, drop: 1 - stay };
}

/** Runs Jev over one transcript and pairs its answers with Laya's per-call states. */
export async function labelSession(
  session: string,
  messages: readonly Message[],
  asker: JevAsker,
  recent: number,
): Promise<{ labels: JevLabel[]; summary: JevSession }> {
  const result = await jevCompact(messages, asker, { preserveRecentMessages: recent });
  const decisions = new Map(result.decisions.map((d) => [d.id, d]));
  const calls = collectToolCalls(messages, recent);
  const goal = goalFromMessages(messages);
  const { maxCallStateChars } = resolveOptions();
  const labels = calls
    .filter((call) => !call.pinned)
    .map((call): JevLabel => {
      const decision = decisions.get(call.id);
      if (!decision || decision.tool !== call.tool) throw new Error(`Jev and Laya disagree on call ${call.id}`);
      return {
        session,
        messages: messages.length,
        recent,
        id: call.id,
        tool: call.tool,
        call: callLine(call),
        superseded: supersededBy(call, calls) !== undefined,
        state: callState(call, messages, calls, goal, maxCallStateChars),
        jev: { keepCall: decision.keepCall, keepResult: decision.keepResult, action: decision.action },
      };
    });
  return {
    labels,
    summary: {
      session,
      messages: messages.length,
      calls: calls.length,
      scored: labels.length,
      reduction: jevReduction(result),
      requests: result.stats.requests,
      ms: result.stats.ms,
    },
  };
}

export function readJsonl<T>(path: string): T[] {
  if (!existsSync(path)) return [];
  return readFileSync(path, 'utf8')
    .split('\n')
    .filter((line) => line.trim())
    .map((line) => JSON.parse(line) as T);
}

async function main(): Promise<void> {
  const { values, positionals } = parseArgs({
    allowPositionals: true,
    options: { out: { type: 'string', default: 'data' }, recent: { type: 'string', default: '6' } },
  });
  if (positionals.length === 0) {
    console.error('usage: npm run jev-labels -- <session.jsonl>… [--out data] [--recent 6]');
    process.exit(2);
  }
  if (!process.env.TYPESAFE_API_KEY) {
    console.error('TYPESAFE_API_KEY is not set: put it in .env (git-ignored) as TYPESAFE_API_KEY=…');
    process.exit(2);
  }
  mkdirSync(values.out, { recursive: true });
  const labelsPath = join(values.out, 'jev-labels.jsonl');
  const sessionsPath = join(values.out, 'jev-sessions.jsonl');
  const done = new Set(readJsonl<JevSession>(sessionsPath).map((s) => s.session));
  const asker = new JevClient();
  for (const file of positionals.map((p) => resolve(p))) {
    if (done.has(file)) {
      console.log(`skip (labelled) ${file}`);
      continue;
    }
    try {
      const { labels, summary } = await labelSession(
        file,
        parseTranscript(readFileSync(file, 'utf8')),
        asker,
        Number(values.recent),
      );
      for (const label of labels) appendFileSync(labelsPath, `${JSON.stringify(label)}\n`);
      appendFileSync(sessionsPath, `${JSON.stringify(summary)}\n`);
      console.log(
        `${Math.round(summary.reduction * 100)}% Jev reduction, ${summary.scored} calls, ${summary.requests} request(s) in ${summary.ms} ms: ${file}`,
      );
    } catch (error) {
      console.error(`failed (${error instanceof Error ? error.message : String(error)}): ${file}`);
    }
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) await main();
