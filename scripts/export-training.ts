/**
 * Turns Jev's labels into fine-tuning data for Laya, in the row format of
 * Laya's own training notebook (the `LocalLLaMA/typed-decisions` dataset):
 * `state`, `questions` and `gold` as JSON strings, with Jev's answer as soft
 * target probabilities. Calls our superseded rule removes are left out: the
 * plugin never asks Laya about them. Whole sessions are held out, so the
 * held-out score says how Laya does on sessions it never saw.
 *
 *   npm run export-training -- [--labels data] [--variant current] [--holdout 0.2]
 *
 * Writes `<labels>/laya-train.jsonl` and `<labels>/laya-holdout.jsonl`.
 */
import { writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { pathToFileURL } from 'node:url';
import type { ChoiceQuestion } from '../src/index.js';
import { labelledSessions, variantState, type LabelledSession } from './agreement.js';
import { jevTarget, readJsonl, type JevLabel } from './jev-labels.js';
import { VARIANTS, type Variant } from './variants.js';

export interface TrainingRow {
  state: string;
  questions: string;
  gold: string;
}

/** One notebook row per call: the variant's state and question, Jev's answer as the target. */
export function trainingRows(sessions: readonly LabelledSession[], variant: Variant): TrainingRow[] {
  return sessions.flatMap((s) =>
    s.labels
      .filter((label) => !label.superseded)
      .map((label) => {
        const probabilities = jevTarget(label.jev);
        const best = (Object.keys(probabilities) as (keyof typeof probabilities)[]).reduce((a, b) =>
          probabilities[b] > probabilities[a] ? b : a,
        );
        const question: ChoiceQuestion = variant.question;
        return {
          state: JSON.stringify(variantState(variant, s, label)),
          questions: JSON.stringify({ action: question }),
          gold: JSON.stringify({ action: { type: 'choice', label: best, probabilities } }),
        };
      }),
  );
}

/**
 * Every `1/fraction`-th session (by sorted path) is held out, at least one
 * when there are two or more, so the split is stable as sessions are added.
 */
export function splitSessions<T extends { session: string }>(sessions: readonly T[], fraction: number): { train: T[]; holdout: T[] } {
  const sorted = [...sessions].sort((a, b) => a.session.localeCompare(b.session));
  if (sorted.length < 2 || fraction <= 0) return { train: sorted, holdout: [] };
  const every = Math.max(2, Math.round(1 / fraction));
  const holdout = sorted.filter((_, i) => i % every === every - 1);
  const chosen = holdout.length > 0 ? holdout : [sorted[sorted.length - 1]!];
  return { train: sorted.filter((s) => !chosen.includes(s)), holdout: chosen };
}

async function main(): Promise<void> {
  const { values } = parseArgs({
    options: {
      labels: { type: 'string', default: 'data' },
      variant: { type: 'string', default: 'current' },
      holdout: { type: 'string', default: '0.2' },
    },
  });
  const variant = VARIANTS[values.variant];
  if (!variant) {
    console.error(`unknown variant ${values.variant}; one of: ${Object.keys(VARIANTS).join(', ')}`);
    process.exit(2);
  }
  const labels = readJsonl<JevLabel>(join(values.labels, 'jev-labels.jsonl'));
  if (labels.length === 0) {
    console.error(`no labels in ${values.labels}/jev-labels.jsonl; run npm run jev-labels first`);
    process.exit(2);
  }
  const { train, holdout } = splitSessions(labelledSessions(labels), Number(values.holdout));
  for (const [name, sessions] of [['laya-train', train], ['laya-holdout', holdout]] as const) {
    const rows = trainingRows(sessions, variant);
    writeFileSync(join(values.labels, `${name}.jsonl`), rows.map((r) => JSON.stringify(r)).join('\n') + (rows.length ? '\n' : ''));
    console.log(`${name}: ${rows.length} rows from ${sessions.length} session(s)`);
  }
  if (variant.headMaxLen) console.log(`train with head_max_len >= ${variant.headMaxLen} (variant ${values.variant})`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) await main();
