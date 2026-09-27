import type { CallState, ChoiceQuestion, LayaScores } from './types.js';

/** Laya's checkpoint fine-tuned for typed decisions; the base ones are near chance zero-shot. */
export const DEFAULT_MODEL = 'typed-decisions';

/** The scoring script, relative to the plugin (and package) root. */
export const SCRIPT_PATH = 'backend/laya_compact.py';

/** What to run once before the first compaction; the hook names the installed script's absolute path. */
export function setupHint(script: string = SCRIPT_PATH): string {
  return `Laya is not set up, run once: uv run --script ${script} --warmup`;
}

export const SETUP_HINT = setupHint();

/** The script's exit code for a checkpoint missing from the Hugging Face cache. */
const EXIT_NOT_CACHED = 3;

/** How uv reports a script environment it cannot build under `--offline`. */
const UV_OFFLINE = /network was disabled|not found in the cache/i;

export interface LayaRequestOptions {
  /** `typed-decisions` (default), `multilingual` or `english`. */
  model?: string;
  /** Torch device; auto (mps on Apple Silicon) when absent. */
  device?: string;
  /** Tokens per question row: head (question and options) plus state. */
  maxLen?: number;
  headMaxLen?: number;
  /** States per forward pass. */
  batchSize?: number;
}

/**
 * The command that scores one compaction. `--offline` keeps a hook from ever
 * building the torch environment inline: without the one-time setup it fails
 * fast and the caller falls back.
 */
export function layaArgv(uvPath: string, root: string): string[] {
  return [uvPath, 'run', '--quiet', '--offline', '--script', `${root}/${SCRIPT_PATH}`];
}

/** The script's stdin for one compaction: one question over every call state. */
export function buildLayaRequest(
  states: readonly CallState[],
  question: ChoiceQuestion,
  options: LayaRequestOptions = {},
): string {
  return JSON.stringify({
    model: options.model ?? DEFAULT_MODEL,
    ...(options.device ? { device: options.device } : {}),
    max_len: options.maxLen ?? 512,
    head_max_len: options.headMaxLen ?? 128,
    batch_size: options.batchSize ?? 32,
    question,
    states,
  });
}

function lastLine(text: string): string {
  const lines = text.split('\n').map((line) => line.trim()).filter(Boolean);
  return (lines[lines.length - 1] ?? '').slice(0, 200);
}

/** Validates the script's output; throws a message fit for a toast on anything else. */
export function parseLayaResponse(
  exitCode: number,
  stdout: string,
  stderr: string,
  script: string = SCRIPT_PATH,
): LayaScores {
  if (exitCode === EXIT_NOT_CACHED || (exitCode !== 0 && UV_OFFLINE.test(stderr))) {
    throw new Error(setupHint(script));
  }
  if (exitCode !== 0) {
    throw new Error(`Laya exited with ${exitCode}: ${lastLine(stderr) || 'no output'}`);
  }
  let parsed: unknown;
  try {
    // The answer is the last line; anything a library printed before it is noise.
    parsed = JSON.parse(stdout.trim().split('\n').pop() ?? '');
  } catch {
    throw new Error('Laya returned malformed JSON');
  }
  const body = parsed as Record<string, unknown> | null;
  if (!body || typeof body !== 'object' || !body.scores || typeof body.scores !== 'object') {
    throw new Error('Laya response is missing scores');
  }
  return {
    model: String(body.model ?? ''),
    device: String(body.device ?? ''),
    loadMs: Number(body.load_ms) || 0,
    inferMs: Number(body.infer_ms) || 0,
    scores: body.scores as LayaScores['scores'],
  };
}
