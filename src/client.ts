import { spawn } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildLayaRequest, layaArgv, parseLayaResponse, type LayaRequestOptions } from './laya.js';
import type { CallState, ChoiceQuestion, LayaScorer, LayaScores } from './types.js';

export interface NodeLayaScorerOptions extends LayaRequestOptions {
  /** Defaults to `uv` on PATH. */
  uvPath?: string;
  /** Directory holding `backend/laya_compact.py`; defaults to this package's root. */
  root?: string;
  /** Defaults to 120 s. */
  timeoutMs?: number;
}

/** The package root: one level above `src/` (tsx) or `dist/` (built). */
const PACKAGE_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');

/**
 * Scores call states in a one-shot `uv run` of the Laya script, from Node.
 * The plugin hook does the same through `$.process.run`; this one serves the
 * library entry points and the dry-run tool.
 */
export class NodeLayaScorer implements LayaScorer {
  constructor(private readonly options: NodeLayaScorerOptions = {}) {}

  score(states: readonly CallState[], question: ChoiceQuestion): Promise<LayaScores> {
    const argv = layaArgv(
      this.options.uvPath ?? 'uv',
      this.options.root ?? PACKAGE_ROOT,
    );
    const [command, ...args] = argv;
    const timeoutMs = this.options.timeoutMs ?? 120_000;
    return new Promise((resolvePromise, reject) => {
      const child = spawn(command!, args, { stdio: 'pipe', timeout: timeoutMs });
      let stdout = '';
      let stderr = '';
      child.stdout.setEncoding('utf8').on('data', (chunk: string) => (stdout += chunk));
      child.stderr.setEncoding('utf8').on('data', (chunk: string) => (stderr += chunk));
      child.on('error', reject);
      // `killed` is set only by the timeout. uv forwards the signal and exits
      // 143 rather than dying of it, and a grandchild (uv's python) may still
      // hold the pipes, so a timeout settles on `exit` without waiting for `close`.
      child.on('exit', () => {
        if (!child.killed) return;
        child.stdout.destroy();
        child.stderr.destroy();
        reject(new Error(`Laya timed out after ${timeoutMs / 1000}s`));
      });
      child.on('close', (code, signal) => {
        if (child.killed) return;
        if (signal) {
          reject(new Error(`Laya was killed by ${signal}`));
          return;
        }
        try {
          resolvePromise(parseLayaResponse(code ?? 1, stdout, stderr, argv[argv.length - 1]));
        } catch (error) {
          reject(error);
        }
      });
      child.stdin.end(buildLayaRequest(states, question, this.options));
    });
  }
}
