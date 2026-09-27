import type {
  On,
  PluginOptions,
  Register,
  SessionMessage,
  ToolResultSummary,
  ToolUseSummary,
  TurnCompleteInput,
} from 'claude-code';

import { compact, reductionRatio, resolveOptions } from '../src/compact.js';
import { buildLayaRequest, DEFAULT_MODEL, layaArgv, parseLayaResponse } from '../src/laya.js';
import type {
  CompactOptions,
  CompactResult,
  LayaScorer,
  Message,
  ToolResult,
  ToolUse,
} from '../src/types.js';

const HOOK_DEFAULTS = {
  compactAtPercent: 60,
  minReductionRatio: 0.25,
  model: DEFAULT_MODEL,
  uvPath: 'uv',
  timeoutSeconds: 120,
};

export type HookRunInit = {
  stdin?: string;
  timeoutMs?: number;
};

export type HookRunResult = {
  exitCode: number;
  stdout: string;
  stderr: string;
};

/** The shape of `$.process.run`, so the hook can be driven without an engine. */
export type HookRun = (argv: readonly string[], init?: HookRunInit) => Promise<HookRunResult>;

export type HookConfig = CompactOptions & {
  compactAtPercent: number;
  minReductionRatio: number;
  model: string;
  uvPath: string;
  device?: string;
  timeoutSeconds: number;
};

function optionNumber(options: PluginOptions, key: string, fallback: number): number {
  const value = options[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function optionString(options: PluginOptions, key: string): string | undefined {
  const value = options[key];
  return typeof value === 'string' && value.length > 0 ? value : undefined;
}

/** Reads the plugin's `userConfig` values; anything missing takes the defaults. */
export function resolveHookConfig(options: PluginOptions): HookConfig {
  const numbers: Partial<Omit<CompactOptions, 'goal'>> = {};
  for (const key of [
    'keepThreshold',
    'preserveRecentMessages',
    'maxCallStateChars',
    'maxScoredCalls',
    'truncateHeadChars',
  ] as const) {
    const value = options[key];
    if (typeof value === 'number' && Number.isFinite(value)) numbers[key] = value;
  }
  const config: HookConfig = {
    ...numbers,
    compactAtPercent: optionNumber(options, 'compactAtPercent', HOOK_DEFAULTS.compactAtPercent),
    minReductionRatio: optionNumber(
      options,
      'minReductionRatio',
      HOOK_DEFAULTS.minReductionRatio,
    ),
    // A fine-tuned checkpoint directory, when set, replaces the published checkpoint.
    model: optionString(options, 'checkpointPath') ?? optionString(options, 'model') ?? HOOK_DEFAULTS.model,
    uvPath: optionString(options, 'uvPath') ?? HOOK_DEFAULTS.uvPath,
    timeoutSeconds: Math.max(
      1,
      optionNumber(options, 'timeoutSeconds', HOOK_DEFAULTS.timeoutSeconds),
    ),
  };
  const device = optionString(options, 'device');
  if (device) config.device = device;
  const goal = optionString(options, 'goal');
  if (goal) config.goal = goal;
  return config;
}

/**
 * A `LayaScorer` over the engine's `$.process.run`: one short-lived Laya
 * process per compaction, the request on stdin, the scores on stdout.
 * `$.process.run` rejects both when the command cannot start and when it
 * outlives the timeout; the elapsed time tells the two apart.
 */
export function processScorer(
  run: HookRun,
  argv: readonly string[],
  config: Pick<HookConfig, 'model' | 'device' | 'timeoutSeconds'>,
): LayaScorer {
  return {
    async score(states, question) {
      const timeoutMs = config.timeoutSeconds * 1000;
      const stdin = buildLayaRequest(states, question, { model: config.model, device: config.device });
      const started = Date.now();
      let result: HookRunResult;
      try {
        result = await run(argv, { stdin, timeoutMs });
      } catch (error) {
        if (Date.now() - started >= timeoutMs - 1000) {
          throw new Error(`Laya timed out after ${config.timeoutSeconds}s`);
        }
        throw new Error(`Laya could not start (${error instanceof Error ? error.message : String(error)})`);
      }
      return parseLayaResponse(result.exitCode, result.stdout, result.stderr, argv[argv.length - 1]);
    },
  };
}

function toolUseSummary(tool: ToolUse): ToolUseSummary {
  const summary: ToolUseSummary = {
    tool_use_id: tool.tool_use_id,
    tool: tool.tool,
    input: tool.input,
  };
  if (tool.text !== undefined) summary.text = tool.text;
  if (tool.isError) summary.isError = true;
  return summary;
}

function toolResultSummary(result: ToolResult): ToolResultSummary {
  return {
    tool_use_id: result.tool_use_id,
    text: result.text,
    isError: result.isError ?? false,
  };
}

/**
 * Maps the library's output back onto session messages. Whatever came back
 * unchanged (a message, a tool use, a tool result) is the engine's own object,
 * handle included; anything rebuilt is a fresh message without a handle, so the
 * engine takes the edited content instead of its original.
 */
export function toSessionMessages(
  input: readonly SessionMessage[],
  output: readonly Message[],
): SessionMessage[] {
  const messages = new Map<Message, SessionMessage>();
  const uses = new Map<ToolUse, ToolUseSummary>();
  const results = new Map<ToolResult, ToolResultSummary>();
  for (const message of input) {
    messages.set(message, message);
    for (const tool of message.toolUses) uses.set(tool, tool);
    for (const result of message.toolResults ?? []) results.set(result, result);
  }
  return output.map((message) => {
    const own = messages.get(message);
    if (own) return own;
    const rebuilt: SessionMessage = {
      role: message.role,
      text: message.text,
      toolUses: message.toolUses.map((tool) => uses.get(tool) ?? toolUseSummary(tool)),
    };
    if (message.toolResults && message.toolResults.length > 0) {
      rebuilt.toolResults = message.toolResults.map(
        (result) => results.get(result) ?? toolResultSummary(result),
      );
    }
    return rebuilt;
  });
}

export type SessionCompaction = {
  result: CompactResult;
  messages: SessionMessage[];
};

/** Runs the library over a session transcript; throws when Laya cannot run or fails. */
export async function compactSession(
  messages: readonly SessionMessage[],
  config: HookConfig,
  run: HookRun,
  pluginRoot: string,
): Promise<SessionCompaction> {
  const scorer = processScorer(run, layaArgv(config.uvPath, pluginRoot), config);
  const result = await compact(messages, scorer, config);
  return { result, messages: toSessionMessages(messages, result.messages) };
}

function percent(ratio: number): string {
  return `${Math.round(ratio * 100)}%`;
}

function seconds(ms: number): string {
  return `${(ms / 1000).toFixed(1)}s`;
}

export function summarize(result: CompactResult): string {
  const { stats } = result;
  const parts = [
    stats.kept > 0 ? `${stats.kept} kept` : '',
    stats.resultsDropped > 0 ? `${stats.resultsDropped} results truncated` : '',
    stats.callsDropped > 0 ? `${stats.callsDropped} call_dropped` : '',
    stats.superseded > 0 ? `${stats.superseded} superseded calls removed` : '',
    stats.pinned > 0 ? `${stats.pinned} pinned` : '',
    stats.unscored > 0 ? `${stats.unscored} unscored` : '',
  ].filter(Boolean);
  const laya = stats.model
    ? `${stats.model} on ${stats.device}, load ${seconds(stats.loadMs)} + infer ${seconds(stats.inferMs)}`
    : 'Laya not called';
  return `${percent(reductionRatio(result))} reduction; ${parts.join(', ') || 'no tool calls'}; ${laya}`;
}

const UI_LOG_MAX_CHARS = 4096;

export function decisionLog(result: CompactResult): string {
  return result.decisions
    .filter((d) => d.reason !== 'pinned' && d.reason !== 'unscored')
    .map((d) => {
      const p = d.probabilities;
      return `${d.id}:${d.tool}:${d.action}/${p ? `keep=${p.keep.toFixed(2)}/truncate=${p.truncate.toFixed(2)}` : d.reason}`;
    })
    .join(' ');
}

export function decisionLogLines(
  result: CompactResult,
  maxChars: number = UI_LOG_MAX_CHARS,
): string[] {
  const entries = decisionLog(result).split(' ').filter(Boolean);
  if (entries.length === 0) return ['decisions: (none)'];
  const chunks: string[] = [];
  let current = '';
  for (const entry of entries) {
    const next = current ? `${current} ${entry}` : entry;
    if (current && next.length > maxChars - 24) {
      chunks.push(current);
      current = entry;
    } else current = next;
  }
  chunks.push(current);
  return chunks.map((chunk, index) =>
    chunks.length === 1
      ? `decisions: ${chunk}`
      : `decisions (${index + 1}/${chunks.length}): ${chunk}`,
  );
}

function notify(
  $: {
    ui: {
      log: (text: string) => void;
      toast: (text: string, options?: { timeoutMs?: number }) => void;
    };
  },
  text: string,
): void {
  $.ui.log(text);
  $.ui.toast(text, { timeoutMs: 15_000 });
}

export const register: Register = (on: On, options: PluginOptions) => {
  const config = resolveHookConfig(options);
  let compacting = false;
  // One Laya process at a time: each holds a checkpoint (~2-3 GB), so a
  // subagent's or an ahead-of-time compaction arriving meanwhile falls back.
  let scoring = false;

  on('session.compact', async ($, event, next) => {
    if (scoring) {
      notify($, 'fallback to built-in summary (another Laya compaction is running)');
      return next(event);
    }
    scoring = true;
    try {
      const { result, messages } = await compactSession(
        event.messages,
        config,
        (argv, init) => $.process.run(argv, init),
        $.plugin.root,
      );
      for (const line of decisionLogLines(result)) $.ui.log(line);
      if (reductionRatio(result) < config.minReductionRatio) {
        notify(
          $,
          `fallback to built-in summary (below ${percent(config.minReductionRatio)} minimum: ${summarize(result)})`,
        );
        return next(event);
      }
      notify(
        $,
        `kept ${messages.length}/${event.messages.length} messages, no summary (${summarize(result)})`,
      );
      return { messages };
    } catch (error) {
      notify(
        $,
        `fallback to built-in summary (${error instanceof Error ? error.message : String(error)})`,
      );
      return next(event);
    } finally {
      scoring = false;
    }
  });

  on('turn.complete', async ($, event: TurnCompleteInput, next) => {
    if (compacting) return next(event);
    try {
      const { context } = await $.session.usage();
      if ((context.percent ?? 0) < config.compactAtPercent) return next(event);
      compacting = true;
      await $.session.compact();
    } catch (error) {
      $.ui.log(
        `auto-compact skipped (${error instanceof Error ? error.message : String(error)})`,
      );
    } finally {
      compacting = false;
    }
    return next(event);
  });
};

export { resolveOptions };
