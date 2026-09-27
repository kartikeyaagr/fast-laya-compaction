import { describe, expect, it } from 'vitest';
import {
  compactSession,
  decisionLog,
  decisionLogLines,
  register,
  resolveHookConfig,
  summarize,
  toSessionMessages,
  type HookRun,
  type HookRunInit,
} from '../hooks/fast-laya.ts';
import {
  applyDecisions,
  collectToolCalls,
  decideCall,
  setupHint,
  type CallProbabilities,
  type Message,
} from '../src/index.js';

type SessionMessage = Message & { handle?: string };

function message(role: Message['role'], text: string, extra: Partial<SessionMessage> = {}): SessionMessage {
  return { role, text, toolUses: [], ...extra };
}

function call(id: string, tool: string, input: Record<string, unknown>, text: string): SessionMessage {
  return message('assistant', '', {
    toolUses: [{ tool_use_id: id, tool, input, text }],
    handle: `h-${id}`,
  });
}

function result(id: string, text: string, isError = false): SessionMessage {
  return message('user', '', { toolResults: [{ tool_use_id: id, text, isError }], handle: `r-${id}` });
}

const fileA = 'export const a = 1;\n'.repeat(50);

function transcript(): SessionMessage[] {
  return [
    message('user', 'Fix the failing test.', { handle: 'h-0' }),
    call('tool-1', 'Read', { file_path: 'src/a.ts' }, fileA),
    result('tool-1', fileA),
    call('tool-2', 'Bash', { command: 'npm test' }, 'FAIL'),
    result('tool-2', 'FAIL b.test.ts: expected 2 to be 3', true),
    message('assistant', 'Fixing now.', { handle: 'h-5' }),
    message('user', 'go ahead', { handle: 'h-6' }),
  ];
}

const KEEP = { keep: 0.9, truncate: 0.05, drop: 0.05 };
const TRUNCATE = { keep: 0.1, truncate: 0.8, drop: 0.1 };

type Call = { argv: readonly string[]; init?: HookRunInit };

/** A fake `$.process.run` answering like the Laya script. */
function layaRun(answer: (id: string) => CallProbabilities, calls: Call[] = []): HookRun {
  return async (argv, init) => {
    calls.push({ argv, init });
    const { states } = JSON.parse(init?.stdin ?? '{}') as { states: { id: string }[] };
    const scores = Object.fromEntries(states.map((s) => [s.id, answer(s.id)]));
    return {
      exitCode: 0,
      stdout: JSON.stringify({ model: 'typed-decisions', device: 'mps', load_ms: 3200, infer_ms: 800, scores }),
      stderr: '',
    };
  };
}

describe('hook config', () => {
  it('lets a fine-tuned checkpoint path replace the published checkpoint', () => {
    expect(resolveHookConfig({ model: 'english', checkpointPath: '/ckpt/laya-jev' }).model).toBe('/ckpt/laya-jev');
    expect(resolveHookConfig({ model: 'english' }).model).toBe('english');
  });

  it('reads userConfig values and falls back to defaults', () => {
    expect(resolveHookConfig({})).toEqual({
      compactAtPercent: 60,
      minReductionRatio: 0.25,
      model: 'typed-decisions',
      uvPath: 'uv',
      timeoutSeconds: 120,
    });
    expect(
      resolveHookConfig({
        keepThreshold: 0.3,
        maxCallStateChars: 1000,
        maxScoredCalls: 50,
        model: 'multilingual',
        uvPath: '/opt/homebrew/bin/uv',
        device: 'cpu',
        timeoutSeconds: 0,
        goal: 'g',
        compactAtPercent: 'no',
        device2: 'ignored',
        checkpointPath: '',
      }),
    ).toEqual({
      keepThreshold: 0.3,
      maxCallStateChars: 1000,
      maxScoredCalls: 50,
      model: 'multilingual',
      uvPath: '/opt/homebrew/bin/uv',
      device: 'cpu',
      timeoutSeconds: 1,
      goal: 'g',
      compactAtPercent: 60,
      minReductionRatio: 0.25,
    });
  });
});

describe('session message mapping', () => {
  it('returns the engine objects for untouched messages and handle-less copies for rebuilt ones', () => {
    const messages = transcript();
    const calls = collectToolCalls(messages, 0);
    const decisions = [
      decideCall(calls[0]!, { probabilities: TRUNCATE }, { keepThreshold: 0.5 }),
      decideCall(calls[1]!, { probabilities: KEEP }, { keepThreshold: 0.5 }),
    ];
    messages[1]!.toolUses[0]!.text = 'x'.repeat(2000);
    messages[2]!.toolResults![0]!.text = 'x'.repeat(2000);
    const out = toSessionMessages(messages, applyDecisions(messages, decisions, calls, 300));
    expect(out).toHaveLength(messages.length);
    expect(out[0]).toBe(messages[0]);
    expect(out[1]?.handle).toBeUndefined();
    expect(out[1]?.toolUses[0]?.text).toMatch(
      new RegExp(`^${'x'.repeat(300)}\\n\\[fast-laya-compaction truncated 1700 chars`),
    );
    expect(out[2]?.handle).toBeUndefined();
    expect(out[2]?.toolResults?.[0]?.text).toMatch(
      new RegExp(`^${'x'.repeat(300)}\\n\\[fast-laya-compaction truncated 1700 chars`),
    );
    expect(out[2]?.toolResults?.[0]).toMatchObject({ tool_use_id: 'tool-1', isError: false });
    expect(out[3]).toBe(messages[3]);
    expect(out[4]).toBe(messages[4]);
  });

  it('preserves short dropped-result messages and their handles', () => {
    const messages = transcript();
    messages[1]!.toolUses[0]!.text = 'y'.repeat(100);
    messages[2]!.toolResults![0]!.text = 'y'.repeat(100);
    const calls = collectToolCalls(messages, 0);
    const decisions = [
      decideCall(calls[0]!, { probabilities: TRUNCATE }, { keepThreshold: 0.5 }),
      decideCall(calls[1]!, { probabilities: KEEP }, { keepThreshold: 0.5 }),
    ];
    const out = toSessionMessages(messages, applyDecisions(messages, decisions, calls, 300));
    expect(out[1]).toBe(messages[1]);
    expect(out[2]).toBe(messages[2]);
  });
});

describe('compactSession', () => {
  it('runs one Laya process over the engine and reports the outcome', async () => {
    const calls: Call[] = [];
    const config = { ...resolveHookConfig({ preserveRecentMessages: 1, timeoutSeconds: 30 }), device: 'cpu' };
    const { result: output, messages } = await compactSession(
      transcript(),
      config,
      layaRun((id) => (id === 't2' ? KEEP : TRUNCATE), calls),
      '/plugins/fast-laya',
    );
    expect(calls).toHaveLength(1);
    expect(calls[0]!.argv).toEqual([
      'uv',
      'run',
      '--quiet',
      '--offline',
      '--script',
      '/plugins/fast-laya/backend/laya_compact.py',
    ]);
    expect(calls[0]!.init?.timeoutMs).toBe(30_000);
    const request = JSON.parse(calls[0]!.init?.stdin ?? '{}');
    expect(request).toMatchObject({ model: 'typed-decisions', device: 'cpu', question: { type: 'choice' } });
    expect(request.states.map((s: { id: string }) => s.id)).toEqual(['t1', 't2']);
    expect(output.decisions.map((d) => d.action)).toEqual(['drop_result', 'keep']);
    expect(messages.map((m) => m.handle)).toEqual(['h-0', undefined, undefined, 'h-tool-2', 'r-tool-2', 'h-5', 'h-6']);
    expect(summarize(output)).toMatch(
      /^\d+% reduction; 1 kept, 1 results truncated; typed-decisions on mps, load 3\.2s \+ infer 0\.8s$/,
    );
    expect(decisionLog(output)).toBe('t1:Read:drop_result/keep=0.10/truncate=0.80 t2:Bash:keep/keep=0.90/truncate=0.05');
    expect(decisionLogLines(output)).toEqual([`decisions: ${decisionLog(output)}`]);
  });

  it('splits a long decision log into ui.log lines under the host limit', async () => {
    const config = resolveHookConfig({ preserveRecentMessages: 1 });
    const { result: output } = await compactSession(transcript(), config, layaRun(() => TRUNCATE), '/p');
    const lines = decisionLogLines(output, 60);
    expect(lines).toEqual([
      'decisions (1/2): t1:Read:drop_result/keep=0.10/truncate=0.80',
      'decisions (2/2): t2:Bash:drop_result/keep=0.10/truncate=0.80',
    ]);
    expect(lines.every((line) => line.length <= 60)).toBe(true);
    expect(decisionLogLines({ ...output, decisions: [] })).toEqual(['decisions: (none)']);
  });

  it('removes superseded calls without asking Laya and logs them', async () => {
    const calls: Call[] = [];
    const messages = [
      ...transcript().slice(0, 5),
      call('tool-3', 'Bash', { command: 'npm test' }, 'PASS'),
      result('tool-3', 'PASS'),
      ...transcript().slice(5),
    ];
    const config = resolveHookConfig({ preserveRecentMessages: 1 });
    const { result: output } = await compactSession(messages, config, layaRun(() => KEEP, calls), '/p');
    const request = JSON.parse(calls[0]!.init?.stdin ?? '{}');
    expect(request.states.map((s: { id: string }) => s.id)).toEqual(['t1', 't3']);
    expect(decisionLog(output)).toBe(
      't1:Read:keep/keep=0.90/truncate=0.05 t2:Bash:drop_call/superseded t3:Bash:keep/keep=0.90/truncate=0.05',
    );
    expect(summarize(output)).toMatch(/; 2 kept, 1 superseded calls removed;/);
  });

  it('throws on a missing setup, a timeout, a failed start and a crash so the hook falls back', async () => {
    const config = resolveHookConfig({ preserveRecentMessages: 1 });
    const exit = (exitCode: number, stderr: string): HookRun => async () => ({ exitCode, stdout: '', stderr });
    const reject: HookRun = async () => {
      throw new Error('spawn uv ENOENT');
    };
    await expect(compactSession(transcript(), config, exit(3, ''), '/p')).rejects.toThrow(
      'Laya is not set up, run once: uv run --script /p/backend/laya_compact.py --warmup',
    );
    await expect(compactSession(transcript(), config, exit(1, 'boom'), '/p')).rejects.toThrow(
      'Laya exited with 1: boom',
    );
    await expect(compactSession(transcript(), config, reject, '/p')).rejects.toThrow(
      'Laya could not start (spawn uv ENOENT)',
    );
    await expect(
      compactSession(transcript(), { ...config, timeoutSeconds: 1 }, reject, '/p'),
    ).rejects.toThrow('Laya timed out after 1s');
  });
});

describe('session.compact hook', () => {
  type Handler = (...args: unknown[]) => Promise<unknown>;

  function engine(run: HookRun) {
    const handlers = new Map<string, Handler>();
    register(((event: string, handler: Handler) => handlers.set(event, handler)) as never, {
      preserveRecentMessages: 1,
    });
    const toasts: string[] = [];
    const $ = {
      plugin: { name: 'fast-laya-compaction', root: '/p' },
      process: { run },
      ui: { log: () => undefined, toast: (text: string) => toasts.push(text) },
    };
    const next = async () => ({ messages: [], summarized: true });
    const compactEvent = () => handlers.get('session.compact')!($, { trigger: 'manual', messages: transcript() }, next);
    return { compactEvent, toasts };
  }

  it('replaces the history on success and falls back to the built-in summary on failure', async () => {
    const ok = engine(layaRun(() => TRUNCATE));
    const replaced = (await ok.compactEvent()) as { messages: unknown[]; summarized?: boolean };
    expect(replaced.summarized).toBeUndefined();
    expect(replaced.messages).toHaveLength(7);
    expect(ok.toasts[0]).toMatch(/^kept 7\/7 messages, no summary \(\d+% reduction; 2 results truncated/);

    const broken = engine(async () => ({ exitCode: 3, stdout: '', stderr: '' }));
    expect(await broken.compactEvent()).toMatchObject({ summarized: true });
    expect(broken.toasts).toEqual([`fallback to built-in summary (${setupHint('/p/backend/laya_compact.py')})`]);
  });

  it('runs one Laya process at a time; a compaction arriving meanwhile falls back', async () => {
    let release!: () => void;
    const gate = new Promise<void>((resolve) => (release = resolve));
    const fast = layaRun(() => TRUNCATE);
    const { compactEvent, toasts } = engine(async (argv, init) => {
      await gate;
      return fast(argv, init);
    });
    const first = compactEvent();
    expect(await compactEvent()).toMatchObject({ summarized: true });
    expect(toasts).toEqual(['fallback to built-in summary (another Laya compaction is running)']);
    release();
    expect(await first).not.toHaveProperty('summarized');
    expect(await compactEvent()).not.toHaveProperty('summarized');
  });
});
