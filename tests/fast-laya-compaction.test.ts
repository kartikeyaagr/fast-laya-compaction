import { chmodSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  applyDecisions,
  buildLayaRequest,
  CALL_QUESTION,
  callLine,
  callState,
  callTarget,
  collectToolCalls,
  compact,
  compactMessages,
  decideCall,
  callProbabilities,
  goalFromMessages,
  layaArgv,
  maxReduction,
  NodeLayaScorer,
  parseLayaResponse,
  reductionRatio,
  resolveOptions,
  SETUP_HINT,
  supersededBy,
  type CallProbabilities,
  type CallState,
  type LayaScorer,
  type Message,
} from '../src/index.js';

function message(role: Message['role'], text: string, extra: Partial<Message> = {}): Message {
  return { role, text, toolUses: [], ...extra };
}

function call(id: string, tool: string, input: Record<string, unknown>, text: string): Message {
  return message('assistant', '', { toolUses: [{ tool_use_id: id, tool, input, text }] });
}

function result(id: string, text: string, isError = false): Message {
  return message('user', '', { toolResults: [{ tool_use_id: id, text, isError }] });
}

const fileA = 'export const a = 1;\n'.repeat(50);
const fileB = 'export const b = 2;\n'.repeat(50);

function transcript(): Message[] {
  return [
    message('user', 'Never edit anything under src/generated. Fix the failing test.'),
    call('tool-1', 'Read', { file_path: 'src/a.ts' }, fileA),
    result('tool-1', fileA),
    message('assistant', 'a.ts looks fine; checking b.ts'),
    call('tool-2', 'Read', { file_path: 'src/b.ts' }, fileB),
    result('tool-2', fileB),
    call('tool-3', 'Bash', { command: 'npm test' }, 'FAIL b.test.ts'),
    result('tool-3', 'FAIL b.test.ts: expected 2 to be 3', true),
    message('assistant', 'The failure is in b.test.ts; fixing now.'),
    message('user', 'go ahead'),
  ];
}

function fakeScorer(answer: (id: string) => CallProbabilities, seen: CallState[][] = []): LayaScorer {
  return {
    async score(states, question) {
      expect(question).toBe(CALL_QUESTION);
      seen.push([...states]);
      return {
        model: 'typed-decisions',
        device: 'cpu',
        loadMs: 10,
        inferMs: 20,
        scores: Object.fromEntries(states.map((s) => [s.id, answer(s.id)])),
      };
    },
  };
}

const KEEP = { keep: 0.8, truncate: 0.15, drop: 0.05 };
const TRUNCATE = { keep: 0.1, truncate: 0.7, drop: 0.2 };
const DROP = { keep: 0.05, truncate: 0.15, drop: 0.8 };

describe('options', () => {
  it('fills in defaults and ignores non-finite values', () => {
    expect(resolveOptions()).toMatchObject({
      keepThreshold: 0.5,
      preserveRecentMessages: 6,
      maxCallStateChars: 1600,
      maxScoredCalls: 80,
      truncateHeadChars: 300,
      targetReduction: 0,
    });
    expect(resolveOptions({
      keepThreshold: Number.NaN,
      preserveRecentMessages: 2.7,
      truncateHeadChars: -1.2,
      maxCallStateChars: 10,
    })).toMatchObject({
      keepThreshold: 0.5,
      preserveRecentMessages: 2,
      truncateHeadChars: 0,
      maxCallStateChars: 200,
    });
  });
});

describe('tool call collection', () => {
  it('pairs each tool call with its result and pins recent ones', () => {
    const calls = collectToolCalls(transcript(), 3);
    expect(calls.map((c) => [c.id, c.tool, c.callIndex, c.resultIndex, c.pinned])).toEqual([
      ['t1', 'Read', 1, 2, false],
      ['t2', 'Read', 4, 5, false],
      ['t3', 'Bash', 6, 7, true],
    ]);
    expect(calls[2]?.isError).toBe(true);
    expect(calls[0]?.resultChars).toBe(fileA.length);
  });

  it('ignores calls without a result', () => {
    expect(collectToolCalls([message('user', 'hi'), call('x', 'Read', {}, '')], 0)).toHaveLength(0);
  });
});

describe('call state', () => {
  function session(): Message[] {
    return [
      message('user', 'Fix the failing test in b.ts.'),
      message('assistant', 'Reading a.ts first.'),
      call('r1', 'Read', { file_path: 'src/a.ts' }, fileA),
      result('r1', fileA),
      call('g1', 'Grep', { pattern: 'TODO', path: 'src' }, 'src/a.ts:1'),
      result('g1', 'src/a.ts:1'),
      call('e1', 'Edit', { file_path: 'src/a.ts', old_string: 'a = 1', new_string: 'a = 2' }, 'ok'),
      result('e1', 'ok'),
      call('r2', 'Read', { file_path: 'src/b.ts' }, fileB),
      result('r2', fileB),
      message('assistant', 'b.ts still exports b; running the tests.'),
      call('b1', 'Bash', { command: 'npm test' }, 'FAIL'),
      result('b1', 'FAIL', true),
      call('b2', 'Bash', { command: 'npm test' }, 'PASS'),
      result('b2', 'PASS'),
    ];
  }

  it('names what a call operates on and renders it on one line', () => {
    expect(callTarget({ input: { file_path: 'src/a.ts', old_string: 'x' } })).toBe('src/a.ts');
    expect(callTarget({ input: { pattern: 'TODO', path: 'src' } })).toBe('src TODO');
    expect(callTarget({ input: { description: 'x' } })).toBeUndefined();
    expect(callLine({ tool: 'Read', input: { file_path: 'src/a.ts' } })).toBe('Read file_path=src/a.ts');
    expect(callLine({ tool: 'Write', input: { content: 'x'.repeat(500) } })).toHaveLength(200);
    expect(callLine({ tool: 'Edit', input: { replace_all: false, old_string: 'a', file_path: 'src/a.ts' } })).toBe(
      'Edit file_path=src/a.ts replace_all=false old_string=a',
    );
  });

  it('puts the decisive facts first, in a fixed order', () => {
    const messages = session();
    const calls = collectToolCalls(messages, 0);
    const state = callState(calls[0]!, messages, calls, 'Fix the failing test in b.ts.', 1600);
    expect(Object.keys(state)).toEqual([
      'tool_call',
      'outcome',
      'messages_since',
      'superseded',
      'later_calls',
      'mentioned_later',
      'goal',
      'intent',
      'output_head',
    ]);
    expect(state).toMatchObject({
      tool_call: 'Read file_path=src/a.ts',
      outcome: `ok, ${fileA.length} chars`,
      messages_since: messages.length - 1 - 3,
      superseded: true,
      later_calls: ['Edit file_path=src/a.ts old_string=a = 1 new_string=a = 2'],
      mentioned_later: false,
      intent: 'Reading a.ts first.',
    });
    expect(String(state.output_head)).toHaveLength(400);
  });

  it('flags later calls on the same target and later mentions by base name', () => {
    const messages = session();
    const calls = collectToolCalls(messages, 0);
    const byTool = (id: string) => calls.find((c) => c.tool_use_id === id)!;
    const b1 = callState(byTool('b1'), messages, calls, '', 1600);
    expect(b1).toMatchObject({ superseded: true, later_calls: ['Bash command=npm test'], outcome: 'error, 4 chars' });
    const r2 = callState(byTool('r2'), messages, calls, '', 1600);
    expect(r2).toMatchObject({ superseded: false, mentioned_later: true, intent: 'Reading a.ts first.' });
    expect(r2).not.toHaveProperty('later_calls');
    expect(r2).not.toHaveProperty('goal');
    expect(callState(byTool('b2'), messages, calls, '', 1600)).toMatchObject({ superseded: false });
  });

  it('trims the least decisive fields first to stay under the cap', () => {
    const messages = session();
    const calls = collectToolCalls(messages, 0);
    const goal = 'g'.repeat(300);
    const full = callState(calls[0]!, messages, calls, goal, 10_000);
    expect(JSON.stringify(full).length).toBeGreaterThan(900);

    const capped = callState(calls[0]!, messages, calls, goal, 700);
    expect(JSON.stringify(capped).length).toBeLessThanOrEqual(700);
    expect(capped).toMatchObject({ tool_call: 'Read file_path=src/a.ts', goal });
    expect(String(capped.output_head).length).toBeLessThan(400);

    const shorter = callState(calls[0]!, messages, calls, goal, 300);
    expect(JSON.stringify(shorter).length).toBeLessThanOrEqual(300);
    expect(shorter).not.toHaveProperty('output_head');
    expect(shorter).not.toHaveProperty('intent');
    expect(String(shorter.goal).length).toBeLessThan(300);

    const tight = callState(calls[0]!, messages, calls, goal, 120);
    expect(Object.keys(tight)).toEqual(['tool_call', 'outcome', 'messages_since', 'superseded', 'mentioned_later']);
  });

  it('supersedes a read-only call when a later call repeats it or changes its target', () => {
    const messages = session();
    const calls = collectToolCalls(messages, 0);
    const byTool = (id: string) => calls.find((c) => c.tool_use_id === id)!;
    expect(supersededBy(byTool('r1'), calls)?.tool_use_id).toBe('e1');
    expect(supersededBy(byTool('b1'), calls)?.tool_use_id).toBe('b2');
    for (const id of ['g1', 'e1', 'r2', 'b2']) expect(supersededBy(byTool(id), calls)).toBeUndefined();
  });

  it('never supersedes a change, a different read of the same file, or a call without a target', () => {
    const messages = [
      message('user', 'go'),
      call('e1', 'Edit', { file_path: 'a.ts', old_string: '1', new_string: '2' }, 'ok'),
      result('e1', 'ok'),
      call('e2', 'Edit', { file_path: 'a.ts', old_string: '2', new_string: '3' }, 'ok'),
      result('e2', 'ok'),
      call('r1', 'Read', { file_path: 'b.ts', offset: 1, limit: 50 }, 'x'),
      result('r1', 'x'),
      call('r2', 'Read', { file_path: 'b.ts', offset: 50, limit: 50 }, 'y'),
      result('r2', 'y'),
      call('t1', 'TodoWrite', { todos: [] }, 'ok'),
      result('t1', 'ok'),
      call('t2', 'TodoWrite', { todos: [] }, 'ok'),
      result('t2', 'ok'),
      call('b1', 'Bash', { command: 'npm test', description: 'Run tests' }, 'FAIL'),
      result('b1', 'FAIL'),
      call('b2', 'Bash', { command: 'npm test', description: 'Re-run the tests' }, 'PASS'),
      result('b2', 'PASS'),
    ];
    const calls = collectToolCalls(messages, 0);
    const byTool = (id: string) => calls.find((c) => c.tool_use_id === id)!;
    for (const id of ['e1', 'r1', 't1']) expect(supersededBy(byTool(id), calls)).toBeUndefined();
    expect(supersededBy(byTool('b1'), calls)?.tool_use_id).toBe('b2');
  });

  it('defaults the goal to the latest user prompt', () => {
    expect(goalFromMessages(transcript())).toBe('go ahead');
    expect(goalFromMessages(transcript(), 2)).toBe(
      'Never edit anything under src/generated. Fix the failing test.\ngo ahead',
    );
  });
});

describe('decisions', () => {
  const options = { keepThreshold: 0.5 };
  const unpinned = { id: 't1', tool: 'Read', pinned: false };

  it('keeps, truncates or drops per Laya, removes superseded calls and leaves the rest alone', () => {
    expect(decideCall(unpinned, { probabilities: KEEP }, options)).toMatchObject({
      action: 'keep',
      reason: 'kept',
      probabilities: KEEP,
    });
    expect(decideCall(unpinned, { probabilities: TRUNCATE }, options)).toMatchObject({
      action: 'drop_result',
      reason: 'result_dropped',
    });
    expect(decideCall(unpinned, { probabilities: DROP }, options)).toMatchObject({ action: 'drop_call', reason: 'call_dropped' });
    expect(decideCall(unpinned, { superseded: true }, options)).toMatchObject({ action: 'drop_call', reason: 'superseded' });
    expect(decideCall(unpinned, {}, options)).toMatchObject({ action: 'keep', reason: 'unscored' });
    expect(decideCall({ ...unpinned, pinned: true }, { superseded: true, probabilities: DROP }, options)).toMatchObject({
      action: 'keep',
      reason: 'pinned',
    });
  });

  it('removes dropped calls and truncates dropped results', () => {
    const messages = transcript();
    messages[4]!.toolUses[0]!.text = 'x'.repeat(2000);
    messages[5]!.toolResults![0]!.text = 'x'.repeat(2000);
    const calls = collectToolCalls(messages, 0);
    const decisions = [
      decideCall(calls[0]!, { superseded: true }, options),
      decideCall(calls[1]!, { probabilities: TRUNCATE }, options),
      decideCall(calls[2]!, { probabilities: KEEP }, options),
    ];
    const kept = applyDecisions(messages, decisions, calls, 300);

    expect(kept.map((m) => m.text || m.toolUses[0]?.tool_use_id || m.toolResults?.[0]?.tool_use_id)).toEqual([
      'Never edit anything under src/generated. Fix the failing test.',
      'a.ts looks fine; checking b.ts',
      'tool-2',
      'tool-2',
      'tool-3',
      'tool-3',
      'The failure is in b.test.ts; fixing now.',
      'go ahead',
    ]);
    expect(kept[0]).toBe(messages[0]);
    expect(kept[2]).not.toBe(messages[4]);
    expect(kept[2]?.toolUses[0]?.text).toMatch(
      new RegExp(`^${'x'.repeat(300)}\\n\\[fast-laya-compaction truncated 1700 chars`),
    );
    expect(kept[3]?.toolResults?.[0]?.text).toMatch(
      new RegExp(`^${'x'.repeat(300)}\\n\\[fast-laya-compaction truncated 1700 chars`),
    );
    expect(kept[2]).not.toBe(messages[4]);
    expect(kept[3]).not.toBe(messages[5]);
    expect(kept[4]).toBe(messages[6]);
    expect(kept[5]?.toolResults?.[0]?.text).toContain('expected 2 to be 3');

    const shortMessages = transcript();
    shortMessages[4]!.toolUses[0]!.text = 'y'.repeat(100);
    shortMessages[5]!.toolResults![0]!.text = 'y'.repeat(100);
    const shortKept = applyDecisions(shortMessages, decisions, calls, 300);
    expect(shortKept[2]).toBe(shortMessages[4]);
    expect(shortKept[3]).toBe(shortMessages[5]);
  });

  it('honours truncateHeadChars, including a zero head', () => {
    const messages = transcript();
    const calls = collectToolCalls(messages, 0);
    const decisions = [decideCall(calls[0]!, { probabilities: TRUNCATE }, { keepThreshold: 0.5 })];
    const original = messages[2]!.toolResults![0]!.text;
    const total = original.length;

    const kept = applyDecisions(messages, decisions, calls, 50);
    expect(kept[2]?.toolResults?.[0]?.text).toBe(
      `${original.slice(0, 50)}\n[fast-laya-compaction truncated ${total - 50} chars of this tool result; re-run the tool if needed]`,
    );
    expect(kept[1]?.toolUses[0]?.text).toBe(kept[2]?.toolResults?.[0]?.text);

    const noHead = applyDecisions(messages, decisions, calls, 0);
    expect(noHead[2]?.toolResults?.[0]?.text).toBe(
      `[fast-laya-compaction truncated ${total} chars of this tool result; re-run the tool if needed]`,
    );
  });
});

describe('compact', () => {
  it('scores every candidate in one request, one state per call', async () => {
    const seen: CallState[][] = [];
    const messages = transcript();
    const output = await compact(messages, fakeScorer(() => TRUNCATE, seen), { preserveRecentMessages: 1 });

    expect(seen).toHaveLength(1);
    expect(seen[0]!.map((s) => s.id)).toEqual(['t1', 't2', 't3']);
    expect(seen[0]![0]!.state).toMatchObject({ tool_call: 'Read file_path=src/a.ts', goal: 'go ahead' });
    expect(output.decisions.map((d) => d.action)).toEqual(['drop_result', 'drop_result', 'drop_result']);
    expect(output.messages).toHaveLength(messages.length);
    expect(output.stats).toMatchObject({
      resultsDropped: 3,
      kept: 0,
      callsDropped: 0,
      superseded: 0,
      pinned: 0,
      model: 'typed-decisions',
      device: 'cpu',
      loadMs: 10,
      inferMs: 20,
    });
    expect(reductionRatio(output)).toBeGreaterThan(0);
  });

  it('drops, truncates or keeps each call per its probabilities', async () => {
    const answers: Record<string, CallProbabilities> = { t1: DROP, t2: TRUNCATE, t3: KEEP };
    const output = await compact(transcript(), fakeScorer((id) => answers[id]!), { preserveRecentMessages: 1 });
    expect(output.decisions.map((d) => d.action)).toEqual(['drop_call', 'drop_result', 'keep']);
    expect(output.decisions.map((d) => d.probabilities)).toEqual([DROP, TRUNCATE, KEEP]);
    expect(output.stats).toMatchObject({ callsDropped: 1, resultsDropped: 1, kept: 1 });
  });

  it('removes superseded calls with their results without asking Laya', async () => {
    const seen: CallState[][] = [];
    const messages = [
      message('user', 'Fix the test.'),
      call('r1', 'Read', { file_path: 'src/a.ts' }, fileA),
      result('r1', fileA),
      call('b1', 'Bash', { command: 'npm test' }, 'FAIL'),
      result('b1', 'FAIL', true),
      call('e1', 'Edit', { file_path: 'src/a.ts', old_string: 'a = 1', new_string: 'a = 2' }, 'ok'),
      result('e1', 'ok'),
      call('b2', 'Bash', { command: 'npm test' }, 'PASS'),
      result('b2', 'PASS'),
      message('assistant', 'Fixed.'),
    ];
    const output = await compact(messages, fakeScorer(() => KEEP, seen), { preserveRecentMessages: 3 });
    expect(seen[0]!.map((s) => s.id)).toEqual(['t3']);
    expect(output.decisions.map((d) => [d.id, d.reason])).toEqual([
      ['t1', 'superseded'],
      ['t2', 'superseded'],
      ['t3', 'kept'],
      ['t4', 'pinned'],
    ]);
    expect(output.messages.map((m) => m.text || m.toolUses[0]?.tool_use_id || m.toolResults?.[0]?.tool_use_id)).toEqual([
      'Fix the test.',
      'e1',
      'e1',
      'b2',
      'b2',
      'Fixed.',
    ]);
    expect(output.stats).toMatchObject({ superseded: 2, kept: 1, pinned: 1 });
  });

  it('does not start Laya when every candidate is superseded', async () => {
    const seen: CallState[][] = [];
    const messages = [
      message('user', 'go'),
      call('b1', 'Bash', { command: 'ls' }, 'a'),
      result('b1', 'a'),
      call('b2', 'Bash', { command: 'ls' }, 'a b'),
      result('b2', 'a b'),
    ];
    const output = await compact(messages, fakeScorer(() => KEEP, seen), { preserveRecentMessages: 2 });
    expect(seen).toHaveLength(0);
    expect(output.stats).toMatchObject({ superseded: 1, model: '' });
  });

  it('scores at most maxScoredCalls, oldest first, and keeps the rest', async () => {
    const seen: CallState[][] = [];
    const output = await compact(transcript(), fakeScorer(() => TRUNCATE, seen), {
      preserveRecentMessages: 1,
      maxScoredCalls: 2,
    });
    expect(seen[0]!.map((s) => s.id)).toEqual(['t1', 't2']);
    expect(output.decisions.map((d) => d.action)).toEqual(['drop_result', 'drop_result', 'keep']);
    expect(output.decisions[2]?.reason).toBe('unscored');
    expect(output.stats).toMatchObject({ calls: 3, resultsDropped: 2, kept: 0, unscored: 1 });
  });

  it('trims outputs Laya would keep, lowest P(keep) first, until targetReduction is met', async () => {
    const messages = transcript();
    messages[4]!.toolUses[0]!.text = 'y'.repeat(1000);
    messages[5]!.toolResults![0]!.text = 'y'.repeat(1000);
    const answers: Record<string, CallProbabilities> = {
      t1: { keep: 0.7, truncate: 0.2, drop: 0.1 },
      t2: { keep: 0.55, truncate: 0.3, drop: 0.15 },
    };
    const options = { preserveRecentMessages: 4 };
    const none = await compact(messages, fakeScorer((id) => answers[id]!), options);
    expect(none.decisions.map((d) => d.reason)).toEqual(['kept', 'kept', 'pinned']);

    const some = await compact(messages, fakeScorer((id) => answers[id]!), { ...options, targetReduction: 0.2 });
    expect(some.decisions.map((d) => d.reason)).toEqual(['kept', 'trimmed', 'pinned']);
    expect(some.stats.trimmed).toBe(1);
    expect(reductionRatio(some)).toBeGreaterThanOrEqual(0.2);

    const all = await compact(messages, fakeScorer((id) => answers[id]!), { ...options, targetReduction: 0.9 });
    expect(all.decisions.map((d) => d.reason)).toEqual(['trimmed', 'trimmed', 'pinned']);
    expect(reductionRatio(all)).toBeCloseTo(maxReduction(messages, options));
  });

  it('bounds the reduction any answer could reach before asking Laya', () => {
    const messages = transcript();
    expect(maxReduction(messages, { preserveRecentMessages: 1 })).toBeGreaterThan(0.5);
    expect(maxReduction(messages, { preserveRecentMessages: 9 })).toBe(0);
    expect(maxReduction([message('user', 'hi')])).toBe(0);
  });

  it('uses the configured goal instead of the latest prompt', async () => {
    const seen: CallState[][] = [];
    await compact(transcript(), fakeScorer(() => KEEP, seen), { preserveRecentMessages: 1, goal: 'ship it' });
    expect(seen[0]![0]!.state.goal).toBe('ship it');
  });

  it('keeps everything without calling Laya when no tool call is a candidate', async () => {
    const seen: CallState[][] = [];
    const messages = [message('user', 'hello'), message('assistant', 'hi')];
    const output = await compact(messages, fakeScorer(() => DROP, seen));
    expect(seen).toHaveLength(0);
    expect(output.stats).toMatchObject({ model: '', device: '', loadMs: 0, calls: 0 });
    expect(output.messages).toEqual(messages);
  });

  it('reports no reduction when Laya wants everything kept', async () => {
    const output = await compact(transcript(), fakeScorer(() => KEEP), { preserveRecentMessages: 1 });
    expect(output.decisions.every((d) => d.action === 'keep')).toBe(true);
    expect(reductionRatio(output)).toBe(0);
  });

  it('rejects missing and malformed answers', async () => {
    const partial: LayaScorer = {
      score: async () => ({ model: 'm', device: 'cpu', loadMs: 0, inferMs: 0, scores: { t1: KEEP } }),
    };
    await expect(compact(transcript(), partial, { preserveRecentMessages: 1 })).rejects.toThrow(
      /Invalid Laya answer for t2/,
    );
    expect(callProbabilities({ t1: { ...KEEP, extra: 1 } }, 't1')).toEqual(KEEP);
    for (const bad of [Number.NaN, 1.5, -0.1]) {
      expect(() => callProbabilities({ t1: { ...KEEP, keep: bad } }, 't1')).toThrow(/Invalid Laya answer for t1/);
    }
    expect(() => callProbabilities({ t1: { keep: 0.5, truncate: 0.5 } }, 't1')).toThrow(/Invalid Laya answer/);
  });
});

describe('Laya process protocol', () => {
  it('builds the command and the stdin request', () => {
    expect(layaArgv('uv', '/p')).toEqual(['uv', 'run', '--quiet', '--offline', '--script', '/p/backend/laya_compact.py']);
    const states = [{ id: 't1', state: { tool_call: 'Read' } }];
    expect(JSON.parse(buildLayaRequest(states, CALL_QUESTION))).toEqual({
      model: 'typed-decisions',
      max_len: 512,
      head_max_len: 128,
      batch_size: 32,
      question: CALL_QUESTION,
      states,
    });
    expect(JSON.parse(buildLayaRequest(states, CALL_QUESTION, { model: 'english', device: 'cpu' }))).toMatchObject({
      model: 'english',
      device: 'cpu',
    });
  });

  it('parses the last stdout line and turns failures into readable errors', () => {
    const answer = '{"model":"typed-decisions","device":"mps","load_ms":3100,"infer_ms":900,"scores":{"t1":{"keep":0.2}}}';
    expect(parseLayaResponse(0, `noise\n${answer}\n`, '')).toEqual({
      model: 'typed-decisions',
      device: 'mps',
      loadMs: 3100,
      inferMs: 900,
      scores: { t1: { keep: 0.2 } },
    });
    expect(() => parseLayaResponse(3, '', 'laya_compact: not cached')).toThrow(SETUP_HINT);
    expect(() =>
      parseLayaResponse(1, '', 'hint: Packages were unavailable because the network was disabled.'),
    ).toThrow(SETUP_HINT);
    expect(() => parseLayaResponse(1, '', 'warning\nlaya_compact: RuntimeError: boom\n')).toThrow(
      'Laya exited with 1: laya_compact: RuntimeError: boom',
    );
    expect(() => parseLayaResponse(0, 'not json', '')).toThrow(/malformed/);
    expect(() => parseLayaResponse(0, '', '')).toThrow(/malformed/);
    expect(() => parseLayaResponse(0, '{"model":"m"}', '')).toThrow(/missing scores/);
  });

  function fakeUv(script: string): string {
    const dir = mkdtempSync(join(tmpdir(), 'fake-uv-'));
    const path = join(dir, 'uv');
    writeFileSync(path, `#!/bin/sh\n${script}\n`);
    chmodSync(path, 0o755);
    return path;
  }

  it('runs the script from Node with the request on stdin', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'stdin-'));
    const uvPath = fakeUv(
      `cat > ${dir}/stdin.json; echo "$@" > ${dir}/argv; echo '{"model":"typed-decisions","device":"cpu","load_ms":1,"infer_ms":2,"scores":{"t1":${JSON.stringify(DROP)},"t2":${JSON.stringify(TRUNCATE)},"t3":${JSON.stringify(KEEP)}}}'`,
    );
    const output = await compactMessages(transcript(), { uvPath, root: '/root', preserveRecentMessages: 1 });
    expect(output.decisions.map((d) => d.action)).toEqual(['drop_call', 'drop_result', 'keep']);
    const { readFileSync } = await import('node:fs');
    expect(readFileSync(join(dir, 'argv'), 'utf8').trim()).toBe(
      'run --quiet --offline --script /root/backend/laya_compact.py',
    );
    expect(JSON.parse(readFileSync(join(dir, 'stdin.json'), 'utf8')).states).toHaveLength(3);
  });

  it('reports a timeout, a missing setup and a missing uv', async () => {
    const states = [{ id: 't1', state: {} }];
    await expect(
      new NodeLayaScorer({ uvPath: fakeUv('sleep 5'), timeoutMs: 200 }).score(states, CALL_QUESTION),
    ).rejects.toThrow('Laya timed out after 0.2s');
    // uv forwards the signal and exits 143 instead of dying of it
    await expect(
      new NodeLayaScorer({ uvPath: fakeUv("trap 'exit 143' TERM; sleep 5 & wait"), timeoutMs: 200 }).score(
        states,
        CALL_QUESTION,
      ),
    ).rejects.toThrow('Laya timed out after 0.2s');
    await expect(
      new NodeLayaScorer({ uvPath: fakeUv('cat >/dev/null; exit 3'), root: '/r' }).score(states, CALL_QUESTION),
    ).rejects.toThrow('run once: uv run --script /r/backend/laya_compact.py --warmup');
    await expect(
      new NodeLayaScorer({ uvPath: '/nonexistent/uv' }).score(states, CALL_QUESTION),
    ).rejects.toThrow(/ENOENT/);
  });
});
