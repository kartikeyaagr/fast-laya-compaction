import { describe, expect, it } from 'vitest';
import { decideCall as jevDecide, type JevAsker } from 'fast-jev-compaction';
import { collectToolCalls, decideCall, type Message } from '../src/index.js';
import { confusion, reductionUnder, spearman } from '../scripts/agreement.js';
import { jevTarget, labelSession } from '../scripts/jev-labels.js';

function message(role: Message['role'], text: string, extra: Partial<Message> = {}): Message {
  return { role, text, toolUses: [], ...extra };
}

function call(id: string, tool: string, input: Record<string, unknown>, text: string): Message {
  return message('assistant', '', { toolUses: [{ tool_use_id: id, tool, input, text }] });
}

function result(id: string, text: string, isError = false): Message {
  return message('user', '', { toolResults: [{ tool_use_id: id, text, isError }] });
}

const file = 'export const a = 1;\n'.repeat(50);

function transcript(): Message[] {
  return [
    message('user', 'Fix the failing test.'),
    call('r1', 'Read', { file_path: 'src/a.ts' }, file),
    result('r1', file),
    call('b1', 'Bash', { command: 'npm test' }, 'FAIL'),
    result('b1', 'FAIL', true),
    call('e1', 'Edit', { file_path: 'src/a.ts', old_string: 'a = 1', new_string: 'a = 2' }, 'ok'),
    result('e1', 'ok'),
    call('b2', 'Bash', { command: 'npm test' }, 'PASS'),
    result('b2', 'PASS'),
    message('assistant', 'Fixed.'),
  ];
}

/** Answers every Jev question from a table keyed by question name (`call_t1`, `result_t1`, …). */
function fakeJev(answers: Record<string, number>, seen: string[][] = []): JevAsker {
  return {
    async ask(_state, questions) {
      seen.push(Object.keys(questions));
      return { answers: Object.fromEntries(Object.keys(questions).map((q) => [q, { noul: answers[q] ?? 0.5 }])) };
    },
  };
}

describe('Jev as teacher', () => {
  it('maps Jev’s two answers onto a target that reproduces Jev’s decision at any threshold', () => {
    let seed = 7;
    const random = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
    const unpinned = { id: 't1', tool: 'Read', pinned: false };
    for (let i = 0; i < 500; i++) {
      const jev = { keepCall: random(), keepResult: random() };
      const target = jevTarget(jev);
      expect(target.keep + target.truncate + target.drop).toBeCloseTo(1);
      for (const keepThreshold of [0.3, 0.5, 0.7]) {
        expect(decideCall(unpinned, { probabilities: target }, { keepThreshold }).action).toBe(
          jevDecide(unpinned, jev, { keepThreshold }).action,
        );
      }
    }
  });

  it('labels every scored call with Jev’s answers, Laya’s state and the superseded rule', async () => {
    const seen: string[][] = [];
    const answers = { call_t1: 0.2, result_t1: 0.1, call_t2: 0.9, result_t2: 0.2, call_t3: 0.95, result_t3: 0.9 };
    const { labels, summary } = await labelSession('/s.jsonl', transcript(), fakeJev(answers, seen), 3);

    expect(seen.flat().sort()).toEqual(['call_t1', 'call_t2', 'call_t3', 'result_t1', 'result_t2', 'result_t3']);
    expect(labels.map((l) => [l.id, l.tool, l.superseded, l.jev.action])).toEqual([
      ['t1', 'Read', true, 'drop_call'],
      ['t2', 'Bash', true, 'drop_result'],
      ['t3', 'Edit', false, 'keep'],
    ]);
    expect(labels[0]).toMatchObject({ session: '/s.jsonl', messages: 10, recent: 3, call: 'Read file_path=src/a.ts' });
    expect(labels[0]!.state).toMatchObject({ tool_call: 'Read file_path=src/a.ts', superseded: true });
    expect(labels[2]!.jev).toEqual({ keepCall: 0.95, keepResult: 0.9, action: 'keep' });
    expect(summary).toMatchObject({ session: '/s.jsonl', messages: 10, calls: 4, scored: 3, requests: 1 });
    expect(summary.reduction).toBeGreaterThan(0);
  });
});

describe('scoreboard metrics', () => {
  it('computes rank correlation with ties averaged', () => {
    expect(spearman([1, 2, 3, 4], [10, 20, 30, 40])).toBeCloseTo(1);
    expect(spearman([1, 2, 3, 4], [4, 3, 2, 1])).toBeCloseTo(-1);
    expect(spearman([1, 1, 2, 3], [5, 5, 6, 7])).toBeCloseTo(1);
    expect(spearman([1, 1, 1], [1, 2, 3])).toBeNaN();
  });

  it('counts Jev-by-Laya decisions and the reduction a set of actions gives', () => {
    const table = confusion([
      ['keep', 'keep'],
      ['keep', 'drop_result'],
      ['drop_call', 'drop_result'],
    ]);
    expect(table.keep).toEqual({ keep: 1, drop_result: 1, drop_call: 0 });
    expect(table.drop_call.drop_result).toBe(1);

    const messages = transcript();
    const calls = collectToolCalls(messages, 3);
    expect(reductionUnder(messages, calls, new Map())).toBe(0);
    const dropped = reductionUnder(messages, calls, new Map([['t1', 'drop_call']]));
    const truncated = reductionUnder(messages, calls, new Map([['t1', 'drop_result']]));
    expect(dropped).toBeGreaterThan(truncated);
    expect(truncated).toBeGreaterThan(0);
  });
});
