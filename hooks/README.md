# fast-laya-compaction Claude Code mod

This plugin uses Claude Code function hooks to replace a compaction with the
original messages, minus the tool calls and tool results Laya judged no longer
needed. `hooks/fast-laya.ts` is a thin adapter: it reads the plugin options,
hands `session.compact` transcripts to the library in `src/` (the plugin folder
is the repository root, so the hook imports it directly) and maps the result
back onto session messages. User and assistant text is never touched.

Read-only calls that a later call repeated or made stale (a re-run command,
a re-read, a file edited after it was read) are removed with their results by
rule. Every other old call is scored by Laya as keep, truncate (its output) or
drop (with its output).

The hook module runs in Claude Code's sandbox (no Node, no Python), so scoring
happens in a separate process started through `$.process.run`:

```
uv run --quiet --offline --script <plugin root>/backend/laya_compact.py
```

The request (one state per tool call and the shared question) goes in on
stdin; one JSON line of probabilities comes back on stdout. The process loads
the checkpoint, scores every state in one batched pass and exits, so no memory
is held between compactions. Only one Laya process runs at a time: a
compaction that arrives while another is scoring (a subagent's, or one ahead
of time) falls back to the built-in summary.

Install it as described in the root [README](../README.md#install): enable
function hooks (`CLAUDE_CODE_ENABLE_FUNCTION_HOOKS=1`), install from the
marketplace, and run the one-time `--warmup` of `backend/laya_compact.py`.

For local development:

```sh
CLAUDE_CODE_ENABLE_FUNCTION_HOOKS=1 claude --plugin-dir .
```

## Configuration

The plugin declares these `userConfig` values in
`.claude-plugin/plugin.json`; see the root README for what each does.

| Option | Default |
| --- | ---: |
| `model` | `typed-decisions` |
| `checkpointPath` | unset |
| `keepThreshold` | `0.5` |
| `preserveRecentMessages` | `6` |
| `maxScoredCalls` | `80` |
| `maxCallStateChars` | `1600` |
| `truncateHeadChars` | `300` |
| `compactAtPercent` | `60` |
| `minReductionRatio` | `0.25` |
| `timeoutSeconds` | `120` |
| `uvPath` | `uv` |
| `device` | auto |
| `goal` | latest user prompt |

The hook falls back to Claude Code's built-in compaction, with the reason in
a toast and the log, when:

- the setup has not run (`Laya is not set up, run once: …`);
- Laya takes longer than `timeoutSeconds`, cannot start (`uvPath`), or exits
  with an error;
- the answer is malformed;
- the estimated reduction is below `minReductionRatio`.

On success the toast shows the reduction, per-reason counts, the checkpoint,
device and load/inference times; a per-call `decisions:` line with both
probabilities is logged for diagnosis. The `turn.complete` hook requests
compaction when `context.percent` reaches `compactAtPercent`, with an
in-flight guard.

## Scope and caveat

Function hooks are early access and may change between Claude Code releases.
This mod uses the generated declarations from 2.1.274 in
`types/claude-code.d.ts`; regenerate and review that file after a Claude Code
upgrade.

References:

- [Claude Code plugins](https://code.claude.com/docs/en/plugins)
- [Claude Code plugins reference](https://code.claude.com/docs/en/plugins-reference)
- [Claude Code hooks](https://code.claude.com/docs/en/hooks)
- [Laya](https://github.com/NandhaKishorM/laya)
