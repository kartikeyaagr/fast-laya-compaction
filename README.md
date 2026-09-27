# fast-laya-compaction

Claude Code compaction that keeps your conversation verbatim and trims old tool
output instead of summarizing, decided by a small model running on your own
machine.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## About

Claude Code's built-in compaction asks a model to summarize the conversation.
Summaries are lossy: a file path, an exact error or a constraint can vanish
even when it matters later. This plugin never rewrites anything. When the
context fills up, it removes or truncates **old tool calls and their outputs**
that are no longer needed, and leaves every user and assistant message exactly
as written.

The decisions come from two places:

- **A rule** removes a read-only call that a later call repeated exactly, or
  whose file a later edit changed.
- **[Laya](https://github.com/NandhaKishorM/laya)**, a 421M-parameter
  classifier that runs locally, scores every other old call as *keep*,
  *truncate* (keep a 300-character head) or *drop*.

No API key, no network at compaction time, nothing leaves your machine. It is
a port of [fast-jev-compaction](https://github.com/tamaratran/fast-jev-compaction),
which asks TypeSafe's hosted Jev model instead.

**Results on three real sessions:** 27–33% smaller in 19–47 s, against 79 s for a
built-in compaction on the same machine. Details, and what did not work, in
[BENCHMARK.md](BENCHMARK.md).

## How it works

1. Tool calls are paired with their results. The first message and the newest
   `preserveRecentMessages` messages are never touched.
2. **Superseded calls are removed** with their results, without a model: a
   read-only call (Read, Grep, Bash, WebFetch, …) repeated exactly later, or a
   read of a file a later Edit/Write changed. Edits and writes themselves are
   never removed.
3. The oldest `maxScoredCalls` remaining calls each get a small state for Laya:
   the call, its outcome, how long ago it ran, whether later calls touched the
   same target, the current goal and the start of its output.
4. Laya answers *keep / truncate / drop* for all of them in one batched pass.
   `P(keep) ≥ keepThreshold` keeps the call; otherwise `P(keep) + P(truncate) ≥
   keepThreshold` keeps the call with a truncated output; otherwise it is
   removed.
5. If the result is at least `minReductionRatio` smaller, it replaces the
   history. Otherwise, or if anything fails, Claude Code's built-in summary
   runs as usual.

The hook runs `backend/laya_compact.py` once per compaction with
`uv run --offline`, so no model stays in memory between compactions.

## Requirements

- Claude Code **2.1.274+** (function hooks, early access)
- [uv](https://docs.astral.sh/uv/)
- ~1.6 GB of disk for torch and the checkpoint, ~2–3 GB of free memory while compacting
- Apple Silicon (`mps`), an NVIDIA GPU (`cuda`) or CPU

## Install

1. **Enable function hooks** wherever Claude Code runs, for example in
   `~/.claude/settings.json`:

   ```json
   { "env": { "CLAUDE_CODE_ENABLE_FUNCTION_HOOKS": "1" } }
   ```

2. **Install the plugin:**

   ```sh
   claude plugin marketplace add kartikeyaagr/fast-laya-compaction
   claude plugin install fast-laya-compaction@fast-laya-compaction
   ```

3. **Download Laya once** (torch and ~843 MB of weights, pinned to
   `laya==0.3.20` and a reviewed Hugging Face commit). The hook never
   downloads; until this has run, compaction falls back to the built-in summary
   with a toast that shows this exact command:

   ```sh
   uv run --script ~/.claude/plugins/cache/fast-laya-compaction/fast-laya-compaction/0.1.0/backend/laya_compact.py --warmup
   ```

   It ends with `laya_compact: ready (typed-decisions on mps, …)`.

4. **Check it:** in a session with some tool calls, run `/compact`. The toast
   reads `kept N/M messages, no summary (…)`, or
   `fallback to built-in summary (<reason>)`.

## Configuration

Set options with `/plugin configure fast-laya-compaction@fast-laya-compaction`
inside Claude Code, or at install time with
`claude plugin install … --config keepThreshold=0.55`.

| Option | Default | What it does |
| --- | --- | --- |
| `keepThreshold` | `0.5` | Minimum probability for a call (or its full output) to stay; higher trims more |
| `preserveRecentMessages` | `6` | Newest messages never touched (the first is always kept) |
| `compactAtPercent` | `60` | Context percentage at which the plugin requests compaction |
| `minReductionRatio` | `0.25` | Minimum reduction to replace the history instead of summarizing |
| `truncateHeadChars` | `300` | Characters of a truncated output kept before its note |
| `maxScoredCalls` | `80` | Most calls scored per compaction, oldest first; bounds the time |
| `timeoutSeconds` | `120` | Longest a Laya run may take before falling back |
| `model` | `typed-decisions` | Laya checkpoint: `typed-decisions`, `multilingual` (faster, weaker) or `english` |
| `checkpointPath` | unset | Absolute path of a fine-tuned checkpoint; overrides `model` |
| `maxCallStateChars` | `1600` | Size cap on what Laya reads about one call |
| `goal` | latest user prompt | Task description Laya weighs calls against |
| `uvPath` | `uv` | uv executable, if it is not on Claude Code's PATH |
| `device` | auto | `mps`, `cpu` or `cuda` |

## Tune before you trust it

Laya is only weakly decisive on this task: its `keep` probabilities mostly fall
between 0.3 and 0.65, ranking edits and writes above exploratory `ls`, `git`
and search calls, and it rarely chooses *drop*. See what it would do to your
own sessions before relying on it. The dry-run changes nothing:

```sh
git clone https://github.com/kartikeyaagr/fast-laya-compaction && cd fast-laya-compaction
npm install
npm run dry-run -- ~/.claude/projects/<project>/<session>.jsonl --threshold 0.5
```

It prints every call with its keep/truncate/drop probabilities and action,
the reduction, the timings, and whether the hook would replace the history.

## Getting closer to Jev

Prompt changes move Laya's answers a lot (renaming one key in its input
changed a session from 33% to 23%) without saying which way is closer to Jev,
and Laya cannot read a whole conversation the way Jev does
([measured](BENCHMARK.md#design-experiments)). So Jev is used as a teacher:

1. **Label sessions with Jev.** Put `TYPESAFE_API_KEY=…` in `.env`
   (git-ignored). This **sends those transcripts to TypeSafe** and stores Jev's
   per-call answers in `data/` (git-ignored). Aim for 20–40 sessions:
   `npm run jev-labels -- ~/.claude/projects/<project>/*.jsonl`
2. **Score Laya against Jev** and compare prompt variants from
   `scripts/variants.ts`: `npm run agreement -- --variant timeline`. It reports
   decision agreement, a Jev-by-Laya confusion table, rank correlation and
   reduction under both. Jev's two probabilities map onto Laya's three so that
   matching the target reproduces Jev's decision at any threshold.
3. **Export training data** with the best variant:
   `npm run export-training -- --variant <best>` writes rows in the format of
   Laya's own training notebook, with whole sessions held out.
4. **Fine-tune on a CUDA GPU** (a free Kaggle 2×T4 session; an 8 GB Mac cannot)
   with Laya's `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`,
   changed to read `laya-train.jsonl`, start from the `typed-decisions`
   checkpoint, keep `max_len`/`head_max_len` at 512/128, fit the temperature on
   `laya-holdout.jsonl`, and delete `temperature_by_options` from the saved
   `rl_agent_config.json`. This step has not been run here.
5. **Use it:** `npm run agreement -- --model /abs/path/to/checkpoint`, then set
   the `checkpointPath` option.

## Publishing

To ship your own fork or a new version:

1. Bump `version` in `.claude-plugin/plugin.json`, `.claude-plugin/marketplace.json`
   and `package.json`.
2. `npm run typecheck && npm test && npm run test:backend && npm run validate:plugin`.
3. Commit and push to GitHub. The repository is its own marketplace
   (`.claude-plugin/marketplace.json`), so `claude plugin marketplace add
   <owner>/<repo>` works as soon as the push lands. Only committed files are
   installed; `data/`, `tasks/`, `docs/` and `.env` are git-ignored.
4. Users update with:

   ```sh
   claude plugin marketplace update fast-laya-compaction
   claude plugin update fast-laya-compaction@fast-laya-compaction   # then restart Claude Code
   ```

   A new version installs to a new cache directory; the Laya download is shared,
   so setup does not need to run again unless `laya` or the checkpoint changes.

For local testing without publishing:
`CLAUDE_CODE_ENABLE_FUNCTION_HOOKS=1 claude --plugin-dir /path/to/fast-laya-compaction`,
or `claude plugin marketplace add /path/to/fast-laya-compaction` to install from
a local directory.

## Library usage

The compaction logic is also a TypeScript library (not published to npm; build
it with `npm run build` and import from `dist/`):

```ts
import { compactMessages, reductionRatio, type Message } from 'fast-laya-compaction';

const result = await compactMessages(transcript, { preserveRecentMessages: 4 });
console.log(result.decisions, result.stats);
if (reductionRatio(result) < 0.25) {
  // not worth it: keep the original transcript, or summarize instead
}
```

`Message` is a subset of Claude Code's `SessionMessage`. `compactMessages` runs
the same Laya script through `uv` from Node. To bring your own scorer, implement
`LayaScorer` and call `compact(messages, scorer, options)`; `collectToolCalls`,
`supersededBy`, `callState`, `decideCall` and `applyDecisions` are exported too.

## Limitations

- Only tool calls and their outputs are ever removed; messages are never
  shortened.
- A probability is not proof an output is safe to trim. Truncation keeps a head
  and a note, and the assistant can re-run the tool.
- Speed depends heavily on free memory (0.2–0.8 s per scored call on an 8 GB M2).
- Function hooks are early access and may change between Claude Code releases;
  `types/claude-code.d.ts` was generated by Claude Code 2.1.274.
- Laya is young and moves fast, so the package and weights are pinned.

## Development

```sh
npm install
npm run typecheck && npm test      # library, hook and scripts (vitest)
npm run test:backend               # Python contract tests (fake model, no torch)
npm run validate:plugin
CLAUDE_CODE_ENABLE_FUNCTION_HOOKS=1 claude --plugin-dir .
```

| Path | What |
| --- | --- |
| `hooks/fast-laya.ts` | The Claude Code hook: config, the Laya process, fallback, auto-compact |
| `src/` | The library: call pairing, the superseded rule, per-call states, decisions |
| `backend/laya_compact.py` | One-shot Laya scorer (stdin → stdout), run with `uv run --script` |
| `scripts/` | `dry-run`, benchmark ceilings, and the Jev teacher tools |
| `BENCHMARK.md` | Measurements |

## Acknowledgments

- [fast-jev-compaction](https://github.com/tamaratran/fast-jev-compaction) (MIT):
  the original design and most of the compaction code.
- [Laya](https://github.com/NandhaKishorM/laya) by Convai Innovations
  (Apache-2.0, code and weights).

## License

MIT. See [LICENSE](LICENSE).
