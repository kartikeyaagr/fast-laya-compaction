# Benchmarks

What fast-laya-compaction does to real Claude Code sessions, how long it takes,
and what was tried on the way. Every number below was measured on one machine;
treat them as that machine's numbers, not guarantees.

**At a glance** (default settings, three real sessions):

| | Session A | Session B | Session C |
| --- | ---: | ---: | ---: |
| Messages / tool calls | 55 / 27 | 128 / 51 | 329 / 163 |
| Reduction (characters) | **33%** | **27%** | **31%** (41% uncapped) |
| Wall time, dry-run | 19 s | 21 s | 47 s* |
| Hook outcome | replaces history | replaces history | replaces history |

\* Measured while another Laya process was running; the scoring itself took 25 s.
For comparison, one built-in auto-compaction on the same machine took **79 s**
(187k → 27k tokens).

## Setup

| | |
| --- | --- |
| Machine | Apple M2, 8 GB RAM, macOS; 4–8 GB of swap in use throughout |
| Device | `mps` (Apple GPU) unless noted |
| Laya | `laya==0.3.20`, `typed-decisions` checkpoint, Hugging Face commit `55cf4c4` |
| Settings | defaults: `max_len` 512, `head_max_len` 128, batch 32, `keepThreshold` 0.5, `preserveRecentMessages` 6, `maxScoredCalls` 80 |
| Claude Code | 2.1.283 with function hooks |
| Sessions | three real transcripts: **A** research and planning (Bash, Write, Edit, WebSearch), **B** mixed (Bash, Agent, Read, Write), **C** a Bash-heavy build session (155 of 163 calls are Bash) |

Reduction is the share of characters (message text, tool inputs, tool
results) removed, as reported by `reductionRatio`. The hook only replaces the
history when it reaches `minReductionRatio` (25%).

Reproduce:

```sh
npm run dry-run -- <session.jsonl>                     # reduction, decisions, timings
npm run dry-run -- <session.jsonl> --max-scored 1000   # uncapped
npm run bench:ceilings -- <session.jsonl>…             # ceilings and Jev's request shape (no model)
```

## Reduction

### Final design: superseded rule + keep/truncate/drop

| Session | Scored | Kept | Truncated | Dropped by Laya | Removed by rule | Reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 25 | 8 | 17 | 0 | 0 | 33% |
| B | 48 | 10 | 38 | 0 | 0 | 27% |
| C (80 scored) | 80 | 9 | 71 | 0 | 1 | 31% |
| C (uncapped) | 159 | 16 | 143 | 0 | 1 | 41% |

Laya never chose *drop* on these sessions, so its reduction comes from
truncating old tool outputs. The superseded rule fired once: these sessions
rarely re-read a file or re-run the exact same command. In a live
read-edit-test session it removed 2 of 8 calls (see below).

### Ceilings

The same keep/truncate/drop mechanics bound what any model could achieve on
these sessions:

| Session | Truncate every scored call | Drop every scored call | Laya |
| --- | ---: | ---: | ---: |
| A | 60% | 90% | 33% |
| B | 29% | 72% | 27% |
| C | 51% | 96% | 41% (uncapped) |

Truncation alone cannot pass the first column; only removing whole calls (the
rule, or a *drop* answer) can go beyond it.

### Jev (not measured)

Jev's actual decisions were not measured: it needs a TypeSafe API key. What
the original fast-jev-compaction *would* send for the same sessions, from its
own code (`npm run bench:ceilings`):

| Session | Jev state | Requests | Tokens sent |
| --- | --- | ---: | ---: |
| A | ~9.1k tokens, whole history | 1 | ~12k |
| B | ~24.8k tokens, whole history | 2 | ~55k |
| C | ~18.9k tokens, tool inputs cut to 60 chars | 2 | ~58k |

At the $0.042 per million tokens quoted in Laya's README (a third-party
figure), that is well under a cent per compaction. `npm run jev-labels` and
`npm run agreement` measure Laya against Jev once a key is available (see the
README).

## Speed

| | Measured |
| --- | --- |
| First setup (`--warmup`) | ~4 min download (torch, 843 MB of weights); import 8.7 s |
| Start and load, per compaction | 5.4–8.5 s typical, up to 16 s under memory pressure |
| Scoring, `mps`, one warm process | 170–220 ms per call |
| Scoring, `mps`, per compaction | 0.2–0.8 s per call, depending on swap |
| Scoring, `cpu` | 395–457 ms per call (40 calls) |
| Peak memory | ~1.1 GB process, ~3.4 GB on the GPU side while scoring |

`maxScoredCalls` (80) exists because of the last rows: at 0.8 s per call, 160
calls plus loading would pass the 120 s timeout.

Tried and rejected:

| Change | Result |
| --- | --- |
| `max_len` 256 instead of 512 | up to ~1.7× faster in one run, but changed 15 of 66 decisions |
| fp16 weights on `mps` | crashes (Metal dtype assertion in Laya's heads) |
| `torch.mps.empty_cache()` between batches | slower: 396 vs 217 ms per call |
| Scoring in chunks under `inference_mode` | no reliable gain: 207 and 494 ms per call in two identical runs |

## Live Claude Code runs

Headless `claude -p` sessions (Haiku) on a throwaway project, then `/compact`:

| Run | Messages | Reduction | Tokens | Compaction time | Notes |
| --- | --- | ---: | --- | ---: | --- |
| 5 reads, `ls`, `wc`, grep | 22 → 22 | 84% | 38,365 → 8,771 | 14.1 s | 5 old reads truncated to 300 chars (earlier design, no rule) |
| read, edit, repeated `wc`, reads | 27 → 23 | 67% | 35,817 → 9,593 | 8.7 s | rule removed the read of the later-edited file and the first `wc` |

In both, Claude Code logged that the hook's messages stood and the built-in
compaction never ran, and the resumed session answered questions about the
earlier work correctly.

Fallbacks, each ending in the built-in summary with the reason in a toast:

| Case | Result |
| --- | --- |
| Weights not downloaded | exit 3 after 251 ms: "Laya is not set up, run once: …" |
| `uv` not on PATH | "Laya could not start (… ENOENT …)" |
| `timeoutSeconds: 1` | "Laya timed out after 1s" |
| No tool calls to score | "below 25% minimum … Laya not called" |

## Design experiments

| Experiment | Result | Kept? |
| --- | --- | --- |
| Three wordings of the keep/truncate/drop question (58 calls) | spread of P(keep) 0.069–0.080; rankings driven by tool type (edits and writes high, `ls`, `git` and search low) | kept the first wording |
| Yes/no question with yes/no labels and answer descriptions | P(yes) median 0.60–0.62 → nothing truncated at 0.5: **0%** on all three sessions | no |
| Bare yes/no question | median P(yes) ≈ 0.48 → 24% / 19% / 25% (all below 25%) | no |
| Yes/no, opposite polarity ("can it be trimmed?") | rank correlation −0.44 with the original on P(keep): Laya said yes to the same calls both ways | no |
| Renaming one state key, `superseded` → `touched_later` | reduction 33% → 23% (A), 27% → 24% (B) | reverted; key names are part of the prompt |
| Jev's whole-history state (2,963 tokens, `max_len` 4096), one question per call | ~3 s per call; P(keep) 0.320–0.336 for every real call and 0.329 for a call that does not exist; all calls at once thrashed the machine | no: Laya cannot find a call in a long history |
| `timeline` variant (per-call state plus what happened after, 1024-token rows) | ~1.5× the cost of the default | pending a Jev comparison |

## Caveats

- One machine, under heavy memory pressure; timings vary run to run by up to
  2× for that reason. More free memory is the biggest speed-up.
- Three sessions from one user. Reduction depends on how much of a session is
  large tool output and how often calls repeat.
- Reduction is not quality. Whether the right calls were trimmed is only
  measurable against a reference such as Jev (`npm run agreement`).
