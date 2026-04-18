# Vignettes plan

Planning doc for the remaining tdc docs. The `quickstart.md` is done (2,530
prose words, 0 anti-AI findings, 7 runnable examples under
`docs/examples/`). Everything below is still to write.

Doc stack: MkDocs Material + `pymdownx.snippets`. Examples live under
`docs/examples/` and are built by CMake (see `TDC_DOC_EXAMPLES` list in
`CMakeLists.txt` — add new `.c` files there as they are written). Nav
lives in `mkdocs.yml`.

## Skill to use

`~/.claude/skills/tdc-vignette/SKILL.md`. Enforces:

- Prose word floor / ceiling per vignette type (table below).
- REQUIRED section checklist with min-prose and min-worked-examples per
  section.
- Library-author voice rules (no teacher mode, no "not X but Y", flat
  statements, specific numbers).
- Final anti-AI pass using `~/dotfiles/python/scan_ai.py`.

Counter: `python ~/.claude/skills/tdc-vignette/scripts/count_prose_md.py <path>`.

Scanner (run before committing):
`python ~/dotfiles/python/scan_ai.py <path> --tier1 --passive --catalogues --pairs --contrast --halves --emdash`.

## Priority order

Three tiers. Do tier 1 first — those are the broken links from
`quickstart.md`.

### Tier 1 — unblock quickstart outbound links

| # | File | Type | Floor | Why first |
|---|---|---|---|---|
| 1 | `backends/models.md` | Codec walkthrough | 3,000 | Linked 4x from quickstart (delta1d, pred2d, pred3d, plane2d mentions). Covers all model backends. |
| 2 | `backends/transforms.md` | Codec walkthrough | 3,000 | zigzag, byte_shuffle, quantize. Used in quickstart codec_compare example. |
| 3 | `backends/entropy.md` | Deep codec | 5,000 | LZ is the hot path. Matter of institutional memory from vectra/CLAUDE.md (no Robin Hood, no batched parse-then-copy). |
| 4 | `format/on-disk.md` | Format spec | 2,000 | Block record byte-by-byte. Needs a hex-dump example. |
| 5 | `extending/custom-backends.md` | Extending | 2,500 | Walk through adding a model vtable to `src/core/registry.c`. Only vignette allowed to reference internal headers. |
| 6 | `integration.md` | Integration | 2,000 | CMake subproject, vendoring (covers the `-D_POSIX_C_SOURCE` / `-lm` memory on Linux vendor builds), allocator wiring. |

### Tier 2 — depth coverage

| # | File | Type | Floor | Notes |
|---|---|---|---|---|
| 7 | `theory/predictors.md` | Algorithm theory | 2,500 | Paeth derivation, GRAD3D math, PLANE_2D least-squares. Pseudocode + correctness argument. |
| 8 | `theory/lz-inner-loop.md` | Algorithm theory + Deep | 2,500 | Hash table design (FNV-1a + 70% load, no Robin Hood), matcher, wildcopy, fast/safe decode split. Cite vectra benchmarks. |
| 9 | `performance/tuning.md` | Performance tuning | 2,000 | At least 3 codec_spec configurations benchmarked on the same input. Knob walk (quantize step, LZ level, shuffle grouping). |

### Tier 3 — reference

| # | File | Type | Floor | Notes |
|---|---|---|---|---|
| 10 | `reference/troubleshooting.md` | Troubleshooting | 1,500 | Every `tdc_status` value with one likely cause + one fix. |
| 11 | `reference/migration-vtr.md` | Migration guide | 1,500 | Side-by-side vectra call -> tdc call. Lean heavily on `VECTRA_REWIRE.md`. Be honest about "stay on vectra until P2-P5 land." |

## Per-vignette section gates

All REQUIRED sections + min-prose / min-worked-examples live in
`~/.claude/skills/tdc-vignette/SKILL.md` under "Enforcement: Section
Requirements by Vignette Type". Don't skip sections; the skill fails the
vignette if any REQUIRED heading is missing.

Quick reference for the most common types:

**Codec walkthrough (3,000 prose floor, 4+ worked examples):**
Introduction, What it does, When to use / when NOT, Worked example,
Parameter tuning, Benchmarks, Edge cases, Integration notes.

**Deep codec (4,150 prose floor, 7+ worked examples):** adds Inner loop
/ hot path, Multiple variants, Failure modes.

**Format spec (2,000 prose floor):** Scope and guarantees, Container
header, Block record, On-disk example, Evolution policy.

**Extending (2,500 prose floor):** Scope, The vtable, Registering,
Wiring tests, Performance checklist, Plugin API forward compat.

## Process per vignette

1. Invoke the skill with the target file (`Skill: tdc-vignette` with
   args `write backends/models.md`).
2. Skill reads public headers under `include/tdc/` first — never invent
   function names, enum ids, or status codes.
3. Writes the `.md` + any new `.c` examples under `docs/examples/`.
4. Adds new example names to `TDC_DOC_EXAMPLES` in `CMakeLists.txt`.
5. Builds and runs every example; byte counts in prose come from real
   runs, not guesses.
6. Runs `count_prose_md.py`, expands until >= floor.
7. Runs `scan_ai.py`; fixes until 0 findings (or all findings are
   justified).
8. Rebuilds the library and tests to confirm no CMake regression.

## Hard rules (project-specific)

From `tdc/CLAUDE.md`, carry into every vignette:

- Public API only (from `include/tdc/`). The only vignette allowed to
  reach into `src/` is `extending/custom-backends.md`.
- Allocator convention: `realloc_fn(user, ptr, n)`. No bare malloc/free
  in example code.
- Little-endian only — do not write "if your platform is big-endian"
  branches.
- Stage layering is sacred. A model owns iteration; a transform sees
  bytes; an entropy coder sees a flat byte buffer.

## Known gotchas from the quickstart pass

- `DELTA_1D` accepts floats (F16/F32/F64) via the XOR-delta variant,
  despite what the header comment implies. Don't write error-handling
  examples that assume float -> `TDC_E_DTYPE`. Use layout mismatch for
  guaranteed-reject paths instead.
- Every block carries an 80-byte header. The "RAW" spec grows the output
  by 80 bytes, not shrinks. Say this up front in any benchmark section.
- The encoder has a zero-residual fast path
  (`TDC_BLOCK_FLAG_ZERO_RESIDUAL`). Any synthetic input the model can
  predict exactly (linear ramp, tri-affine volume, etc.) will hit it and
  the transform / entropy stages will store zero bytes. Pick benchmark
  inputs carefully — too-clean inputs mask the real tradeoffs.

## Suggested parallelism

Tier 1 items 1, 2, 4, 5, 6 can run in parallel (independent files, no
shared examples). Item 3 (`backends/entropy.md`) depends on
`theory/lz-inner-loop.md` being planned, so either do LZ-theory first
and link from entropy, or do entropy first and backfill the theory link.

Tier 2 and 3 can all run in parallel once tier 1 lands.
