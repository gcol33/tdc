# v1 release — reconciled status and open decisions

The design agents drew sketches unaware that `include/tdc/stream.h`
(366 lines) and `include/tdc/plugin.h` (81 lines) already exist and
implement most of what TODO.md called "blockers". This doc reconciles
the TODO against reality and lists only the decisions actually left.

## Reconciled status

| TODO item | Reality | Real gap |
|---|---|---|
| 1. Streaming API | `tdc_stream_encoder` / `decoder` exist with I/O callbacks, schema, row-group index, seek. Block-at-a-time. | "Push arbitrary bytes → library splits into blocks" is the missing flavor. Vectra doesn't need it (it already works per-column). Mark **done unless a consumer asks**. |
| 2. Container I/O | 64-byte header + trailing row-group index already wired via stream.h. Schema serializer in place. Stats stub exists but **not wired into stream.h**. No `tdc_inspect` CLI. | (a) Wire `stats.c` into stream encoder/decoder. (b) Write `tdc_inspect`. The 128-byte file-header redesign the agent proposed is not needed. |
| 3. Plugin API | `tdc_{model,xform,entropy}_register` + `tdc_plugin_clear` live in plugin.h, 16 user slots, id-range enforcement. | Only `examples/plugin/` is missing. Effectively **done**. |
| 4. Vectra rewire | Vendor + wire up. | Needs the two new items below (SPARSE_ZERO, decode_into) before it can start. |
| 5. Stress + fuzz | 3 new test files, 33/33 pass, 450 stress iters + 840 corrupt decodes, no crashes. **Done this session.** | One flaky first-run observed; worth watching. |
| 6. Doxygen | 110 symbols documented across 9 headers. **Done this session.** | — |

## Remaining real work

1. **`SPARSE_ZERO_1D` model** — vectra uses it for ≥75%-zero int64/f64.
   Not in tdc. Needs `src/model/sparse_zero_1d.c` + tests + enum id.
2. **`decode_into` variant** — vectra hands R-owned buffers to decode;
   tdc always allocates its own via arena. Need a variant that writes
   into caller-provided memory (zero-copy path).
3. **Wire `stats.c` into streaming** — min/max/null-count emitted at
   `end_rowgroup`, read at `decoder_open`. Semantics for `side_meta`
   need nailing down.
4. **`tdc_inspect` CLI** — dump header/schema/rowgroup table without
   decoding payloads. Small.
5. **Examples/plugin/** template — one self-contained C file + README.
6. **Vectra rewire** — vendor tdc, drop legacy codec, run vectra tests.
   Blocked on items 1–2.
7. **Flaky stress-test investigation** — one-off failure on first run,
   passed on subsequent runs, fixed seed. Either extrinsic or there's
   hidden non-determinism. Read the first-run log before shrugging.

## Decisions (resolved 2026-04-17)

1. **Push/pull-over-bytes streaming**: **BUILD.** Caller-facing zstd/zlib
   UX. New header (name TBD — likely `pushpull.h` to keep frozen
   `stream.h` intact).
2. **`SPARSE_ZERO_1D` signature**: **Match `DICT_1D` shape.** No vectra
   on-disk copy.
3. **`decode_into` API**: **New function `tdc_decode_block_into`.** No
   flag-on-existing.
4. **Adaptive encode wrapper**: **Vectra owns it.** tdc stays spec-static.
5. **Zone-map content for v1**: **`{has_stats, min[16], max[16],
   null_count}` per (rowgroup, column).** No distinct-count / histogram.
6. **String codec**: **tdc adds UTF-8 DICT_1D + entropy path now.**
   Cleaner than shipping vectra in a half-state.

## Not blockers

- Container-format redesign (128-byte file header). Existing 64-byte
  `tdc_container_header` + trailing index stays.
- Global-vs-context registry rework. Setup-then-use singleton is fine.
