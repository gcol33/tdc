# Plugin example: XOR entropy backend

`xor_entropy.c` is a minimal, self-contained demonstration of the
runtime plugin API declared in `include/tdc/plugin.h`. It registers a
user-defined entropy backend that XORs every byte with 0xA5, resolves
it through `tdc_entropy_get`, and runs a round-trip on a small buffer.

The backend is not a compressor. The point is to show the lifecycle
(define vtable, register, look up, use), not entropy coding.

## Id-range rules

Every tdc stage (model, transform, entropy) reserves id ranges:

| Range            | Purpose                                    |
| ---------------- | ------------------------------------------ |
| `0x0001..0x00FF` | core (statically compiled; not registrable) |
| `0x0100..0x01FF` | experimental (statically compiled; not registrable) |
| `0x0200..0xFEFF` | reserved                                   |
| `0xFF00..0xFFFF` | user-defined (the only registrable range)  |

This example uses id `0xFF01`. Registration of any id outside the
user-defined range returns `TDC_E_INVAL`.

## Vtable lifetime

`tdc_entropy_register` stores the vtable pointer, it does not copy the
struct. The pointer MUST remain valid for the lifetime of the process
(or until `tdc_plugin_clear`). Put the vtable in `static` storage --
never on the stack, never in a heap block you might free.

## Build and run

From the repository root:

```
cmake -B build_examples -DTDC_BUILD_EXAMPLES=ON
cmake --build build_examples --config Release
```

Then run the resulting executable. On Windows with a multi-config
generator:

```
./build_examples/Release/tdc_example_plugin
```

On single-config generators:

```
./build_examples/tdc_example_plugin
```

Expected output: `OK` on stdout, exit code 0.
