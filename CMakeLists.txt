# tdc — typed dimensional compression library
#
# Builds:
#   - libtdc.a              static library
#   - tdc_smoke             header-compile + link smoke test
#
# Vectra (the primary consumer) does NOT use CMake — it vendors the tdc
# source tree under vectra/src/tdc/ and compiles it via R's Makevars.
# This CMake file exists so tdc can be developed and tested standalone
# (incremental builds, ctest, sanitizers, IDE integration).

cmake_minimum_required(VERSION 3.15)
project(tdc C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(TDC_BUILD_TESTS    "Build tdc tests" ON)
option(TDC_BUILD_BENCH    "Build tdc throughput bench" ON)
option(TDC_BUILD_TOOLS    "Build tdc command-line tools (tdc_inspect)" ON)
option(TDC_BUILD_EXAMPLES "Build tdc examples (plugin demo)" OFF)
option(TDC_HAVE_ZLIB      "Enable TDC_ENTROPY_DEFLATE backend (requires zlib)" OFF)

option(TDC_ENABLE_AVX2 "Enable AVX2 intrinsics (requires AVX2-capable CPU)" ON)

if(MSVC)
    add_compile_options(/W4 /permissive-)
    if(TDC_ENABLE_AVX2)
        add_compile_options(/arch:AVX2)
    endif()
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wstrict-prototypes
                        -Wmissing-prototypes -Wcast-align)
    if(TDC_ENABLE_AVX2)
        add_compile_options(-mavx2)
    endif()
endif()

# ----- Source layout (kept in lock-step with the design tree) ---------------

set(TDC_SOURCES
    src/core/block.c
    src/core/stream.c
    src/core/arena.c
    src/core/registry.c
    src/core/error.c
    src/core/decode_profile.c

    src/layout/traverse.c
    src/layout/tiling.c

    src/model/raw.c
    src/model/delta1d.c
    src/model/dict.c
    src/model/pred2d.c
    src/model/plane2d.c
    src/model/stack2d.c
    src/model/pred3d.c
    src/model/pred3d_float.c
    src/model/delta2_1d.c
    src/model/fpc1d.c
    src/model/dict_numeric.c
    src/model/sparse_zero_1d.c

    src/transform/quantize.c
    src/transform/zigzag.c
    src/transform/shuffle.c
    src/transform/bitshuffle.c

    src/symbols/residuals.c
    src/symbols/runs.c

    src/entropy/none.c
    src/entropy/lz.c
    src/entropy/lz_opt.c
    src/entropy/lz_streams.c
    src/entropy/mf_hashchain.c
    src/entropy/mf_btree.c
    src/entropy/deflate.c
    src/entropy/huffman.c
    src/entropy/huffman4.c
    src/entropy/fse.c
    src/entropy/lane.c

    src/format/header.c
    src/format/block_record.c
    src/format/metadata.c
    src/format/schema.c
    src/format/rowgroup.c
    src/format/stats.c

    src/api/encode.c
    src/api/decode.c
    src/api/decode_ex.c
    src/api/decode_impl.c
    src/api/decode_into.c
    src/api/stream_encode.c
    src/api/stream_decode.c
    src/api/pushpull_encode.c
    src/api/pushpull_decode.c
)

add_library(tdc STATIC ${TDC_SOURCES})
target_include_directories(tdc PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

if(TDC_HAVE_ZLIB)
    find_package(ZLIB REQUIRED)
    target_link_libraries(tdc PUBLIC ZLIB::ZLIB)
    target_compile_definitions(tdc PUBLIC TDC_HAVE_ZLIB=1)
endif()

# ----- Tests ----------------------------------------------------------------

if(TDC_BUILD_TESTS)
    enable_testing()
    add_executable(tdc_smoke tests/test_smoke.c)
    target_link_libraries(tdc_smoke PRIVATE tdc)
    add_test(NAME smoke COMMAND tdc_smoke)

    add_executable(test_lz_roundtrip tests/test_lz_roundtrip.c)
    target_link_libraries(test_lz_roundtrip PRIVATE tdc)
    add_test(NAME lz_roundtrip COMMAND test_lz_roundtrip)

    add_executable(test_lz_opt_roundtrip tests/test_lz_opt_roundtrip.c)
    target_link_libraries(test_lz_opt_roundtrip PRIVATE tdc)
    add_test(NAME lz_opt_roundtrip COMMAND test_lz_opt_roundtrip)

    add_executable(test_lz_streams_roundtrip tests/test_lz_streams_roundtrip.c)
    target_link_libraries(test_lz_streams_roundtrip PRIVATE tdc)
    add_test(NAME lz_streams_roundtrip COMMAND test_lz_streams_roundtrip)

    add_executable(test_huffman_roundtrip tests/test_huffman_roundtrip.c)
    target_link_libraries(test_huffman_roundtrip PRIVATE tdc)
    add_test(NAME huffman_roundtrip COMMAND test_huffman_roundtrip)

    add_executable(test_fse_roundtrip tests/test_fse_roundtrip.c)
    target_link_libraries(test_fse_roundtrip PRIVATE tdc)
    add_test(NAME fse_roundtrip COMMAND test_fse_roundtrip)

    add_executable(test_byte_shuffle_roundtrip tests/test_byte_shuffle_roundtrip.c)
    target_link_libraries(test_byte_shuffle_roundtrip PRIVATE tdc)
    add_test(NAME byte_shuffle_roundtrip COMMAND test_byte_shuffle_roundtrip)

    add_executable(test_quantize_roundtrip tests/test_quantize_roundtrip.c)
    target_link_libraries(test_quantize_roundtrip PRIVATE tdc)
    if(NOT MSVC)
        target_link_libraries(test_quantize_roundtrip PRIVATE m)
    endif()
    add_test(NAME quantize_roundtrip COMMAND test_quantize_roundtrip)

    add_executable(test_zigzag_roundtrip tests/test_zigzag_roundtrip.c)
    target_link_libraries(test_zigzag_roundtrip PRIVATE tdc)
    add_test(NAME zigzag_roundtrip COMMAND test_zigzag_roundtrip)

    add_executable(test_delta1d_roundtrip tests/test_delta1d_roundtrip.c)
    target_link_libraries(test_delta1d_roundtrip PRIVATE tdc)
    add_test(NAME delta1d_roundtrip COMMAND test_delta1d_roundtrip)

    add_executable(test_delta2_fpc_roundtrip tests/test_delta2_fpc_roundtrip.c)
    target_link_libraries(test_delta2_fpc_roundtrip PRIVATE tdc)
    add_test(NAME delta2_fpc_roundtrip COMMAND test_delta2_fpc_roundtrip)

    add_executable(test_raw_roundtrip tests/test_raw_roundtrip.c)
    target_link_libraries(test_raw_roundtrip PRIVATE tdc)
    add_test(NAME raw_roundtrip COMMAND test_raw_roundtrip)

    add_executable(test_pred2d_roundtrip tests/test_pred2d_roundtrip.c)
    target_link_libraries(test_pred2d_roundtrip PRIVATE tdc)
    add_test(NAME pred2d_roundtrip COMMAND test_pred2d_roundtrip)

    add_executable(test_pred3d_roundtrip tests/test_pred3d_roundtrip.c)
    target_link_libraries(test_pred3d_roundtrip PRIVATE tdc)
    add_test(NAME pred3d_roundtrip COMMAND test_pred3d_roundtrip)

    add_executable(test_dict1d_roundtrip tests/test_dict1d_roundtrip.c)
    target_link_libraries(test_dict1d_roundtrip PRIVATE tdc)
    add_test(NAME dict1d_roundtrip COMMAND test_dict1d_roundtrip)

    add_executable(test_dict_numeric_roundtrip tests/test_dict_numeric_roundtrip.c)
    target_link_libraries(test_dict_numeric_roundtrip PRIVATE tdc)
    add_test(NAME dict_numeric_roundtrip COMMAND test_dict_numeric_roundtrip)

    add_executable(test_sparse_zero_roundtrip tests/test_sparse_zero_roundtrip.c)
    target_link_libraries(test_sparse_zero_roundtrip PRIVATE tdc)
    add_test(NAME sparse_zero_roundtrip COMMAND test_sparse_zero_roundtrip)

    add_executable(test_dnum_e2e tests/test_dnum_e2e.c)
    target_link_libraries(test_dnum_e2e PRIVATE tdc)
    add_test(NAME dnum_e2e COMMAND test_dnum_e2e)

    add_executable(test_lane_roundtrip tests/test_lane_roundtrip.c)
    target_link_libraries(test_lane_roundtrip PRIVATE tdc)
    add_test(NAME lane_roundtrip COMMAND test_lane_roundtrip)

    add_executable(test_pipeline_roundtrip tests/test_pipeline_roundtrip.c)
    target_link_libraries(test_pipeline_roundtrip PRIVATE tdc)
    add_test(NAME pipeline_roundtrip COMMAND test_pipeline_roundtrip)

    add_executable(test_format_validate tests/test_format_validate.c)
    target_link_libraries(test_format_validate PRIVATE tdc)
    add_test(NAME format_validate COMMAND test_format_validate)

    add_executable(test_stack2d_roundtrip tests/test_stack2d_roundtrip.c)
    target_link_libraries(test_stack2d_roundtrip PRIVATE tdc)
    add_test(NAME stack2d_roundtrip COMMAND test_stack2d_roundtrip)

    add_executable(test_bitshuffle_roundtrip tests/test_bitshuffle_roundtrip.c)
    target_link_libraries(test_bitshuffle_roundtrip PRIVATE tdc)
    add_test(NAME bitshuffle_roundtrip COMMAND test_bitshuffle_roundtrip)

    add_executable(test_deflate_roundtrip tests/test_deflate_roundtrip.c)
    target_link_libraries(test_deflate_roundtrip PRIVATE tdc)
    add_test(NAME deflate_roundtrip COMMAND test_deflate_roundtrip)

    add_executable(test_stream_roundtrip tests/test_stream_roundtrip.c)
    target_link_libraries(test_stream_roundtrip PRIVATE tdc)
    add_test(NAME stream_roundtrip COMMAND test_stream_roundtrip)

    add_executable(test_pushpull_roundtrip tests/test_pushpull_roundtrip.c)
    target_link_libraries(test_pushpull_roundtrip PRIVATE tdc)
    add_test(NAME pushpull_roundtrip COMMAND test_pushpull_roundtrip)
    set_tests_properties(pushpull_roundtrip PROPERTIES TIMEOUT 180)

    add_executable(test_schema tests/test_schema.c)
    target_link_libraries(test_schema PRIVATE tdc)
    add_test(NAME schema COMMAND test_schema)

    add_executable(test_decode_ex tests/test_decode_ex.c)
    target_link_libraries(test_decode_ex PRIVATE tdc)
    add_test(NAME decode_ex COMMAND test_decode_ex)

    add_executable(test_decode_into tests/test_decode_into.c)
    target_link_libraries(test_decode_into PRIVATE tdc)
    add_test(NAME decode_into COMMAND test_decode_into)

    add_executable(test_rowgroup_index tests/test_rowgroup_index.c)
    target_link_libraries(test_rowgroup_index PRIVATE tdc)
    add_test(NAME rowgroup_index COMMAND test_rowgroup_index)

    add_executable(test_stats tests/test_stats.c)
    target_link_libraries(test_stats PRIVATE tdc)
    add_test(NAME stats COMMAND test_stats)

    add_executable(test_rowgroup_stats tests/test_rowgroup_stats.c)
    target_link_libraries(test_rowgroup_stats PRIVATE tdc)
    add_test(NAME rowgroup_stats COMMAND test_rowgroup_stats)

    add_executable(test_plugin tests/test_plugin.c)
    target_link_libraries(test_plugin PRIVATE tdc)
    add_test(NAME plugin COMMAND test_plugin)

    add_executable(test_adversarial_inputs tests/test_adversarial_inputs.c)
    target_link_libraries(test_adversarial_inputs PRIVATE tdc)
    add_test(NAME adversarial_inputs COMMAND test_adversarial_inputs)

    add_executable(test_entropy_corrupt tests/test_entropy_corrupt.c)
    target_link_libraries(test_entropy_corrupt PRIVATE tdc)
    add_test(NAME entropy_corrupt COMMAND test_entropy_corrupt)

    add_executable(test_stress_roundtrip tests/test_stress_roundtrip.c)
    target_link_libraries(test_stress_roundtrip PRIVATE tdc)
    add_test(NAME stress_roundtrip COMMAND test_stress_roundtrip)

endif()

# ----- Bench ----------------------------------------------------------------

if(TDC_BUILD_BENCH)
    add_executable(bench_throughput bench/bench_throughput.c)
    target_link_libraries(bench_throughput PRIVATE tdc)
    if(NOT MSVC)
        target_link_libraries(bench_throughput PRIVATE m)
    endif()

    add_executable(bench_pred3d_avg3 bench/bench_pred3d_avg3.c)

    add_executable(bench_lz_opt bench/bench_lz_opt.c)
    target_link_libraries(bench_lz_opt PRIVATE tdc)

    add_executable(bench_plane_decode bench/bench_plane_decode.c)
    target_link_libraries(bench_plane_decode PRIVATE tdc)

    add_executable(bench_lz_streams_quick bench/bench_lz_streams_quick.c)
    target_link_libraries(bench_lz_streams_quick PRIVATE tdc)

    add_executable(bench_lz_decode_micro bench/bench_lz_decode_micro.c)
    target_link_libraries(bench_lz_decode_micro PRIVATE tdc)
    # Round-trip smoke gate: runs every bench pipeline once on a tiny block
    # and asserts bytes match. Sub-second; catches any pipeline that breaks
    # round-trip between bench runs. SPEEDUP-TODO P3.2.
    if(TDC_BUILD_TESTS)
        add_test(NAME bench_smoke COMMAND bench_throughput --smoke)
    endif()
endif()

# ----- Tools ----------------------------------------------------------------

if(TDC_BUILD_TOOLS)
    add_executable(tdc_inspect tools/tdc_inspect.c)
    target_link_libraries(tdc_inspect PRIVATE tdc)
endif()

# ----- Examples -------------------------------------------------------------

if(TDC_BUILD_EXAMPLES)
    add_executable(tdc_example_plugin examples/plugin/xor_entropy.c)
    target_link_libraries(tdc_example_plugin PRIVATE tdc)
endif()
