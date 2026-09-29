# Gemma4 routed experts

Experimental BF16 execution for the text decoder of `google/gemma-4-26B-A4B-it`:
128 experts, top eight routing, hidden size 2816, expert intermediate size 704.
Requires a CUDA build targeting SM80 or later. Only H100 has been exercised.
Other Gemma4 expert shapes and quantized weights are rejected.

SM90 builds additionally compile a Hopper SM90a TMA/WGMMA implementation.
It is selected for at least 2048 compact tokens on runtime SM90 devices;
smaller batches and other builds use the portable grouped GEMM. Both paths
share routing, packing, GELU and combination kernels, including scalar fallbacks
for unaligned input/output views. Hopper reserves 2 MiB of additional GEMM
workspace and checks the CUTLASS requirement before launch. The Hopper tile is
128 x 256 x 64; this improved the tested long-request medians by another
2–6% over the initial 128 x 128 x 64 tile, with unchanged regression scores.

Routing, token grouping, both expert GEMMs, GELU gating and weighted reduction
run on the caller's CUDA stream. Routing counts stay on the device. The Rust
binding checks tensor shapes, dtypes, contiguous storage and device identity,
and owns the scratch/output allocations. Shared dense MLP and expert branch
normalization are composed in `models/gemma4.rs`.

CUTLASS v4.5.0 is fetched at pinned commit
`e406c186f510a15091cce01f782020ceb7ba8eb5` by cudaforge. Its license is included
in `LICENSE.cutlass`.

GELU rounds to BF16 before multiplying the up branch. The down projection keeps
FP32 accumulation until routing weights are applied, then rounds each expert's
weighted output to BF16 before summing in FP32. The Hopper path applies the
routing weights in the GEMM epilogue and stores BF16 weighted outputs, avoiding
the larger FP32 intermediate buffer. The portable path retains its separate
FP32 output and weighting step. This follows vLLM's intermediate rounding.
Different GEMM/attention accumulation orders can still produce different model
scores; routing agreement alone does not qualify full-model accuracy.

## Expert-block diagnostic

From this directory, with CUDA, the pinned CUTLASS checkout, PyTorch, safetensors
and vLLM 0.30 installed:

```sh
nvcc -std=c++17 -O3 --expt-relaxed-constexpr -arch=sm_90 \
  -shared -Xcompiler -fPIC -I/path/to/cutlass/include \
  tests/probe.cu -o /tmp/gemma4_moe.so
CUDA_VISIBLE_DEVICES=0 python tests/compare_vllm.py \
  --model /path/to/gemma-4-26B-A4B-it \
  --library /tmp/gemma4_moe.so --output /tmp/gemma4-moe-results.json
```

This compares actual layer-zero weights at 1, 17, 257 and 1024 tokens, with
relative RMS error below 0.1% and cosine above 0.999999 as primitive-test gates.
It also records CUDA-event timings. These tolerances are not model-quality
acceptance criteria. Keep a separate full-model decision/scoring comparison.

The harness also checks exact GELU-plus-multiply agreement with vLLM across six
input scales, to catch changes to intermediate BF16 rounding. It additionally
checks every BF16 gate bit pattern against the original scalar arithmetic,
with unit/random up values, widths 13/704 and aligned/offset buffers. NaNs
are treated as equivalent; finite outputs must match bit for bit.

For the Hopper implementation, build both translation units and select its
entry points in the harness:

```sh
nvcc -std=c++17 -O3 -DNDEBUG --expt-relaxed-constexpr -arch=sm_90a \
  -shared -Xcompiler -fPIC -I/path/to/cutlass/include \
  -I/path/to/cutlass/tools/util/include \
  tests/probe.cu kernels/grouped_gemm_hopper.cu -o /tmp/gemma4_moe_hopper.so
CUDA_VISIBLE_DEVICES=0 python tests/compare_vllm.py \
  --model /path/to/gemma-4-26B-A4B-it --hopper --tokens 2048 4096 \
  --library /tmp/gemma4_moe_hopper.so --output /tmp/gemma4-moe-hopper.json
```

Use `--unaligned-io` and `--concentrated` separately to exercise offset views
and routing concentrated on eight experts (including 120 empty experts).
`-DNDEBUG` avoids CUTLASS device assertions that serialize WGMMA instructions;
host shape, workspace and launch-status checks remain enabled.

To verify a kernel change against a prior build, add `--baseline-library /path/to/previous.so`
to the diagnostic. It requires identical BF16 output bits using the selected
portable/Hopper entry point, in addition to the existing vLLM comparison.
