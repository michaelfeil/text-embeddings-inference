"""Real-weight expert-block diagnostic; this does not qualify full-model accuracy."""

import argparse
import ctypes
import json
from pathlib import Path

import torch
from safetensors import safe_open
from vllm.model_executor.layers.fused_moe.fused_moe import MoEActivation, fused_experts
from vllm.model_executor.models.gemma4 import gemma4_fused_routing_kernel_triton

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--library", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
lib = ctypes.CDLL(str(args.library.resolve()))
P = ctypes.c_void_p
I = ctypes.c_int
S = ctypes.c_size_t
lib.gemma4_moe_workspace_bytes.argtypes = [I, I, I]
lib.gemma4_moe_workspace_bytes.restype = S
fn = lib.gemma4_moe_forward_bf16
fn.argtypes = [P] * 6 + [I] * 3 + [P, S, P]


def weight(name):
    for f in args.model.glob("*.safetensors"):
        with safe_open(f, framework="pt", device="cpu") as s:
            if name in s.keys():  # noqa: SIM118 - safetensors handle is not a dict
                return s.get_tensor(name).cuda()
    raise KeyError(name)


prefix = "model.language_model.layers.0."
w1 = weight(prefix + "experts.gate_up_proj")
w2 = weight(prefix + "experts.down_proj")
scales = weight(prefix + "router.per_expert_scale").float()
torch.manual_seed(20)
results = []
for t in [1, 17, 257, 1024]:
    x = torch.randn(t, 2816, device="cuda", dtype=torch.bfloat16)
    logits = torch.randn(t, 128, device="cuda")
    out = torch.empty_like(x)
    scratch = torch.empty(
        lib.gemma4_moe_workspace_bytes(t, 2816, 704), device="cuda", dtype=torch.uint8
    )
    status = fn(
        logits.data_ptr(),
        scales.data_ptr(),
        x.data_ptr(),
        w1.data_ptr(),
        w2.data_ptr(),
        out.data_ptr(),
        t,
        2816,
        704,
        scratch.data_ptr(),
        scratch.numel(),
        torch.cuda.current_stream().cuda_stream,
    )
    assert status == 0, status
    tw, ti = gemma4_fused_routing_kernel_triton(logits, 8, scales)
    ref = fused_experts(x, w1, w2, tw, ti, activation=MoEActivation.GELU_TANH)
    torch.cuda.synchronize()
    err = out.float() - ref.float()
    rms = (err.square().mean() / ref.float().square().mean()).sqrt().item()
    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.float().flatten(), dim=0
    ).item()

    def ours(logits=logits, x=x, out=out, t=t, scratch=scratch):
        status = fn(
            logits.data_ptr(),
            scales.data_ptr(),
            x.data_ptr(),
            w1.data_ptr(),
            w2.data_ptr(),
            out.data_ptr(),
            t,
            2816,
            704,
            scratch.data_ptr(),
            scratch.numel(),
            torch.cuda.current_stream().cuda_stream,
        )
        assert status == 0, status

    def baseline(logits=logits, x=x):
        tw, ti = gemma4_fused_routing_kernel_triton(logits, 8, scales)
        return fused_experts(x, w1, w2, tw, ti, activation=MoEActivation.GELU_TANH)

    def bench(call):
        for _ in range(5):
            call()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(30):
            call()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / 30

    item = {
        "tokens": t,
        "max_abs_error": err.abs().max().item(),
        "relative_rmse": rms,
        "cosine": cos,
        "scratch_bytes": scratch.numel(),
        "cuda_ms": bench(ours),
        "vllm_ms": bench(baseline),
    }
    print(item, flush=True)
    results.append(item)
    assert rms < 0.01 and cos > 0.9999, item
args.output.write_text(json.dumps(results, indent=2))
