"""Generate independent Transformers references for the released multimodal checkpoint.

python integration_tests/embeddinggemma2_reference.py /path/to/embeddinggemma-2 /tmp/eg2-reference
EMBEDDINGGEMMA2_MODEL_ROOT=... EMBEDDINGGEMMA2_FIXTURE_DIR=... cargo test \
  -p text-embeddings-backend-candle --features flash-attn --test test_embedding_gemma2 -- --ignored
"""
import argparse
import json
import pathlib
import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
from transformers import AutoModel, AutoProcessor

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("model", type=pathlib.Path)
parser.add_argument("output", type=pathlib.Path)
parser.add_argument("--attention", choices=["sdpa", "eager"], default="sdpa", help="Generate an eager control to measure BF16 attention implementation variation")
parser.add_argument("--audio-convolution", choices=["bf16", "fp32"], default="bf16", help="Generate a Transformers control with FP32 audio convolutions and BF16 outputs, matching Candle's convolution precision")
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
torch.backends.cuda.matmul.allow_tf32 = False
processor = AutoProcessor.from_pretrained(args.model)
model = AutoModel.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation=args.attention).cuda().eval()
if args.attention == "eager":
    # Gemma4 audio implements its own attention and requires boolean masks.
    # The eager mask factory emits additive masks instead; preserve SDPA's
    # boolean mask contract while comparing text/vision attention numerics.
    model.audio_tower.config._attn_implementation = "sdpa"
if args.audio_convolution == "fp32":
    for convolution in model.audio_tower.modules():
        if isinstance(convolution, (torch.nn.Conv1d, torch.nn.Conv2d)):
            convolution.float()
            # Hooks preserve causal padding in Gemma4AudioCausalConv1d.forward.
            convolution.register_forward_pre_hook(lambda module, inputs: (inputs[0].float(),))
            convolution.register_forward_hook(lambda module, inputs, output: output.bfloat16())
red = Image.new("RGB", (64, 48), (255, 0, 0))
green = Image.new("RGB", (48, 64), (0, 255, 0))
frames = np.stack([np.array(red), np.full((48, 64, 3), [0, 0, 255], dtype=np.uint8)])
audio = (0.25 * np.sin(np.arange(16000, dtype=np.float32) * (2 * np.pi * 440 / 16000))).astype(np.float32)
# Reference the samples actually represented by the encoded PCM16 WAV, so
# endpoint parity measures inference rather than waveform quantization drift.
audio = (audio * 32767).astype("<i2").astype(np.float32) / 32768
cases = [
    ("text", {"text": ["task: search result | query: What is artificial intelligence?"]}),
    ("text_batch", {"text": ["title: none | text: Artificial intelligence is a field of computer science.", "Hello.", "A longer sentence about the weather and the sun."]}),
    ("text_window", {"text": ["hello " * 550, "short"]}),
    ("image", {"text": ["<|image|>"], "images": [red]}),
    ("text_image", {"text": ["A colorful image: <|image|>"], "images": [green]}),
    ("images", {"text": ["<|image|><|image|>"], "images": [red, green]}),
    ("video", {"text": ["<|video|>"], "videos": [frames], "do_sample_frames": False}),
    ("audio", {"text": ["<|audio|>"], "audio": [audio]}),
    ("audio_batch", {"text": ["<|audio|>", "<|audio|>"], "audio": [audio[:1591], np.tile(audio, 3)[:37891]]}),
    ("audios", {"text": ["<|audio|><|audio|>"], "audio": [audio[:1591], audio]}),
    ("audio_limit", {"text": ["<|audio|>"], "audio": [np.tile(audio, 30)]}),
    ("mixed", {"text": ["Compare these inputs. <|image|><|video|><|audio|>"], "images": [red], "videos": [frames], "audio": [audio], "do_sample_frames": False}),
]
# Thirty-three one-fps source frames exercise the processor's uniform 32-frame cap.
limit_frames = np.stack([np.array(red) if i % 2 == 0 else frames[1] for i in range(33)])
limit_indices = np.linspace(0, 32, 32, dtype=int)
cases.insert(7, ("video_limit", {"text": ["<|video|>"], "videos": [limit_frames[limit_indices]], "do_sample_frames": False}))
manifest = []
for name, kwargs in cases:
    inputs = processor(**kwargs, return_tensors="pt", padding=True)
    # CPU input tensors retain their original dtype for processor parity tests.
    tensors = {k: (v.to(torch.uint8) if v.dtype == torch.bool else v).contiguous() for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    save_file(tensors, str(args.output / f"{name}.inputs.safetensors"))
    with torch.no_grad():
        out = model(**inputs.to(device="cuda", dtype=torch.bfloat16)).last_hidden_state
    mask = tensors["attention_mask"]
    raw = [out[i, : int(mask[i].sum())].float().cpu() for i in range(mask.shape[0])]
    pooled = [x.mean(0) for x in raw]
    save_file({"raw": torch.cat(raw), "pooled": torch.stack(pooled)}, str(args.output / f"{name}.outputs.safetensors"))
    manifest.append({"name": name, "sequences": [tensors["input_ids"][i, : int(mask[i].sum())].tolist() for i in range(mask.shape[0])]})
    print(name, [len(row) for row in manifest[-1]["sequences"]], flush=True)
# Independent CPU frontend reference exercises the exact Hann/log-mel/mask semantics.
features = processor.feature_extractor([audio], return_tensors="pt")
save_file({"samples": torch.from_numpy(audio), **{k:v.contiguous() for k,v in features.items()}}, str(args.output / "audio_features.safetensors"))
(args.output / "audio_features.json").write_text(json.dumps({
    "samples": audio.tolist(), "values": features["input_features"].flatten().tolist(),
    "mask": features["input_features_mask"].flatten().tolist(),
}))
(args.output / "manifest.json").write_text(json.dumps(manifest))
red.save(args.output / "red.png")
green.save(args.output / "green.png")
# Encoded media fixtures for endpoint tests: preserve frame indices 0 and 2 at 1 fps.
import subprocess, wave
for name, samples in [("audio.wav", audio), ("audio-short.wav", audio[:1591]), ("audio-long.wav", np.tile(audio, 3)[:37891]), ("audio-30s.wav", np.tile(audio, 30))]:
    with wave.open(str(args.output / name), "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000)
        f.writeframes((samples * 32768).astype("<i2").tobytes())
for i, frame in enumerate([np.array(red), np.array(red), frames[1], frames[1]]):
    Image.fromarray(frame).save(args.output / f"video-{i:02}.png")
subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", "2", "-i", str(args.output / "video-%02d.png"), "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv444p", str(args.output / "video.mp4")], check=True)

for i, frame in enumerate(limit_frames):
    Image.fromarray(frame).save(args.output / f"video-limit-{i:02}.png")
subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", "1", "-i", str(args.output / "video-limit-%02d.png"), "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv444p", str(args.output / "video-limit.mp4")], check=True)
