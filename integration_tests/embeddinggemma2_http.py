"""Check a running router against embeddinggemma2_reference.py fixtures.

python integration_tests/embeddinggemma2_http.py http://localhost:8088 /tmp/eg2-reference
Requires numpy and safetensors. Exercises batching, all modalities, projection,
normalization, task prompts, raw outputs, and invalid media rejection.
"""
import argparse
import base64
import concurrent.futures
import json
import io
import wave
import pathlib
import subprocess
import tempfile
import urllib.error
import urllib.request

import numpy as np
from safetensors.numpy import load_file
from PIL import Image

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("url")
parser.add_argument("fixtures", type=pathlib.Path)
args = parser.parse_args()


def request(path, body):
    req = urllib.request.Request(
        args.url.rstrip("/") + path,
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response)


def encoded(name):
    return base64.b64encode((args.fixtures / name).read_bytes()).decode()


def image(name):
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded(name)}}


video = {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + encoded("video.mp4")}}
limit_video = {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + encoded("video-limit.mp4")}}
audio = {"type": "input_audio", "input_audio": {"format": "wav", "data": encoded("audio.wav")}}
short_audio = {"type": "input_audio", "input_audio": {"format": "wav", "data": encoded("audio-short.wav")}}
long_audio = {"type": "input_audio", "input_audio": {"format": "wav", "data": encoded("audio-long.wav")}}
limit_audio = {"type": "input_audio", "input_audio": {"format": "wav", "data": encoded("audio-30s.wav")}}


def messages(parts):
    return [{"role": "user", "content": parts}]


cases = {
    "text": "task: search result | query: What is artificial intelligence?",
    "text_batch": ["title: none | text: Artificial intelligence is a field of computer science.", "Hello.", "A longer sentence about the weather and the sun."],
    "text_window": ["hello " * 550, "short"],
    "image": messages([image("red.png")]),
    "text_image": messages([{"type": "text", "text": "A colorful image: "}, image("green.png")]),
    "images": messages([image("red.png"), image("green.png")]),
    "video": messages([video]),
    "video_limit": messages([limit_video]),
    "audio": messages([audio]),
    "audios": messages([short_audio, audio]),
    "audio_limit": messages([limit_audio]),
    "mixed": messages([{"type": "text", "text": "Compare these inputs. "}, image("red.png"), video, audio]),
}


def check(item):
    name, inputs = item
    result = np.asarray(request("/embed", {"inputs": inputs, "normalize": False}))
    expected = load_file(str(args.fixtures / f"{name}.outputs.safetensors"))["pooled"]
    assert result.shape == expected.shape, (name, result.shape, expected.shape)
    cosines = (result * expected).sum(-1) / (np.linalg.norm(result, axis=-1) * np.linalg.norm(expected, axis=-1))
    assert np.all(cosines > 0.995), (name, cosines)
    openai = request("/v1/embeddings", {"model": "google/embeddinggemma-2", "input": inputs})
    vectors = np.asarray([row["embedding"] for row in openai["data"]])
    assert vectors.shape == expected.shape, (name, vectors.shape)
    openai_cosines = (vectors * expected).sum(-1) / (np.linalg.norm(vectors, axis=-1) * np.linalg.norm(expected, axis=-1))
    assert np.all(openai_cosines > 0.995), (name, openai_cosines)
    raw = request("/embed_all", {"inputs": inputs})
    sequences = next(c["sequences"] for c in json.loads((args.fixtures / "manifest.json").read_text()) if c["name"] == name)
    assert [len(row) for row in raw] == [len(row) for row in sequences], name
    assert all(len(token) == 768 for row in raw for token in row), name
    print(name, "pooled cosine", cosines.tolist(), "raw lengths", [len(row) for row in raw], flush=True)


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(check, cases.items()))

# Independent requests exercise unequal-length audio batching in the queue.
expected_audio = load_file(str(args.fixtures / "audio_batch.outputs.safetensors"))["pooled"]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    results = list(pool.map(lambda part: request("/embed", {"inputs": messages([part]), "normalize": False}), [short_audio, long_audio]))
for actual, expected in zip(results, expected_audio):
    actual = np.asarray(actual)[0]
    similarity = np.dot(actual, expected) / (np.linalg.norm(actual) * np.linalg.norm(expected))
    assert similarity > 0.995, ("audio_batch", similarity)

for dimensions in [768, 512, 256, 128]:
    result = np.asarray(request("/embed", {"inputs": "Hello.", "dimensions": dimensions, "normalize": True}))
    assert result.shape == (1, dimensions)
    assert np.allclose(np.linalg.norm(result, axis=-1), 1, atol=1e-5)

prompted = request("/embed", {"inputs": "What is artificial intelligence?", "prompt_name": "query"})
explicit = request("/embed", {"inputs": cases["text"]})
assert np.allclose(prompted, explicit, atol=1e-5)
manifest = json.loads((args.fixtures / "manifest.json").read_text())
text_ids = next(case["sequences"][0] for case in manifest if case["name"] == "text")
assert np.allclose(request("/embed", {"inputs": text_ids}), explicit, atol=1e-5)
conversation = [{"role": "user", "content": "What is artificial intelligence?"},
                {"role": "system", "content": "task: search result | query: "}]
assert np.allclose(request("/embed", {"inputs": conversation}), explicit, atol=1e-5)
manual = messages([{"type": "text", "text": "Compare these inputs. <|image|><|video|><|audio|>"},
                   image("red.png"), video, audio])
assert np.allclose(request("/embed", {"inputs": manual}), request("/embed", {"inputs": cases["mixed"]}), atol=1e-5)
openai = request("/v1/embeddings", {"model": "google/embeddinggemma-2", "input": "Hello.", "dimensions": 128})
assert len(openai["data"][0]["embedding"]) == 128
with tempfile.TemporaryDirectory() as directory:
    directory = pathlib.Path(directory)
    for extension, mime in [("jpg", "image/jpeg"), ("webp", "image/webp")]:
        path = directory / f"red.{extension}"
        Image.open(args.fixtures / "red.png").save(path)
        data = base64.b64encode(path.read_bytes()).decode()
        result = request("/embed", {"inputs": messages([{"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}])})
        assert np.asarray(result).shape == (1, 768)
        assert np.isfinite(result).all()
    for source, target, codec in [("audio.wav", "audio.mp3", "libmp3lame"), ("video.mp4", "video.webm", "libvpx-vp9")]:
        path = directory / target
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(args.fixtures / source), "-threads", "1", "-c:a" if target.endswith("mp3") else "-c:v", codec, str(path)], check=True)
        data = base64.b64encode(path.read_bytes()).decode()
        part = ({"type": "input_audio", "input_audio": {"format": "mp3", "data": data}}
                if target.endswith("mp3") else {"type": "video_url", "video_url": {"url": "data:video/webm;base64," + data}})
        result = request("/embed", {"inputs": messages([part])})
        assert np.asarray(result).shape == (1, 768)
        assert np.isfinite(result).all()
# Exercise decoder duration boundaries without adding encoded fixture files.
def silence(samples):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)
        wav.writeframes(bytes(samples * 2))
    return messages([{"type": "input_audio", "input_audio": {
        "format": "wav", "data": base64.b64encode(buffer.getvalue()).decode()}}])

minimum_audio = np.asarray(request("/embed", {"inputs": silence(321)}))
assert minimum_audio.shape == (1, 768) and np.isfinite(minimum_audio).all()
maximum_text = np.asarray(request("/embed", {"inputs": [2] * 8192, "normalize": True}))
assert maximum_text.shape == (1, 768) and np.isfinite(maximum_text).all()
assert np.allclose(np.linalg.norm(maximum_text, axis=-1), 1, atol=1e-5)
for invalid in [
    [2] * 8193,
    silence(0), silence(320), silence(480001),
    messages([{"type": "input_audio", "input_audio": {"format": "wav", "data": "invalid"}}]),
    messages([{"type": "text", "text": "<|image|>"}]),
    messages([{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,aW52YWxpZA=="}}]),
    messages([image("red.png")] * 5),
]:
    try:
        request("/embed", {"inputs": invalid})
    except urllib.error.HTTPError as error:
        assert 400 <= error.code < 500, error.code
    else:
        raise AssertionError("invalid media was accepted")
print("projection, normalization, prompts, OpenAI endpoint, JPEG/WebP/MP3/WebM, and media rejection passed")
