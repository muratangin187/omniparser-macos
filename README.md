# OmniParser macOS Runner

Mac-focused OmniParser setup with:

- a verified single-image CLI
- a persistent stdio worker for hot requests
- a local Gradio app
- Apple Silicon `mps` support
- subsecond `balanced`, `turbo`, and `ultra` presets
- Swift-friendly JSON and annotated image outputs

This repository is based on [microsoft/OmniParser](https://github.com/microsoft/OmniParser) and keeps the original project files, while adding the pieces needed to run it reliably on a modern Mac.

## What Changed

Compared with upstream, this repo adds and fixes:

- a real CLI entry point: `omniparser_cli.py`
- a persistent worker process: `omniparser_worker.py`
- a reusable parser wrapper in `util/omniparser.py`
- lazy OCR backend loading so startup is more reliable on macOS
- `mps` support for the Florence caption model path
- tuned `balanced`, `turbo`, and `ultra` presets for Apple Silicon
- caption caching for repeated screenshots in a hot process
- worker warm-up so the first real screenshot is faster
- direct image saving for CLI and worker outputs without a base64 round-trip
- optional JPEG or BMP output for lower save latency
- a Gradio launcher that does not assume CUDA or `share=True`
- a smaller parser-only dependency file: `requirements-parser.txt`

## Quick Start

```bash
git clone https://github.com/muratangin187/omniparser-macos.git
cd omniparser-macos
uv venv --python 3.12 .venv
.venv/bin/python -m ensurepip --upgrade
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-parser.txt
```

Download the OmniParser v2 weights:

```bash
mkdir -p weights
for f in icon_detect/{train_args.yaml,model.pt,model.yaml} icon_caption/{config.json,generation_config.json,model.safetensors}; do
  .venv/bin/hf download microsoft/OmniParser-v2.0 "$f" --local-dir weights
done
if [ -d weights/icon_caption ] && [ ! -d weights/icon_caption_florence ]; then
  mv weights/icon_caption weights/icon_caption_florence
fi
if [ -f weights/icon_caption/model.safetensors ]; then
  mv weights/icon_caption/model.safetensors weights/icon_caption_florence/model.safetensors
  rmdir weights/icon_caption 2>/dev/null || true
fi
```

Expected weight layout:

```text
weights/
  icon_detect/
    model.pt
    model.yaml
    train_args.yaml
  icon_caption_florence/
    config.json
    generation_config.json
    model.safetensors
```

## CLI Usage

Run the parser on a single screenshot:

```bash
.venv/bin/python omniparser_cli.py \
  --image /absolute/path/to/screenshot.png \
  --preset balanced \
  --output-dir outputs
```

Generated files:

- `outputs/<name>_annotated.<png|jpg|bmp>`
- `outputs/<name>_parsed.json`

Example:

```bash
.venv/bin/python omniparser_cli.py \
  --image /Users/you/Desktop/test.png \
  --preset balanced \
  --output-dir outputs
```

Useful flags:

- `--device auto|cpu|mps|cuda`
- `--som-device auto|cpu|mps|cuda`
- `--preset full|balanced|turbo|ultra`
- `--no-ocr`
- `--no-semantics`
- `--scale-img`
- `--icon-crop-size 32`
- `--max-new-tokens 3`
- `--output-format png|jpg|bmp`
- `--use-paddleocr`
- `--fast`

## Performance Modes

`balanced` is the main Apple Silicon speed preset and currently applies:

```text
--device mps
--som-device cpu
--no-ocr
--scale-img
--box-threshold 0.18
--imgsz 384
--batch-size 192
--icon-crop-size 32
--max-new-tokens 3
```

`turbo` is the more aggressive semantic preset:

```text
--device mps
--som-device cpu
--no-ocr
--scale-img
--box-threshold 0.15
--imgsz 320
--batch-size 192
--icon-crop-size 32
--max-new-tokens 3
```

`ultra` is the box-only preset:

```text
--device mps
--som-device cpu
--no-ocr
--no-semantics
--scale-img
--box-threshold 0.18
--imgsz 384
--batch-size 192
```

Measured on the validation image used during setup on an Apple M4 Pro, using `/Users/muratangin/madlen/misc/test.png`:

- older cold CPU path: about `18.85s`
- older cold MPS path: about `23.62s`
- current cold CLI `balanced` wall time: about `5.38s`
- current cold CLI `turbo` wall time: about `5.29s`
- current hot-worker first `balanced` request wall time with JPEG output: about `507ms`
- current hot-worker first `turbo` request wall time with JPEG output: about `412ms`
- current hot-worker first `ultra` request wall time with JPEG output: about `188ms`
- current repeated cached `balanced` request wall time with JPEG output: about `117ms`
- current repeated cached `turbo` request wall time with JPEG output: about `117ms`

Those numbers will vary by screenshot size and GUI density, but the tuned hot-process path is now well under `1s` for semantic parsing and well under `250ms` for box-only parsing.

## Persistent Worker

For a real app, do not spawn the CLI for every screenshot. Start the worker once and keep it alive:

```bash
.venv/bin/python omniparser_worker.py --preset balanced --output-format jpg
```

The worker speaks JSON over stdin/stdout.

Request:

```json
{"image_path":"/absolute/path/to/screenshot.png","output_dir":"/tmp/omni-out"}
```

Response:

```json
{
  "ok": true,
  "stats": {
    "total_ms": 492.75,
    "detect_ms": 135.5,
    "caption_ms": 305.14,
    "caption_cache_hits": 0,
    "caption_cache_misses": 51,
    "save_ms": 12.31
  },
  "output_image": "/tmp/omni-out/test_annotated.jpg",
  "output_json": "/tmp/omni-out/test_parsed.json",
  "items": 51
}
```

Shutdown:

```json
{"action":"shutdown"}
```

## Gradio App

Start the local UI:

```bash
.venv/bin/python gradio_demo.py \
  --device mps \
  --default-box-threshold 0.18 \
  --default-imgsz 384 \
  --batch-size 192 \
  --ocr-batch-size 16 \
  --ocr-canvas-size 1920
```

Then open:

```text
http://127.0.0.1:7861
```

## JSON Output Shape

The CLI writes a JSON file like:

```json
{
  "input_image": "/abs/path/to/input.png",
  "output_image": "/abs/path/to/input_annotated.png",
  "label_coordinates": {
    "0": [0.1, 0.2, 0.3, 0.4]
  },
  "parsed_content_list": [
    {
      "type": "text",
      "bbox": [0.1, 0.2, 0.3, 0.4],
      "interactivity": false,
      "content": "Chrome",
      "source": "box_ocr_content_ocr"
    }
  ]
}
```

That format is convenient for native apps because you can:

- display the annotated image directly
- read the structured text/icon list
- map label IDs back to normalized coordinates

## Swift Integration

The easiest integration from a Swift macOS app is:

1. capture a screenshot
2. save it as a temporary PNG
3. execute `omniparser_cli.py`
4. read back the JSON and annotated PNG

Typical command:

```bash
/absolute/path/to/.venv/bin/python \
/absolute/path/to/omniparser_cli.py \
--image /absolute/path/to/screenshot.png \
--preset turbo \
--output-format jpg \
--output-dir /absolute/path/to/output-dir
```

Recommended production path for Swift:

1. start `omniparser_worker.py` once when the app launches
2. keep stdin/stdout pipes open
3. send one JSON request per screenshot
4. reuse the in-memory model and caption cache

That is the path that gets semantic responses to roughly `400-500ms` on the first request and roughly `100-120ms` on repeated identical screenshots.

## Repository Layout

Relevant files added or changed for the macOS workflow:

- `omniparser_cli.py`
- `omniparser_worker.py`
- `gradio_demo.py`
- `requirements-parser.txt`
- `util/omniparser.py`
- `util/utils.py`

## Notes

- `weights/` is intentionally gitignored and is not committed.
- `outputs*/` is also gitignored.
- This repository still includes the upstream Microsoft project files and license.
- The project is PyTorch-based, so Apple acceleration here uses `mps`, not MLX.
- I tested an MLX route with `mlx-vlm` and `mlx-community/Florence-2-base-ft-4bit`; on this machine it failed to load Florence cleanly, so MLX is not currently a drop-in replacement for the caption path in this repo.

## Upstream

- Upstream repo: <https://github.com/microsoft/OmniParser>
- Models: <https://huggingface.co/microsoft/OmniParser-v2.0>
- Paper: <https://arxiv.org/abs/2408.00203>

## License

See the upstream [LICENSE](LICENSE) file and the model licensing notes from the original project. In particular, model checkpoint licenses differ by component.
