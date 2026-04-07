# OmniParser macOS Runner

Mac-focused OmniParser setup with:

- a verified single-image CLI
- a local Gradio app
- Apple Silicon `mps` support
- a faster `--fast` preset
- Swift-friendly JSON and annotated PNG outputs

This repository is based on [microsoft/OmniParser](https://github.com/microsoft/OmniParser) and keeps the original project files, while adding the pieces needed to run it reliably on a modern Mac.

## What Changed

Compared with upstream, this repo adds and fixes:

- a real CLI entry point: `omniparser_cli.py`
- a reusable parser wrapper in `util/omniparser.py`
- lazy OCR backend loading so startup is more reliable on macOS
- `mps` support for the Florence caption model path
- a tuned `--fast` mode for Apple Silicon
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
  --fast \
  --output-dir outputs
```

Generated files:

- `outputs/<name>_annotated.png`
- `outputs/<name>_parsed.json`

Example:

```bash
.venv/bin/python omniparser_cli.py \
  --image /Users/you/Desktop/test.png \
  --fast \
  --output-dir outputs
```

Useful flags:

- `--device auto|cpu|mps|cuda`
- `--box-threshold 0.10`
- `--imgsz 512`
- `--batch-size 128`
- `--ocr-batch-size 16`
- `--ocr-canvas-size 1920`
- `--use-paddleocr`
- `--fast`

## Fast Mode

`--fast` is tuned for Apple Silicon and currently applies:

```text
--device mps
--box-threshold 0.10
--imgsz 512
--batch-size 128
--ocr-batch-size 16
--ocr-canvas-size 1920
```

On the validation image used during setup on an Apple M4 Pro:

- older cold CPU path: about `18.85s`
- older cold MPS path: about `23.62s`
- current cold `--fast` path: about `11.23s`
- best warm in-process benchmark: about `4.52s`

Those numbers will vary by screenshot size and GUI density, but the tuned path is materially faster than the original Mac run path.

## Gradio App

Start the local UI:

```bash
.venv/bin/python gradio_demo.py \
  --device mps \
  --default-box-threshold 0.1 \
  --default-imgsz 512 \
  --batch-size 128 \
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
--fast \
--output-dir /absolute/path/to/output-dir
```

If your app processes many screenshots, spawning Python for every frame will become the bottleneck. In that case, the next step is to run a persistent local server or helper process that keeps the model loaded in memory.

## Repository Layout

Relevant files added or changed for the macOS workflow:

- `omniparser_cli.py`
- `gradio_demo.py`
- `requirements-parser.txt`
- `util/omniparser.py`
- `util/utils.py`

## Notes

- `weights/` is intentionally gitignored and is not committed.
- `outputs*/` is also gitignored.
- This repository still includes the upstream Microsoft project files and license.
- The project is PyTorch-based, so Apple acceleration here uses `mps`, not MLX.

## Upstream

- Upstream repo: <https://github.com/microsoft/OmniParser>
- Models: <https://huggingface.co/microsoft/OmniParser-v2.0>
- Paper: <https://arxiv.org/abs/2408.00203>

## License

See the upstream [LICENSE](LICENSE) file and the model licensing notes from the original project. In particular, model checkpoint licenses differ by component.
