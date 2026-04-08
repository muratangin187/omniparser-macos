import argparse
import base64
import io
import json
import sys
from pathlib import Path

from PIL import Image

from omniparser_cli import apply_preset
from util.omniparser import Omniparser


def parse_args():
    parser = argparse.ArgumentParser(description="Persistent OmniParser worker over stdio.")
    parser.add_argument("--weights-dir", default="weights")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--som-device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--preset", default="balanced", choices=["full", "balanced", "ultra"])
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def build_parser_config(args):
    namespace = argparse.Namespace(
        device=args.device,
        som_device=args.som_device,
        preset=args.preset,
        fast=False,
        box_threshold=0.05,
        iou_threshold=0.1,
        imgsz=640,
        batch_size=128,
        ocr_batch_size=1,
        ocr_canvas_size=2560,
        use_paddleocr=False,
        no_ocr=False,
        no_semantics=False,
        scale_img=False,
        icon_crop_size=64,
        max_new_tokens=20,
    )
    apply_preset(namespace)
    return {
        "weights_dir": args.weights_dir,
        "device": namespace.device,
        "som_device": namespace.som_device,
        "enable_ocr": not namespace.no_ocr,
        "use_local_semantics": not namespace.no_semantics,
        "scale_img": namespace.scale_img,
        "BOX_TRESHOLD": namespace.box_threshold,
        "iou_threshold": namespace.iou_threshold,
        "use_paddleocr": namespace.use_paddleocr,
        "easyocr_args": {
            "text_threshold": 0.8,
            "batch_size": namespace.ocr_batch_size,
            "canvas_size": namespace.ocr_canvas_size,
        },
        "imgsz": namespace.imgsz,
        "batch_size": namespace.batch_size,
        "icon_crop_size": namespace.icon_crop_size,
        "max_new_tokens": namespace.max_new_tokens,
        "verbose": args.verbose,
    }


def save_result(result, image_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_image_path = output_dir / f"{image_path.stem}_annotated.png"
    output_json_path = output_dir / f"{image_path.stem}_parsed.json"
    annotated_image = Image.open(io.BytesIO(base64.b64decode(result["annotated_image_base64"])))
    annotated_image.save(output_image_path)
    payload = {
        "input_image": str(image_path),
        "output_image": str(output_image_path),
        "label_coordinates": result["label_coordinates"],
        "parsed_content_list": result["parsed_content_list"],
        "stats": result["stats"],
    }
    output_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_image_path, output_json_path, payload


def main():
    args = parse_args()
    parser = Omniparser(build_parser_config(args))
    sys.stdout.write(json.dumps({"ok": True, "event": "ready"}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            if request.get("action") == "shutdown":
                sys.stdout.write(json.dumps({"ok": True, "event": "bye"}) + "\n")
                sys.stdout.flush()
                break

            image_path = Path(request["image_path"]).expanduser().resolve()
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            result = parser.parse_image(image_path)
            response = {"ok": True, "stats": result["stats"]}

            if request.get("output_dir"):
                output_dir = Path(request["output_dir"]).expanduser().resolve()
                output_image_path, output_json_path, payload = save_result(result, image_path, output_dir)
                response.update(
                    {
                        "output_image": str(output_image_path),
                        "output_json": str(output_json_path),
                        "items": len(payload["parsed_content_list"]),
                    }
                )
            else:
                response["result"] = result

            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
        except Exception as exc:
            sys.stdout.write(json.dumps({"ok": False, "error": str(exc)}) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
