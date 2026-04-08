import argparse
import base64
import io
import json
from pathlib import Path

from PIL import Image

from util.omniparser import Omniparser


def parse_args():
    parser = argparse.ArgumentParser(description="Run OmniParser on a single image.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--weights-dir", default="weights")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--som-device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--preset", default="full", choices=["full", "balanced", "ultra"])
    parser.add_argument("--box-threshold", type=float, default=0.05)
    parser.add_argument("--iou-threshold", type=float, default=0.1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ocr-batch-size", type=int, default=1)
    parser.add_argument("--ocr-canvas-size", type=int, default=2560)
    parser.add_argument("--use-paddleocr", action="store_true")
    parser.add_argument("--fast", action="store_true", help="Use tuned faster settings for Apple Silicon.")
    parser.add_argument("--no-ocr", action="store_true")
    parser.add_argument("--no-semantics", action="store_true")
    parser.add_argument("--scale-img", action="store_true")
    parser.add_argument("--icon-crop-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def apply_preset(args):
    if args.fast and args.preset == "full":
        args.preset = "balanced"

    if args.preset == "balanced":
        if args.device == "auto":
            args.device = "mps"
        if args.som_device == "auto":
            args.som_device = "cpu"
        args.no_ocr = True
        args.scale_img = True
        args.box_threshold = 0.15
        args.imgsz = 512
        args.batch_size = 128
        args.icon_crop_size = 48
        args.max_new_tokens = 6
    elif args.preset == "ultra":
        if args.device == "auto":
            args.device = "mps"
        if args.som_device == "auto":
            args.som_device = "cpu"
        args.no_ocr = True
        args.no_semantics = True
        args.scale_img = True
        args.box_threshold = 0.20
        args.imgsz = 384
        args.batch_size = 128
        args.icon_crop_size = 48
        args.max_new_tokens = 8


def main():
    args = parse_args()
    apply_preset(args)

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_image_path = output_dir / f"{image_path.stem}_annotated.png"
    output_json_path = output_dir / f"{image_path.stem}_parsed.json"

    omniparser = Omniparser(
        {
            "weights_dir": args.weights_dir,
            "device": args.device,
            "som_device": args.som_device,
            "enable_ocr": not args.no_ocr,
            "use_local_semantics": not args.no_semantics,
            "scale_img": args.scale_img,
            "BOX_TRESHOLD": args.box_threshold,
            "iou_threshold": args.iou_threshold,
            "use_paddleocr": args.use_paddleocr,
            "easyocr_args": {
                "text_threshold": 0.8,
                "batch_size": args.ocr_batch_size,
                "canvas_size": args.ocr_canvas_size,
            },
            "imgsz": args.imgsz,
            "batch_size": args.batch_size,
            "icon_crop_size": args.icon_crop_size,
            "max_new_tokens": args.max_new_tokens,
        }
    )
    result = omniparser.parse_image(
        image_path,
        enable_ocr=not args.no_ocr,
        box_threshold=args.box_threshold,
        iou_threshold=args.iou_threshold,
        use_paddleocr=args.use_paddleocr,
        easyocr_args={
            "text_threshold": 0.8,
            "batch_size": args.ocr_batch_size,
            "canvas_size": args.ocr_canvas_size,
        },
        use_local_semantics=not args.no_semantics,
        scale_img=args.scale_img,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        icon_crop_size=args.icon_crop_size,
        max_new_tokens=args.max_new_tokens,
    )

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

    print(f"Saved annotated image to {output_image_path}")
    print(f"Saved parsed output to {output_json_path}")
    print(f"Stats: {json.dumps(result['stats'])}")
    print("First parsed elements:")
    for index, item in enumerate(result["parsed_content_list"][:10]):
        print(f"{index}: {item['type']} | {item['content']}")


if __name__ == "__main__":
    main()
