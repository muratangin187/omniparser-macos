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
    parser.add_argument("--box-threshold", type=float, default=0.05)
    parser.add_argument("--iou-threshold", type=float, default=0.1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ocr-batch-size", type=int, default=1)
    parser.add_argument("--ocr-canvas-size", type=int, default=2560)
    parser.add_argument("--use-paddleocr", action="store_true")
    parser.add_argument("--fast", action="store_true", help="Use tuned faster settings for Apple Silicon.")
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.fast:
        if args.device == "auto":
            args.device = "mps"
        args.box_threshold = 0.10
        args.imgsz = 512
        args.batch_size = 128
        args.ocr_batch_size = 16
        args.ocr_canvas_size = 1920

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
        }
    )
    result = omniparser.parse_image(
        image_path,
        box_threshold=args.box_threshold,
        iou_threshold=args.iou_threshold,
        use_paddleocr=args.use_paddleocr,
        easyocr_args={
            "text_threshold": 0.8,
            "batch_size": args.ocr_batch_size,
            "canvas_size": args.ocr_canvas_size,
        },
        imgsz=args.imgsz,
        batch_size=args.batch_size,
    )

    annotated_image = Image.open(io.BytesIO(base64.b64decode(result["annotated_image_base64"])))
    annotated_image.save(output_image_path)

    payload = {
        "input_image": str(image_path),
        "output_image": str(output_image_path),
        "label_coordinates": result["label_coordinates"],
        "parsed_content_list": result["parsed_content_list"],
    }
    output_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved annotated image to {output_image_path}")
    print(f"Saved parsed output to {output_json_path}")
    print("First parsed elements:")
    for index, item in enumerate(result["parsed_content_list"][:10]):
        print(f"{index}: {item['type']} | {item['content']}")


if __name__ == "__main__":
    main()
