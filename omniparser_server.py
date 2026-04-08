import argparse
import base64
import io
import json
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel, Field

from omniparser_cli import OUTPUT_FORMATS
from omniparser_worker import build_parser_config
from util.omniparser import Omniparser


def parse_args():
    parser = argparse.ArgumentParser(description="Run OmniParser as a local HTTP server.")
    parser.add_argument("--weights-dir", default="weights")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--som-device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--preset", default="turbo", choices=["full", "balanced", "recall", "turbo", "ultra"])
    parser.add_argument("--box-threshold", type=float, default=None)
    parser.add_argument("--iou-threshold", type=float, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--ocr-batch-size", type=int, default=None)
    parser.add_argument("--ocr-canvas-size", type=int, default=None)
    parser.add_argument("--icon-crop-size", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--use-paddleocr", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--no-ocr", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--no-semantics", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--scale-img", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--png-compress-level", type=int, default=1)
    parser.add_argument("--output-format", default="jpg", choices=sorted(OUTPUT_FORMATS))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


class ParsePathRequest(BaseModel):
    image_path: str
    output_dir: Optional[str] = None
    output_format: Optional[str] = Field(default=None, pattern="^(png|jpg|bmp)$")
    include_annotated_image: bool = False
    enable_ocr: Optional[bool] = None
    use_local_semantics: Optional[bool] = None
    scale_img: Optional[bool] = None
    box_threshold: Optional[float] = None
    iou_threshold: Optional[float] = None
    imgsz: Optional[int] = None
    batch_size: Optional[int] = None
    icon_crop_size: Optional[int] = None
    max_new_tokens: Optional[int] = None


def get_save_kwargs(output_format: str, png_compress_level: int):
    image_format, image_suffix = OUTPUT_FORMATS[output_format]
    save_kwargs = {"format": image_format}
    if image_format == "PNG":
        save_kwargs["compress_level"] = png_compress_level
    elif image_format == "JPEG":
        save_kwargs["quality"] = 90
    return image_format, image_suffix, save_kwargs


def serialize_image(image, output_format: str, png_compress_level: int):
    _, image_suffix, save_kwargs = get_save_kwargs(output_format, png_compress_level)

    buffer = io.BytesIO()
    image.save(buffer, **save_kwargs)
    mime_type = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "bmp": "image/bmp",
    }[output_format]
    return base64.b64encode(buffer.getvalue()).decode("ascii"), image_suffix, mime_type, save_kwargs


def save_outputs(result, image_name: str, output_dir: Path, output_format: str, png_compress_level: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_name).stem or "image"
    _, image_suffix, save_kwargs = get_save_kwargs(output_format, png_compress_level)
    mime_type = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "bmp": "image/bmp",
    }[output_format]
    output_image_path = output_dir / f"{stem}_annotated{image_suffix}"
    output_json_path = output_dir / f"{stem}_parsed.json"
    result["annotated_image"].save(output_image_path, **save_kwargs)
    payload = {
        "input_image": image_name,
        "output_image": str(output_image_path),
        "label_coordinates": result["label_coordinates"],
        "parsed_content_list": result["parsed_content_list"],
        "stats": result["stats"],
    }
    output_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "output_image": str(output_image_path),
        "output_json": str(output_json_path),
        "output_image_mime_type": mime_type,
        "items": len(result["parsed_content_list"]),
    }


def build_response(result, image_name: str, output_dir: Optional[str], output_format: str, include_annotated_image: bool, png_compress_level: int):
    response = {
        "ok": True,
        "input_image": image_name,
        "stats": result["stats"],
        "label_coordinates": result["label_coordinates"],
        "parsed_content_list": result["parsed_content_list"],
    }
    if output_dir:
        response.update(
            save_outputs(
                result,
                image_name=image_name,
                output_dir=Path(output_dir).expanduser().resolve(),
                output_format=output_format,
                png_compress_level=png_compress_level,
            )
        )
    if include_annotated_image:
        annotated_image_base64, _, mime_type, _ = serialize_image(
            result["annotated_image"],
            output_format=output_format,
            png_compress_level=png_compress_level,
        )
        response["annotated_image_base64"] = annotated_image_base64
        response["annotated_image_mime_type"] = mime_type
    return response


def parse_kwargs_from_request(
    enable_ocr=None,
    use_local_semantics=None,
    scale_img=None,
    box_threshold=None,
    iou_threshold=None,
    imgsz=None,
    batch_size=None,
    icon_crop_size=None,
    max_new_tokens=None,
):
    kwargs = {}
    if enable_ocr is not None:
        kwargs["enable_ocr"] = enable_ocr
    if use_local_semantics is not None:
        kwargs["use_local_semantics"] = use_local_semantics
    if scale_img is not None:
        kwargs["scale_img"] = scale_img
    if box_threshold is not None:
        kwargs["box_threshold"] = box_threshold
    if iou_threshold is not None:
        kwargs["iou_threshold"] = iou_threshold
    if imgsz is not None:
        kwargs["imgsz"] = imgsz
    if batch_size is not None:
        kwargs["batch_size"] = batch_size
    if icon_crop_size is not None:
        kwargs["icon_crop_size"] = icon_crop_size
    if max_new_tokens is not None:
        kwargs["max_new_tokens"] = max_new_tokens
    return kwargs


def create_app(args):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        parser = Omniparser(build_parser_config(args))
        if not args.no_warmup:
            parser.warmup()
        app.state.parser = parser
        app.state.lock = threading.Lock()
        yield

    app = FastAPI(title="OmniParser Server", version="1.0.0", lifespan=lifespan)

    @app.get("/")
    def index():
        return {
            "ok": True,
            "service": "omniparser",
            "preset": args.preset,
            "device": args.device,
            "som_device": args.som_device,
            "box_threshold": app.state.parser.config["BOX_TRESHOLD"],
            "imgsz": app.state.parser.config["imgsz"],
            "scale_img": app.state.parser.config["scale_img"],
            "default_output_format": args.output_format,
            "endpoints": ["/healthz", "/parse", "/parse-path"],
        }

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "status": "ready", "preset": args.preset}

    @app.post("/parse-path")
    def parse_path(request: ParsePathRequest):
        image_path = Path(request.image_path).expanduser().resolve()
        if not image_path.exists():
            raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")
        output_format = request.output_format or args.output_format
        parse_kwargs = parse_kwargs_from_request(
            enable_ocr=request.enable_ocr,
            use_local_semantics=request.use_local_semantics,
            scale_img=request.scale_img,
            box_threshold=request.box_threshold,
            iou_threshold=request.iou_threshold,
            imgsz=request.imgsz,
            batch_size=request.batch_size,
            icon_crop_size=request.icon_crop_size,
            max_new_tokens=request.max_new_tokens,
        )

        with app.state.lock:
            result = app.state.parser.parse_image(image_path, encode_output=False, **parse_kwargs)

        return build_response(
            result,
            image_name=str(image_path),
            output_dir=request.output_dir,
            output_format=output_format,
            include_annotated_image=request.include_annotated_image,
            png_compress_level=args.png_compress_level,
        )

    @app.post("/parse")
    def parse_upload(
        image: UploadFile = File(...),
        output_dir: Optional[str] = Form(default=None),
        output_format: str = Form(default=args.output_format),
        include_annotated_image: bool = Form(default=True),
        enable_ocr: Optional[bool] = Form(default=None),
        use_local_semantics: Optional[bool] = Form(default=None),
        scale_img: Optional[bool] = Form(default=None),
        box_threshold: Optional[float] = Form(default=None),
        iou_threshold: Optional[float] = Form(default=None),
        imgsz: Optional[int] = Form(default=None),
        batch_size: Optional[int] = Form(default=None),
        icon_crop_size: Optional[int] = Form(default=None),
        max_new_tokens: Optional[int] = Form(default=None),
    ):
        if output_format not in OUTPUT_FORMATS:
            raise HTTPException(status_code=400, detail=f"Unsupported output format: {output_format}")
        try:
            image_bytes = image.file.read()
            parsed_image = Image.open(io.BytesIO(image_bytes))
            parse_kwargs = parse_kwargs_from_request(
                enable_ocr=enable_ocr,
                use_local_semantics=use_local_semantics,
                scale_img=scale_img,
                box_threshold=box_threshold,
                iou_threshold=iou_threshold,
                imgsz=imgsz,
                batch_size=batch_size,
                icon_crop_size=icon_crop_size,
                max_new_tokens=max_new_tokens,
            )
            with app.state.lock:
                result = app.state.parser.parse_image(parsed_image, encode_output=False, **parse_kwargs)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to parse uploaded image: {exc}") from exc

        return build_response(
            result,
            image_name=image.filename or "upload.png",
            output_dir=output_dir,
            output_format=output_format,
            include_annotated_image=include_annotated_image,
            png_compress_level=args.png_compress_level,
        )

    return app


def main():
    args = parse_args()
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
