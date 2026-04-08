import base64
import io
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from PIL import Image
from util.utils import get_som_labeled_img, get_caption_model_processor, get_yolo_model, check_ocr_box


DEFAULT_CONFIG = {
    "weights_dir": "weights",
    "caption_model_name": "florence2",
    "device": "auto",
    "BOX_TRESHOLD": 0.05,
    "iou_threshold": 0.1,
    "use_paddleocr": False,
    "easyocr_args": {"text_threshold": 0.8},
    "imgsz": 640,
    "batch_size": 128,
    "icon_crop_size": 64,
    "max_new_tokens": 20,
    "enable_ocr": True,
    "use_local_semantics": True,
    "scale_img": False,
    "som_device": "auto",
    "verbose": False,
}


def resolve_device(device: Optional[str]) -> str:
    if device and device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_som_device(device: Optional[str]) -> str:
    if device and device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Omniparser(object):
    def __init__(self, config: Dict):
        merged = {**DEFAULT_CONFIG, **config}
        merged["device"] = resolve_device(merged.get("device"))
        merged["som_device"] = resolve_som_device(merged.get("som_device"))
        weights_dir = Path(merged["weights_dir"])
        merged.setdefault("som_model_path", str(weights_dir / "icon_detect" / "model.pt"))
        merged.setdefault("caption_model_path", str(weights_dir / "icon_caption_florence"))
        self.config = merged

        self.som_model = get_yolo_model(model_path=self.config['som_model_path'])
        self.caption_cache: Dict[str, str] = {}
        self.caption_model_processor = get_caption_model_processor(
            model_name=self.config['caption_model_name'],
            model_name_or_path=self.config['caption_model_path'],
            device=self.config['device'],
        )
        if self.config.get("verbose"):
            print('Omniparser initialized!!!')

    def parse_image(
        self,
        image_source: Union[str, Path, Image.Image],
        enable_ocr: Optional[bool] = None,
        box_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        use_paddleocr: Optional[bool] = None,
        easyocr_args: Optional[Dict[str, Any]] = None,
        use_local_semantics: Optional[bool] = None,
        scale_img: Optional[bool] = None,
        imgsz: Optional[int] = None,
        batch_size: Optional[int] = None,
        icon_crop_size: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        if isinstance(image_source, (str, Path)):
            image = Image.open(image_source)
        else:
            image = image_source

        enable_ocr = self.config["enable_ocr"] if enable_ocr is None else enable_ocr
        box_threshold = self.config["BOX_TRESHOLD"] if box_threshold is None else box_threshold
        iou_threshold = self.config["iou_threshold"] if iou_threshold is None else iou_threshold
        use_paddleocr = self.config["use_paddleocr"] if use_paddleocr is None else use_paddleocr
        easyocr_args = self.config["easyocr_args"] if easyocr_args is None else easyocr_args
        use_local_semantics = self.config["use_local_semantics"] if use_local_semantics is None else use_local_semantics
        scale_img = self.config["scale_img"] if scale_img is None else scale_img
        imgsz = self.config["imgsz"] if imgsz is None else imgsz
        batch_size = self.config["batch_size"] if batch_size is None else batch_size
        icon_crop_size = self.config["icon_crop_size"] if icon_crop_size is None else icon_crop_size
        max_new_tokens = self.config["max_new_tokens"] if max_new_tokens is None else max_new_tokens

        stats: Dict[str, Any] = {
            "enable_ocr": enable_ocr,
            "use_local_semantics": use_local_semantics,
            "scale_img": scale_img,
            "box_threshold": box_threshold,
            "imgsz": imgsz,
            "batch_size": batch_size,
            "icon_crop_size": icon_crop_size,
            "max_new_tokens": max_new_tokens,
        }
        total_started = time.time()

        box_overlay_ratio = max(image.size) / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }

        if enable_ocr:
            ocr_started = time.time()
            (text, ocr_bbox), _ = check_ocr_box(
                image,
                display_img=False,
                output_bb_format='xyxy',
                easyocr_args=easyocr_args,
                use_paddleocr=use_paddleocr,
            )
            stats["ocr_ms"] = round((time.time() - ocr_started) * 1000, 2)
            stats["ocr_boxes"] = len(text)
        else:
            text, ocr_bbox = [], None
            stats["ocr_ms"] = 0.0
            stats["ocr_boxes"] = 0

        dino_labeled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
            image,
            self.som_model,
            BOX_TRESHOLD=box_threshold,
            output_coord_in_ratio=True,
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config,
            caption_model_processor=self.caption_model_processor,
            ocr_text=text,
            use_local_semantics=use_local_semantics,
            iou_threshold=iou_threshold,
            scale_img=scale_img,
            imgsz=imgsz,
            batch_size=batch_size,
            icon_crop_size=icon_crop_size,
            max_new_tokens=max_new_tokens,
            caption_cache=self.caption_cache,
            stats=stats,
            device=self.config["som_device"],
        )
        stats["total_ms"] = round((time.time() - total_started) * 1000, 2)

        return {
            "annotated_image_base64": dino_labeled_img,
            "label_coordinates": label_coordinates,
            "parsed_content_list": parsed_content_list,
            "stats": stats,
        }

    def parse(self, image_base64: str):
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        result = self.parse_image(image)
        return result["annotated_image_base64"], result["parsed_content_list"]
