import argparse
import base64
import io

import gradio as gr
from PIL import Image

from util.omniparser import Omniparser


MARKDOWN = """
# OmniParser for Pure Vision Based General GUI Agent

Upload a screenshot to detect UI regions and label them with local descriptions.
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Run OmniParser in Gradio.")
    parser.add_argument("--weights-dir", default="weights")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--default-box-threshold", type=float, default=0.05)
    parser.add_argument("--default-iou-threshold", type=float, default=0.1)
    parser.add_argument("--default-imgsz", type=int, default=640)
    parser.add_argument("--default-use-paddleocr", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ocr-batch-size", type=int, default=1)
    parser.add_argument("--ocr-canvas-size", type=int, default=2560)
    return parser.parse_args()


def build_demo(omniparser: Omniparser, args):
    def process(image_input, box_threshold, iou_threshold, use_paddleocr, imgsz):
        result = omniparser.parse_image(
            image_input,
            box_threshold=box_threshold,
            iou_threshold=iou_threshold,
            use_paddleocr=use_paddleocr,
            imgsz=imgsz,
        )
        image = Image.open(io.BytesIO(base64.b64decode(result["annotated_image_base64"])))
        parsed_content = "\n".join(
            f"{index}: {item['type']} | {item['content']}"
            for index, item in enumerate(result["parsed_content_list"])
        )
        return image, parsed_content

    with gr.Blocks() as demo:
        gr.Markdown(MARKDOWN)
        with gr.Row():
            with gr.Column():
                image_input_component = gr.Image(type="pil", label="Upload image")
                box_threshold_component = gr.Slider(
                    label="Box Threshold",
                    minimum=0.01,
                    maximum=1.0,
                    step=0.01,
                    value=args.default_box_threshold,
                )
                iou_threshold_component = gr.Slider(
                    label="IOU Threshold",
                    minimum=0.01,
                    maximum=1.0,
                    step=0.01,
                    value=args.default_iou_threshold,
                )
                use_paddleocr_component = gr.Checkbox(
                    label="Use PaddleOCR",
                    value=args.default_use_paddleocr,
                )
                imgsz_component = gr.Slider(
                    label="Icon Detect Image Size",
                    minimum=640,
                    maximum=1920,
                    step=32,
                    value=args.default_imgsz,
                )
                submit_button_component = gr.Button(value="Submit", variant="primary")
            with gr.Column():
                image_output_component = gr.Image(type="pil", label="Image Output")
                text_output_component = gr.Textbox(
                    label="Parsed screen elements",
                    placeholder="Text Output",
                    lines=20,
                )

        submit_button_component.click(
            fn=process,
            inputs=[
                image_input_component,
                box_threshold_component,
                iou_threshold_component,
                use_paddleocr_component,
                imgsz_component,
            ],
            outputs=[image_output_component, text_output_component],
        )
    return demo


def main():
    args = parse_args()
    omniparser = Omniparser(
        {
            "weights_dir": args.weights_dir,
            "device": args.device,
            "BOX_TRESHOLD": args.default_box_threshold,
            "iou_threshold": args.default_iou_threshold,
            "use_paddleocr": args.default_use_paddleocr,
            "easyocr_args": {
                "text_threshold": 0.8,
                "batch_size": args.ocr_batch_size,
                "canvas_size": args.ocr_canvas_size,
            },
            "imgsz": args.default_imgsz,
            "batch_size": args.batch_size,
        }
    )
    demo = build_demo(omniparser, args)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
