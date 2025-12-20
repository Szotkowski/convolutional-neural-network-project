import os
import io
import json
import base64
from typing import List, Dict

import torch
from PIL import Image
from rfdetr import RFDETRMedium

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


def _clip_xyxy(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(0, min(int(round(x2)), w - 1))
    y2 = max(0, min(int(round(y2)), h - 1))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def _parse_dtype(s: str):
    s = (s or "fp16").lower()
    if s in ("fp32", "float32"): return torch.float32
    if s in ("bf16", "bfloat16"): return torch.bfloat16
    # default
    return torch.float16


def init_context(context):
    model_path = os.environ.get("MODEL_PATH", "/opt/nuclio/models/weights.pth")
    # Change object names to match your custom model
    classes_csv = os.environ.get("CLASSES", "lightbulb,sea_shell")
    default_thresh = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.5"))
    one_indexed = os.environ.get("CLASS_IDS_START_AT_ONE", "1") in ("1", "true", "True")
    compile_opt = os.environ.get("COMPILE_OPTIMIZED", "1") in ("1", "true", "True")

    if torch.cuda.is_available():
        device = "cuda"
        context.logger.info(f"CUDA is available. Using {torch.cuda.get_device_name(0)} GPU for inference.")
        default_infer_dtype = "fp16"
    else:
        device = "cpu"
        context.logger.info("CUDA is not available. Using CPU for inference.")
        default_infer_dtype = "fp32"

    infer_dtype = _parse_dtype(os.environ.get("INFER_DTYPE", default_infer_dtype))
    context.logger.info(f"Loading weights: {model_path}")
    model = RFDETRMedium(pretrain_weights=model_path, device=device)

    try:
        if compile_opt:
            if infer_dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported() and device == "cuda":
                context.logger.warn("bf16 not supported on this GPU; falling back to fp16")
                infer_dtype = torch.float16
            else:
                if infer_dtype is not torch.float32 and device == "cpu":
                    context.logger.warn("Only fp32 is supported on CPU; falling back to fp32")
                    infer_dtype = torch.float32

            model.optimize_for_inference(compile=True, batch_size=1, dtype=infer_dtype)
            context.logger.info(f"Optimized model compiled with dtype={infer_dtype}")
    except Exception as e:
        context.logger.warn(f"Could not optimize model for inference: {e}")

    context.user_data.model = model
    context.user_data.labels: List[str] = [c.strip() for c in classes_csv.split(",") if c.strip()]
    context.user_data.default_thresh: float = default_thresh
    context.user_data.one_indexed: bool = one_indexed


def handler(context, event):
    try:
        body = event.body
        if isinstance(body, (bytes, bytearray)):
            body = body.decode("utf-8")
        payload = json.loads(body) if isinstance(body, str) else body

        if "image" not in payload:
            raise ValueError("Missing 'image' field (base64) in request body.")

        thresh = float(payload.get("threshold", context.user_data.default_thresh))

        img_bytes = base64.b64decode(payload["image"])
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h = image.size

        detections = context.user_data.model.predict(image, threshold=thresh)

        labels = context.user_data.labels
        one_indexed = context.user_data.one_indexed

        results: List[Dict] = []
        for (x1, y1, x2, y2), cls_id, score in zip(
            detections.xyxy, detections.class_id, detections.confidence
        ):
            idx = int(cls_id)
            if one_indexed:
                idx -= 1
            label_name = labels[idx] if 0 <= idx < len(labels) else str(cls_id)
            xyxy = _clip_xyxy(x1, y1, x2, y2, w, h)

            results.append(
                {
                    "label": label_name,
                    "points": xyxy,
                    "type": "rectangle",
                    "confidence": float(score),
                }
            )

        return context.Response(
            body=json.dumps(results),
            content_type="application/json",
            status_code=200,
        )

    except Exception as e:
        context.logger.error(f"Error: {repr(e)}")
        return context.Response(
            body=json.dumps({"error": repr(e)}),
            content_type="application/json",
            status_code=500,
        )