from typing import List, Dict
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
from PIL import Image
from wasabi import msg
import numpy as np

from scripts.constants import BASE_MODEL


def _load_model(model_path: str):
    msg.text(f"Loading model from {model_path}")
    model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
    return model


def _load_processor(model: str = BASE_MODEL):
    msg.text(f"Loading processor {model}")
    processor = LayoutLMv3Processor.from_pretrained(model)
    return processor


def _unnormalize_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]


def infer(model_path: str, examples: List[Dict], labels: List[str]) -> List[str]:
    model = _load_model(model_path)
    processor = _load_processor()
    id2label = {v: k for v, k in enumerate(labels)}

    all_preds = []
    all_bboxes = []
    for eg in examples:
        image = Image.open(eg["path"]).convert("RGB")
        width, height = image.size

        # Encode the image
        encoding = processor(
            # fmt: off
            image, 
            truncation=True, 
            return_offsets_mapping=True, 
            return_tensors="pt"
            # fmt: on
        )
        offset_mapping = encoding.pop("offset_mapping")

        # Perform forward pass
        outputs = model(**encoding)

        # Get the predictions
        predictions = outputs.logits.argmax(-1).squeeze().tolist()
        token_boxes = encoding.bbox.squeeze().tolist()

        # Only keep non-subword predictions
        is_subword = np.array(offset_mapping.squeeze().tolist())[:, 0] != 0
        true_predictions = [
            id2label[pred]
            for idx, pred in enumerate(predictions)
            if not is_subword[idx]
        ]
        true_boxes = [
            _unnormalize_box(box, width, height)
            for idx, box in enumerate(token_boxes)
            if not is_subword[idx]
        ]

        all_preds.append(true_predictions)
        all_bboxes.append(true_boxes)

    breakpoint()
