import os
from typing import Dict, List, Tuple, Any

import numpy as np
import torch.nn.functional as nn
from PIL import Image
from tqdm import tqdm
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
from wasabi import msg

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


def _filter(
    values: List[Any], probabilities: List[float], threshold: float
) -> List[Any]:
    return [value for probs, value in zip(probabilities, values) if probs >= threshold]


def infer(
    model_path: str, examples: List[Dict], labels: List[str], threshold: float
) -> Tuple[List, List]:
    msg.info(f"Performing inference using the model at {model_path}")
    model = _load_model(model_path)
    processor = _load_processor()
    id2label = {v: k for v, k in enumerate(labels)}

    all_preds = []
    all_bboxes = []

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    msg.info(
        f"Tokenizer parallelism: {os.environ.get('TOKENIZERS_PARALLELISM', 'true')}"
    )

    for eg in tqdm(examples):
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

        # Get the predictions and probabilities
        probs = nn.softmax(outputs.logits.squeeze(), dim=1).max(dim=1).values.tolist()
        _predictions = outputs.logits.argmax(-1).squeeze().tolist()
        _token_boxes = encoding.bbox.squeeze().tolist()

        # Filter the predictions and bounding boxes based on a threshold
        predictions = _filter(_predictions, probs, threshold)
        token_boxes = _filter(_token_boxes, probs, threshold)

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

    return (all_preds, all_bboxes)
