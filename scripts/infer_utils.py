from typing import List, Dict
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
from PIL import Image

from scripts.constants import BASE_MODEL


def _load_model(model_path: str):
    model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
    return model


def _load_processor(model: str = BASE_MODEL):
    processor = LayoutLMv3Processor.from_pretrained(model)
    return processor


# def _process_image(image, model, processor, labels, id2label):
#     width, height = image.size

#     # Encode the image


def infer(model_path: str, examples: List[Dict], labels: List[str]) -> List[str]:
    model = _load_model(model_path)
    processor = _load_processor()
    id2label = {v: k for v, k in enumerate(labels)}

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
        breakpoint()
