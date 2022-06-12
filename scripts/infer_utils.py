from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

from scripts.constants import BASE_MODEL


def load_model(model_path: str):
    model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
    return model


def load_processor(model: str = BASE_MODEL):
    processor = LayoutLMv3Processor.from_pretrained(model)
    return processor


def infer(example, model):
    pass
