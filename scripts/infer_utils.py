from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor


def load_model(model_path: str):
    model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
    return model


def infer(example, model):
    pass
