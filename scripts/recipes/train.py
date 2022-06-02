from pathlib import Path
from platform import processor
from typing import List, Union, Dict

from datasets import ClassLabel, Dataset, Array2D, Array3D
from datasets import Features, Image, Sequence, Value, load_metric
from PIL import Image as PILImage
from transformers import AutoProcessor
from wasabi import msg


try:
    import prodigy
    from prodigy import set_hashes
    from prodigy.components import db
    from prodigy.types import TaskType
except ImportError:
    msg.fail("No installation of prodigy found", exits=1)


CLASS_NAMES = ["other", "header", "question", "answer"]


def get_examples(database: db.Database, set_id: str) -> List[TaskType]:
    """Load examples given a dataset"""
    if set_id not in database:
        msg.fail(f"Can't find '{set_id}' in database '{database.db_name}'", exits=1)
    examples = database.get_dataset(set_id)

    result = []
    for eg in examples:
        if eg["answer"] != "ignore":
            result.append(set_hashes(eg, overwrite=True))
    return result


def examples_to_hf_dataset(examples: List[TaskType], test_size: float = 0.2) -> Dataset:
    """Convert a list of Prodigy examples into a dataset"""
    features = Features(
        {
            # fmt: off
            "id": Value(dtype="string", id=None),
            "tokens": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
            "bboxes": Sequence(feature=Sequence(feature=Value(dtype="int64", id=None), length=-1, id=None), length=-1, id=None),
            "ner_tags": Sequence(feature=ClassLabel(num_classes=len(CLASS_NAMES), names=CLASS_NAMES, id=None), length=-1, id=None),
            "image": Image(decode=True, id=None)
            # fmt: on
        }
    )

    dataset_dict = {
        # fmt: off
        "id": list(range(0, len(examples))),
        "tokens": [[span["text"] for span in eg["spans"]] for eg in examples],
        "bboxes": [[span["bbox"] for span in eg["spans"]] for eg in examples],
        "ner_tags": [[CLASS_NAMES.index(span["label"]) for span in eg["spans"]] for eg in examples],
        "image": [PILImage.open(eg["image"]).convert("RGB") for eg in examples],
        # fmt: on
    }

    dataset = Dataset.from_dict(dataset_dict, features=features).train_test_split(
        test_size=test_size
    )
    return dataset


def preprocess_dataset(
    dataset: Dataset, model: str = "microsoft/layoutlmv3-base"
) -> Union[Dataset, Dataset, Dict, Dict]:
    """Preprocess the whole dataset to make it compatible with the LayoutLMv3 model

    Source: https://github.com/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv3/Fine_tune_LayoutLMv3_on_FUNSD_(HuggingFace_Trainer).ipynb

    Returns the processed trained and eval datasets and id2label and label2id
    mappings respectively.
    """
    processor = AutoProcessor.from_pretrained(model, apply_ocr=False)

    features = dataset["train"].features
    column_names = dataset["train"].column_names
    image_column_name = "image"
    text_column_name = "tokens"
    boxes_column_name = "bboxes"
    label_column_name = "ner_tags"

    def _get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    if isinstance(features[label_column_name].feature, ClassLabel):
        label_list = features[label_column_name].feature.names
    else:
        label_list = _get_label_list(dataset["train"][label_column_name])

    id2label = {k: v for k, v in enumerate(label_list)}
    label2id = {v: k for k, v in enumerate(label_list)}

    def _prepare_examples(examples):
        """Mapper function"""
        images = examples[image_column_name]
        words = examples[text_column_name]
        boxes = examples[boxes_column_name]
        word_labels = examples[label_column_name]
        encoding = processor(
            images,
            words,
            boxes=boxes,
            word_labels=word_labels,
            truncation=True,
            padding="max_length",
        )
        return encoding

    new_features = Features(
        {
            "pixel_values": Array3D(dtype="float32", shape=(3, 224, 224)),
            "input_ids": Sequence(feature=Value(dtype="int64")),
            "attention_mask": Sequence(Value(dtype="int64")),
            "bbox": Array2D(dtype="int64", shape=(512, 4)),
            "labels": Sequence(ClassLabel(names=label_list)),
        }
    )

    train_dataset = dataset["train"].map(
        _prepare_examples,
        batched=True,
        remove_columns=column_names,
        features=new_features,
    )
    train_dataset.set_format("torch")
    dev_dataset = dataset["test"].map(
        _prepare_examples,
        batched=True,
        remove_columns=column_names,
        features=new_features,
    )

    return train_dataset, dev_dataset, id2label, label2id


@prodigy.recipe(
    "train-image",
    dataset=("Dataset to train the model on", "positional", None, str),
    output_dir=("Output directory for the trained pipeline", "positional", None, str),
    verbose=("Enable verbose logging", "flag", "V", bool),
    silent=("Don't output any status or logs", "flag", "S", bool),
)
def train_image(
    dataset: str,
    output_dir: Union[str, Path],
    verbose: bool = False,
    silent: bool = False,
):
    DB = db.connect()
    examples = get_examples(DB, set_id=dataset)
    hf_dataset = examples_to_hf_dataset(examples)
    msg.good(f"Loaded examples from '{dataset}'")

    train, dev, id2label, label2id = preprocess_dataset(hf_dataset)
    msg.text("Training dataset features")
    for k, v in train[0].items():
        msg.text(f"{k}: {v.shape}", show=verbose)

    # Train model
    pass
