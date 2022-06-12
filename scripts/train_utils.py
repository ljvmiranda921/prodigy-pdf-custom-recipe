from typing import Callable, Dict, List, NamedTuple, Union

import numpy as np
from datasets import Array2D, Array3D, ClassLabel, Dataset, Features, Image
from datasets import Sequence, Value, load_metric
from PIL import Image as PILImage
from transformers import AutoProcessor, LayoutLMv3ForTokenClassification
from transformers import Trainer, TrainingArguments
from transformers.data.data_collator import default_data_collator
from wasabi import msg

try:
    from prodigy.types import TaskType
except ImportError:
    msg.fail("No installation of prodigy found", exits=1)


from scripts.constants import BASE_MODEL, CLASS_NAMES


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


def preprocess_dataset(dataset: Dataset, model: str = BASE_MODEL):
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

    return train_dataset, dev_dataset, processor, id2label, label2id


def compute_metrics(
    label_list: List, return_entity_level_metrics: bool = False
) -> Callable:
    """Returns a metric function that is passed to the Trainer for computing metrics

    Source: https://github.com/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv3/Fine_tune_LayoutLMv3_on_FUNSD_(HuggingFace_Trainer).ipynb
    """
    metric = load_metric("seqeval")

    def _compute_metrics(p: NamedTuple) -> Dict:
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        if return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

    return _compute_metrics


def setup_trainer(
    train_dataset: Dataset,
    dev_dataset: Dataset,
    processor,
    id2label: Dict,
    label2id: Dict,
    compute_metrics: Callable,
    output_dir: str,
    max_steps: int = 1000,
    per_device_train_batch_size: int = 2,
    per_device_eval_batch_size: int = 2,
    learning_rate: float = 1e-5,
    evaluation_strategy: str = "steps",
    eval_steps: int = 100,
    load_best_model_at_end: bool = True,
    metric_for_best_model: str = "f1",
) -> Trainer:
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        BASE_MODEL, id2label=id2label, label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=processor,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )

    return trainer
