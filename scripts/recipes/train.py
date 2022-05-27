from pathlib import Path
from typing import List, Union

from datasets import ClassLabel, Dataset
from datasets import Features, Image, Sequence, Value, load_dataset
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


def examples_to_hf_dataset(examples: List[TaskType]) -> Dataset:
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
        "image": [eg["image"] for eg in examples],
        # fmt: on
    }

    dataset = Dataset.from_dict(dataset_dict, features=features)
    dataset = dataset.cast_column("image", Image())
    return dataset


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
    dataset = examples_to_hf_dataset(examples)

    # Train model
    pass
