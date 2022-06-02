from pathlib import Path
from typing import List, Union

from wasabi import msg

from scripts.train_utils import examples_to_hf_dataset, preprocess_dataset

try:
    import prodigy
    from prodigy import set_hashes
    from prodigy.components import db
    from prodigy.types import TaskType
except ImportError:
    msg.fail("No installation of prodigy found", exits=1)


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
