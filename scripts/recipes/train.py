from pathlib import Path
from typing import List, Union

from wasabi import msg

from scripts.train_utils import compute_metrics, examples_to_hf_dataset
from scripts.train_utils import preprocess_dataset, setup_trainer

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
    dry_run=("Disable training", "flag", "D", bool),
    eval=("Perform evaluation", "flag", "E", bool),
    verbose=("Enable verbose logging", "flag", "V", bool),
)
def train_image(
    dataset: str,
    output_dir: Union[str, Path],
    dry_run: bool = False,
    eval: bool = False,
    verbose: bool = False,
):
    DB = db.connect()
    examples = get_examples(DB, set_id=dataset)
    hf_dataset = examples_to_hf_dataset(examples)
    msg.good(f"Loaded examples from '{dataset}'")

    train, dev, processor, id2label, label2id = preprocess_dataset(hf_dataset)
    msg.text("Training dataset features", show=verbose)
    for k, v in train[0].items():
        msg.text(f"{k}: {v.shape}", show=verbose)

    label_list = list(id2label.values())

    # Train the model
    msg.info(f"Training the model and saving the progress at '{output_dir}'")
    trainer = setup_trainer(
        train_dataset=train,
        dev_dataset=dev,
        processor=processor,
        id2label=id2label,
        label2id=label2id,
        compute_metrics=compute_metrics(label_list, return_entity_level_metrics=False),
        output_dir=output_dir,
    )

    if not dry_run:
        trainer.train()

    if not dry_run and eval:
        eval_scores = trainer.evaluate()
        msg.text("Evaluation scores")
        msg.text(eval_scores)
