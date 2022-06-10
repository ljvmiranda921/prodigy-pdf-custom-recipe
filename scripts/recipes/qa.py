from pathlib import Path
from typing import Iterable, List, Optional, Union

from torch import Stream
from wasabi import msg

from scripts.infer_utils import infer, load_model
from scripts.train_utils import examples_to_hf_dataset

try:
    import prodigy
    from prodigy.components.loaders import Images
    from prodigy.types import StreamType
    from prodigy.util import get_labels, split_string
except ImportError:
    msg.fail("No installation of prodigy found", exits=1)


def make_tasks(model_path: str, stream: StreamType) -> StreamType:
    """Add a 'spans' key to each example, with predicted entities."""
    model = load_model(model_path)
    examples = [examples_to_hf_dataset(eg) for eg in stream]


@prodigy.recipe(
    "image.qa",
    # fmt: off
    dataset=("Dataset to save annotations to", "positional", None, str),
    source=("Data to qa/annotate (directory of images, file path or '-' to read from standard input)", "positional", None, str),
    model_path=("Path to the finetuned model for inference", "positional", str),
    label=("Comma-separated label(s) to annotate or text file with one label per line", "option", "l", get_labels),
    exclude=("Comma-separated list of dataset IDs whose annotations to exclude", "option", "e", split_string),
    darken=("Darken image to make boxes stand out more", "flag", "D", bool),
    # fmt: on
)
def qa(
    dataset: str,
    source: Union[str, Iterable[dict]],
    model_path: str,
    label: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    darken: bool = False,
):
    """
    Annotate documents with the help of a layout model.
    """
    # Much of the proceeding blocks of code is based from image.manual
    # Source: https://github.com/explosion/prodigy-recipes/blob/master/image/image_manual.py
    stream = Images(source)

    return {
        "view_id": "image_manual",  # Annotation interface to use
        "dataset": dataset,  # Name of dataset to save annotations
        "stream": stream,  # Incoming stream of examples
        "exclude": exclude,  # List of dataset names to exclude
        "config": {  # Additional config settings, mostly for app UI
            "label": ", ".join(label) if label is not None else "all",
            "labels": label,  # Selectable label options,
            "darken_image": 0.3 if darken else 0,
        },
    }
