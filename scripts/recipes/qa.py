import copy
import json
from pathlib import Path
from typing import Iterable, List, Optional, Union, Dict

from wasabi import msg

from scripts.infer_utils import infer, load_model
from scripts.train_utils import examples_to_hf_dataset
from scripts.recipes.loaders import get_image, get_labels, create_task

try:
    import prodigy
    from prodigy.components.loaders import Images
    from prodigy.types import StreamType
    from prodigy.util import split_string
except ImportError:
    msg.fail("No installation of prodigy found", exits=1)


def make_tasks(model_path: str, stream: StreamType) -> StreamType:
    """Add a 'spans' key to each example, with predicted entities."""
    # model = load_model(model_path)
    examples = examples_to_hf_dataset(list(stream), test_size=1.0)
    breakpoint()


def make_bboxes(bbox_path: str, stream: StreamType) -> StreamType:
    """Add bounding boxes to the spans key"""

    def _get_labels(id: str) -> Dict:
        annot_fp = Path(bbox_path) / f"{id}.json"
        with annot_fp.open() as f:
            annot = json.load(f)
        return annot

    for eg in stream:
        task = copy.deepcopy(eg)
        # The filename of the task (without extension) is in the 'text' key
        labels = _get_labels(task["text"])
        breakpoint()
        task["spans"] = [{"bbox": labels["box"], "text": labels["text"]}]
        yield task


@prodigy.recipe(
    "image.qa",
    # fmt: off
    dataset=("Dataset to save annotations to", "positional", None, str),
    source=("Data to qa/annotate (directory of images, file path or '-' to read from standard input)", "positional", None, str),
    bbox_path=("Path to the bounding box annotations file (this model doesn't have OCR installed)", "positional", None, str),
    model_path=("Path to the finetuned model for inference", "positional", None, str),
    label=("Comma-separated label(s) to annotate or text file with one label per line", "option", "l"),
    exclude=("Comma-separated list of dataset IDs whose annotations to exclude", "option", "e", split_string),
    darken=("Darken image to make boxes stand out more", "flag", "D", bool),
    # fmt: on
)
def qa(
    dataset: str,
    source: str,
    bbox_path: str,
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
    stream = make_bboxes(bbox_path, stream)
    breakpoint()
    make_tasks(model_path, stream)

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
