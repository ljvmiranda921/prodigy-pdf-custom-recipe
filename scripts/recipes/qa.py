import copy
import json
from pathlib import Path
from typing import Iterable, List, Optional, Union, Dict

from wasabi import msg
from scripts.infer_utils import infer
from scripts.constants import CLASS_NAMES

try:
    import prodigy
    from prodigy.components.loaders import Images
    from prodigy.types import StreamType
    from prodigy.util import split_string, get_labels
except ImportError:
    msg.fail("No installation of prodigy found", exits=1)


def make_labels(model_path: str, stream: StreamType) -> StreamType:
    """Add the predicted labels in the 'labels' key of the image spans"""
    predictions = infer(model_path, examples=list(stream), labels=CLASS_NAMES)

    for eg, pred in zip(stream, predictions):
        task = copy.deepcopy(eg)
        task["spans"]["labels"] = pred
        yield task


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
        label = _get_labels(task["text"])
        task["spans"] = [{"bbox": a["box"], "text": a["text"]} for a in label["form"]]
        yield task


@prodigy.recipe(
    "image.qa",
    # fmt: off
    dataset=("Dataset to save annotations to", "positional", None, str),
    source=("Data to qa/annotate (directory of images, file path or '-' to read from standard input)", "positional", None, str),
    bbox_path=("Path to the bounding box annotations file (this model doesn't have OCR installed)", "positional", None, str),
    model_path=("Path to the finetuned model for inference", "positional", None, str),
    label=("Comma-separated label(s) to annotate or text file with one label per line", "option", "l", get_labels),
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
    # Update the stream to add bounding boxes (based from annotations) and labels (based from the
    # finetuned model). It's possible to update the make_bboxes function and replace it with an
    # actual OCR engine. For now, we'll just use what's been annotated.
    stream = make_bboxes(bbox_path, stream)
    stream = make_labels(model_path, stream)

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
