import copy
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

from wasabi import msg

from scripts.constants import CLASS_NAMES
from scripts.infer_utils import infer
from scripts.recipes.loaders import _to_poly

try:
    import prodigy
    from prodigy.components.loaders import Images
    from prodigy.types import StreamType
    from prodigy.util import get_labels, split_string, set_hashes
except ImportError:
    msg.fail("No installation of prodigy found", exits=1)


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


def make_labels(model_path: str, stream: StreamType, threshold: float) -> StreamType:
    """Add the predicted labels in the 'labels' key of the image spans"""
    examples = list(stream)
    predictions = infer(
        model_path, examples=examples, labels=CLASS_NAMES, threshold=threshold
    )
    labels, bboxes = predictions

    for eg, label, bbox in zip(examples, labels, bboxes):
        task = copy.deepcopy(eg)
        # Repace the spans with predicted values
        task["spans"] = [
            {"points": _to_poly(b), "label": l} for l, b in zip(label, bbox)
        ]
        task = set_hashes(task)
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
    threshold=("Threshold to filter the predictions (0 - 1)", "option", "t", float),
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
    threshold: int = 0.9,
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
    stream = make_labels(model_path, stream, threshold)

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
