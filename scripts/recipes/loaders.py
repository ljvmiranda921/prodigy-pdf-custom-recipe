import base64
import json
from pathlib import Path
from typing import Dict, List, Union

from wasabi import msg

from scripts.constants import TRAIN_IMAGES, TRAIN_LABELS

try:
    import prodigy
    from prodigy import set_hashes
    from prodigy.components.db import connect
except ImportError:
    msg.fail("No installation of prodigy found", exits=1)


def _to_poly(bbox: List) -> List[List[int]]:
    """Convert bounding box into polygon coordinates"""
    x_left, y_top, x_right, y_bottom = bbox
    return [
        [x_left, y_top],
        [x_right, y_top],
        [x_right, y_bottom],
        [x_left, y_bottom],
    ]


def get_ids(reference: Path) -> List:
    files = reference.glob("**/*")
    ids = [f.stem for f in files if f.is_file()]
    return ids


def get_image(id: str, encode_b64: bool, images_dir: Union[Path, str]) -> str:
    """Get the image and return its path or base64 representation"""
    image_fp = images_dir / f"{id}.png"
    with open(image_fp, "rb") as f:
        img = f.read()
    source = base64.encodebytes(img).decode("utf-8") if encode_b64 else str(image_fp)
    return source


def get_labels(id: str, labels_dir: Union[Path, str]) -> Dict:
    """Get the labels given a file ID"""
    annot_fp = labels_dir / f"{id}.json"
    with open(annot_fp) as f:
        annot = json.load(f)
    return annot


def create_task(image: str, label: Dict, answer: str = "accept") -> Dict:
    """Create a Prodigy task given the image and its label"""
    # https://prodi.gy/docs/api-interfaces#image_manual
    spans = [{"points": _to_poly(a["box"]), "label": a["label"]} for a in label["form"]]
    task = set_hashes({"image": image, "spans": spans, "answer": answer})
    return task


@prodigy.recipe(
    "db-in-image",
    # fmt: off
    set_id=("Dataset to import annotations to", "positional", None, str),
    images_dir=("Path to the images directory", "option", "i", str),
    labels_dir=("Path to the annotations directory", "option", "l", str),
    answer=("Set this answer key if none is present", "option", "a", str),
    encode_b64=("Encode to base64 before storing to db", "flag", "e", bool)
    # fmt: on
)
def db_in_image(
    set_id: str,
    images_dir: Union[str, Path] = TRAIN_IMAGES,
    labels_dir: Union[str, Path] = TRAIN_LABELS,
    answer: str = "accept",
    encode_b64: bool = False,
):
    """Import annotations to the database

    This assumes that the filename of the annotations is the same as the
    filename of the images. The images can be encoded into their base-64
    representation and then saved to the database.
    """

    # Get all file IDs
    ids = get_ids(labels_dir)
    msg.info(f"Found {len(ids)} images and annotations")

    examples = []
    for id in ids:
        image = get_image(id, encode_b64=encode_b64, images_dir=images_dir)
        label = get_labels(id, labels_dir=labels_dir)
        task = create_task(image, label, answer)
        examples.append(task)

    db = connect()
    db.add_dataset(set_id)
    db.add_examples(examples, [set_id])
