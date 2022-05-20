import base64
import json
import sys
import zipfile
from pathlib import Path
from typing import Union


from prodigy.components import connect
from wasabi import msg

try:
    import prodigy
except ImportError:
    msg.fail("No installation of prodigy found")
    raise


@prodigy.recipe(
    "db-in-image",
    set_id=("Dataset to import annotations to", "positional", None, str),
    in_dir=("Path to images annotations directory", None, str),
    answer=("Set this answer key if none is present", "option", "a", str),
)
def db_in_image(set_id: str, in_file: Union[str, Path], answer: str = "accept"):
    """Import annotations to the database

    This assumes that the filename of the annotations is the same as the
    filename of the images. The images are encoded into its base-64
    representation and then saved to the database.

    """
    pass


def unzip_file(path: Path):
    pass
