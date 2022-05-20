import sys
from wasabi import msg

try:
    import prodigy
except ImportError:
    msg.fail("No installation of prodigy found")
    sys.exit()


@prodigy.recipe(
    "train-layout",
    dataset=("Dataset to train the model on", "positional", None, str),
    output_dir=("Output directory for the trained pipeline", "positional", None, str),
    verbose=("Enable verbose logging", "flag", "V", bool),
    silent=("Don't output any status or logs", "flag", "S", bool),
)
def train_layout_model(
    dataset: str, output_dir: str, verbose: bool = False, silent: bool = False
):
    pass
