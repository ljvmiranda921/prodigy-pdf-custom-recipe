from pathlib import Path

# The zipfile itself has a parent file named 'dataset', in addition to our
# own 'dataset' directory, hence the repetition
DATASET_PATH = Path(__file__).parent.parent / "dataset" / "dataset"

TRAIN_DATA = DATASET_PATH / "training_data"
TRAIN_IMAGES = TRAIN_DATA / "images"
TRAIN_LABELS = TRAIN_DATA / "annotations"

TEST_DATA = DATASET_PATH / "testing_data"
TEST_IMAGES = TEST_DATA / "images"
TEST_LABELS = TEST_DATA / "annotations"
