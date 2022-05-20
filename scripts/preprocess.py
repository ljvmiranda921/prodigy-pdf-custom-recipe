from pathlib import Path
import zipfile
import typer


def unzip_file(
    in_dir: Path = typer.Argument(..., exists=True),
    out_dir: Path = typer.Argument(..., exists=False),
):
    with zipfile.ZipFile(in_dir, "r") as f:
        f.extractall(out_dir)


if __name__ == "__main__":
    typer.run(unzip_file)
