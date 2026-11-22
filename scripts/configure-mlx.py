"""Register an existing Whisper large-v3-turbo MLX snapshot for Handy MLX."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
from tempfile import NamedTemporaryFile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, help="Local MLX snapshot directory")
    parser.add_argument("--data-dir", type=Path, default=Path.home() /
                        "Library/Application Support/com.pais.handy.mlx")
    args = parser.parse_args()

    if sys.platform != "darwin" or platform.machine() != "arm64":
        parser.exit(1, "Use an arm64 Python on an Apple Silicon Mac.\n")
    try:
        import mlx_whisper  # The app starts this same Python in isolated mode.
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import LocalEntryNotFoundError
    except ImportError as error:
        parser.exit(1, f"Install mlx-whisper in this Python environment first: {error}\n")

    snapshot = args.snapshot
    if snapshot is None:
        try:
            snapshot = Path(snapshot_download("mlx-community/whisper-large-v3-turbo",
                                              local_files_only=True))
        except LocalEntryNotFoundError:
            parser.exit(1, "No cached main revision found. Pass --snapshot /path/to/snapshot.\n")
    snapshot = snapshot.expanduser().resolve()
    try:
        config = json.loads((snapshot / "config.json").read_text())
    except (OSError, ValueError) as error:
        parser.exit(1, f"Cannot read the snapshot configuration: {error}\n")
    if (not (snapshot / "weights.safetensors").is_file()
            or config.get("n_mels") != 128 or config.get("n_text_layer") != 4
            or config.get("n_audio_layer") != 32):
        parser.exit(1, "Expected a Whisper large-v3-turbo MLX safetensors snapshot.\n")

    data_dir = args.data_dir.expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / "whisper-mlx.json"
    # Preserve the venv executable path. Resolving its symlink would select base Python.
    desired = {"python": sys.executable, "snapshot": str(snapshot)}
    with NamedTemporaryFile("w", dir=data_dir, delete=False) as fp:
        temp = Path(fp.name)
        json.dump(desired, fp, indent=2)
        fp.write("\n")
    try:
        try:
            os.link(temp, target)
        except FileExistsError:
            if json.loads(target.read_text()) != desired:
                parser.exit(1, f"An existing configuration is in use. To register a different environment or snapshot, move {target} to a backup path and rerun setup.\n")
    finally:
        temp.unlink(missing_ok=True)
    print(f"Configured shared:whisper-mlx in {target}\nSnapshot: {snapshot}")


if __name__ == "__main__":
    main()
