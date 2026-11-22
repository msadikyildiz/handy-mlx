"""MLX Whisper worker. stdin carries JSON lines followed by mono 16 kHz float32 PCM."""
import json
import os
from pathlib import Path
import sys

# Keep native libraries' stdout separate from the reply pipe.
protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8")
os.dup2(sys.stderr.fileno(), sys.stdout.fileno())


def reply(**message):
    print(json.dumps(message, ensure_ascii=False), file=protocol, flush=True)


def main():
    import mlx.core as mx
    import mlx_whisper
    import numpy as np
    from mlx_whisper.transcribe import ModelHolder

    snapshot = Path(sys.argv[1])
    if not snapshot.is_dir():
        raise FileNotFoundError(snapshot)
    ModelHolder.get_model(str(snapshot), mx.float16)
    reply(text="")
    source = sys.stdin.buffer
    for header in source:
        request = json.loads(header)
        size = request["samples"] * 4
        pcm = source.read(size)
        if len(pcm) != size:
            return
        language = request["language"]
        try:
            result = mlx_whisper.transcribe(
                np.frombuffer(pcm, dtype="<f4"),
                path_or_hf_repo=str(snapshot),
                language=None if language == "auto" else language,
                initial_prompt=", ".join(request["custom_words"]) or None,
                verbose=None,
            )
        except Exception as error:
            # The complete PCM frame was consumed. The next request can still run.
            reply(error=str(error))
            continue
        reply(text=result["text"].strip(), language=result.get("language"))


try:
    main()
except Exception as error:
    reply(error=str(error))
    sys.exit(1)
