"""Run with an MLX Python environment and MLX_WHISPER_SNAPSHOT set to a local snapshot."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


class WorkerRecovery(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("MLX_WHISPER_SNAPSHOT"), "requires local MLX weights")
    def test_recoverable_decode_error_keeps_the_worker_alive(self):
        script = Path(__file__).resolve().parents[1] / "src-tauri/src/managers/mlx_worker.py"
        samples = 16000

        def frame(language):
            header = {"samples": samples, "language": language, "custom_words": []}
            return json.dumps(header).encode() + b"\n" + bytes(samples * 4)

        worker = subprocess.run(
            [sys.executable, "-I", script, os.environ["MLX_WHISPER_SNAPSHOT"]],
            env={**os.environ, "HF_HUB_OFFLINE": "1"},
            input=frame("invalid-language") + frame("en"),
            capture_output=True, timeout=90,
        )
        replies = [json.loads(line) for line in worker.stdout.splitlines()]
        self.assertEqual(worker.returncode, 0, (replies, worker.stderr))
        ready, failed, recovered = replies
        self.assertEqual(ready, {"text": ""})
        self.assertIn("error", failed)
        self.assertIn("text", recovered)


if __name__ == "__main__":
    unittest.main()
