import argparse
import shutil
import sys

from .config import Config
from .pipeline import Engine
from .server import Server


def main():
    config = Config()
    parser = argparse.ArgumentParser(prog="transcriber", description=__doc__)
    parser.add_argument("--host", default=config.host, help="address to bind (default: %(default)s)")
    parser.add_argument("--port", type=int, default=config.port, help="port to bind (default: %(default)s)")
    args = parser.parse_args()
    config.host, config.port = args.host, args.port

    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg is required but was not found on PATH. Install it with: brew install ffmpeg")

    engine = Engine(config)
    server = Server(config, engine)
    engine.start()
    print(f"Web UI:               http://{config.host}:{config.port}/")
    print(f"Parallel jobs:        {config.max_parallel} (set TRANSCRIBER_MAX_PARALLEL to change)")
    print(f"Whisper-compatible:   http://{config.host}:{config.port}/v1/audio/transcriptions", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
