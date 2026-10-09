import argparse
import shutil
import socket
import sys

from .config import Config
from .pipeline import Engine
from .server import Server


def _reachable_host(host: str) -> str:
    """A host other machines can put in a URL: wildcard binds become this Mac's LAN address."""
    if host in {"0.0.0.0", "::"}:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                probe.connect(("192.0.2.1", 9))  # picks the outbound interface; sends nothing
                return probe.getsockname()[0]
        except OSError:
            return "localhost"
    return f"[{host}]" if ":" in host else host


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
    base = f"http://{_reachable_host(config.host)}:{config.port}"
    print(f"Web UI:               {base}/")
    print(f"Parallel jobs:        {config.max_parallel} (set TRANSCRIBER_MAX_PARALLEL to change)")
    print(f"Whisper-compatible:   {base}/v1/audio/transcriptions", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
