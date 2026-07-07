"""A real minimal LSP server for resolver-hardening tests (no mocks).

argv[1] mode:
  stall          - initialize OK, then never answer textDocument/definition
  crash          - initialize OK, then exit(1) on first textDocument/definition
  stall_shutdown - initialize OK, never answer shutdown (forces kill path)
"""
import json
import sys


def _read():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        line = line.decode("ascii").strip()
        if not line:
            break
        k, _, v = line.partition(":")
        headers[k.lower()] = v.strip()
    length = int(headers.get("content-length", 0))
    return json.loads(sys.stdin.buffer.read(length).decode("utf-8"))


def _send(msg):
    data = json.dumps(msg).encode("utf-8")
    sys.stdout.buffer.write(b"Content-Length: %d\r\n\r\n" % len(data) + data)
    sys.stdout.buffer.flush()


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "stall"
    while True:
        msg = _read()
        if msg is None:
            return
        method = msg.get("method")
        if method == "initialize":
            _send({"jsonrpc": "2.0", "id": msg["id"], "result": {"capabilities": {}}})
        elif method == "textDocument/definition":
            if mode == "crash":
                sys.exit(1)
            # stall: never reply -> client's timeout fires
        elif method == "shutdown":
            if mode == "stall_shutdown":
                continue   # never reply -> client kill path
            _send({"jsonrpc": "2.0", "id": msg["id"], "result": None})
        elif method == "exit":
            return
        # initialized / didOpen / didClose / others: ignore


if __name__ == "__main__":
    main()
