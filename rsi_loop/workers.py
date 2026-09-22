"""Isolate simulator, JAX VLA and Torch vision in the same conda environment."""
from multiprocessing.connection import Client
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import time


class Worker:
    def __init__(self, module, args, *, output, env=None, timeout=600):
        self.directory = tempfile.TemporaryDirectory(prefix="rsi-ipc-")
        self.socket = Path(self.directory.name) / "worker.sock"
        self.key = secrets.token_bytes(32)
        variables = dict(os.environ, RSI_IPC_AUTH=self.key.hex())
        variables.update(env or {})
        self.log = Path(output).open("w")
        self.process = subprocess.Popen([sys.executable, "-u", "-m", module, "--socket", str(self.socket), *args],
                                        stdout=self.log, stderr=subprocess.STDOUT, env=variables)
        self.connection = None
        deadline = time.monotonic()+timeout
        try:
            while not self.socket.exists():
                if self.process.poll() is not None:
                    raise RuntimeError(f"{module} failed; inspect {output}")
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"{module} did not initialize; inspect {output}")
                time.sleep(.2)
            self.connection = Client(str(self.socket), family="AF_UNIX", authkey=self.key)
            self.call("metadata", timeout=timeout)
        except BaseException:
            self.close()
            raise

    def call(self, op, timeout=180, **args):
        self.connection.send(dict(op=op, args=args))
        if not self.connection.poll(timeout):
            # No retries after uncertain execution. The owner must stop.
            raise TimeoutError(f"Worker {op} acknowledgement timed out")
        response = self.connection.recv()
        if not response["ok"]:
            raise RuntimeError(response["error"])
        return response["result"]

    def close(self):
        if self.connection is not None:
            try:
                self.call("close", timeout=10)
            except (OSError, EOFError, TimeoutError, RuntimeError):
                pass
            self.connection.close()
            self.connection = None
        if self.process.poll() is None:
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        self.log.close()
        self.directory.cleanup()


def serve(address, dispatch):
    from multiprocessing.connection import Listener
    import traceback
    key = bytes.fromhex(os.environ.pop("RSI_IPC_AUTH"))
    with Listener(address, family="AF_UNIX", authkey=key) as listener:
        with listener.accept() as connection:
            while True:
                try:
                    request = connection.recv()
                except EOFError:
                    break
                try:
                    result = dispatch(request["op"], request["args"])
                    connection.send(dict(ok=True, result=result))
                except Exception as error:
                    traceback.print_exc()
                    connection.send(dict(ok=False, error=f"{type(error).__name__}: {error}"))
                    break
                if request["op"] == "close":
                    break
