import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Optional

import requests
import runpod

MODEL_PATH = os.environ["MODEL_PATH"].strip()
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "supergemma4-26b")
LLAMA_SERVER_HOST = os.environ.get("LLAMA_SERVER_HOST", "127.0.0.1")
LLAMA_SERVER_PORT = int(os.environ.get("LLAMA_SERVER_PORT", "8000"))
LLAMA_SERVER_ARGS = os.environ.get(
    "LLAMA_SERVER_ARGS", "--ctx-size 4096 -ngl 999 --embeddings"
)
STARTUP_TIMEOUT_SEC = int(os.environ.get("STARTUP_TIMEOUT_SEC", "1800"))
REQUEST_TIMEOUT_SEC = int(os.environ.get("REQUEST_TIMEOUT_SEC", "1800"))

_http = requests.Session()
_llama_process = None


def _check_llama_process(process: Optional[subprocess.Popen]) -> None:
    if process is None:
        return

    return_code = process.poll()
    if return_code is None:
        return

    raise RuntimeError(
        f"llama-server exited before becoming ready with code {return_code}"
    )


def resolve_model_path() -> Path:
    if not MODEL_PATH:
        raise ValueError("MODEL_PATH must not be empty")

    model_path = Path(MODEL_PATH)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"MODEL_PATH does not exist or is not a file: {model_path}"
        )

    return model_path


def wait_for_llama_server(process: Optional[subprocess.Popen] = None) -> None:
    url = f"http://{LLAMA_SERVER_HOST}:{LLAMA_SERVER_PORT}/health"
    deadline = time.time() + STARTUP_TIMEOUT_SEC
    last_error = "unknown"

    while time.time() < deadline:
        _check_llama_process(process)

        try:
            response = _http.get(url, timeout=5)
            if response.status_code == 200:
                return
            last_error = f"status={response.status_code}, body={response.text}"
        except Exception as exc:
            last_error = repr(exc)

        time.sleep(2)

    _check_llama_process(process)
    raise TimeoutError(f"llama-server did not become ready in time: {last_error}")


def start_llama_server() -> subprocess.Popen:
    model_path = resolve_model_path()

    print(f"[startup] model_path={model_path}")

    command = [
        "/opt/llama.cpp/build/bin/llama-server",
        "-m",
        str(model_path),
        "--host",
        LLAMA_SERVER_HOST,
        "--port",
        str(LLAMA_SERVER_PORT),
        "--alias",
        MODEL_ALIAS,
    ] + shlex.split(LLAMA_SERVER_ARGS)

    print(f"[startup] launching: {' '.join(command)}")
    process = subprocess.Popen(command)

    wait_for_llama_server(process)
    print("[startup] llama-server is ready")
    return process


def proxy_to_llama(path: str, payload: dict):
    url = f"http://{LLAMA_SERVER_HOST}:{LLAMA_SERVER_PORT}{path}"

    body = dict(payload) if isinstance(payload, dict) else payload
    if isinstance(body, dict) and path.startswith("/v1/") and "model" not in body:
        body["model"] = MODEL_ALIAS

    response = _http.post(url, json=body, timeout=REQUEST_TIMEOUT_SEC)

    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        data = response.json()
    else:
        data = {"text": response.text}

    if not response.ok:
        return {
            "error": True,
            "status_code": response.status_code,
            "response": data,
        }

    return data


def handler(job):
    job_input = job.get("input", {})

    if not isinstance(job_input, dict):
        return {"error": True, "message": "input must be a JSON object"}

    if "payload" in job_input:
        path = job_input.get("path", "/v1/chat/completions")
        payload = job_input["payload"]
    else:
        path = job_input.get("path", "/v1/chat/completions")
        payload = {key: value for key, value in job_input.items() if key != "path"}

    try:
        return proxy_to_llama(path, payload)
    except Exception as exc:
        return {"error": True, "message": repr(exc)}


if __name__ == "__main__":
    _llama_process = start_llama_server()
    runpod.serverless.start({"handler": handler})
