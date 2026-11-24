from __future__ import annotations

import atexit
import json
import logging
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger("vettor.llama_server")

_MANAGED_PROCESS: subprocess.Popen | None = None
_SERVER_LOG_HANDLE = None


def find_llama_executable(configured_bin: str = "") -> str | None:
    """Finds the local llama-server or llama CLI executable."""
    env_bin = configured_bin or os.getenv("QA_LLAMA_BIN", "")
    if env_bin and Path(env_bin).exists():
        return env_bin

    for cmd in ["llama-server", "llama-server.exe", "llama", "llama.exe"]:
        found = shutil.which(cmd)
        if found and Path(found).exists():
            return found

    win_apps = Path.home() / "AppData" / "Local" / "Microsoft" / "WindowsApps" / "llama.exe"
    if win_apps.exists():
        return str(win_apps)

    return None


def find_model_gguf(configured_path: str = "") -> str | None:
    """Checks configured_path, QA_MODEL_PATH, ./models/*.gguf, and ~/.cache/huggingface/hub/**/*.gguf.

    Prefers compact 1.5B–4B models and ignores mmproj files.
    """
    env_path = configured_path or os.getenv("QA_MODEL_PATH", "")
    if env_path and Path(env_path).exists():
        return env_path

    # 1. Local ./models directory
    models_dir = Path.cwd() / "models"
    if models_dir.exists():
        local_ggufs = sorted(models_dir.glob("*.gguf"), key=lambda p: p.stat().st_mtime, reverse=True)
        for gguf in local_ggufs:
            if "mmproj" not in gguf.name.lower():
                return str(gguf)

    # 2. Hugging Face cache (prefer compact 1.5B-4B models)
    hf_hub = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_hub.exists():
        all_ggufs = sorted(hf_hub.glob("**/*.gguf"), key=lambda p: p.stat().st_mtime, reverse=True)
        for tag in ("1.7b", "1.5b", "2b", "3b", "4b", "lfm"):
            for gguf in all_ggufs:
                name_lower = gguf.name.lower()
                if tag in name_lower and "mmproj" not in name_lower:
                    return str(gguf)
        for gguf in all_ggufs:
            if "mmproj" not in gguf.name.lower():
                return str(gguf)

    # 3. Workspace fallback
    for gguf in Path.cwd().glob("**/*.gguf"):
        if "mmproj" not in gguf.name.lower():
            return str(gguf)

    return None


def is_loopback_url(url: str) -> bool:
    """Returns True if the URL points to localhost or loopback address."""
    if not url:
        return False
    host = (urlparse(url if "://" in url else f"http://{url}").hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


def is_server_alive(base_url: str, timeout: float = 1.0, api_key: str = "") -> bool:
    """Checks /health and /v1/models to determine if the server is responsive."""
    clean_url = base_url.rstrip("/").removesuffix("/v1")
    headers = {"User-Agent": "Vettor/2.0"}
    if api_key and api_key != "none":
        headers["Authorization"] = f"Bearer {api_key}"

    for ep in (f"{clean_url}/health", f"{clean_url}/v1/models"):
        try:
            req = urllib.request.Request(ep, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            continue
    return False


def get_server_model_id(base_url: str, timeout: float = 2.0, api_key: str = "") -> str | None:
    """Queries /v1/models and returns the first model ID."""
    clean_url = base_url.rstrip("/").removesuffix("/v1")
    headers = {"User-Agent": "Vettor/2.0"}
    if api_key and api_key != "none":
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        req = urllib.request.Request(f"{clean_url}/v1/models", headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        data = payload.get("data") or []
        if data and isinstance(data[0], dict):
            return data[0].get("id") or data[0].get("model")
    except Exception as exc:
        logger.debug("Could not read /v1/models from %s: %s", base_url, exc)
    return None


def stop_managed_server() -> None:
    """Terminates any child llama-server process started by this module."""
    global _MANAGED_PROCESS, _SERVER_LOG_HANDLE
    if _MANAGED_PROCESS is not None:
        try:
            _MANAGED_PROCESS.terminate()
            _MANAGED_PROCESS.wait(timeout=3)
        except Exception:
            try:
                _MANAGED_PROCESS.kill()
            except Exception:
                pass
        finally:
            _MANAGED_PROCESS = None
        if _SERVER_LOG_HANDLE is not None:
            try:
                _SERVER_LOG_HANDLE.close()
            except Exception:
                pass
            _SERVER_LOG_HANDLE = None


atexit.register(stop_managed_server)


def ensure_llama_server(
    base_url: str = "http://127.0.0.1:8080/v1",
    configured_bin: str = "",
    configured_model: str = "",
    context_size: int = 4096,
    use_jinja: bool = True,
    chat_template: str = "",
) -> bool:
    """Ensures a local llama-server instance is running, starting one if needed."""
    global _MANAGED_PROCESS, _SERVER_LOG_HANDLE
    if is_server_alive(base_url):
        return True

    llama_bin = find_llama_executable(configured_bin)
    model_path = find_model_gguf(configured_model)
    if not llama_bin or not model_path:
        logger.warning("Local llama.cpp binary or GGUF model not found.")
        return False

    parsed_url = urlparse(base_url if "://" in base_url else f"http://{base_url}")
    effective_host = parsed_url.hostname or "127.0.0.1"
    effective_port = parsed_url.port or 8080

    bin_name = Path(llama_bin).name.lower()
    cmd = [llama_bin]
    if "server" not in bin_name:
        cmd.append("serve")
    cmd.extend([
        "-m", model_path,
        "--host", effective_host,
        "--port", str(effective_port),
        "-c", str(context_size),
    ])
    if chat_template:
        cmd.extend(["--chat-template", chat_template])
    elif use_jinja:
        cmd.append("--jinja")

    log_path = Path.cwd() / "artifacts" / "llama_server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _SERVER_LOG_HANDLE = log_path.open("ab")
        _MANAGED_PROCESS = subprocess.Popen(cmd, stdout=_SERVER_LOG_HANDLE, stderr=subprocess.STDOUT)
    except Exception as exc:
        logger.error("Failed to launch llama server: %s", exc)
        return False

    for _ in range(30):
        if is_server_alive(base_url, timeout=0.8):
            return True
        if _MANAGED_PROCESS.poll() is not None:
            stop_managed_server()
            return False
        time.sleep(0.5)

    stop_managed_server()
    return False
