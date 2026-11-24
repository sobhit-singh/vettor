import atexit
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger("vettor.llm_manager")

_MANAGED_PROCESS: Optional[subprocess.Popen] = None
_SERVER_LOG_HANDLE = None


def find_llama_executable(configured_bin: str = "") -> Optional[str]:
    """Finds llama.exe or llama-server executable."""
    env_bin = configured_bin or os.getenv("QA_LLAMA_BIN", "")
    if env_bin and Path(env_bin).exists():
        return env_bin

    for cmd in ["llama", "llama.exe", "llama-server", "llama-server.exe"]:
        found = shutil.which(cmd)
        if found and Path(found).exists():
            return found

    # Check WindowsApps store install
    win_apps = Path.home() / "AppData" / "Local" / "Microsoft" / "WindowsApps" / "llama.exe"
    if win_apps.exists():
        return str(win_apps)

    return None


def find_model_gguf(configured_path: str = "") -> Optional[str]:
    """Finds a GGUF model, prioritizing LFM3-1.7B in the Hugging Face cache."""
    env_path = configured_path or os.getenv("QA_MODEL_PATH", "")
    if env_path and Path(env_path).exists():
        return env_path

    hf_hub = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_hub.exists():
        # Check specific model folder first
        target_dir = hf_hub / "models--mradermacher--LFM3-1.7B-Uncensored-GGUF"
        if target_dir.exists():
            for gguf in target_dir.glob("**/*.gguf"):
                return str(gguf)

        # Search for any LFM3 model
        for gguf in hf_hub.glob("**/*lfm3*/**/*.gguf"):
            return str(gguf)

        # Search for any 1.7B or 2B GGUF
        for gguf in hf_hub.glob("**/*1.7b*.gguf"):
            return str(gguf)

        for gguf in hf_hub.glob("**/*.gguf"):
            return str(gguf)

    # Check local workspace
    for gguf in Path.cwd().glob("**/*.gguf"):
        return str(gguf)

    return None


def get_server_model_id(base_url: str, timeout: float = 2.0) -> Optional[str]:
    """Returns the model id advertised by an OpenAI-compatible local server."""
    clean_url = base_url.rstrip("/").removesuffix("/v1")
    try:
        with urllib.request.urlopen(f"{clean_url}/v1/models", timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        data = payload.get("data") or []
        if data and isinstance(data[0], dict):
            return data[0].get("id") or data[0].get("model")
    except Exception as exc:
        logger.debug("Could not read /v1/models from %s: %s", base_url, exc)
    return None


def is_server_alive(base_url: str, timeout: float = 1.0) -> bool:
    """Checks if llama.cpp server is already running and healthy."""
    clean_url = base_url.rstrip("/").removesuffix("/v1")
    endpoints = [f"{clean_url}/health", f"{clean_url}/v1/models"]
    for ep in endpoints:
        try:
            req = urllib.request.Request(ep, headers={"User-Agent": "Vettor/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            continue
    return False


def stop_managed_server():
    """Terminates the background llama server if launched by this process."""
    global _MANAGED_PROCESS, _SERVER_LOG_HANDLE
    if _MANAGED_PROCESS is not None:
        try:
            logger.info("Stopping managed llama.cpp server (pid=%s)...", _MANAGED_PROCESS.pid)
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
    host: str = "127.0.0.1",
    port: int = 8080,
    context_size: int = 4096,
    use_jinja: bool = True,
    chat_template: str = "",
) -> bool:
    """
    Ensures that a llama.cpp server is running.
    If already running at base_url, returns True immediately.
    Otherwise, discovers the llama binary and model GGUF, starts the server,
    and waits until /health responds.
    """
    global _MANAGED_PROCESS, _SERVER_LOG_HANDLE
    parsed_url = urlparse(base_url if "://" in base_url else f"http://{base_url}")
    effective_host = parsed_url.hostname or host
    effective_port = parsed_url.port or port

    if is_server_alive(base_url):
        logger.info("llama.cpp server is already running at %s", base_url)
        return True

    llama_bin = find_llama_executable(configured_bin)
    if not llama_bin:
        logger.warning("No llama.exe or llama-server executable found.")
        return False

    model_path = find_model_gguf(configured_model)
    if not model_path:
        logger.warning("No GGUF model found in Hugging Face cache or configured path.")
        return False

    logger.info("Auto-starting local llama.cpp server with model: %s", model_path)
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
    logger.info("llama.cpp server command: %s | output: %s", " ".join(cmd), log_path)
    try:
        _SERVER_LOG_HANDLE = log_path.open("ab")
        _MANAGED_PROCESS = subprocess.Popen(
            cmd,
            stdout=_SERVER_LOG_HANDLE,
            stderr=subprocess.STDOUT,
        )
    except Exception as exc:
        logger.error("Failed to launch llama server process: %s", exc)
        return False

    # Wait for server ready
    logger.info("Waiting for llama.cpp server to become ready on port %d...", effective_port)
    for _ in range(30):
        if is_server_alive(base_url, timeout=0.8):
            logger.info("llama.cpp server is ready and serving requests.")
            return True
        if _MANAGED_PROCESS.poll() is not None:
            logger.error("llama server process exited prematurely with code %s; see %s", _MANAGED_PROCESS.returncode, log_path)
            _MANAGED_PROCESS = None
            if _SERVER_LOG_HANDLE is not None:
                _SERVER_LOG_HANDLE.close()
                _SERVER_LOG_HANDLE = None
            return False
        time.sleep(0.5)

    logger.warning("Timed out waiting for llama server to report healthy.")
    stop_managed_server()
    return False
