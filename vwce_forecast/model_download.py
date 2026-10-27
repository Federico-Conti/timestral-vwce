from __future__ import annotations

import hashlib
import logging
import os
import shutil
import time
from pathlib import Path

import requests
from filelock import FileLock
from huggingface_hub import hf_hub_download, hf_hub_url, try_to_load_from_cache

from .model import MODEL_ID, MODEL_REVISION

LOGGER = logging.getLogger(__name__)
WEIGHTS_SHA256 = "1456effb9927e3daa8c7234f10497e3424233869245a9c72b8ecebaf7ad2b7fa"
WEIGHTS_SIZE = 837271648
MODEL_CACHE = Path.home() / ".cache" / "vwce-forecast" / f"Timestral-209M-{MODEL_REVISION}"


def _valid_weights(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size != WEIGHTS_SIZE:
        return False
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest() == WEIGHTS_SHA256


def ensure_model_files() -> Path:
    cached_weights = try_to_load_from_cache(MODEL_ID, "model.safetensors", revision=MODEL_REVISION)
    cached_config = try_to_load_from_cache(MODEL_ID, "config.json", revision=MODEL_REVISION)
    if isinstance(cached_weights, str) and isinstance(cached_config, str):
        if _valid_weights(Path(cached_weights)) and Path(cached_weights).parent == Path(cached_config).parent:
            LOGGER.info("Pesi ufficiali verificati nella cache Hugging Face.")
            return Path(cached_weights).parent

    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    weights = MODEL_CACHE / "model.safetensors"
    partial = MODEL_CACHE / "model.safetensors.part"
    config = MODEL_CACHE / "config.json"
    reject_partial = MODEL_CACHE / ".ignore-hub-partial"
    with FileLock(str(MODEL_CACHE / ".download.lock")):
        if _valid_weights(weights) and config.is_file():
            return MODEL_CACHE
        config_file = hf_hub_download(MODEL_ID, "config.json", revision=MODEL_REVISION)
        if Path(config_file).resolve() != config.resolve():
            shutil.copyfile(config_file, config)
        if not partial.exists() and not reject_partial.exists():
            hub_root = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
            hub_blobs = hub_root / "models--mistralai--Timestral-209M" / "blobs"
            candidates = [path for path in hub_blobs.glob(f"{WEIGHTS_SHA256}*.incomplete")
                          if 0 < path.stat().st_size < WEIGHTS_SIZE]
            if candidates:
                candidate = max(candidates, key=lambda path: path.stat().st_size)
                LOGGER.info("Riutilizzo la cache parziale Hugging Face: %s (%d byte).", candidate, candidate.stat().st_size)
                shutil.copyfile(candidate, partial)

        size = partial.stat().st_size if partial.exists() else 0
        if size > WEIGHTS_SIZE:
            raise ValueError(f"Download parziale più grande del modello: {partial}")
        if size < WEIGHTS_SIZE:
            LOGGER.info("Download riprendibile dei pesi ufficiali: %d/%d byte.", size, WEIGHTS_SIZE)
            url = hf_hub_url(MODEL_ID, "model.safetensors", revision=MODEL_REVISION)
            headers = {"Range": f"bytes={size}-"} if size else {}
            with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                if size and (response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {size}-")):
                    raise RuntimeError("Il server non supporta una ripresa sicura del download; file parziale conservato.")
                with partial.open("ab" if size else "wb") as stream:
                    last_log = time.monotonic()
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            stream.write(chunk)
                            size += len(chunk)
                            if size > WEIGHTS_SIZE:
                                raise ValueError("Il download supera la dimensione ufficiale del modello.")
                            if time.monotonic() - last_log >= 30:
                                LOGGER.info("Pesi scaricati: %.1f%% (%d/%d byte).", 100 * size / WEIGHTS_SIZE, size, WEIGHTS_SIZE)
                                last_log = time.monotonic()
        if not _valid_weights(partial):
            if partial.exists() and partial.stat().st_size == WEIGHTS_SIZE:
                reject_partial.touch()
            raise ValueError(f"Download incompleto o SHA-256 non valido: {partial}. Non verrà caricato.")
        os.replace(partial, weights)
        LOGGER.info("Download completato e SHA-256 verificato: %s", weights)
        return MODEL_CACHE


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(ensure_model_files())
