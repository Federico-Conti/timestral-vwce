import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vwce_forecast import model_download


class Response:
    status_code = 206
    headers = {"Content-Range": "bytes 3-5/6"}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield b"def"


class ModelDownloadTests(unittest.TestCase):
    def test_resume_and_verify_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "model"
            cache.mkdir()
            (cache / "model.safetensors.part").write_bytes(b"abc")
            config = root / "config.json"
            config.write_text("{}")
            digest = hashlib.sha256(b"abcdef").hexdigest()
            with patch.object(model_download, "MODEL_CACHE", cache), \
                 patch.object(model_download, "WEIGHTS_SHA256", digest), \
                 patch.object(model_download, "WEIGHTS_SIZE", 6), \
                 patch.object(model_download, "try_to_load_from_cache", return_value=None), \
                 patch.object(model_download, "hf_hub_download", return_value=str(config)), \
                 patch.object(model_download.requests, "get", return_value=Response()) as request:
                self.assertEqual(model_download.ensure_model_files(), cache)
                self.assertEqual((cache / "model.safetensors").read_bytes(), b"abcdef")
                self.assertEqual(request.call_args.kwargs["headers"], {"Range": "bytes=3-"})
                self.assertFalse((cache / "model.safetensors.part").exists())
                model_download.ensure_model_files()
                self.assertEqual(request.call_count, 1)

    def test_rejects_resume_when_server_ignores_range(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "model"
            cache.mkdir()
            partial = cache / "model.safetensors.part"
            partial.write_bytes(b"abc")
            config = root / "config.json"
            config.write_text("{}")
            response = Response()
            response.status_code = 200
            with patch.object(model_download, "MODEL_CACHE", cache), \
                 patch.object(model_download, "WEIGHTS_SHA256", hashlib.sha256(b"abcdef").hexdigest()), \
                 patch.object(model_download, "WEIGHTS_SIZE", 6), \
                 patch.object(model_download, "try_to_load_from_cache", return_value=None), \
                 patch.object(model_download, "hf_hub_download", return_value=str(config)), \
                 patch.object(model_download.requests, "get", return_value=response):
                with self.assertRaisesRegex(RuntimeError, "ripresa sicura"):
                    model_download.ensure_model_files()
                self.assertEqual(partial.read_bytes(), b"abc")


if __name__ == "__main__":
    unittest.main()
