from __future__ import annotations

import ast
import csv
import gc
import inspect
import json
import math
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
try:
    import accelerate
except Exception:
    accelerate = None

try:
    import peft
except Exception:
    peft = None
import transformers
try:
    from accelerate import Accelerator
except Exception:
    Accelerator = None
from datasets import load_dataset
try:
    from peft import LoraConfig, PeftModel, get_peft_model
except Exception:
    LoraConfig = None
    PeftModel = None
    get_peft_model = None
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoImageProcessor,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_scheduler,
)

try:
    from transformers import ColModernVBertForRetrieval, ColModernVBertProcessor
    TRANSFORMERS_COLMODERNVBERT_IMPORT_ERROR = None
except Exception as exc:
    TRANSFORMERS_COLMODERNVBERT_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
    ColModernVBertForRetrieval = None
    ColModernVBertProcessor = None

try:
    from colpali_engine.models import ColModernVBert as ColPaliModernVBert
    from colpali_engine.models import ColModernVBertProcessor as ColPaliModernVBertProcessor
    COLPALI_MODERNVBERT_IMPORT_ERROR = None
except Exception as exc:
    COLPALI_MODERNVBERT_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
    ColPaliModernVBert = None
    ColPaliModernVBertProcessor = None

PaddleOCR = None
PADDLEOCR_IMPORT_ERROR = "disabled in retrieval module; run OCR in 01_prepare_synthetic_dataset_paddleocr.ipynb"

easyocr = None
EASYOCR_IMPORT_ERROR = "disabled in retrieval module"


def in_kaggle() -> bool:
    return Path("/kaggle/working").exists()


def in_colab() -> bool:
    return Path("/content").exists()


def default_workdir(project_name: str = "modernvbert_ru") -> Path:
    if in_kaggle():
        root = Path("/kaggle/working")
    elif in_colab():
        root = Path("/content")
    else:
        root = Path.cwd()
    out = root / project_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def pick_device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


DEVICE, DEFAULT_DTYPE = pick_device_and_dtype()
PINNED_TRANSFORMERS_VERSION = "5.5.4"
PINNED_COLPALI_ENGINE_VERSION = "0.3.15"
PINNED_PEFT_VERSION = "0.18.0"
PINNED_ACCELERATE_VERSION = "0.34.2"
PINNED_PADDLEOCR_VERSION = "2.7.3"
PINNED_PADDLEPADDLE_VERSION = "2.6.2"
PINNED_NUMPY_VERSION = "1.26.4"
PINNED_OPENCV_VERSION = "4.10.0.84"
PINNED_SCIPY_VERSION = "1.14.1"
PINNED_SCIKIT_IMAGE_VERSION = "0.25.2"


@dataclass
class SourceRecord:
    row_id: int
    doc_id: str
    query: str
    source: str
    answer: str | None = None
    options: str | None = None
    original_image_path: str | None = None
    synthetic_image_path: str | None = None
    query_ru: str | None = None
    answer_ru: str | None = None
    split: str | None = None
    image_variant: str = "original_en"
    source_lang_query: str = "en"
    target_lang_query: str = "en"
    translation_model: str | None = None
    seed: int | None = None
    ocr_boxes_detected: int = 0
    ocr_boxes_translated: int = 0
    synthetic_fallback_used: bool = False
    synthetic_mode: str | None = None
    ocr_backend_requested: str | None = None
    ocr_backend_used: str | None = None
    ocr_backend_detail: str | None = None
    ocr_fallback_reason: str | None = None


@dataclass
class ManualRussianRecord:
    query_id: str
    doc_id: str
    query: str
    image_path: str
    split: str = "manual_test"
    image_variant: str = "real_ru"
    source_lang_query: str = "ru"
    target_lang_query: str = "ru"
    category: str | None = None
    difficulty: str | None = None
    notes: str | None = None


@dataclass
class RetrievalRuntime:
    requested_model_name: str
    resolved_model_name: str
    device: str
    processor: Any
    model: Any
    versions: dict[str, str]
    backend: str = "transformers"


@dataclass
class RetrieverSourceSpec:
    requested_model_name: str
    processor_name: str
    resolved_model_name: str
    built_in_adapter_dir: str | None = None
    backend: str = "transformers"


def slugify(text: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text))
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "item"


def export_json(path: str | Path, obj: Any):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def export_jsonl(rows: Iterable[dict[str, Any]], output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _row_to_record(row: dict[str, Any], row_id: int) -> SourceRecord:
    query = str(row.get("query", "")).strip()
    source = str(row.get("source", "vidore/colpali_train_set"))
    answer = row.get("answer")
    options = row.get("options")
    raw_doc_id = row.get("id") or row.get("doc_id") or row.get("document_id") or f"doc_{row_id}"
    doc_id = slugify(str(raw_doc_id))
    return SourceRecord(
        row_id=row_id,
        doc_id=doc_id,
        query=query,
        source=source,
        answer=str(answer) if answer is not None else None,
        options=str(options) if options is not None else None,
    )


def load_colpali_records(
    max_rows: int | None = None,
    seed: int = 42,
    streaming: bool = True,
    dataset_name: str = "vidore/colpali_train_set",
    split: str = "train",
):
    if streaming:
        print(f"Loading {dataset_name} in streaming mode...")
        ds = load_dataset(dataset_name, split=split, streaming=True)
        if max_rows is not None:
            buffer_size = min(2000, max(100, max_rows * 4))
            ds = ds.shuffle(seed=seed, buffer_size=buffer_size)
            print(f"Streaming with shuffle buffer_size={buffer_size}, max_rows={max_rows}")
        rows = []
        for i, row in enumerate(ds):
            if max_rows is not None and i >= max_rows:
                break
            rows.append(row)
        records = [_row_to_record(row, i) for i, row in enumerate(rows)]
        return records, rows

    print(f"Loading {dataset_name} in standard mode...")
    ds = load_dataset(dataset_name, split=split)
    if max_rows is not None and max_rows < len(ds):
        ds = ds.shuffle(seed=seed).select(range(max_rows))
    rows = list(ds)
    records = [_row_to_record(row, i) for i, row in enumerate(rows)]
    return records, rows


def save_original_images(rows: list[dict[str, Any]], records: list[SourceRecord], out_dir: str | Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, (record, row) in enumerate(zip(records, rows)):
        image = row["image"]
        output_path = out_dir / f"{record.doc_id}.png"
        if isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            pil_img = image["image"].convert("RGB")
        pil_img.save(output_path)
        record.original_image_path = str(output_path)
        if (idx + 1) % 100 == 0:
            print("saved", idx + 1, "images")


def create_split_map(
    records: list[SourceRecord],
    output_path: str | Path,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
):
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    unique_doc_ids = sorted({record.doc_id for record in records})
    rng = random.Random(seed)
    rng.shuffle(unique_doc_ids)

    n_docs = len(unique_doc_ids)
    n_train = max(1, int(n_docs * train_ratio))
    n_val = max(1, int(n_docs * val_ratio))
    if n_train + n_val >= n_docs:
        n_val = max(1, min(n_val, n_docs - n_train - 1))
    n_test = max(1, n_docs - n_train - n_val)

    train_docs = unique_doc_ids[:n_train]
    val_docs = unique_doc_ids[n_train : n_train + n_val]
    test_docs = unique_doc_ids[n_train + n_val : n_train + n_val + n_test]

    split_map = {doc_id: "train" for doc_id in train_docs}
    split_map.update({doc_id: "val" for doc_id in val_docs})
    split_map.update({doc_id: "test" for doc_id in test_docs})

    artifact = {
        "seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "splits": {
            "train": train_docs,
            "val": val_docs,
            "test": test_docs,
        },
        "counts": {
            "train_docs": len(train_docs),
            "val_docs": len(val_docs),
            "test_docs": len(test_docs),
            "total_docs": n_docs,
        },
    }
    export_json(output_path, artifact)
    return split_map, artifact


def apply_split_map(records: list[SourceRecord], split_map: dict[str, str], seed: int):
    for record in records:
        record.split = split_map[record.doc_id]
        record.seed = seed


class MarianTranslator:
    def __init__(
        self,
        model_name: str = "Helsinki-NLP/opus-mt-en-ru",
        device: str = "cpu",
        max_new_tokens: int = 64,
        max_input_length: int = 256,
    ):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.max_input_length = max_input_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device).eval()

    def translate(self, text: str | None) -> str | None:
        if text is None:
            return None
        text = str(text).strip()
        if not text:
            return text
        batch = self.tokenizer(
            [text],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_input_length,
        ).to(self.device)
        with torch.no_grad():
            generated = self.model.generate(**batch, max_new_tokens=self.max_new_tokens)
        out = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return out


class OCRBackendAdapter:
    def __init__(
        self,
        backend: str = "paddleocr",
        langs: list[str] | None = None,
        gpu: bool | None = None,
        use_angle_cls: bool = True,
    ):
        self.backend = backend.lower()
        self.langs = langs or ["en"]
        self.gpu = torch.cuda.is_available() if gpu is None else gpu
        self.use_angle_cls = use_angle_cls
        self._easyocr_fallback_reader = None
        self.last_backend_trace = {
            "requested_backend": self.backend,
            "used_backend": self.backend,
            "detail": "not_run_yet",
            "fallback_reason": None,
        }
        self.reader = self._build_reader()

    def _build_reader(self):
        if self.backend == "paddleocr":
            if PaddleOCR is None:
                raise ImportError(
                    "PaddleOCR is not installed. Install `paddleocr` and `paddlepaddle`, "
                    "or switch to backend='easyocr'."
                )
            paddle_lang = self._map_langs_for_paddle(self.langs)
            candidate_kwargs = [
                {
                    "lang": paddle_lang,
                    "use_angle_cls": self.use_angle_cls,
                    "device": "gpu" if self.gpu else "cpu",
                },
                {
                    "lang": paddle_lang,
                    "use_angle_cls": self.use_angle_cls,
                    "device": "gpu" if self.gpu else "cpu",
                    "show_log": False,
                },
                {
                    "lang": paddle_lang,
                    "use_angle_cls": self.use_angle_cls,
                    "use_gpu": self.gpu,
                },
                {
                    "lang": paddle_lang,
                    "use_angle_cls": self.use_angle_cls,
                    "use_gpu": self.gpu,
                    "show_log": False,
                },
                {
                    "lang": paddle_lang,
                    "use_angle_cls": self.use_angle_cls,
                },
                {
                    "lang": paddle_lang,
                },
            ]
            last_error = None
            for kwargs in candidate_kwargs:
                try:
                    return PaddleOCR(**kwargs)
                except (TypeError, ValueError) as exc:
                    last_error = exc
                    continue
            raise RuntimeError(
                "Failed to initialize PaddleOCR with all compatibility fallbacks. "
                f"Last error: {last_error}"
            )

        if self.backend == "easyocr":
            if easyocr is None:
                raise ImportError("EasyOCR is not installed. Install `easyocr` or switch OCR backend.")
            return easyocr.Reader(self.langs, gpu=self.gpu)

        raise ValueError(f"Unsupported OCR backend: {self.backend}")

    @staticmethod
    def _map_langs_for_paddle(langs: list[str]) -> str:
        normalized = [lang.lower() for lang in langs]
        if normalized == ["en"]:
            return "en"
        if "ru" in normalized:
            return "ru"
        return "en"

    def readtext(self, image_array: np.ndarray):
        if self.backend == "easyocr":
            self.last_backend_trace = {
                "requested_backend": self.backend,
                "used_backend": "easyocr",
                "detail": "easyocr.readtext(detail=1)",
                "fallback_reason": None,
            }
            return self.reader.readtext(image_array, detail=1)

        try:
            raw = self._run_paddle_ocr(image_array)
        except RuntimeError as exc:
            if easyocr is None:
                raise
            print(f"PaddleOCR failed, falling back to EasyOCR. Reason: {exc}")
            self.last_backend_trace = {
                "requested_backend": self.backend,
                "used_backend": "easyocr",
                "detail": "easyocr_fallback_after_paddle_failure",
                "fallback_reason": str(exc),
            }
            raw = self._get_easyocr_fallback_reader().readtext(image_array, detail=1)
            return raw
        return self._normalize_paddle_result(raw)

    def _get_easyocr_fallback_reader(self):
        if self._easyocr_fallback_reader is None:
            self._easyocr_fallback_reader = easyocr.Reader(self.langs, gpu=self.gpu)
        return self._easyocr_fallback_reader

    def _run_paddle_ocr(self, image_array: np.ndarray):
        candidate_calls = [
            (
                "predict(safe_doc_disabled)",
                lambda: self.reader.predict(
                    image_array,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                ),
            ),
            ("ocr(cls=use_angle_cls)", lambda: self.reader.ocr(image_array, cls=self.use_angle_cls)),
            ("ocr()", lambda: self.reader.ocr(image_array)),
            ("predict(cls=use_angle_cls)", lambda: self.reader.predict(image_array, cls=self.use_angle_cls)),
            ("predict()", lambda: self.reader.predict(image_array)),
        ]
        last_error = None
        last_call_name = None
        for call_name, call in candidate_calls:
            try:
                raw = call()
                self.last_backend_trace = {
                    "requested_backend": self.backend,
                    "used_backend": "paddleocr",
                    "detail": call_name,
                    "fallback_reason": None,
                }
                return raw
            except (AttributeError, TypeError, ValueError, IndexError, NotImplementedError, RuntimeError) as exc:
                last_call_name = call_name
                last_error = exc
                continue
        raise RuntimeError(
            "Failed to run PaddleOCR with all compatibility fallbacks. "
            f"Last call: {last_call_name}. Last error: {type(last_error).__name__}: {last_error}"
        )

    @staticmethod
    def _normalize_bbox(bbox):
        if bbox is None:
            return None
        if hasattr(bbox, "tolist"):
            bbox = bbox.tolist()
        if not isinstance(bbox, (list, tuple)) or len(bbox) == 0:
            return None
        if isinstance(bbox[0], (int, float)) and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        if isinstance(bbox[0], (list, tuple)) and len(bbox[0]) >= 2:
            return [[float(pt[0]), float(pt[1])] for pt in bbox[:4]]
        return None

    @staticmethod
    def _coerce_float(value, default=0.0):
        try:
            return float(value)
        except Exception:
            return float(default)

    def _normalize_paddle_result(self, raw):
        if raw is None:
            return []

        # Older API often returns [[bbox, (text, conf)], ...] wrapped in a list.
        if isinstance(raw, list) and raw and isinstance(raw[0], list):
            first = raw[0]
            if first and isinstance(first[0], (list, tuple)) and len(first[0]) >= 2:
                maybe_old_api = []
                for item in first:
                    if not item or len(item) < 2:
                        continue
                    bbox = self._normalize_bbox(item[0])
                    rec = item[1]
                    if bbox is None or not isinstance(rec, (list, tuple)) or len(rec) < 2:
                        continue
                    maybe_old_api.append((bbox, str(rec[0]), self._coerce_float(rec[1], default=0.0)))
                if maybe_old_api:
                    return maybe_old_api

        # Newer API may return a list of result objects or dict-like structures.
        candidates = raw if isinstance(raw, list) else [raw]
        normalized = []
        for item in candidates:
            if item is None:
                continue

            if isinstance(item, dict):
                texts = item.get("rec_texts") or item.get("texts") or []
                scores = item.get("rec_scores") or item.get("scores") or []
                polys = item.get("rec_polys") or item.get("dt_polys") or item.get("polys") or []
                count = max(len(texts), len(scores), len(polys))
                for idx in range(count):
                    text = str(texts[idx]) if idx < len(texts) else ""
                    conf = self._coerce_float(scores[idx], default=0.0) if idx < len(scores) else 0.0
                    bbox = self._normalize_bbox(polys[idx]) if idx < len(polys) else None
                    if bbox is not None and text:
                        normalized.append((bbox, text, conf))
                continue

            # Paddle result object with attributes
            texts = getattr(item, "rec_texts", None) or getattr(item, "texts", None)
            scores = getattr(item, "rec_scores", None) or getattr(item, "scores", None)
            polys = getattr(item, "rec_polys", None) or getattr(item, "dt_polys", None) or getattr(item, "polys", None)
            if texts is not None or polys is not None:
                texts = list(texts) if texts is not None else []
                scores = list(scores) if scores is not None else []
                polys = list(polys) if polys is not None else []
                count = max(len(texts), len(scores), len(polys))
                for idx in range(count):
                    text = str(texts[idx]) if idx < len(texts) else ""
                    conf = self._coerce_float(scores[idx], default=0.0) if idx < len(scores) else 0.0
                    bbox = self._normalize_bbox(polys[idx]) if idx < len(polys) else None
                    if bbox is not None and text:
                        normalized.append((bbox, text, conf))
                continue

            # Fallback for tuple/list items that already look normalized.
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                bbox = self._normalize_bbox(item[0])
                rec = item[1]
                if bbox is not None and isinstance(rec, (list, tuple)) and len(rec) >= 2:
                    normalized.append((bbox, str(rec[0]), self._coerce_float(rec[1], default=0.0)))

        return normalized


def translate_records(records: list[SourceRecord], translator: MarianTranslator):
    for idx, record in enumerate(records):
        record.query_ru = translator.translate(record.query)
        record.source_lang_query = "en"
        record.target_lang_query = "ru"
        record.translation_model = translator.model_name
        if (idx + 1) % 20 == 0:
            print("translated", idx + 1, "queries")


class NoOCRBackendAdapter:
    def __init__(self, backend: str = "none", reason: str | None = None):
        self.backend = "none"
        self.reason = reason or "OCR disabled."
        self.last_backend_trace = {
            "requested_backend": backend,
            "used_backend": "none",
            "detail": "no_ocr_empty_results",
            "fallback_reason": self.reason,
        }

    def readtext(self, image_array: np.ndarray):
        return []


def get_ocr_reader(
    langs: list[str] | None = None,
    gpu: bool | None = None,
    backend: str = "paddleocr",
    use_angle_cls: bool = True,
    strict: bool = True,
):
    requested_backend = (backend or "paddleocr").lower()
    if requested_backend in {"none", "disabled", "skip", "no_ocr"}:
        if strict:
            raise RuntimeError(
                "OCR is disabled but strict_ocr=True. Synthetic Russian image generation "
                "requires a working OCR backend for thesis runs."
            )
        print("OCR is disabled; synthetic pages will use fallback originals.")
        return NoOCRBackendAdapter(backend=requested_backend, reason="OCR disabled by configuration.")

    if requested_backend == "paddleocr" and PaddleOCR is None:
        if easyocr is not None:
            print("PaddleOCR is not installed; falling back to EasyOCR.")
            requested_backend = "easyocr"
        else:
            if strict:
                raise ImportError(
                    "PaddleOCR is not installed, and strict_ocr=True. Synthetic Russian image "
                    "generation requires PaddleOCR for thesis runs.\n"
                    f"PaddleOCR import error: {PADDLEOCR_IMPORT_ERROR}\n"
                    f"Recommended Kaggle reinstall cell:\n{build_kaggle_paddleocr_reinstall_command()}\n"
                    "Restart the runtime after reinstalling, then rerun the import/preflight cells."
                )
            print("PaddleOCR is not installed and EasyOCR is unavailable; continuing with no OCR fallback.")
            return NoOCRBackendAdapter(
                backend=backend,
                reason="PaddleOCR and EasyOCR are not installed. Synthetic pages will be copied from originals.",
            )

    if requested_backend == "easyocr" and easyocr is None:
        if strict:
            raise ImportError(
                "EasyOCR is not installed and strict_ocr=True. Use ocr_backend='paddleocr' with "
                "a working PaddleOCR install, or set strict_ocr=False only for debug runs.\n"
                f"EasyOCR import error: {EASYOCR_IMPORT_ERROR}"
            )
        print("EasyOCR is not installed; continuing with no OCR fallback.")
        return NoOCRBackendAdapter(
            backend=backend,
            reason="EasyOCR is not installed. Synthetic pages will be copied from originals.",
        )

    try:
        return OCRBackendAdapter(
            backend=requested_backend,
            langs=langs or ["en"],
            gpu=gpu,
            use_angle_cls=use_angle_cls,
        )
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        install_hint = build_kaggle_paddleocr_reinstall_command()
        if strict:
            raise RuntimeError(
                "Failed to initialize the requested OCR backend with strict_ocr=True.\n"
                f"Requested backend={requested_backend!r}\n"
                f"Error={message}\n"
                "For Kaggle, this is often caused by incompatible PaddleOCR/PaddleX/PaddlePaddle "
                "versions, including the PaddlePredictorOption initialization error.\n"
                f"Recommended reinstall cell:\n{install_hint}\n"
                "Restart the runtime after reinstalling, then rerun the import/preflight cells."
            ) from exc
        print(f"{requested_backend} initialization failed; continuing with no OCR fallback. Error: {message}")
        return NoOCRBackendAdapter(backend=backend, reason=message)


def preprocess_image_for_ocr(image: Image.Image) -> np.ndarray:
    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray)
    gray = ImageEnhance.Contrast(gray).enhance(1.8)
    gray = gray.filter(ImageFilter.MedianFilter(size=3))
    enhanced = gray.convert("RGB")
    return np.array(enhanced, dtype=np.uint8)


def normalize_ocr_input(image: Image.Image, preprocess_for_ocr: bool = True) -> np.ndarray:
    ocr_image = preprocess_image_for_ocr(image) if preprocess_for_ocr else np.array(image, dtype=np.uint8)
    if ocr_image.ndim == 2:
        ocr_image = np.stack([ocr_image] * 3, axis=-1)
    elif ocr_image.ndim == 3 and ocr_image.shape[2] == 1:
        ocr_image = np.repeat(ocr_image, 3, axis=2)
    return np.ascontiguousarray(ocr_image.astype(np.uint8, copy=False))


def run_ocr_on_image(
    image_path: str | Path,
    reader,
    preprocess_for_ocr: bool = True,
    return_trace: bool = False,
):
    image = Image.open(image_path).convert("RGB")
    ocr_input = normalize_ocr_input(image, preprocess_for_ocr=preprocess_for_ocr)
    ocr_results = reader.readtext(ocr_input)
    ocr_trace = getattr(
        reader,
        "last_backend_trace",
        {
            "requested_backend": None,
            "used_backend": None,
            "detail": None,
            "fallback_reason": None,
        },
    )
    image.close()
    if return_trace:
        return ocr_results, ocr_input, dict(ocr_trace)
    return ocr_results, ocr_input


def draw_ocr_debug_overlay(
    image_path: str | Path,
    ocr_results,
    output_path: str | Path | None = None,
    max_boxes: int = 80,
):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = get_font_for_box(size=18)

    for idx, item in enumerate(ocr_results[:max_boxes]):
        bbox, text, conf = item
        xs = [int(p[0]) for p in bbox]
        ys = [int(p[1]) for p in bbox]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        label = f"{idx + 1}: {text[:40]} ({conf:.2f})"
        label_y = max(0, y1 - 18)
        draw.rectangle([x1, label_y, min(image.width, x1 + 420), y1], fill=(255, 255, 204))
        draw.text((x1 + 2, label_y), label, fill=(0, 0, 0), font=font)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
    return image


def ocr_debug_summary(ocr_results, top_k: int = 20, ocr_trace: dict[str, Any] | None = None):
    preview = []
    for idx, item in enumerate(ocr_results[:top_k]):
        bbox, text, conf = item
        preview.append(
            {
                "idx": idx + 1,
                "text": str(text),
                "conf": float(conf),
                "bbox": bbox,
            }
        )
    summary = {
        "num_boxes": len(ocr_results),
        "preview": preview,
    }
    if ocr_trace:
        summary["ocr_trace"] = ocr_trace
    return summary


def get_font_for_box(font_path: str | None = None, size: int = 24):
    if font_path and Path(font_path).exists():
        return ImageFont.truetype(font_path, size=size)
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def sample_background_color(img_np, x1, y1, x2, y2, pad=4):
    h, w = img_np.shape[:2]
    xa = max(0, x1 - pad)
    ya = max(0, y1 - pad)
    xb = min(w, x2 + pad)
    yb = min(h, y2 + pad)
    ring_pixels = []
    if ya < y1:
        ring_pixels.append(img_np[ya:y1, xa:xb])
    if y2 < yb:
        ring_pixels.append(img_np[y2:yb, xa:xb])
    if xa < x1:
        ring_pixels.append(img_np[y1:y2, xa:x1])
    if x2 < xb:
        ring_pixels.append(img_np[y1:y2, x2:xb])
    valid = [p.reshape(-1, 3) for p in ring_pixels if p.size > 0]
    if valid:
        pixels = np.concatenate(valid, axis=0)
    else:
        patch = img_np[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)]
        pixels = patch.reshape(-1, 3) if patch.size > 0 else np.array([[245, 245, 245]])
    color = np.median(pixels, axis=0).astype(int)
    return tuple(color.tolist())


def contrast_text_color(background_color):
    r, g, b = background_color
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (20, 20, 20) if luminance >= 160 else (245, 245, 245)


def sample_text_color(img_np, x1, y1, x2, y2, background_color):
    patch = img_np[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
    if patch.size == 0:
        return contrast_text_color(background_color)

    pixels = patch.reshape(-1, 3).astype(np.float32)
    if len(pixels) == 0:
        return contrast_text_color(background_color)

    bg = np.array(background_color, dtype=np.float32)
    distances = np.linalg.norm(pixels - bg, axis=1)

    min_distance = 30.0
    percentile_threshold = np.percentile(distances, 85)
    selected = pixels[distances >= max(min_distance, percentile_threshold)]

    if len(selected) < max(8, len(pixels) // 20):
        top_k = max(8, min(len(pixels), max(1, len(pixels) // 8)))
        top_indices = np.argsort(distances)[-top_k:]
        selected = pixels[top_indices]

    if len(selected) == 0:
        return contrast_text_color(background_color)

    text_color = tuple(np.median(selected, axis=0).astype(int).tolist())
    contrast = float(np.linalg.norm(np.array(text_color, dtype=np.float32) - bg))
    if contrast < 40:
        return contrast_text_color(background_color)
    return text_color


def wrap_text_to_width(draw, text, font, max_width):
    words = text.split()
    if not words:
        return []
    lines = []
    cur = words[0]
    for word in words[1:]:
        trial = cur + " " + word
        bbox = draw.textbbox((0, 0), trial, font=font)
        if bbox[2] - bbox[0] <= max_width:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    lines.append(cur)
    return lines


def fit_text_into_box(draw, text, box_w, box_h, font_path=None, min_size=10, max_size=28):
    for size in range(max_size, min_size - 1, -1):
        font = get_font_for_box(font_path, size=size)
        lines = wrap_text_to_width(draw, text, font, box_w)
        if not lines:
            return font, [""]
        line_heights = []
        max_line_width = 0
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            max_line_width = max(max_line_width, bbox[2] - bbox[0])
            line_heights.append(bbox[3] - bbox[1])
        total_h = sum(line_heights) + max(0, len(lines) - 1) * 2
        if max_line_width <= box_w and total_h <= box_h:
            return font, lines
    font = get_font_for_box(font_path, size=min_size)
    lines = wrap_text_to_width(draw, text, font, box_w)
    max_lines = max(1, box_h // (min_size + 2))
    return font, lines[:max_lines]


def translate_image_text_only(
    image_path: str | Path,
    output_path: str | Path,
    translator: MarianTranslator,
    reader,
    font_path: str | None = None,
    min_conf: float = 0.35,
    min_chars: int = 2,
    expand_box: int = 2,
    preprocess_for_ocr: bool = True,
):
    image = Image.open(image_path).convert("RGB")
    img_np = np.array(image)
    draw = ImageDraw.Draw(image)
    ocr_results, _, ocr_trace = run_ocr_on_image(
        image_path=image_path,
        reader=reader,
        preprocess_for_ocr=preprocess_for_ocr,
        return_trace=True,
    )

    boxes_detected = len(ocr_results)
    boxes_translated = 0

    for item in ocr_results:
        bbox, text, conf = item
        text = str(text).strip()
        if conf < min_conf:
            continue
        if len(text) < min_chars:
            continue
        if not any(ch.isalpha() for ch in text):
            continue

        xs = [int(p[0]) for p in bbox]
        ys = [int(p[1]) for p in bbox]
        x1 = max(0, min(xs) - expand_box)
        y1 = max(0, min(ys) - expand_box)
        x2 = min(image.width, max(xs) + expand_box)
        y2 = min(image.height, max(ys) + expand_box)

        box_w = max(10, x2 - x1)
        box_h = max(10, y2 - y1)
        translated = translator.translate(text)
        if translated is None:
            continue
        translated = translated.strip()
        if not translated:
            continue

        bg = sample_background_color(img_np, x1, y1, x2, y2, pad=5)
        text_color = sample_text_color(img_np, x1, y1, x2, y2, background_color=bg)
        draw.rectangle([x1, y1, x2, y2], fill=bg)
        font, lines = fit_text_into_box(
            draw,
            translated,
            box_w=box_w,
            box_h=box_h,
            font_path=font_path,
            min_size=10,
            max_size=28,
        )

        line_heights = []
        for line in lines:
            bb = draw.textbbox((0, 0), line, font=font)
            line_heights.append(bb[3] - bb[1])

        total_h = sum(line_heights) + max(0, len(lines) - 1) * 2
        cur_y = y1 + max(0, (box_h - total_h) // 2)
        for line, lh in zip(lines, line_heights):
            bb = draw.textbbox((0, 0), line, font=font)
            line_w = bb[2] - bb[0]
            cur_x = x1 + max(0, (box_w - line_w) // 2)
            draw.text((cur_x, cur_y), line, fill=text_color, font=font)
            cur_y += lh + 2
        boxes_translated += 1

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    image.close()
    return {
        "ocr_boxes_detected": boxes_detected,
        "ocr_boxes_translated": boxes_translated,
        "synthetic_fallback_used": boxes_translated == 0,
        "ocr_backend_requested": ocr_trace.get("requested_backend"),
        "ocr_backend_used": ocr_trace.get("used_backend"),
        "ocr_backend_detail": ocr_trace.get("detail"),
        "ocr_fallback_reason": ocr_trace.get("fallback_reason"),
    }


def build_synthetic_subset_ocr(
    records: list[SourceRecord],
    translator: MarianTranslator,
    out_dir: str | Path,
    split: str,
    subset_size: int | None = None,
    font_path: str | None = None,
    seed: int = 42,
    min_conf: float = 0.35,
    fallback_to_original: bool = True,
    reader=None,
    preprocess_for_ocr: bool = True,
):
    reader = reader or get_ocr_reader(["en"], backend="paddleocr")
    selected = [record for record in records if record.split == split]
    rng = random.Random(seed)
    selected = sorted(selected, key=lambda r: r.doc_id)
    if subset_size is not None and subset_size < len(selected):
        selected = rng.sample(selected, subset_size)
        selected = sorted(selected, key=lambda r: r.doc_id)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    synthetic_records: list[SourceRecord] = []

    stats = {
        "split": split,
        "attempted_pages": 0,
        "pages_with_ocr_boxes": 0,
        "pages_with_translated_boxes": 0,
        "fallback_pages": 0,
        "average_translated_boxes_per_page": 0.0,
        "ocr_backend_usage": {},
    }
    translated_box_counts: list[int] = []

    for idx, record in enumerate(selected):
        new_record = SourceRecord(**asdict(record))
        dst = out_dir / split / f"{new_record.doc_id}_ru.png"
        synthetic_stats = translate_image_text_only(
            image_path=new_record.original_image_path,
            output_path=dst,
            translator=translator,
            reader=reader,
            font_path=font_path,
            min_conf=min_conf,
            preprocess_for_ocr=preprocess_for_ocr,
        )
        stats["attempted_pages"] += 1
        if synthetic_stats["ocr_boxes_detected"] > 0:
            stats["pages_with_ocr_boxes"] += 1
        if synthetic_stats["ocr_boxes_translated"] > 0:
            stats["pages_with_translated_boxes"] += 1
        if synthetic_stats["synthetic_fallback_used"]:
            stats["fallback_pages"] += 1
            if fallback_to_original:
                shutil.copy(new_record.original_image_path, dst)
        translated_box_counts.append(synthetic_stats["ocr_boxes_translated"])
        backend_used = synthetic_stats.get("ocr_backend_used") or "unknown"
        stats["ocr_backend_usage"][backend_used] = stats["ocr_backend_usage"].get(backend_used, 0) + 1

        new_record.synthetic_image_path = str(dst)
        new_record.image_variant = "synthetic_ru"
        new_record.target_lang_query = "ru"
        new_record.translation_model = translator.model_name
        new_record.synthetic_mode = "ocr_box_translation"
        new_record.ocr_boxes_detected = synthetic_stats["ocr_boxes_detected"]
        new_record.ocr_boxes_translated = synthetic_stats["ocr_boxes_translated"]
        new_record.synthetic_fallback_used = synthetic_stats["synthetic_fallback_used"]
        new_record.ocr_backend_requested = synthetic_stats.get("ocr_backend_requested")
        new_record.ocr_backend_used = synthetic_stats.get("ocr_backend_used")
        new_record.ocr_backend_detail = synthetic_stats.get("ocr_backend_detail")
        new_record.ocr_fallback_reason = synthetic_stats.get("ocr_fallback_reason")
        synthetic_records.append(new_record)

        if (idx + 1) % 10 == 0:
            print(
                f"processed {idx + 1}/{len(selected)} | "
                f"translated_pages={stats['pages_with_translated_boxes']} | "
                f"fallback={stats['fallback_pages']} | "
                f"ocr_backend={backend_used}"
            )

    if translated_box_counts:
        stats["average_translated_boxes_per_page"] = sum(translated_box_counts) / len(translated_box_counts)
    return synthetic_records, stats


def build_synthetic_gallery(records: list[SourceRecord], output_dir: str | Path, max_items: int = 12):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gallery_rows = []
    for record in records[:max_items]:
        if not record.original_image_path or not record.synthetic_image_path:
            continue
        original = Image.open(record.original_image_path).convert("RGB")
        synthetic = Image.open(record.synthetic_image_path).convert("RGB")
        width = max(original.width, synthetic.width)
        height = original.height + synthetic.height + 40
        canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        canvas.paste(original, (0, 20))
        canvas.paste(synthetic, (0, 20 + original.height))
        font = get_font_for_box(size=18)
        draw.text((10, 0), f"Original: {record.doc_id}", fill=(0, 0, 0), font=font)
        draw.text((10, 20 + original.height), "Synthetic RU", fill=(0, 0, 0), font=font)
        out_path = output_dir / f"{record.doc_id}_compare.png"
        canvas.save(out_path)
        original.close()
        synthetic.close()
        canvas.close()
        gallery_rows.append({"doc_id": record.doc_id, "gallery_path": str(out_path)})
    return gallery_rows


def normalize_synthetic_subset_sizes(
    synthetic_subset_size: int | dict[str, int | None] | None,
    splits: tuple[str, ...],
):
    if synthetic_subset_size is None:
        return {split: None for split in splits}
    if isinstance(synthetic_subset_size, int):
        return {split: synthetic_subset_size for split in splits}
    return {split: synthetic_subset_size.get(split) for split in splits}


def record_to_manifest_row(record: SourceRecord, image_path: str, query_text: str, image_variant: str):
    return {
        "query_id": f"q_{record.row_id}",
        "doc_id": record.doc_id,
        "query": query_text,
        "image_path": image_path,
        "original_image_path": record.original_image_path,
        "split": record.split,
        "image_variant": image_variant,
        "source_lang_query": record.source_lang_query,
        "target_lang_query": record.target_lang_query,
        "translation_model": record.translation_model,
        "ocr_boxes_detected": record.ocr_boxes_detected,
        "ocr_boxes_translated": record.ocr_boxes_translated,
        "synthetic_fallback_used": record.synthetic_fallback_used,
        "synthetic_mode": record.synthetic_mode,
        "ocr_backend_requested": record.ocr_backend_requested,
        "ocr_backend_used": record.ocr_backend_used,
        "ocr_backend_detail": record.ocr_backend_detail,
        "ocr_fallback_reason": record.ocr_fallback_reason,
        "seed": record.seed,
        "source": record.source,
        "answer": record.answer,
        "original_query": record.query,
    }


def export_split_manifests(
    records: list[SourceRecord],
    output_dir: str | Path,
    manifest_prefix: str,
    query_lang: str,
    image_variant: str,
    include_fallback: bool = True,
    source_lang_query: str | None = None,
    target_lang_query: str | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split in ["train", "val", "test"]:
        split_rows = []
        for record in records:
            if record.split != split:
                continue
            if image_variant == "synthetic_ru" and not include_fallback and record.synthetic_fallback_used:
                continue
            query_text = record.query if query_lang == "en" else (record.query_ru or record.query)
            image_path = record.original_image_path if image_variant == "original_en" else record.synthetic_image_path
            if not image_path:
                continue
            manifest_record = SourceRecord(
                **{
                    **asdict(record),
                    "source_lang_query": source_lang_query or record.source_lang_query,
                    "target_lang_query": target_lang_query or record.target_lang_query,
                }
            )
            split_rows.append(record_to_manifest_row(manifest_record, image_path, query_text, image_variant))
        path = output_dir / f"{manifest_prefix}_{split}.jsonl"
        export_jsonl(split_rows, path)
        paths[split] = str(path)
    return paths


def export_manual_russian_template(output_path: str | Path):
    example_rows = [
        {
            "query_id": "manual_q_001",
            "doc_id": "manual_doc_001",
            "query": "Найдите номер договора в документе",
            "image_path": "/absolute/path/to/russian_doc_001.png",
            "split": "manual_test",
            "image_variant": "real_ru",
            "source_lang_query": "ru",
            "target_lang_query": "ru",
            "category": "contract",
            "difficulty": "medium",
            "notes": "Optional free-text note",
        }
    ]
    export_jsonl(example_rows, output_path)


def read_manifest_doc_ids(manifest_path: str | Path) -> list[str]:
    rows = read_jsonl(manifest_path)
    seen = set()
    doc_ids = []
    for row in rows:
        doc_id = row["doc_id"]
        if doc_id not in seen:
            seen.add(doc_id)
            doc_ids.append(doc_id)
    return doc_ids


def filter_manifest_rows(
    manifest_path: str | Path,
    output_path: str | Path,
    allowed_doc_ids: set[str] | None = None,
    allowed_query_ids: set[str] | None = None,
):
    rows = read_jsonl(manifest_path)
    filtered = []
    for row in rows:
        if allowed_doc_ids is not None and row["doc_id"] not in allowed_doc_ids:
            continue
        if allowed_query_ids is not None and row["query_id"] not in allowed_query_ids:
            continue
        filtered.append(row)
    export_jsonl(filtered, output_path)
    return str(output_path), len(filtered)


def build_aligned_test_manifests(
    baseline_test_manifest: str | Path,
    synthetic_attempted_test_manifest: str | Path,
    synthetic_primary_test_manifest: str | Path,
    output_dir: str | Path,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_doc_ids = set(read_manifest_doc_ids(baseline_test_manifest))
    synthetic_attempted_doc_ids = set(read_manifest_doc_ids(synthetic_attempted_test_manifest))
    synthetic_primary_doc_ids = set(read_manifest_doc_ids(synthetic_primary_test_manifest))

    attempted_shared_doc_ids = baseline_doc_ids & synthetic_attempted_doc_ids
    primary_shared_doc_ids = baseline_doc_ids & synthetic_primary_doc_ids

    artifacts = {}
    baseline_attempted_path, baseline_attempted_rows = filter_manifest_rows(
        baseline_test_manifest,
        output_dir / "baseline_aligned_to_synthetic_attempted_test.jsonl",
        allowed_doc_ids=attempted_shared_doc_ids,
    )
    synthetic_attempted_path, synthetic_attempted_rows = filter_manifest_rows(
        synthetic_attempted_test_manifest,
        output_dir / "synthetic_attempted_aligned_test.jsonl",
        allowed_doc_ids=attempted_shared_doc_ids,
    )

    baseline_primary_path, baseline_primary_rows = filter_manifest_rows(
        baseline_test_manifest,
        output_dir / "baseline_aligned_to_synthetic_primary_test.jsonl",
        allowed_doc_ids=primary_shared_doc_ids,
    )
    synthetic_primary_path, synthetic_primary_rows = filter_manifest_rows(
        synthetic_primary_test_manifest,
        output_dir / "synthetic_primary_aligned_test.jsonl",
        allowed_doc_ids=primary_shared_doc_ids,
    )

    artifacts["synthetic_attempted"] = {
        "baseline_manifest": baseline_attempted_path,
        "synthetic_manifest": synthetic_attempted_path,
        "shared_doc_count": len(attempted_shared_doc_ids),
        "baseline_rows": baseline_attempted_rows,
        "synthetic_rows": synthetic_attempted_rows,
    }
    artifacts["synthetic_primary"] = {
        "baseline_manifest": baseline_primary_path,
        "synthetic_manifest": synthetic_primary_path,
        "shared_doc_count": len(primary_shared_doc_ids),
        "baseline_rows": baseline_primary_rows,
        "synthetic_rows": synthetic_primary_rows,
    }
    export_json(output_dir / "aligned_test_manifest_summary.json", artifacts)
    return artifacts


def validate_split_integrity(manifest_paths: dict[str, str]):
    split_docs = {}
    for split, path in manifest_paths.items():
        split_docs[split] = {row["doc_id"] for row in read_jsonl(path)}
    overlap_report = {}
    keys = list(split_docs.keys())
    for i, left in enumerate(keys):
        for right in keys[i + 1 :]:
            overlap = sorted(split_docs[left] & split_docs[right])
            overlap_report[f"{left}__{right}"] = overlap[:20]
            if overlap:
                raise ValueError(f"doc_id leakage detected between {left} and {right}")
    return overlap_report


def build_eval_sets_from_manifest(
    manifest_path: str | Path,
    candidate_manifest_path: str | Path | None = None,
    limit_queries: int | None = None,
    limit_docs: int | None = None,
):
    rows = read_jsonl(manifest_path)
    if limit_queries is not None:
        rows = rows[:limit_queries]

    candidate_rows = rows if candidate_manifest_path is None else read_jsonl(candidate_manifest_path)
    doc_rows = []
    seen_docs = set()
    for row in candidate_rows:
        doc_id = row["doc_id"]
        if doc_id not in seen_docs:
            doc_rows.append({"doc_id": doc_id, "image_path": row["image_path"]})
            seen_docs.add(doc_id)

    if limit_docs is not None:
        doc_rows = doc_rows[:limit_docs]
        allowed = {d["doc_id"] for d in doc_rows}
        rows = [r for r in rows if r["doc_id"] in allowed]

    query_rows = [{"query_id": r["query_id"], "query": r["query"], "doc_id": r["doc_id"]} for r in rows]
    return query_rows, doc_rows


def get_runtime_versions() -> dict[str, str]:
    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": getattr(peft, "__version__", None),
        "accelerate": getattr(accelerate, "__version__", None),
    }


def build_kaggle_reinstall_command(
    transformers_version: str = PINNED_TRANSFORMERS_VERSION,
    colpali_engine_version: str = PINNED_COLPALI_ENGINE_VERSION,
    peft_version: str = PINNED_PEFT_VERSION,
    accelerate_version: str = PINNED_ACCELERATE_VERSION,
) -> str:
    return (
        "!pip -q install -U --upgrade-strategy only-if-needed "
        f"\"transformers=={transformers_version}\" "
        f"\"colpali-engine=={colpali_engine_version}\" "
        f"\"peft=={peft_version}\" "
        f"\"accelerate=={accelerate_version}\""
    )


def build_kaggle_paddleocr_reinstall_command(
    paddleocr_version: str = PINNED_PADDLEOCR_VERSION,
    paddlepaddle_version: str = PINNED_PADDLEPADDLE_VERSION,
    numpy_version: str = PINNED_NUMPY_VERSION,
    opencv_version: str = PINNED_OPENCV_VERSION,
    scipy_version: str = PINNED_SCIPY_VERSION,
    scikit_image_version: str = PINNED_SCIKIT_IMAGE_VERSION,
) -> str:
    return (
        "%pip -q uninstall -y easyocr paddleocr paddlex paddlepaddle paddlepaddle-gpu opencv-python opencv-contrib-python opencv-python-headless scipy scikit-image\n"
        "%pip -q install -U pip setuptools wheel\n"
        f"%pip -q install --force-reinstall --no-cache-dir \"numpy=={numpy_version}\"\n"
        f"%pip -q install --no-deps --force-reinstall --no-cache-dir --prefer-binary \"opencv-python-headless=={opencv_version}\"\n"
        f"%pip -q install --no-deps --force-reinstall --no-cache-dir --prefer-binary \"scipy=={scipy_version}\" \"scikit-image=={scikit_image_version}\"\n"
        f"%pip -q install --prefer-binary \"paddlepaddle=={paddlepaddle_version}\"\n"
        "%pip -q install --prefer-binary pyclipper lmdb rapidfuzz premailer attrdict visualdl fire\n"
        f"%pip -q install --no-deps --prefer-binary --only-binary=:all: \"paddleocr=={paddleocr_version}\"\n"
        f"%pip -q install --force-reinstall --no-cache-dir \"numpy=={numpy_version}\"\n"
        "import os\n"
        "os.kill(os.getpid(), 9)"
    )


def ensure_colmodernvbert_available():
    has_transformers_backend = ColModernVBertForRetrieval is not None and ColModernVBertProcessor is not None
    has_colpali_backend = ColPaliModernVBert is not None and ColPaliModernVBertProcessor is not None
    if not has_transformers_backend and not has_colpali_backend:
        raise ImportError(
            "This pipeline requires either a transformers build that exports "
            "ColModernVBertForRetrieval and ColModernVBertProcessor, or "
            "colpali_engine with ColModernVBert support. "
            f"Installed transformers={transformers.__version__}. "
            f"Install the pinned Kaggle stack first: {build_kaggle_reinstall_command()}. "
            "If public ModernVBERT checkpoints still fail, install "
            "`git+https://github.com/illuin-tech/colpali.git@vbert` and restart the runtime."
        )


def format_exception_brief(exc: Exception) -> str:
    message = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def ensure_transformers_peft_compatibility():
    if peft is None:
        return
    try:
        import peft.utils.save_and_load as peft_save_and_load
    except Exception as exc:
        raise RuntimeError(
            "Failed to import PEFT save/load helpers required by the retrieval stack.\n"
            f"Installed transformers={transformers.__version__}\n"
            f"Installed peft={getattr(peft, '__version__', None)}\n"
            f"Installed accelerate={getattr(accelerate, '__version__', None)}\n"
            f"Recommended reinstall: {build_kaggle_reinstall_command()}\n"
            f"Error={format_exception_brief(exc)}"
        ) from exc

    try:
        from transformers.integrations import peft as transformers_peft_integration
    except Exception:
        return

    expects_tp_shard_helper = False
    try:
        integration_source = inspect.getsource(transformers_peft_integration)
        expects_tp_shard_helper = "_maybe_shard_state_dict_for_tp" in integration_source
    except (OSError, TypeError):
        expects_tp_shard_helper = False

    if expects_tp_shard_helper and not hasattr(peft_save_and_load, "_maybe_shard_state_dict_for_tp"):
        raise RuntimeError(
            "Installed transformers and peft are incompatible for adapter loading.\n"
            f"Installed transformers={transformers.__version__}\n"
            f"Installed peft={getattr(peft, '__version__', None)}\n"
            f"Installed accelerate={getattr(accelerate, '__version__', None)}\n"
            f"Recommended reinstall: {build_kaggle_reinstall_command()}\n"
            "ColModernVBERT is published as an adapter-backed retriever, so broad "
            "transformers version ranges can break auto-loading in Kaggle."
        )


def _trim_embedding_rows(embedding, attention_mask=None):
    if attention_mask is None:
        return embedding
    valid = attention_mask.bool()
    return embedding[valid]


def _batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def is_local_colmodernvbert_checkpoint(model_name: str | Path) -> bool:
    path = Path(model_name).expanduser()
    if not path.exists() or not path.is_dir():
        return False
    if (path / "adapter_config.json").exists():
        return False

    path_hint = any(token in path.name.lower() for token in ("modernvbert", "colmodernvbert"))
    processor_files = ("preprocessor_config.json", "tokenizer_config.json", "config.json")
    if not (path_hint or any((path / filename).exists() for filename in processor_files)):
        return False

    config_path = path / "config.json"
    if not config_path.exists():
        return path_hint

    try:
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return path_hint

    payload_text = json.dumps(config_payload, ensure_ascii=False).lower()
    return "modernvbert" in payload_text or "colmodernvbert" in payload_text or path_hint


def load_colmodernvbert_processor(model_name: str | Path):
    ensure_colmodernvbert_available()
    model_name = str(model_name)
    errors: list[tuple[str, Exception]] = []

    if ColPaliModernVBertProcessor is not None and (
        model_name.startswith("ModernVBERT/") or is_local_colmodernvbert_checkpoint(model_name)
    ):
        try:
            return ColPaliModernVBertProcessor.from_pretrained(model_name)
        except Exception as exc:
            errors.append(("colpali_processor_from_pretrained", exc))

    try:
        return ColModernVBertProcessor.from_pretrained(model_name)
    except Exception as exc:
        errors.append(("processor_from_pretrained", exc))

    try:
        image_processor = AutoImageProcessor.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return ColModernVBertProcessor(image_processor=image_processor, tokenizer=tokenizer)
    except Exception as exc:
        errors.append(("explicit_construction", exc))

    detail = "; ".join(f"{label}={format_exception_brief(exc)}" for label, exc in errors)
    raise RuntimeError(f"Failed to load ColModernVBertProcessor for {model_name}. {detail}")


def set_processor_image_size(processor, target_size: tuple[int, int] | None):
    if target_size is None:
        return processor

    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return processor

    width, height = int(target_size[0]), int(target_size[1])
    size_dict = {"width": width, "height": height}
    longest_edge_dict = {"longest_edge": max(width, height)}

    for attr in ("size", "crop_size", "image_size"):
        value = getattr(image_processor, attr, None)
        if isinstance(value, dict):
            if "longest_edge" in value and "height" not in value and "width" not in value:
                setattr(image_processor, attr, dict(longest_edge_dict))
            else:
                setattr(image_processor, attr, dict(size_dict))
        elif value is None:
            if attr == "size":
                setattr(image_processor, attr, dict(longest_edge_dict))
            else:
                setattr(image_processor, attr, dict(size_dict))
        elif isinstance(value, int):
            setattr(image_processor, attr, max(width, height))
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            setattr(image_processor, attr, (width, height))

    max_image_size = getattr(image_processor, "max_image_size", None)
    if isinstance(max_image_size, dict) or max_image_size is None:
        image_processor.max_image_size = dict(longest_edge_dict)
    elif isinstance(max_image_size, int):
        image_processor.max_image_size = max(width, height)

    if hasattr(image_processor, "do_image_splitting"):
        image_processor.do_image_splitting = False
    if hasattr(image_processor, "split_image"):
        image_processor.split_image = False

    return processor


def align_colmodernvbert_token_embeddings(model, processor):
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return model

    tokenizer_size = len(tokenizer)
    embedding_layer = model.get_input_embeddings()
    embedding_size = embedding_layer.weight.shape[0]
    if tokenizer_size <= embedding_size:
        return model

    print(
        f"Resizing token embeddings from {embedding_size} to {tokenizer_size} "
        "to match the tokenizer vocabulary."
    )
    model.resize_token_embeddings(tokenizer_size)

    if hasattr(model.config, "vocab_size"):
        model.config.vocab_size = tokenizer_size
    vlm_config = getattr(model.config, "vlm_config", None)
    if vlm_config is not None and hasattr(vlm_config, "vocab_size"):
        vlm_config.vocab_size = tokenizer_size
    text_config = getattr(vlm_config, "text_config", None) if vlm_config is not None else None
    if text_config is not None and hasattr(text_config, "vocab_size"):
        text_config.vocab_size = tokenizer_size

    return model


def _normalize_processor_output(output):
    if isinstance(output, dict):
        return dict(output)
    if hasattr(output, "items"):
        return dict(output.items())
    raise TypeError(f"Unsupported processor output type: {type(output)}")


def resolve_retriever_source(model_name: str | Path) -> RetrieverSourceSpec:
    model_name = str(model_name)
    redirected_model_name = LEGACY_RETRIEVER_MODEL_REDIRECTS.get(model_name, model_name)
    if is_local_colmodernvbert_checkpoint(redirected_model_name):
        if ColPaliModernVBert is None or ColPaliModernVBertProcessor is None:
            raise RuntimeError(
                "Local merged ModernVBERT retrieval checkpoints require colpali_engine "
                "with ModernVBERT support.\n"
                f"Install `colpali-engine=={PINNED_COLPALI_ENGINE_VERSION}` (or newer) "
                "and restart the runtime."
            )
        return RetrieverSourceSpec(
            requested_model_name=model_name,
            processor_name=redirected_model_name,
            resolved_model_name=redirected_model_name,
            built_in_adapter_dir=None,
            backend="colpali",
        )
    if redirected_model_name == "ModernVBERT/colmodernvbert-merged":
        if ColPaliModernVBert is None or ColPaliModernVBertProcessor is None:
            raise RuntimeError(
                "Public ModernVBERT retrieval checkpoints require colpali_engine with ModernVBERT support.\n"
                f"Install `colpali-engine=={PINNED_COLPALI_ENGINE_VERSION}` (or newer) and restart the runtime."
            )
        backend = "colpali" if ColPaliModernVBert is not None and ColPaliModernVBertProcessor is not None else "transformers"
        return RetrieverSourceSpec(
            requested_model_name=model_name,
            processor_name=PUBLIC_RETRIEVER_PROCESSOR_NAME,
            resolved_model_name=redirected_model_name,
            built_in_adapter_dir=None,
            backend=backend,
        )

    ensure_transformers_peft_compatibility()
    try:
        peft_config = peft.PeftConfig.from_pretrained(redirected_model_name) if peft is not None else None
        if peft_config is None:
            raise RuntimeError("PEFT is not installed.")
    except Exception:
        return RetrieverSourceSpec(
            requested_model_name=model_name,
            processor_name=redirected_model_name,
            resolved_model_name=redirected_model_name,
            built_in_adapter_dir=None,
            backend="transformers",
        )

    base_model_name = getattr(peft_config, "base_model_name_or_path", None)
    if not base_model_name:
        raise RuntimeError(
            f"Retriever adapter config for {model_name!r} is missing base_model_name_or_path."
        )

    return RetrieverSourceSpec(
        requested_model_name=model_name,
        processor_name=redirected_model_name,
        resolved_model_name=base_model_name,
        built_in_adapter_dir=redirected_model_name,
        backend="transformers",
    )


def attach_retriever_adapter(model, adapter_dir: str, adapter_name: str = "default"):
    if hasattr(model, "load_adapter"):
        model.load_adapter(adapter_dir, adapter_name=adapter_name)
        if hasattr(model, "set_adapter"):
            model.set_adapter(adapter_name)
        return model
    if PeftModel is None:
        raise ImportError("peft is required to attach external retriever adapters.")
    return PeftModel.from_pretrained(model, adapter_dir, adapter_name=adapter_name)


def export_merged_retriever_checkpoint(
    model_name: str | Path,
    adapter_dir: str | Path,
    output_dir: str | Path,
    processor=None,
):
    adapter_dir = Path(adapter_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime = load_retriever(model_name=str(model_name), device="cpu")
    if PeftModel is None:
        raise ImportError("peft is required to merge legacy LoRA adapters.")
    merged_model = PeftModel.from_pretrained(runtime.model, str(adapter_dir))
    if not hasattr(merged_model, "merge_and_unload"):
        raise RuntimeError("The trained retriever adapter does not support merge_and_unload().")
    merged_model = merged_model.merge_and_unload()
    merged_model.save_pretrained(output_dir)

    processor_to_save = processor or runtime.processor
    processor_to_save.save_pretrained(output_dir)
    return str(output_dir)


def load_retriever_candidate(
    source_spec: RetrieverSourceSpec,
    adapter_dir: str | None = None,
    device: str | None = None,
):
    ensure_colmodernvbert_available()
    if source_spec.backend != "colpali" or adapter_dir is not None or source_spec.built_in_adapter_dir is not None:
        ensure_transformers_peft_compatibility()
    runtime_device = device or DEVICE
    _, dtype = pick_device_and_dtype()
    processor = load_colmodernvbert_processor(source_spec.processor_name)
    model_load_name = source_spec.resolved_model_name
    if source_spec.backend == "colpali":
        model = ColPaliModernVBert.from_pretrained(
            model_load_name,
            torch_dtype=dtype if runtime_device != "cpu" else torch.float32,
            trust_remote_code=True,
        )
        if adapter_dir is not None:
            raise RuntimeError("adapter_dir is not supported with the colpali_engine ColModernVBert backend.")
        model = model.to(runtime_device).eval()
    else:
        model = ColModernVBertForRetrieval.from_pretrained(
            model_load_name,
            torch_dtype=dtype if runtime_device != "cpu" else torch.float32,
        )
        model = align_colmodernvbert_token_embeddings(model, processor)
        if source_spec.built_in_adapter_dir is not None:
            model = attach_retriever_adapter(model, source_spec.built_in_adapter_dir, adapter_name="default")
        if adapter_dir is not None:
            model = attach_retriever_adapter(model, adapter_dir, adapter_name="task_adapter")
        model = model.to(runtime_device).eval()
    return RetrievalRuntime(
        requested_model_name=source_spec.requested_model_name,
        resolved_model_name=model_load_name,
        device=runtime_device,
        processor=processor,
        model=model,
        versions=get_runtime_versions(),
        backend=source_spec.backend,
    )


def load_retriever(
    model_name: str | Path = DEFAULT_RETRIEVAL_MODEL_NAME,
    adapter_dir: str | None = None,
    device: str | None = None,
):
    ensure_colmodernvbert_available()
    try:
        source_spec = resolve_retriever_source(model_name)
        return load_retriever_candidate(
            source_spec=source_spec,
            adapter_dir=adapter_dir,
            device=device,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load the requested ColModernVBert retriever.\n"
            f"Installed transformers={transformers.__version__}\n"
            f"Installed peft={getattr(peft, '__version__', None)}\n"
            f"Installed accelerate={getattr(accelerate, '__version__', None)}\n"
            f"Requested model_name={model_name!r}\n"
            f"Recommended reinstall: {build_kaggle_reinstall_command()}\n"
            f"Error={format_exception_brief(exc)}"
        ) from exc


def get_tensor_shape(value) -> list[int] | None:
    if torch.is_tensor(value):
        return list(value.shape)
    if isinstance(value, np.ndarray):
        return list(value.shape)
    if isinstance(value, (list, tuple)):
        return [len(value)]
    return None


def process_query_batch(processor, texts: list[str]):
    candidate_calls = []
    candidate_calls.extend(
        [
            lambda: processor.process_queries(texts, return_tensors="pt"),
            lambda: processor.process_queries(texts),
            lambda: processor.process_texts(texts),
            lambda: processor.process_queries(texts),
            lambda: processor(text=texts, return_tensors="pt", padding=True, truncation=True),
        ]
    )
    last_error = None
    for call in candidate_calls:
        try:
            return _normalize_processor_output(call())
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Failed to process query batch: {format_exception_brief(last_error)}")


def process_image_batch(processor, images: list[Image.Image]):
    candidate_calls = []
    blank_texts = [""] * len(images)
    candidate_calls.extend(
        [
            lambda: processor(images=images, return_tensors="pt"),
            lambda: processor(images=images, text=blank_texts, return_tensors="pt"),
            lambda: processor.process_images(images, return_tensors="pt"),
            lambda: processor.process_images(images),
        ]
    )
    last_error = None
    for call in candidate_calls:
        try:
            output = _normalize_processor_output(call())
            if output.get("input_ids") is None:
                last_error = TypeError("Processor image batch did not include input_ids.")
                continue
            return output
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Failed to process image batch: {format_exception_brief(last_error)}")


def get_position_embedding_count(model) -> int | None:
    for _, module in model.named_modules():
        num_embeddings = getattr(module, "num_embeddings", None)
        if isinstance(num_embeddings, int) and num_embeddings > 0:
            return num_embeddings
        weight = getattr(module, "weight", None)
        if torch.is_tensor(weight) and weight.ndim >= 2 and "position" in type(module).__name__.lower():
            return int(weight.shape[0])
    return None


def get_processed_patch_grid(pixel_values: torch.Tensor, patch_size: int) -> tuple[int, int] | None:
    if not torch.is_tensor(pixel_values):
        return None
    if pixel_values.ndim == 5:
        height = int(pixel_values.shape[-2])
        width = int(pixel_values.shape[-1])
    elif pixel_values.ndim == 4:
        height = int(pixel_values.shape[-2])
        width = int(pixel_values.shape[-1])
    else:
        return None
    if patch_size <= 0 or height % patch_size != 0 or width % patch_size != 0:
        return None
    return height // patch_size, width // patch_size


def build_retrieval_image_contract(runtime: RetrievalRuntime, image_inputs: dict[str, Any]) -> dict[str, Any]:
    model_contract = build_model_image_contract(runtime.model, image_inputs)
    model_contract["resolved_model_name"] = runtime.resolved_model_name
    model_contract["processor_image_size"] = list(get_processor_image_size(runtime.processor) or []) or None
    return model_contract


def build_model_image_contract(model, image_inputs: dict[str, Any]) -> dict[str, Any]:
    pixel_values = image_inputs.get("pixel_values")
    model_image_size = get_model_image_size(model)
    patch_size = infer_patch_size(model)
    pixel_shuffle_factor = infer_pixel_shuffle_factor(model)
    processed_grid = (
        get_processed_patch_grid(pixel_values, patch_size)
        if patch_size is not None and torch.is_tensor(pixel_values)
        else None
    )
    processed_token_count = None
    if processed_grid is not None:
        processed_token_count = int(processed_grid[0] * processed_grid[1])
    expected_grid = None
    if model_image_size is not None and patch_size is not None:
        expected_grid = (
            max(1, int(model_image_size[1] // patch_size)),
            max(1, int(model_image_size[0] // patch_size)),
        )
    position_embedding_count = None
    if expected_grid is not None:
        position_embedding_count = int(expected_grid[0] * expected_grid[1])
    else:
        position_embedding_count = get_position_embedding_count(model)
        if position_embedding_count is not None:
            grid_side = int(round(math.sqrt(position_embedding_count)))
            if grid_side * grid_side == position_embedding_count:
                expected_grid = (grid_side, grid_side)
    return {
        "model_image_size": list(model_image_size or []) or None,
        "patch_size": patch_size,
        "pixel_shuffle_factor": pixel_shuffle_factor,
        "position_embedding_count": position_embedding_count,
        "expected_patch_grid": list(expected_grid) if expected_grid is not None else None,
        "pixel_values_shape": get_tensor_shape(pixel_values),
        "processed_patch_grid": list(processed_grid) if processed_grid is not None else None,
        "processed_patch_token_count": processed_token_count,
        "pixel_attention_mask_shape": get_tensor_shape(image_inputs.get("pixel_attention_mask")),
    }


def validate_retrieval_image_contract(contract: dict[str, Any]):
    position_embedding_count = contract.get("position_embedding_count")
    processed_token_count = contract.get("processed_patch_token_count")
    patch_size = contract.get("patch_size")
    pixel_values_shape = contract.get("pixel_values_shape")
    if position_embedding_count is None:
        raise RuntimeError(
            "Unable to infer the vision tower position embedding count before image forward. "
            f"Contract={json.dumps(contract, ensure_ascii=False)}"
        )
    if patch_size is None:
        raise RuntimeError(
            "Unable to infer the vision patch size before image forward. "
            f"Contract={json.dumps(contract, ensure_ascii=False)}"
        )
    if pixel_values_shape is None:
        raise RuntimeError(
            "Processor output does not expose a usable `pixel_values` tensor before image forward. "
            f"Contract={json.dumps(contract, ensure_ascii=False)}"
        )
    if processed_token_count is None:
        raise RuntimeError(
            "Unable to derive the processed image patch grid before image forward. "
            f"Contract={json.dumps(contract, ensure_ascii=False)}"
        )

    # ColModernVBert can legitimately receive split-image batches from its processor,
    # which produce 5D pixel tensors like [batch, num_tiles, channels, height, width].
    # In that mode, comparing a single-tile patch grid against model position embeddings
    # is misleading and rejects valid inputs before model forward.
    if isinstance(pixel_values_shape, list) and len(pixel_values_shape) == 5:
        return

    if processed_token_count != position_embedding_count:
        raise RuntimeError(
            "Processor/model image contract mismatch detected before image forward. "
            f"Expected {position_embedding_count} vision position tokens but processor produced "
            f"{processed_token_count}. Contract={json.dumps(contract, ensure_ascii=False)}"
        )


def score_retrieval_with_processor(
    processor,
    query_embeddings,
    doc_embeddings,
    score_batch_size=64,
):
    if hasattr(processor, "score_multi_vector"):
        return processor.score_multi_vector(query_embeddings, doc_embeddings)
    if hasattr(processor, "score"):
        return processor.score(query_embeddings, doc_embeddings)
    return processor.score_retrieval(
        query_embeddings=query_embeddings,
        passage_embeddings=doc_embeddings,
        batch_size=score_batch_size,
        output_device="cpu",
    )


def run_model_for_embeddings(model, inputs: dict[str, Any], is_image_batch: bool = False):
    kwargs = dict(inputs)
    if is_image_batch:
        validate_retrieval_image_contract(build_model_image_contract(model, kwargs))
        try:
            output = model(**kwargs, interpolate_pos_encoding=True)
        except TypeError:
            output = model(**kwargs)
    else:
        output = model(**kwargs)
    embeddings = getattr(output, "embeddings", None)
    if embeddings is None and torch.is_tensor(output):
        embeddings = output
    if embeddings is None:
        raise RuntimeError("Model output does not expose `.embeddings`.")
    return embeddings


def run_retrieval_preflight(
    model_name: str = DEFAULT_RETRIEVAL_MODEL_NAME,
    device: str | None = None,
):
    runtime = load_retriever(model_name=model_name, device=device)
    image_size = get_processor_image_size(runtime.processor) or (448, 448)
    sample_image = Image.new("RGB", image_size, color=(255, 255, 255))
    sample_query = "What is written in the document?"
    query_inputs = process_query_batch(runtime.processor, [sample_query])
    image_inputs = process_image_batch(runtime.processor, [sample_image])
    query_inputs = {k: v.to(runtime.device) for k, v in query_inputs.items()}
    image_inputs = {k: v.to(runtime.device) for k, v in image_inputs.items()}
    image_contract = build_retrieval_image_contract(runtime, image_inputs)
    validate_retrieval_image_contract(image_contract)
    query_embeddings = run_model_for_embeddings(runtime.model, query_inputs, is_image_batch=False)
    image_embeddings = run_model_for_embeddings(runtime.model, image_inputs, is_image_batch=True)
    score_retrieval_with_processor(
        runtime.processor,
        [query_embeddings[0].detach().cpu()],
        [image_embeddings[0].detach().cpu()],
        score_batch_size=1,
    )
    return {
        "requested_model_name": runtime.requested_model_name,
        "resolved_model_name": runtime.resolved_model_name,
        "device": runtime.device,
        "versions": runtime.versions,
        "image_contract": image_contract,
        "text_forward_ok": True,
        "image_forward_ok": True,
        "score_ok": True,
    }


@torch.no_grad()
def encode_queries(model, processor, query_rows, batch_size=8, device: str | None = None):
    runtime_device = device or DEVICE
    encoded = []
    for batch in tqdm(list(_batched(query_rows, batch_size)), desc="Encoding queries"):
        texts = [row["query"] for row in batch]
        inputs = process_query_batch(processor, texts)
        input_ids = inputs.get("input_ids")
        if input_ids is not None:
            vocab_size = model.get_input_embeddings().weight.shape[0]
            max_input_id = int(input_ids.max().item())
            min_input_id = int(input_ids.min().item())
            if max_input_id >= vocab_size or min_input_id < 0:
                raise ValueError(
                    "Tokenizer/model vocabulary mismatch detected before query encoding. "
                    f"input_id_range=[{min_input_id}, {max_input_id}], vocab_size={vocab_size}, "
                    f"sample_query={texts[0]!r}"
                )
        inputs = {k: v.to(runtime_device) for k, v in inputs.items()}
        outputs = run_model_for_embeddings(model, inputs, is_image_batch=False)
        attention_mask = inputs.get("attention_mask")
        for i, row in enumerate(batch):
            emb = _trim_embedding_rows(outputs[i], attention_mask[i] if attention_mask is not None else None).detach().cpu()
            encoded.append((row["query_id"], emb))
    return encoded


def get_processor_image_size(processor) -> tuple[int, int] | None:
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return None

    candidates: list[tuple[int, int]] = []
    for attr in ("max_image_size", "crop_size", "image_size", "size"):
        value = getattr(image_processor, attr, None)
        if isinstance(value, dict):
            height = value.get("height") or value.get("shortest_edge") or value.get("longest_edge")
            width = value.get("width") or value.get("shortest_edge") or value.get("longest_edge")
            if height and width:
                candidates.append((int(width), int(height)))
        if isinstance(value, (tuple, list)) and len(value) == 2:
            candidates.append((int(value[0]), int(value[1])))
        if isinstance(value, int):
            candidates.append((int(value), int(value)))

    if not candidates:
        return None
    return candidates[0]


def prepare_images_for_processor(images: list[Image.Image], processor) -> list[Image.Image]:
    target_size = get_processor_image_size(processor)
    if target_size is None:
        return images
    target_width, target_height = target_size
    prepared = []
    for image in images:
        if image.size == target_size:
            prepared.append(image)
        else:
            prepared.append(image.resize((target_width, target_height), Image.Resampling.BICUBIC))
    return prepared


def get_model_image_size(model) -> tuple[int, int] | None:
    config_candidates = [
        getattr(model, "config", None),
        getattr(getattr(model, "config", None), "vlm_config", None),
        getattr(getattr(getattr(model, "config", None), "vlm_config", None), "vision_config", None),
        getattr(getattr(model, "config", None), "vision_config", None),
        getattr(getattr(model, "vision_tower", None), "config", None),
        getattr(getattr(model, "vision_model", None), "config", None),
    ]

    for config in config_candidates:
        if config is None:
            continue
        size = getattr(config, "image_size", None)
        if isinstance(size, int) and size > 0:
            return (size, size)
        if isinstance(size, (tuple, list)) and len(size) == 2:
            return (int(size[0]), int(size[1]))
        if isinstance(size, dict):
            height = size.get("height") or size.get("shortest_edge")
            width = size.get("width") or size.get("shortest_edge")
            if height and width:
                return (int(width), int(height))
        vision_model_name = getattr(config, "vision_model_name", None)
        if isinstance(vision_model_name, str):
            if "patch16-512" in vision_model_name or vision_model_name.endswith("-512"):
                return (512, 512)
    return None


def infer_image_size_from_vision_modules(model) -> tuple[int, int] | None:
    position_embeddings = []
    patch_sizes = []

    for _, module in model.named_modules():
        num_embeddings = getattr(module, "num_embeddings", None)
        if isinstance(num_embeddings, int) and num_embeddings > 0:
            position_embeddings.append(num_embeddings)

        patch_size = getattr(module, "patch_size", None)
        if isinstance(patch_size, int) and patch_size > 0:
            patch_sizes.append(patch_size)
        elif isinstance(patch_size, (tuple, list)) and patch_size:
            value = patch_size[0]
            if isinstance(value, int) and value > 0:
                patch_sizes.append(value)

        kernel_size = getattr(module, "kernel_size", None)
        if isinstance(kernel_size, int) and kernel_size > 0:
            patch_sizes.append(kernel_size)
        elif isinstance(kernel_size, (tuple, list)) and kernel_size:
            value = kernel_size[0]
            if isinstance(value, int) and value > 0:
                patch_sizes.append(value)

    for num_embeddings in position_embeddings:
        grid_size = int(round(math.sqrt(num_embeddings)))
        if grid_size * grid_size != num_embeddings:
            continue
        for patch_size in patch_sizes or [16]:
            image_size = grid_size * patch_size
            if image_size > 0:
                return (image_size, image_size)
    return None


def infer_patch_size(model) -> int | None:
    config_candidates = [
        getattr(model, "config", None),
        getattr(getattr(model, "config", None), "vlm_config", None),
        getattr(getattr(getattr(model, "config", None), "vlm_config", None), "vision_config", None),
        getattr(getattr(model, "config", None), "vision_config", None),
        getattr(getattr(model, "vision_tower", None), "config", None),
        getattr(getattr(model, "vision_model", None), "config", None),
    ]

    for config in config_candidates:
        if config is None:
            continue
        patch_size = getattr(config, "patch_size", None)
        if isinstance(patch_size, int) and patch_size > 0:
            return patch_size
        if isinstance(patch_size, (tuple, list)) and patch_size:
            value = patch_size[0]
            if isinstance(value, int) and value > 0:
                return value

    for _, module in model.named_modules():
        patch_size = getattr(module, "patch_size", None)
        if isinstance(patch_size, int) and patch_size > 0:
            return patch_size
        if isinstance(patch_size, (tuple, list)) and patch_size:
            value = patch_size[0]
            if isinstance(value, int) and value > 0:
                return value
        kernel_size = getattr(module, "kernel_size", None)
        if isinstance(kernel_size, int) and kernel_size > 0:
            return kernel_size
        if isinstance(kernel_size, (tuple, list)) and kernel_size:
            value = kernel_size[0]
            if isinstance(value, int) and value > 0:
                return value
    return None


def infer_pixel_shuffle_factor(model) -> int | None:
    config_candidates = [
        getattr(model, "config", None),
        getattr(getattr(model, "config", None), "vlm_config", None),
        getattr(getattr(getattr(model, "config", None), "vlm_config", None), "vision_config", None),
        getattr(getattr(model, "config", None), "vision_config", None),
    ]

    for config in config_candidates:
        if config is None:
            continue
        for attr in ("pixel_shuffle_factor", "vision_pixel_shuffle_factor"):
            value = getattr(config, attr, None)
            if isinstance(value, int) and value > 0:
                return value

    for _, module in model.named_modules():
        for attr in ("pixel_shuffle_factor", "vision_pixel_shuffle_factor"):
            value = getattr(module, attr, None)
            if isinstance(value, int) and value > 0:
                return value
    return None


@torch.no_grad()
def encode_images(model, processor, doc_rows, batch_size=2, device: str | None = None):
    runtime_device = device or DEVICE
    encoded = []
    for batch in tqdm(list(_batched(doc_rows, batch_size)), desc="Encoding images"):
        images = [Image.open(row["image_path"]).convert("RGB") for row in batch]
        inputs = process_image_batch(processor, images)
        inputs = {k: v.to(runtime_device) for k, v in inputs.items()}
        outputs = run_model_for_embeddings(model, inputs, is_image_batch=True)
        for i, row in enumerate(batch):
            emb = _trim_embedding_rows(outputs[i], None).detach().cpu()
            encoded.append((row["doc_id"], emb))
        for img in images:
            img.close()
    return encoded


def compute_rankings_with_processor(processor, query_embeddings, doc_embeddings, score_batch_size=64):
    query_ids = [qid for qid, _ in query_embeddings]
    q_embs = [emb for _, emb in query_embeddings]
    doc_ids = [did for did, _ in doc_embeddings]
    d_embs = [emb for _, emb in doc_embeddings]
    scores = score_retrieval_with_processor(processor, q_embs, d_embs, score_batch_size=score_batch_size)
    rankings = {}
    for i, query_id in enumerate(query_ids):
        row_scores = scores[i]
        order = torch.argsort(row_scores, descending=True).tolist()
        rankings[query_id] = [(doc_ids[j], float(row_scores[j].item())) for j in order]
    return rankings


def compute_retrieval_metrics(rankings, query_rows, ks=(1, 5, 10), num_candidate_docs: int | None = None):
    total = len(query_rows)
    recall_hits = {k: 0 for k in ks}
    mrr_sum = 0.0
    for row in query_rows:
        query_id = row["query_id"]
        gold_doc_id = row["doc_id"]
        ranked_doc_ids = [doc_id for doc_id, _ in rankings[query_id]]
        for k in ks:
            if gold_doc_id in ranked_doc_ids[:k]:
                recall_hits[k] += 1
        rr = 0.0
        for rank, doc_id in enumerate(ranked_doc_ids, start=1):
            if doc_id == gold_doc_id:
                rr = 1.0 / rank
                break
        mrr_sum += rr
    metrics = {f"Recall@{k}": recall_hits[k] / total for k in ks}
    metrics["MRR"] = mrr_sum / total
    metrics["num_queries"] = total
    metrics["num_candidate_docs"] = num_candidate_docs if num_candidate_docs is not None else len({row["doc_id"] for row in query_rows})
    return metrics


def evaluate_manifest_with_colmodernvbert(
    manifest_path,
    model_name=DEFAULT_RETRIEVAL_MODEL_NAME,
    adapter_dir: str | None = None,
    candidate_manifest_path: str | Path | None = None,
    limit_queries: int | None = None,
    limit_docs: int | None = None,
    query_batch_size: int = 8,
    image_batch_size: int = 2,
    score_batch_size: int = 64,
    output_path: str | Path | None = None,
    run_name: str | None = None,
    device: str | None = None,
):
    query_rows, doc_rows = build_eval_sets_from_manifest(
        manifest_path,
        candidate_manifest_path=candidate_manifest_path,
        limit_queries=limit_queries,
        limit_docs=limit_docs,
    )
    runtime = load_retriever(
        model_name=model_name,
        adapter_dir=adapter_dir,
        device=device,
    )
    model, processor = runtime.model, runtime.processor
    query_embeddings = encode_queries(
        model,
        processor,
        query_rows,
        batch_size=query_batch_size,
        device=device,
    )
    doc_embeddings = encode_images(
        model,
        processor,
        doc_rows,
        batch_size=image_batch_size,
        device=device,
    )
    rankings = compute_rankings_with_processor(processor, query_embeddings, doc_embeddings, score_batch_size=score_batch_size)
    metrics = compute_retrieval_metrics(rankings, query_rows, num_candidate_docs=len(doc_rows))
    preview = []
    for row in query_rows[:10]:
        preview.append(
            {
                "query_id": row["query_id"],
                "query": row["query"],
                "gold_doc_id": row["doc_id"],
                "top_docs": rankings[row["query_id"]][:5],
            }
        )
    result = {
        "run_name": run_name,
        "manifest_path": str(manifest_path),
        "candidate_manifest_path": str(candidate_manifest_path) if candidate_manifest_path is not None else str(manifest_path),
        "adapter_dir": adapter_dir,
        "resolved_model_name": runtime.resolved_model_name,
        "device": runtime.device,
        "versions": runtime.versions,
        "metrics": metrics,
        "preview": preview,
    }
    if output_path is not None:
        export_json(output_path, result)
    return result


def save_results_table(result_rows: list[dict[str, Any]], output_prefix: str | Path):
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    export_json(output_prefix.with_suffix(".json"), result_rows)
    if not result_rows:
        return
    keys = sorted({key for row in result_rows for key in row.keys()})
    with open(output_prefix.with_suffix(".csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(result_rows)


def plot_results_table(
    result_rows: list[dict[str, Any]],
    output_path: str | Path | None = None,
    metric_names: tuple[str, ...] = ("Recall@1", "Recall@5", "Recall@10", "MRR"),
    title: str = "Experiment Comparison",
):
    if not result_rows:
        return None
    run_names = [row["run_name"] for row in result_rows]
    x = np.arange(len(run_names))
    width = 0.18 if len(metric_names) >= 4 else 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(run_names) * 1.2), 6))
    offsets = np.linspace(-width * (len(metric_names) - 1) / 2, width * (len(metric_names) - 1) / 2, len(metric_names))

    for metric_idx, metric_name in enumerate(metric_names):
        values = [row.get(metric_name, 0.0) for row in result_rows]
        ax.bar(x + offsets[metric_idx], values, width=width, label=metric_name)

    ax.set_title(title)
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(run_names, rotation=35, ha="right")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig


def export_thesis_tables(
    result_rows: list[dict[str, Any]],
    output_dir: str | Path,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    formatted_rows = format_results_for_thesis(result_rows)
    main_table = []
    transfer_table = []
    for row in formatted_rows:
        target = transfer_table if row["run_name"].startswith("manual_real_ru") else main_table
        target.append(
            {
                "condition": row.get("condition_label", row["run_name"]),
                "run_name": row["run_name"],
                "Recall@1": row.get("Recall@1"),
                "Recall@5": row.get("Recall@5"),
                "Recall@10": row.get("Recall@10"),
                "MRR": row.get("MRR"),
                "num_queries": row.get("num_queries"),
                "num_candidate_docs": row.get("num_candidate_docs"),
            }
        )

    save_results_table(main_table, output_dir / "thesis_main_table")
    save_results_table(transfer_table, output_dir / "thesis_transfer_table")
    plot_results_table(main_table, output_dir / "thesis_main_plot.png", title="Source Data Evaluation")
    if transfer_table:
        plot_results_table(transfer_table, output_dir / "thesis_transfer_plot.png", title="Manual Russian Transfer")
    return {
        "main_table_json": str((output_dir / "thesis_main_table.json")),
        "main_table_csv": str((output_dir / "thesis_main_table.csv")),
        "main_plot": str((output_dir / "thesis_main_plot.png")),
        "transfer_table_json": str((output_dir / "thesis_transfer_table.json")),
        "transfer_table_csv": str((output_dir / "thesis_transfer_table.csv")),
        "transfer_plot": str((output_dir / "thesis_transfer_plot.png")),
    }


def thesis_label_map() -> dict[str, str]:
    return {
        "english_reference": "EN query + EN image (zero-shot)",
        "russian_baseline_zero_shot": "RU query + EN image (zero-shot)",
        "russian_baseline_lora": "RU query + EN image (+ LoRA)",
        "synthetic_zero_shot": "RU query + synthetic RU image (zero-shot)",
        "synthetic_lora": "RU query + synthetic RU image (+ LoRA)",
        "synthetic_attempted_zero_shot": "RU query + synthetic-attempted image (zero-shot)",
        "synthetic_attempted_lora": "RU query + synthetic-attempted image (+ LoRA)",
        "baseline_aligned_to_synthetic_attempted_zero_shot": "RU query + EN image, aligned to synthetic-attempted subset (zero-shot)",
        "synthetic_attempted_aligned_zero_shot": "RU query + synthetic-attempted image, aligned subset (zero-shot)",
        "baseline_aligned_to_synthetic_primary_zero_shot": "RU query + EN image, aligned to synthetic-primary subset (zero-shot)",
        "synthetic_primary_aligned_zero_shot": "RU query + synthetic RU image, aligned subset (zero-shot)",
        "baseline_aligned_to_synthetic_primary_lora": "RU query + EN image, aligned to synthetic-primary subset (+ LoRA)",
        "synthetic_primary_aligned_lora": "RU query + synthetic RU image, aligned subset (+ LoRA)",
        "manual_real_ru_zero_shot": "Manual real RU set (zero-shot)",
        "manual_real_ru_baseline_lora": "Manual real RU set with baseline LoRA",
        "manual_real_ru_synthetic_lora": "Manual real RU set with synthetic LoRA",
    }


def format_results_for_thesis(
    result_rows: list[dict[str, Any]],
    decimals: int = 4,
):
    labels = thesis_label_map()
    formatted = []
    for row in result_rows:
        out = dict(row)
        out["condition_label"] = labels.get(row["run_name"], row["run_name"])
        for key in ("Recall@1", "Recall@5", "Recall@10", "MRR"):
            if key in out and out[key] is not None:
                out[key] = round(float(out[key]), decimals)
        formatted.append(out)
    return formatted


def thesis_table_columns() -> list[str]:
    return [
        "condition_label",
        "run_name",
        "Recall@1",
        "Recall@5",
        "Recall@10",
        "MRR",
        "num_queries",
        "num_candidate_docs",
    ]


def read_manifest_for_training(manifest_path, one_query_per_doc: bool = True):
    rows = read_jsonl(manifest_path)
    if one_query_per_doc:
        deduped = {}
        for row in rows:
            deduped.setdefault(row["doc_id"], row)
        rows = list(deduped.values())
    return rows


class ManifestPairDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        return {
            "query_id": row["query_id"],
            "doc_id": row["doc_id"],
            "query": row["query"],
            "image_path": row["image_path"],
        }


def collate_pair_batch(examples, processor):
    queries = [ex["query"] for ex in examples]
    images = [Image.open(ex["image_path"]).convert("RGB") for ex in examples]
    query_inputs = process_query_batch(processor, queries)
    image_inputs = process_image_batch(processor, images)
    for img in images:
        img.close()
    return {
        "query_ids": [ex["query_id"] for ex in examples],
        "doc_ids": [ex["doc_id"] for ex in examples],
        "query_inputs": dict(query_inputs),
        "image_inputs": dict(image_inputs),
    }


def move_to_device(obj, device):
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if torch.is_tensor(obj):
        return obj.to(device)
    return obj


def suggest_lora_target_modules(model):
    linear_names = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_names.append(name)
    preferred_suffixes = ["Wqkv", "Wo", "embedding_proj_layer", "q_proj", "k_proj", "v_proj", "o_proj"]
    picked = []
    for suffix in preferred_suffixes:
        if any(name.endswith(suffix) for name in linear_names):
            picked.append(suffix)
    if not picked:
        picked = sorted({name.split(".")[-1] for name in linear_names[:12]})
    return picked


def build_lora_retriever(
    model_name=DEFAULT_RETRIEVAL_MODEL_NAME,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=None,
):
    runtime = load_retriever(model_name=model_name, device=DEVICE)
    processor = runtime.processor
    base_model = runtime.model
    if target_modules is None:
        target_modules = suggest_lora_target_modules(base_model)
    if LoraConfig is None or get_peft_model is None:
        raise ImportError("peft is required for legacy LoRA training. v2 query_adapter does not require PEFT.")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(base_model, lora_config)
    return model, processor, target_modules


def contrastive_retrieval_loss(model, processor, batch):
    q_inputs = batch["query_inputs"]
    d_inputs = batch["image_inputs"]
    q_outputs = run_model_for_embeddings(model, q_inputs, is_image_batch=False)
    d_outputs = run_model_for_embeddings(model, d_inputs, is_image_batch=True)
    q_mask = q_inputs.get("attention_mask")
    q_embs = [
        _trim_embedding_rows(q_outputs[i], q_mask[i] if q_mask is not None else None)
        for i in range(q_outputs.size(0))
    ]
    d_embs = [_trim_embedding_rows(d_outputs[i], None) for i in range(d_outputs.size(0))]
    scores = processor.score_retrieval(
        query_embeddings=q_embs,
        passage_embeddings=d_embs,
        batch_size=max(16, len(q_embs)),
        output_device=q_outputs.device,
    )
    targets = torch.arange(scores.size(0), device=scores.device)
    loss_q2d = F.cross_entropy(scores, targets)
    loss_d2q = F.cross_entropy(scores.t(), targets)
    return 0.5 * (loss_q2d + loss_d2q)


def train_lora_retriever(
    train_manifest,
    output_dir,
    model_name=DEFAULT_RETRIEVAL_MODEL_NAME,
    val_manifest: str | None = None,
    num_epochs=1,
    batch_size=4,
    grad_accum_steps=1,
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_ratio=0.05,
    max_train_rows=None,
    one_query_per_doc=True,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=None,
    seed=42,
    val_limit_queries: int | None = None,
):
    random.seed(seed)
    torch.manual_seed(seed)

    rows = read_manifest_for_training(train_manifest, one_query_per_doc=one_query_per_doc)
    if not rows:
        raise ValueError(f"No training rows found in manifest: {train_manifest}")
    if max_train_rows is not None:
        rows = rows[:max_train_rows]
    if not rows:
        raise ValueError(
            f"No training rows remain after applying max_train_rows={max_train_rows} "
            f"to manifest: {train_manifest}"
        )
    dataset = ManifestPairDataset(rows)
    model, processor, target_modules = build_lora_retriever(
        model_name=model_name,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )
    collate_fn = lambda examples: collate_pair_batch(examples, processor)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    total_steps = max(1, num_epochs * math.ceil(len(train_loader) / max(1, grad_accum_steps)))
    warmup_steps = int(total_steps * warmup_ratio)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    if Accelerator is None:
        raise ImportError("accelerate is required for legacy LoRA training. Install accelerate before calling train_lora_retriever().")
    mixed_precision = "fp16" if torch.cuda.is_available() else "no"
    accelerator = Accelerator(gradient_accumulation_steps=grad_accum_steps, mixed_precision=mixed_precision)
    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(model, optimizer, train_loader, lr_scheduler)

    history = []
    best_val_mrr = float("-inf")
    best_epoch = None
    best_adapter_dir = None
    best_model_dir = None
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        completed_steps = 0
        progress_bar = tqdm(train_loader, desc=f"Train epoch {epoch + 1}/{num_epochs}")
        for batch in progress_bar:
            batch["query_inputs"] = move_to_device(batch["query_inputs"], accelerator.device)
            batch["image_inputs"] = move_to_device(batch["image_inputs"], accelerator.device)
            with accelerator.accumulate(model):
                loss = contrastive_retrieval_loss(model, processor, batch)
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                completed_steps += 1
                running_loss += loss.detach().float().item()
                progress_bar.set_postfix({"loss": round(running_loss / completed_steps, 4)})

        accelerator.wait_for_everyone()
        unwrapped = accelerator.unwrap_model(model)
        epoch_dir = Path(output_dir) / f"epoch_{epoch + 1}"
        merged_epoch_dir = Path(output_dir) / f"epoch_{epoch + 1}_merged"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        unwrapped.save_pretrained(epoch_dir)
        processor.save_pretrained(epoch_dir)
        merged_model_dir = export_merged_retriever_checkpoint(
            model_name=model_name,
            adapter_dir=epoch_dir,
            output_dir=merged_epoch_dir,
            processor=processor,
        )

        epoch_summary = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(1, completed_steps),
            "adapter_dir": str(epoch_dir),
            "model_dir": merged_model_dir,
        }

        if val_manifest is not None:
            val_eval = evaluate_manifest_with_colmodernvbert(
                manifest_path=val_manifest,
                model_name=merged_model_dir,
                limit_queries=val_limit_queries,
                run_name=f"val_epoch_{epoch + 1}",
            )
            epoch_summary["val_metrics"] = val_eval["metrics"]
            val_mrr = val_eval["metrics"]["MRR"]
            if val_mrr > best_val_mrr:
                best_val_mrr = val_mrr
                best_epoch = epoch + 1
                best_adapter_dir = str(epoch_dir)
                best_model_dir = merged_model_dir
        else:
            if epoch == num_epochs - 1:
                best_epoch = epoch + 1
                best_adapter_dir = str(epoch_dir)
                best_model_dir = merged_model_dir

        history.append(epoch_summary)

    training_summary = {
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest) if val_manifest is not None else None,
        "target_modules": target_modules,
        "num_rows": len(rows),
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "best_epoch": best_epoch,
        "best_adapter_dir": best_adapter_dir,
        "best_model_dir": best_model_dir,
        "history": history,
    }
    export_json(Path(output_dir) / "training_summary.json", training_summary)
    return training_summary


def evaluate_experiment_suite(
    manifest_groups: dict[str, dict[str, str]],
    output_dir: str | Path,
    model_name=DEFAULT_RETRIEVAL_MODEL_NAME,
    baseline_model_dir: str | None = None,
    synthetic_model_dir: str | None = None,
    manual_manifest_path: str | None = None,
    aligned_test_manifests: dict[str, dict[str, Any]] | None = None,
    device: str | None = None,
    baseline_adapter_dir: str | None = None,
    synthetic_adapter_dir: str | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_rows = []

    if baseline_model_dir is None and baseline_adapter_dir is not None and is_local_colmodernvbert_checkpoint(baseline_adapter_dir):
        baseline_model_dir = baseline_adapter_dir
    if synthetic_model_dir is None and synthetic_adapter_dir is not None and is_local_colmodernvbert_checkpoint(synthetic_adapter_dir):
        synthetic_model_dir = synthetic_adapter_dir

    evaluations = [
        ("english_reference", manifest_groups["english_reference"]["test"], model_name),
        ("russian_baseline_zero_shot", manifest_groups["russian_baseline"]["test"], model_name),
        ("russian_baseline_lora", manifest_groups["russian_baseline"]["test"], baseline_model_dir),
        ("synthetic_zero_shot", manifest_groups["synthetic_primary"]["test"], model_name),
        ("synthetic_lora", manifest_groups["synthetic_primary"]["test"], synthetic_model_dir),
        ("synthetic_attempted_zero_shot", manifest_groups["synthetic_attempted"]["test"], model_name),
        ("synthetic_attempted_lora", manifest_groups["synthetic_attempted"]["test"], synthetic_model_dir),
    ]

    for run_name, manifest_path, run_model_name in evaluations:
        if manifest_path is None:
            continue
        if "lora" in run_name and run_model_name is None:
            continue
        eval_result = evaluate_manifest_with_colmodernvbert(
            manifest_path=manifest_path,
            model_name=run_model_name,
            output_path=output_dir / f"{run_name}.json",
            run_name=run_name,
            device=device,
        )
        row = {"run_name": run_name, **eval_result["metrics"]}
        result_rows.append(row)

    if aligned_test_manifests:
        aligned_runs = [
            (
                "baseline_aligned_to_synthetic_attempted_zero_shot",
                aligned_test_manifests["synthetic_attempted"]["baseline_manifest"],
                model_name,
            ),
            (
                "synthetic_attempted_aligned_zero_shot",
                aligned_test_manifests["synthetic_attempted"]["synthetic_manifest"],
                model_name,
            ),
            (
                "baseline_aligned_to_synthetic_primary_zero_shot",
                aligned_test_manifests["synthetic_primary"]["baseline_manifest"],
                model_name,
            ),
            (
                "synthetic_primary_aligned_zero_shot",
                aligned_test_manifests["synthetic_primary"]["synthetic_manifest"],
                model_name,
            ),
            (
                "baseline_aligned_to_synthetic_primary_lora",
                aligned_test_manifests["synthetic_primary"]["baseline_manifest"],
                baseline_model_dir,
            ),
            (
                "synthetic_primary_aligned_lora",
                aligned_test_manifests["synthetic_primary"]["synthetic_manifest"],
                synthetic_model_dir,
            ),
        ]
        for run_name, manifest_path, run_model_name in aligned_runs:
            if "lora" in run_name and run_model_name is None:
                continue
            eval_result = evaluate_manifest_with_colmodernvbert(
                manifest_path=manifest_path,
                model_name=run_model_name,
                output_path=output_dir / f"{run_name}.json",
                run_name=run_name,
                device=device,
            )
            row = {"run_name": run_name, **eval_result["metrics"]}
            result_rows.append(row)

    if manual_manifest_path:
        manual_runs = [
            ("manual_real_ru_zero_shot", model_name),
            ("manual_real_ru_baseline_lora", baseline_model_dir),
            ("manual_real_ru_synthetic_lora", synthetic_model_dir),
        ]
        for run_name, run_model_name in manual_runs:
            if "lora" in run_name and run_model_name is None:
                continue
            eval_result = evaluate_manifest_with_colmodernvbert(
                manifest_path=manual_manifest_path,
                model_name=run_model_name,
                output_path=output_dir / f"{run_name}.json",
                run_name=run_name,
                device=device,
            )
            row = {"run_name": run_name, **eval_result["metrics"]}
            result_rows.append(row)

    save_results_table(result_rows, output_dir / "experiment_results")
    return result_rows


def prepare_thesis_experiment(
    output_dir: str | Path | None = None,
    max_rows: int = 300,
    synthetic_subset_size: int | dict[str, int | None] | None = 100,
    translation_model: str = "Helsinki-NLP/opus-mt-en-ru",
    font_path: str | None = None,
    seed: int = 42,
    streaming: bool = True,
    translator_device: str = "cpu",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    synthetic_splits: tuple[str, ...] = ("train", "val", "test"),
    ocr_backend: str = "paddleocr",
    ocr_langs: tuple[str, ...] = ("en",),
    ocr_use_angle_cls: bool = True,
    preprocess_for_ocr: bool = True,
    strict_ocr: bool = True,
):
    output_dir = Path(output_dir or default_workdir("modernvbert_ru"))
    output_dir.mkdir(parents=True, exist_ok=True)

    records, rows = load_colpali_records(max_rows=max_rows, seed=seed, streaming=streaming)
    save_original_images(rows, records, output_dir / "original_images")

    split_map, split_artifact = create_split_map(
        records,
        output_dir / "artifacts" / "doc_splits.json",
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    apply_split_map(records, split_map, seed=seed)

    del rows
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    translator = MarianTranslator(
        model_name=translation_model,
        device=translator_device,
        max_new_tokens=64,
        max_input_length=256,
    )
    translate_records(records, translator)
    export_jsonl([asdict(r) for r in records], output_dir / "artifacts" / "source_records_full.jsonl")

    english_reference = export_split_manifests(
        records,
        output_dir / "manifests",
        manifest_prefix="english_reference",
        query_lang="en",
        image_variant="original_en",
        source_lang_query="en",
        target_lang_query="en",
    )
    russian_baseline = export_split_manifests(
        records,
        output_dir / "manifests",
        manifest_prefix="russian_baseline",
        query_lang="ru",
        image_variant="original_en",
        source_lang_query="en",
        target_lang_query="ru",
    )

    reader = get_ocr_reader(
        list(ocr_langs),
        backend=ocr_backend,
        use_angle_cls=ocr_use_angle_cls,
        strict=strict_ocr,
    )
    subset_sizes = normalize_synthetic_subset_sizes(synthetic_subset_size, synthetic_splits)
    synthetic_records = []
    synthetic_stats = {}
    for split in synthetic_splits:
        split_records, split_stats = build_synthetic_subset_ocr(
            records=records,
            translator=translator,
            out_dir=output_dir / "synthetic_ru_images",
            split=split,
            subset_size=subset_sizes[split],
            font_path=font_path,
            seed=seed,
            reader=reader,
            preprocess_for_ocr=preprocess_for_ocr,
        )
        synthetic_records.extend(split_records)
        synthetic_stats[split] = split_stats

    gallery_rows = build_synthetic_gallery(
        [r for r in synthetic_records if r.ocr_boxes_translated > 0],
        output_dir / "artifacts" / "synthetic_gallery",
        max_items=12,
    )
    export_json(output_dir / "artifacts" / "synthetic_stats.json", synthetic_stats)
    export_jsonl([asdict(r) for r in synthetic_records], output_dir / "artifacts" / "synthetic_records_full.jsonl")
    export_json(output_dir / "artifacts" / "synthetic_gallery.json", gallery_rows)

    synthetic_attempted = export_split_manifests(
        synthetic_records,
        output_dir / "manifests",
        manifest_prefix="synthetic_attempted",
        query_lang="ru",
        image_variant="synthetic_ru",
        include_fallback=True,
        source_lang_query="en",
        target_lang_query="ru",
    )
    synthetic_primary = export_split_manifests(
        synthetic_records,
        output_dir / "manifests",
        manifest_prefix="synthetic_primary",
        query_lang="ru",
        image_variant="synthetic_ru",
        include_fallback=False,
        source_lang_query="en",
        target_lang_query="ru",
    )
    aligned_test_manifests = build_aligned_test_manifests(
        baseline_test_manifest=russian_baseline["test"],
        synthetic_attempted_test_manifest=synthetic_attempted["test"],
        synthetic_primary_test_manifest=synthetic_primary["test"],
        output_dir=output_dir / "manifests" / "aligned",
    )

    export_manual_russian_template(output_dir / "manual_russian_set_template.jsonl")

    validate_split_integrity(english_reference)
    validate_split_integrity(russian_baseline)

    summary = {
        "output_dir": str(output_dir),
        "split_artifact": str(output_dir / "artifacts" / "doc_splits.json"),
        "manifests": {
            "english_reference": english_reference,
            "russian_baseline": russian_baseline,
            "synthetic_attempted": synthetic_attempted,
            "synthetic_primary": synthetic_primary,
        },
        "aligned_test_manifests": aligned_test_manifests,
        "synthetic_stats": synthetic_stats,
        "ocr_backend": ocr_backend,
        "ocr_langs": list(ocr_langs),
        "strict_ocr": strict_ocr,
        "preprocess_for_ocr": preprocess_for_ocr,
        "split_counts": split_artifact["counts"],
        "manual_russian_template": str(output_dir / "manual_russian_set_template.jsonl"),
    }
    export_json(output_dir / "artifacts" / "prepare_summary.json", summary)
    return summary


def run_tiny_pipeline_sanity_check(
    output_dir: str | Path | None = None,
    experiment_overrides: dict[str, Any] | None = None,
    model_name: str = DEFAULT_RETRIEVAL_MODEL_NAME,
    eval_limit_queries: int = 4,
    eval_limit_docs: int = 4,
    run_training: bool = False,
    eval_device: str = "cpu",
):
    output_dir = Path(output_dir or default_workdir("modernvbert_ru")) / "sanity_check"
    output_dir.mkdir(parents=True, exist_ok=True)

    sanity_config = {
        "output_dir": output_dir,
        "max_rows": 12,
        "synthetic_subset_size": {"train": 4, "val": 2, "test": 2},
        "translation_model": "Helsinki-NLP/opus-mt-en-ru",
        "font_path": None,
        "seed": 42,
        "streaming": True,
        "translator_device": "cpu",
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "synthetic_splits": ("train", "val", "test"),
        "ocr_backend": "paddleocr",
        "ocr_langs": ("en",),
        "ocr_use_angle_cls": True,
        "preprocess_for_ocr": True,
        "strict_ocr": True,
    }
    if experiment_overrides:
        sanity_config.update(experiment_overrides)
    sanity_config["output_dir"] = Path(sanity_config["output_dir"])
    preflight = run_retrieval_preflight(
        model_name=model_name,
        device=eval_device,
    )

    summary = prepare_thesis_experiment(**sanity_config)
    evals = {}

    for run_name, manifest_path in [
        ("english_reference_sanity", summary["manifests"]["english_reference"]["test"]),
        ("russian_baseline_sanity", summary["manifests"]["russian_baseline"]["test"]),
    ]:
        evals[run_name] = evaluate_manifest_with_colmodernvbert(
            manifest_path=manifest_path,
            model_name=model_name,
            limit_queries=eval_limit_queries,
            limit_docs=eval_limit_docs,
            run_name=run_name,
            output_path=sanity_config["output_dir"] / "results" / f"{run_name}.json",
            device=eval_device,
        )

    synthetic_primary_test = summary["manifests"]["synthetic_primary"]["test"]
    synthetic_primary_rows = read_jsonl(synthetic_primary_test)
    if synthetic_primary_rows:
        evals["synthetic_primary_sanity"] = evaluate_manifest_with_colmodernvbert(
            manifest_path=synthetic_primary_test,
            model_name=model_name,
            limit_queries=min(eval_limit_queries, len(synthetic_primary_rows)),
            limit_docs=min(eval_limit_docs, len({row["doc_id"] for row in synthetic_primary_rows})),
            run_name="synthetic_primary_sanity",
            output_path=sanity_config["output_dir"] / "results" / "synthetic_primary_sanity.json",
            device=eval_device,
        )

    training_summary = None
    if run_training:
        training_summary = train_lora_retriever(
            train_manifest=summary["manifests"]["russian_baseline"]["train"],
            val_manifest=summary["manifests"]["russian_baseline"]["val"],
            output_dir=sanity_config["output_dir"] / "adapters" / "baseline_sanity_lora",
            model_name=model_name,
            num_epochs=1,
            batch_size=2,
            grad_accum_steps=1,
            learning_rate=1e-4,
            max_train_rows=4,
            one_query_per_doc=True,
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            seed=sanity_config["seed"],
            val_limit_queries=eval_limit_queries,
        )

    result = {
        "config": {**sanity_config, "output_dir": str(sanity_config["output_dir"])},
        "prepare_summary_path": str(sanity_config["output_dir"] / "artifacts" / "prepare_summary.json"),
        "eval_device": eval_device,
        "retrieval_preflight": preflight,
        "synthetic_stats": summary["synthetic_stats"],
        "manifests": summary["manifests"],
        "evaluations": {name: payload["metrics"] for name, payload in evals.items()},
        "training": training_summary,
    }
    export_json(sanity_config["output_dir"] / "artifacts" / "sanity_check_summary.json", result)
    return result


# ---- V2 thesis extensions ----


import csv
import importlib
import importlib.metadata as importlib_metadata
import io
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_RETRIEVAL_MODEL_NAME = "ModernVBERT/colmodernvbert-merged"
DEFAULT_K_VALUES = (1, 5, 10)
V2_VERSION = "2026-04-24"


def _legacy():
    import sys
    return sys.modules[__name__]


def export_json(path: str | Path, obj: Any):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def export_jsonl(rows: Iterable[dict[str, Any]], output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_results_table(result_rows: list[dict[str, Any]], output_prefix: str | Path):
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    export_json(output_prefix.with_suffix(".json"), result_rows)
    if not result_rows:
        return
    keys = sorted({key for row in result_rows for key in row.keys()})
    with open(output_prefix.with_suffix(".csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(result_rows)


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def build_kaggle_v2_install_command(include_lora: bool = False) -> str:
    packages = [
        '"pillow==11.3.0"',
        '"PyYAML==6.0.2"',
        '"datasets==2.21.0"',
        "git+https://github.com/huggingface/transformers.git",
        '"colpali-engine==0.3.15"',
        '"accelerate==0.34.2"',
        '"sentencepiece"',
        '"sacremoses"',
        '"tqdm"',
        '"paddleocr"',
        '"paddlepaddle"',
        '"easyocr"',
    ]
    if include_lora:
        packages.append('"peft==0.18.0"')
    return "!pip -q install -U --upgrade-strategy only-if-needed " + " ".join(packages)


def environment_preflight(
    model_name: str = DEFAULT_RETRIEVAL_MODEL_NAME,
    device: str = "cpu",
    run_model_check: bool = False,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "v2_version": V2_VERSION,
        "model_name": model_name,
        "device": device,
        "versions": {
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "colpali-engine": _package_version("colpali-engine"),
            "peft": _package_version("peft"),
            "accelerate": _package_version("accelerate"),
            "datasets": _package_version("datasets"),
        },
        "import_errors_at_module_load": {
            "transformers_colmodernvbert": TRANSFORMERS_COLMODERNVBERT_IMPORT_ERROR,
            "colpali_modernvbert": COLPALI_MODERNVBERT_IMPORT_ERROR,
        },
        "imports": {},
        "modes": {
            "zero_shot": {"enabled": True, "reason": "No training required."},
            "query_adapter": {"enabled": True, "reason": "Uses frozen retriever embeddings and a small trainable adapter."},
            "dual_adapter": {"enabled": True, "reason": "Uses frozen retriever embeddings and small query/image adapters."},
            "transformers_lora_experimental": {"enabled": False, "reason": "Not checked yet."},
        },
    }

    for module_name in ("torch", "transformers", "colpali_engine", "peft", "accelerate", "datasets", "PIL"):
        try:
            importlib.import_module(module_name)
            report["imports"][module_name] = {"ok": True}
        except Exception as exc:
            report["imports"][module_name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    try:
        transformers = importlib.import_module("transformers")
        has_tf_model = hasattr(transformers, "ColModernVBertForRetrieval") and hasattr(transformers, "ColModernVBertProcessor")
        peft_ok = report["imports"].get("peft", {}).get("ok", False)
        report["modes"]["transformers_lora_experimental"] = {
            "enabled": bool(has_tf_model and peft_ok),
            "reason": (
                "Transformers ColModernVBertForRetrieval and PEFT are importable."
                if has_tf_model and peft_ok
                else "Skipped: requires Transformers ColModernVBertForRetrieval plus PEFT; never uses colpali_engine adapter_dir."
            ),
        }
    except Exception as exc:
        report["modes"]["transformers_lora_experimental"] = {
            "enabled": False,
            "reason": f"Skipped: Transformers inspection failed: {type(exc).__name__}: {exc}",
        }

    if run_model_check:
        try:
            report["retrieval_preflight"] = _legacy().run_retrieval_preflight(model_name=model_name, device=device)
        except Exception as exc:
            report["retrieval_preflight"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    if output_path is not None:
        export_json(output_path, report)
    return report


def assert_v2_environment_ready(
    require_ocr: bool = False,
    require_retriever: bool = True,
    model_name: str = DEFAULT_RETRIEVAL_MODEL_NAME,
    device: str = "cpu",
) -> dict[str, Any]:
    report = environment_preflight(model_name=model_name, device=device, run_model_check=require_retriever)
    problems = []

    if require_ocr:
        problems.append(
            "This retrieval module intentionally does not import OCR libraries. "
            "Run PaddleOCR preparation in 01_prepare_synthetic_dataset_paddleocr.ipynb."
        )

    if require_retriever:
        retrieval = report.get("retrieval_preflight")
        if isinstance(retrieval, dict) and retrieval.get("ok") is False:
            problems.append(f"ModernVBERT retrieval preflight failed: {retrieval.get('error')}")

    if problems:
        raise RuntimeError(
            "Environment is not ready for the v2 thesis pipeline.\n\n"
            + "\n\n".join(f"{idx + 1}. {problem}" for idx, problem in enumerate(problems))
            + "\n\nAfter changing installs in Kaggle, restart the runtime and rerun the import cell."
        )
    return report


def prepare_thesis_experiment_v2(*args, **kwargs):
    return _legacy().prepare_thesis_experiment(*args, **kwargs)


def _rebase_artifact_path(value: str | Path | None, prepared_dir: str | Path) -> str | None:
    if value is None:
        return None
    raw = str(value)
    if not raw:
        return raw
    path = Path(raw)
    if path.exists():
        return str(path)
    prepared_dir = Path(prepared_dir)
    parts = path.parts
    for anchor in ("original_images", "synthetic_ru_images", "manifests", "artifacts"):
        if anchor in parts:
            idx = parts.index(anchor)
            candidate = prepared_dir.joinpath(*parts[idx:])
            if candidate.exists():
                return str(candidate)
            return str(candidate)
    candidate = prepared_dir / path.name
    return str(candidate)


def _rewrite_manifest_paths(manifest_path: str | Path, prepared_dir: str | Path, output_dir: str | Path) -> tuple[str, dict[str, Any]]:
    rows = read_jsonl(manifest_path)
    missing = 0
    rewritten = []
    for row in rows:
        out = dict(row)
        for key in ("image_path", "original_image_path", "clean_image_path"):
            if key in out and out[key]:
                out[key] = _rebase_artifact_path(out[key], prepared_dir)
        image_path = out.get("image_path")
        if image_path and not Path(image_path).exists():
            missing += 1
        rewritten.append(out)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / Path(manifest_path).name
    export_jsonl(rewritten, output_path)
    return str(output_path), {"rows": len(rewritten), "missing_images": missing}


def _validate_manifest_split_leakage(manifest_paths: dict[str, str]) -> dict[str, Any]:
    split_docs = {}
    for split, path in manifest_paths.items():
        split_docs[split] = {row["doc_id"] for row in read_jsonl(path)}
    overlaps = {}
    keys = list(split_docs)
    for idx, left in enumerate(keys):
        for right in keys[idx + 1:]:
            overlap = sorted(split_docs[left] & split_docs[right])
            overlaps[f"{left}__{right}"] = overlap[:20]
    return {"ok": all(not values for values in overlaps.values()), "overlaps": overlaps}


def load_prepared_experiment_artifacts(
    prepared_dir: str | Path,
    work_dir: str | Path | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    prepared_dir = Path(prepared_dir)
    summary_path = prepared_dir / "artifacts" / "prepare_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing prepared artifact summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    work_dir = Path(work_dir or prepared_dir)
    rewritten_manifest_dir = work_dir / "manifests_rebased"

    manifest_report = {}
    rebased_manifest_groups = {}
    for group_name, split_paths in summary.get("manifests", {}).items():
        rebased_manifest_groups[group_name] = {}
        for split, old_path in split_paths.items():
            old_path_rebased = _rebase_artifact_path(old_path, prepared_dir)
            if old_path_rebased is None or not Path(old_path_rebased).exists():
                raise FileNotFoundError(f"Missing manifest for {group_name}/{split}: {old_path}")
            new_path, report = _rewrite_manifest_paths(
                old_path_rebased,
                prepared_dir=prepared_dir,
                output_dir=rewritten_manifest_dir / group_name,
            )
            rebased_manifest_groups[group_name][split] = new_path
            manifest_report[f"{group_name}/{split}"] = report

    rebased_aligned = {}
    for group_name, payload in (summary.get("aligned_test_manifests") or {}).items():
        rebased_aligned[group_name] = dict(payload)
        for key in ("baseline_manifest", "synthetic_manifest"):
            if payload.get(key):
                old_path = _rebase_artifact_path(payload[key], prepared_dir)
                if old_path and Path(old_path).exists():
                    new_path, report = _rewrite_manifest_paths(
                        old_path,
                        prepared_dir=prepared_dir,
                        output_dir=rewritten_manifest_dir / "aligned" / group_name,
                    )
                    rebased_aligned[group_name][key] = new_path
                    manifest_report[f"aligned/{group_name}/{key}"] = report

    validation = {
        "prepared_dir": str(prepared_dir),
        "summary_path": str(summary_path),
        "manifest_report": manifest_report,
        "split_leakage": {},
        "synthetic_primary_nonfallback_rows": None,
        "synthetic_fallback_rows": 0,
        "ocr_backend_usage": {},
    }
    if validate:
        for group_name, split_paths in rebased_manifest_groups.items():
            validation["split_leakage"][group_name] = _validate_manifest_split_leakage(split_paths)
        synthetic_primary_test = rebased_manifest_groups.get("synthetic_primary", {}).get("test")
        if synthetic_primary_test:
            rows = read_jsonl(synthetic_primary_test)
            validation["synthetic_primary_nonfallback_rows"] = len(rows)
            validation["synthetic_fallback_rows"] = sum(1 for row in rows if row.get("synthetic_fallback_used"))
        stats = summary.get("synthetic_stats") or {}
        for split_payload in stats.values():
            usage = split_payload.get("ocr_backend_usage", {}) if isinstance(split_payload, dict) else {}
            for backend, count in usage.items():
                validation["ocr_backend_usage"][backend] = validation["ocr_backend_usage"].get(backend, 0) + count
        missing_total = sum(item["missing_images"] for item in manifest_report.values())
        if missing_total:
            raise FileNotFoundError(f"Prepared artifact validation found {missing_total} missing manifest images.")
        bad_leakage = {
            name: report for name, report in validation["split_leakage"].items()
            if not report.get("ok")
        }
        if bad_leakage:
            raise ValueError(f"Prepared artifact validation found split leakage: {bad_leakage}")
        if validation["synthetic_primary_nonfallback_rows"] == 0:
            raise ValueError("Prepared artifact validation found no synthetic-primary rows.")

    loaded = dict(summary)
    loaded["prepared_dir"] = str(prepared_dir)
    loaded["manifests"] = rebased_manifest_groups
    loaded["aligned_test_manifests"] = rebased_aligned
    loaded["validation"] = validation
    return loaded


def _dcg(relevances: list[float]) -> float:
    return sum((2.0**rel - 1.0) / math.log2(idx + 2) for idx, rel in enumerate(relevances))


def compute_rich_retrieval_metrics(
    rankings: dict[str, list[tuple[str, float]]],
    query_rows: list[dict[str, Any]],
    ks: tuple[int, ...] = DEFAULT_K_VALUES,
    qrels: dict[str, dict[str, float]] | None = None,
    num_candidate_docs: int | None = None,
) -> dict[str, float | int]:
    if not query_rows:
        return {f"Recall@{k}": 0.0 for k in ks} | {"MRR": 0.0, "num_queries": 0, "num_candidate_docs": num_candidate_docs or 0}

    if qrels is None:
        qrels = {row["query_id"]: {row["doc_id"]: 1.0} for row in query_rows}

    totals = {f"Recall@{k}": 0.0 for k in ks}
    totals.update({f"MAP@{k}": 0.0 for k in ks})
    totals.update({f"nDCG@{k}": 0.0 for k in ks})
    mrr_sum = 0.0

    for row in query_rows:
        query_id = row["query_id"]
        relevant = qrels.get(query_id, {})
        ranked_doc_ids = [doc_id for doc_id, _ in rankings.get(query_id, [])]
        relevant_doc_ids = {doc_id for doc_id, score in relevant.items() if score > 0}
        if not relevant_doc_ids:
            continue

        rr = 0.0
        for rank, doc_id in enumerate(ranked_doc_ids, start=1):
            if doc_id in relevant_doc_ids:
                rr = 1.0 / rank
                break
        mrr_sum += rr

        for k in ks:
            top_k = ranked_doc_ids[:k]
            hit_count = len(set(top_k) & relevant_doc_ids)
            totals[f"Recall@{k}"] += hit_count / max(1, len(relevant_doc_ids))

            precision_hits = 0
            ap_sum = 0.0
            for rank, doc_id in enumerate(top_k, start=1):
                if doc_id in relevant_doc_ids:
                    precision_hits += 1
                    ap_sum += precision_hits / rank
            totals[f"MAP@{k}"] += ap_sum / max(1, min(len(relevant_doc_ids), k))

            gains = [float(relevant.get(doc_id, 0.0)) for doc_id in top_k]
            ideal_gains = sorted([float(v) for v in relevant.values() if v > 0], reverse=True)[:k]
            ideal = _dcg(ideal_gains)
            totals[f"nDCG@{k}"] += (_dcg(gains) / ideal) if ideal > 0 else 0.0

    n = len(query_rows)
    metrics = {key: value / n for key, value in totals.items()}
    metrics["MRR"] = mrr_sum / n
    metrics["num_queries"] = n
    metrics["num_candidate_docs"] = int(num_candidate_docs or 0)
    return metrics


def _batched(items: list[Any], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _torch():
    return importlib.import_module("torch")


def _pil_image():
    return importlib.import_module("PIL.Image")


def _pil_filter():
    return importlib.import_module("PIL.ImageFilter")


def _pil_enhance():
    return importlib.import_module("PIL.ImageEnhance")


class _Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.seconds = time.perf_counter() - self.start


def _tensor_bytes(tensor: Any) -> int:
    return int(tensor.nelement() * tensor.element_size())


def compute_index_stats(doc_embeddings: list[tuple[str, Any]]) -> dict[str, float | int]:
    if not doc_embeddings:
        return {
            "index_size_mb": 0.0,
            "index_size_bytes": 0,
            "avg_doc_token_vectors": 0.0,
            "max_doc_token_vectors": 0,
            "num_documents_indexed": 0,
        }
    total_bytes = sum(_tensor_bytes(emb) for _, emb in doc_embeddings)
    token_counts = [int(emb.shape[0]) for _, emb in doc_embeddings]
    return {
        "index_size_mb": total_bytes / (1024 * 1024),
        "index_size_bytes": total_bytes,
        "avg_doc_token_vectors": sum(token_counts) / len(token_counts),
        "max_doc_token_vectors": max(token_counts),
        "num_documents_indexed": len(doc_embeddings),
    }


def _adapter_config_path(adapter_dir: str | Path) -> Path:
    return Path(adapter_dir) / "adapter_config.json"


def load_embedding_adapter(adapter_dir: str | Path, device: str | None = None):
    torch = _torch()
    adapter_dir = Path(adapter_dir)
    config = json.loads(_adapter_config_path(adapter_dir).read_text(encoding="utf-8"))
    adapter = ResidualEmbeddingAdapter(
        embedding_dim=int(config["embedding_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config.get("dropout", 0.0)),
        residual_scale=float(config.get("residual_scale", 0.1)),
    )
    state = torch.load(adapter_dir / "adapter.pt", map_location=device or "cpu")
    adapter.load_state_dict(state)
    adapter.to(device or "cpu").eval()
    return adapter, config


def save_embedding_adapter(adapter: Any, adapter_dir: str | Path, config: dict[str, Any]):
    torch = _torch()
    adapter_dir = Path(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    torch.save(adapter.state_dict(), adapter_dir / "adapter.pt")
    export_json(adapter_dir / "adapter_config.json", config)
    return str(adapter_dir)


def _resolve_adapter_pair(query_adapter_dir: str | Path | None, image_adapter_dir: str | Path | None, device: str):
    query_adapter = None
    image_adapter = None
    query_config = None
    image_config = None
    if query_adapter_dir is not None:
        query_adapter, query_config = load_embedding_adapter(query_adapter_dir, device=device)
    if image_adapter_dir is not None:
        image_adapter, image_config = load_embedding_adapter(image_adapter_dir, device=device)
    return query_adapter, image_adapter, query_config, image_config


def _apply_adapter_to_embeddings(adapter: Any, embeddings: list[tuple[str, Any]], device: str) -> list[tuple[str, Any]]:
    if adapter is None:
        return embeddings
    torch = _torch()
    out = []
    adapter.eval()
    with torch.no_grad():
        for item_id, emb in embeddings:
            adapted = adapter(emb.to(device)).detach().cpu()
            out.append((item_id, adapted))
    return out


def _encode_queries_v2(model, processor, query_rows, batch_size=8, device=None, attention_mode: str = "trim"):
    torch = _torch()
    legacy = _legacy()
    runtime_device = device or legacy.DEVICE
    encoded = []
    with torch.no_grad():
        for batch in _batched(query_rows, batch_size):
            texts = [row["query"] for row in batch]
            inputs = legacy.process_query_batch(processor, texts)
            inputs = {k: v.to(runtime_device) for k, v in inputs.items()}
            outputs = legacy.run_model_for_embeddings(model, inputs, is_image_batch=False)
            attention_mask = inputs.get("attention_mask")
            for i, row in enumerate(batch):
                mask = attention_mask[i] if attention_mode == "trim" and attention_mask is not None else None
                emb = legacy._trim_embedding_rows(outputs[i], mask).detach().cpu()
                encoded.append((row["query_id"], emb))
    return encoded


def _encode_images_v2(model, processor, doc_rows, batch_size=2, device=None, image_resolution: int | None = None):
    torch = _torch()
    Image = _pil_image()
    legacy = _legacy()
    runtime_device = device or legacy.DEVICE
    encoded = []
    if image_resolution is not None:
        legacy.set_processor_image_size(processor, (int(image_resolution), int(image_resolution)))
    with torch.no_grad():
        for batch in _batched(doc_rows, batch_size):
            images = [Image.open(row["image_path"]).convert("RGB") for row in batch]
            inputs = legacy.process_image_batch(processor, images)
            inputs = {k: v.to(runtime_device) for k, v in inputs.items()}
            outputs = legacy.run_model_for_embeddings(model, inputs, is_image_batch=True)
            for i, row in enumerate(batch):
                emb = legacy._trim_embedding_rows(outputs[i], None).detach().cpu()
                encoded.append((row["doc_id"], emb))
            for img in images:
                img.close()
    return encoded


def evaluate_manifest_v2(
    manifest_path: str | Path,
    model_name: str = DEFAULT_RETRIEVAL_MODEL_NAME,
    candidate_manifest_path: str | Path | None = None,
    query_adapter_dir: str | Path | None = None,
    image_adapter_dir: str | Path | None = None,
    limit_queries: int | None = None,
    limit_docs: int | None = None,
    query_batch_size: int = 8,
    image_batch_size: int = 2,
    score_batch_size: int = 64,
    image_resolution: int | None = None,
    attention_mode: str = "trim",
    run_name: str | None = None,
    output_path: str | Path | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    legacy = _legacy()
    query_rows, doc_rows = legacy.build_eval_sets_from_manifest(
        manifest_path,
        candidate_manifest_path=candidate_manifest_path,
        limit_queries=limit_queries,
        limit_docs=limit_docs,
    )
    runtime = legacy.load_retriever(model_name=model_name, device=device)
    runtime_device = runtime.device
    if image_resolution is not None:
        legacy.set_processor_image_size(runtime.processor, (int(image_resolution), int(image_resolution)))

    query_adapter, image_adapter, query_config, image_config = _resolve_adapter_pair(
        query_adapter_dir=query_adapter_dir,
        image_adapter_dir=image_adapter_dir,
        device=runtime_device,
    )

    with _Timer() as query_timer:
        query_embeddings = _encode_queries_v2(
            runtime.model,
            runtime.processor,
            query_rows,
            batch_size=query_batch_size,
            device=runtime_device,
            attention_mode=attention_mode,
        )
        query_embeddings = _apply_adapter_to_embeddings(query_adapter, query_embeddings, runtime_device)

    with _Timer() as image_timer:
        doc_embeddings = _encode_images_v2(
            runtime.model,
            runtime.processor,
            doc_rows,
            batch_size=image_batch_size,
            device=runtime_device,
            image_resolution=image_resolution,
        )
        doc_embeddings = _apply_adapter_to_embeddings(image_adapter, doc_embeddings, runtime_device)

    with _Timer() as score_timer:
        rankings = legacy.compute_rankings_with_processor(
            runtime.processor,
            query_embeddings,
            doc_embeddings,
            score_batch_size=score_batch_size,
        )

    metrics = compute_rich_retrieval_metrics(rankings, query_rows, num_candidate_docs=len(doc_rows))
    index_stats = compute_index_stats(doc_embeddings)
    efficiency = {
        "query_encoding_latency_s": query_timer.seconds,
        "image_encoding_latency_s": image_timer.seconds,
        "search_latency_s": score_timer.seconds,
        "query_encoding_latency_ms_per_query": 1000 * query_timer.seconds / max(1, len(query_rows)),
        "image_encoding_latency_ms_per_doc": 1000 * image_timer.seconds / max(1, len(doc_rows)),
        "search_latency_ms_per_query": 1000 * score_timer.seconds / max(1, len(query_rows)),
        **index_stats,
    }
    preview = []
    for row in query_rows[:10]:
        preview.append(
            {
                "query_id": row["query_id"],
                "query": row["query"],
                "gold_doc_id": row["doc_id"],
                "top_docs": rankings.get(row["query_id"], [])[:5],
            }
        )
    result = {
        "run_name": run_name,
        "manifest_path": str(manifest_path),
        "candidate_manifest_path": str(candidate_manifest_path) if candidate_manifest_path else str(manifest_path),
        "model_name": model_name,
        "resolved_model_name": runtime.resolved_model_name,
        "backend": runtime.backend,
        "device": runtime.device,
        "query_adapter_dir": str(query_adapter_dir) if query_adapter_dir else None,
        "image_adapter_dir": str(image_adapter_dir) if image_adapter_dir else None,
        "query_adapter_config": query_config,
        "image_adapter_config": image_config,
        "image_resolution": image_resolution,
        "attention_mode": attention_mode,
        "metrics": metrics,
        "efficiency": efficiency,
        "preview": preview,
    }
    if output_path is not None:
        export_json(output_path, result)
    return result


def _late_interaction_score_matrix(query_embeddings: list[Any], doc_embeddings: list[Any]):
    torch = _torch()
    rows = []
    for q_emb in query_embeddings:
        row_scores = []
        for d_emb in doc_embeddings:
            sim = torch.matmul(q_emb, d_emb.transpose(0, 1))
            row_scores.append(sim.max(dim=1).values.sum())
        rows.append(torch.stack(row_scores))
    return torch.stack(rows)


def _infer_embedding_dim(manifest_path: str | Path, model_name: str, device: str, image_resolution: int | None):
    legacy = _legacy()
    query_rows, doc_rows = legacy.build_eval_sets_from_manifest(manifest_path, limit_queries=1, limit_docs=1)
    runtime = legacy.load_retriever(model_name=model_name, device=device)
    if image_resolution is not None:
        legacy.set_processor_image_size(runtime.processor, (int(image_resolution), int(image_resolution)))
    query_embeddings = _encode_queries_v2(runtime.model, runtime.processor, query_rows, batch_size=1, device=runtime.device)
    return int(query_embeddings[0][1].shape[-1])


def build_embedding_adapter(embedding_dim: int, hidden_dim: int = 512, dropout: float = 0.05, residual_scale: float = 0.1):
    return ResidualEmbeddingAdapter(embedding_dim, hidden_dim, dropout, residual_scale)


def train_embedding_adapter(
    train_manifest: str | Path,
    output_dir: str | Path,
    model_name: str = DEFAULT_RETRIEVAL_MODEL_NAME,
    val_manifest: str | Path | None = None,
    mode: str = "query_adapter",
    num_epochs: int = 1,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.01,
    hidden_dim: int = 512,
    dropout: float = 0.05,
    residual_scale: float = 0.1,
    max_train_rows: int | None = None,
    one_query_per_doc: bool = True,
    image_resolution: int | None = None,
    attention_mode: str = "trim",
    seed: int = 42,
    device: str | None = None,
    val_limit_queries: int | None = None,
) -> dict[str, Any]:
    if mode not in {"query_adapter", "dual_adapter"}:
        raise ValueError("mode must be 'query_adapter' or 'dual_adapter'")

    torch = _torch()
    legacy = _legacy()
    random.seed(seed)
    torch.manual_seed(seed)

    rows = legacy.read_manifest_for_training(train_manifest, one_query_per_doc=one_query_per_doc)
    if max_train_rows is not None:
        rows = rows[:max_train_rows]
    if not rows:
        raise ValueError(f"No training rows found in manifest: {train_manifest}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = legacy.load_retriever(model_name=model_name, device=device)
    runtime_device = runtime.device
    if image_resolution is not None:
        legacy.set_processor_image_size(runtime.processor, (int(image_resolution), int(image_resolution)))
    runtime.model.eval()
    for parameter in runtime.model.parameters():
        parameter.requires_grad_(False)

    embedding_dim = _infer_embedding_dim(train_manifest, model_name, runtime_device, image_resolution)
    query_adapter = build_embedding_adapter(embedding_dim, hidden_dim, dropout, residual_scale).to(runtime_device)
    image_adapter = build_embedding_adapter(embedding_dim, hidden_dim, dropout, residual_scale).to(runtime_device) if mode == "dual_adapter" else None
    parameters = list(query_adapter.parameters()) + (list(image_adapter.parameters()) if image_adapter is not None else [])
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)

    history = []
    best_metric = float("-inf")
    best_query_adapter_dir = None
    best_image_adapter_dir = None

    for epoch in range(num_epochs):
        random.shuffle(rows)
        query_adapter.train()
        if image_adapter is not None:
            image_adapter.train()
        running_loss = 0.0
        steps = 0

        for batch in _batched(rows, batch_size):
            query_rows = [{"query_id": row["query_id"], "query": row["query"], "doc_id": row["doc_id"]} for row in batch]
            doc_rows = [{"doc_id": row["doc_id"], "image_path": row["image_path"]} for row in batch]
            q_pairs = _encode_queries_v2(
                runtime.model,
                runtime.processor,
                query_rows,
                batch_size=len(batch),
                device=runtime_device,
                attention_mode=attention_mode,
            )
            d_pairs = _encode_images_v2(
                runtime.model,
                runtime.processor,
                doc_rows,
                batch_size=len(batch),
                device=runtime_device,
                image_resolution=image_resolution,
            )
            q_embs = [query_adapter(emb.to(runtime_device)) for _, emb in q_pairs]
            if image_adapter is None:
                d_embs = [emb.to(runtime_device) for _, emb in d_pairs]
            else:
                d_embs = [image_adapter(emb.to(runtime_device)) for _, emb in d_pairs]

            scores = _late_interaction_score_matrix(q_embs, d_embs)
            targets = torch.arange(scores.size(0), device=runtime_device)
            loss = 0.5 * (
                torch.nn.functional.cross_entropy(scores, targets)
                + torch.nn.functional.cross_entropy(scores.transpose(0, 1), targets)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu().item())
            steps += 1

        epoch_query_dir = output_dir / f"epoch_{epoch + 1}" / "query_adapter"
        query_config = {
            "adapter_type": "residual_embedding_adapter",
            "mode": mode,
            "side": "query",
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "residual_scale": residual_scale,
            "image_resolution": image_resolution,
            "attention_mode": attention_mode,
            "model_name": model_name,
        }
        save_embedding_adapter(query_adapter, epoch_query_dir, query_config)

        epoch_image_dir = None
        if image_adapter is not None:
            epoch_image_dir = output_dir / f"epoch_{epoch + 1}" / "image_adapter"
            image_config = {**query_config, "side": "image"}
            save_embedding_adapter(image_adapter, epoch_image_dir, image_config)

        epoch_summary: dict[str, Any] = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(1, steps),
            "query_adapter_dir": str(epoch_query_dir),
            "image_adapter_dir": str(epoch_image_dir) if epoch_image_dir else None,
        }

        if val_manifest is not None:
            val_result = evaluate_manifest_v2(
                val_manifest,
                model_name=model_name,
                query_adapter_dir=epoch_query_dir,
                image_adapter_dir=epoch_image_dir,
                limit_queries=val_limit_queries,
                image_resolution=image_resolution,
                attention_mode=attention_mode,
                run_name=f"val_epoch_{epoch + 1}",
                device=runtime_device,
            )
            epoch_summary["val_metrics"] = val_result["metrics"]
            metric = float(val_result["metrics"]["MRR"])
        else:
            metric = -float(epoch + 1)

        if best_query_adapter_dir is None or metric > best_metric:
            best_metric = metric
            best_query_adapter_dir = str(epoch_query_dir)
            best_image_adapter_dir = str(epoch_image_dir) if epoch_image_dir else None

        history.append(epoch_summary)

    summary = {
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest) if val_manifest else None,
        "mode": mode,
        "num_rows": len(rows),
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "best_query_adapter_dir": best_query_adapter_dir,
        "best_image_adapter_dir": best_image_adapter_dir,
        "history": history,
    }
    export_json(output_dir / "training_summary.json", summary)
    return summary


def create_degraded_image_manifest(
    manifest_path: str | Path,
    output_manifest_path: str | Path,
    output_image_dir: str | Path,
    degradation: str = "jpeg_low",
    jpeg_quality: int = 35,
    blur_radius: float = 1.5,
    downscale_factor: float = 0.5,
    contrast_factor: float = 0.6,
) -> str:
    Image = _pil_image()
    ImageFilter = _pil_filter()
    ImageEnhance = _pil_enhance()
    rows = read_jsonl(manifest_path)
    output_image_dir = Path(output_image_dir)
    output_image_dir.mkdir(parents=True, exist_ok=True)
    new_rows = []
    seen: dict[str, str] = {}

    for row in rows:
        src = row["image_path"]
        if src not in seen:
            image = Image.open(src).convert("RGB")
            if degradation == "jpeg_low":
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=jpeg_quality)
                buffer.seek(0)
                image = Image.open(buffer).convert("RGB")
            elif degradation == "blur":
                image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            elif degradation == "downscale":
                small_size = (max(1, int(image.width * downscale_factor)), max(1, int(image.height * downscale_factor)))
                image = image.resize(small_size, Image.Resampling.BICUBIC).resize(image.size, Image.Resampling.BICUBIC)
            elif degradation == "grayscale":
                image = image.convert("L").convert("RGB")
            elif degradation == "low_contrast":
                image = ImageEnhance.Contrast(image).enhance(contrast_factor)
            else:
                raise ValueError(f"Unsupported degradation: {degradation}")
            dst = output_image_dir / f"{Path(src).stem}_{degradation}.jpg"
            image.save(dst, quality=jpeg_quality if degradation == "jpeg_low" else 90)
            image.close()
            seen[src] = str(dst)
        new_row = dict(row)
        new_row["clean_image_path"] = row["image_path"]
        new_row["image_path"] = seen[src]
        new_row["image_degradation"] = degradation
        new_rows.append(new_row)

    export_jsonl(new_rows, output_manifest_path)
    return str(output_manifest_path)


def create_perturbed_query_manifest(
    manifest_path: str | Path,
    output_manifest_path: str | Path,
    mode: str = "shorten",
    typo_probability: float = 0.04,
    seed: int = 42,
) -> str:
    rng = random.Random(seed)
    rows = read_jsonl(manifest_path)
    out = []
    for row in rows:
        query = str(row["query"])
        if mode == "shorten":
            words = query.split()
            query_out = " ".join(words[: max(2, math.ceil(len(words) * 0.65))])
        elif mode == "typos":
            chars = []
            for ch in query:
                if ch.isalpha() and rng.random() < typo_probability:
                    continue
                chars.append(ch)
            query_out = "".join(chars) or query
        elif mode == "formatting":
            query_out = "  ".join(query.split())
        else:
            raise ValueError(f"Unsupported perturbation mode: {mode}")
        new_row = dict(row)
        new_row["clean_query"] = query
        new_row["query"] = query_out
        new_row["query_perturbation"] = mode
        out.append(new_row)
    export_jsonl(out, output_manifest_path)
    return str(output_manifest_path)


def evaluate_robustness_suite(
    clean_manifest_path: str | Path,
    output_dir: str | Path,
    model_name: str = DEFAULT_RETRIEVAL_MODEL_NAME,
    query_adapter_dir: str | Path | None = None,
    image_adapter_dir: str | Path | None = None,
    degradations: tuple[str, ...] = ("jpeg_low", "blur", "downscale", "grayscale", "low_contrast"),
    query_perturbations: tuple[str, ...] = ("shorten", "typos", "formatting"),
    limit_queries: int | None = None,
    limit_docs: int | None = None,
    device: str | None = None,
) -> list[dict[str, Any]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    clean = evaluate_manifest_v2(
        clean_manifest_path,
        model_name=model_name,
        query_adapter_dir=query_adapter_dir,
        image_adapter_dir=image_adapter_dir,
        limit_queries=limit_queries,
        limit_docs=limit_docs,
        run_name="clean",
        output_path=output_dir / "clean.json",
        device=device,
    )
    clean_metrics = clean["metrics"]
    rows.append({"run_name": "clean", "robustness_type": "clean", **clean_metrics, **clean["efficiency"]})

    for degradation in degradations:
        degraded_manifest = create_degraded_image_manifest(
            clean_manifest_path,
            output_dir / "manifests" / f"image_{degradation}.jsonl",
            output_dir / "images" / degradation,
            degradation=degradation,
        )
        result = evaluate_manifest_v2(
            degraded_manifest,
            model_name=model_name,
            query_adapter_dir=query_adapter_dir,
            image_adapter_dir=image_adapter_dir,
            limit_queries=limit_queries,
            limit_docs=limit_docs,
            run_name=f"image_{degradation}",
            output_path=output_dir / f"image_{degradation}.json",
            device=device,
        )
        row = _robustness_row(f"image_{degradation}", "image", clean_metrics, result)
        rows.append(row)

    for perturbation in query_perturbations:
        perturbed_manifest = create_perturbed_query_manifest(
            clean_manifest_path,
            output_dir / "manifests" / f"query_{perturbation}.jsonl",
            mode=perturbation,
        )
        result = evaluate_manifest_v2(
            perturbed_manifest,
            model_name=model_name,
            query_adapter_dir=query_adapter_dir,
            image_adapter_dir=image_adapter_dir,
            limit_queries=limit_queries,
            limit_docs=limit_docs,
            run_name=f"query_{perturbation}",
            output_path=output_dir / f"query_{perturbation}.json",
            device=device,
        )
        rows.append(_robustness_row(f"query_{perturbation}", "query", clean_metrics, result))

    save_results_table(rows, output_dir / "robustness_results")
    return rows


def _robustness_row(run_name: str, robustness_type: str, clean_metrics: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    row = {"run_name": run_name, "robustness_type": robustness_type, **result["metrics"], **result["efficiency"]}
    for metric in ("Recall@1", "Recall@5", "Recall@10", "MRR", "MAP@10", "nDCG@10"):
        if metric in clean_metrics and metric in result["metrics"]:
            clean_value = float(clean_metrics[metric])
            value = float(result["metrics"][metric])
            row[f"{metric}_absolute_drop"] = clean_value - value
            row[f"{metric}_relative_drop_pct"] = 100.0 * (clean_value - value) / clean_value if clean_value else 0.0
    return row


def export_mws_vision_retrieval_manifest(
    output_dir: str | Path,
    split: str = "train",
    config_name: str = "default",
    max_rows: int | None = 200,
    seed: int = 42,
) -> dict[str, str]:
    datasets = importlib.import_module("datasets")
    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    manifest_dir = output_dir / "manifests"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    ds = datasets.load_dataset("MTSAIR/MWS-Vision-Bench", config_name, split=split)
    if max_rows is not None and max_rows < len(ds):
        ds = ds.shuffle(seed=seed).select(range(max_rows))

    rows = []
    for idx, item in enumerate(ds):
        image = item["image"].convert("RGB")
        doc_id = f"mws_doc_{item.get('id', idx)}"
        query_id = f"mws_q_{item.get('id', idx)}"
        image_path = image_dir / f"{doc_id}.png"
        image.save(image_path)
        rows.append(
            {
                "query_id": query_id,
                "doc_id": doc_id,
                "query": str(item.get("question", "")).strip(),
                "image_path": str(image_path),
                "split": "mws_test",
                "image_variant": "real_ru",
                "source_lang_query": "ru",
                "target_lang_query": "ru",
                "dataset": "MTSAIR/MWS-Vision-Bench",
                "mws_id": item.get("id"),
                "mws_type": item.get("type"),
                "mws_dataset_name": item.get("dataset_name"),
                "answers": item.get("answers"),
            }
        )
        image.close()

    manifest_path = manifest_dir / "mws_vision_retrieval.jsonl"
    export_jsonl(rows, manifest_path)
    summary = {
        "manifest_path": str(manifest_path),
        "num_rows": len(rows),
        "config_name": config_name,
        "split": split,
    }
    export_json(output_dir / "mws_manifest_summary.json", summary)
    return summary


def evaluate_by_metadata_group(
    manifest_path: str | Path,
    group_key: str,
    output_dir: str | Path,
    **eval_kwargs,
) -> list[dict[str, Any]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(manifest_path)
    groups = sorted({str(row.get(group_key, "unknown")) for row in rows})
    result_rows = []
    for group in groups:
        group_rows = [row for row in rows if str(row.get(group_key, "unknown")) == group]
        if not group_rows:
            continue
        group_manifest = output_dir / "manifests" / f"{group_key}_{_slugify(group)}.jsonl"
        export_jsonl(group_rows, group_manifest)
        result = evaluate_manifest_v2(
            group_manifest,
            run_name=f"{group_key}_{group}",
            output_path=output_dir / f"{group_key}_{_slugify(group)}.json",
            **eval_kwargs,
        )
        result_rows.append({"group_key": group_key, "group_value": group, **result["metrics"], **result["efficiency"]})
    save_results_table(result_rows, output_dir / f"{group_key}_group_results")
    return result_rows


def _simple_text_tokens(text: str) -> list[str]:
    token = []
    tokens = []
    for ch in str(text).lower():
        if ch.isalnum():
            token.append(ch)
        elif token:
            tokens.append("".join(token))
            token = []
    if token:
        tokens.append("".join(token))
    return tokens


def _row_document_text(row: dict[str, Any], fields: tuple[str, ...]) -> str:
    parts = []
    for field in fields:
        value = row.get(field)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            parts.extend(str(item) for item in value)
        else:
            parts.append(str(value))
    text = "\n".join(part for part in parts if part.strip())
    if text.strip():
        return text
    fallback_parts = [
        str(row.get("answer", "") or ""),
        str(row.get("original_query", "") or ""),
        str(row.get("query", "") or ""),
    ]
    return "\n".join(part for part in fallback_parts if part.strip())


def evaluate_early_fusion_text_proxy(
    manifest_path: str | Path,
    output_dir: str | Path,
    run_name: str = "early_fusion_text_proxy_bm25",
    candidate_manifest_path: str | Path | None = None,
    document_text_fields: tuple[str, ...] = ("ocr_text", "document_text", "text", "markdown", "answer"),
    limit_queries: int | None = None,
    limit_docs: int | None = None,
    k1: float = 1.5,
    b: float = 0.75,
) -> dict[str, Any]:
    legacy = _legacy()
    query_rows, doc_rows = legacy.build_eval_sets_from_manifest(
        manifest_path,
        candidate_manifest_path=candidate_manifest_path,
        limit_queries=limit_queries,
        limit_docs=limit_docs,
    )
    source_rows = read_jsonl(candidate_manifest_path or manifest_path)
    row_by_doc = {}
    for row in source_rows:
        row_by_doc.setdefault(row["doc_id"], row)

    documents = []
    for doc in doc_rows:
        row = row_by_doc.get(doc["doc_id"], {})
        tokens = _simple_text_tokens(_row_document_text(row, document_text_fields))
        documents.append({"doc_id": doc["doc_id"], "tokens": tokens})

    doc_freq: dict[str, int] = {}
    for doc in documents:
        for tok in set(doc["tokens"]):
            doc_freq[tok] = doc_freq.get(tok, 0) + 1
    avgdl = sum(len(doc["tokens"]) for doc in documents) / max(1, len(documents))
    num_docs = len(documents)
    rankings = {}
    with _Timer() as timer:
        for query in query_rows:
            q_tokens = _simple_text_tokens(query["query"])
            scores = []
            for doc in documents:
                freqs: dict[str, int] = {}
                for tok in doc["tokens"]:
                    freqs[tok] = freqs.get(tok, 0) + 1
                score = 0.0
                dl = len(doc["tokens"])
                for tok in q_tokens:
                    tf = freqs.get(tok, 0)
                    if tf == 0:
                        continue
                    df = doc_freq.get(tok, 0)
                    idf = math.log(1 + (num_docs - df + 0.5) / (df + 0.5))
                    denom = tf + k1 * (1 - b + b * dl / max(1e-9, avgdl))
                    score += idf * (tf * (k1 + 1)) / denom
                scores.append((doc["doc_id"], score))
            rankings[query["query_id"]] = sorted(scores, key=lambda item: item[1], reverse=True)

    metrics = compute_rich_retrieval_metrics(rankings, query_rows, num_candidate_docs=len(documents))
    result = {
        "run_name": run_name,
        "baseline_type": "early_fusion_text_proxy_bm25",
        "manifest_path": str(manifest_path),
        "candidate_manifest_path": str(candidate_manifest_path) if candidate_manifest_path else str(manifest_path),
        "document_text_fields": list(document_text_fields),
        "metrics": metrics,
        "efficiency": {
            "search_latency_s": timer.seconds,
            "search_latency_ms_per_query": 1000 * timer.seconds / max(1, len(query_rows)),
            "index_size_mb": sum(len(" ".join(doc["tokens"]).encode("utf-8")) for doc in documents) / (1024 * 1024),
            "avg_doc_token_vectors": sum(len(doc["tokens"]) for doc in documents) / max(1, len(documents)),
            "max_doc_token_vectors": max([len(doc["tokens"]) for doc in documents] or [0]),
            "num_documents_indexed": len(documents),
        },
        "preview": [
            {
                "query_id": row["query_id"],
                "query": row["query"],
                "gold_doc_id": row["doc_id"],
                "top_docs": rankings.get(row["query_id"], [])[:5],
            }
            for row in query_rows[:10]
        ],
    }
    output_dir = Path(output_dir)
    export_json(output_dir / f"{run_name}.json", result)
    save_results_table([{ "run_name": run_name, **metrics, **result["efficiency"] }], output_dir / f"{run_name}_results")
    return result


def _slugify(text: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text))
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "item"


def evaluate_main_v2_suite(
    manifest_groups: dict[str, dict[str, str]],
    output_dir: str | Path,
    model_name: str = DEFAULT_RETRIEVAL_MODEL_NAME,
    query_adapter_dir: str | Path | None = None,
    synthetic_query_adapter_dir: str | Path | None = None,
    dual_query_adapter_dir: str | Path | None = None,
    dual_image_adapter_dir: str | Path | None = None,
    aligned_test_manifests: dict[str, dict[str, Any]] | None = None,
    device: str | None = None,
) -> list[dict[str, Any]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs: list[tuple[str, str, str | Path | None, str | Path | None]] = [
        ("english_reference_zero_shot", manifest_groups["english_reference"]["test"], None, None),
        ("ru_query_original_image_zero_shot", manifest_groups["russian_baseline"]["test"], None, None),
        ("ru_query_synthetic_image_zero_shot", manifest_groups["synthetic_primary"]["test"], None, None),
        ("ru_query_original_image_query_adapter", manifest_groups["russian_baseline"]["test"], query_adapter_dir, None),
        ("ru_query_synthetic_image_query_adapter", manifest_groups["synthetic_primary"]["test"], synthetic_query_adapter_dir or query_adapter_dir, None),
        ("ru_query_synthetic_image_dual_adapter", manifest_groups["synthetic_primary"]["test"], dual_query_adapter_dir, dual_image_adapter_dir),
    ]
    if aligned_test_manifests:
        runs.extend(
            [
                (
                    "aligned_original_zero_shot",
                    aligned_test_manifests["synthetic_primary"]["baseline_manifest"],
                    None,
                    None,
                ),
                (
                    "aligned_synthetic_zero_shot",
                    aligned_test_manifests["synthetic_primary"]["synthetic_manifest"],
                    None,
                    None,
                ),
            ]
        )

    result_rows = []
    for run_name, manifest_path, q_adapter, i_adapter in runs:
        if q_adapter is None and "adapter" in run_name:
            continue
        if run_name.endswith("dual_adapter") and (q_adapter is None or i_adapter is None):
            continue
        result = evaluate_manifest_v2(
            manifest_path=manifest_path,
            model_name=model_name,
            query_adapter_dir=q_adapter,
            image_adapter_dir=i_adapter,
            run_name=run_name,
            output_path=output_dir / f"{run_name}.json",
            device=device,
        )
        result_rows.append({"run_name": run_name, **result["metrics"], **result["efficiency"]})
    save_results_table(result_rows, output_dir / "main_v2_results")
    return result_rows


def run_tiny_v2_sanity_check(
    output_dir: str | Path | None = None,
    model_name: str = DEFAULT_RETRIEVAL_MODEL_NAME,
    device: str = "cpu",
    experiment_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    legacy = _legacy()
    output_dir = Path(output_dir or legacy.default_workdir("modernvbert_ru")) / "v2_sanity_check"
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight = environment_preflight(model_name=model_name, device=device, run_model_check=True, output_path=output_dir / "preflight.json")
    config = {
        "output_dir": output_dir,
        "max_rows": 12,
        "synthetic_subset_size": {"train": 4, "val": 2, "test": 2},
        "translation_model": "Helsinki-NLP/opus-mt-en-ru",
        "font_path": None,
        "seed": 42,
        "streaming": True,
        "translator_device": "cpu",
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "synthetic_splits": ("train", "val", "test"),
        "ocr_backend": "paddleocr",
        "ocr_langs": ("en",),
        "ocr_use_angle_cls": True,
        "preprocess_for_ocr": True,
    }
    if experiment_overrides:
        config.update(experiment_overrides)
    summary = prepare_thesis_experiment_v2(**config)
    zero_shot = evaluate_manifest_v2(
        summary["manifests"]["russian_baseline"]["test"],
        model_name=model_name,
        limit_queries=4,
        limit_docs=4,
        run_name="tiny_zero_shot",
        output_path=output_dir / "results" / "tiny_zero_shot.json",
        device=device,
    )
    training = train_embedding_adapter(
        train_manifest=summary["manifests"]["russian_baseline"]["train"],
        val_manifest=summary["manifests"]["russian_baseline"]["val"],
        output_dir=output_dir / "adapters" / "query_adapter",
        model_name=model_name,
        mode="query_adapter",
        num_epochs=1,
        batch_size=2,
        max_train_rows=4,
        val_limit_queries=4,
        device=device,
    )
    adapted = evaluate_manifest_v2(
        summary["manifests"]["russian_baseline"]["test"],
        model_name=model_name,
        query_adapter_dir=training["best_query_adapter_dir"],
        limit_queries=4,
        limit_docs=4,
        run_name="tiny_query_adapter",
        output_path=output_dir / "results" / "tiny_query_adapter.json",
        device=device,
    )
    robustness = evaluate_robustness_suite(
        clean_manifest_path=summary["manifests"]["russian_baseline"]["test"],
        output_dir=output_dir / "robustness",
        model_name=model_name,
        query_adapter_dir=training["best_query_adapter_dir"],
        degradations=("jpeg_low",),
        query_perturbations=("shorten",),
        limit_queries=4,
        limit_docs=4,
        device=device,
    )
    result = {
        "preflight": preflight,
        "prepare_summary": summary,
        "zero_shot_metrics": zero_shot["metrics"],
        "training": training,
        "adapted_metrics": adapted["metrics"],
        "robustness_rows": robustness,
    }
    export_json(output_dir / "sanity_summary.json", result)
    return result


def transformers_lora_experimental_status(output_path: str | Path | None = None) -> dict[str, Any]:
    report = environment_preflight(run_model_check=False)
    status = report["modes"]["transformers_lora_experimental"]
    payload = {
        "enabled": status["enabled"],
        "reason": status["reason"],
        "policy": "LoRA is experimental in v2 and must use Transformers ColModernVBertForRetrieval only; adapter_dir is never passed to colpali_engine.",
    }
    if output_path is not None:
        export_json(output_path, payload)
    return payload


def export_thesis_tables_v2(result_rows: list[dict[str, Any]], output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered_metrics = [
        "Recall@1",
        "Recall@5",
        "Recall@10",
        "MRR",
        "MAP@1",
        "MAP@5",
        "MAP@10",
        "nDCG@1",
        "nDCG@5",
        "nDCG@10",
        "index_size_mb",
        "search_latency_ms_per_query",
        "avg_doc_token_vectors",
    ]
    table = []
    for row in result_rows:
        out = {"run_name": row.get("run_name")}
        for metric in ordered_metrics:
            if metric in row:
                value = row[metric]
                out[metric] = round(float(value), 4) if isinstance(value, (float, int)) else value
        table.append(out)
    save_results_table(table, output_dir / "thesis_v2_table")
    return {
        "table_json": str(output_dir / "thesis_v2_table.json"),
        "table_csv": str(output_dir / "thesis_v2_table.csv"),
    }


def _define_residual_embedding_adapter():
    torch = _torch()

    class ResidualEmbeddingAdapter(torch.nn.Module):
        def __init__(self, embedding_dim: int, hidden_dim: int = 512, dropout: float = 0.05, residual_scale: float = 0.1):
            super().__init__()
            self.embedding_dim = int(embedding_dim)
            self.hidden_dim = int(hidden_dim)
            self.dropout = float(dropout)
            self.residual_scale = float(residual_scale)
            self.net = torch.nn.Sequential(
                torch.nn.LayerNorm(self.embedding_dim),
                torch.nn.Linear(self.embedding_dim, self.hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(self.dropout),
                torch.nn.Linear(self.hidden_dim, self.embedding_dim),
            )

        def forward(self, embeddings):
            return embeddings + self.residual_scale * self.net(embeddings)

    return ResidualEmbeddingAdapter


class _ResidualEmbeddingAdapterProxy:
    def __call__(self, *args, **kwargs):
        return _define_residual_embedding_adapter()(*args, **kwargs)


ResidualEmbeddingAdapter = _ResidualEmbeddingAdapterProxy()
