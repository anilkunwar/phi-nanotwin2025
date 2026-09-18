# ============================================================================
# ███ PLASTICITY PARAMETER INTELLIGENT RECOMMENDER v8.0 ███
# FAISS Retrieval · Ollama LLM · LatentMoE Scoring · Learned Priors · Histograms
# ----------------------------------------------------------------------------
# Drop this module verbatim at the bottom of your Laser‑MPEA file.
# It targets exactly these five JSON files inside ./json_metadatabase/:
#     friction_lattice_stress_metadatabase.json
#     initial_dislocation_density_metadatabase.json
#     reference_strain_rate_metadatabase.json
#     shear_modulus_metadatabase.json
#     strain_rate_sensitivity_metadatabase.json
# ============================================================================

from __future__ import annotations
import os
import re
import json
import math
import time
import pickle
import hashlib
import logging
import threading
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Iterable

import numpy as np
import pandas as pd
import streamlit as st

try:
    import requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

try:
    import faiss
    _FAISS_OK = True
except ImportError:
    _FAISS_OK = False

try:
    from sentence_transformers import SentenceTransformer
    _SBERT_OK = True
except ImportError:
    _SBERT_OK = False

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# 0.  PHYSICAL ONTOLOGY
# ----------------------------------------------------------------------------
PLASTICITY_ONTOLOGY: Dict[str, Dict[str, Any]] = {
    "rho0": {
        "label": "Initial Dislocation Density",
        "symbol": "ρ₀",
        "aliases": [
            "initial dislocation density", "dislocation density",
            "rho_0", "rho0", "ρ₀", "forest dislocation density",
            "mobile dislocation density", "immobile dislocation density",
            "total dislocation density", "geometrically necessary dislocation",
        ],
        "unit": "m^-2", "ui_unit": "m⁻²", "ui_scale": 1.0,
        "valid_range": (1e10, 1e17), "soft_range": (1e11, 1e16),
        "defaults": {"Cu": 1e12, "Al": 1e12, "Ni": 1e13,
                     "CoCrFeNi": 1e13, "Fe": 1e14},
        "expected_file": "initial_dislocation_density_metadatabase.json",
    },
    "mu": {
        "label": "Shear Modulus",
        "symbol": "μ",
        "aliases": [
            "shear modulus", "mu", "G", "elastic shear modulus",
            "c44", "c_44", "second lame parameter", "second lamé parameter",
            "rigidity modulus",
        ],
        "unit": "Pa", "ui_unit": "GPa", "ui_scale": 1e9,
        "valid_range": (1e9, 5e11), "soft_range": (20e9, 200e9),
        "defaults": {"Cu": 48e9, "Al": 26e9, "Ni": 80e9,
                     "CoCrFeNi": 82e9, "Fe": 82e9},
        "expected_file": "shear_modulus_metadatabase.json",
    },
    "gamma0_dot": {
        "label": "Reference Strain Rate",
        "symbol": "γ̇₀",
        "aliases": [
            "reference strain rate", "reference strain-rate",
            "gamma0_dot", "gamma_dot_0", "γ̇₀", "reference shear rate",
            "pre-exponential strain rate", "attempt frequency",
            "reference shear strain rate",
        ],
        "unit": "s^-1", "ui_unit": "s⁻¹", "ui_scale": 1.0,
        "valid_range": (1e-6, 1e12), "soft_range": (1e-4, 1e6),
        "defaults": {"Cu": 1e-3, "Al": 1e-3, "Ni": 1e-3,
                     "CoCrFeNi": 1e-3, "Fe": 1e-3},
        "expected_file": "reference_strain_rate_metadatabase.json",
    },
    "srs": {
        "label": "Strain‑Rate Sensitivity Exponent",
        "symbol": "m",
        "aliases": [
            "strain rate sensitivity", "srs", "m exponent",
            "stress exponent", "rate sensitivity", "strain-rate sensitivity",
            "rate sensitivity exponent", "n exponent", "viscous exponent",
        ],
        "unit": "dimensionless", "ui_unit": "–", "ui_scale": 1.0,
        "valid_range": (1.0, 200.0), "soft_range": (5.0, 50.0),
        "defaults": {"Cu": 20.0, "Al": 20.0, "Ni": 20.0,
                     "CoCrFeNi": 20.0, "Fe": 20.0},
        "expected_file": "strain_rate_sensitivity_metadatabase.json",
    },
    "sigma0": {
        "label": "Friction / Initial Yield Stress",
        "symbol": "σ₀",
        "aliases": [
            "initial yield stress", "friction stress", "sigma_0", "σ₀",
            "lattice friction", "peierls stress", "peierls-nabarro stress",
            "athermal stress", "yield strength", "lattice resistance",
            "friction lattice stress",
        ],
        "unit": "Pa", "ui_unit": "MPa", "ui_scale": 1e6,
        "valid_range": (1e5, 2e9), "soft_range": (10e6, 500e6),
        "defaults": {"Cu": 50e6, "Al": 30e6, "Ni": 70e6,
                     "CoCrFeNi": 120e6, "Fe": 100e6},
        "expected_file": "friction_lattice_stress_metadatabase.json",
    },
}

PARAM_ORDER: List[str] = ["rho0", "mu", "gamma0_dot", "srs", "sigma0"]

SOLVER_KEY_MAP: Dict[str, str] = {
    "rho0": "rho0",
    "mu": "mu",
    "gamma0_dot": "gamma0_dot",
    "srs": "m",
    "sigma0": "sigma0",
}

TARGET_JSON_FILES: List[str] = [
    "friction_lattice_stress_metadatabase.json",
    "initial_dislocation_density_metadatabase.json",
    "reference_strain_rate_metadatabase.json",
    "shear_modulus_metadatabase.json",
    "strain_rate_sensitivity_metadatabase.json",
]

# ----------------------------------------------------------------------------
# 1.  UTILITY HELPERS
# ----------------------------------------------------------------------------
def _hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _normalize_unit(value: float, unit: str, param: str) -> float:
    u = (unit or "").strip().lower().replace("μ", "u").replace("µ", "u")
    v = float(value)
    if param in ("mu", "sigma0"):
        if "gpa" in u:  return v * 1e9
        if "mpa" in u:  return v * 1e6
        if "kpa" in u:  return v * 1e3
        if "pa" in u:   return v
    if param == "rho0":
        if "cm^-2" in u or "cm-2" in u: return v * 1e4
        if "mm^-2" in u or "mm-2" in u: return v * 1e6
        if "m^-3" in u or "m-3" in u:   return v ** (2 / 3)  # rare, approximate
    return v


def _clamp(value: float, param: str) -> Tuple[float, bool]:
    lo, hi = PLASTICITY_ONTOLOGY[param]["valid_range"]
    if value < lo:  return lo, True
    if value > hi:  return hi, True
    return value, False


def _fmt(param: str, si_value: float) -> str:
    spec = PLASTICITY_ONTOLOGY[param]
    ui_val = si_value / spec["ui_scale"]
    if param in ("rho0", "gamma0_dot"):
        return f"{ui_val:.2e} {spec['ui_unit']}"
    if param == "mu":
        return f"{ui_val:.2f} {spec['ui_unit']}"
    if param == "sigma0":
        return f"{ui_val:.1f} {spec['ui_unit']}"
    return f"{ui_val:.2f} {spec['ui_unit']}"


# ----------------------------------------------------------------------------
# 2.  OLLAMA CLIENT
# ----------------------------------------------------------------------------
class OllamaClient:
    """Thin wrapper around Ollama /api/generate with retries + JSON coercion."""

    def __init__(
        self,
        url: str = "http://localhost:11434",
        model: str = "qwen2.5:7b",
        timeout: float = 90.0,
        max_retries: int = 2,
    ):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    @staticmethod
    def is_available(url: str = "http://localhost:11434/api/tags") -> bool:
        if not _REQUESTS_OK:
            return False
        try:
            r = requests.get(url, timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False

    @staticmethod
    def list_models(url: str = "http://localhost:11434") -> List[str]:
        if not _REQUESTS_OK:
            return []
        try:
            r = requests.get(f"{url.rstrip('/')}/api/tags", timeout=3.0)
            if r.status_code == 200:
                return sorted(m.get("name", "") for m in r.json().get("models", []))
        except Exception:
            pass
        return []

    def generate_json(self, prompt: str, system: Optional[str] = None) -> Optional[Any]:
        if not _REQUESTS_OK:
            return None
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1, "top_p": 0.9, "num_predict": 2048},
        }
        if system:
            payload["system"] = system

        for attempt in range(self.max_retries + 1):
            try:
                r = requests.post(
                    f"{self.url}/api/generate", json=payload, timeout=self.timeout
                )
                r.raise_for_status()
                raw = r.json().get("response", "")
                return self._parse_lenient(raw)
            except Exception as e:
                logger.warning("Ollama attempt %d failed: %s", attempt + 1, e)
                if attempt == self.max_retries:
                    return None
                time.sleep(0.75 * (attempt + 1))
        return None

    @staticmethod
    def _parse_lenient(raw: str) -> Any:
        if raw is None:
            return None
        s = raw.strip()
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
        for oc, cc in (("[", "]"), ("{", "}")):
            i, j = s.find(oc), s.rfind(cc)
            if i != -1 and j > i:
                try:
                    return json.loads(s[i : j + 1])
                except Exception:
                    continue
        try:
            return json.loads(s)
        except Exception:
            return None


# ----------------------------------------------------------------------------
# 3.  CORPUS LOADER (targets the 5 canonical JSON files)
# ----------------------------------------------------------------------------
class PlasticityCorpus:
    """Loads, chunks, and caches text records from the 5 metadatabases."""

    _CACHE_VERSION = "v8"

    def __init__(self, db_dir: str = "json_metadatabase", max_chars: int = 4000):
        self.db_dir = db_dir
        self.max_chars = max_chars

    def discover_files(self) -> List[str]:
        found: List[str] = []
        for fname in TARGET_JSON_FILES:
            fpath = os.path.join(self.db_dir, fname)
            if os.path.exists(fpath):
                found.append(fpath)
        # Fallback: any *.json that matches the synonyms
        if os.path.isdir(self.db_dir):
            for fname in os.listdir(self.db_dir):
                if not fname.lower().endswith(".json"):
                    continue
                fpath = os.path.join(self.db_dir, fname)
                if fpath in found:
                    continue
                low = fname.lower()
                if any(k in low for k in [
                    "dislocation", "shear", "strain_rate", "strain-rate",
                    "friction", "lattice", "yield", "sensitivity",
                ]):
                    found.append(fpath)
        return found

    def load(self, force_reload: bool = False) -> List[Dict[str, Any]]:
        """Return a list of records. Each record = {source, title, text, raw}."""
        cache_key = f"pl_corpus_{self._CACHE_VERSION}"
        if not force_reload and cache_key in st.session_state:
            return st.session_state[cache_key]

        corpus: List[Dict[str, Any]] = []
        for fpath in self.discover_files():
            try:
                with open(fpath, "r", encoding="utf-8-sig") as f:
                    data = json.load(f)
            except Exception as e:
                logger.warning("Cannot load %s: %s", fpath, e)
                continue

            # Records can be a list, or a dict with a records/data/papers key
            if isinstance(data, dict):
                for key in ("records", "data", "papers", "entries", "items"):
                    if key in data and isinstance(data[key], list):
                        data = data[key]
                        break
                else:
                    data = [data]
            if not isinstance(data, list):
                continue

            fname = os.path.basename(fpath)
            for item in data:
                if not isinstance(item, dict):
                    continue
                title = str(
                    item.get("Title") or item.get("title")
                    or item.get("name") or ""
                )
                abstract = str(
                    item.get("Abstract") or item.get("abstract")
                    or item.get("summary") or ""
                )
                full = str(
                    item.get("Full Text") or item.get("full_text")
                    or item.get("text") or item.get("content") or ""
                )
                text = (
                    f"Title: {title}. "
                    f"Abstract: {abstract}. "
                    f"Full Text: {full[:self.max_chars]}"
                )
                corpus.append({
                    "source": fname,
                    "title": title,
                    "text": text,
                    "raw": item,
                })

        st.session_state[cache_key] = corpus
        return corpus

    @staticmethod
    def keyword_prefilter(
        corpus: List[Dict[str, Any]], material: str, k: int = 30
    ) -> List[Dict[str, Any]]:
        syn = {
            "cu": ["cu", "copper"],
            "al": ["al", "aluminium", "aluminum"],
            "ni": ["ni", "nickel"],
            "fe": ["fe", "iron", "steel"],
            "cocrfeni": ["cocrfeni", "co-cr-fe-ni", "hea", "mpea",
                         "high entropy", "high-entropy"],
            "ti": ["ti", "titanium"],
            "mg": ["mg", "magnesium"],
        }
        keys = syn.get(material.lower(), [material.lower()])
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for doc in corpus:
            low = doc["text"].lower()
            score = 0.0
            for kk in keys:
                score += low.count(kk) * 2.0
            for spec in PLASTICITY_ONTOLOGY.values():
                for alias in spec["aliases"]:
                    if alias in low:
                        score += 1.0
                        break
            scored.append((score, doc))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [d for _, d in scored[:k]]


# ----------------------------------------------------------------------------
# 4.  FAISS RETRIEVAL LAYER
# ----------------------------------------------------------------------------
class FAISSRetriever:
    """Dense retrieval over the corpus using SentenceTransformer + FAISS.

    Falls back to NumPy brute force if FAISS is unavailable.
    Index is persisted to `.plasticity_cache/faiss_index.pkl`.
    """

    CACHE_DIR = ".plasticity_cache"
    INDEX_FILE = "faiss_index.pkl"

    def __init__(self, embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embed_model_name = embed_model_name
        self._model: Optional[SentenceTransformer] = None
        self._index = None
        self._doc_vectors: Optional[np.ndarray] = None
        self._docs: List[Dict[str, Any]] = []
        self._dim: int = 0
        self._lock = threading.Lock()

    # ---- lazy model load ----------------------------------------------------
    @property
    def model(self) -> Optional[SentenceTransformer]:
        if self._model is None and _SBERT_OK:
            try:
                self._model = SentenceTransformer(self.embed_model_name, device="cpu")
            except Exception as e:
                logger.warning("SentenceTransformer failed: %s", e)
                self._model = None
        return self._model

    # ---- index build --------------------------------------------------------
    def build(self, corpus: List[Dict[str, Any]], force: bool = False) -> None:
        if not corpus:
            self._docs, self._index, self._doc_vectors = [], None, None
            return

        fingerprint = _hash(json.dumps(
            [d["source"] + "|" + d["title"] for d in corpus], sort_keys=False
        ))
        cache_path = os.path.join(self.CACHE_DIR, self.INDEX_FILE)

        if not force and os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    payload = pickle.load(f)
                if payload.get("fingerprint") == fingerprint:
                    self._docs = payload["docs"]
                    self._doc_vectors = payload["vectors"]
                    self._dim = payload["dim"]
                    self._attach_index()
                    return
            except Exception as e:
                logger.warning("FAISS cache load failed: %s", e)

        m = self.model
        if m is None:
            # No SBERT: use TF‑IDF fallback vectors via sklearn if available
            self._docs = list(corpus)
            self._doc_vectors = self._tfidf_fallback_vectors(corpus)
            self._dim = self._doc_vectors.shape[1] if self._doc_vectors.size else 0
            self._attach_index()
            self._persist(fingerprint, cache_path)
            return

        with self._lock:
            texts = [d["text"][:1500] for d in corpus]
            vectors = m.encode(
                texts, batch_size=32, show_progress_bar=False,
                convert_to_numpy=True, normalize_embeddings=True,
            ).astype(np.float32)

        self._docs = list(corpus)
        self._doc_vectors = vectors
        self._dim = vectors.shape[1]
        self._attach_index()
        self._persist(fingerprint, cache_path)

    def _tfidf_fallback_vectors(self, corpus: List[Dict[str, Any]]) -> np.ndarray:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vec = TfidfVectorizer(max_features=2048, stop_words="english")
            X = vec.fit_transform([d["text"] for d in corpus]).toarray().astype(np.float32)
            norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
            return X / norms
        except Exception:
            return np.zeros((len(corpus), 1), dtype=np.float32)

    def _attach_index(self) -> None:
        if self._doc_vectors is None or self._doc_vectors.size == 0:
            self._index = None
            return
        if _FAISS_OK:
            try:
                index = faiss.IndexFlatIP(self._doc_vectors.shape[1])
                index.add(self._doc_vectors)
                self._index = index
                return
            except Exception as e:
                logger.warning("FAISS index build failed: %s", e)
        # Fallback → mark as None and use numpy search
        self._index = None

    def _persist(self, fingerprint: str, path: str) -> None:
        try:
            os.makedirs(self.CACHE_DIR, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump({
                    "fingerprint": fingerprint,
                    "docs": self._docs,
                    "vectors": self._doc_vectors,
                    "dim": self._dim,
                }, f)
        except Exception as e:
            logger.warning("FAISS persist failed: %s", e)

    # ---- query --------------------------------------------------------------
    def search(
        self,
        query: str,
        k: int = 15,
        material_hint: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not self._docs or self._doc_vectors is None or self._doc_vectors.size == 0:
            return []

        m = self.model
        if m is not None:
            qvec = m.encode(
                [query], batch_size=1, show_progress_bar=False,
                convert_to_numpy=True, normalize_embeddings=True,
            ).astype(np.float32)
        else:
            qvec = np.zeros((1, self._dim), dtype=np.float32)
            for i, d in enumerate(self._docs):
                if query.lower()[:30] in d["text"].lower():
                    qvec[0, i % self._dim] += 1.0
            nrm = np.linalg.norm(qvec, axis=1, keepdims=True) + 1e-12
            qvec = qvec / nrm

        # Optionally oversample so we can rerank with a keyword bonus
        top_k = min(len(self._docs), max(k * 3, k))
        if self._index is not None:
            D, I = self._index.search(qvec, top_k)
            sims, idxs = D[0], I[0]
        else:
            sims_all = (self._doc_vectors @ qvec[0])
            idxs = np.argsort(-sims_all)[:top_k]
            sims = sims_all[idxs]

        hits: List[Tuple[float, Dict[str, Any]]] = []
        for sim, idx in zip(sims, idxs):
            if idx < 0 or idx >= len(self._docs):
                continue
            doc = self._docs[int(idx)]
            bonus = 0.0
            if material_hint:
                low = doc["text"].lower()
                mat_low = material_hint.lower()
                if mat_low in low:
                    bonus += 0.05
            hits.append((float(sim) + bonus, doc))
        hits.sort(key=lambda t: t[0], reverse=True)
        return [d for _, d in hits[:k]]


# ----------------------------------------------------------------------------
# 5.  EXTRACTION: LLM prompt + heuristic fallback
# ----------------------------------------------------------------------------
_EXTRACT_SCHEMA = (
    '{"param": "rho0|mu|gamma0_dot|srs|sigma0", '
    '"value": <number>, "unit": "<string>", '
    '"material": "<string>", "temp": <number in K or null>, '
    '"strain_rate": <number in s^-1 or null>, '
    '"method": "<experiment|MD|DFT|review|unknown>", '
    '"confidence": <0.0-1.0>, '
    '"evidence": "<short quoted snippet>"}'
)

_EXTRACT_PROMPT = """You are a strict materials-science NER system.
Extract ONLY the following five plasticity parameters from the text:
  1. rho0        – initial dislocation density              [m^-2]
  2. mu          – shear modulus                            [Pa or GPa]
  3. gamma0_dot  – reference strain rate                    [s^-1]
  4. srs         – strain-rate sensitivity exponent (m)     [dimensionless]
  5. sigma0      – friction / initial yield stress          [Pa or MPa]

Target context (do NOT invent values for it):
  Material       = "{material}"
  Temperature    = "{temp_k}" K
  Strain rate    = "{strain_rate}" s^-1

Return ONLY a JSON ARRAY of objects. Each object must match this schema:
  {schema}
If a parameter is not present, omit it. If a field is unknown, use null.
Do NOT include markdown, comments, or explanations.

TEXT:
\"\"\"{text}\"\"\"
"""


class ParameterExtractor:
    """LLM extraction with heuristic fallback."""

    NUM_RE = re.compile(
        r"(-?\d+(?:\.\d+)?)\s*(?:[×xX\*]\s*10\s*\^?\s*\{?(-?\d+)\}?)?\s*"
        r"(GPa|MPa|kPa|Pa|m\^?-?2|m-2|s\^?-?1|/s)?",
        re.I,
    )

    def __init__(self, client: Optional[OllamaClient], cache: Optional[Dict[str, Any]] = None):
        self.client = client
        self.cache = cache if cache is not None else {}

    def extract(
        self, text: str, material: str, temp_k: float,
        strain_rate: float, use_llm: bool = True,
    ) -> List[Dict[str, Any]]:
        key = _hash(f"{text[:2000]}|{material}|{temp_k}|{strain_rate}|{use_llm}")
        if key in self.cache:
            return self.cache[key]

        out: List[Dict[str, Any]] = []
        if use_llm and self.client is not None:
            prompt = _EXTRACT_PROMPT.format(
                material=material, temp_k=temp_k, strain_rate=strain_rate,
                schema=_EXTRACT_SCHEMA, text=text[:3500],
            )
            raw = self.client.generate_json(prompt)
            out = self._validate(raw)
        if not out:
            out = self._heuristic(text, material, temp_k, strain_rate)
        self.cache[key] = out
        return out

    @staticmethod
    def _validate(raw: Any) -> List[Dict[str, Any]]:
        if raw is None:
            return []
        if isinstance(raw, dict):
            raw = [raw]
        if not isinstance(raw, list):
            return []
        out: List[Dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            p = str(item.get("param", "")).strip().lower()
            if p not in PLASTICITY_ONTOLOGY:
                continue
            try:
                v = float(item.get("value"))
            except (TypeError, ValueError):
                continue
            out.append({
                "param": p,
                "value": v,
                "unit": str(item.get("unit") or ""),
                "material": str(item.get("material") or ""),
                "temp": item.get("temp"),
                "strain_rate": item.get("strain_rate"),
                "method": str(item.get("method") or "unknown").lower(),
                "confidence": float(item.get("confidence", 0.5) or 0.5),
                "evidence": str(item.get("evidence") or "")[:240],
            })
        return out

    @classmethod
    def _heuristic(
        cls, text: str, material: str, temp_k: float, strain_rate: float
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        low = text.lower()
        for param, spec in PLASTICITY_ONTOLOGY.items():
            found = False
            for alias in spec["aliases"]:
                idx = low.find(alias.lower())
                if idx < 0:
                    continue
                window = text[max(0, idx - 80): idx + 300]
                m = cls.NUM_RE.search(window)
                if not m:
                    continue
                try:
                    mantissa = float(m.group(1))
                    exponent = int(m.group(2)) if m.group(2) else 0
                    unit = (m.group(3) or "").strip()
                    value = mantissa * (10 ** exponent)
                except Exception:
                    continue
                out.append({
                    "param": param,
                    "value": value,
                    "unit": unit,
                    "material": material,
                    "temp": temp_k,
                    "strain_rate": strain_rate,
                    "method": "heuristic",
                    "confidence": 0.35,
                    "evidence": window[:180].strip(),
                })
                found = True
                break
            if found:
                continue
        return out


# ----------------------------------------------------------------------------
# 6.  LATENT MoE SCORER
# ----------------------------------------------------------------------------
@dataclass
class Candidate:
    param: str
    value_si: float
    raw_value: float
    raw_unit: str
    score: float
    confidence: float
    material: str
    temp_k: Optional[float]
    strain_rate: Optional[float]
    method: str
    source_file: str
    source_title: str
    evidence: str
    clamped: bool = False

    def to_display(self) -> Dict[str, Any]:
        spec = PLASTICITY_ONTOLOGY[self.param]
        return {
            "value": self.value_si / spec["ui_scale"],
            "unit": spec["ui_unit"],
            "score": round(self.score, 3),
            "confidence": round(self.confidence, 3),
            "material": self.material or "n/a",
            "temp_K": self.temp_k,
            "method": self.method,
            "source": f"{self.source_file} — {self.source_title[:60]}",
            "clamped": self.clamped,
            "evidence": self.evidence[:140],
        }


class LatentMoEScorer:
    """Five‑expert attention aggregator."""

    def __init__(
        self,
        w_material: float = 0.45,
        w_thermal: float = 0.25,
        w_strain: float = 0.10,
        w_method: float = 0.10,
        w_confidence: float = 0.10,
        thermal_sigma: float = 100.0,
    ):
        self.w_material = w_material
        self.w_thermal = w_thermal
        self.w_strain = w_strain
        self.w_method = w_method
        self.w_confidence = w_confidence
        self.thermal_sigma = thermal_sigma

    def _material_expert(self, ext_mat: str, target_mat: str) -> float:
        if not ext_mat:
            return 0.4
        a, b = ext_mat.lower().strip(), target_mat.lower().strip()
        if a == b:
            return 1.0
        if a in b or b in a:
            return 0.85
        fcc = {"cu", "al", "ni", "ag", "au", "pt", "pb", "cocrfeni", "hea"}
        bcc = {"fe", "w", "mo", "cr", "v", "nb"}
        hcp = {"mg", "ti", "zn", "co", "zr"}
        for grp in (fcc, bcc, hcp):
            if a in grp and b in grp:
                return 0.5
        return 0.25

    def _thermal_expert(self, ext_temp: Optional[float], target_temp: float) -> float:
        if ext_temp is None:
            return 0.5
        try:
            t = float(ext_temp)
        except (TypeError, ValueError):
            return 0.5
        diff = t - float(target_temp)
        return float(np.exp(-(diff ** 2) / (2.0 * self.thermal_sigma ** 2)))

    def _strain_expert(self, ext_rate: Optional[float], target_rate: float) -> float:
        if ext_rate is None:
            return 0.5
        try:
            r = float(ext_rate)
            if r <= 0 or target_rate <= 0:
                return 0.5
            ratio = math.log10(r / target_rate)
            return float(np.exp(-(ratio ** 2) / 8.0))
        except (TypeError, ValueError):
            return 0.5

    @staticmethod
    def _method_expert(method: str) -> float:
        return {
            "experiment": 1.0, "review": 0.8,
            "md": 0.6, "molecular dynamics": 0.6,
            "dft": 0.5, "first-principles": 0.5, "ab initio": 0.5,
            "heuristic": 0.3,
        }.get((method or "").lower(), 0.5)

    def score(
        self,
        extractions: List[Dict[str, Any]],
        target_material: str,
        target_temp: float,
        target_strain_rate: float = 1e-3,
        top_k: int = 8,
    ) -> Dict[str, List[Candidate]]:
        buckets: Dict[str, List[Candidate]] = {p: [] for p in PARAM_ORDER}

        for ext in extractions:
            p = ext.get("param")
            if p not in buckets:
                continue
            try:
                v_si = _normalize_unit(ext["value"], ext.get("unit", ""), p)
            except Exception:
                continue
            v_si_clamped, was_clamped = _clamp(v_si, p)

            s_mat = self._material_expert(ext.get("material", ""), target_material)
            s_temp = self._thermal_expert(ext.get("temp"), target_temp)
            s_strain = self._strain_expert(ext.get("strain_rate"), target_strain_rate)
            s_method = self._method_expert(ext.get("method", "unknown"))
            s_conf = float(ext.get("confidence", 0.5) or 0.5)

            score = (
                self.w_material * s_mat
                + self.w_thermal * s_temp
                + self.w_strain * s_strain
                + self.w_method * s_method
                + self.w_confidence * s_conf
            )
            buckets[p].append(Candidate(
                param=p,
                value_si=v_si_clamped,
                raw_value=float(ext["value"]),
                raw_unit=str(ext.get("unit", "")),
                score=score,
                confidence=s_conf,
                material=str(ext.get("material", "")),
                temp_k=ext.get("temp"),
                strain_rate=ext.get("strain_rate"),
                method=str(ext.get("method", "unknown")),
                source_file=str(ext.get("_source_file", "")),
                source_title=str(ext.get("_source_title", "")),
                evidence=str(ext.get("evidence", "")),
                clamped=was_clamped,
            ))

        for p in buckets:
            buckets[p].sort(key=lambda c: c.score, reverse=True)
            buckets[p] = buckets[p][:top_k]
        return buckets


# ----------------------------------------------------------------------------
# 7.  LEARNED PER‑MATERIAL PRIORS
# ----------------------------------------------------------------------------
class MaterialPriorLearner:
    """Builds a per-(material, parameter) table directly from the corpus."""

    def __init__(self, extractor: ParameterExtractor, scorer: LatentMoEScorer):
        self.extractor = extractor
        self.scorer = scorer

    def learn(
        self,
        corpus: List[Dict[str, Any]],
        use_llm: bool = False,
        max_docs: int = 200,
    ) -> pd.DataFrame:
        """Return a DataFrame with columns:
        material, param, n, mean, median, std, p10, p90.
        """
        raw: Dict[Tuple[str, str], List[float]] = {}
        for doc in corpus[:max_docs]:
            text = doc["text"]
            # Pass material="?" so the scorer's material filter is neutral;
            # we want whatever material the extractor pulls from the text.
            extractions = self.extractor.extract(
                text, material="?", temp_k=300, strain_rate=1e-3, use_llm=use_llm,
            )
            for ext in extractions:
                mat = (ext.get("material") or "").strip() or "unknown"
                p = ext.get("param")
                if p not in PLASTICITY_ONTOLOGY:
                    continue
                try:
                    v_si = _normalize_unit(ext["value"], ext.get("unit", ""), p)
                except Exception:
                    continue
                v_si, _ = _clamp(v_si, p)
                raw.setdefault((mat, p), []).append(v_si)

        rows: List[Dict[str, Any]] = []
        for (mat, p), values in raw.items():
            if len(values) == 0:
                continue
            arr = np.array(values, dtype=float)
            rows.append({
                "material": mat,
                "param": p,
                "n": int(arr.size),
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "std": float(arr.std(ddof=0)),
                "p10": float(np.percentile(arr, 10)),
                "p90": float(np.percentile(arr, 90)),
            })
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=[
                "material", "param", "n", "mean", "median", "std", "p10", "p90"
            ])
        return df.sort_values(["param", "material"]).reset_index(drop=True)

    @staticmethod
    def suggest_from_priors(
        priors_df: pd.DataFrame, material: str, param: str
    ) -> Optional[float]:
        if priors_df.empty:
            return None
        sub = priors_df[
            (priors_df["material"].str.lower() == material.lower())
            & (priors_df["param"] == param)
        ]
        if sub.empty:
            return None
        # Weight the median more when sample size is small
        row = sub.iloc[0]
        if row["n"] < 3:
            return float(row["median"])
        return float(0.5 * row["median"] + 0.5 * row["mean"])


# ----------------------------------------------------------------------------
# 8.  QUANTITATIVE HISTOGRAM PLOTTER
# ----------------------------------------------------------------------------
def render_candidate_histograms(
    candidates_by_param: Dict[str, List[Candidate]],
    theme: Optional[Dict[str, str]] = None,
    log_scale_params: Optional[set] = None,
) -> None:
    """Render one histogram per parameter (side‑by‑side in a 3×2 grid)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    log_scale_params = log_scale_params or {"rho0", "gamma0_dot"}

    available = [p for p in PARAM_ORDER if candidates_by_param.get(p)]
    if not available:
        st.info("No candidates to plot.")
        return

    n = len(available)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[PLASTICITY_ONTOLOGY[p]["label"] for p in available],
        horizontal_spacing=0.10, vertical_spacing=0.18,
    )

    for idx, p in enumerate(available):
        r = idx // cols + 1
        c = idx % cols + 1
        spec = PLASTICITY_ONTOLOGY[p]
        ui_vals = [cd.value_si / spec["ui_scale"] for cd in candidates_by_param[p]]
        scores = [cd.score for cd in candidates_by_param[p]]
        mats = [cd.material or "n/a" for cd in candidates_by_param[p]]

        hover = [
            f"<b>{spec['symbol']}</b> = {v:.4g} {spec['ui_unit']}<br>"
            f"material: {mm}<br>score: {ss:.3f}"
            for v, mm, ss in zip(ui_vals, mats, scores)
        ]

        fig.add_trace(
            go.Histogram(
                x=ui_vals,
                marker=dict(color="#3b82f6", line=dict(color="#1e3a8a", width=1)),
                name=spec["symbol"],
                hovertemplate="%{x:.4g}<br>count: %{y}<extra></extra>",
                nbinsx=max(4, min(20, len(ui_vals) * 2)),
            ),
            row=r, col=c,
        )
        # Overlay the top‑scoring candidate
        top_idx = int(np.argmax(scores))
        fig.add_trace(
            go.Scatter(
                x=[ui_vals[top_idx]], y=[1],
                mode="markers",
                marker=dict(color="#ef4444", size=12, symbol="star",
                            line=dict(color="white", width=1)),
                name="⭐ Best",
                hovertemplate=hover[top_idx] + "<extra></extra>",
                showlegend=(idx == 0),
            ),
            row=r, col=c,
        )
        # Log scale when the parameter spans orders of magnitude
        if p in log_scale_params and min(ui_vals) > 0:
            fig.update_xaxes(type="log", row=r, col=c)

    fig.update_layout(
        height=300 * rows,
        showlegend=True,
        margin=dict(l=40, r=20, t=60, b=40),
        paper_bgcolor=(theme or {}).get("plotly_paper", "#ffffff"),
        plot_bgcolor=(theme or {}).get("plotly_bg", "#f8f9fa"),
        font=dict(color=(theme or {}).get("font", "#1e293b")),
        bargap=0.08,
    )
    st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------------------------
# 9.  RECOMMENDER ORCHESTRATOR (with FAISS + priors + histograms)
# ----------------------------------------------------------------------------
@dataclass
class RecommendationBundle:
    material: str
    temp_k: float
    strain_rate: float
    candidates: Dict[str, List[Candidate]]
    defaults: Dict[str, float]
    priors_df: pd.DataFrame
    retrieval_backend: str
    llm_used: bool
    timestamp: float = field(default_factory=time.time)

    def best(self, param: str) -> Optional[Candidate]:
        lst = self.candidates.get(param, [])
        return lst[0] if lst else None

    def prior_suggestion(self, param: str) -> Optional[float]:
        return MaterialPriorLearner.suggest_from_priors(
            self.priors_df, self.material, param
        )


class PlasticityRecommender:
    """End‑to‑end orchestrator: FAISS retrieve → LLM extract → LatentMoE rank
    → learned priors → histogram plot."""

    CACHE_DIR = ".plasticity_cache"

    def __init__(
        self,
        db_dir: str = "json_metadatabase",
        ollama_model: str = "qwen2.5:7b",
        use_llm: bool = True,
        top_k_retrieval: int = 20,
    ):
        self.corpus = PlasticityCorpus(db_dir)
        self.client = OllamaClient(model=ollama_model)
        self.llm_available = use_llm and OllamaClient.is_available()
        self.retriever = FAISSRetriever()
        self.extractor = ParameterExtractor(
            self.client if self.llm_available else None,
            cache=self._load_disk_cache(),
        )
        self.scorer = LatentMoEScorer()
        self.prior_learner = MaterialPriorLearner(self.extractor, self.scorer)
        self.top_k_retrieval = top_k_retrieval

    # ---- disk cache ---------------------------------------------------------
    def _load_disk_cache(self) -> Dict[str, Any]:
        try:
            os.makedirs(self.CACHE_DIR, exist_ok=True)
            path = os.path.join(self.CACHE_DIR, "llm_cache.json")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning("Cache load failed: %s", e)
        return {}

    def _save_disk_cache(self) -> None:
        try:
            os.makedirs(self.CACHE_DIR, exist_ok=True)
            path = os.path.join(self.CACHE_DIR, "llm_cache.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.extractor.cache, f)
        except Exception as e:
            logger.warning("Cache save failed: %s", e)

    # ---- main entry ---------------------------------------------------------
    def recommend(
        self,
        material: str,
        temp_k: float,
        strain_rate: float = 1e-3,
        max_docs: int = 20,
        build_priors: bool = True,
        progress_callback=None,
    ) -> RecommendationBundle:
        corpus = self.corpus.load()
        if not corpus:
            st.warning("⚠️ No JSON metadatabases found. Using ontology defaults.")
            return self._default_bundle(material, temp_k, strain_rate)

        # ---------- (a) retrieval layer ----------
        retrieval_backend = "keyword-prefilter"
        if not self.retriever._docs:
            try:
                self.retriever.build(corpus, force=False)
                if self.retriever._docs:
                    retrieval_backend = (
                        "faiss+dense" if _FAISS_OK and self.retriever._index is not None
                        else "numpy+dense" if self.retriever.model is not None
                        else "tfidf-fallback"
                    )
            except Exception as e:
                logger.warning("Retrieval build failed: %s", e)

        if self.retriever._docs:
            query = (
                f"{material} plasticity parameters: dislocation density, "
                f"shear modulus, reference strain rate, strain-rate sensitivity, "
                f"friction stress; temperature ~{temp_k} K."
            )
            docs = self.retriever.search(
                query, k=max(self.top_k_retrieval, max_docs),
                material_hint=material,
            )
            docs = docs[:max_docs]
        else:
            docs = PlasticityCorpus.keyword_prefilter(corpus, material, k=max_docs)

        # ---------- (b) LLM / heuristic extraction ----------
        extractions: List[Dict[str, Any]] = []
        for i, doc in enumerate(docs):
            if progress_callback:
                progress_callback(i + 1, len(docs), doc.get("title", "")[:60])
            ext = self.extractor.extract(
                doc["text"], material, temp_k, strain_rate,
                use_llm=self.llm_available,
            )
            for e in ext:
                e["_source_file"] = doc["source"]
                e["_source_title"] = doc["title"]
            extractions.extend(ext)
        self._save_disk_cache()

        # ---------- (c) LatentMoE ranking ----------
        candidates = self.scorer.score(
            extractions, material, temp_k,
            target_strain_rate=strain_rate, top_k=8,
        )

        # ---------- (d) learned per‑material priors ----------
        priors_df = pd.DataFrame()
        if build_priors:
            try:
                priors_df = self.prior_learner.learn(
                    corpus, use_llm=False, max_docs=min(150, len(corpus)),
                )
            except Exception as e:
                logger.warning("Prior learning failed: %s", e)

        defaults = {
            p: PLASTICITY_ONTOLOGY[p]["defaults"].get(
                material, PLASTICITY_ONTOLOGY[p]["defaults"]["Cu"]
            )
            for p in PARAM_ORDER
        }
        return RecommendationBundle(
            material=material, temp_k=temp_k, strain_rate=strain_rate,
            candidates=candidates, defaults=defaults,
            priors_df=priors_df,
            retrieval_backend=retrieval_backend,
            llm_used=self.llm_available,
        )

    @staticmethod
    def _default_bundle(material, temp_k, strain_rate) -> RecommendationBundle:
        defaults = {
            p: PLASTICITY_ONTOLOGY[p]["defaults"].get(
                material, PLASTICITY_ONTOLOGY[p]["defaults"]["Cu"]
            )
            for p in PARAM_ORDER
        }
        return RecommendationBundle(
            material=material, temp_k=temp_k, strain_rate=strain_rate,
            candidates={p: [] for p in PARAM_ORDER}, defaults=defaults,
            priors_df=pd.DataFrame(),
            retrieval_backend="none", llm_used=False,
        )


# ----------------------------------------------------------------------------
# 10.  STREAMLIT SIDEBAR UI
# ----------------------------------------------------------------------------
_SS = "pl_rec_"


def _ss_get(key: str, default=None):
    return st.session_state.get(_SS + key, default)


def _ss_set(key: str, value):
    st.session_state[_SS + key] = value


def _reset_pl_state():
    for p in PARAM_ORDER:
        st.session_state.pop(f"{_SS}{p}_choice", None)
        st.session_state.pop(f"{_SS}{p}_manual", None)
        st.session_state.pop(f"{_SS}{p}_value_si", None)
    st.session_state.pop(f"{_SS}bundle", None)
    st.session_state.pop("plasticity_overrides", None)


def _render_parameter_selector(param: str, bundle: RecommendationBundle):
    spec = PLASTICITY_ONTOLOGY[param]
    st.markdown(f"**{spec['symbol']} — {spec['label']}**")

    options: List[Tuple[str, float, Optional[Candidate]]] = []

    best = bundle.best(param)
    if best is not None:
        options.append((
            f"⭐ Recommended: {_fmt(param, best.value_si)} "
            f"(score {best.score:.2f})",
            best.value_si, best,
        ))

    for i, c in enumerate(bundle.candidates.get(param, [])[1:], start=1):
        options.append((
            f"   Alt {i}: {_fmt(param, c.value_si)} (score {c.score:.2f})",
            c.value_si, c,
        ))

    prior = bundle.prior_suggestion(param)
    if prior is not None and (best is None or abs(prior - best.value_si) > 1e-12):
        options.append((
            f"📚 Learned prior ({bundle.material}): {_fmt(param, prior)}",
            prior, None,
        ))

    default_si = bundle.defaults[param]
    options.append((
        f"⚙️  Default ({bundle.material}): {_fmt(param, default_si)}",
        default_si, None,
    ))
    options.append(("✏️  Manual override…", float("nan"), None))

    labels = [o[0] for o in options]
    prev = _ss_get(f"{param}_choice", labels[0])
    if prev not in labels:
        prev = labels[0]
    idx = labels.index(prev)

    choice = st.radio(
        label=f"Select {param}",
        options=labels,
        index=idx,
        key=f"{_SS}{param}_radio",
        label_visibility="collapsed",
    )
    _ss_set(f"{param}_choice", choice)

    sel = labels.index(choice)
    _, value_si, chosen_cand = options[sel]

    if math.isnan(value_si):
        ui_default = default_si / spec["ui_scale"]
        manual = st.number_input(
            f"Manual {param} ({spec['ui_unit']})",
            value=float(_ss_get(f"{param}_manual", ui_default)),
            format="%.6g" if param in ("rho0", "gamma0_dot") else "%.4f",
            key=f"{_SS}{param}_manual_input",
        )
        _ss_set(f"{param}_manual", manual)
        value_si = manual * spec["ui_scale"]
        chosen_cand = None

    # Provenance
    if chosen_cand is not None:
        st.caption(
            f"📚 {chosen_cand.source_file} — {chosen_cand.source_title[:70]} | "
            f"mat={chosen_cand.material or 'n/a'}, T={chosen_cand.temp_k}, "
            f"method={chosen_cand.method}, conf={chosen_cand.confidence:.2f}"
            + (" | ⚠️ clamped" if chosen_cand.clamped else "")
        )
        if chosen_cand.evidence:
            with st.expander("Evidence snippet"):
                st.code(chosen_cand.evidence, language="text")

    st.session_state[f"{_SS}{param}_value_si"] = value_si
    st.markdown("---")


def render_plasticity_recommender_sidebar(
    default_material: str = "Cu",
    default_temp: float = 300.0,
    default_strain_rate: float = 1e-3,
    ollama_model: str = "qwen2.5:7b",
):
    """Full sidebar with retrieval + LatentMoE + priors + histograms."""
    st.subheader("🤖 Intelligent Plasticity Recommender v8")
    st.caption(
        "FAISS + SentenceTransformer retrieval · Ollama NER · LatentMoE scoring · "
        "learned per‑material priors · candidate histograms."
    )

    col1, col2 = st.columns(2)
    with col1:
        material = st.text_input(
            "Target material", value=default_material, key=f"{_SS}material"
        )
    with col2:
        temp_k = st.number_input(
            "Temperature (K)", value=float(default_temp),
            min_value=1.0, step=10.0, key=f"{_SS}temp",
        )
    strain_rate = st.number_input(
        "Reference strain rate (s⁻¹)",
        value=float(default_strain_rate), format="%.2e",
        key=f"{_SS}rate",
    )
    ollama_model = st.text_input(
        "Ollama model", value=ollama_model, key=f"{_SS}ollama_model"
    )

    # Availability badges
    llm_ok = OllamaClient.is_available()
    backend_txt = (
        "faiss+dense" if _FAISS_OK and _SBERT_OK
        else "numpy+dense" if _SBERT_OK
        else "tfidf-fallback"
    )
    badge_col1, badge_col2 = st.columns(2)
    with badge_col1:
        st.caption(f"{'✅' if llm_ok else '⚠️'} Ollama "
                   f"{'available' if llm_ok else 'unreachable'}")
    with badge_col2:
        st.caption(f"🔎 Retrieval: `{backend_txt}`")

    btn_col1, btn_col2, btn_col3 = st.columns(3)
    with btn_col1:
        run_btn = st.button("🔍 Analyse JSON databases",
                            use_container_width=True, type="primary")
    with btn_col2:
        refresh_btn = st.button("🔄 Force reload corpus",
                                use_container_width=True)
    with btn_col3:
        if st.button("♻️ Reset recommendations",
                     use_container_width=True):
            _reset_pl_state()
            st.rerun()

    if refresh_btn:
        st.session_state.pop(f"pl_corpus_{PlasticityCorpus._CACHE_VERSION}", None)
        try:
            idx = os.path.join(FAISSRetriever.CACHE_DIR, FAISSRetriever.INDEX_FILE)
            if os.path.exists(idx):
                os.remove(idx)
        except Exception:
            pass
        st.success("Corpus cache cleared.")

    if run_btn:
        recommender = PlasticityRecommender(
            ollama_model=ollama_model, use_llm=llm_ok,
        )
        progress = st.progress(0.0)
        status = st.empty()

        def _cb(i, n, title):
            progress.progress(i / max(n, 1))
            status.caption(f"Scanning {i}/{n}: {title}…")

        try:
            with st.spinner("Retrieving candidates + running LatentMoE…"):
                bundle = recommender.recommend(
                    material=material, temp_k=float(temp_k),
                    strain_rate=float(strain_rate),
                    progress_callback=_cb,
                )
            _ss_set("bundle", bundle)
            progress.empty()
            status.success(
                f"✅ Retrieved {sum(len(v) for v in bundle.candidates.values())} "
                f"candidates across {sum(1 for v in bundle.candidates.values() if v)} params."
            )
        except Exception as e:
            progress.empty()
            status.error(f"Recommendation failed: {e}")
            st.exception(e)

    bundle: Optional[RecommendationBundle] = _ss_get("bundle")
    if bundle is None:
        return

    st.caption(
        f"Retrieval: **{bundle.retrieval_backend}** · "
        f"LLM: **{'yes' if bundle.llm_used else 'no (heuristic)'}**"
    )

    # ---------- Per‑parameter selection (one by one) ----------
    st.markdown("### Choose values (one by one)")
    for param in PARAM_ORDER:
        _render_parameter_selector(param, bundle)

    # ---------- Apply button ----------
    if st.button("✅ Apply selected values to solver",
                 type="primary", use_container_width=True):
        overrides = {}
        for param in PARAM_ORDER:
            v = st.session_state.get(f"{_SS}{param}_value_si")
            if v is not None:
                overrides[SOLVER_KEY_MAP[param]] = float(v)
        st.session_state["plasticity_overrides"] = overrides
        st.success(f"Applied {len(overrides)} parameters. "
                   "The solver will use them on the next run.")

    # ---------- (b) Histograms ----------
    st.markdown("### 📊 Candidate Distributions")
    render_candidate_histograms(bundle.candidates)

    # ---------- (c) Learned per‑material prior table ----------
    if not bundle.priors_df.empty:
        with st.expander("📚 Learned per‑material priors (from corpus)",
                         expanded=False):
            st.caption(
                "Aggregated from all heuristic extractions in the corpus. "
                "Use this to sanity‑check values for materials not directly "
                "represented in the retrieved documents."
            )
            styled = bundle.priors_df.copy()
            for col in ["mean", "median", "std", "p10", "p90"]:
                if col in styled.columns:
                    styled[col] = styled.apply(
                        lambda r, c=col: f"{r[c]:.3e}", axis=1
                    )
            st.dataframe(styled, use_container_width=True, hide_index=True)

    # ---------- Full audit ----------
    with st.expander("📋 Full candidate audit (LatentMoE ranking)"):
        for param in PARAM_ORDER:
            st.markdown(f"**{PLASTICITY_ONTOLOGY[param]['label']}**")
            cands = bundle.candidates.get(param, [])
            if not cands:
                st.caption("No candidates found — using default.")
                continue
            rows = [c.to_display() for c in cands]
            st.dataframe(pd.DataFrame(rows), use_container_width=True,
                         hide_index=True)


# ----------------------------------------------------------------------------
# 11.  SOLVER HOOK
# ----------------------------------------------------------------------------
def apply_plasticity_overrides(solver) -> None:
    """Merge user‑chosen plasticity values into solver.mat_props['plasticity']."""
    try:
        overrides = st.session_state.get("plasticity_overrides", None)
        if not overrides:
            return
        solver.mat_props["plasticity"].update(overrides)
        logger.info("Applied plasticity overrides: %s", overrides)
    except Exception as e:
        logger.warning("Could not apply plasticity overrides: %s", e)
