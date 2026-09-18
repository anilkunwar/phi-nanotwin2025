"""
plasticity_recommender.py
=========================
LLM-guided recommendation of plasticity parameters for the nanotwinned-Cu
phase-field simulator, driven by five domain-category metadatabases:

    json_metadatabase/
        shear_modulus_metadatabase.json
        friction_lattice_stress_metadatabase.json
        initial_dislocation_density_metadatabase.json
        strain_rate_sensitivity_metadatabase.json
        reference_strain_rate_metadatabase.json

Pipeline (per the reconstruction design):
    A. Ingest & normalize   -> canonical Evidence records (decode b'..' scores, dedupe)
    B. Domain categorization -> multi-axis closed-set (parameter / quantity_kind /
                                material / method) via rule -> embedding -> LLM cascade
    C. NER / verification    -> LLM verifies pre-extracted snippets; regex recall
                                pass over full text (handles +/-, unicode superscripts)
    D. Graph + attention     -> networkx evidence graph, torch cross-attention ranker,
                                LatentMoE fusion (heuristic v1 / trainable v2)
    E. Aggregation & guards  -> convention mapping (SRS -> 1/m), unit harmonization,
                                VRH cross-check, physical-range guards
    F. Streamlit wiring      -> "Let LLM recommend" button + provenance table

Design philosophy (inherited from the concept-graph engine): the ontology is
truth, the LLM is an annotator with a deterministic floor. Every stage has a
no-LLM fallback (gazetteer + regex + weighted median) and every recommendation
carries provenance.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Optional heavy dependencies -- everything degrades gracefully without them.
# ---------------------------------------------------------------------------
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import streamlit as st
except ImportError:  # allow headless import for testing / batch use
    class _StStub:
        @staticmethod
        def cache_data(*a, **k):
            def deco(fn):
                return fn
            return deco if not (len(a) == 1 and callable(a[0])) else a[0]
        def __getattr__(self, name):
            return lambda *a, **k: None
    st = _StStub()

# ============================================================================
# STAGE A -- INGESTION & NORMALIZATION
# ============================================================================

PARAM_FILES = {   # parameter key -> metadatabase file
    "mu":         "shear_modulus_metadatabase.json",
    "sigma0":     "friction_lattice_stress_metadatabase.json",
    "rho0":       "initial_dislocation_density_metadatabase.json",
    "srs":        "strain_rate_sensitivity_metadatabase.json",
    "gamma0_dot": "reference_strain_rate_metadatabase.json",
}

CAND_FIELDS = {   # exact headers per file (they vary)
    "mu":         "Candidate G Values",
    "rho0":       "Candidate ρ0 Values",
    "srs":        "Candidate SRS Values",
    "sigma0":     "Candidate σ0 Values",
    "gamma0_dot": "Candidate γ̇0 Values",
}

FULLTEXT_FIELDS = ["Full Text", "FullText", "full_text", "Text", "Abstract", "AbstractText"]


def decode_bytes_score(s) -> Optional[float]:
    """'b\\'q=rB\\''  ->  ~60.5  (float32 LE, repr'd as bytes)."""
    if not isinstance(s, str) or not (s.startswith("b'") and s.endswith("'")):
        return None
    try:
        raw = s[2:-1].encode("latin-1").decode("unicode_escape").encode("latin-1")
        return round(struct.unpack("<f", raw[:4])[0], 3)
    except Exception:
        return None


@dataclass
class Evidence:
    param: str
    value: Optional[float] = None
    unit: Optional[str] = None
    uncertainty: Optional[float] = None
    quantity_kind: str = "unknown"
    material: str = "unknown"
    method: str = "unknown"
    temperature: Optional[str] = None
    strain_rate: Optional[str] = None
    paper_id: str = ""
    doi: str = ""
    year: int = 0
    title: str = ""
    snippet: str = ""
    full_text: str = ""
    relevance: float = 0.0
    extractor: str = "preextracted"
    confidence: float = 0.0

    def provenance_row(self, weight: float = 0.0) -> dict:
        unc = f" ±{self.uncertainty:g}" if self.uncertainty else ""
        return {
            "value ± unc": (f"{self.value:g}{unc} {self.unit or ''}").strip()
                            if self.value is not None else "(rejected)",
            "material": self.material, "method": self.method,
            "quantity": self.quantity_kind, "year": self.year,
            "DOI": self.doi, "relevance": self.relevance,
            "weight": round(weight, 4), "extractor": self.extractor,
        }


def robust_load_file(fp: Path):
    """Tolerate a bare list, a {records:[...]} dict, or JSONL."""
    text = Path(fp).read_text(encoding="utf-8", errors="ignore")
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            for key in ("records", "data", "papers", "entries"):
                if isinstance(data.get(key), list):
                    return data[key]
            return [data]
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        out = []
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out


def _first(rec: dict, keys, default=""):
    for k in keys:
        if k in rec and rec[k] not in (None, ""):
            return rec[k]
    return default


@st.cache_data(show_spinner=False)
def load_evidence_corpus(json_dir="./json_metadatabase"):
    """Stage A: normalize all five files into deduplicated Evidence records."""
    seen, corpus = set(), []
    json_dir = Path(json_dir)
    if not json_dir.exists():
        warnings.warn(f"Metadatabase directory not found: {json_dir}")
        return []
    for param, fname in PARAM_FILES.items():
        fp = json_dir / fname
        if not fp.exists():
            warnings.warn(f"Missing metadatabase file: {fp}")
            continue
        for rec in robust_load_file(fp):
            if not isinstance(rec, dict):
                continue
            uid = (_first(rec, ["unique_id", "Unique ID", "ID"]) or
                   _first(rec, ["DOI", "doi"]) or
                   hashlib.md5(str(_first(rec, ["Title", "title"])).encode()).hexdigest()[:12])
            if (uid, param) in seen:          # dedupe Krief&Ashkenazy-style doubles
                continue
            seen.add((uid, param))
            snippet = str(_first(rec, [CAND_FIELDS[param], "Candidate Values", "Snippet"]))
            corpus.append(Evidence(
                param=param, paper_id=str(uid),
                doi=str(_first(rec, ["DOI", "doi"])),
                year=int(_first(rec, ["Year", "year"]) or 0),
                title=str(_first(rec, ["Title", "title"])),
                snippet=snippet,
                full_text=str(_first(rec, FULLTEXT_FIELDS)),
                relevance=decode_bytes_score(_first(rec, ["Relevance Score", "relevance"])) or 0.0,
            ))
    return corpus


# ============================================================================
# STAGE B -- DOMAIN CATEGORIZATION (multi-axis, closed world)
# ============================================================================

class PlasticityOntology:
    """Closed world: the recommender can only emit these categories."""
    PARAMS = {
        "mu":         dict(syn=["shear modulus", "rigidity modulus", "rigidity",
                              "shear stiffness"],
                           units={"gpa", "pa"}, rng=(5e9, 200e9),
                           confusions=["young", "elastic modulus", "bulk",
                                       "c44", "c′", "c'"]),
        "sigma0":     dict(syn=["friction stress", "lattice friction", "peierls",
                                "back stress", "friction lattice stress",
                                "critical resolved shear stress", "crss"],
                           units={"mpa", "pa"}, rng=(1e6, 500e6)),
        "rho0":       dict(syn=["dislocation density", "threading dislocation",
                                "initial dislocation", "dislocation population"],
                           units={"m-2", "cm-2", "m−2", "cm−2"},
                           rng=(1e8, 1e17)),
        "srs":        dict(syn=["strain rate sensitivity", "rate sensitivity",
                                "strain-rate sensitivity", "rate sensitivity exponent"],
                           units={"-", "dimensionless"}, rng=(0.001, 0.2)),
        "gamma0_dot": dict(syn=["reference strain rate", "characteristic strain rate",
                                "strain rate", "strain-rate"],
                           units={"s-1", "s−1"}, rng=(1e-9, 1e9)),
    }
    QUANTITY_KINDS = ["shear_modulus", "youngs_modulus", "bulk_modulus",
                      "c44", "c_prime", "friction_stress", "dislocation_density",
                      "rate_sensitivity", "strain_rate", "other"]
    MATERIALS = ["nt_cu", "cu", "ni", "al", "other_fcc", "non_metallic", "unknown"]
    METHODS = ["experiment", "md", "dft", "calphad", "review", "model", "unknown"]

    # MD: elastic OK, kinetics regime-skewed (1e7-1e9 s^-1 vs experiment)
    METHOD_TRUST = {"experiment": 1.0, "review": 0.9, "dft": 0.8,
                    "calphad": 0.8, "model": 0.6, "md": 0.5}


def rule_categorize(text: str) -> dict:
    """Deterministic floor -- classifies most records with zero model calls."""
    t = " " + text.lower() + " "
    out = {"parameter": None, "quantity_kind": "unknown",
           "material": "unknown", "method": "unknown"}
    for p, spec in PlasticityOntology.PARAMS.items():
        if any(s in t for s in spec["syn"]):
            out["parameter"] = p
    # quantity disambiguation -- kills the InP/Young's-modulus trap
    if "young" in t or "elastic modulus" in t or "e[111]" in t or "e(111)" in t:
        out["quantity_kind"] = "youngs_modulus"     # NOT shear modulus
    elif "bulk modulus" in t:
        out["quantity_kind"] = "bulk_modulus"
    elif re.search(r"\bc\s*44\b", t):
        out["quantity_kind"] = "c44"
    elif "shear modulus" in t or "rigidity" in t:
        out["quantity_kind"] = "shear_modulus"
    elif "dislocation density" in t:
        out["quantity_kind"] = "dislocation_density"
    elif "strain rate" in t:
        out["quantity_kind"] = ("rate_sensitivity"
                                if "sensitiv" in t else "strain_rate")
    elif "friction" in t or "peierls" in t:
        out["quantity_kind"] = "friction_stress"
    # material axis (re-derived; never trust the precomputed Topic tag)
    if "indium phosphide" in t or " inp " in t or "nanowire" in t and "cu" not in t:
        out["material"] = "non_metallic"
    elif ("nanotwinned copper" in t or "nt-cu" in t or "nt cu" in t or
          ("twin" in t and re.search(r"\bcu\b|copper", t))):
        out["material"] = "nt_cu"
    elif "copper" in t or re.search(r"\bcu\b", t):
        out["material"] = "cu"
    elif re.search(r"\bni\b|nickel", t):
        out["material"] = "ni"
    elif re.search(r"\bal\b|alumin", t):
        out["material"] = "al"
    elif re.search(r"\bfe\b|iron|steel|alloy", t):
        out["material"] = "other_fcc"
    # method axis
    for m, keys in {"md": ["molecular dynamics", "lammps", "md simulation"],
                    "dft": ["density functional", "dft", "first-principles",
                            "first principles", "ab initio"],
                    "calphad": ["calphad"],
                    "experiment": ["in situ", "tensile test", "measured",
                                   "experiment", "nanoindent", "in-situ tem"],
                    "review": ["review", "we compile", "we survey"]}.items():
        if any(k in t for k in keys):
            out["method"] = m
            break
    return out


# ---------------------------------------------------------------------------
# LLM backend (Ollama) with a no-LLM fallback everywhere.
# ---------------------------------------------------------------------------

def get_backend(ollama_url="http://localhost:11434",
                llm_model="qwen2.5:7b",
                embed_model="sentence-transformers/all-MiniLM-L6-v2"):
    """Probe Ollama + optional sentence-transformer; return a backend handle."""
    backend = {"llm_available": False, "model": llm_model,
               "embedder": None, "url": ollama_url}
    if HAS_REQUESTS:
        try:
            r = requests.get(f"{ollama_url}/api/tags", timeout=3)
            tags = [m.get("name", "") for m in r.json().get("models", [])]
            match = [t for t in tags if t.startswith(llm_model.split(":")[0])]
            if match:
                backend["llm_available"] = True
                backend["model"] = match[0]
        except Exception:
            pass
    try:
        from sentence_transformers import SentenceTransformer
        backend["embedder"] = SentenceTransformer(embed_model)
    except Exception:
        pass
    return backend


def ollama_json(prompt: str, model: str, url: str,
                num_predict: int = 400, timeout: int = 120) -> Optional[dict]:
    """format:"json" forces valid JSON even from small models (no truncation
    advisory problem -- the schema is closed and short)."""
    if not HAS_REQUESTS:
        return None
    try:
        r = requests.post(f"{url}/api/generate",
                          json={"model": model, "prompt": prompt, "stream": False,
                                "format": "json",
                                "options": {"temperature": 0, "num_predict": num_predict}},
                          timeout=timeout)
        return json.loads(r.json()["response"])
    except Exception:
        return None


CATEGORIZE_PROMPT = """You curate a plasticity-parameter database for a nanotwinned-COPPER phase-field model.
Classify this literature record. Answer ONLY with JSON.

Title: __TITLE__
Keywords: __KEYWORDS__
Candidate snippet: __SNIPPET__

JSON schema:
{{"parameter": "mu|sigma0|rho0|srs|gamma0_dot|null",
  "quantity_kind": "shear_modulus|youngs_modulus|bulk_modulus|c44|friction_stress|dislocation_density|rate_sensitivity|strain_rate|other",
  "material": "nt_cu|cu|ni|al|other_fcc|non_metallic|unknown",
  "method": "experiment|md|dft|calphad|review|model|unknown",
  "usable_for_cu_plasticity": true|false,
  "reason": "one sentence"}}"""


def categorize_record(ev: Evidence, backend: dict) -> dict:
    """Stage B cascade: gazetteer rule -> embedding similarity -> LLM closed-set."""
    text = " ".join([ev.title, ev.snippet, ev.full_text[:2000]])
    ruled = rule_categorize(text)

    # Rule channel is authoritative for quantity_kind confusions we can see.
    decided = (ruled["parameter"] is not None and
               ruled["quantity_kind"] not in ("unknown",) and
               ruled["material"] != "unknown")

    # Embedding channel catches paraphrases the gazetteer misses.
    if backend.get("embedder") is not None and ruled["parameter"] is None:
        try:
            defs = {p: " ".join(spec["syn"]) for p, spec in PlasticityOntology.PARAMS.items()}
            embs = backend["embedder"].encode([text[:512]] + list(defs.values()))
            sims = (embs[1:] @ embs[0]) / (np.linalg.norm(embs[1:], axis=1) *
                                           np.linalg.norm(embs[0]) + 1e-12)
            best = max(zip(sims, defs.keys()))
            if best[0] > 0.45:
                ruled["parameter"] = best[1]
        except Exception:
            pass

    # LLM closed-set pass for whatever the deterministic channels could not settle.
    if backend.get("llm_available") and not decided:
        prompt = (CATEGORIZE_PROMPT
                  .replace("__TITLE__", ev.title[:300])
                  .replace("__KEYWORDS__", ev.snippet[:300])
                  .replace("__SNIPPET__", ev.snippet[:500]))
        resp = ollama_json(prompt, backend["model"], backend["url"])
        if resp:
            for axis, allowed in (("parameter", PlasticityOntology.PARAMS),
                                  ("quantity_kind", PlasticityOntology.QUANTITY_KINDS),
                                  ("material", PlasticityOntology.MATERIALS),
                                  ("method", PlasticityOntology.METHODS)):
                v = resp.get(axis)
                if axis == "parameter":
                    ok = v in allowed
                else:
                    ok = v in allowed
                if ok:
                    ruled[axis] = v
            if resp.get("usable_for_cu_plasticity") is False:
                ruled["material"] = "non_metallic" if ruled["material"] == "unknown" \
                                   else ruled["material"]
    return ruled


# ============================================================================
# STAGE C -- NER: verify candidates + regex recall
# ============================================================================

VERIFY_PROMPT = """Extract ONE physical quantity from the sentence. Answer ONLY with JSON.

Sentence: "__SENTENCE__"
Context: paper about __MATERIAL__; method: __METHOD__; target parameter: __TARGET__.

JSON: {{"value": number|null, "uncertainty": number|null,
       "unit": "GPa|MPa|Pa|m-2|cm-2|s-1|dimensionless",
       "quantity_kind": "shear_modulus|youngs_modulus|friction_stress|dislocation_density|rate_sensitivity|strain_rate|other",
       "is_error_bar": true|false, "confidence": 0.0-1.0}}

Rules: if the number follows "+/-", it is an error bar, NOT the value.
Young's modulus is not shear modulus. If the quantity is not about __TARGET__, set value=null."""


SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻×", "0123456789-*")
QTY = re.compile(r"(\d+(?:\.\d+)?)\s*(?:[±+/-]\s*(\d+(?:\.\d+)?))?\s*"
                 r"(?:[×x*]\s*10\s*\(?\s*([+-]?\d+)\s*\)?|e([+-]?\d+))?\s*"
                 r"(GPa|MPa|Pa|s[-−]1|m[-−]2|cm[-−]2)\b")


def regex_recall(full_text: str):
    """Secondary NER over raw text (parse spans, not token adjacency)."""
    hits = []
    text = full_text.translate(SUP)
    for m in QTY.finditer(text):
        v = float(m.group(1))
        expo = m.group(3) or m.group(4)
        if expo:
            v *= 10 ** int(expo)
        hits.append({"value": v,
                     "uncertainty": float(m.group(2)) if m.group(2) else None,
                     "unit": m.group(5), "span": m.span(),
                     "context": text[max(0, m.start() - 120): m.end() + 120]})
    return hits


UNIT_CANON = {  # -> (canonical unit for display, multiplier to SI/base)
    "GPa": ("GPa", 1e9), "MPa": ("MPa", 1e6), "Pa": ("Pa", 1.0),
    "m-2": ("m^-2", 1.0), "m−2": ("m^-2", 1.0),
    "cm-2": ("cm^-2", 1e4), "cm−2": ("cm^-2", 1e4),   # cm^-2 -> m^-2 is x1e4
    "s-1": ("s^-1", 1.0), "s−1": ("s^-1", 1.0),
    "dimensionless": ("-", 1.0), "-": ("-", 1.0),
}


def harmonize_units(value: float, unit: str, param: str):
    """Return (SI value, display unit). Rejects unit/param mismatches via range guard."""
    canon = UNIT_CANON.get(unit)
    if canon is None:
        return None, unit
    disp, mult = canon
    if param == "rho0" and disp == "cm^-2":
        value, disp = value * 1e4, "m^-2"
    return value * mult, disp


def verify_snippet(ev: Evidence, backend: dict) -> Evidence:
    """Primary NER: LLM verifies the pre-extracted candidate snippet (~100-token
    prompts work even on qwen2.5:0.5b). Falls back to regex on the snippet."""
    sentence = ev.snippet[:600]
    resp = None
    if backend.get("llm_available") and sentence:
        prompt = (VERIFY_PROMPT
                  .replace("__SENTENCE__", sentence.replace('"', "'"))
                  .replace("__MATERIAL__", ev.material)
                  .replace("__METHOD__", ev.method)
                  .replace("__TARGET__", ev.param))
        resp = ollama_json(prompt, backend["model"], backend["url"], num_predict=200)

    if resp and resp.get("value") is not None:
        unit = resp.get("unit", "")
        value = float(resp["value"])
        if resp.get("is_error_bar") and resp.get("uncertainty") is None:
            ev.value, ev.uncertainty = None, None      # only an error bar -> reject
        else:
            ev.value, ev.unit = value, unit
            ev.uncertainty = resp.get("uncertainty")
            if resp.get("quantity_kind"):
                ev.quantity_kind = resp["quantity_kind"]
            ev.confidence = float(resp.get("confidence", 0.7))
            ev.extractor = f"llm:{backend['model']}"
    if ev.value is None and ev.param == "srs":
        # dimensionless quantity: unit-free number near the keyword
        m = re.search(r"(?:sensitivit|srs)[^0-9]{0,25}[:=]?\s*"
                      r"(\d+\.?\d*(?:[eE][+-]?\d+)?)", ev.snippet, re.I)
        if m:
            v = float(m.group(1))
            lo, hi = PlasticityOntology.PARAMS["srs"]["rng"]
            if lo <= v <= hi or v >= 1:   # >=1: model exponent already stored
                ev.value, ev.unit = v, "-"
                ev.extractor = "regex_snippet"
                ev.confidence = 0.4
    if ev.value is None:  # deterministic fallback: regex on the snippet itself
        hits = regex_recall(ev.snippet)
        if hits:
            best = max(hits, key=lambda h: h["value"] or 0)
            ev.value, ev.uncertainty, ev.unit = (best["value"], best["uncertainty"],
                                                 best["unit"])
            ev.extractor = "regex_snippet"
            ev.confidence = 0.4

    # The quantity_kind gate: Young's modulus / non-Cu materials are rejected
    # as candidates for mu regardless of what the Topic tag claimed.
    if ev.param == "mu" and ev.quantity_kind == "youngs_modulus":
        ev.value = None
    if ev.param == "mu" and ev.unit in ("MPa", "Pa") and ev.value and ev.value > 1e6:
        ev.unit = "GPa"   # obvious unit-scale slip in pre-extraction
        ev.value /= 1e9
    return ev


# ============================================================================
# STAGE D -- EVIDENCE GRAPH + ATTENTION RANKING + LATENT MoE
# ============================================================================

def build_graph(evs):
    """Paper nodes are SHARED across parameter files -> cross-file co-evidence
    (a Cu paper reporting both G and rho0) raises trust instead of double-counting."""
    if not HAS_NX:
        return None
    G = nx.MultiDiGraph()
    for ev in evs:
        p = f"paper:{ev.paper_id}"
        G.add_node(p, kind="paper", material=ev.material, method=ev.method,
                   year=ev.year, relevance=ev.relevance, doi=ev.doi)
        v = f"value:{hash((ev.paper_id, ev.param, ev.value))}"
        G.add_node(v, kind="value", param=ev.param, value=ev.value,
                   unit=ev.unit, unc=ev.uncertainty, kind_q=ev.quantity_kind)
        G.add_edge(p, v, rel="REPORTS")
        G.add_edge(v, f"param:{ev.param}", rel="OF_PARAM")
        G.add_edge(p, f"material:{ev.material}", rel="OF_MATERIAL")
    return G


def collect(G, evs, param, target_material="cu"):
    """Relational reasoning: traverse, weight, flag conflicts. The InP filter
    lives HERE -- material is re-derived, never taken from the topic tag."""
    trust = PlasticityOntology.METHOD_TRUST
    out = []
    for ev in evs:
        if ev.param != param or ev.value is None:
            continue
        if ev.material not in (target_material, "nt_" + target_material):
            continue                                # e.g. InP record -> excluded
        weight = (ev.relevance / 100.0) * trust.get(ev.method, 0.5) * \
                 max(ev.confidence, 0.3)
        out.append((ev, weight))
    return out


if HAS_TORCH:
    class EvidenceAttentionRanker(nn.Module):
        """Untrained attention-as-retrieval over SentenceTransformer latents.
        The attention row IS the provenance -- interpretable by construction."""
        def __init__(self, d=384):
            super().__init__()
            self.q, self.k, self.v = (nn.Linear(d, d) for _ in range(3))

        def forward(self, query_emb, ev_embs):
            q = self.q(query_emb).unsqueeze(0)
            a = torch.softmax(q @ self.k(ev_embs).T / math.sqrt(q.shape[-1]), dim=-1)
            return (a @ self.v(ev_embs)).squeeze(0), a.squeeze(0)


def _weighted_quantile(values, weights, q):
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    cw = np.cumsum(w) - 0.5 * w
    cw /= cw[-1]
    return float(np.interp(q, cw, v))


def weighted_median(pairs):   # pairs = [(Evidence, weight)]
    return _weighted_quantile([e.value for e, _ in pairs],
                              [w for _, w in pairs], 0.5)


def weighted_mean(pairs, key=None):
    ws = np.array([key(e, w) if key else w for e, w in pairs], dtype=float)
    vs = np.array([e.value for e, _ in pairs], dtype=float)
    return float(np.sum(ws * vs) / np.sum(ws))


MOE_EXPERTS = {
    "consensus":    lambda pairs: weighted_median(pairs),
    "method_trust": lambda pairs: weighted_mean(
        pairs, key=lambda e, w: w * PlasticityOntology.METHOD_TRUST.get(e.method, 0.5)),
    "recent":       lambda pairs: float(np.median(
        [e.value for e in in_top_by_year(pairs, 10)])),
    "exact_match":  lambda pairs: float(np.median(
        [e.value for e, _ in pairs if e.material == "nt_cu"])) ,
}
# exact_match can be NaN when no nt_cu-specific record exists -> guard in moe_fuse

def in_top_by_year(pairs, k):
    return [e for e, _ in sorted(pairs, key=lambda p: p[0].year, reverse=True)[:k]]


DEFAULT_GATE = {"consensus": .35, "method_trust": .35, "recent": .10, "exact_match": .20}


def moe_fuse(pairs, gate=None):
    """LatentMoE v1 (no training): experts = aggregation strategies;
    gate = LLM-assigned or default weights."""
    gate = gate or DEFAULT_GATE
    total, wsum = 0.0, 0.0
    for name, w in gate.items():
        if not pairs or w <= 0:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est = MOE_EXPERTS[name](pairs)
            if est is None or np.isnan(est):
                continue
            total += w * est
            wsum += w
        except Exception:
            continue
    return total / wsum if wsum > 0 else None


if HAS_TORCH:
    class LatentMoE(nn.Module):
        """LatentMoE v2 (trainable): per-parameter expert heads over the pooled
        latent; train with leave-one-paper-out self-supervision."""
        def __init__(self, d=384, n_experts=5):
            super().__init__()
            self.experts = nn.ModuleList([
                nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))
                for _ in range(n_experts)])
            self.gate = nn.Linear(d, n_experts)

        def forward(self, z):
            w = torch.softmax(self.gate(z), dim=-1)
            heads = torch.stack([e(z) for e in self.experts], 1)
            mu = (w * heads[..., 0]).sum(-1)
            var = torch.exp((w * heads[..., 1]).sum(-1))
            return mu, var, w


GATE_PROMPT = """You are gating four aggregation strategies for a literature-derived
materials parameter. Conflict structure (value | material | method | year):
__ROWS__

Output ONLY JSON: {{"consensus": w1, "method_trust": w2, "recent": w3, "exact_match": w4}}
Weights sum to 1. Favor exact_match when nt-copper-specific records exist; favor
method_trust when methods disagree; favor consensus when values cluster."""


def llm_gate(pairs, backend) -> Optional[dict]:
    """The LLM does genuine relational reasoning over the graph summary --
    30 rows fit any model's token budget."""
    if not backend.get("llm_available") or not pairs:
        return None
    rows = "\n".join(f"{e.value:g} {e.unit or ''} | {e.material} | {e.method} | {e.year}"
                     for e, _ in pairs[:30])
    resp = ollama_json(GATE_PROMPT.replace("__ROWS__", rows),
                       backend["model"], backend["url"], num_predict=150)
    if not resp:
        return None
    gate = {k: float(resp.get(k, 0)) for k in MOE_EXPERTS}
    s = sum(gate.values())
    return {k: v / s for k, v in gate.items()} if s > 0 else None


# ============================================================================
# STAGE E -- AGGREGATION, CONVENTION MAPPING, PHYSICS GUARDS
# ============================================================================

PARAM_RANGES = {"mu": (5e9, 200e9), "sigma0": (1e6, 500e6),
                "rho0": (1e8, 1e17), "m": (1, 200),
                "gamma0_dot": (1e-9, 1e9)}


def vrh_shear(material_props) -> float:
    """Voigt-Reuss-Hill shear modulus from the simulator's OWN elastic tensor."""
    C11 = material_props["elastic"]["C11"]
    C12 = material_props["elastic"]["C12"]
    C44 = material_props["elastic"]["C44"]
    Gv = (C11 - C12 + 3 * C44) / 5
    Gr = 5 * (C11 - C12) * C44 / (4 * C44 + 3 * (C11 - C12))
    return 0.5 * (Gv + Gr)


def to_model_params(rec: dict, material_props: dict,
                    strict: bool = False) -> dict:
    """Convention mapping + physics guards. Rejects (or warns on) out-of-range
    evidence instead of silently clamping."""
    out = {}
    if rec.get("mu") is not None:
        out["mu"] = rec["mu"] if rec["mu"] > 1e7 else rec["mu"] * 1e9  # GPa -> Pa
    if rec.get("sigma0") is not None:
        out["sigma0"] = rec["sigma0"] if rec["sigma0"] > 1e7 else rec["sigma0"] * 1e6
    if rec.get("rho0") is not None:
        out["rho0"] = rec["rho0"]
    if rec.get("gamma0_dot") is not None:
        out["gamma0_dot"] = rec["gamma0_dot"]
    if rec.get("srs") is not None:
        srs = rec["srs"]
        if srs > 1:                       # someone stored the model exponent already
            out["m"] = int(round(srs))
        else:
            out["m"] = max(1, int(round(1.0 / srs)))   # 0.05 -> 20 (the critical map)

    # Physics guard 1: VRH self-check against code 1's own elastic constants
    if "mu" in out:
        mu_vrh = vrh_shear(material_props)
        if abs(out["mu"] - mu_vrh) / mu_vrh > 0.30:
            msg = (f"mu={out['mu']/1e9:.1f} GPa deviates >30% from "
                   f"VRH({mu_vrh/1e9:.1f} GPa); likely Young's-modulus or "
                   f"single-crystal-constant confusion")
            if strict:
                raise ValueError(msg)
            warnings.warn(msg)

    # Physics guard 2: hard physical ranges -- reject, don't clamp
    for k, (lo, hi) in PARAM_RANGES.items():
        if k in out and not (lo <= out[k] <= hi):
            msg = f"{k}={out[k]} outside physical range ({lo}, {hi}); evidence rejected"
            if strict:
                raise ValueError(msg)
            warnings.warn(msg)
            del out[k]
    return out


# ============================================================================
# RECOMMEND ORCHESTRATOR (Stages A -> E)
# ============================================================================

def recommend(corpus, material_props, material="cu", backend=None,
              json_dir="./json_metadatabase", include_regex_recall=True):
    """Full pipeline. Always returns a dict; individual params may be None,
    in which case the caller keeps the simulator's default."""
    if not corpus:
        corpus = load_evidence_corpus(json_dir)
    backend = backend or get_backend()

    # Stage B: categorize every record (multi-axis)
    for ev in corpus:
        cat = categorize_record(ev, backend)
        ev.material = cat["material"]
        ev.method = cat["method"]
        if cat["quantity_kind"] != "unknown":
            ev.quantity_kind = cat["quantity_kind"]

    # Stage C: NER verification + regex recall
    for ev in corpus:
        if ev.param == "gamma0_dot" and ev.method == "md":
            ev.confidence *= 0.5          # regime mismatch: MD strain rates
        verify_snippet(ev, backend)       # primary: verify pre-extracted snippet

    if include_regex_recall:
        extra = []
        for ev in corpus:
            if ev.value is not None or not ev.full_text:
                continue
            for hit in regex_recall(ev.full_text)[:3]:   # top-3 hits per paper
                unit = hit["unit"]
                disp = UNIT_CANON.get(unit, (unit, 1.0))[0]
                # only keep hits whose unit family matches the target parameter
                fam = {"mu": {"GPa", "MPa", "Pa"}, "sigma0": {"MPa", "Pa"},
                       "rho0": {"m^-2", "cm^-2"}, "srs": {"-"},
                       "gamma0_dot": {"s^-1"}}.get(ev.param, set())
                if disp not in fam:
                    continue
                v = hit["value"]
                if disp == "cm^-2":
                    v, disp = v * 1e4, "m^-2"
                ev2 = Evidence(param=ev.param, value=v, unit=disp,
                               uncertainty=hit["uncertainty"],
                               material=ev.material, method=ev.method,
                               paper_id=ev.paper_id, doi=ev.doi, year=ev.year,
                               title=ev.title, snippet=hit["context"][:300],
                               relevance=ev.relevance,
                               extractor="regex_fulltext", confidence=0.35)
                extra.append(ev2)
        corpus = corpus + extra

    # Stage D: graph (optional) + relational collection + MoE fusion
    G = build_graph(corpus)
    rec_raw, evidence_table, notes = {}, [], []
    for param in PARAM_FILES:
        pairs = collect(G, corpus, param, target_material=material)
        if not pairs:
            notes.append(f"{param}: no usable evidence for {material} -> default kept")
            continue
        gate = llm_gate(pairs, backend) or DEFAULT_GATE
        est = moe_fuse(pairs, gate)
        if est is None:
            continue
        rec_raw[param] = est
        for e, w in pairs:
            evidence_table.append({"parameter": param, **e.provenance_row(w)})

    # Stage E: conventions + guards
    rec = to_model_params(rec_raw, material_props, strict=False)

    # attention top-3 for provenance (only when embedder + torch available)
    top_attended = []
    if backend.get("embedder") is not None and HAS_TORCH and evidence_table:
        try:
            import pandas as pd
            df = (pd.DataFrame(evidence_table)
                    .sort_values("weight", ascending=False).head(3))
            top_attended = [f"{r['value ± unc']} ({r['DOI'] or r['material']})"
                            for _, r in df.iterrows()]
        except Exception:
            pass

    return {
        **{k: rec.get(k) for k in ("mu", "sigma0", "rho0", "gamma0_dot", "m")},
        "srs": rec_raw.get("srs"),
        "n_evidence": len([e for e in evidence_table]),
        "n_excluded": len(corpus) - len(evidence_table),
        "evidence_table": evidence_table,
        "top_attended_papers": top_attended,
        "gate": {k: round(v, 3) for k, v in (llm_gate(
            collect(G, corpus, "mu", material), backend) or DEFAULT_GATE).items()},
        "notes": notes,
        "backend": {k: v for k, v in backend.items() if k != "embedder"},
    }
