# --------------------------------------------------------------
# Strain Rate Sensitivity (m, activation volume) in nanotwinned Cu — HARVESTER
# Pivot from σ₀ (friction/lattice stress) to SRS (m, V*, strain-rate windows)
# ✅ SRS vocabulary + DOMAIN_TERMS union (SRS ∪ friction ∪ material)
# ✅ analyze_srs_relevance() — same schema flags, new semantics
# ✅ extract_candidate_srs() alongside extract_candidate_sigma0()
# ✅ topic column + idempotent migration → multi-campaign separation
# ✅ query_arxiv takes topic as an argument (cache-safe)
# ✅ export_scopus_format filters by topic and emits provenance column
# ✅ Full text stored in SQLite universe DB is the single source of truth
# ✅ Partial-coverage warnings; backfill; BLOB-score repair retained
# ✅ Scopus-format CSV: 22 official columns + 15 appended, exact order, BOM, 1-based
# --------------------------------------------------------------

# Standard library imports
import os
import re
import struct
import sqlite3
import json
import io
import zipfile
import logging
import time
import datetime
from datetime import datetime
import tempfile
import hashlib
import gc
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Third-party scientific and data libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# PDF handling
import fitz  # PyMuPDF

# Web and API clients
import requests
import arxiv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Concurrency and system utilities
import concurrent.futures
import psutil

# Machine Learning and NLP
from transformers import AutoTokenizer, AutoModel
import torch

# Retry and robustness decorators
from tenacity import retry, stop_after_attempt, wait_fixed, wait_exponential

# Numerical acceleration
from numba import njit

# Streamlit for UI
import streamlit as st


# ========================= ENVIRONMENT DETECTION =========================
def is_streamlit_cloud():
    if os.getenv("HOME") == "/home/appuser":
        return True
    if "streamlitapp.com" in os.getenv("HOSTNAME", ""):
        return True
    if os.getenv("IS_STREAMLIT_CLOUD", "false").lower() == "true":
        return True
    return False


IS_CLOUD = is_streamlit_cloud()


# ========================= CAMPAIGN IDENTIFIER =========================
# Every paper row carries a 'topic' so σ₀ and SRS results share one DB without
# contaminating each other's CSV exports. Change this to switch campaigns.
CURRENT_TOPIC = "srs_ntcu"


# ========================= PAGE CONFIGURATION (MUST BE FIRST) =========================
if "page_config_set" not in st.session_state:
    st.set_page_config(
        page_title="Strain Rate Sensitivity (m) Harvester — Nanotwinned Cu",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.session_state.page_config = {
        "title": "Strain Rate Sensitivity (m) Harvester — Nanotwinned Cu",
        "layout": "wide"
    }
    st.session_state.page_config_set = True


# =================================================================
# -------------------------- DOMAIN-SPECIFIC TERM DEFINITIONS --------------------------
# Campaign target: strain-rate sensitivity (m) and activation volume (V*) of
# nanotwinned Cu and related FCC microstructures. σ₀ (friction/lattice stress)
# vocabulary is retained because σ = σ₀ + σ_th in nt-Cu and the SRS literature
# routinely co-reports both.


# SRS (strain-rate sensitivity) terms — the new primary target vocabulary
SRS_TERMS = {
    'strain rate sensitivity', 'strain-rate sensitivity', 'rate sensitivity',
    'rate sensitivity exponent', 'activation volume', 'activation energy',
    'thermal activation', 'thermally activated', 'strain rate', 'strain-rate',
    'high strain rate', 'dynamic deformation', 'dynamic loading',
    'dislocation velocity', 'phonon drag', 'detwinning',
    'twin boundary migration', 'rate-dependent',
}


# σ₀ vocabulary — retained because the two phenomena are physically linked in nt-Cu
FRICTION_STRESS_TERMS = {
    'peierls stress', 'peierls-nabarro', 'peierls barrier', 'peierls potential',
    'lattice friction', 'friction stress', 'frictional stress', 'lattice resistance',
    'hall-petch', 'hall–petch', 'h-p intercept', 'frictional resistance',
    'critical resolved shear stress', 'crss', 'kink-pair', 'kink pair',
    'kink migration', 'thermal activation', 'activation volume', 'activation energy',
    'dislocation glide', 'flow stress', 'yield stress',
}


# Material / microstructure vocabulary
MATERIAL_TERMS = {
    'copper', 'nanotwinned', 'nanotwins', 'nanotwin', 'twin boundary', 'twin boundaries',
    'coherent twin', 'nanocrystalline', 'ultrafine-grained', 'face-centered cubic',
    'fcc', 'single crystal', 'polycrystalline', 'dislocation', 'molecular dynamics',
    'crystal plasticity', 'molecular statics',
}


# Base material identifiers (name kept for session-state compatibility)
PVDF_TERMS = {'copper', 'nanotwin', 'nanotwinned'}


# Union used anywhere the previous code wrote (FRICTION_STRESS_TERMS | MATERIAL_TERMS)
DOMAIN_TERMS = SRS_TERMS | FRICTION_STRESS_TERMS | MATERIAL_TERMS


# NOTE: DB column names (dopant_present, beta_phase_present, pvdf_present) are deliberately
# preserved so DB schema, store_paper_metadata, and get_paper_info require zero schema changes.
# Semantically they now mean:
#   dopant_present      → SRS terms present  (was: friction-stress terms present)
#   beta_phase_present  → material terms present
#   pvdf_present        → copper/nanotwin present


# -------------------------- DIRECTORY AND DATABASE CONFIGURATION --------------------------
# These DB files host multiple campaigns. The 'topic' column (below) separates
# σ₀ rows (topic='sigma0', the historical default) from SRS rows (topic='srs_ntcu').


if IS_CLOUD:
    DB_DIR = "/tmp"
    st.info("🌐 Running on Streamlit Cloud: Using temporary storage")
else:
    DB_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "friction_stress_data")
    os.makedirs(DB_DIR, exist_ok=True)


METADATA_DB    = os.path.join(DB_DIR, "friction_lattice_stress_metadata.db")
UNIVERSE_DB    = os.path.join(DB_DIR, "friction_lattice_stress_universe.db")
PDF_STORAGE_DB = os.path.join(DB_DIR, "friction_lattice_stress_pdfs.db")

TEMP_DIR = os.path.join(DB_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

log_file = os.path.join(DB_DIR, "friction_lattice_stress_query.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# -------------------------- SESSION STATE DEFAULTS --------------------------


DEFAULT_STATE = {
    "log_buffer": [],
    "processing": False,
    "search_results": None,
    "relevant_papers": None,
    "downloaded_pdfs": {},
    "zip_buffer": None,
    "processing_time": 0.0,
    "db_stats": {},
    "search_session_id": None,
    "temp_files": [],
    "selected_papers": set(),
    "batch_download_index": 0,
}


for key, default_value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# -------------------------- LOGGING UTILITY --------------------------


def update_log(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {message}"
    st.session_state.log_buffer.append(entry)
    if len(st.session_state.log_buffer) > 50:
        st.session_state.log_buffer.pop(0)
    logging.info(message)


# -------------------------- SCIBERT LOADING (CACHED) --------------------------


@st.cache_resource
def load_scibert():
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
    update_log("SciBERT model and tokenizer loaded from cache")
    return tokenizer, model


# -------------------------- EMBEDDING AND SIMILARITY UTILITIES --------------------------


def get_embedding(text: str) -> np.ndarray:
    tokenizer, model = load_scibert()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denominator = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    return np.dot(a, b) / denominator


# -------------------------- TEXT NORMALIZATION (for extractors) --------------------------


def _normalize_flattened_text(t: str) -> str:
    """
    PyMuPDF flattens superscripts, Unicode minus signs and multiplication
    symbols when extracting text. Normalize the most common offenders so that
    regexes targeting 's-1', 'x10-4', 'b3', 'nm3', 'A3' match what the PDF
    actually produces.
    """
    for a, b in {'−': '-', '–': '-', '⁻': '-', '×': 'x', '≈': '=',
                 '³': '3', '²': '2', '¹': '1', 'Å': 'A', 'Å': 'A'}.items():
        t = t.replace(a, b)
    return t


# -------------------------- DOMAIN-SPECIFIC TEXT ANALYSIS --------------------------


def analyze_srs_relevance(text: str) -> Dict[str, Any]:
    """
    Analyze text for presence of SRS, material, and Cu/nanotwin terms.
    Keeps the DB-flag schema (dopant_present / beta_phase_present / pvdf_present)
    so store/get/display paths need zero changes; only the semantics shift.
    """
    text_lower = text.lower()
    has_srs      = any(term in text_lower for term in SRS_TERMS)
    has_material = any(term in text_lower for term in MATERIAL_TERMS)
    has_cu_nt    = ('copper' in text_lower) or ('nanotwin' in text_lower)
    return {
        "dopant_present":     bool(has_srs),        # now = SRS terms present
        "beta_phase_present": bool(has_material),   # unchanged
        "pvdf_present":       bool(has_cu_nt),      # unchanged
    }


# Backwards-compatible aliases (older call sites keep working)
analyze_sigma0_relevance = analyze_srs_relevance
analyze_dopant_beta_relevance = analyze_srs_relevance


# -------------------------- DATABASE MANAGER CLASS --------------------------


class DatabaseManager:
    """
    Manages three SQLite databases:
    - Metadata (paper info, scores, topic)
    - Full-text (extracted text, FTS5 index)
    - PDF storage (BLOBs, deduplicated by hash)
    """

    def __init__(self):
        self.metadata_db = METADATA_DB
        self.universe_db = UNIVERSE_DB
        self.pdf_db = PDF_STORAGE_DB
        self.init_databases()
        update_log("Database manager initialized")

    def init_databases(self):
        # ------------------ Metadata Database ------------------
        conn = sqlite3.connect(self.metadata_db)
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS papers (
            id TEXT PRIMARY KEY,
            arxiv_id TEXT UNIQUE,
            title TEXT NOT NULL,
            authors TEXT,
            year INTEGER,
            categories TEXT,
            abstract TEXT,
            pdf_url TEXT,
            published_date TEXT,
            updated_date TEXT,
            doi TEXT,
            relevance_score REAL,
            matched_terms TEXT,
            download_status TEXT,
            pdf_stored BOOLEAN DEFAULT 0,
            fulltext_stored BOOLEAN DEFAULT 0,
            pdf_size INTEGER,
            download_time TIMESTAMP,
            enhanced_relevance_score REAL,
            dopant_present BOOLEAN,
            beta_phase_present BOOLEAN,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS search_sessions (
            session_id TEXT PRIMARY KEY,
            query TEXT,
            categories TEXT,
            start_year INTEGER,
            end_year INTEGER,
            max_results INTEGER,
            threshold REAL,
            total_found INTEGER,
            relevant_found INTEGER,
            downloaded_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")

        # Migration: add journal_ref column if it doesn't exist (idempotent)
        try:
            c.execute("ALTER TABLE papers ADD COLUMN journal_ref TEXT")
        except sqlite3.OperationalError:
            pass  # column already exists

        # Migration: add topic column if it doesn't exist (idempotent).
        # Default 'sigma0' backfills legacy rows so they stay in the σ₀ campaign;
        # new SRS searches write topic='srs_ntcu' explicitly.
        try:
            c.execute("ALTER TABLE papers ADD COLUMN topic TEXT DEFAULT 'sigma0'")
        except sqlite3.OperationalError:
            pass  # column already exists

        c.execute("CREATE INDEX IF NOT EXISTS idx_year ON papers(year)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_score ON papers(relevance_score)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_enhanced_score ON papers(enhanced_relevance_score)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_status ON papers(download_status)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_topic ON papers(topic)")
        conn.commit()
        conn.close()

        # ------------------ Full-text Database ------------------
        conn = sqlite3.connect(self.universe_db)
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS papers_fulltext (
            paper_id TEXT PRIMARY KEY,
            title TEXT,
            abstract TEXT,
            full_text TEXT,
            text_hash TEXT UNIQUE,
            word_count INTEGER,
            page_count INTEGER,
            extraction_status TEXT,
            extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS extracted_entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id TEXT,
            entity_type TEXT,
            entity_text TEXT,
            context TEXT,
            page_number INTEGER,
            confidence REAL,
            extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (paper_id) REFERENCES papers_fulltext(paper_id)
        )""")
        c.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts
            USING fts5(paper_id, title, abstract, full_text, tokenize='porter')""")
        conn.commit()
        conn.close()

        # ------------------ PDF Storage Database ------------------
        conn = sqlite3.connect(self.pdf_db)
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS pdf_storage (
            paper_id TEXT PRIMARY KEY,
            pdf_data BLOB NOT NULL,
            pdf_hash TEXT UNIQUE,
            original_url TEXT,
            file_size INTEGER,
            page_count INTEGER,
            compression_method TEXT DEFAULT 'none',
            stored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS pdf_chunks (
            chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id TEXT,
            chunk_index INTEGER,
            chunk_data BLOB,
            chunk_hash TEXT,
            stored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (paper_id) REFERENCES pdf_storage(paper_id),
            UNIQUE(paper_id, chunk_index)
        )""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_pdf_hash ON pdf_storage(pdf_hash)")
        conn.commit()
        conn.close()

    def get_db_stats(self) -> Dict[str, Any]:
        stats = {}
        try:
            conn = sqlite3.connect(self.metadata_db)
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM papers")
            stats['total_papers'] = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM papers WHERE pdf_stored = 1")
            stats['pdfs_stored'] = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM papers WHERE fulltext_stored = 1")
            stats['fulltext_stored'] = c.fetchone()[0]
            c.execute("SELECT COUNT(DISTINCT year) FROM papers")
            stats['years_covered'] = c.fetchone()[0]
            # Per-topic counts (safe if topic column absent — falls back silently)
            try:
                c.execute("SELECT topic, COUNT(*) FROM papers GROUP BY topic")
                stats['by_topic'] = {row[0] or 'sigma0': row[1] for row in c.fetchall()}
            except sqlite3.OperationalError:
                stats['by_topic'] = {}
            conn.close()

            conn = sqlite3.connect(self.universe_db)
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM papers_fulltext")
            stats['fulltext_count'] = c.fetchone()[0]
            c.execute("SELECT SUM(word_count) FROM papers_fulltext")
            stats['total_words'] = c.fetchone()[0] or 0
            conn.close()

            conn = sqlite3.connect(self.pdf_db)
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM pdf_storage")
            stats['pdf_storage_count'] = c.fetchone()[0]
            c.execute("SELECT SUM(file_size) FROM pdf_storage")
            total_bytes = c.fetchone()[0] or 0
            stats['total_pdf_size_mb'] = round(total_bytes / (1024 * 1024), 2)
            conn.close()
        except Exception as e:
            update_log(f"Error getting DB stats: {e}")
        return stats

    def store_paper_metadata(self, paper: Dict[str, Any]) -> bool:
        """
        Insert or update paper metadata.
        Preserves existing download state (pdf_stored / fulltext_stored /
        download_status / pdf_size / download_time / enhanced_relevance_score)
        so a fresh search does not wipe previously-downloaded papers.
        Writes 'topic' so campaigns stay separated in the shared DB.
        """
        try:
            conn = sqlite3.connect(self.metadata_db)
            c = conn.cursor()

            # Preserve prior download state if this row was already downloaded
            row = c.execute(
                "SELECT download_status, pdf_stored, fulltext_stored, pdf_size, "
                "download_time, enhanced_relevance_score FROM papers WHERE id=?",
                (paper.get('id'),)
            ).fetchone()
            if row and row[1]:  # truthy pdf_stored → prior download exists
                paper = {
                    **paper,
                    'download_status': row[0],
                    'pdf_stored': row[1],
                    'fulltext_stored': row[2],
                    'pdf_size': row[3],
                    'download_time': row[4],
                    'enhanced_relevance_score':
                        row[5] if row[5] is not None else paper.get('enhanced_relevance_score', 0.0),
                }

            c.execute("""INSERT OR REPLACE INTO papers
                (id, arxiv_id, title, authors, year, categories, abstract,
                pdf_url, published_date, updated_date, doi, journal_ref,
                relevance_score, matched_terms, download_status, pdf_stored,
                fulltext_stored, pdf_size, download_time, enhanced_relevance_score,
                dopant_present, beta_phase_present, topic)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    paper.get('id'), paper.get('arxiv_id'), paper.get('title'),
                    paper.get('authors'), paper.get('year'), paper.get('categories'),
                    paper.get('abstract'), paper.get('pdf_url'), paper.get('published_date'),
                    paper.get('updated_date'), paper.get('doi'), paper.get('journal_ref'),
                    paper.get('relevance_score'), paper.get('matched_terms'),
                    paper.get('download_status'), paper.get('pdf_stored', 0),
                    paper.get('fulltext_stored', 0), paper.get('pdf_size', 0),
                    paper.get('download_time'), paper.get('enhanced_relevance_score'),
                    paper.get('dopant_present'), paper.get('beta_phase_present'),
                    paper.get('topic', 'sigma0')
                ))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            update_log(f"Failed to store metadata for {paper.get('id')}: {e}")
            return False

    def store_pdf_data(self, paper_id: str, pdf_bytes: bytes, pdf_url: str) -> bool:
        try:
            pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()
            file_size = len(pdf_bytes)
            try:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                page_count = len(doc)
                doc.close()
            except:
                page_count = 0
            conn = sqlite3.connect(self.pdf_db)
            c = conn.cursor()
            c.execute("SELECT paper_id FROM pdf_storage WHERE pdf_hash = ?", (pdf_hash,))
            existing = c.fetchone()
            if existing:
                update_log(f"PDF already exists for {paper_id}")
                c.execute("UPDATE papers SET pdf_stored = 1, pdf_size = ? WHERE id = ?",
                          (file_size, paper_id))
            else:
                c.execute("""INSERT OR REPLACE INTO pdf_storage
                    (paper_id, pdf_data, pdf_hash, original_url, file_size, page_count)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (paper_id, sqlite3.Binary(pdf_bytes), pdf_hash, pdf_url, file_size, page_count))
                c.execute("UPDATE papers SET pdf_stored = 1, pdf_size = ? WHERE id = ?",
                          (file_size, paper_id))
            conn.commit()
            conn.close()
            update_log(f"Stored PDF for {paper_id} ({file_size/1024:.1f} KB, {page_count} pages)")
            return True
        except Exception as e:
            update_log(f"Failed to store PDF for {paper_id}: {e}")
            return False

    def store_fulltext(self, paper_id: str, title: str, abstract: str,
                       full_text: str, page_count: int = 0) -> bool:
        try:
            text_hash = hashlib.md5(full_text.encode()).hexdigest()
            word_count = len(full_text.split())
            conn = sqlite3.connect(self.universe_db)
            c = conn.cursor()
            c.execute("""INSERT OR REPLACE INTO papers_fulltext
                (paper_id, title, abstract, full_text, text_hash, word_count, page_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (paper_id, title, abstract, full_text, text_hash, word_count, page_count))
            c.execute("""INSERT OR REPLACE INTO papers_fts
                (paper_id, title, abstract, full_text)
                VALUES (?, ?, ?, ?)""",
                (paper_id, title, abstract, full_text))
            conn.commit()
            conn.close()
            conn = sqlite3.connect(self.metadata_db)
            c = conn.cursor()
            c.execute("UPDATE papers SET fulltext_stored = 1 WHERE id = ?", (paper_id,))
            conn.commit()
            conn.close()
            update_log(f"Stored full text for {paper_id} ({word_count} words)")
            return True
        except Exception as e:
            update_log(f"Failed to store full text for {paper_id}: {e}")
            return False

    def get_pdf(self, paper_id: str) -> Optional[bytes]:
        try:
            conn = sqlite3.connect(self.pdf_db)
            c = conn.cursor()
            c.execute("SELECT pdf_data FROM pdf_storage WHERE paper_id = ?", (paper_id,))
            result = c.fetchone()
            conn.close()
            return result[0] if result else None
        except Exception as e:
            update_log(f"Failed to retrieve PDF for {paper_id}: {e}")
            return None

    def get_paper_info(self, paper_id: str) -> Optional[Dict[str, Any]]:
        try:
            conn = sqlite3.connect(self.metadata_db)
            c = conn.cursor()
            c.execute("""SELECT title, authors, year, abstract, pdf_url,
                relevance_score, enhanced_relevance_score, dopant_present,
                beta_phase_present, download_status, pdf_stored, fulltext_stored,
                journal_ref
                FROM papers WHERE id = ?""", (paper_id,))
            meta = c.fetchone()
            conn.close()
            if not meta:
                return None
            pdf_bytes = self.get_pdf(paper_id)
            conn = sqlite3.connect(self.universe_db)
            c = conn.cursor()
            c.execute("SELECT word_count FROM papers_fulltext WHERE paper_id = ?", (paper_id,))
            fulltext_info = c.fetchone()
            conn.close()
            return {
                'title': meta[0], 'authors': meta[1], 'year': meta[2],
                'abstract': meta[3], 'pdf_url': meta[4],
                'relevance_score': meta[5], 'enhanced_relevance_score': meta[6],
                'dopant_present': meta[7], 'beta_phase_present': meta[8],
                'download_status': meta[9], 'has_pdf': meta[10],
                'has_fulltext': meta[11], 'journal_ref': meta[12],
                'pdf_bytes': pdf_bytes,
                'word_count': fulltext_info[0] if fulltext_info else 0
            }
        except Exception as e:
            update_log(f"Failed to get paper info for {paper_id}: {e}")
            return None

    def create_zip_from_db(self, paper_ids: List[str]) -> io.BytesIO:
        zip_buffer = io.BytesIO()
        try:
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for paper_id in paper_ids:
                    pdf_data = self.get_pdf(paper_id)
                    if pdf_data:  # ✅ was "if pdf_" (syntax error)
                        info = self.get_paper_info(paper_id)
                        if info:
                            title = re.sub(r'[^\w\s-]', '', info['title'])[:100]
                            authors = info['authors'].split(',')[0][:50] if info['authors'] else 'unknown'
                            filename = f"{paper_id}_{authors}_{info['year']}_{title}.pdf"
                            filename = re.sub(r'\s+', '_', filename)
                        else:
                            filename = f"{paper_id}.pdf"
                        zip_file.writestr(filename, pdf_data)
            zip_buffer.seek(0)
            update_log(f"Created ZIP with {len(paper_ids)} PDFs")
        except Exception as e:
            update_log(f"Failed to create ZIP: {e}")
        return zip_buffer

    def export_metadata(self, format: str = "csv",
                        topic: Optional[str] = None) -> io.BytesIO:
        """
        Raw metadata export (CSV/JSON/Excel). Optionally filter by topic so
        an SRS export doesn't include legacy σ₀ rows.
        """
        try:
            conn = sqlite3.connect(self.metadata_db)
            sql = "SELECT * FROM papers"
            params: Tuple = ()
            if topic:
                sql += " WHERE topic = ?"
                params = (topic,)

            if format.lower() == "csv":
                df = pd.read_sql_query(sql, conn, params=params)
                output = io.BytesIO()
                df.to_csv(output, index=False)
                output.seek(0)
            elif format.lower() == "json":
                df = pd.read_sql_query(sql, conn, params=params)
                output = io.BytesIO()
                df.to_json(output, orient="records", indent=2)
                output.seek(0)
            elif format.lower() == "excel":
                df = pd.read_sql_query(sql, conn, params=params)
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Papers')
                output.seek(0)
            else:
                output = io.BytesIO()
            conn.close()
            return output
        except Exception as e:
            update_log(f"Export failed: {e}")
            return io.BytesIO()


db_manager = DatabaseManager()


# -------------------------- PDF DOWNLOAD AND TEXT EXTRACTION --------------------------


def download_pdf_bytes(pdf_url: str) -> Optional[bytes]:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/pdf,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _download():
        session = requests.Session()
        session.mount('https://', HTTPAdapter(max_retries=3))
        response = session.get(pdf_url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.content
    try:
        pdf_bytes = _download()
        if len(pdf_bytes) < 1024:
            raise ValueError("PDF file too small")
        return pdf_bytes
    except Exception as e:
        update_log(f"Download failed for {pdf_url}: {e}")
        return None


def extract_text_from_bytes(pdf_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page_num in range(min(50, len(doc))):
            text += doc[page_num].get_text()
        doc.close()
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:1000000]
    except Exception as e:
        return f"Error extracting text: {str(e)}"


def handle_paper_download(paper: Dict[str, Any], manual_download: bool = False) -> Dict[str, Any]:
    paper_id = paper['id']
    if manual_download and paper_id in st.session_state.downloaded_pdfs:
        update_log(f"PDF for {paper_id} already in session")
        return paper
    try:
        update_log(f"Downloading PDF for {paper_id}...")
        pdf_bytes = download_pdf_bytes(paper['pdf_url'])
        if pdf_bytes is None:
            paper['download_status'] = "Failed to download"
            return paper
        full_text = extract_text_from_bytes(pdf_bytes)
        pdf_stored = db_manager.store_pdf_data(paper_id, pdf_bytes, paper['pdf_url'])
        if not full_text.startswith("Error"):
            analysis = analyze_srs_relevance(full_text)
            paper['dopant_present']     = analysis['dopant_present']
            paper['beta_phase_present'] = analysis['beta_phase_present']
            if 'query_emb' in st.session_state:
                full_emb = get_embedding(full_text)
                sim1 = cosine_sim(st.session_state.query_emb, full_emb)
                sim2 = cosine_sim(st.session_state.key_emb, full_emb)
                paper['enhanced_relevance_score'] = round((0.7 * sim1 + 0.3 * sim2) * 100, 2)
            else:
                paper['enhanced_relevance_score'] = 0.0
            text_stored = db_manager.store_fulltext(paper_id, paper['title'],
                                                    paper['abstract'], full_text)
        else:
            text_stored = False
        paper['pdf_stored'] = 1 if pdf_stored else 0
        paper['fulltext_stored'] = 1 if text_stored else 0
        paper['pdf_size'] = len(pdf_bytes)
        paper['download_time'] = datetime.now().isoformat()
        paper['download_status'] = "Successfully downloaded and stored"
        st.session_state.downloaded_pdfs[paper_id] = {
            'pdf_bytes': pdf_bytes,
            'title': paper['title'],
            'authors': paper['authors'],
            'year': paper['year']
        }
        db_manager.store_paper_metadata(paper)
        update_log(f"✅ Successfully processed {paper_id}")
    except Exception as e:
        paper['download_status'] = f"Failed: {str(e)[:100]}"
        update_log(f"❌ Failed to process {paper_id}: {e}")
    return paper


@st.cache_data(ttl=3600)
def query_arxiv(query: str, categories: List[str], max_results: int,
                start_year: int, end_year: int,
                topic: str = "srs_ntcu") -> List[Dict[str, Any]]:
    """
    Query arXiv for the given search string. The 'topic' argument is part of
    the cache key, so switching campaigns never serves stale results from the
    other topic's cache. Each returned paper dict carries its 'topic' so
    store_paper_metadata writes the correct campaign label.
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=min(max_results * 2, 500),
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending
    )
    results = []
    for result in client.results(search):
        if not (any(cat in result.categories for cat in categories) and
                start_year <= result.published.year <= end_year):
            continue

        paper = {
            'id': result.entry_id.split('/')[-1],
            'arxiv_id': result.entry_id,
            'title': result.title,
            'authors': ', '.join(a.name for a in result.authors),
            'year': result.published.year,
            'categories': ', '.join(result.categories),
            'abstract': result.summary,
            'pdf_url': result.pdf_url,
            'published_date': result.published.isoformat(),
            'updated_date': result.updated.isoformat() if result.updated else result.published.isoformat(),
            'doi': result.doi if hasattr(result, 'doi') and result.doi else None,
            'journal_ref': getattr(result, 'journal_ref', None),
            'relevance_score': 0.0,
            'matched_terms': '',
            'download_status': 'Pending',
            'pdf_stored': 0,
            'fulltext_stored': 0,
            'pdf_size': 0,
            'download_time': None,
            'enhanced_relevance_score': 0.0,
            'dopant_present': False,
            'beta_phase_present': False,
            'topic': topic,
        }

        text_lower = (paper['title'] + ' ' + paper['abstract']).lower()
        paper['matched_terms'] = '; '.join(sorted(
            t for t in DOMAIN_TERMS if t in text_lower
        ))

        results.append(paper)
        if len(results) >= max_results:
            break
    return results[:max_results]


# -------------------------- SCOPUS-FORMAT EXPORT (SRS) --------------------------


SCOPUS_COLUMNS = [
    "Authors", "Author full names", "Author(s) ID", "Title", "Year",
    "Source title", "Volume", "Issue", "Art. No.", "Page start", "Page end",
    "Cited by", "DOI", "Link", "Abstract", "Author Keywords", "Index Keywords",
    "Document Type", "Publication Stage", "Open Access", "Source", "EID",
]


EXTRA_COLUMNS = [
    "Full Text", "arXiv ID", "Categories", "PDF URL", "Journal Ref",
    "Relevance Score", "Enhanced Relevance Score", "Matched Terms",
    "Candidate SRS Values", "Candidate σ0 Values",
    "Download Status", "PDF Stored", "Full Text Stored",
    "PDF Size (KB)", "Word Count", "Topic",
]


def format_scopus_authors_short(authors_str: str) -> str:
    """'Junwan Li, Jifa Mei' -> 'Li J.; Mei J.'   (Scopus 'Authors' style)"""
    out = []
    for name in [a.strip() for a in (authors_str or '').split(',') if a.strip()]:
        parts = name.split()
        if len(parts) == 1:
            out.append(parts[0])
        else:
            surname = parts[-1]
            initials = ''.join(p[0].upper() for p in parts[:-1] if p)
            out.append(f"{surname} {initials}")
    return '; '.join(out)


def format_scopus_authors_full(authors_str: str) -> str:
    """'Junwan Li' -> 'Li, Junwan'   (Scopus 'Author full names' style)"""
    out = []
    for name in [a.strip() for a in (authors_str or '').split(',') if a.strip()]:
        parts = name.split()
        if len(parts) >= 2:
            out.append(f"{parts[-1]}, {' '.join(parts[:-1])}")
        else:
            out.append(name)
    return '; '.join(out)


def parse_journal_ref(jr: str) -> Dict[str, str]:
    """Best-effort parse of arXiv journal_ref, e.g.
    'Scripta Mater. 55 (2006) 319-322'  or  'Phys. Rev. B 80, 174104 (2009)'."""
    out = {"src": "", "vol": "", "pg_start": "", "pg_end": "", "yr": ""}
    if not jr:
        return out
    s = jr.strip()
    ym = re.search(r'(19|20)\d{2}', s)
    if ym:
        out["yr"] = ym.group(0)
    vm = re.match(r'^(?P<src>[A-Za-z][^,;]*?)\s+(?P<vol>\d+)\s*[,(]', s)
    if vm:
        out["src"] = vm.group('src').strip()
        out["vol"] = vm.group('vol')
    pm = re.search(r'(?P<a>\d+)[-–](?P<b>\d+)', s)
    if pm:
        out["pg_start"], out["pg_end"] = pm.group('a'), pm.group('b')
    else:
        am = re.search(r'[,\s](\d{4,6})(?:\s*\(\d{4}\))?$', s)
        if am:
            out["pg_start"] = am.group(1)
    return out


def extract_candidate_sigma0(full_text: str, max_hits: int = 15) -> str:
    """
    Candidate σ₀ snippets: keyword near number+unit.
    A pointer system for downstream NER, not a final extraction.
    """
    if not full_text or full_text.startswith("Error"):
        return ""
    kw = re.compile(
        r'peierls|lattice\s+friction|friction\s+stress|lattice\s+resistance|'
        r'critical\s+resolved\s+shear\s+stress|hall[-–]petch', re.IGNORECASE)
    num_unit = re.compile(
        r'\d+(?:\.\d+)?(?:\s*[x×]\s*10\s*[-−]?\s*\d+)?\s*'
        r'(?:GPa|MPa|kPa|g/mm\s*2|kgf/mm\s*2|N/m\s*2)', re.IGNORECASE)
    hits, seen = [], set()
    for m in kw.finditer(full_text):
        window = re.sub(r'\s+', ' ',
                        full_text[max(0, m.start() - 80):m.end() + 160]).strip()
        vals = num_unit.findall(window)
        if vals and window[:60] not in seen:
            seen.add(window[:60])
            hits.append(f"{window[:180]}  [→ {', '.join(vals[:3])}]")
        if len(hits) >= max_hits:
            break
    return " || ".join(hits)


def extract_candidate_srs(full_text: str, max_hits: int = 15) -> str:
    """
    Candidate SRS snippets: 'strain-rate sensitivity', 'activation volume',
    'thermally activated', etc., near a number that looks like m (0.0xx),
    a strain-rate window (1x10-4 s-1, 0.001 s-1), or an activation volume
    (9 b3, 8.5 nm3, 1.5x10-28 m3).
    Returns '||'-joined snippets; a pointer system, not a final extraction.
    """
    if not full_text or full_text.startswith("Error"):
        return ""
    t = _normalize_flattened_text(full_text)
    kw = re.compile(
        r'strain[- ]?rate\s+sensitivity|rate\s+sensitivity|activation\s+volume|'
        r'thermally\s+activated|thermal\s+activation|rate[- ]sensitivity\s+exponent',
        re.IGNORECASE)
    num_unit = re.compile(
        r'\bm\s*[=:~]?\s*0\.0\d+'                                        # m = 0.025
        r'|\d+(?:\.\d+)?\s*(?:x\s*10|e)\s*[-^]?\s*\d+\s*s\s*[-^]?\s*1'   # 1x10-4 s-1
        r'|\b\d+(?:\.\d+)?\s*s\s*-\s*1'                                  # 0.001 s-1
        r'|\d+(?:\.\d+)?\s*(?:b3|nm3|A3)'                                # 9 b3, 8.5 nm3
        r'|\d+(?:\.\d+)?\s*x\s*10\s*-\s*\d+\s*m3',                       # 1.5x10-28 m3
        re.IGNORECASE)
    hits, seen = [], set()
    for m in kw.finditer(t):
        window = re.sub(r'\s+', ' ', t[max(0, m.start() - 80): m.end() + 160]).strip()
        vals = num_unit.findall(window)
        if vals and window[:60] not in seen:
            seen.add(window[:60])
            hits.append(f"{window[:180]}  [→ {', '.join(v.strip() for v in vals[:3])}]")
        if len(hits) >= max_hits:
            break
    return " || ".join(hits)


def export_scopus_format(include_fulltext: bool = True,
                          fulltext_char_limit: Optional[int] = None,
                          only_with_fulltext: bool = False,
                          require_metadata_flag: bool = False,
                          topic: Optional[str] = None) -> io.BytesIO:
    """
    Build a Scopus-compatible CSV.

    Parameters
    ----------
    include_fulltext : bool
        Emit the appended "Full Text" column populated from papers_fulltext.
    fulltext_char_limit : int | None
        Truncate each paper's full text to this many chars (None = no limit).
    only_with_fulltext : bool
        Skip rows that have no full_text payload in the universe DB.
    require_metadata_flag : bool
        Skip rows where metadata.fulltext_stored is not 1.
    topic : str | None
        Restrict export to a single campaign (e.g. 'srs_ntcu'). None = all rows.
    """
    try:
        conn = sqlite3.connect(db_manager.metadata_db)
        sql = "SELECT * FROM papers"
        params: Tuple = ()
        if topic:
            sql += " WHERE topic = ?"
            params = (topic,)
        df_meta = pd.read_sql_query(sql, conn, params=params)
        conn.close()
        if df_meta.empty:
            return io.BytesIO()

        conn = sqlite3.connect(db_manager.universe_db)
        df_text = pd.read_sql_query(
            "SELECT paper_id, full_text, word_count FROM papers_fulltext", conn)
        conn.close()
        text_map = {r['paper_id']: (r['full_text'], r['word_count'])
                    for _, r in df_text.iterrows()}

        rows = []
        for _, r in df_meta.iterrows():
            full_text, word_count = text_map.get(r['id'], ("", 0))

            if require_metadata_flag and not bool(r.get('fulltext_stored')):
                continue
            if only_with_fulltext and not full_text:
                continue

            ft = (full_text or "")[:fulltext_char_limit] if fulltext_char_limit else (full_text or "")

            jr = parse_journal_ref(r.get('journal_ref'))
            arxiv_short = str(r.get('arxiv_id', '')).split('/abs/')[-1] or str(r.get('id', ''))
            abs_link = f"https://arxiv.org/abs/{arxiv_short}"

            corpus = ((r.get('abstract') or '') + ' ' + (full_text or '')).lower()
            index_kw = '; '.join(sorted(t for t in DOMAIN_TERMS if t in corpus))

            row = {
                # ---------- exact Scopus columns, exact order ----------
                "Authors": format_scopus_authors_short(r.get('authors') or ""),
                "Author full names": format_scopus_authors_full(r.get('authors') or ""),
                "Author(s) ID": "",
                "Title": r.get('title') or "",
                "Year": jr["yr"] or r.get('year') or "",
                "Source title": jr["src"] or "arXiv",
                "Volume": jr["vol"],
                "Issue": "",
                "Art. No.": jr["pg_start"] if jr["pg_start"] and not jr["pg_end"] else "",
                "Page start": jr["pg_start"] if jr["pg_end"] else "",
                "Page end": jr["pg_end"],
                # Semantic Scholar Cited-by intentionally disabled:
                # one HTTPS call per row is not acceptable by default.
                "Cited by": "",
                "DOI": r.get('doi') or "",
                "Link": abs_link,
                "Abstract": r.get('abstract') or "",
                "Author Keywords": "",
                "Index Keywords": index_kw,
                "Document Type": "Article",
                "Publication Stage": "Final",
                "Open Access": "All Open Access",
                "Source": "arXiv",
                "EID": f"arxiv-{arxiv_short}",
                # ---------- appended extra columns ----------
                "Full Text": ft if include_fulltext else "",
                "arXiv ID": arxiv_short,
                "Categories": r.get('categories') or "",
                "PDF URL": r.get('pdf_url') or "",
                "Journal Ref": r.get('journal_ref') or "",
                "Relevance Score": r.get('relevance_score') or 0.0,
                "Enhanced Relevance Score": r.get('enhanced_relevance_score') or 0.0,
                "Matched Terms": r.get('matched_terms') or "",
                "Candidate SRS Values": extract_candidate_srs(full_text),
                "Candidate σ0 Values":  extract_candidate_sigma0(full_text),
                "Download Status": r.get('download_status') or "",
                "PDF Stored": r.get('pdf_stored') or 0,
                "Full Text Stored": r.get('fulltext_stored') or 0,
                "PDF Size (KB)": round((r.get('pdf_size') or 0) / 1024, 1),
                "Word Count": word_count or 0,
                "Topic": r.get('topic') or "sigma0",
            }
            rows.append(row)

        df_out = pd.DataFrame(rows, columns=SCOPUS_COLUMNS + EXTRA_COLUMNS)
        df_out.index = range(1, len(df_out) + 1)          # Scopus-style row-number column
        output = io.BytesIO()
        df_out.to_csv(output, index=True, encoding='utf-8-sig')  # BOM = exact Scopus encoding
        output.seek(0)
        update_log(f"Scopus-format CSV exported: {len(df_out)} rows "
                   f"(topic={topic}, only_with_fulltext={only_with_fulltext}, "
                   f"require_metadata_flag={require_metadata_flag})")
        return output
    except Exception as e:
        update_log(f"Scopus-format export failed: {e}")
        return io.BytesIO()


# -------------------------- MAINTENANCE HELPERS --------------------------


def get_missing_fulltext_ids(topic: Optional[str] = None) -> List[str]:
    """Return IDs of metadata rows (optionally a single topic) that have a
    pdf_url but no stored full text."""
    try:
        conn = sqlite3.connect(db_manager.metadata_db)
        c = conn.cursor()
        sql = ("SELECT id FROM papers "
               "WHERE (fulltext_stored = 0 OR fulltext_stored IS NULL) "
               "AND pdf_url IS NOT NULL")
        params: Tuple = ()
        if topic:
            sql += " AND topic = ?"
            params = (topic,)
        rows = c.execute(sql, params).fetchall()
        conn.close()
        return [r[0] for r in rows]
    except Exception as e:
        update_log(f"get_missing_fulltext_ids failed: {e}")
        return []


def get_fulltext_coverage(topic: Optional[str] = None) -> Dict[str, Any]:
    """
    Aggregate coverage counters used to decide whether the CSV will carry text.
    'with_ft_universe_rows' is the number that actually backs the Full Text column.
    When 'topic' is provided, all counters are restricted to that campaign.
    """
    where = ""
    params: Tuple = ()
    if topic:
        where = " WHERE topic = ?"
        params = (topic,)

    conn = sqlite3.connect(db_manager.metadata_db)
    total = conn.execute(f"SELECT COUNT(*) FROM papers{where}", params).fetchone()[0]
    with_pdf = conn.execute(
        f"SELECT COUNT(*) FROM papers{where}{' AND' if where else ' WHERE'} pdf_stored = 1",
        params).fetchone()[0]
    with_ft_meta = conn.execute(
        f"SELECT COUNT(*) FROM papers{where}{' AND' if where else ' WHERE'} fulltext_stored = 1",
        params).fetchone()[0]
    # IDs of the topic's rows to intersect with the universe DB
    ids = [row[0] for row in conn.execute(
        f"SELECT id FROM papers{where}", params).fetchall()]
    conn.close()

    conn = sqlite3.connect(db_manager.universe_db)
    if ids:
        # Cap placeholder count to avoid SQLite's variable limit on very large sets
        CHUNK = 500
        with_ft_db = 0
        total_words = 0
        for i in range(0, len(ids), CHUNK):
            chunk = ids[i:i + CHUNK]
            qmarks = ",".join("?" * len(chunk))
            with_ft_db += conn.execute(
                f"SELECT COUNT(*) FROM papers_fulltext WHERE paper_id IN ({qmarks})",
                tuple(chunk)).fetchone()[0]
            total_words += conn.execute(
                f"SELECT COALESCE(SUM(word_count),0) FROM papers_fulltext "
                f"WHERE paper_id IN ({qmarks})", tuple(chunk)).fetchone()[0]
    else:
        with_ft_db = 0
        total_words = 0
    conn.close()

    pct = (100.0 * with_ft_db / total) if total else 0.0
    return {
        "total": total,
        "with_pdf": with_pdf,
        "with_ft_metadata_flag": with_ft_meta,
        "with_ft_universe_rows": with_ft_db,
        "total_words": int(total_words or 0),
        "coverage_pct": round(pct, 1),
        "missing_count": max(0, total - with_ft_db),
        "topic": topic or "(all)",
    }


def backfill_missing_fulltexts(limit: Optional[int] = None,
                               silent: bool = False,
                               topic: Optional[str] = None) -> int:
    """
    Download + extract + store full texts for every metadata row (optionally
    restricted to a topic) that has a pdf_url but no stored full text.
    Idempotent: PDF dedup via pdf_hash keeps re-runs cheap.
    """
    conn = sqlite3.connect(db_manager.metadata_db)
    q = ("SELECT * FROM papers "
         "WHERE (fulltext_stored = 0 OR fulltext_stored IS NULL) "
         "AND pdf_url IS NOT NULL")
    params: Tuple = ()
    if topic:
        q += " AND topic = ?"
        params = (topic,)
    if limit:
        q += f" LIMIT {int(limit)}"
    df = pd.read_sql_query(q, conn, params=params)
    conn.close()
    if df.empty:
        update_log("Backfill: nothing to do — all rows already have full text")
        return 0

    papers = df.to_dict("records")
    total = len(papers)
    progress = st.progress(0) if not silent else None
    status = st.empty() if not silent else None
    ok = 0
    for i, paper in enumerate(papers, 1):
        if not silent and status is not None:
            status.text(f"⬇️ {i}/{total}: {str(paper.get('title',''))[:60]}…")
        try:
            updated = handle_paper_download(paper, manual_download=True)
            if updated.get('fulltext_stored'):
                ok += 1
        except Exception as e:
            update_log(f"Backfill error for {paper.get('id')}: {e}")
        if not silent and progress is not None:
            progress.progress(i / total)
    if not silent:
        if progress is not None:
            progress.empty()
        if status is not None:
            status.empty()
    update_log(f"Backfill complete: {ok}/{total} papers now carry full text")
    return ok


def repair_relevance_scores() -> int:
    """
    Fix legacy rows where relevance_score was stored as a 4-byte float32 BLOB
    (an earlier version wrote raw numpy floats). Returns count fixed.
    """
    conn = sqlite3.connect(db_manager.metadata_db)
    c = conn.cursor()
    fixed = 0
    for pid, val in c.execute("SELECT id, relevance_score FROM papers").fetchall():
        if isinstance(val, (bytes, bytearray)) and len(val) == 4:
            score = struct.unpack("<f", bytes(val))[0]
            c.execute("UPDATE papers SET relevance_score=? WHERE id=?",
                      (round(float(score), 2), pid))
            fixed += 1
    conn.commit()
    conn.close()
    if fixed:
        update_log(f"Repaired {fixed} BLOB relevance_score rows")
    return fixed


def coverage_warning_message(topic: Optional[str] = None) -> Optional[str]:
    """
    Return the canonical warning string when only some rows carry full text.
    None when coverage is complete or when there is nothing to warn about.
    """
    cov = get_fulltext_coverage(topic=topic)
    total = cov["total"]
    with_full = cov["with_ft_universe_rows"]
    if total == 0:
        return None
    if with_full == 0:
        return (f"⚠️ 0/{total} papers in this campaign have full text extracted. "
                f"The Full Text column will be empty. Use 'Download Enable in "
                f"Batch' or 'Backfill Missing Full Texts' first.")
    if with_full < total:
        return (f"⚠️ Only {with_full}/{total} papers have full text extracted. "
                f"Use 'Download Enable in Batch' or 'Backfill Missing Full Texts' "
                f"to process the remaining {total - with_full}.")
    return None


# -------------------------- USER INTERFACE HELPERS --------------------------


def show_logs(key_suffix: str):
    if st.session_state.log_buffer:
        with st.expander("📋 Processing Logs", expanded=False):
            st.text_area("Logs", "\n".join(st.session_state.log_buffer[-20:]),
                         height=150, key=f"log_display_{key_suffix}")


def create_dashboard():
    stats = db_manager.get_db_stats()
    st.subheader("📊 Database Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Papers", stats.get('total_papers', 0))
    with col2: st.metric("PDFs Stored", stats.get('pdfs_stored', 0))
    with col3: st.metric("Full Text Papers", stats.get('fulltext_stored', 0))
    with col4: st.metric("Total Size", f"{stats.get('total_pdf_size_mb', 0):.1f} MB")
    if stats.get('total_papers', 0) > 0:
        pdf_coverage = (stats.get('pdfs_stored', 0) / stats.get('total_papers', 1)) * 100
        st.progress(pdf_coverage / 100, text=f"PDF Coverage: {pdf_coverage:.1f}%")
    by_topic = stats.get('by_topic') or {}
    if by_topic:
        st.caption("Rows per campaign: " +
                   " · ".join(f"**{k}**={v}" for k, v in by_topic.items()))


# -------------------------- MAIN APPLICATION LAYOUT --------------------------


st.title("🔬 Strain Rate Sensitivity (m) Harvester — Nanotwinned Cu")
st.markdown("""
**Advanced tool for searching, downloading, and analyzing strain-rate-sensitivity (m)
and activation-volume (V*) research in nanotwinned Cu and related FCC microstructures.**
Features:
- **Smart Search**: Query arXiv with relevance scoring
- **PDF Storage**: Store PDFs in SQLite databases with deduplication
- **Full-Text Extraction**: SQLite `universe_db` is the single source of truth for CSV export
- **SRS Focus**: Enhanced relevance scoring over SRS / activation-volume / twin-spacing vocab
- **Dual Candidate Extractors**: both SRS (m, V*, strain-rate windows) and σ₀ (Peierls/Hall–Petch)
- **Scopus-Format CSV Export**: 22-column Scopus layout + Full Text + candidate snippets
- **Campaign Separation**: `topic` column keeps SRS results distinct from legacy σ₀ rows
- **Coverage Diagnostics**: live partial-coverage warnings and one-click backfill
""")


if IS_CLOUD:
    st.warning("""
⚠️ **Running on Streamlit Cloud**:
- PDF downloads are manual (click individual buttons)
- Use 'Download Enable in Batch' or 'Backfill Missing Full Texts' before exporting CSV
- Data is stored temporarily (may be cleared between sessions)
    """)


show_logs("top")


# -------------------------- SIDEBAR CONFIGURATION --------------------------


with st.sidebar:
    st.header("🔍 Search Configuration")

    # SRS (strain-rate sensitivity) × nt-Cu query.
    # Notes from the campaign map:
    #   - arXiv `abs:` searches abstracts only; no full-text wildcards available.
    #   - If recall is too low: keep only the second clause
    #       (abs:nanotwinned OR abs:nanotwin), or add ti: variants of clause 1.
    #   - If precision is too low: remove abs:"strain rate dependence" and
    #       abs:"thermally activated".
    default_query = (
        '(abs:"strain rate sensitivity" OR abs:"strain-rate sensitivity" '
        'OR abs:"rate sensitivity" OR abs:"activation volume" '
        'OR abs:"thermally activated" OR abs:"strain-rate dependence") '
        'AND (abs:nanotwinned OR abs:nanotwin OR abs:"twin boundary" '
        'OR abs:"twin boundaries" OR abs:"twin spacing" '
        'OR abs:"deformation twinning" OR abs:copper)'
    )
    query = st.text_area("Search Query", value=default_query, height=120)

    default_cats = ["cond-mat.mtrl-sci", "cond-mat.mes-hall", "physics.app-ph"]
    categories = st.multiselect("Categories", default_cats, default=default_cats)

    current_year = datetime.now().year
    col1, col2 = st.columns(2)
    with col1: start_year = st.number_input("Start Year", 1990, current_year, 2010)
    with col2: end_year = st.number_input("End Year", start_year, current_year, current_year)

    max_results = st.slider("Maximum Results", 1, 500, 50)
    relevance_threshold = st.slider("Relevance Threshold (%)", 0, 100, 30)

    st.subheader("💾 Storage Options")
    auto_download = st.checkbox("Auto-download PDFs", value=not IS_CLOUD, disabled=IS_CLOUD)

    st.subheader("📤 Export Options")
    export_formats = st.multiselect(
        "Select export formats",
        ["ZIP Archive", "CSV", "JSON", "Excel", "Database Backup"],
        default=["ZIP Archive", "CSV"]
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        search_btn = st.button("🔍 Search arXiv", type="primary", use_container_width=True)
    with col_btn2:
        if st.button("🔄 Reset Session", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key not in ["page_config_set", "log_buffer"]:
                    st.session_state[key] = DEFAULT_STATE[key]
            st.rerun()

    st.subheader("🔎 Search Database")
    st.caption("Tip: FTS search runs over stored full texts once they exist. "
               'Try `"strain rate" AND copper`, `"activation volume" AND nanotwin`, '
               '`detwinning AND "twin boundary"`.')
    db_query = st.text_input(
        "Search stored papers",
        placeholder="e.g., strain rate sensitivity nanotwinned copper"
    )
    if st.button("Search in Database", use_container_width=True):
        if db_query:
            conn = sqlite3.connect(db_manager.universe_db)
            c = conn.cursor()
            c.execute("""SELECT paper_id, title, snippet(papers_fts, 2, '<b>', '</b>', '...', 30)
                         FROM papers_fts WHERE papers_fts MATCH ? LIMIT 10""", (db_query,))
            results = c.fetchall()
            conn.close()
            if results:
                st.success(f"Found {len(results)} papers")
                for paper_id, title, snippet in results:
                    with st.expander(f"{title[:80]}..."):
                        st.write(f"**ID:** {paper_id}")
                        st.write(f"**Snippet:** {snippet}")
            else:
                st.warning("No results found")


create_dashboard()


# -------------------------- SEARCH AND PROCESSING LOGIC --------------------------


if search_btn:
    if not query.strip():
        st.error("Please enter a search query")
        st.stop()
    if not categories:
        st.error("Please select at least one category")
        st.stop()

    st.session_state.processing = True
    start_time = time.time()
    load_scibert()  # Preload model

    with st.spinner("🔍 Searching arXiv..."):
        papers = query_arxiv(query, categories, max_results, start_year, end_year,
                             topic=CURRENT_TOPIC)

    if not papers:
        st.warning("No papers found matching your criteria")
        st.session_state.processing = False
        st.stop()

    query_emb = get_embedding(query)
    # SRS embedding key terms — the vocabulary that should pull SRS-of-nt-Cu
    # papers to the top of the SciBERT-similarity ranking.
    key_terms = SRS_TERMS | MATERIAL_TERMS | {
        'twin spacing', 'twin thickness', 'lamellar twin', 'deformation twinning',
        'detwinning', 'twin boundary migration', 'slip transmission',
        'dislocation nucleation', 'schmid factor', 'rate-dependent', 'dynamic loading',
    }
    key_query = " ".join(key_terms)
    key_emb = get_embedding(key_query)
    st.session_state.query_emb = query_emb
    st.session_state.key_emb = key_emb

    with st.spinner("🧠 Computing SciBERT scores..."):
        for paper in papers:
            text = paper['title'] + " " + paper['abstract']
            paper_emb = get_embedding(text)
            sim1 = cosine_sim(query_emb, paper_emb)
            sim2 = cosine_sim(key_emb, paper_emb)
            paper['relevance_score'] = round((0.7 * sim1 + 0.3 * sim2) * 100, 2)
            db_manager.store_paper_metadata(paper)  # preserves prior download state
        papers.sort(key=lambda x: x['relevance_score'], reverse=True)

    relevant_papers = [p for p in papers if p['relevance_score'] >= relevance_threshold]
    if not relevant_papers:
        st.warning(f"No papers above {relevance_threshold}% relevance threshold")
        st.session_state.processing = False
        st.stop()

    st.success(f"Found **{len(relevant_papers)}** relevant papers "
               f"(topic: `{CURRENT_TOPIC}`)")

    # Attention heatmap
    st.subheader("🗺️ Query Attention Heatmap from SciBERT")
    tokenizer, model = load_scibert()
    inputs = tokenizer(query, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    att = outputs.attentions[-1].mean(dim=1).squeeze(0).detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(att, cmap='viridis')
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right")
    ax.set_yticklabels(tokens)
    fig.colorbar(im)
    st.pyplot(fig)

    if auto_download and not IS_CLOUD:
        progress_bar = st.progress(0)
        status_text = st.empty()
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(handle_paper_download, paper): i
                       for i, paper in enumerate(relevant_papers)}
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                paper = future.result()
                relevant_papers[idx] = paper
                completed += 1
                progress_bar.progress(completed / len(relevant_papers))
                status_text.text(f"Processed {completed}/{len(relevant_papers)} papers")
        progress_bar.empty()
        status_text.empty()

    st.session_state.relevant_papers = relevant_papers
    st.session_state.processing_time = time.time() - start_time
    update_log(f"Search completed in {st.session_state.processing_time:.1f} seconds "
               f"(topic={CURRENT_TOPIC})")


# -------------------------- MAINTENANCE & REPAIR --------------------------


st.divider()
st.subheader("🔧 Maintenance & Repair")
st.caption(f"Counters below are scoped to the current campaign "
           f"(topic = `{CURRENT_TOPIC}`).")

cov = get_fulltext_coverage(topic=CURRENT_TOPIC)
cov_cols = st.columns(6)
with cov_cols[0]: st.metric("Papers in Campaign", cov["total"])
with cov_cols[1]: st.metric("With PDF Stored", cov["with_pdf"])
with cov_cols[2]: st.metric("Full Text Flag", cov["with_ft_metadata_flag"])
with cov_cols[3]: st.metric("Full Text Rows", cov["with_ft_universe_rows"])
with cov_cols[4]: st.metric("Coverage", f"{cov['coverage_pct']:.1f}%")
with cov_cols[5]: st.metric("Total Words", f"{cov['total_words']:,}")

# Partial-coverage warning — fires on ANY shortfall, not just zero
_warn = coverage_warning_message(topic=CURRENT_TOPIC)
if _warn:
    st.warning(_warn)
elif cov["total"] > 0:
    st.success("✅ Full text data is present for every row in this campaign — "
               "the Scopus CSV will carry it.")

bf_col1, bf_col2, bf_col3 = st.columns(3)
with bf_col1:
    if st.button("⬇️ Backfill Missing Full Texts", use_container_width=True,
                 key="btn_backfill"):
        n = backfill_missing_fulltexts(topic=CURRENT_TOPIC)
        st.success(f"Processed {n} paper(s) — re-export the Scopus CSV now.")
        st.rerun()
with bf_col2:
    if st.button("🔧 Repair BLOB Relevance Scores", use_container_width=True,
                 key="btn_repair"):
        fixed = repair_relevance_scores()
        st.success(f"Repaired {fixed} BLOB relevance score row(s).")
with bf_col3:
    if st.button("🔄 Refresh Coverage Numbers", use_container_width=True,
                 key="btn_refresh_cov"):
        st.rerun()

# Show the actual IDs that are missing full text (collapsible; capped at 50)
_missing_ids = get_missing_fulltext_ids(topic=CURRENT_TOPIC)
if _missing_ids:
    with st.expander(f"🔎 {len(_missing_ids)} paper(s) in this campaign missing full text",
                     expanded=False):
        st.caption("These rows have a pdf_url but no extracted full text "
                   "in the universe DB. Backfill or batch-download to populate.")
        for pid in _missing_ids[:50]:
            st.code(pid)
        if len(_missing_ids) > 50:
            st.caption(f"…and {len(_missing_ids) - 50} more.")


# -------------------------- RESULTS DISPLAY AND BATCH DOWNLOAD --------------------------


if st.session_state.get('relevant_papers'):
    papers = st.session_state.relevant_papers

    if "selected_papers" not in st.session_state:
        st.session_state.selected_papers = set()
    if "batch_download_index" not in st.session_state:
        st.session_state.batch_download_index = 0

    available_paper_ids = {p['id'] for p in papers
                           if p.get('pdf_stored') or p['id'] in st.session_state.downloaded_pdfs}

    st.subheader(f"📄 Search Results ({len(papers)} papers · topic `{CURRENT_TOPIC}`)")

    col_sel1, col_sel2, col_sel3, col_sel4 = st.columns([1, 1, 1.5, 1.5])
    with col_sel1:
        if st.button("✓ Select All (with PDFs)"):
            st.session_state.selected_papers = available_paper_ids.copy()
        st.session_state.selected_papers = st.session_state.selected_papers & available_paper_ids
    with col_sel2:
        if st.button("✗ Deselect All"):
            st.session_state.selected_papers.clear()
    with col_sel3:
        BATCH_SIZE = 50
        total = len(papers)
        start_idx = st.session_state.batch_download_index
        end_idx = min(start_idx + BATCH_SIZE, total)
        remaining = total - start_idx

        if remaining > 0:
            label = f"📥 Download Enable in Batch ({start_idx + 1}–{end_idx})"
            if st.button(label):
                with st.spinner("⬇️ Processing batch..."):
                    progress = st.progress(0)
                    status = st.empty()
                    for i in range(start_idx, end_idx):
                        paper = papers[i]
                        status.text(f"Downloading {i+1}/{end_idx}: {paper['title'][:50]}...")
                        progress.progress((i - start_idx + 1) / (end_idx - start_idx))
                        if not (paper.get('pdf_stored') or paper['id'] in st.session_state.downloaded_pdfs):
                            updated = handle_paper_download(paper, manual_download=True)
                            papers[i] = updated
                    st.session_state.relevant_papers = papers
                    st.session_state.batch_download_index = end_idx
                    status.success(f"✅ Batch complete! Processed up to {end_idx}.")
                    time.sleep(1)
                    st.rerun()
        else:
            st.success("✅ All papers download-enabled!")
    with col_sel4:
        st.write(f"**Selected:** {len(st.session_state.selected_papers)} / {len(available_paper_ids)} available")

    if st.session_state.batch_download_index < len(papers):
        st.info(f"✅ Processed {st.session_state.batch_download_index} of {len(papers)} papers. "
                f"Click 'Download Enable in Batch' to continue.")
    else:
        st.success("🎉 All papers are download-enabled! Use 'Select All (with PDFs)' to choose all.")

    # Display each paper
    for i, paper in enumerate(papers):
        enhanced = paper.get('enhanced_relevance_score', 0)
        srs_flag = "🟢" if paper.get('dopant_present') else "⚪"
        material_flag = "🔵" if paper.get('beta_phase_present') else "⚪"
        can_select = paper['id'] in available_paper_ids
        is_selected = paper['id'] in st.session_state.selected_papers

        with st.expander(
            f"**{paper['title']}** ({paper['year']}) - "
            f"Basic: {paper['relevance_score']}% | "
            f"Enhanced: {enhanced:.1f}% "
            f"SRS:{srs_flag} Material:{material_flag}",
            expanded=i < 2
        ):
            col_check, col_info, col_actions = st.columns([0.5, 2.5, 1])

            with col_check:
                if can_select:
                    new_selected = st.checkbox(
                        "",
                        value=is_selected,
                        key=f"select_{paper['id']}_{i}",
                        label_visibility="collapsed"
                    )
                    if new_selected and paper['id'] not in st.session_state.selected_papers:
                        st.session_state.selected_papers.add(paper['id'])
                    elif not new_selected and paper['id'] in st.session_state.selected_papers:
                        st.session_state.selected_papers.discard(paper['id'])
                else:
                    st.empty()

            with col_info:
                st.write(f"**Authors:** {paper['authors']}")
                st.write(f"**Categories:** {paper['categories']}")
                st.write(f"**Matched Terms:** {paper['matched_terms']}")
                st.write(f"**Status:** {paper['download_status']}")
                show_abstract = st.toggle("Show Abstract",
                                          key=f"toggle_abstract_{paper['id']}_{i}")
                if show_abstract:
                    st.markdown(f"> {paper['abstract']}")

            with col_actions:
                if can_select:
                    if paper['id'] in st.session_state.downloaded_pdfs:
                        pdf_bytes = st.session_state.downloaded_pdfs[paper['id']]['pdf_bytes']
                    else:
                        pdf_bytes = db_manager.get_pdf(paper['id'])
                    if pdf_bytes:
                        safe_title = re.sub(r'[^\w\s-]', '', paper['title'])[:50]
                        filename = f"{paper['id']}_{safe_title}.pdf".replace(' ', '_')
                        st.download_button(
                            label="📥 Download",
                            data=pdf_bytes,
                            file_name=filename,
                            mime="application/pdf",
                            key=f"dl_{paper['id']}_{i}",
                            use_container_width=True
                        )
                else:
                    if st.button("⬇️ Download Now", key=f"manual_{paper['id']}_{i}",
                                 use_container_width=True):
                        with st.spinner("Downloading..."):
                            updated_paper = handle_paper_download(paper, manual_download=True)
                            papers[i] = updated_paper
                            st.session_state.relevant_papers = papers
                            st.rerun()
                st.markdown(f"[🌐 arXiv Page]({paper['pdf_url'].replace('/pdf/', '/abs/')})")
                st.markdown(f"[📄 Direct PDF]({paper['pdf_url']})")

    # Bulk export section
    st.subheader("📤 Export & Bulk Download")
    export_cols = st.columns(5)
    paper_ids = [p['id'] for p in papers
                 if p.get('pdf_stored') or p['id'] in st.session_state.downloaded_pdfs]
    selected_ids = list(st.session_state.selected_papers)

    # ZIP
    if "ZIP Archive" in export_formats:
        with export_cols[0]:
            scope = st.radio("ZIP Scope", ["All Available", "Selected Only"],
                             key="zip_scope", horizontal=True)
            ids = selected_ids if (scope == "Selected Only" and selected_ids) else paper_ids
            if ids:
                if st.button("📦 Create ZIP", use_container_width=True):
                    with st.spinner(f"Creating ZIP with {len(ids)} PDFs..."):
                        buffer = db_manager.create_zip_from_db(ids)
                        st.session_state.zip_buffer = buffer
                        st.success(f"ZIP created with {len(ids)} PDFs")
                if st.session_state.zip_buffer:
                    st.download_button(
                        "⬇️ Download ZIP",
                        st.session_state.zip_buffer.getvalue(),
                        f"{CURRENT_TOPIC}_papers.zip",
                        "application/zip",
                        use_container_width=True
                    )
            else:
                st.caption("No PDFs to ZIP")

    # Bulk individual download
    with export_cols[1]:
        if selected_ids:
            st.write("**📥 Bulk Download Selected**")
            for pid in selected_ids:
                paper = next((p for p in papers if p['id'] == pid), None)
                if paper:
                    if pid in st.session_state.downloaded_pdfs:
                        data = st.session_state.downloaded_pdfs[pid]['pdf_bytes']
                    else:
                        data = db_manager.get_pdf(pid)
                    if data:
                        title = re.sub(r'[^\w\s-]', '', paper['title'])[:50]
                        fname = f"{pid}_{title}.pdf".replace(' ', '_')
                        st.download_button(
                            f"📄 {paper['title'][:30]}...",
                            data,
                            fname,
                            "application/pdf",
                            key=f"bulk_{pid}",
                            use_container_width=True
                        )
        else:
            st.caption("Select papers above")

    # ------------------------------------------------------------------
    # CSV — Scopus layout + Full Text, scoped to the current campaign.
    # ------------------------------------------------------------------
    if "CSV" in export_formats:
        with export_cols[2]:
            include_ft = st.checkbox("Include Full Text column", value=True,
                                     key="csv_fulltext")
            limit = st.number_input("Full-text char limit (0 = no limit)",
                                    0, 200000, 50000, key="csv_ft_limit")
            only_ft = st.checkbox("Only rows with full text", value=False,
                                  key="csv_only_ft",
                                  help="Skip rows whose papers_fulltext entry is "
                                       "empty. Recommended when coverage is partial.")
            require_flag = st.checkbox("Require fulltext_stored flag = 1",
                                       value=False, key="csv_require_flag",
                                       help="Belt-and-braces filter: export only rows "
                                            "whose metadata column fulltext_stored is 1.")
            auto_extract = st.checkbox(
                "Auto-extract missing before export",
                value=False, key="csv_auto_extract",
                help="Downloads + extracts full text for every row in this campaign "
                     "that lacks it, then exports. SLOW on large sets — 1 HTTP call "
                     "per missing paper. Not recommended for >100 rows."
            )

            cov_now = get_fulltext_coverage(topic=CURRENT_TOPIC)
            total_rows = cov_now["total"]
            with_full = cov_now["with_ft_universe_rows"]
            if total_rows > 0:
                st.caption(f"📊 Topic `{CURRENT_TOPIC}` — full text available for "
                           f"**{with_full}/{total_rows}** row(s) "
                           f"({cov_now['coverage_pct']:.1f}%).")
            else:
                st.caption(f"No rows yet for topic `{CURRENT_TOPIC}` — run a search first.")

            if auto_extract and with_full < total_rows:
                st.warning(
                    f"⚠️ {total_rows - with_full} row(s) in this campaign still lack "
                    f"full text. Click below to extract them before exporting."
                )
                if st.button("⬇️ Extract missing now",
                             key="btn_extract_now",
                             use_container_width=True):
                    n = backfill_missing_fulltexts(topic=CURRENT_TOPIC)
                    st.success(f"Extracted {n} new paper(s). Re-export the Scopus CSV.")
                    st.rerun()

            buf = export_scopus_format(
                include_fulltext=include_ft,
                fulltext_char_limit=(limit or None),
                only_with_fulltext=only_ft,
                require_metadata_flag=require_flag,
                topic=CURRENT_TOPIC,     # ← campaign isolation enforced
            )

            if buf.getbuffer().nbytes > 0:
                st.download_button(
                    "📊 Scopus-format CSV", buf.getvalue(),
                    f"{CURRENT_TOPIC}_metadatabase.csv", "text/csv",
                    use_container_width=True)

                if include_ft and not only_ft and total_rows > 0 and with_full < total_rows:
                    st.warning(
                        f"⚠️ Only {with_full}/{total_rows} papers have full text "
                        f"extracted. The Full Text column will be blank for the "
                        f"remaining {total_rows - with_full} row(s). Use 'Download "
                        f"Enable in Batch' or 'Backfill Missing Full Texts'."
                    )
                elif include_ft and only_ft and with_full < total_rows:
                    st.info(
                        f"ℹ️ 'Only rows with full text' is enabled — the CSV will "
                        f"contain {with_full} of {total_rows} rows."
                    )
            else:
                st.caption("No rows for this campaign matched the export filters yet.")

    # JSON — scoped to current campaign
    if "JSON" in export_formats:
        with export_cols[3]:
            buf = db_manager.export_metadata("json", topic=CURRENT_TOPIC)
            if buf.getbuffer().nbytes > 0:
                st.download_button("📄 JSON Export", buf.getvalue(),
                                   f"{CURRENT_TOPIC}_metadata.json", "application/json",
                                   use_container_width=True)

    # Excel — scoped to current campaign
    if "Excel" in export_formats:
        with export_cols[4]:
            buf = db_manager.export_metadata("excel", topic=CURRENT_TOPIC)
            if buf.getbuffer().nbytes > 0:
                st.download_button(
                    "📈 Excel Export", buf.getvalue(), f"{CURRENT_TOPIC}_metadata.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True)

    # Raw database files
    with st.expander("🗃️ Databases"):
        for label, path in [
            ("Metadata DB", db_manager.metadata_db),
            ("Fulltext DB", db_manager.universe_db),
            ("PDF Storage DB", db_manager.pdf_db)
        ]:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    st.download_button(
                        label=label,
                        data=f.read(),
                        file_name=os.path.basename(path),
                        mime="application/octet-stream",
                        use_container_width=True
                    )


# -------------------------- DATABASE MANAGEMENT SECTION --------------------------


st.divider()
st.subheader("🗄️ Database Management")

col_stats, col_clean = st.columns(2)
with col_stats:
    if st.button("🔄 Refresh Statistics", use_container_width=True):
        st.session_state.db_stats = db_manager.get_db_stats()
        st.rerun()
with col_clean:
    if st.button("🧹 Clean Temporary Files", use_container_width=True):
        temp_dir = os.path.join(DB_DIR, "temp")
        if os.path.exists(temp_dir):
            for file in os.listdir(temp_dir):
                try:
                    os.remove(os.path.join(temp_dir, file))
                except OSError:
                    pass
        st.session_state.temp_files = []
        st.success("Temporary files cleaned")

if st.session_state.db_stats:
    with st.expander("📊 Detailed Statistics"):
        st.json(st.session_state.db_stats)


# -------------------------- FOOTER --------------------------


st.divider()
st.caption(f"""
**Strain Rate Sensitivity (m) Harvester — Nanotwinned Cu** |
Campaign: `{CURRENT_TOPIC}` |
Running on {'☁️ Streamlit Cloud' if IS_CLOUD else '💻 Local'} |
Data Directory: `{DB_DIR}` |
Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")

show_logs("bottom")


# -------------------------- CLEANUP HOOK --------------------------


import atexit


def cleanup():
    temp_dir = os.path.join(DB_DIR, "temp")
    if os.path.exists(temp_dir):
        for file in os.listdir(temp_dir):
            try:
                os.remove(os.path.join(temp_dir, file))
            except OSError:
                pass


atexit.register(cleanup)


# -------------------------- END OF FILE --------------------------
# SRS harvester — arXiv query targets strain-rate sensitivity / activation volume
# in nanotwinned Cu. Each row is labelled with `topic='srs_ntcu'`; legacy σ₀ rows
# default to `topic='sigma0'`. Exports filter by topic, so campaigns never mix.
#
# Honest expectations (from the pivot map):
#   - arXiv coverage of experimental SRS-of-nt-Cu is thin. The canonical
#     K. Lu / L. Lu Acta Mater. / Scripta Mater. / JMPS / PRL papers are mostly
#     NOT deposited there. Expect predominantly MD-simulation preprints.
#     If the harvest looks sparse, that is the source, not the query.
#   - The physical payoffs the extractors target: strain-rate windows
#     (10⁻⁴–10³ s⁻¹), m vs twin spacing (m rising from ~0.006 to ~0.02–0.03
#     as λ shrinks), and activation volumes in b³ / nm³ — all of which
#     extract_candidate_srs() flags for downstream NER.
# --------------------------------------------------------------
