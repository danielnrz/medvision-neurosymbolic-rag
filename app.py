import streamlit as st
import pandas as pd
import numpy as np
import torch
import open_clip
import cv2
import pickle
import os
import uuid
from io import BytesIO
from PIL import Image, ImageEnhance
from sklearn.metrics.pairwise import cosine_similarity
import datetime
import matplotlib.pyplot as plt
import hashlib

from rdflib import Graph, Namespace, RDF
from rdflib.namespace import SKOS
from typing import Optional, List, Tuple, Dict

# --- NEW: for local LLM (Ollama / DeepSeek-R1) ---
import re
from openai import OpenAI

# Minimum cosine similarity to consider an image as an in-scope chest X-ray
XRAY_SIM_THRESHOLD = 0.75

# ==========================================
# 1. Page settings
# ==========================================
st.set_page_config(
    page_title="MedVision PACS | Neuro-Symbolic AI",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CSS for light theme and professional report layout ---
st.markdown(
    """
<style>
    /* General theme */
    .stApp { background-color: #f8fafc; color: #1e293b; font-family: 'Segoe UI', sans-serif; }

    /* Patient header */
    .patient-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 15px 20px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-left: 5px solid #991b1b;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .patient-text { font-family: 'Courier New', monospace; font-size: 14px; color: #64748b; font-weight: bold; }
    .patient-val { color: #0f172a; font-weight: bold; margin-right: 15px; }

    /* AI diagnosis box */
    .prediction-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .pred-title { color: #64748b; font-size: 0.85em; text-transform: uppercase; letter-spacing: 1.2px; font-weight: 700; }
    .pred-value { font-size: 1.8em; font-weight: 800; margin: 10px 0; line-height: 1.2; }
    .pred-conf { background: #f1f5f9; color: #475569; padding: 4px 12px; border-radius: 20px; font-size: 0.85em; font-weight: 600; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #fff;
        border-radius: 8px 8px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        border: 1px solid #e2e8f0;
        color: #64748b;
    }
    .stTabs [aria-selected="true"] {
        background-color: #eff6ff;
        color: #343434;
        border-color: #bfdbfe;
        font-weight: bold;
    }

    /* Highlighted words */
    .hl-red { background-color: #fef2f2; color: #991b1b; padding: 2px 6px; border-radius: 4px; border: 1px solid #fecaca; font-weight: bold; cursor: help; }
    .hl-green { background-color: #f0fdf4; color: #166534; padding: 2px 6px; border-radius: 4px; border: 1px solid #bbf7d0; font-weight: bold; cursor: help; }
    .hl-blue { background-color: #eff6ff; color: #1e40af; padding: 2px 6px; border-radius: 4px; border: 1px solid #bfdbfe; font-weight: bold; }

    /* Final report style (rule-based) */
    .paper-report {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-top: 6px solid #991b1b;
        border-radius: 8px;
        padding: 40px;
        font-family: 'Segoe UI', Tahoma, sans-serif;
        color: #334155;
        line-height: 1.8;
        margin-top: 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
        width: 100%;
    }
    .report-header {
        display: flex;
        justify-content: space-between;
        border-bottom: 2px solid #f1f5f9;
        padding-bottom: 15px;
        margin-bottom: 20px;
    }
    .report-section { margin-bottom: 20px; }
    .section-title {
        color: #FFFFFF;
        font-weight: 700;
        text-transform: uppercase;
        font-size: 0.9em;
        letter-spacing: 1px;
        margin-bottom: 8px;
        border-left: 3px solid #991b1b;
        padding-left: 10px;
    }
    .report-footer {
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px dashed #cbd5e1;
        color: #64748b;
        font-size: 0.85em;
        font-style: italic;
        display: flex;
        justify-content: space-between;
    }

    /* AI disclaimer box */
    .ai-warning {
        margin-top: 30px;
        background-color: #fffbeb;
        border: 1px solid #fcd34d;
        color: #92400e;
        padding: 15px;
        border-radius: 6px;
        font-size: 0.9em;
        text-align: center;
        font-weight: 500;
    }

    .stButton button { width: 100%; border-radius: 6px; font-weight: 600; height: 45px; }
</style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# 2. Loading resources
# ==========================================
@st.cache_resource
def load_kg(kg_path: str, disease_classes: Optional[List[str]] = None):
    if not kg_path or not os.path.exists(kg_path):
        return None, {}

    g = Graph()
    g.parse(kg_path, format="turtle")

    EX = Namespace("http://example.org/medical_project/")
    kg_info: Dict[str, Dict] = {}

    if disease_classes is None:
        disease_classes = [
            "Cardiomegaly",
            "Pneumonia",
            "Atelectasis",
            "Edema",
            "PleuralEffusion",
            "Pneumothorax",
            "Fracture",
            "Normal",
        ]

    for d in disease_classes:
        du = EX[d]
        locs = [str(o).split("/")[-1] for o in g.objects(du, EX.locatedIn)]
        definition = next(iter(g.objects(du, SKOS.definition)), None)

        kg_info[d] = {
            "locations": locs,
            "definition": str(definition) if definition else "",
        }

    return g, kg_info


@st.cache_resource
def load_resources():
    device = "cpu"

    model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        device=device,
    )

    db_df = None
    ai_classifier = None
    mlb = None
    scaler = None
    all_embeddings = None
    kg_graph = None
    kg_info = {}
    thr_vec = None
    calibrators_aligned = None  # optional per-class probability calibrators

    try:
        # 1) Load embeddings database
        if os.path.exists("dataset_final_fixed.pkl"):
            with open("dataset_final_fixed.pkl", "rb") as f:
                db_df = pickle.load(f)
        elif os.path.exists("dataset_with_embeddings_fixed.pkl"):
            with open("dataset_with_embeddings_fixed.pkl", "rb") as f:
                db_df = pickle.load(f)
        elif os.path.exists("dataset_with_embeddings.pkl"):
            with open("dataset_with_embeddings.pkl", "rb") as f:
                db_df = pickle.load(f)
        else:
            st.error("Dataset file not found. Please run the notebook first.")
            return (None,) * 12

        # 2) Load classifier model
        if os.path.exists("smart_adapter.pkl"):
            with open("smart_adapter.pkl", "rb") as f:
                ai_classifier = pickle.load(f)

        # 3) MultiLabelBinarizer
        if os.path.exists("multilabel_binarizer.pkl"):
            with open("multilabel_binarizer.pkl", "rb") as f:
                mlb = pickle.load(f)

        # 4) Scaler
        if os.path.exists("scaler.pkl"):
            with open("scaler.pkl", "rb") as f:
                scaler = pickle.load(f)

        # 4.5) Optional probability calibrators (sigmoid / Platt scaling)
        # If present, probabilities shown/used in the UI can be "calibrated" rather than raw model scores.
        if os.path.exists("calibrators.pkl") and mlb is not None:
            try:
                with open("calibrators.pkl", "rb") as f:
                    cpack = pickle.load(f)

                cal_classes = list(cpack.get("classes", []))
                cal_list = cpack.get("calibrators", None)

                # Align calibrators to mlb.classes_
                if isinstance(cal_list, list) and len(cal_list) > 0:
                    if cal_classes and len(cal_classes) == len(cal_list):
                        cal_map = {c: cal for c, cal in zip(cal_classes, cal_list)}
                        aligned = [cal_map.get(c) for c in mlb.classes_]
                        calibrators_aligned = aligned if all(x is not None for x in aligned) else None
                    elif len(cal_list) == len(mlb.classes_):
                        calibrators_aligned = cal_list
            except Exception:
                calibrators_aligned = None

        # 5) Embedding matrix
        all_embeddings = np.stack(db_df["embedding"].values)

        # 6) Per-class thresholds (aligned to mlb.classes_)
        if os.path.exists("per_class_thresholds.pkl") and mlb is not None:
            with open("per_class_thresholds.pkl", "rb") as f:
                pack = pickle.load(f)
            cls_list = pack.get("classes", [])
            thr_list = pack.get("thresholds", [])
            thr_map = {c: float(t) for c, t in zip(cls_list, thr_list)}
            thr_vec = np.array([thr_map.get(c, 0.35) for c in mlb.classes_], dtype=float)

        # 7) Knowledge graph (optional)
        kg_graph, kg_info = load_kg(
            "medical_kg_comprehensive.ttl",
            list(mlb.classes_) if mlb is not None else None,
        )

    except Exception as e:
        st.error(f"Error loading files: {e}")
        return (None,) * 12

    return (
        model,
        preprocess,
        device,
        db_df,
        all_embeddings,
        ai_classifier,
        mlb,
        scaler,
        kg_graph,
        kg_info,
        thr_vec,
        calibrators_aligned,
    )


model, preprocess, device, df_final, database_embeddings, clf, mlb, scaler, kg_graph, kg_info, thr_vec, calibrators_aligned = load_resources()

# simple dict: disease -> definition (for LLM prompt)
kg_definitions = {
    d: (info.get("definition") or f"{d} (no detailed definition in KG)")
    for d, info in kg_info.items()
}

# ==========================================
# 3. Utility functions
# ==========================================
def process_display_image(img: Image.Image, brightness: float, contrast: float) -> Image.Image:
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast)
    return img


def prepare_for_ai(pil_image: Image.Image) -> Image.Image:
    return pil_image.convert("RGB")


def get_image_embedding(image: Image.Image) -> np.ndarray:
    inputs = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(inputs)
        emb /= emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().flatten()


def compute_clip_saliency(pil_img, model, preprocess, device):
    """
    Image-only saliency: gradient of embedding norm wrt pixels.
    """
    model.eval()
    img_t = preprocess(pil_img).unsqueeze(0).to(device)
    img_t.requires_grad_(True)

    with torch.enable_grad():
        img_feat = model.encode_image(img_t)
        score = img_feat.norm(dim=-1).sum()
        score.backward()

        grad = img_t.grad.detach().abs().mean(dim=1).squeeze()
        grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)
        heat = grad.cpu().numpy()
    return heat


def predict_probs_from_embedding(emb_raw: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
    """
    Returns (probs, source) where probs is shape [C] aligned with mlb.classes_.
    Uses calibrated probabilities if calibrators are available; otherwise raw clf probs.
    """
    if clf is None or mlb is None or scaler is None:
        return None, "unavailable"

    emb_scaled = scaler.transform(emb_raw.reshape(1, -1))
    raw_probs = clf.predict_proba(emb_scaled)[0]

    # calibrated (Platt / sigmoid) per class, if present and aligned
    if calibrators_aligned is not None:
        try:
            probs = np.array([cal.predict_proba(emb_scaled)[:, 1][0] for cal in calibrators_aligned], dtype=float)
            return probs, "calibrated"
        except Exception:
            pass

    return raw_probs.astype(float), "raw"


def predict_probs_from_pil(pil_img: Image.Image) -> Tuple[Optional[np.ndarray], str]:
    emb = get_image_embedding(prepare_for_ai(pil_img))
    return predict_probs_from_embedding(emb)


def filter_rag_caption_for_labels(caption: str, predicted_labels: List[str], max_sentences: int = 2) -> str:
    """
    Keep only the most relevant evidence sentences for the predicted labels to reduce RAG-induced hallucinations.
    - If no label-specific sentence is found, fall back to the first sentence.
    """
    if not caption:
        return ""

    labels = [l for l in (predicted_labels or []) if l and l != "Normal"]
    if not labels:
        # If only Normal, keep a short generic context.
        sents = _sentences(caption)
        return sents[0] if sents else caption

    # Build regex patterns for the predicted labels using the same lexicon used by the verifier
    pats: List[re.Pattern] = []
    for lab in labels:
        for p in LABEL_PATTERNS.get(lab, []):
            try:
                pats.append(re.compile(p, flags=re.IGNORECASE))
            except re.error:
                continue

    sents = _sentences(caption)
    keep: List[str] = []
    for s in sents:
        if any(p.search(s) for p in pats):
            keep.append(s)
        if len(keep) >= max_sentences:
            break

    if not keep:
        keep = sents[:1] if sents else [caption]

    return ". ".join(keep).strip()


@st.cache_data(show_spinner=False)
def compute_occlusion_heatmap(
    image_bytes: bytes,
    label_name: str,
    grid: int = 6,
) -> Optional[np.ndarray]:
    """
    Model-agnostic, class-specific explainability:
    occlude a coarse grid over the image and measure the change in the classifier probability for `label_name`.

    Returns a heatmap in the original image resolution (H,W) normalized to [0,1], or None if model unavailable.
    """
    if mlb is None:
        return None
    if label_name not in mlb.classes_:
        return None

    pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
    W, H = pil_img.size

    probs0, _src = predict_probs_from_pil(pil_img)
    if probs0 is None:
        return None

    label_idx = list(mlb.classes_).index(label_name)
    base = float(probs0[label_idx])

    # grid heatmap
    heat = np.zeros((grid, grid), dtype=np.float32)

    # occlusion box size (in pixels)
    step_x = max(1, W // grid)
    step_y = max(1, H // grid)

    # gray mask color
    mask_rgb = (128, 128, 128)

    for gy in range(grid):
        for gx in range(grid):
            x0 = gx * step_x
            y0 = gy * step_y
            x1 = min(W, x0 + step_x)
            y1 = min(H, y0 + step_y)

            oc = pil_img.copy()
            oc_arr = np.array(oc)
            oc_arr[y0:y1, x0:x1, :] = mask_rgb
            oc2 = Image.fromarray(oc_arr)

            probs1, _ = predict_probs_from_pil(oc2)
            if probs1 is None:
                return None

            p1 = float(probs1[label_idx])
            heat[gy, gx] = max(0.0, base - p1)

    # normalize and upscale to image size
    if heat.max() > 0:
        heat = heat / (heat.max() + 1e-8)

    heat_up = cv2.resize(heat, (W, H), interpolation=cv2.INTER_CUBIC)
    heat_up = np.clip(heat_up, 0.0, 1.0)
    return heat_up


def kg_validate_diseases(diseases: List[str], kg_info_dict: Dict) -> Tuple[List[str], List[str]]:
    warnings: List[str] = []
    if not kg_info_dict:
        return diseases, warnings

    allowed = set(kg_info_dict.keys())
    clean = [d for d in diseases if d in allowed]
    unknown = [d for d in diseases if d not in allowed]
    if unknown:
        warnings.append(f"KG: removed unknown labels: {unknown}")

    if len(clean) > 1 and "Normal" in clean:
        clean = [d for d in clean if d != "Normal"]
        warnings.append("KG: removed 'Normal' because other pathologies exist.")

    if not clean:
        clean = ["Normal"]
        warnings.append("KG: fallback to 'Normal' (empty prediction after validation).")

    return clean, warnings


def select_pred_indices(
    probs: np.ndarray,
    conf_threshold: float,
    use_per_class_thr: bool,
    thr_vec_aligned: Optional[np.ndarray],
) -> np.ndarray:
    if use_per_class_thr and (thr_vec_aligned is not None) and (len(thr_vec_aligned) == len(probs)):
        return np.where(probs > thr_vec_aligned)[0]
    return np.where(probs > conf_threshold)[0]


def highlight_text(text: str, entities: List[str]) -> str:
    keyword_map = {
        "cardiomegaly": "Cardiomegaly",
        "enlarged heart": "Cardiomegaly",
        "heart": "Cardiomegaly",
        "cardiac": "Cardiomegaly",
        "ctr": "Cardiomegaly",
        "pneumonia": "Pneumonia",
        "infection": "Pneumonia",
        "consolidation": "Pneumonia",
        "infiltration": "Pneumonia",
        "opacity": "Pneumonia",
        "opacities": "Pneumonia",
        "airspace": "Pneumonia",
        "edema": "Edema",
        "congestion": "Edema",
        "fluid overload": "Edema",
        "interstitial": "Edema",
        "peribronchial": "Edema",
        "kerley": "Edema",
        "effusion": "PleuralEffusion",
        "pleural effusion": "PleuralEffusion",
        "blunting": "PleuralEffusion",
        "fluid": "PleuralEffusion",
        "costophrenic": "PleuralEffusion",
        "atelectasis": "Atelectasis",
        "collapse": "Atelectasis",
        "volume loss": "Atelectasis",
        "pneumothorax": "Pneumothorax",
        "lucency": "Pneumothorax",
        "fracture": "Fracture",
        "broken": "Fracture",
        "deformity": "Fracture",
        "rib": "Fracture",
        "clavicle": "Fracture",
        "normal": "Normal",
        "clear": "Normal",
        "unremarkable": "Normal",
        "intact": "Normal",
        "stable": "Normal",
    }
    anatomy = [
        "lung",
        "lungs",
        "pleural",
        "spine",
        "ribs",
        "silhouette",
        "mediastinal",
        "diaphragm",
        "osseous",
    ]

    if not isinstance(entities, list):
        entities = ["Normal"]

    words = text.split()
    out = []

    for w in words:
        cln = w.lower().strip(".,")
        matched_concept = None

        for key, concept in keyword_map.items():
            if key in cln:
                matched_concept = concept
                break

        if matched_concept and matched_concept in entities:
            color = "hl-green" if matched_concept == "Normal" else "hl-red"
            out.append(f'<span class="{color}" title="{matched_concept}">{w}</span>')
        elif any(anat in cln for anat in anatomy):
            out.append(f'<span class="hl-blue">{w}</span>')
        else:
            out.append(w)

    return " ".join(out)


# ==========================================
# 3.2 Neuro-symbolic verifier (post-hoc fact checking)
# ==========================================
NEG_PATTERNS = [
    r"\bno\b",
    r"\bwithout\b",
    r"\babsence of\b",
    r"\bnegative for\b",
    r"\bdenies\b",
]

# Simple label lexicon (kept lightweight + transparent for the course)
LABEL_PATTERNS: Dict[str, List[str]] = {
    "PleuralEffusion": [r"pleural effusion", r"\beffusion\b"],
    "Pneumothorax": [r"pneumothorax"],
    "Edema": [r"pulmonary edema", r"interstitial edema", r"\bedema\b", r"vascular congestion"],
    "Pneumonia": [r"pneumonia", r"consolidation", r"airspace opacity", r"\binfiltrate\b", r"\bopacit(?:y|ies)\b"],
    "Atelectasis": [r"atelectasis", r"collapse", r"volume loss"],
    "Cardiomegaly": [r"cardiomegaly", r"enlarged heart", r"cardiac silhouette is enlarged", r"enlarged cardiac silhouette"],
    "Fracture": [r"fracture", r"rib fracture", r"\bbroken rib\b"],
    "Normal": [r"\bnormal\b", r"unremarkable", r"clear lungs"],
}


# Location hint synonyms to reduce false warnings (reports often use "right base", "cardiomediastinal", etc.)
GENERIC_LOC_SYNONYMS: Dict[str, List[str]] = {
    "Heart": [
        "heart",
        "cardiac",
        "cardiomediastinal",
        "mediastinum",
        "cardiac silhouette",
        "cardiomediastinal silhouette",
    ],
    "RightLung": [
        "right",
        "right lung",
        "right hemithorax",
        "right upper lobe",
        "right middle lobe",
        "right lower lobe",
        "right base",
        "right apex",
        "right mid",
        "right mid zone",
        "rul",
        "rml",
        "rll",
    ],
    "LeftLung": [
        "left",
        "left lung",
        "left hemithorax",
        "left upper lobe",
        "left lower lobe",
        "left base",
        "left apex",
        "left mid",
        "lul",
        "lll",
    ],
    "Lungs": ["lung", "lungs", "pulmonary"],
}


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"[\.\n]+", text) if s.strip()]

def _is_negated(sentence_lc: str) -> bool:
    return any(re.search(p, sentence_lc) for p in NEG_PATTERNS)

def verify_report_with_kg(
    report_text: str,
    predicted: List[str],
    kg_info_dict: Dict,
    allowed_labels: List[str],
) -> Tuple[List[str], List[str]]:
    """
    Lightweight post-hoc verifier:
    - Rejects any PRESENT claim that is not in predicted labels (out-of-scope hallucination).
    - Rejects any ABSENT (negated) claim for a label that the model predicted present.
    - Optionally warns if a PRESENT label lacks a location hint from KG.
    Returns (violations, warnings).
    """
    violations: List[str] = []
    warnings: List[str] = []

    if not report_text:
        return ["Empty report"], warnings

    present = set(predicted or [])
    allowed = set(allowed_labels or [])

    for s in _sentences(report_text):
        slc = s.lower()
        neg = _is_negated(slc)

        # Hard anatomical sanity checks (cheap but effective)
        # Humans have 12 rib pairs; reject impossible rib indices (e.g., 'rib 18').
        m_rib = re.search(r"\brib\s*(\d{1,2})\b", slc)
        if m_rib:
            try:
                rib_n = int(m_rib.group(1))
                if rib_n > 12:
                    violations.append(f"Impossible anatomy: rib {rib_n} (max 12) | sentence: '{s}'")
            except Exception:
                pass

        for label, pats in LABEL_PATTERNS.items():
            if not any(re.search(p, slc) for p in pats):
                continue

            if label not in allowed:
                violations.append(f"Mentions out-of-scope label: {label} | sentence: '{s}'")
                continue

            # PRESENT but not predicted (except "Normal")
            if (not neg) and (label not in present) and (label != "Normal"):
                violations.append(f"Claims PRESENT but not predicted: {label} | sentence: '{s}'")

            # ABSENT but predicted present (except "Normal")
            if neg and (label in present) and (label != "Normal"):
                violations.append(f"Claims ABSENT but predicted present: {label} | sentence: '{s}'")

            # Optional warning: location hint missing for PRESENT labels
            if (not neg) and kg_info_dict and (label in kg_info_dict):
                locs = (kg_info_dict.get(label, {}) or {}).get('locations', [])
                if locs:
                    loc_variants = set()
                    for loc in locs:
                        loc_variants.add(loc.lower())
                        loc_variants.add(loc.replace('_', ' ').lower())
                        loc_variants.add(re.sub(r'(?<!^)(?=[A-Z])', ' ', loc).lower())
                        # add common synonyms for better matching
                        for syn in GENERIC_LOC_SYNONYMS.get(loc, []):
                            loc_variants.add(str(syn).lower())
                    if not any(v in slc for v in loc_variants):
                        warnings.append(f"Missing location hint for {label} (expected one of {locs}) | sentence: '{s}'")

    return violations, warnings


def generate_report_html_rule_based(diseases_list: List[str]) -> str:
    """
    Old rule-based template (kept as fallback).
    """
    date_str = datetime.date.today().strftime("%Y-%m-%d")
    findings_html = ""

    if "Cardiomegaly" in diseases_list:
        findings_html += "The cardiac silhouette is <b>moderately enlarged</b> (Cardiomegaly). "
    else:
        findings_html += "The cardiac silhouette is normal in size and contour. "

    lung_findings = []
    if "Pneumonia" in diseases_list:
        lung_findings.append("focal opacities/consolidation consistent with pneumonia")
    if "Edema" in diseases_list:
        lung_findings.append("vascular congestion and interstitial edema")
    if "Atelectasis" in diseases_list:
        lung_findings.append("linear densities indicating atelectasis")
    if "Normal" in diseases_list:
        lung_findings.append("clear fields bilaterally")

    if lung_findings:
        findings_html += f"The lungs show {', '.join(lung_findings)}. "
    elif "Cardiomegaly" not in diseases_list:
        findings_html += "The lungs are clear. "

    pleura_findings = []
    if "PleuralEffusion" in diseases_list:
        pleura_findings.append("blunting of costophrenic angles (effusion)")
    if "Pneumothorax" in diseases_list:
        pleura_findings.append("visible visceral pleural line (pneumothorax)")

    if pleura_findings:
        findings_html += f"Pleural evaluation reveals {', '.join(pleura_findings)}. "
    else:
        findings_html += "No pleural effusion or pneumothorax is identified. "

    if "Fracture" in diseases_list:
        findings_html += "There is evidence of <b>osseous discontinuity</b> suggestive of fracture. "
    else:
        findings_html += "Osseous structures appear intact. "

    impression_text = (
        f"1. Findings consistent with <b>{', '.join(diseases_list).upper()}</b>.<br>"
        "2. Clinical correlation recommended."
    )

    html_code = f"""
<div class="paper-report">
    <div class="report-header">
        <div><b>PATIENT ID:</b> [AUTO-GENERATED]</div>
        <div><b>DATE:</b> {date_str}</div>
        <div><b>STATUS:</b> <span style="color:#16a34a; font-weight:bold;">FINALIZED</span></div>
    </div>
    <div class="report-section">
        <div class="section-title">Findings</div>
        <div>{findings_html}</div>
    </div>
    <div class="report-section">
        <div class="section-title">Impression</div>
        <div>{impression_text}</div>
    </div>
    <div class="report-footer">
        <div>Generated by MedVision AI (Rule-based Engine)</div>
        <div>Electronically Signed</div>
    </div>
    <div class="ai-warning">
        <b>DISCLAIMER:</b> This report is an AI-generated draft. It must be reviewed and finalized by a certified Radiologist before clinical use.
    </div>
</div>
"""
    return html_code


def generate_llm_report_html(report_text: str) -> str:
    """
    Wrap LLM text in a dark "AI Copilot" panel.
    """
    return f"""
<div style="
    margin-top: 20px;
    background: #FFFFFF;
    color: #000000;
    padding: 24px;
    border-radius: 12px;
    border: 1px solid #1e293b;
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    white-space: pre-wrap;
    line-height: 1.6;
">
    <div style="color:#000000; font-weight:700; margin-bottom:10px; font-size:0.95rem;">
        AI Copilot Report
    </div>
    {report_text}
    <div style="margin-top:16px; font-size:0.75rem; color:#9ca3af;">
        NOTE: This draft is generated by a local LLM (DeepSeek-R1 via Neuro-Symbolic pipeline) and must be reviewed by a certified radiologist.
    </div>
</div>
"""


# ==========================================
# 3.1 LLM wiring (DeepSeek-R1 via Ollama)
# ==========================================
def build_radiology_prompt(predicted_labels_text, rag_context_text, kg_definitions):
    """
    Constructs the prompt for the LLM specifically for Radiology Report Generation.
    """
    kg_str_lines = []
    for d, definition in kg_definitions.items():
        kg_str_lines.append(f"- {d}: {definition}")
    kg_block = "\n".join(kg_str_lines)

    prompt = f"""
You are an expert radiologist assistant specialized in chest imaging.
Your task is to generate a clinically accurate and concise chest X-ray report.

INPUT DATA:
1. PREDICTED PATHOLOGIES (from Vision Model):
{predicted_labels_text}

2. SIMILAR CASES CONTEXT (from RAG System):
{rag_context_text}

3. MEDICAL KNOWLEDGE RULES (from Knowledge Graph):
{kg_block}

TASK:
- Write a structured report with two sections: FINDINGS and IMPRESSION.
- In FINDINGS: describe the heart, lungs, pleura, and bones clearly.
- In IMPRESSION: summarize the key diagnoses concisely (1–3 bullet points).

STRICT RULES:
- You MUST NOT introduce any new diseases or findings that are not among the predicted pathologies listed above.
- Limit all findings and abnormalities to these possible labels:
  Cardiomegaly, Pneumonia, Atelectasis, Edema, PleuralEffusion, Pneumothorax, Fracture, Normal.
- You may describe a disease as present or absent, but do not invent extra conditions.
- Use only medically factual statements supported by the input data.
- Avoid speculative terms like "might" or "could be".
- Do NOT invent laterality (right/left) or lobar/segmental anatomy (e.g., "right upper lobe", "RUL"). Only mention such specific locations if the exact term appears in the SIMILAR CASES CONTEXT.
- If location evidence is weak, use generic wording (e.g., "in the lungs" / "in the right lung") rather than precise lobes.

- Do not mention that you are an AI. Do not include your reasoning steps.

OUTPUT FORMAT:
Return the report as plain text using this structure:

FINDINGS:
...

IMPRESSION:
...
"""
    return prompt.strip()


def generate_report_draft(predicted_labels_text: str, rag_context_text: str, kg_definitions: Dict[str, str]) -> str:
    """
    Call local DeepSeek-R1 (Ollama) and strip <think> blocks.
    """
    prompt = build_radiology_prompt(predicted_labels_text, rag_context_text, kg_definitions)

    try:
        client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # dummy key required by SDK
        )

        response = client.chat.completions.create(
            model="deepseek-r1:latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000,
        )

        raw_content = response.choices[0].message.content
        clean_content = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
        return clean_content or "[LLM returned empty content]"

    except Exception as e:
        return f"[Error] Could not connect to local LLM (DeepSeek-R1/Ollama): {e}"

def revise_report_with_violations(
    original_report: str,
    violations: List[str],
    predicted_labels_text: str,
    rag_context_text: str,
    kg_definitions: Dict[str, str],
) -> str:
    """
    One-shot repair: ask the local LLM to rewrite the report and fix verifier violations.
    Keeps the same constraints as the main prompt.
    """
    base_prompt = build_radiology_prompt(predicted_labels_text, rag_context_text, kg_definitions)

    fix_prompt = base_prompt + "\n\n" + f"""REVISION REQUEST:
Your previous report violated constraints.

VIOLATIONS:
{chr(10).join("- " + v for v in violations)}

INSTRUCTIONS:
- Rewrite the report to remove ALL violations.
- Keep the same FINDINGS / IMPRESSION format.
- Do NOT add any new diagnoses beyond: {predicted_labels_text}
- If you mention a label not in predicted list, it must be only as explicitly ABSENT.
"""

    try:
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        response = client.chat.completions.create(
            model="deepseek-r1:latest",
            messages=[{"role": "user", "content": fix_prompt}],
            temperature=0.2,
            max_tokens=1000,
        )
        raw = response.choices[0].message.content
        clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        return clean or original_report
    except Exception:
        return original_report



# ==========================================
# 4. UI
# ==========================================
with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("Upload X-Ray", type=["jpg", "png"])

    # Persist the uploaded image bytes across reruns (prevents disappearing images)
    # and reset analysis state when a NEW image is uploaded (prevents cross-case contamination).
    if uploaded_file is not None:
        _bytes = uploaded_file.getvalue()
        _sig = hashlib.md5(_bytes).hexdigest()
        if st.session_state.get('uploaded_sig') != _sig:
            st.session_state['uploaded_sig'] = _sig
            st.session_state['uploaded_name'] = uploaded_file.name
            st.session_state['image_bytes'] = _bytes
            # Reset state for a new case
            st.session_state['res'] = None
            st.session_state['show_report'] = False
            st.session_state['llm_report'] = None
            st.session_state['xai_heat'] = None
            st.session_state['xai_label'] = None
            st.session_state['xai_grid'] = None
        else:
            # Keep bytes fresh (some browsers re-send the object)
            st.session_state['image_bytes'] = _bytes

    st.markdown("### Viewport")
    b_val = st.slider("Brightness", 0.5, 2.0, 1.0)
    c_val = st.slider("Contrast", 0.5, 2.0, 1.0)

    st.markdown("### AI Logic")
    num_results = st.slider("Retrieval Count", 1, 5, 3)

    conf_threshold = st.slider("Global Threshold", 0.1, 0.9, 0.35)
    use_per_class_thr = st.checkbox(
        "Use per-class thresholds",
        value=(thr_vec is not None),
        disabled=(thr_vec is None),
        help="Uses thresholds optimized on a validation split. Falls back to the global threshold if not available.",
    )

    analyze_btn = False
    if uploaded_file and df_final is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("RUN ANALYSIS", type="primary")


if uploaded_file and df_final is not None:
    st.markdown(
        f"""
    <div class="patient-card">
        <div><span class="patient-text">ID:</span> <span class="patient-val">{uploaded_file.name.split('.')[0].upper()}</span></div>
        <div><span class="patient-text">DOB:</span> <span class="patient-val">N/A</span></div>
        <div><span class="patient-text">SEX:</span> <span class="patient-val">Unknown</span></div>
        <div><span class="patient-text">MODALITY:</span> <span class="patient-val">Uploaded image</span></div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1.5], gap="large")

    # ---------- Left column: image + saliency ----------
    with col1:
        img_bytes = st.session_state.get("image_bytes") or uploaded_file.getvalue()
        raw_img = Image.open(BytesIO(img_bytes)).convert("RGB")
        disp_img = process_display_image(raw_img, b_val, c_val)
        st.image(disp_img, use_container_width=True, caption="Patient Radiograph")

        show_xai = st.checkbox("Show CLIP embedding saliency heatmap (not disease-specific)", value=False)

        if show_xai and st.session_state.get("res") is not None:
            if st.session_state.res.get("is_xray_like", True):
                heat = compute_clip_saliency(raw_img, model, preprocess, device)
                heat_resized = cv2.resize(heat, (raw_img.size[0], raw_img.size[1]))

                fig, ax = plt.subplots()
                ax.imshow(raw_img, cmap="gray")
                ax.imshow(heat_resized, cmap="jet", alpha=0.4)
                ax.axis("off")
                st.pyplot(fig)


                # Persistent state for label-specific XAI across reruns
        if "xai_heat" not in st.session_state:
            st.session_state.xai_heat = None
        if "xai_label" not in st.session_state:
            st.session_state.xai_label = None
        if "xai_grid" not in st.session_state:
            st.session_state.xai_grid = 6

        show_xai_occl = st.checkbox("Show disease-specific occlusion heatmap (slow)", value=False, key="chk_xai_occl")

        if show_xai_occl:
            if st.session_state.get("res") is None:
                st.info("Run analysis first to compute predictions, then you can generate an explanation.")
            else:
                _res = st.session_state.res
                if not _res.get("is_xray_like", True):
                    st.info("Explainability is available only for in-scope chest X-rays.")
                else:
                    pred_labels = [l for l in (_res.get("pred") or []) if l != "Normal"]
                    if not pred_labels:
                        pred_labels = list(_res.get("pred") or [])
                    if pred_labels:
                        # Keep the selection stable across reruns
                        default_label = st.session_state.xai_label if (st.session_state.xai_label in pred_labels) else pred_labels[0]
                        label_to_explain = st.selectbox(
                            "Explainability target label",
                            pred_labels,
                            index=pred_labels.index(default_label),
                            key="sel_xai_label"
                        )
                        grid = st.slider(
                            "Occlusion grid (higher = slower)",
                            4, 10,
                            int(st.session_state.xai_grid) if st.session_state.xai_grid else 6,
                            key="sld_xai_grid"
                        )

                        if st.button("Compute label-specific explanation", key="btn_xai_occl"):
                            st.session_state.xai_label = label_to_explain
                            st.session_state.xai_grid = grid
                            heat2 = compute_occlusion_heatmap(img_bytes, label_to_explain, grid=grid)
                            st.session_state.xai_heat = heat2

                        # IMPORTANT: show the latest computed explanation on every rerun
                        heat2 = st.session_state.get("xai_heat")
                        if heat2 is None:
                            st.caption("No explanation computed yet for this case.")
                        else:
                            fig2, ax2 = plt.subplots()
                            ax2.imshow(raw_img, cmap="gray")
                            ax2.imshow(heat2, cmap="jet", alpha=0.4)
                            ax2.axis("off")
                            st.pyplot(fig2)
                            st.caption(
                                "Occlusion sensitivity for **{}** (grid={}). Bright regions mean masking that area reduced the label probability the most.".format(
                                    st.session_state.xai_label, st.session_state.xai_grid
                                )
                            )


# ---------- Right column: prediction + retrieval + reports ----------
    with col2:
        if "res" not in st.session_state:
            st.session_state.res = None
        if "show_report" not in st.session_state:
            st.session_state.show_report = False
        if "llm_report" not in st.session_state:
            st.session_state.llm_report = None

        if analyze_btn:
            st.session_state.show_report = False
            st.session_state.llm_report = None
            st.session_state.xai_heat = None

            with st.spinner("Initializing Neuro-Symbolic Engine..."):
                ai_img = prepare_for_ai(raw_img)
                emb_raw = get_image_embedding(ai_img)

                sims_all = cosine_similarity(emb_raw.reshape(1, -1), database_embeddings)[0]
                max_sim = float(sims_all.max())
                is_xray_like = max_sim >= XRAY_SIM_THRESHOLD

                kg_warnings: List[str] = []

                probs_list = None
                class_names = None

                if (not is_xray_like) or (clf is None) or (mlb is None) or (scaler is None):
                    pred_diseases = ["Non-chest image (out of scope)"] if not is_xray_like else ["Model not available"]
                    avg_conf = max_sim
                    filtered_top_k = []
                    sims = sims_all
                else:
                    emb_scaled = scaler.transform(emb_raw.reshape(1, -1))
                    raw_probs = clf.predict_proba(emb_scaled)[0]

                    probs_source = "raw"
                    probs = raw_probs

                    # If calibrators exist, use calibrated probabilities for UI + thresholding
                    if calibrators_aligned is not None:
                        try:
                            probs = np.array([cal.predict_proba(emb_scaled)[:, 1][0] for cal in calibrators_aligned], dtype=float)
                            probs_source = "calibrated"
                        except Exception:
                            probs = raw_probs
                            probs_source = "raw"


                    probs_list = [float(x) for x in probs]
                    class_names = list(mlb.classes_)


                    pred_indices = select_pred_indices(
                        probs=probs,
                        conf_threshold=conf_threshold,
                        use_per_class_thr=use_per_class_thr,
                        thr_vec_aligned=thr_vec,
                    )

                    if len(pred_indices) > 0:
                        pred_diseases = [mlb.classes_[i] for i in pred_indices]
                        pred_diseases, kg_warnings = kg_validate_diseases(pred_diseases, kg_info)

                        kept_indices = [i for i in pred_indices if mlb.classes_[i] in pred_diseases]
                        avg_conf = float(np.mean([probs[i] for i in kept_indices])) if kept_indices else float(
                            np.max(probs)
                        )
                    else:
                        pred_diseases = ["Normal"]
                        pred_diseases, kg_warnings = kg_validate_diseases(pred_diseases, kg_info)
                        avg_conf = float(np.max(probs))

                    sims = sims_all
                    top_k = sims.argsort()[::-1]
                    filtered_top_k = []
                    for idx in top_k:
                        if sims[idx] < 0.9999:
                            filtered_top_k.append(idx)
                        if len(filtered_top_k) >= num_results:
                            break

                st.session_state.res = {
                    "pred": pred_diseases,
                    "conf": avg_conf,
                    "conf_label": f"Mean predicted probability (selected labels) — {probs_source}" if "probs_source" in locals() else "Mean predicted probability (selected labels)",
                    "probs": probs_list,
                    "raw_probs": [float(x) for x in raw_probs] if "raw_probs" in locals() else None,
                    "probs_source": probs_source if "probs_source" in locals() else "raw",
                    "class_names": class_names,
                    "top_k": filtered_top_k,
                    "sims": sims,
                    "search_df": df_final,
                    "is_xray_like": is_xray_like,
                    "max_sim": max_sim,
                    "kg_warnings": kg_warnings,
                }

        if st.session_state.res:
            res = st.session_state.res
            search_df = res["search_df"]

            pred_text = ", ".join(res["pred"])
            if not res.get("is_xray_like", True):
                pred_color = "#64748b"
            else:
                pred_color = "#10b981" if ("Normal" in res["pred"] and len(res["pred"]) == 1) else "#ef4444"

            st.markdown(
                f"""
            <div class="prediction-card">
                <div class="pred-title">AI Diagnosis (Multi-Label)</div>
                <div class="pred-value" style="color: {pred_color}">{pred_text}</div>
                <span class="pred-conf">{res.get("conf_label","Mean predicted probability (selected labels)")}: {res["conf"]:.2%}</span><br/>
                <span class="pred-conf">Max similarity to chest database: {res['max_sim']:.2%}</span>
            </div>
            """,
                unsafe_allow_html=True,
            )

            if res.get("probs") is not None and res.get("class_names") is not None:
                if len(res["probs"]) == len(res["class_names"]):
                    with st.expander(f"View class probabilities ({res.get('probs_source','raw')})"):
                        prob_df = pd.DataFrame({"label": res["class_names"], "probability": res["probs"]})
                        prob_df = prob_df.sort_values("probability", ascending=False)
                        st.dataframe(prob_df, use_container_width=True)
                        st.caption("Calibrated probabilities (sigmoid / Platt scaling) on a held-out set. Still not a clinical device." if res.get("probs_source")=="calibrated" else "These are raw model scores and may not be clinically calibrated.")


            if res.get("kg_warnings"):
                for w in res["kg_warnings"]:
                    st.warning(w)

            if not res.get("is_xray_like", True):
                st.warning(
                    "Uploaded image does not look like a chest X-ray. "
                    "No similar radiology cases are retrieved and no formal report is generated."
                )
            else:
                # KG facts
                if kg_info:
                    with st.expander("Knowledge Graph Facts (KG)", expanded=False):
                        for d in res["pred"]:
                            info = kg_info.get(d)
                            if not info:
                                continue
                            locs = info.get("locations", [])
                            definition = info.get("definition", "")
                            st.markdown(f"**{d}**")
                            st.write(f"Locations: {locs}" if locs else "Locations: N/A")
                            if definition:
                                st.caption(definition)
                            st.markdown("---")

                # Retrieval
                st.markdown("##### Evidence Retrieval (Similar Cases)")
                rag_context_lines: List[str] = []  # we'll reuse for LLM

                if len(res["top_k"]) > 0:
                    tabs = st.tabs([f"Match #{i+1}" for i in range(len(res["top_k"]))])

                    for i, tab in enumerate(tabs):
                        idx = res["top_k"][i]
                        row = search_df.iloc[idx]
                        score = res["sims"][idx]

                        ents = (
                            row["detected_entities"]
                            if ("detected_entities" in row and isinstance(row["detected_entities"], list))
                            else ["Normal"]
                        )

                        # build RAG text
                        rag_line = filter_rag_caption_for_labels(str(row.get('caption','')), res.get('pred', []))

                        rag_context_lines.append(f"Similar Case {i+1}: {rag_line}")

                        with tab:
                            c_a, c_b = st.columns([1, 2.5])
                            with c_a:
                                if os.path.exists(row["image_path"]):
                                    st.image(row["image_path"], use_container_width=True)
                                else:
                                    st.warning("Original image file not found.")
                                st.caption(f"**Similarity:** {score:.2%}")

                            with c_b:
                                st.markdown("**Diagnosed Pathology:** " + " ".join([f"`{e}`" for e in ents]))
                                pred_set = set(res.get('pred', [])) if isinstance(res.get('pred'), list) else set()
                                ent_set = set(ents) if isinstance(ents, list) else set()
                                overlap = sorted(list((pred_set & ent_set) - {'Normal'}))
                                st.caption('Label overlap with prediction: ' + (', '.join(overlap) if overlap else 'None'))

                                with st.expander("View Radiologist Report", expanded=True):
                                    st.markdown(highlight_text(row["caption"], ents), unsafe_allow_html=True)
                else:
                    st.info("No similar cases retrieved for RAG context.")

                # ---------- LLM report generation ----------
                if res.get("is_xray_like", True):
                    if st.button("Generate Final Report Draft (LLM)", use_container_width=True):
                        st.session_state.show_report = True
                        predicted_labels_text = ", ".join(res["pred"])
                        rag_context_text = "\n".join(rag_context_lines)

                        with st.spinner("Calling local LLM (DeepSeek-R1 via Ollama)..."):
                            llm_text = generate_report_draft(
                                predicted_labels_text, rag_context_text, kg_definitions
                            )
                        # --- Neuro-symbolic verifier + automatic revision (up to 2 passes) ---
                        allowed_labels = list(mlb.classes_) if mlb is not None else (res.get('pred', []) if isinstance(res.get('pred'), list) else [])

                        revision_history = [llm_text]
                        final_text = llm_text
                        revised = False

                        violations_now, warnings_now = verify_report_with_kg(
                            final_text, res.get('pred', []), kg_info, allowed_labels
                        )

                        # Try to automatically revise the report if it violates constraints.
                        # IMPORTANT: we only show warnings/violations for the FINAL report (avoid confusing carry-over).
                        for _attempt in range(2):
                            if not violations_now:
                                break
                            revised = True
                            final_text = revise_report_with_violations(
                                final_text,
                                violations_now,
                                predicted_labels_text,
                                rag_context_text,
                                kg_definitions,
                            )
                            revision_history.append(final_text)
                            violations_now, warnings_now = verify_report_with_kg(
                                final_text, res.get('pred', []), kg_info, allowed_labels
                            )

                        st.session_state.llm_report = final_text
                        st.session_state.res['verifier_violations'] = list(violations_now)
                        st.session_state.res['verifier_warnings'] = list(warnings_now)
                        st.session_state.res['verifier_revised'] = revised
                        st.session_state.res['revision_history'] = revision_history


                    if st.session_state.get("show_report"):
                        if st.session_state.llm_report:
                            if res.get('verifier_revised'):
                                st.info('LLM draft was automatically revised to satisfy KG constraints.')

                            if res.get('verifier_violations'):
                                with st.expander('Verifier violations (needs manual review)', expanded=True):
                                    for v in res['verifier_violations']:
                                        st.error(v)

                            if res.get('verifier_warnings'):
                                with st.expander('Verifier warnings', expanded=False):
                                    for w in res['verifier_warnings']:
                                        st.warning(w)

                            if res.get('revision_history') and isinstance(res.get('revision_history'), list) and len(res.get('revision_history')) > 1:
                                with st.expander('Revision trail (draft → final)', expanded=False):
                                    for j, txt in enumerate(res['revision_history']):
                                        st.markdown(f"**Version {j+1}**")
                                        st.code(txt)
                                        st.markdown('---')

                            if res.get('verifier_violations'):
                                st.warning("LLM draft still violates constraints; showing rule-based report. You can inspect the draft below.")
                                with st.expander("LLM draft (violates constraints)", expanded=False):
                                    st.markdown(
                                        generate_llm_report_html(st.session_state.llm_report),
                                        unsafe_allow_html=True,
                                    )
                                rb_html = generate_report_html_rule_based(res["pred"])
                                st.markdown(rb_html, unsafe_allow_html=True)
                            else:
                                st.markdown(
                                    generate_llm_report_html(st.session_state.llm_report),
                                    unsafe_allow_html=True,
                                )
                        else:
                            # fallback rule-based
                            rb_html = generate_report_html_rule_based(res["pred"])
                            st.markdown(rb_html, unsafe_allow_html=True)
        # end if res

elif df_final is None:
    st.error("Database/Model files missing. Please run the Jupyter Notebook first to generate the .pkl files.")
else:
    st.markdown(
        "<div style='text-align:center; padding:50px; color:#aaa;'><h3>Ready for Analysis</h3></div>",
        unsafe_allow_html=True,
    )
