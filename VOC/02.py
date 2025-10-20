#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit VOC Topic Explorer (JSON input)
----------------------------------------

기능 요약
- JSON/NDJSON VOC 파일 업로드 → 임베딩 → (선택) UMAP → HDBSCAN 클러스터링 → c-TF-IDF 키워드 → 자동 라벨 → 대시보드 시각화
- 혹은, 이전 파이프라인 산출물(문서별 docs.jsonl + summary.json) 업로드하여 즉시 시각화
- UMAP 2D 인터랙티브 산점도(Plotly), 클러스터 크기 막대그래프, 군집별 Top 키워드, 샘플 문장, (선택) 시계열 추이
- 결과(문서별, 요약)를 JSON/CSV로 다운로드

실행 방법
1) 의존성 설치:
   pip install -U streamlit pandas numpy scikit-learn sentence-transformers hdbscan umap-learn plotly
   # (옵션) BERTopic 사용시
   pip install -U bertopic

2) 앱 실행:
   streamlit run app.py

입력 형식
- 원본 JSON: 배열 JSON([{}, {}, ...]) 또는 NDJSON(줄마다 하나의 JSON)
- 텍스트 필드가 여러 곳에 있으면 점표기 경로로 다중 선택(예: title, body, user.comment)
- 미리 계산된 결과 사용: voc_topics_docs.jsonl + voc_topics_summary.json 업로드

메모
- 대용량에서 UMAP/임베딩은 시간이 걸릴 수 있습니다. 샘플링 옵션을 제공하며, 최소 파라미터로 먼저 돌려보세요.
- 색상은 기본 팔레트 사용(지정하지 않음), 한 차트에 한 Figure 원칙은 matplotlib에서의 제약이지만 여기선 Plotly만 사용합니다.
"""

import io
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# 임베딩/차원축소/클러스터링
from sentence_transformers import SentenceTransformer

try:
    import umap  # type: ignore
    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False

import hdbscan  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from bertopic import BERTopic  # type: ignore
    _HAS_BERTOPIC = True
except Exception:
    _HAS_BERTOPIC = False

# ---------------------------
# 유틸: JSON 읽기/플래튼/텍스트 결합
# ---------------------------
_URL_RE = re.compile(r"http[s]?://\S+")
_EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+")
_PHONE_RE = re.compile(r"\b\d{2,4}[-\s]?\d{2,4}[-\s]?\d{2,4}\b")


def clean_text(s: str) -> str:
    s = str(s)
    s = _URL_RE.sub("<URL>", s)
    s = _EMAIL_RE.sub("<EMAIL>", s)
    s = _PHONE_RE.sub("<PHONE>", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _read_json_or_ndjson(file: io.BytesIO) -> List[Dict[str, Any]]:
    raw = file.getvalue().decode("utf-8", errors="ignore").strip()
    if not raw:
        return []
    if raw.startswith("[") and raw.endswith("]"):
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("Top-level JSON must be a list.")
        return data
    # NDJSON
    rows = []
    for ln, line in enumerate(raw.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception as e:
            raise ValueError(f"Invalid NDJSON at line {ln}: {e}")
        rows.append(obj)
    return rows


def _flatten(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.extend(_flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def discover_keys(records: List[Dict[str, Any]], sample_n: int = 200) -> List[str]:
    keys = set()
    for rec in records[:sample_n]:
        if isinstance(rec, dict):
            flat = _flatten(rec)
            for k in flat.keys():
                keys.add(k)
    return sorted(keys)


def join_fields(flat: Dict[str, Any], fields: List[str]) -> str:
    parts: List[str] = []
    for f in fields:
        val = flat.get(f)
        if val is None:
            continue
        if isinstance(val, (list, tuple)):
            parts.append(" ".join([str(x) for x in val if x is not None]))
        else:
            parts.append(str(val))
    return " ".join([p for p in parts if p]).strip()


# ---------------------------
# 캐시: 임베딩 모델/계산
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_model(name: str) -> SentenceTransformer:
    return SentenceTransformer(name)


@st.cache_data(show_spinner=True)
def compute_embeddings(texts: List[str], model_name: str) -> np.ndarray:
    model = load_model(model_name)
    emb = model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(emb, dtype=np.float32)


@st.cache_data(show_spinner=True)
def run_umap(emb: np.ndarray, n_components: int = 2, random_state: int = 42) -> np.ndarray:
    if not _HAS_UMAP:
        raise RuntimeError("umap-learn 미설치: pip install umap-learn")
    reducer = umap.UMAP(n_neighbors=15, n_components=n_components, min_dist=0.0, metric="cosine", random_state=random_state)
    coords = reducer.fit_transform(emb)
    return coords


@st.cache_data(show_spinner=True)
def run_hdbscan(X: np.ndarray, min_cluster_size: int = 20) -> np.ndarray:
    cl = hdbscan.HDBSCAN(min_cluster_size=max(5, int(min_cluster_size)), metric="euclidean", cluster_selection_method="eom", prediction_data=True)
    return cl.fit_predict(X)


# ---------------------------
# 키워드/라벨
# ---------------------------
@st.cache_data(show_spinner=False)
def c_tf_idf_keywords(big_docs: List[str], topn: int = 10) -> Tuple[List[List[str]], List[int]]:
    vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2), analyzer="word", lowercase=True)
    X = vec.fit_transform(big_docs)
    terms = np.array(vec.get_feature_names_out())
    top_idx_mat = np.argsort(X.toarray(), axis=1)[:, ::-1][:, :topn]
    top_terms = [[terms[j] for j in row] for row in top_idx_mat]
    vocab_size = int(len(terms))
    return top_terms, [vocab_size] * len(big_docs)


_KO_STOP = set([
    "그리고","그러나","하지만","또는","또","및","그래서","정도","부분","사용","문제","현상",
    "문의","요청","관련","발생","이슈","해결","가능","확인","제대로","전체","일부","사용자",
    "발송","진행","상태","이후","현재","필요","적용","업데이트","버전","정보","데이터"
])


def label_from_keywords(kws: List[str], max_tokens: int = 3) -> str:
    filtered: List[str] = []
    for w in kws:
        w2 = w.replace("_", " ").replace("-", " ").strip()
        if not w2 or len(w2) <= 1 or w2 in _KO_STOP:
            continue
        filtered.append(w2)
    if not filtered:
        filtered = kws[:max_tokens]
    return " / ".join(filtered[:max_tokens])


# ---------------------------
# SEED 로딩/매칭 + 하이브리드 파이프라인 유틸
# ---------------------------
def load_seeds_from_inputs(file, text_input: str):
    """CSV(category,type,value) 또는 JSON 배열을 받아 표준 dict로 정규화
    return {category: {"keyword": [...], "regex": [...], "example": [...]}}
    """
    import io as _io
    seeds = {}
    # 파일 우선
    if file is not None:
        content = file.getvalue().decode("utf-8", errors="ignore").strip()
        if content:
            if file.name.lower().endswith(".json"):
                try:
                    arr = json.loads(content)
                    assert isinstance(arr, list)
                    for row in arr:
                        cat = str(row.get("category",""))
                        typ = str(row.get("type","keyword")).lower()
                        val = row.get("value", "")
                        if not cat or not val:
                            continue
                        seeds.setdefault(cat, {"keyword":[],"regex":[],"example":[]})
                        if isinstance(val, list):
                            seeds[cat][typ].extend([str(v) for v in val])
                        else:
                            seeds[cat][typ].append(str(val))
                except Exception:
                    st.warning("Seed JSON 파싱 실패: CSV로 다시 시도합니다.")
            else:
                try:
                    df = pd.read_csv(_io.StringIO(content))
                    for _, r in df.iterrows():
                        cat = str(r.get("category",""))
                        typ = str(r.get("type","keyword")).lower()
                        val = str(r.get("value",""))
                        if not cat or not val:
                            continue
                        seeds.setdefault(cat, {"keyword":[],"regex":[],"example":[]})
                        seeds[cat][typ].append(val)
                except Exception as e:
                    st.error(f"Seed CSV 파싱 실패: {e}")
    # 텍스트 입력(선택)
    if text_input and text_input.strip():
        try:
            arr = json.loads(text_input)
            assert isinstance(arr, list)
            for row in arr:
                cat = str(row.get("category",""))
                typ = str(row.get("type","keyword")).lower()
                val = row.get("value", "")
                if not cat or not val:
                    continue
                seeds.setdefault(cat, {"keyword":[],"regex":[],"example":[]})
                if isinstance(val, list):
                    seeds[cat][typ].extend([str(v) for v in val])
                else:
                    seeds[cat][typ].append(str(val))
        except Exception as e:
            st.error(f"Seed JSON 텍스트 파싱 실패: {e}")
    return seeds


def seed_assignments(df: pd.DataFrame, seeds: dict, model_name: str,
                     use_kw_regex: bool = True, use_examples: bool = True, example_thresh: float = 0.55,
                     doc_emb: Optional[np.ndarray] = None):
    """사전 카테고리 할당. 반환: (df[seed_* 컬럼 채움], 임베딩)"""
    df = df.copy()
    df["seed_category"] = None
    df["seed_method"] = None
    df["seed_score"] = 0.0
    if not seeds:
        return df, doc_emb

    # 1) 키워드/정규식 매칭
    if use_kw_regex:
        texts_lower = df["text_clean"].astype(str).str.lower().tolist()
        compiled_regex = {cat: [re.compile(p, flags=re.IGNORECASE) for p in spec.get("regex", []) if str(p).strip()]
                          for cat, spec in seeds.items()}
        for i, t in enumerate(texts_lower):
            best_cat, best_score = None, 0
            for cat, spec in seeds.items():
                score = 0
                for kw in spec.get("keyword", []):
                    k = str(kw).lower().strip()
                    if not k:
                        continue
                    score += t.count(k)
                for rgx in compiled_regex.get(cat, []):
                    score += len(rgx.findall(t))
                if score > best_score:
                    best_cat, best_score = cat, score
            if best_cat and best_score > 0:
                df.loc[df.index[i], ["seed_category","seed_method","seed_score"]] = [best_cat, "kw/regex", float(best_score)]

    # 2) 예시문장 임베딩 매칭
    if use_examples:
        if doc_emb is None:
            doc_emb = compute_embeddings(df["text_clean"].astype(str).tolist(), model_name)
        cat_proto = {}
        for cat, spec in seeds.items():
            exs = [clean_text(x) for x in spec.get("example", []) if str(x).strip()]
            if not exs:
                continue
            vecs = compute_embeddings(exs, model_name)
            proto = vecs.mean(axis=0)
            proto = proto / (np.linalg.norm(proto) + 1e-12)
            cat_proto[cat] = proto
        if cat_proto:
            protos = np.stack(list(cat_proto.values()))
            cats = list(cat_proto.keys())
            sims = doc_emb @ protos.T
            mask_un = df["seed_category"].isna().values
            for i, un in enumerate(mask_un):
                if not un:
                    continue
                row = sims[i]
                j = int(np.argmax(row))
                s = float(row[j])
                if s >= example_thresh:
                    df.loc[df.index[i], ["seed_category","seed_method","seed_score"]] = [cats[j], "example", s]
    return df, doc_emb


def pipeline_from_raw_hybrid(records: List[Dict[str, Any]], text_fields: List[str], id_field: Optional[str], model_name: str,
                             min_topic_size: int, umap_components: int, engine: str,
                             seeds: Optional[dict] = None,
                             use_kw_regex: bool = True, use_examples: bool = True, example_thresh: float = 0.55,
                             exclude_seeded_from_clustering: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # 원본 → DF
    rows = []
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            continue
        flat = _flatten(rec)
        doc_id = flat.get(id_field) if id_field else None
        if doc_id is None:
            doc_id = f"doc_{i+1}"
        text = join_fields(flat, text_fields)
        rows.append({"id": doc_id, "text": text})
    df = pd.DataFrame(rows)
    df["text_clean"] = df["text"].map(clean_text)
    df = df[df["text_clean"].str.len() > 5].drop_duplicates(subset=["text_clean"]).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("유효한 텍스트가 없습니다.")

    # SEED 사전할당
    emb_for_examples = None
    if seeds:
        df, emb_for_examples = seed_assignments(df, seeds, model_name, use_kw_regex, use_examples, example_thresh, None)
    else:
        df["seed_category"], df["seed_method"], df["seed_score"] = None, None, 0.0

    # 클러스터링 대상
    if exclude_seeded_from_clustering and seeds:
        df_cl = df[df["seed_category"].isna()].copy()
    else:
        df_cl = df.copy()

    # 임베딩 및 클러스터링
    emb = compute_embeddings(df_cl["text_clean"].tolist(), model_name)
    if engine == "bertopic":
        if not _HAS_BERTOPIC:
            raise RuntimeError("BERTopic 미설치: pip install bertopic")
        topic_model = BERTopic(language="multilingual", n_gram_range=(1,2), min_topic_size=max(5, int(min_topic_size)), calculate_probabilities=True, verbose=False)
        topics, _ = topic_model.fit_transform(df_cl["text_clean"].tolist(), embeddings=emb)
        df_cl["cluster"] = topics
        info = topic_model.get_topic_info()
        summary_records = []
        for _, row in info.iterrows():
            cid = int(row["Topic"])
            if cid == -1:
                continue
            topic_terms = topic_model.get_topic(cid) or []
            top_terms = [t for t, _ in topic_terms[:10]]
            auto_label = label_from_keywords(top_terms, max_tokens=3)
            idxs = df_cl.index[df_cl["cluster"] == cid].tolist()
            sample_texts = [df_cl.loc[k, "text_clean"] for k in idxs[: min(3, len(idxs))]]
            summary_records.append({"cluster": cid, "size": int(row["Count"]), "auto_label": auto_label, "keywords": top_terms, "sample_texts": sample_texts})
        sum_df = pd.DataFrame(summary_records).sort_values("size", ascending=False).reset_index(drop=True)
    else:
        X = emb
        if umap_components > 0:
            X = run_umap(emb, n_components=umap_components)
        labels = run_hdbscan(X, min_cluster_size=min_topic_size)
        df_cl["cluster"] = labels
        by_cluster = {}
        for i, c in enumerate(labels):
            if c == -1:
                continue
            by_cluster.setdefault(int(c), []).append(df_cl.index[i])
        summary_records, cid_to_terms, cid_to_label = [], {}, {}
        if by_cluster:
            big_docs = [" ".join(df_cl.loc[idxs, "text_clean"]) for _, idxs in sorted(by_cluster.items())]
            clusters_sorted = [cid for cid, _ in sorted(by_cluster.items())]
            top_terms_list, _ = c_tf_idf_keywords(big_docs, topn=10)
            for cid, terms in zip(clusters_sorted, top_terms_list):
                auto_label = label_from_keywords(terms, max_tokens=3)
                cid_to_terms[cid] = terms
                cid_to_label[cid] = auto_label
                idxs = by_cluster[cid]
                sample_texts = [df_cl.loc[k, "text_clean"] for k in idxs[: min(3, len(idxs))]]
                summary_records.append({"cluster": cid, "size": len(idxs), "auto_label": auto_label, "keywords": terms, "sample_texts": sample_texts})
        sum_df = pd.DataFrame(summary_records).sort_values("size", ascending=False).reset_index(drop=True)

    # 원본 df에 병합 및 최종 라벨
    if "cluster" in df.columns:
        df = df.drop(columns=["cluster"])
    df = df.merge(df_cl[["cluster"]], left_index=True, right_index=True, how="left")
    df["cluster"] = df["cluster"].fillna(-2).astype(int)  # -2: SEED로만 할당된 문서

    cid_to_label = dict(zip(sum_df["cluster"], sum_df["auto_label"])) if not sum_df.empty else {}
    df["auto_label"] = df["cluster"].map(lambda c: cid_to_label.get(c, "기타/노이즈") if c != -2 else None)

    df["final_label"] = df.apply(lambda r: r["seed_category"] if pd.notna(r["seed_category"]) and r["seed_category"] else (r["auto_label"] if r["auto_label"] else "기타/노이즈"), axis=1)
    df["label_source"] = df.apply(lambda r: "SEED" if pd.notna(r["seed_category"]) and r["seed_category"] else ("AUTO" if r["auto_label"] else "NOISE"), axis=1)

    return df, sum_df

# ---------------------------
# 파이프라인: 원본 JSON → 토픽 생성
# ---------------------------
@st.cache_data(show_spinner=True)
def pipeline_from_raw(records: List[Dict[str, Any]], text_fields: List[str], id_field: Optional[str], model_name: str,
                      min_topic_size: int, umap_components: int, engine: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            continue
        flat = _flatten(rec)
        doc_id = flat.get(id_field) if id_field else None
        if doc_id is None:
            doc_id = f"doc_{i+1}"
        text = join_fields(flat, text_fields)
        rows.append({"id": doc_id, "text": text})

    df = pd.DataFrame(rows)
    df["text_clean"] = df["text"].map(clean_text)
    df = df[df["text_clean"].str.len() > 5].drop_duplicates(subset=["text_clean"]).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("유효한 텍스트가 없습니다.")

    emb = compute_embeddings(df["text_clean"].tolist(), model_name)

    if engine == "bertopic":
        if not _HAS_BERTOPIC:
            raise RuntimeError("BERTopic 미설치: pip install bertopic")
        topic_model = BERTopic(language="multilingual", n_gram_range=(1,2), min_topic_size=max(5, int(min_topic_size)), calculate_probabilities=True, verbose=False)
        topics, probs = topic_model.fit_transform(df["text_clean"].tolist(), embeddings=emb)
        df["cluster"] = topics
        info = topic_model.get_topic_info()
        summary_records = []
        for _, row in info.iterrows():
            cid = int(row["Topic"])
            if cid == -1:
                continue
            topic_terms = topic_model.get_topic(cid) or []
            top_terms = [t for t, _ in topic_terms[:10]]
            auto_label = label_from_keywords(top_terms, max_tokens=3)
            idxs = df.index[df["cluster"] == cid].tolist()
            sample_texts = [df.loc[k, "text_clean"] for k in idxs[: min(3, len(idxs))]]
            summary_records.append({"cluster": cid, "size": int(row["Count"]), "auto_label": auto_label, "keywords": top_terms, "sample_texts": sample_texts})
        cid_to_label = {r["cluster"]: r["auto_label"] for r in summary_records}
        cid_to_terms = {r["cluster"]: r["keywords"] for r in summary_records}
        df["auto_label"] = df["cluster"].map(lambda c: cid_to_label.get(c, "기타/노이즈"))
        df["top_keywords"] = df["cluster"].map(lambda c: cid_to_terms.get(c, []))
        sum_df = pd.DataFrame(summary_records).sort_values("size", ascending=False).reset_index(drop=True)
    else:
        # custom: HDBSCAN + c-TF-IDF
        X = emb
        if umap_components > 0:
            X = run_umap(emb, n_components=umap_components)
        labels = run_hdbscan(X, min_cluster_size=min_topic_size)
        df["cluster"] = labels
        # 군집 big-doc
        by_cluster: Dict[int, List[int]] = {}
        for i, c in enumerate(labels):
            if c == -1:
                continue
            by_cluster.setdefault(int(c), []).append(i)
        summary_records = []
        cid_to_terms: Dict[int, List[str]] = {}
        cid_to_label: Dict[int, str] = {}
        if by_cluster:
            big_docs = [" ".join(df.loc[idxs, "text_clean"]) for _, idxs in sorted(by_cluster.items())]
            clusters_sorted = [cid for cid, _ in sorted(by_cluster.items())]
            top_terms_list, _ = c_tf_idf_keywords(big_docs, topn=10)
            for cid, terms in zip(clusters_sorted, top_terms_list):
                auto_label = label_from_keywords(terms, max_tokens=3)
                cid_to_terms[cid] = terms
                cid_to_label[cid] = auto_label
                idxs = by_cluster[cid]
                sample_texts = [df.loc[k, "text_clean"] for k in idxs[: min(3, len(idxs))]]
                summary_records.append({"cluster": cid, "size": len(idxs), "auto_label": auto_label, "keywords": terms, "sample_texts": sample_texts})
        df["auto_label"] = df["cluster"].map(lambda c: cid_to_label.get(int(c), "기타/노이즈") if c != -1 else "기타/노이즈")
        df["top_keywords"] = df["cluster"].map(lambda c: cid_to_terms.get(int(c), []) if c != -1 else [])
        sum_df = pd.DataFrame(summary_records).sort_values("size", ascending=False).reset_index(drop=True)

    return df, sum_df


# ---------------------------
# 시각화 유틸
# ---------------------------
def plot_cluster_sizes(sum_df: pd.DataFrame, topn: int = 15):
    if sum_df.empty:
        st.info("요약 데이터가 비어 있습니다.")
        return
    top = sum_df.sort_values("size", ascending=False).head(topn)
    fig = px.bar(top, x="size", y=top.apply(lambda r: f"#{int(r['cluster'])}  {r['auto_label']}", axis=1), orientation='h', title="Cluster Sizes (Top-N)")
    st.plotly_chart(fig, use_container_width=True)


def plot_umap_scatter(docs_df: pd.DataFrame, model_name: str, max_points: int = 5000, n_components: int = 2):
    if len(docs_df) == 0:
        st.warning("문서가 없습니다.")
        return
    texts = docs_df.get("text", docs_df.get("text_clean")).astype(str).tolist()
    # 샘플링(성능 보호)
    if len(texts) > max_points:
        docs_df = docs_df.sample(max_points, random_state=42).reset_index(drop=True)
        texts = docs_df.get("text", docs_df.get("text_clean")).astype(str).tolist()
        st.caption(f"UMAP 산점도는 성능을 위해 {max_points}건 샘플로 표시합니다.")
    emb = compute_embeddings(texts, model_name)
    if not _HAS_UMAP:
        st.error("umap-learn 미설치: pip install umap-learn")
        return
    coords = run_umap(emb, n_components=n_components)
    docs_df = docs_df.copy()
    docs_df["x"] = coords[:, 0]
    docs_df["y"] = coords[:, 1]
    fig = px.scatter(
        docs_df,
        x="x", y="y",
        color=docs_df["cluster"].astype(str),
        hover_data={"id": True, "auto_label": True, "text_clean": True, "x": False, "y": False},
        title="UMAP 2D Scatter by Cluster",
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_cluster_keywords(sum_df: pd.DataFrame, cluster_id: int, topn: int = 10):
    row = sum_df[sum_df["cluster"] == cluster_id]
    if row.empty:
        st.info("해당 군집을 찾을 수 없습니다.")
        return
    kws = (row.iloc[0]["keywords"] or [])[:topn]
    if not kws:
        st.info("키워드가 없습니다.")
        return
    dfk = pd.DataFrame({"keyword": kws, "rank": list(range(len(kws), 0, -1))})
    fig = px.bar(dfk, x="rank", y="keyword", orientation='h', title=f"Top Keywords of Cluster #{cluster_id}")
    st.plotly_chart(fig, use_container_width=True)


def plot_timeline(docs_df: pd.DataFrame, time_field: str, top_clusters: List[int], freq: str = "W-MON"):
    if time_field not in docs_df.columns:
        st.info(f"문서에 '{time_field}' 필드가 없습니다.")
        return
    ts = pd.to_datetime(docs_df[time_field], errors="coerce")
    if ts.notna().sum() == 0:
        st.info("유효한 날짜가 없습니다.")
        return
    tmp = docs_df.copy()
    tmp["_week"] = ts.dt.to_period(freq).astype(str)
    tmp = tmp[tmp["cluster"].isin(top_clusters)]
    pv = tmp.pivot_table(index="_week", columns="cluster", values="id", aggfunc="count").fillna(0).sort_index()
    pv = pv.reset_index().melt(id_vars="_week", var_name="cluster", value_name="count")
    pv["cluster"] = pv["cluster"].astype(int)
    fig = px.line(pv, x="_week", y="count", color="cluster", markers=True, title="Weekly Trends by Cluster (Top-N)")
    fig.update_layout(xaxis_title="Week", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="VOC Topic Explorer", layout="wide")
st.title("📊 VOC Topic Explorer (JSON)")
st.caption("카테고리 미지정 비지도 토픽화 + 대시보드 시각화")

with st.sidebar:
    st.header("모드 선택")
    mode = st.radio("입력 모드", ["원본 JSON에서 생성", "미리 계산된 결과 시각화"], index=0)
    st.divider()
    st.header("공통 설정")
    model_name = st.text_input("Sentence-Transformer 모델", value="paraphrase-multilingual-MiniLM-L12-v2")
    topn_clusters = st.slider("Cluster Top-N (차트)", 5, 50, 15)
    topn_keywords = st.slider("Keywords Top-K", 3, 20, 10)
    st.divider()
    st.markdown("**다운로드 옵션**")
    want_docs_csv = st.checkbox("문서별 결과 CSV 버튼 표시", True)
    want_sum_csv = st.checkbox("요약 CSV 버튼 표시", True)
    
    # 하이브리드: Seed 옵션
    st.divider()
    st.header("사전 정의 카테고리(선택)")
    seed_file = st.file_uploader("Seed 파일 (CSV 또는 JSON)", type=["csv","json"], accept_multiple_files=False)
    seed_text = st.text_area("또는 JSON 배열 직접 입력", height=140, help='예: [{"category":"결제","type":"keyword","value":"이중 결제"}]')
    use_kw = st.checkbox("키워드/정규식 매칭 사용", True)
    use_ex = st.checkbox("예시문장 임베딩 매칭 사용", True)
    ex_thresh = st.slider("예시문장 임계 (cos sim)", 0.30, 0.90, 0.55, 0.01)
    exclude_seeded = st.checkbox("SEED로 할당된 문서는 클러스터링에서 제외", True)


# 상태 보관
DOCS_DF: Optional[pd.DataFrame] = None
SUM_DF: Optional[pd.DataFrame] = None

if mode == "미리 계산된 결과 시각화":
    st.subheader("미리 계산된 결과 업로드")
    c1, c2 = st.columns(2)
    with c1:
        docs_file = st.file_uploader("문서별 JSONL (voc_topics_docs.jsonl)", type=["jsonl","txt"], accept_multiple_files=False)
    with c2:
        sum_file = st.file_uploader("요약 JSON (voc_topics_summary.json)", type=["json","txt"], accept_multiple_files=False)

    if docs_file and sum_file:
        # 로드
        docs_rows = []
        for line in io.StringIO(docs_file.getvalue().decode("utf-8", errors="ignore")).read().splitlines():
            line = line.strip()
            if not line:
                continue
            docs_rows.append(json.loads(line))
        DOCS_DF = pd.DataFrame(docs_rows)
        SUM_DF = pd.DataFrame(json.loads(sum_file.getvalue().decode("utf-8", errors="ignore")))

elif mode == "원본 JSON에서 생성":
    st.subheader("원본 JSON/NDJSON 업로드 → 자동 토픽 생성")
    raw_file = st.file_uploader("JSON/NDJSON 파일", type=["json","ndjson","jsonl","txt"], accept_multiple_files=False)
    engine = st.selectbox("엔진", ["custom(HDBSCAN)", "bertopic"], index=0)
    engine_key = "bertopic" if engine == "bertopic" else "custom"
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("문서 수", len(DOCS_DF))
    with c2:
        st.metric("군집 수(노이즈 제외)", int((SUM_DF['cluster'] != -1).sum()))
    with c3:
        st.metric("최대 군집 크기", int(SUM_DF['size'].max() if not SUM_DF.empty else 0))
    with c4:
        seed_cov = float((DOCS_DF['label_source'] == 'SEED').mean()*100.0) if 'label_source' in DOCS_DF.columns else 0.0
        st.metric("SEED 적용 비율", f"{seed_cov:.1f}%")")

    text_fields: List[str] = []
    if raw_file is not None:
        try:
            records = _read_json_or_ndjson(raw_file)
            st.caption(f"레코드 수: {len(records)}")
            if len(records) == 0:
                st.stop()
            keys = discover_keys(records)
            st.markdown("**텍스트 필드 선택(여러 개 가능)**")
            text_fields = st.multiselect("점표기 키로 선택", options=keys, default=[k for k in keys if k.lower() in ("title","body","message","msg.title","msg.detail")])
            # Seed 구성
            seeds = load_seeds_from_inputs(seed_file, seed_text)
            if seeds:
                st.success(f"Seed 카테고리 {len(seeds)}개 로드 완료")
            run_btn = st.button("토픽 생성 실행", type="primary")
            if run_btn:
                with st.spinner("임베딩/클러스터링 수행 중..."):
                    DOCS_DF, SUM_DF = pipeline_from_raw_hybrid(
                        records, text_fields, id_field_in or None, model_name,
                        int(min_topic_size), int(umap_components), engine_key,
                        seeds=seeds if seeds else None,
                        use_kw_regex=use_kw,
                        use_examples=use_ex,
                        example_thresh=float(ex_thresh),
                        exclude_seeded_from_clustering=bool(exclude_seeded),
                    )
        except Exception as e:
            st.error(f"로딩/처리 중 오류: {e}")

# ---------------------------
# 시각화/탐색
# ---------------------------
if DOCS_DF is not None and SUM_DF is not None:
    st.success(f"문서 {len(DOCS_DF)}건, 군집 {SUM_DF['cluster'].nunique()}개")

    # 상단 KPI
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("문서 수", len(DOCS_DF))
    with c2:
        st.metric("군집 수(노이즈 제외)", int((SUM_DF['cluster'] != -1).sum()))
    with c3:
        st.metric("최대 군집 크기", int(SUM_DF['size'].max() if not SUM_DF.empty else 0))

    st.subheader("클러스터 크기 (Top-N)")
    plot_cluster_sizes(SUM_DF, topn=topn_clusters)

    st.subheader("UMAP 2D 산점도")
    with st.expander("보기/설정", expanded=True):
        max_points = st.slider("산점도 샘플 최대 수", 1000, 20000, 5000, 1000)
        umap_dim = st.radio("UMAP 차원", [2, 3], index=0)
        plot_btn = st.button("UMAP 재계산", type="secondary")
        if plot_btn:
            plot_umap_scatter(DOCS_DF, model_name=model_name, max_points=max_points, n_components=int(umap_dim))
        else:
            # 최초 1회 자동 표시
            plot_umap_scatter(DOCS_DF, model_name=model_name, max_points=max_points, n_components=int(umap_dim))

    st.subheader("군집 상세")
    # 군집 선택
    default_cid = int(SUM_DF.sort_values("size", ascending=False).iloc[0]["cluster"]) if not SUM_DF.empty else -1
    cid = st.number_input("군집 ID", value=default_cid, step=1)
    plot_cluster_keywords(SUM_DF, int(cid), topn=topn_keywords)

    # 샘플 문서 표시
    st.markdown("**샘플 문장 (최대 20건)**")
    samples = DOCS_DF[DOCS_DF["cluster"] == int(cid)].head(20)
    if samples.empty:
        st.info("해당 군집에 문서가 없습니다.")
    else:
        show_cols = [c for c in ["id","final_label","label_source","seed_method","seed_score","cluster","auto_label","top_keywords","text_clean"] if c in samples.columns]
        st.dataframe(samples[show_cols], use_container_width=True, hide_index=True)

    # 시계열(옵션)
    with st.expander("시계열 추이 (옵션)"):
        time_field = st.text_input("날짜/시간 칼럼명(예: created_at)")
        topk = st.slider("Top-N 군집(시계열)", 3, 20, 6)
        if time_field:
            top_clusters = SUM_DF.sort_values("size", ascending=False).head(topk)["cluster"].astype(int).tolist()
            plot_timeline(DOCS_DF, time_field, top_clusters)

    # 다운로드
    st.subheader("다운로드")
    if want_docs_csv:
        csv = DOCS_DF.to_csv(index=False).encode("utf-8")
        st.download_button("문서별 결과 CSV 다운로드", csv, file_name="voc_docs.csv", mime="text/csv")
    if want_sum_csv:
        csv2 = SUM_DF.to_csv(index=False).encode("utf-8")
        st.download_button("군집 요약 CSV 다운로드", csv2, file_name="voc_summary.csv", mime="text/csv")
else:
    st.info("좌측에서 파일을 업로드하고 파이프라인을 실행하거나, 미리 계산된 결과를 업로드해 주세요.")
