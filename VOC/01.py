#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit VOC Topic Explorer (JSON input) — Hierarchical Topics + Spike Alerts
-----------------------------------------------------------------------------

기능 요약
- JSON/NDJSON VOC 업로드 → 임베딩 → (선택) UMAP → HDBSCAN(또는 BERTopic) → c-TF-IDF 키워드 → 자동 라벨 → 대시보드
- **계층형 토픽(상위 슈퍼클러스터 자동 생성)**: 군집 센트로이드 → KMeans로 상위 토픽 묶기 → Sunburst 시각화/필터링
- **신규/급증 토픽 알림**: 날짜 필드 기준 주간 집계 → 기준선 대비 급증(z-score)·완전 신규 토픽 탐지 → 알림 테이블/다운로드
- 기존 산출물(docs.jsonl + summary.json) 업로드 즉시 시각화 지원
- 결과(문서별/요약/계층/알림) CSV 다운로드

실행
1) 의존성:
   pip install -U streamlit pandas numpy scikit-learn sentence-transformers hdbscan umap-learn plotly
   # (옵션) BERTopic 사용 시
   pip install -U bertopic
2) 실행: streamlit run app.py

입력
- 원본 JSON: 배열 JSON([{},...]) 또는 NDJSON(줄당 JSON)
- 텍스트 필드 다중 지정(점표기) 가능: 예) title, body, user.comment
- 사전 계산 결과: voc_topics_docs.jsonl + voc_topics_summary.json 업로드
"""

import io
import json
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
from sklearn.cluster import KMeans

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
# 파이프라인: 원본 JSON → 토픽 생성
# ---------------------------
@st.cache_data(show_spinner=True)
def pipeline_from_raw(records: List[Dict[str, Any]], text_fields: List[str], id_field: Optional[str], model_name: str,
                      min_topic_size: int, umap_components: int, engine: str) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
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

    return df, sum_df, emb


# ---------------------------
# 계층형 토픽(슈퍼클러스터)
# ---------------------------
@st.cache_data(show_spinner=True)
def build_hierarchy(docs_df: pd.DataFrame, sum_df: pd.DataFrame, emb: np.ndarray, super_k: int, topn_kw: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # 유효 군집만
    valid = docs_df[docs_df["cluster"] != -1].copy()
    if valid.empty:
        return pd.DataFrame(), pd.DataFrame()

    # 군집별 센트로이드
    clusters = sorted(valid["cluster"].unique().tolist())
    centroids = []
    for c in clusters:
        idxs = valid.index[valid["cluster"] == c].tolist()
        vec = emb[idxs].mean(axis=0)
        centroids.append(vec)
    centroids = np.vstack(centroids)

    # KMeans로 슈퍼클러스터
    super_k = max(2, int(min(super_k, len(clusters))))
    km = KMeans(n_clusters=super_k, n_init=10, random_state=42)
    super_labels = km.fit_predict(centroids)

    # 매핑: cluster -> super_id
    c2s = {c: int(sid) for c, sid in zip(clusters, super_labels)}
    docs_df = docs_df.copy()
    docs_df["super_id"] = docs_df["cluster"].map(lambda c: c2s.get(int(c), -1) if c != -1 else -1)

    # 슈퍼 별 키워드/라벨: 각 슈퍼에 속한 문서 텍스트 합치기 → c-TF-IDF
    super_big_docs = []
    super_ids_sorted = sorted(set([sid for sid in super_labels.tolist()]))
    for sid in super_ids_sorted:
        idxs = docs_df.index[docs_df["super_id"] == sid].tolist()
        super_big_docs.append(" ".join(docs_df.loc[idxs, "text_clean"]))
    super_terms, _ = c_tf_idf_keywords(super_big_docs, topn=topn_kw)
    super_labels_txt = [label_from_keywords(kws, max_tokens=3) for kws in super_terms]

    # 슈퍼 요약 DF
    super_records = []
    for sid, lbl, kws in zip(super_ids_sorted, super_labels_txt, super_terms):
        size = int((docs_df["super_id"] == sid).sum())
        super_records.append({"super_id": int(sid), "size": size, "super_label": lbl, "super_keywords": kws})
    super_df = pd.DataFrame(super_records).sort_values("size", ascending=False).reset_index(drop=True)

    # 클러스터-슈퍼 매핑 DF
    map_rows = []
    idx_sum = sum_df.set_index("cluster") if not sum_df.empty else None
    for c in clusters:
        sid = c2s[c]
        clabel = idx_sum.loc[c, "auto_label"] if idx_sum is not None and c in idx_sum.index else str(c)
        map_rows.append({"cluster": int(c), "auto_label": clabel, "super_id": int(sid),
                         "super_label": super_df.set_index("super_id").loc[sid, "super_label"]})
    hier_map_df = pd.DataFrame(map_rows).sort_values(["super_id","cluster"]).reset_index(drop=True)

    return super_df, hier_map_df


# ---------------------------
# 스파이크/신규 토픽 알림
# ---------------------------
@st.cache_data(show_spinner=True)
def detect_spikes(docs_df: pd.DataFrame, time_field: str, target: str, window_weeks: int = 4,
                  z_thresh: float = 2.0, min_count: int = 10) -> pd.DataFrame:
    """
    target: 'cluster' 또는 'super_id'
    기준: 마지막 주의 카운트가 이전 window_weeks 평균 대비 z-score >= z_thresh 이면 급증
         이전 window_weeks 동안 0이고 마지막 주 >= min_count 이면 신규
    반환: [id, label, week, count, baseline_mean, z, type]
    """
    if time_field not in docs_df.columns:
        return pd.DataFrame()
    s = pd.to_datetime(docs_df[time_field], errors="coerce")
    df = docs_df.copy()
    df = df[s.notna()].copy()
    if df.empty:
        return pd.DataFrame()

    df["week"] = s.dt.to_period("W-MON").astype(str)
    key = target
    # 라벨 매핑
    if target == "cluster":
        label_map = df.groupby("cluster")["auto_label"].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else str(x.iloc[0])).to_dict()
    else:
        # super 라벨은 계산 시 외부에서 제공될 수 있으나, 일단 super_id 문자열로 표기
        label_map = df.groupby("super_id")["super_id"].first().apply(lambda x: f"super_{int(x)}").to_dict()

    # 주간 피벗
    pv = df.pivot_table(index="week", columns=key, values="id", aggfunc="count").fillna(0).sort_index()
    if len(pv.index) < window_weeks + 1:
        return pd.DataFrame()

    last_week = pv.index[-1]
    base = pv.iloc[-(window_weeks+1):-1]
    cur = pv.iloc[[-1]]
    base_mean = base.mean(axis=0)
    base_std = base.std(axis=0).replace(0, 1e-6)
    cur_cnt = cur.iloc[0]

    z = (cur_cnt - base_mean) / base_std

    alerts = []
    for col in pv.columns:
        count_now = int(cur_cnt[col])
        mean_val = float(base_mean[col])
        z_val = float(z[col])
        if count_now >= min_count and z_val >= z_thresh:
            alerts.append({"id": int(col), "label": label_map.get(col, str(col)), "week": last_week,
                           "count": count_now, "baseline_mean": round(mean_val,2), "z": round(z_val,2), "type": "spike"})
        # 신규: 직전 window 모두 0이고 이번 주 최소 건수 이상
        if (base[col] == 0).all() and count_now >= min_count:
            alerts.append({"id": int(col), "label": label_map.get(col, str(col)), "week": last_week,
                           "count": count_now, "baseline_mean": 0.0, "z": None, "type": "new"})

    return pd.DataFrame(alerts).sort_values(["type","count"], ascending=[True, False]).reset_index(drop=True)


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


def plot_umap_scatter(docs_df: pd.DataFrame, model_name: str, max_points: int = 5000, n_components: int = 2, color_by: str = "cluster"):
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
    docs_df["x"], docs_df["y"] = coords[:, 0], coords[:, 1]
    color_series = docs_df[color_by].astype(str) if color_by in docs_df.columns else docs_df["cluster"].astype(str)
    fig = px.scatter(
        docs_df,
        x="x", y="y",
        color=color_series,
        hover_data={"id": True, "auto_label": True, "text_clean": True, "x": False, "y": False},
        title=f"UMAP 2D Scatter by {color_by}",
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


def plot_timeline(docs_df: pd.DataFrame, time_field: str, top_ids: List[int], by: str = "cluster"):
    if time_field not in docs_df.columns:
        st.info(f"문서에 '{time_field}' 필드가 없습니다.")
        return
    ts = pd.to_datetime(docs_df[time_field], errors="coerce")
    if ts.notna().sum() == 0:
        st.info("유효한 날짜가 없습니다.")
        return
    tmp = docs_df.copy()
    tmp["_week"] = ts.dt.to_period("W-MON").astype(str)
    tmp = tmp[tmp[by].isin(top_ids)]
    pv = tmp.pivot_table(index="_week", columns=by, values="id", aggfunc="count").fillna(0).sort_index()
    pv = pv.reset_index().melt(id_vars="_week", var_name=by, value_name="count")
    pv[by] = pv[by].astype(int)
    fig = px.line(pv, x="_week", y="count", color=by, markers=True, title=f"Weekly Trends by {by} (Top-N)")
    fig.update_layout(xaxis_title="Week", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)


def plot_hierarchy_sunburst(docs_df: pd.DataFrame, super_df: pd.DataFrame, hier_map_df: pd.DataFrame):
    if super_df.empty or hier_map_df.empty:
        st.info("계층 데이터가 없습니다.")
        return
    # path: super -> cluster
    merged = hier_map_df.merge(docs_df.groupby(["cluster"]).size().reset_index(name="size"), on="cluster", how="left")
    merged["size"] = merged["size"].fillna(0).astype(int)
    path_df = merged[["super_label","auto_label","size"]].rename(columns={"super_label":"super","auto_label":"cluster_label"})
    fig = px.sunburst(path_df, path=["super","cluster_label"], values="size", title="Hierarchical Topics (Super → Cluster)")
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="VOC Topic Explorer", layout="wide")
st.title("📊 VOC Topic Explorer (JSON) — 계층형 + 알림")
st.caption("카테고리 미지정 비지도 토픽화 + 계층형 토픽 + 신규/급증 알림")

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
    want_docs_csv = st.checkbox("문서별 결과 CSV 버튼", True)
    want_sum_csv = st.checkbox("요약 CSV 버튼", True)
    want_hier_csv = st.checkbox("계층 매핑 CSV 버튼", True)
    want_alerts_csv = st.checkbox("알림 CSV 버튼", True)

# 상태 보관
DOCS_DF: Optional[pd.DataFrame] = None
SUM_DF: Optional[pd.DataFrame] = None
EMB_MAT: Optional[np.ndarray] = None
SUPER_DF: Optional[pd.DataFrame] = None
HIER_MAP_DF: Optional[pd.DataFrame] = None

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
        # 임베딩 재계산(계층/UMAP용)
        with st.spinner("임베딩 재계산 중..."):
            EMB_MAT = compute_embeddings(DOCS_DF.get("text_clean", DOCS_DF.get("text")).astype(str).tolist(), model_name)

elif mode == "원본 JSON에서 생성":
    st.subheader("원본 JSON/NDJSON 업로드 → 자동 토픽 생성")
    raw_file = st.file_uploader("JSON/NDJSON 파일", type=["json","ndjson","jsonl","txt"], accept_multiple_files=False)
    engine = st.selectbox("엔진", ["custom(HDBSCAN)", "bertopic"], index=0)
    engine_key = "bertopic" if engine == "bertopic" else "custom"
    c1, c2, c3 = st.columns(3)
    with c1:
        min_topic_size = st.number_input("min_topic_size", min_value=5, max_value=1000, value=20, step=1)
    with c2:
        umap_components = st.number_input("UMAP 차원 (custom)", min_value=0, max_value=10, value=5, step=1,
                                         help="0이면 UMAP 생략. bertopic 모드에서는 무시")
    with c3:
        id_field_in = st.text_input("ID 필드(선택)")

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
            run_btn = st.button("토픽 생성 실행", type="primary")
            if run_btn:
                with st.spinner("임베딩/클러스터링 수행 중..."):
                    DOCS_DF, SUM_DF, EMB_MAT = pipeline_from_raw(records, text_fields, id_field_in or None, model_name,
                                                                 int(min_topic_size), int(umap_components), engine_key)
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

    # ---- 계층형 토픽 ----
    st.subheader("계층형 토픽 (슈퍼클러스터)")
    with st.expander("설정 및 생성", expanded=False):
        super_k = st.slider("상위 슈퍼클러스터 수(KMeans K)", 2, max(3, min(60, int(max(2, SUM_DF['cluster'].nunique()//2)))), 10)
        build_btn = st.button("계층 생성/업데이트")
        if build_btn:
            if EMB_MAT is None:
                with st.spinner("임베딩 생성 중..."):
                    EMB_MAT = compute_embeddings(DOCS_DF.get("text_clean", DOCS_DF.get("text")).astype(str).tolist(), model_name)
            with st.spinner("계층 구축 중..."):
                SUPER_DF, HIER_MAP_DF = build_hierarchy(DOCS_DF, SUM_DF, EMB_MAT, super_k=super_k, topn_kw=topn_keywords)
                if SUPER_DF is None or SUPER_DF.empty:
                    st.info("생성된 계층이 없습니다.")
    if 'SUPER_DF' in globals() or SUPER_DF is not None:
        if SUPER_DF is not None and not SUPER_DF.empty and HIER_MAP_DF is not None and not HIER_MAP_DF.empty:
            st.markdown("**Sunburst (Super → Cluster)**")
            plot_hierarchy_sunburst(DOCS_DF, SUPER_DF, HIER_MAP_DF)

            st.markdown("**슈퍼/클러스터 매핑 미리보기**")
            st.dataframe(HIER_MAP_DF.head(50), use_container_width=True, hide_index=True)

    # ---- UMAP 산점도 ----
    st.subheader("UMAP 2D 산점도")
    with st.expander("보기/설정", expanded=True):
        max_points = st.slider("산점도 샘플 최대 수", 1000, 20000, 5000, 1000)
        umap_dim = st.radio("UMAP 차원", [2, 3], index=0)
        color_by = st.selectbox("색상 기준", ["cluster", "super_id" if (HIER_MAP_DF is not None and not HIER_MAP_DF.empty) else "cluster"], index=0)
        plot_btn = st.button("UMAP 재계산", type="secondary")
        if plot_btn:
            plot_umap_scatter(DOCS_DF, model_name=model_name, max_points=max_points, n_components=int(umap_dim), color_by=color_by)
        else:
            plot_umap_scatter(DOCS_DF, model_name=model_name, max_points=max_points, n_components=int(umap_dim), color_by=color_by)

    # ---- 군집 상세 ----
    st.subheader("군집 상세")
    default_cid = int(SUM_DF.sort_values("size", ascending=False).iloc[0]["cluster"]) if not SUM_DF.empty else -1
    cid = st.number_input("군집 ID", value=default_cid, step=1)
    plot_cluster_keywords(SUM_DF, int(cid), topn=topn_keywords)

    st.markdown("**샘플 문장 (최대 20건)**")
    samples = DOCS_DF[DOCS_DF["cluster"] == int(cid)].head(20)
    if samples.empty:
        st.info("해당 군집에 문서가 없습니다.")
    else:
        show_cols = [c for c in ["id","auto_label","top_keywords","text_clean"] if c in samples.columns]
        st.dataframe(samples[show_cols], use_container_width=True, hide_index=True)

    # ---- 시계열 & 알림 ----
    st.subheader("신규/급증 토픽 알림")
    with st.expander("설정", expanded=False):
        time_field = st.text_input("날짜/시간 칼럼명(예: created_at, date)")
        target = st.selectbox("분석 단위", ["cluster", "super_id" if (HIER_MAP_DF is not None and not HIER_MAP_DF.empty) else "cluster"], index=0)
        window_weeks = st.slider("기준선 윈도우(주)", 2, 12, 4)
        z_thresh = st.slider("급증 임계(z-score)", 1.0, 5.0, 2.0, 0.1)
        min_count = st.slider("최소 발생 건수", 1, 100, 10)
        detect_btn = st.button("알림 계산")
        if detect_btn and time_field:
            alerts = detect_spikes(DOCS_DF, time_field, target=target, window_weeks=window_weeks, z_thresh=z_thresh, min_count=min_count)
            if alerts is None or alerts.empty:
                st.info("알림 없음 (데이터 기간/임계 조정 필요)")
            else:
                st.success(f"알림 {len(alerts)}건 감지")
                st.dataframe(alerts, use_container_width=True, hide_index=True)
                # 상위 몇 개만 시계열 표시
                top_ids = alerts.sort_values("count", ascending=False).head(6)["id"].astype(int).tolist()
                plot_timeline(DOCS_DF, time_field, top_ids, by=target)
                if want_alerts_csv:
                    st.download_button("알림 CSV 다운로드", alerts.to_csv(index=False).encode("utf-8"), file_name="voc_alerts.csv", mime="text/csv")
        elif detect_btn and not time_field:
            st.warning("날짜/시간 칼럼명을 입력해 주세요.")

    # ---- 다운로드 ----
    st.subheader("다운로드")
    if want_docs_csv:
        csv = DOCS_DF.to_csv(index=False).encode("utf-8")
        st.download_button("문서별 결과 CSV", csv, file_name="voc_docs.csv", mime="text/csv")
    if want_sum_csv:
        csv2 = SUM_DF.to_csv(index=False).encode("utf-8")
        st.download_button("군집 요약 CSV", csv2, file_name="voc_summary.csv", mime="text/csv")
    if want_hier_csv and (HIER_MAP_DF is not None) and (not HIER_MAP_DF.empty):
        csv3 = HIER_MAP_DF.to_csv(index=False).encode("utf-8")
        st.download_button("계층 매핑 CSV", csv3, file_name="voc_hierarchy_map.csv", mime="text/csv")
else:
    st.info("좌측에서 파일을 업로드하고 파이프라인을 실행하거나, 미리 계산된 결과를 업로드해 주세요.")
