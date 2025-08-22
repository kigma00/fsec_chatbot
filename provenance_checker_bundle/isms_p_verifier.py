#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ISMS-P Feedback Verifier (CLI)
- 기준 문서(.pdf/.txt/.md/.xlsx) 인덱싱 → 피드백 JSON의 claim을 실제 근거로 검증
- 답변/피드백 내부 URL의 화이트리스트 검사
- PDF 추출: pdfminer.six → PyPDF2 → pdftotext(있으면)
- CLI 요약 출력 추가 (--print-limit)
"""

import os, re, sys, json, math, argparse, shutil, subprocess, textwrap
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from urllib.parse import urlparse

# -------- env --------
def _load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
_load_env()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
MODEL = os.getenv("MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
ENV_WHITELIST = [d.strip() for d in os.getenv("WHITELIST_DOMAINS", "").split(",") if d.strip()]

# -------- text utils --------
def normalize_text(t: str) -> str:
    t = t.replace("\u00a0", " ")
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def sent_split(text: str) -> List[str]:
    text = normalize_text(text)
    parts = re.split(r"(?<=[\.!\?]|[。！？])\s+", text)
    out = []
    for p in parts:
        p = p.strip()
        if not p: 
            continue
        if len(p) > 600:
            out.extend([s.strip() for s in re.split(r"[;·]| - ", p) if s.strip()])
        else:
            out.append(p)
    return out

def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9가-힣_\- ]+", " ", text)
    return [w for w in text.split() if w]

# -------- PDF/Text/Excel loading --------
def _pdf_with_pdfminer(path: str) -> Optional[str]:
    try:
        from pdfminer.high_level import extract_text
        return extract_text(path)
    except Exception:
        return None

def _pdf_with_pypdf2(path: str) -> Optional[str]:
    try:
        from PyPDF2 import PdfReader
        r = PdfReader(path)
        texts = []
        for pg in r.pages:
            try:
                texts.append(pg.extract_text() or "")
            except Exception:
                continue
        return "\n".join(texts).strip() or None
    except Exception:
        return None

def _pdf_with_pdftotext(path: str) -> Optional[str]:
    if not shutil.which("pdftotext"):
        return None
    try:
        out = subprocess.run(
            ["pdftotext", "-layout", "-q", path, "-"],
            check=False, capture_output=True, text=True
        )
        txt = out.stdout.strip()
        return txt or None
    except Exception:
        return None

def _xlsx_to_text(path: Path, debug: bool=False) -> str:
    try:
        import pandas as pd
    except Exception:
        if debug:
            print(f"[warn] pandas 미설치로 엑셀 스킵: {path.name}")
        return ""
    try:
        xls = pd.ExcelFile(str(path))
        parts = []
        for sheet in xls.sheet_names:
            df = xls.parse(sheet).fillna("")
            # 열 이름 + 값들을 텍스트화
            rows = [", ".join(map(str, df.columns))]
            for _, row in df.iterrows():
                vals = [str(v) for v in row.tolist()]
                rows.append(", ".join(vals))
            block = f"[sheet:{sheet}]\n" + "\n".join(rows)
            parts.append(block)
        txt = "\n\n".join(parts)
        if debug:
            print(f"[load] {path.name}: {len(txt)} chars (excel)")
        return txt
    except Exception as e:
        if debug:
            print(f"[warn] excel 파싱 실패: {path.name} ({e})")
        return ""

def read_text_from_path(p: Path, debug: bool=False) -> str:
    suf = p.suffix.lower()
    if suf in {".txt", ".md", ".markdown"}:
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            if debug: print(f"[load] {p.name}: {len(txt)} chars (text)")
            return txt
        except Exception:
            if debug: print(f"[warn] failed to read text: {p.name}")
            return ""
    if suf == ".pdf":
        for fn, func in [("pdfminer", _pdf_with_pdfminer),
                         ("PyPDF2", _pdf_with_pypdf2),
                         ("pdftotext", _pdf_with_pdftotext)]:
            txt = func(str(p))
            if txt:
                if debug: print(f"[load] {p.name}: {len(txt)} chars (pdf via {fn})")
                return txt
        if debug: print(f"[warn] failed to extract pdf: {p.name}")
        return ""
    if suf in {".xlsx", ".xls"}:
        return _xlsx_to_text(p, debug=debug)
    # fallback
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore")
        if debug: print(f"[load] {p.name}: {len(txt)} chars (fallback)")
        return txt
    except Exception:
        if debug: print(f"[warn] failed to read: {p.name}")
        return ""

def load_docs_from_dir(docs_dir: str, debug: bool=False) -> Dict[str, str]:
    docs: Dict[str, str] = {}
    p = Path(docs_dir)
    for fp in p.rglob("*"):
        if fp.is_file() and fp.suffix.lower() in {".pdf", ".txt", ".md", ".markdown", ".xlsx", ".xls"}:
            txt = read_text_from_path(fp, debug=debug)
            if txt:
                docs[fp.name] = txt
    if debug:
        print(f"[info] loaded {len(docs)} docs from {docs_dir}")
    return docs

# -------- chunking / tf-idf --------
def chunk_text(text: str, max_chars: int = 800, overlap: int = 120) -> List[str]:
    sents = sent_split(text)
    chunks: List[str] = []
    cur = ""
    for s in sents:
        if len(cur) + len(s) + 1 <= max_chars:
            cur = (cur + " " + s).strip()
        else:
            if cur: chunks.append(cur)
            tail = cur[-overlap:] if overlap and cur else ""
            cur = (tail + " " + s).strip()
            if len(cur) > max_chars:
                chunks.append(cur[:max_chars])
                cur = cur[max_chars - overlap:]
    if cur: chunks.append(cur)
    return [c for c in chunks if c.strip()]

def build_index(docs: Dict[str, str], max_chars=800, overlap=120):
    chunks: List[Tuple[str, str]] = []
    for doc_id, text in docs.items():
        for ch in chunk_text(text, max_chars, overlap):
            chunks.append((doc_id, ch))
    vocab: Dict[str, int] = {}
    chunk_tokens: List[List[str]] = []
    for _, ch in chunks:
        toks = tokenize(ch)
        chunk_tokens.append(toks)
        for t in set(toks):
            if t not in vocab:
                vocab[t] = len(vocab)
    df = [0] * len(vocab)
    for toks in chunk_tokens:
        seen = set(toks)
        for t in seen: df[vocab[t]] += 1
    N = max(1, len(chunk_tokens))
    idf = [math.log((N + 1) / (df_i + 1)) + 1.0 for df_i in df]
    tfidf: List[Dict[int, float]] = []
    for toks in chunk_tokens:
        tf: Dict[int, int] = {}
        for t in toks: tf[vocab[t]] = tf.get(vocab[t], 0) + 1
        vec: Dict[int, float] = {j: (cnt / len(toks)) * idf[j] for j, cnt in tf.items()}
        norm = math.sqrt(sum(v*v for v in vec.values())) or 1.0
        tfidf.append({j: v / norm for j, v in vec.items()})
    return chunks, vocab, tfidf, idf

def vectorize(text: str, vocab: Dict[str, int], idf: List[float]) -> Dict[int, float]:
    toks = tokenize(text)
    if not toks: return {}
    tf: Dict[int, int] = {}
    for t in toks:
        if t in vocab:
            j = vocab[t]
            tf[j] = tf.get(j, 0) + 1
    if not tf: return {}
    vec = {j: (cnt / len(toks)) * idf[j] for j, cnt in tf.items()}
    norm = math.sqrt(sum(v*v for v in vec.values())) or 1.0
    return {j: v / norm for j, v in vec.items()}

def cosine_sparse(a: Dict[int, float], b: Dict[int, float]) -> float:
    if not a or not b: return 0.0
    if len(a) > len(b): a, b = b, a
    s = 0.0
    for j, v in a.items():
        if j in b: s += v * b[j]
    return float(s)

def top_k_chunks(query: str, chunks, vocab, tfidf, idf, k=5) -> List[Tuple[int, float]]:
    qv = vectorize(query, vocab, idf)
    sims = [(i, cosine_sparse(qv, tfidf[i])) for i in range(len(tfidf))]
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:k]

# -------- whitelist --------
_URL_RE = re.compile(r"(https?://[^\s)>\]\"'}]+|www\.[^\s)>\]\"'}]+)", re.IGNORECASE)
def _clean_url(u: str) -> str: return u.rstrip(").,!?;:\"'“”‘’]}>")
def extract_urls(text: str) -> List[str]:
    urls = []
    for m in _URL_RE.finditer(text or ""):
        u = _clean_url(m.group(0))
        if u.lower().startswith("www."): u = "http://" + u
        urls.append(u)
    for m in re.finditer(r"\]\((https?://[^)]+)\)", text or "", re.IGNORECASE):
        urls.append(_clean_url(m.group(1)))
    seen, out = set(), []
    for u in urls:
        if u not in seen: out.append(u); seen.add(u)
    return out

def is_allowed_host(host: str, whitelist: List[str]) -> Tuple[bool, Optional[str]]:
    h = (host or "").lower()
    if not h: return False, None
    for w in whitelist:
        w = w.lower()
        if w.startswith("*."):
            if h.endswith(w[1:]): return True, w
        elif h == w or h.endswith("." + w): return True, w
    return False, None

def domain_gate_urls(texts: List[str], whitelist: List[str]) -> Dict:
    urls = []
    for t in texts: urls.extend(extract_urls(t))
    allowed, blocked = [], []
    for u in urls:
        try: host = (urlparse(u).hostname or "").lower()
        except Exception: host = ""
        ok, rule = is_allowed_host(host, whitelist)
        if ok: allowed.append({"url": u, "host": host, "rule": rule})
        else:  blocked.append({"url": u, "host": host, "reason": "not_in_whitelist" if host else "invalid_url"})
    return {"whitelist": whitelist, "urls_found": urls,
            "allowed": allowed, "blocked": blocked,
            "allowed_count": len(allowed), "blocked_count": len(blocked)}

# -------- feedback parsing --------
def _iter_strings(obj: Any):
    if isinstance(obj, str): yield obj
    elif isinstance(obj, list):
        for x in obj: yield from _iter_strings(x)
    elif isinstance(obj, dict):
        for v in obj.values(): yield from _iter_strings(v)

def pick_claims_from_feedback(data: Dict) -> List[Dict]:
    claims = []
    if isinstance(data.get("feedback"), list):
        for it in data["feedback"]:
            claim = {
                "control_id": it.get("control_id"),
                "claim_text": it.get("claim_text") or "",
                "references": it.get("references") or []
            }
            if claim["claim_text"]: claims.append(claim)
    if not claims:
        texts = [t for t in _iter_strings(data) if isinstance(t, str)]
        for t in texts:
            for sent in sent_split(t):
                if len(sent) >= 30:
                    claims.append({"control_id": None, "claim_text": sent, "references": []})
                    if len(claims) >= 20: break
            if len(claims) >= 20: break
    return claims

# -------- support analysis --------
def support_analysis(answer: str, chunks, vocab, tfidf, idf, k=3, thresh=0.22):
    sents = [s for s in sent_split(answer) if len(s) >= 15]
    per, supported = [], 0
    for s in sents:
        top = top_k_chunks(s, chunks, vocab, tfidf, idf, k=k)
        best_i, best_sim = (top[0] if top else (-1, 0.0))
        hit = best_sim >= thresh
        if hit: supported += 1
        per.append({
            "claim": s,
            "topk": [{"doc_id": chunks[i][0], "chunk": chunks[i][1][:400], "similarity": round(sim,4)} for i, sim in top],
            "best_similarity": round(best_sim,4),
            "supported": bool(hit)
        })
    ratio = supported / max(1, len(sents))
    verdict = "supported" if ratio >= 0.85 else ("partial" if ratio >= 0.4 else "unsupported")
    return {"num_claims": len(sents), "supported": supported, "coverage": round(ratio,3),
            "verdict": verdict, "details": per}

def aggregate_sources(details, min_sim=0.25):
    sources = {}
    for d in details:
        if not d.get("topk"): continue
        doc_id = d["topk"][0]["doc_id"]; sim = d["topk"][0]["similarity"]
        if sim < min_sim: continue
        snippet = d["topk"][0]["chunk"]
        sources.setdefault(doc_id, []).append({"similarity": sim, "snippet": snippet})
    return [{"doc_id": k, "matches": sorted(v, key=lambda x: -x["similarity"])[:5]} for k, v in sources.items()]

# -------- LLM summary (optional) --------
def llm_summary(result: Dict) -> str:
    if not OPENAI_API_KEY: return "(요약 생략: OPENAI_API_KEY 미설정)"
    try:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            prompt = ("ISMS-P 피드백 검증 결과를 3줄로 요약하세요. "
                      "overall_verdict/coverage와 차단된 URL 개수를 포함하세요.\n\n"
                      + json.dumps(result, ensure_ascii=False)[:6000])
            resp = client.chat.completions.create(model=MODEL, messages=[{"role":"user","content":prompt}], temperature=0)
            return resp.choices[0].message.content.strip()
        except Exception:
            import openai as openai_legacy
            openai_legacy.api_key = OPENAI_API_KEY
            prompt = ("ISMS-P 피드백 검증 결과를 3줄로 요약하세요. "
                      "overall_verdict/coverage와 차단된 URL 개수를 포함하세요.\n\n"
                      + json.dumps(result, ensure_ascii=False)[:6000])
            resp = openai_legacy.ChatCompletion.create(model=MODEL, messages=[{"role":"user","content":prompt}], temperature=0)
            return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"(요약 실패: {e})"

# -------- verify main --------
def verify_feedback_json(docs_dir: str, feedback_json_path: str,
                         k=3, thresh=0.22, whitelist: Optional[List[str]] = None,
                         use_summary: bool = False, debug: bool=False) -> Dict:
    docs = load_docs_from_dir(docs_dir, debug=debug)
    if not docs:
        raise RuntimeError(f"No documents loaded from: {docs_dir}")
    chunks, vocab, tfidf, idf = build_index(docs)

    data = json.loads(Path(feedback_json_path).read_text(encoding="utf-8"))
    claims = pick_claims_from_feedback(data)
    all_texts = [c["claim_text"] for c in claims]
    dg = domain_gate_urls(all_texts, whitelist or [])

    results_by_item = []
    sup, total = 0, 0
    for c in claims:
        analysis = support_analysis(c["claim_text"], chunks, vocab, tfidf, idf, k=k, thresh=thresh)
        total += analysis["num_claims"]; sup += analysis["supported"]
        top_source = analysis["details"][0]["topk"][0] if (analysis["details"] and analysis["details"][0]["topk"]) else None
        results_by_item.append({
            "control_id": c.get("control_id"),
            "claim_text": c["claim_text"],
            "verdict": analysis["verdict"],
            "coverage": analysis["coverage"],
            "top_source": top_source,
            "analysis": analysis["details"]
        })

    ratio = sup / max(1, total)
    merged_sources = aggregate_sources(
        [{"topk": r["analysis"][0]["topk"]} for r in results_by_item if r.get("analysis")],
        min_sim=max(0.25, thresh)
    )
    result = {
        "verifier": "isms-p-feedback-verifier",
        "docs_dir": str(Path(docs_dir).resolve()),
        "feedback_file": str(Path(feedback_json_path).name),
        "overall_verdict": "supported" if ratio >= 0.85 else ("partial" if ratio >= 0.4 else "unsupported"),
        "overall_coverage": round(ratio, 3),
        "tot_claims": total,
        "supported_claims": sup,
        "sources": merged_sources,
        "by_item": results_by_item,
        "domain_gate": dg,
        "env": {"model": MODEL, "whitelist_domains": whitelist or []}
    }
    if use_summary:
        result["summary"] = llm_summary(result)
    return result

# -------- CLI summary print --------
def print_cli_summary(result: Dict, limit: int = 8):
    wrap = lambda s, w=80: textwrap.shorten(s.replace("\n"," "), width=w, placeholder="…")
    print("\n=== ISMS-P Feedback Verification ===")
    print(f"Docs dir     : {result.get('docs_dir')}")
    print(f"Feedback file: {result.get('feedback_file')}")
    print(f"Verdict      : {result.get('overall_verdict')}  | Coverage: {result.get('overall_coverage')*100:.1f}%")
    print(f"Claims       : {result.get('tot_claims')}  | Supported: {result.get('supported_claims')}")
    dg = result.get("domain_gate", {})
    print(f"URLs         : {dg.get('allowed_count',0)} allowed / {dg.get('blocked_count',0)} blocked")
    if dg.get("blocked"):
        print("  Blocked URLs:")
        for b in dg["blocked"][:limit]:
            print(f"    - {b.get('url')} ({b.get('reason')})")
    if result.get("sources"):
        print("\nTop Sources:")
        for s in result["sources"][:limit]:
            best = s["matches"][0]
            print(f"  - {s['doc_id']}  (sim {best['similarity']:.3f})  :: {wrap(best['snippet'])}")
    print("\nPer-Item:")
    for i, it in enumerate(result.get("by_item", [])[:limit], 1):
        cs = it.get("top_source") or {}
        print(f"  {i:02d}. [{it.get('verdict'):<10}] cov={it.get('coverage'):.2f}  id={it.get('control_id') or '-'}")
        print(f"      claim: {wrap(it.get('claim_text',''), 100)}")
        if cs:
            print(f"      src  : {cs.get('doc_id')} (sim {cs.get('similarity',0):.3f}) :: {wrap(cs.get('chunk',''), 100)}")
    if "summary" in result:
        print("\nSummary(LLM):", result["summary"])
    print("====================================\n")

# -------- CLI --------
def main():
    ap = argparse.ArgumentParser(description="ISMS-P Feedback Verifier (CLI)")
    ap.add_argument("--docs", required=True, help="기준 문서 폴더(.pdf/.txt/.md/.xlsx)")
    ap.add_argument("--feedback", required=True, help="피드백 JSON 경로")
    ap.add_argument("--out", default="verify_result.json", help="출력 JSON 경로")
    ap.add_argument("--thresh", type=float, default=0.22, help="근거 임계치")
    ap.add_argument("--k", type=int, default=3, help="문장당 Top-K 청크 수")
    ap.add_argument("--whitelist", default="", help="CSV: intra.bank,regulator.go.kr,*.gov.kr (없으면 .env)")
    ap.add_argument("--summary", action="store_true", help="3줄 요약 생성(.env 키 필요)")
    ap.add_argument("--debug", action="store_true", help="로딩/인덱싱 로그")
    ap.add_argument("--print-limit", type=int, default=8, help="CLI 요약에 표시할 항목 수")
    args = ap.parse_args()

    whitelist = [d.strip() for d in (args.whitelist or "").split(",") if d.strip()] or ENV_WHITELIST

    try:
        result = verify_feedback_json(
            docs_dir=args.docs,
            feedback_json_path=args.feedback,
            k=args.k,
            thresh=args.thresh,
            whitelist=whitelist,
            use_summary=args.summary,
            debug=args.debug
        )
    except Exception as e:
        print(f"[!] Error: {e}", file=sys.stderr); sys.exit(2)

    # CLI 요약 출력
    print_cli_summary(result, limit=args.print_limit)

    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote JSON to {args.out}")

if __name__ == "__main__":
    main()