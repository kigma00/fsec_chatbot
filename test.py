# NetSketch — FortiGate 로그로 네트워크 토폴로지 자동 스케치 (Streamlit)
# -----------------------------------------------------------------------------
# 실행 전 설치(가상환경 권장):
#   python -m pip install --upgrade pip
#   pip install streamlit pandas networkx pyvis jinja2 "openai>=1.20.0,<2" "httpx<0.28"
# 실행:
#   streamlit run app.py
# -----------------------------------------------------------------------------
# 메모:
# - OpenAI API 키는 아래 OPENAI_API_KEY 변수에 하드코딩하거나(비권장), 사이드바에 입력하세요.
# - CSV/TSV 자동 인식, 대용량은 '최대 처리 행 수'로 제한하세요.
# -----------------------------------------------------------------------------

from __future__ import annotations
import io
import json
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
import ipaddress

# ------------------------ 의존성 점검: networkx/pyvis/jinja2 ------------------------
try:
    import networkx as nx  # type: ignore
except Exception as e:
    st.set_page_config(page_title="NetSketch — FortiGate Topology", page_icon="🕸️", layout="wide")
    st.error("`networkx`가 설치되어 있지 않습니다. 아래 명령으로 설치 후 다시 실행하세요.")
    st.code("pip install networkx", language="bash")
    st.caption(f"ImportError: {type(e).__name__}: {e}")
    st.stop()

try:
    from pyvis.network import Network  # type: ignore
except Exception as e:
    st.set_page_config(page_title="NetSketch — FortiGate Topology", page_icon="🕸️", layout="wide")
    st.error("`pyvis`가 설치되어 있지 않습니다. 아래 명령으로 설치 후 다시 실행하세요.")
    st.code("pip install pyvis", language="bash")
    st.caption(f"ImportError: {type(e).__name__}: {e}")
    st.stop()

try:
    import jinja2  # pyvis HTML 렌더에 필요
except Exception as e:
    st.set_page_config(page_title="NetSketch — FortiGate Topology", page_icon="🕸️", layout="wide")
    st.error("`jinja2`가 설치되어 있지 않아 그래프 HTML 렌더링이 실패합니다. 설치 후 다시 실행하세요.")
    st.code("pip install jinja2", language="bash")
    st.caption(f"ImportError: {type(e).__name__}: {e}")
    st.stop()

# (선택) OpenAI 요약 사용 — 키를 하드코딩(데모용)
OPENAI_API_KEY = "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"  # 값이 있으면 LLM 요약 활성화, 비우면 비활성화
try:
    from openai import OpenAI  # OpenAI Python SDK v1.x
    OA_OK = True
except Exception:
    OpenAI = None
    OA_OK = False  # 설치 안 되어도 앱 핵심 기능은 동작

st.set_page_config(page_title="NetSketch — FortiGate Topology", page_icon="🕸️", layout="wide")
st.title("🕸️ NetSketch — FortiGate 로그로 토폴로지 스케치")

# --------------------------------- Sidebar -----------------------------------
st.sidebar.header("설정")
agg_priv = st.sidebar.slider("내부 대역 집계 프리픽스(/n)", min_value=16, max_value=30, value=24)
agg_pub = st.sidebar.slider("외부 대역 집계 프리픽스(/n)", min_value=16, max_value=32, value=24)
max_rows = st.sidebar.number_input("최대 처리 행 수(성능보정)", min_value=1000, max_value=500000, value=50000, step=1000)

with st.sidebar.expander("컬럼 매핑(자동 감지, 필요시 수동 변경)"):
    st.write("일반적인 FortiGate 컬럼 별칭을 자동 인식합니다. 필요 시 직접 선택하세요.")

# --------------------------------- Columns -----------------------------------
CANON = [
    "time", "srcip", "dstip", "srcport", "dstport", "action",
    "snat", "dnat", "policyid", "in_interface", "out_interface", "proto",
]
SYNONYMS = {
    "time": ["time", "timestamp", "date", "logtime", "eventtime", "receivedtime"],
    "srcip": ["srcip", "src", "source", "src_ip", "srcaddr", "sourceip"],
    "dstip": ["dstip", "dst", "destination", "dst_ip", "dstaddr", "destinationip"],
    "srcport": ["srcport", "sport", "src_port", "s_port"],
    "dstport": ["dstport", "dport", "dst_port", "d_port"],
    "action": ["action", "act", "status", "result"],
    "snat": ["snat", "src_xlated_ip", "xsrcip", "nat_src", "transsrc", "translatedsrc", "nat", "xlated_src_ip"],
    "dnat": ["dnat", "dst_xlated_ip", "xdstip", "nat_dst", "transdst", "translateddst", "xlated_dst_ip"],
    "policyid": ["policyid", "policy_id", "policy", "policyid_"],
    "in_interface": ["in_interface", "inif", "ingress", "srcintf", "srcintfrole"],
    "out_interface": ["out_interface", "outif", "egress", "dstintf", "dstintfrole"],
    "proto": ["proto", "protocol", "ipproto"],
}


def _normalize_cols(cols: List[str]) -> List[str]:
    out = []
    for c in cols:
        out.append(c.strip().lower().replace(" ", "").replace("-", "_"))
    return out


def auto_map_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    mapping: Dict[str, Optional[str]] = {k: None for k in CANON}
    lc = _normalize_cols(list(df.columns))
    for canon, syns in SYNONYMS.items():
        for s in syns:
            if s in lc:
                mapping[canon] = df.columns[lc.index(s)]
                break
    return mapping


def apply_column_mapping(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
    out = pd.DataFrame()
    for canon in CANON:
        col = mapping.get(canon)
        if col and col in df.columns:
            out[canon] = df[col]
        else:
            out[canon] = pd.NA
    for p in ["srcport", "dstport", "policyid"]:
        if out[p].notna().any():
            out[p] = pd.to_numeric(out[p], errors="coerce")
    return out

# -------------------------------- Networking ---------------------------------

def ip_to_subnet(ip: str, agg_priv: int, agg_pub: int) -> Optional[str]:
    try:
        ip_obj = ipaddress.ip_address(str(ip))
    except Exception:
        return None
    prefix = agg_priv if ip_obj.is_private else agg_pub
    try:
        net = ipaddress.ip_network(f"{ip_obj}/{prefix}", strict=False)
        return f"{net.network_address}/{net.prefixlen}"
    except Exception:
        return None


def is_private_ip(ip: str) -> Optional[bool]:
    try:
        return ipaddress.ip_address(str(ip)).is_private
    except Exception:
        return None

# -------------------------------- File Upload --------------------------------
with st.expander("1) 로그 업로드", expanded=True):
    uploaded = st.file_uploader("FortiGate 로그 파일(CSV/TSV)", type=["csv", "tsv", "log", "txt"])   
    sample_btn = st.button("샘플 데이터 생성/로드")

if uploaded or sample_btn:
    if sample_btn and not uploaded:
        data = io.StringIO()
        data.write(
            "time,srcip,srcport,dstip,dstport,action,snat,dnat,policyid,in_interface,out_interface,proto\n"
            "2025-08-18 10:00,10.1.1.10,53123,8.8.8.8,53,accept,203.0.113.5,,100,port1,port2,udp\n"
            "2025-08-18 10:01,10.1.2.20,51789,1.1.1.1,53,accept,203.0.113.5,,100,port1,port2,udp\n"
            "2025-08-18 10:02,10.2.0.15,443,198.51.100.25,443,accept,203.0.113.9,,200,lan,wan,tcp\n"
            "2025-08-18 10:03,198.51.100.25,4443,10.2.0.15,4443,accept,,10.2.0.15,300,wan,lan,tcp\n"
        )
        data.seek(0)
        df_raw = pd.read_csv(data)
        st.info("샘플 데이터 4행 로드")
    else:
        content = uploaded.read()
        buf = io.StringIO(content.decode("utf-8", errors="ignore"))
        try:
            df_raw = pd.read_csv(buf, sep=None, engine="python")
        except Exception:
            buf.seek(0)
            df_raw = pd.read_csv(buf)

    if len(df_raw) > max_rows:
        st.warning(f"행 {len(df_raw):,} → {max_rows:,}로 샘플링")
        df_raw = df_raw.sample(n=max_rows, random_state=7)

    st.subheader("원본 미리보기")
    st.dataframe(df_raw.head(50), use_container_width=True)

    auto_map = auto_map_columns(df_raw)
    with st.sidebar:
        st.markdown("**자동 매핑 결과**")
        edits = {}
        for k in CANON:
            opts = [None] + list(df_raw.columns)
            sel = st.selectbox(k, options=opts, index=opts.index(auto_map[k]) if auto_map[k] in opts else 0, key=f"colmap_{k}")
            edits[k] = sel

    df = apply_column_mapping(df_raw, edits)

    def _valid_ip(s):
        try:
            ipaddress.ip_address(str(s)); return True
        except Exception:
            return False
    mask_valid = df["srcip"].apply(_valid_ip) & df["dstip"].apply(_valid_ip)
    df = df[mask_valid].reset_index(drop=True)
    if df.empty:
        st.error("유효 IP가 있는 행이 없습니다. 컬럼 매핑/파일을 확인하세요.")
        st.stop()

    df["eff_dstip"] = df["dnat"].where(df["dnat"].notna() & (df["dnat"].astype(str).str.len() > 0), df["dstip"]) 
    df["src_subnet"] = df["srcip"].apply(lambda ip: ip_to_subnet(str(ip), agg_priv, agg_pub))
    df["dst_subnet"] = df["eff_dstip"].apply(lambda ip: ip_to_subnet(str(ip), agg_priv, agg_pub))
    df["snat_ip"] = df["snat"].where(df["snat"].astype(str).str.len() > 0, pd.NA)

    # ------------------------------- Build Graph -----------------------------
    G = nx.DiGraph()

    def add_node_if_missing(node_id: str, role: str):
        if node_id not in G:
            color = {"internal": "#34a853", "public": "#4285f4", "nat": "#ea4335", "unknown": "#fbbc05"}.get(role, "#9aa0a6")
            G.add_node(node_id, label=node_id, color=color, role=role)

    subnets: Dict[str, str] = {}
    for col in ["srcip", "eff_dstip"]:
        for ip in df[col].dropna().astype(str).unique():
            subnet = ip_to_subnet(ip, agg_priv, agg_pub)
            if not subnet:
                continue
            role = "internal" if is_private_ip(ip) else "public"
            subnets[subnet] = role

    for subnet, role in subnets.items():
        add_node_if_missing(subnet, role)

    nat_ips = sorted(set([str(x) for x in df["snat_ip"].dropna().unique()]))
    for nip in nat_ips:
        add_node_if_missing(f"NAT {nip}", "nat")

    from collections import defaultdict
    edge_counts = defaultdict(int)
    edge_ports = defaultdict(set)

    for _, r in df.iterrows():
        s_net = r["src_subnet"]
        d_net = r["dst_subnet"]
        if not s_net or not d_net:
            continue
        if pd.notna(r["snat_ip"]) and str(r["snat_ip"]).strip():
            nat_node = f"NAT {str(r['snat_ip']).strip()}"
            edge_counts[(s_net, nat_node)] += 1
            edge_counts[(nat_node, d_net)] += 1
            if not pd.isna(r["dstport"]):
                edge_ports[(nat_node, d_net)].add(int(r["dstport"]))
        else:
            edge_counts[(s_net, d_net)] += 1
            if not pd.isna(r["dstport"]):
                edge_ports[(s_net, d_net)].add(int(r["dstport"]))

    for (u, v), w in edge_counts.items():
        ports = sorted(list(edge_ports[(u, v)]))
        label = f"{w} flows" + (f"\nports: {ports[:5]}" if ports else "")
        G.add_edge(u, v, weight=w, title=label)

    net = Network(height="650px", width="100%", directed=True, bgcolor="#0d1117", font_color="#e6edf3")
    net.toggle_physics(True)
    net.from_nx(G)

    max_w = max(edge_counts.values()) if edge_counts else 1
    for e in net.edges:
        w = edge_counts[(e["from"], e["to"])]
        e["width"] = max(1, 6 * (w / max_w))

    html_path = "netsketch_graph.html"
    try:
        net.write_html(html_path, open_browser=False, notebook=False)  # show() 대신 안전한 경로
        with open(html_path, "r", encoding="utf-8") as f:
            graph_html = f.read()
    except Exception as e:
        st.error("그래프 HTML 생성 중 오류가 발생했습니다. jinja2/pyvis 설치 상태를 확인하세요.")
        st.caption(f"RenderError: {type(e).__name__}: {e}")
        st.stop()

    st.subheader("토폴로지 그래프")
    st.components.v1.html(graph_html, height=680, scrolling=True)

    # --------------------------------- Tables --------------------------------
    st.subheader("요약 테이블")
    sub_df = pd.DataFrame({"subnet": list(subnets.keys()), "role": [subnets[s] for s in subnets]})
    nat_df = df.dropna(subset=["snat_ip"])[["src_subnet", "snat_ip"]].groupby(["src_subnet", "snat_ip"]).size().reset_index(name="flows")
    edges_df = pd.DataFrame([
        {"src": u, "dst": v, "flows": w, "ports_sample": ",".join(map(str, sorted(list(edge_ports[(u, v)])[:5])))}
        for (u, v), w in sorted(edge_counts.items(), key=lambda x: x[1], reverse=True)
    ])

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**서브넷**")
        st.dataframe(sub_df, use_container_width=True, height=250)
        st.download_button("다운로드: subnets.csv", sub_df.to_csv(index=False).encode("utf-8"), "subnets.csv", "text/csv")
    with c2:
        st.markdown("**NAT 매핑 (src_subnet → snat_ip)**")
        st.dataframe(nat_df, use_container_width=True, height=250)
        if not nat_df.empty:
            st.download_button("다운로드: nat_mappings.csv", nat_df.to_csv(index=False).encode("utf-8"), "nat_mappings.csv", "text/csv")
    with c3:
        st.markdown("**에지(흐름) Topology**")
        st.dataframe(edges_df.head(50), use_container_width=True, height=250)
        if not edges_df.empty:
            st.download_button("다운로드: edges.csv", edges_df.to_csv(index=False).encode("utf-8"), "edges.csv", "text/csv")

    st.download_button("다운로드: graph.html", graph_html.encode("utf-8"), "netsketch_graph.html", "text/html")

    # --------------------------------- LLM 요약 --------------------------------
    st.subheader("LLM 네트워크 요약")
    if use_llm:
        if not api_key_input:
            st.warning("API 키가 비어 있어 요약을 건너뜁니다.")
        elif not OA_OK:
            st.error("openai 패키지가 설치되어 있지 않습니다. 설치 후 다시 시도하세요.")
            st.code('pip install "openai>=1.20.0,<2" "httpx<0.28"', language="bash")
        else:
            try:
                client = OpenAI(api_key=api_key_input.strip())
                context = {
                    "subnets": sub_df.to_dict(orient="records")[:80],
                    "nat_ips": nat_ips[:80],
                    "top_edges": edges_df.head(30).to_dict(orient="records"),
                }
                sys_prompt = (
                    "너는 네트워크 포렌식 분석가다. 주어진 서브넷/에지/NAT 정보를 바탕으로, "
                    "내부-외부 세그먼트 구조, 게이트웨이/경계 추정, NAT 체인(SNAT/DNAT) 흐름을 설명하라. "
                    "의심 구간(외부→내부 고포트 유입, 비정상적인 다대일 SNAT 등)이 보이면 근거와 함께 짚어라. "
                    "과장은 금지하고, 사실 기반으로 5~8문장 내에서 한국어로 요약하라."
                )
                user_prompt = "다음은 네트워크 요약 입력이다. JSON을 해석해서 설명하라.\n\n" + json.dumps(context, ensure_ascii=False)
                with st.spinner("LLM이 네트워크 요약을 생성 중..."):
                    resp = client.chat.completions.create(
                        model=model_name,
                        temperature=float(temperature),
                        max_tokens=400,
                        messages=[
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                    )
                st.write(resp.choices[0].message.content.strip())
            except Exception as e:
                st.error(f"LLM 요약 중 오류: {type(e).__name__}")
    else:
        st.info("사이드바에서 'LLM 요약 사용'을 켜고 API 키를 입력하면, 네트워크 구조 요약을 생성합니다.")

else:
    st.info("왼쪽에서 로그 파일을 업로드하거나 '샘플 데이터 생성/로드'를 눌러 시작하세요.")

# --------------------------------- Footer ------------------------------------
st.caption("© 2025 NetSketch demo — 로그 구조가 달라도 컬럼 매핑으로 대응할 수 있습니다. 성능 문제 시 '최대 처리 행 수'를 줄이세요.")

