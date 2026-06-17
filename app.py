import re
import io
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing


st.set_page_config(
    page_title="LFDA Forecast",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --green: #38b249;
        --green-dark: #1a6b2f;
        --green-light: #e8f5e9;
        --blue: #1e7ec8;
        --blue-dark: #155fa0;
        --border: #b2d8b2;
        --muted: #4a7a4a;
        --text: #1a2e1a;
    }

    * { font-family: 'Inter', sans-serif !important; }

    .stApp {
        background: linear-gradient(135deg, #f0f7f0 0%, #f4f9f4 60%, #e8f5e9 100%);
        color: var(--text);
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 2px solid var(--border);
    }

    [data-testid="stSidebar"] * {
        color: var(--text) !important;
        font-family: 'Inter', sans-serif !important;
    }

    .main-header {
        padding: 1.25rem 0 1rem 0;
        border-bottom: 3px solid var(--green);
        margin-bottom: 1.4rem;
    }
    .main-title { font-size: 2.1rem; font-weight: 700; letter-spacing: -0.04em; color: var(--green-dark); margin-bottom: 0.15rem; }
    .main-subtitle { font-size: 1rem; color: var(--muted); font-weight: 400; margin-bottom: 0.45rem; }
    .navline { color: var(--muted); font-size: 0.87rem; }
    .status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--green); margin-right: 6px; box-shadow: 0 0 8px rgba(56,178,73,0.6); }

    div[data-testid="stMetric"] {
        background: #fff;
        border: 1px solid var(--border);
        border-top: 3px solid var(--green);
        border-radius: 14px;
        padding: .8rem 1rem;
        box-shadow: 0 2px 8px rgba(26,107,47,0.07);
    }
    div[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 12px; }

    .stButton>button, .stDownloadButton>button {
        border-radius: 10px; border: 2px solid var(--green-dark);
        background: var(--green-dark); color: #fff; font-weight: 600;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        background: var(--green); border-color: var(--green);
    }
    h1, h2, h3 { font-family: 'Inter', sans-serif !important; letter-spacing: -0.03em; color: var(--green-dark); }

    .capa-hero {
        background: linear-gradient(135deg, #1a6b2f 0%, #2d8a45 50%, #1e7ec8 100%);
        border-radius: 20px; padding: 3rem 2.5rem; color: white; margin-bottom: 2rem;
    }
    .capa-title { font-size: 2.8rem; font-weight: 700; letter-spacing: -0.04em; margin-bottom: 0.4rem; color: white; }
    .capa-subtitle { font-size: 1.15rem; font-weight: 400; opacity: 0.88; margin-bottom: 0.3rem; color: white; }
    .capa-inst { font-size: 0.92rem; opacity: 0.75; margin-bottom: 1.4rem; color: white; }
    .capa-badge {
        display: inline-block; background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.35); border-radius: 20px;
        padding: 0.3rem 1rem; font-size: 0.82rem; font-weight: 500;
        color: white; margin-right: 8px; margin-bottom: 8px;
    }

    .info-card {
        background: #fff; border: 1px solid var(--border);
        border-left: 4px solid var(--green); border-radius: 12px;
        padding: 1.2rem 1.4rem; margin-bottom: 1rem;
        color: var(--text);
    }
    .info-card-blue {
        background: #fff; border: 1px solid #b3d4ef;
        border-left: 4px solid var(--blue); border-radius: 12px;
        padding: 1.2rem 1.4rem; margin-bottom: 1rem;
        color: var(--text);
    }
    .info-card-yellow {
        background: #fffef5; border: 1px solid #e8d88a;
        border-left: 4px solid #d4a017; border-radius: 12px;
        padding: 1.2rem 1.4rem; margin-bottom: 1rem;
        color: var(--text);
    }

    .model-card {
        background: #fff; border: 1px solid var(--border);
        border-radius: 12px; padding: 1.1rem 1.3rem; margin-bottom: 10px;
    }
    .model-title { font-size: 0.95rem; font-weight: 700; color: var(--green-dark); margin-bottom: 0.4rem; }
    .model-desc { font-size: 0.83rem; color: var(--muted); line-height: 1.6; }
    .model-when { font-size: 0.78rem; color: var(--blue-dark); margin-top: 0.4rem; font-weight: 500; }

    .step-flow { display: flex; align-items: flex-start; gap: 0; margin: 1.5rem 0; flex-wrap: wrap; }
    .step-item { display: flex; flex-direction: column; align-items: center; min-width: 90px; flex: 1; }
    .step-circle {
        width: 48px; height: 48px; border-radius: 50%;
        background: var(--green-dark); color: white;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.1rem; font-weight: 700; margin-bottom: 0.5rem;
    }
    .step-label { font-size: 0.77rem; font-weight: 700; color: var(--green-dark); text-align: center; }
    .step-desc { font-size: 0.7rem; color: var(--muted); text-align: center; margin-top: 3px; line-height: 1.4; }
    .step-arrow { font-size: 1.4rem; color: var(--border); margin-top: 12px; padding: 0 2px; flex: 0; }

    .planilha-card {
        background: #fff; border: 1px solid var(--border);
        border-radius: 12px; padding: 1.2rem 1.5rem; margin-bottom: 1rem;
    }
    .col-row { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; padding: 10px 12px; border-radius: 8px; background: #f8fdf8; border: 1px solid #d4ecd4; }
    .col-name { font-weight: 700; color: var(--green-dark); font-size: 0.92rem; min-width: 120px; }
    .col-pos { font-size: 0.78rem; color: var(--blue); font-weight: 600; min-width: 90px; }
    .col-desc { font-size: 0.82rem; color: var(--muted); }
    .col-example { font-size: 0.78rem; color: #888; font-style: italic; }

    .small-muted { color: var(--muted); font-size: .88rem; }
    .green { color: var(--green); }
    .section-title { font-size: 1.15rem; font-weight: 700; color: var(--green-dark); margin: 1.5rem 0 0.8rem 0; border-left: 4px solid var(--green); padding-left: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="main-header">
        <div class="main-title">🌱 LFDA Forecast</div>
        <div class="main-subtitle">Sistema de Previsão de Demanda</div>
        <div class="navline">Upload • Validação • Modelos • Auditoria • Forecast • Exportação &nbsp;&nbsp;<span class="status-dot"></span>Processamento local</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Helpers
# =========================================================

def strip_accents(text):
    text = str(text)
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))

def normalize_col(col):
    col = strip_accents(str(col)).lower().strip()
    col = re.sub(r"[^a-z0-9]+", "_", col)
    return col.strip("_")

def clean_dataframe(df):
    df = df.copy()
    df = df.dropna(axis=1, how="all")
    df = df.loc[:, ~df.columns.astype(str).str.lower().str.contains(r"^unnamed")]
    df = df.dropna(axis=0, how="all")
    return df

def infer_columns(df):
    norm_map = {col: normalize_col(col) for col in df.columns}
    product_keys = ["produto", "product", "item", "sku", "material", "codigo", "descricao"]
    date_keys = ["data", "date", "mes", "month", "periodo", "competencia", "ano_mes"]
    qty_keys = ["quantidade", "quantity", "qty", "qtd", "demanda", "demand", "volume", "vendas"]
    product_col = next((c for c, n in norm_map.items() if any(k in n for k in product_keys)), None)
    date_col = next((c for c, n in norm_map.items() if any(k in n for k in date_keys)), None)
    qty_col = next((c for c, n in norm_map.items() if any(k in n for k in qty_keys)), None)
    if qty_col is None:
        numeric_scores = [(c, pd.to_numeric(df[c], errors="coerce").notna().sum()) for c in df.columns]
        if numeric_scores:
            qty_col = max(numeric_scores, key=lambda x: x[1])[0]
    return product_col, date_col, qty_col

MONTHS_PT = {
    "janeiro": 1, "jan": 1, "fevereiro": 2, "fev": 2,
    "marco": 3, "mar": 3, "março": 3, "abril": 4, "abr": 4,
    "maio": 5, "mai": 5, "junho": 6, "jun": 6, "julho": 7, "jul": 7,
    "agosto": 8, "ago": 8, "setembro": 9, "set": 9,
    "outubro": 10, "out": 10, "novembro": 11, "nov": 11, "dezembro": 12, "dez": 12,
}

def parse_month_date(value):
    if pd.isna(value):
        return pd.NaT
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.to_datetime(value).to_period("M").to_timestamp()
    raw = str(value).strip().lower()
    raw = strip_accents(raw)
    raw = raw.replace(" de ", " ").replace("/", " ").replace("-", " ").replace(".", " ")
    raw = re.sub(r"\s+", " ", raw).strip()
    m = re.match(r"^(\d{1,2})\s+(\d{4})$", raw)
    if m:
        month, year = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12:
            return pd.Timestamp(year=year, month=month, day=1)
    parts = raw.split()
    if len(parts) >= 2:
        year_match = re.search(r"(20\d{2}|19\d{2})", raw)
        if parts[0] in MONTHS_PT and year_match:
            return pd.Timestamp(year=int(year_match.group(1)), month=MONTHS_PT[parts[0]], day=1)
    parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
    if pd.notna(parsed):
        return parsed.to_period("M").to_timestamp()
    return pd.NaT

def prepare_monthly(df, product_col, date_col, qty_col):
    out = df[[product_col, date_col, qty_col]].copy()
    out.columns = ["Produto", "Data", "Quantidade"]
    out["Produto"] = out["Produto"].astype(str).str.strip()
    out["Data"] = out["Data"].apply(parse_month_date)
    out["Quantidade"] = pd.to_numeric(out["Quantidade"], errors="coerce").fillna(0.0)
    out = out.dropna(subset=["Data"])
    out = out[out["Produto"].notna() & (out["Produto"].astype(str).str.len() > 0)]
    out = out.groupby(["Produto", "Data"], as_index=False)["Quantidade"].sum()
    return out.sort_values(["Produto", "Data"])

def monthly_to_quarterly(monthly):
    q = monthly.copy()
    q["Trimestre"] = q["Data"].dt.to_period("Q").dt.to_timestamp()
    q = q.groupby(["Produto", "Trimestre"], as_index=False)["Quantidade"].sum()
    return q.sort_values(["Produto", "Trimestre"])

def fill_missing_quarters(qdf):
    frames = []
    for prod, g in qdf.groupby("Produto"):
        g = g.sort_values("Trimestre")
        full_idx = pd.period_range(g["Trimestre"].min(), g["Trimestre"].max(), freq="Q").to_timestamp()
        aux = pd.DataFrame({"Trimestre": full_idx})
        aux["Produto"] = prod
        aux = aux.merge(g, on=["Produto", "Trimestre"], how="left")
        aux["Quantidade"] = aux["Quantidade"].fillna(0.0)
        frames.append(aux)
    return pd.concat(frames, ignore_index=True) if frames else qdf

def treat_zeros_preserve_total(values):
    y = np.array(values, dtype=float).copy()
    n = len(y)
    i = 0
    while i < n:
        if y[i] == 0:
            start_zero = i
            while i < n and y[i] == 0:
                i += 1
            left, right = start_zero - 1, i
            if left >= 0 and right < n and y[left] > 0 and y[right] > 0:
                k = i - start_zero
                avg = (y[left] + y[right]) / 2.0
                per_zero = avg / k
                y[start_zero:i] = per_zero
                denom = y[left] + y[right]
                if denom > 0:
                    y[left] = max(0.0, y[left] - avg * (y[left] / denom))
                    y[right] = max(0.0, y[right] - avg * (y[right] / denom))
        else:
            i += 1
    return y

def split_train_test(y):
    n = len(y)
    if n >= 17: return y.iloc[:-4], y.iloc[-4:], "13+ treino / 4 teste"
    if 12 <= n <= 16: return y.iloc[:-4], y.iloc[-4:], "Últimos 4 períodos em teste"
    if 8 <= n <= 11: return y.iloc[:-3], y.iloc[-3:], "Últimos 3 períodos em teste"
    if 5 <= n <= 7: return y.iloc[:-2], y.iloc[-2:], "Últimos 2 períodos em teste"
    if 3 <= n <= 4: return y.iloc[:-1], y.iloc[-1:], "Último período em teste"
    return y, pd.Series(dtype=float), "Forecast exploratório sem teste"

def safe_mape(actual, pred):
    actual, pred = np.array(actual, dtype=float), np.array(pred, dtype=float)
    mask = actual != 0
    if mask.sum() == 0: return np.nan
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100)

def agg_mape(actual, pred):
    s, p = float(np.sum(actual)), float(np.sum(pred))
    if s == 0: return np.nan
    return abs(s - p) / abs(s) * 100

def model_params(fit):
    params = getattr(fit, "params", {}) or {}
    return {"Alpha": params.get("smoothing_level", np.nan),
            "Beta": params.get("smoothing_trend", np.nan),
            "Gamma": params.get("smoothing_seasonal", np.nan)}

def fit_forecast_model(model_name, train, steps):
    y = train.astype(float)
    if len(y) < 2:
        return None, pd.Series(np.repeat(y.iloc[-1] if len(y) else 0.0, steps))
    if model_name == "SES":
        fit = SimpleExpSmoothing(y, initialization_method="estimated").fit(optimized=True)
    elif model_name == "Holt":
        fit = Holt(y, initialization_method="estimated").fit(optimized=True)
    elif model_name == "Holt-Winters":
        if len(y) >= 8:
            fit = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=4,
                                       initialization_method="estimated").fit(optimized=True)
        else:
            fit = Holt(y, initialization_method="estimated").fit(optimized=True)
    else:
        raise ValueError("Modelo inválido")
    return fit, fit.forecast(steps)

def future_quarters(last_quarter, periods=4):
    start = pd.Period(last_quarter, freq="Q") + 1
    return list(pd.period_range(start, periods=periods, freq="Q").to_timestamp())

def quarter_label(ts):
    p = pd.Period(ts, freq="Q")
    return f"{p.year}T{p.quarter}"

def run_models(q):
    treated_frames, summaries, audits, forecasts = [], [], [], []
    for prod, g in q.groupby("Produto"):
        g = g.sort_values("Trimestre").copy()
        treated = treat_zeros_preserve_total(g["Quantidade"].astype(float).values)
        g["Valor Tratado"] = treated
        treated_frames.append(g)
        y = pd.Series(treated, index=g["Trimestre"])
        train, test, split_rule = split_train_test(y)
        for model_name in ["SES", "Holt", "Holt-Winters"]:
            test_steps = len(test)
            try:
                fit_bt, pred_test = fit_forecast_model(model_name, train, max(test_steps, 1))
                pred_test = pd.Series(pred_test).iloc[:test_steps].values if test_steps > 0 else np.array([])
                period_mape = safe_mape(test.values, pred_test) if test_steps else np.nan
                aggregate_mape = agg_mape(test.values, pred_test) if test_steps else np.nan
                params = model_params(fit_bt) if fit_bt is not None else {"Alpha": np.nan, "Beta": np.nan, "Gamma": np.nan}
                for idx, dt in enumerate(test.index):
                    actual_o = float(g.loc[g["Trimestre"] == dt, "Quantidade"].iloc[0])
                    actual_t = float(test.iloc[idx])
                    pred = float(pred_test[idx]) if idx < len(pred_test) else np.nan
                    err = actual_t - pred if pd.notna(pred) else np.nan
                    mape_i = abs(err / actual_t) * 100 if actual_t != 0 and pd.notna(err) else np.nan
                    audits.append({"Produto": prod, "Modelo": model_name, "Tipo": "Teste",
                                   "Trimestre": quarter_label(dt), "Valor Real": actual_o,
                                   "Valor Tratado": actual_t, "Valor Previsto": pred,
                                   "Erro": err, "MAPE %": mape_i,
                                   "Alpha": params["Alpha"], "Beta": params["Beta"], "Gamma": params["Gamma"],
                                   "Regra Treino/Teste": split_rule})
                summaries.append({"Produto": prod, "Modelo": model_name, "Períodos": len(y),
                                   "Treino": len(train), "Teste": len(test), "Regra": split_rule,
                                   "MAPE Período Médio %": period_mape, "MAPE Agregado %": aggregate_mape,
                                   "Alpha": params["Alpha"], "Beta": params["Beta"], "Gamma": params["Gamma"]})
                fit_full, fc = fit_forecast_model(model_name, y, 4)
                fc = pd.Series(fc).astype(float).values
                residual_std = np.nan
                if fit_bt is not None and test_steps > 1:
                    residual_std = float(np.nanstd(test.values - pred_test, ddof=1))
                elif fit_full is not None and len(y) > 2:
                    fitted = np.array(getattr(fit_full, "fittedvalues", []), dtype=float)
                    if len(fitted) == len(y):
                        residual_std = float(np.nanstd(y.values - fitted, ddof=1))
                if pd.isna(residual_std) or residual_std == 0:
                    residual_std = float(np.nanstd(y.values, ddof=1)) if len(y) > 1 else 0.0
                z90 = 1.645
                future = future_quarters(g["Trimestre"].max(), 4)
                params_full = model_params(fit_full) if fit_full is not None else params
                for h, dt in enumerate(future, start=1):
                    pv = max(0.0, float(fc[h - 1]))
                    interval = z90 * residual_std * np.sqrt(h)
                    forecasts.append({"Produto": prod, "Modelo": model_name, "Trimestre": quarter_label(dt),
                                      "Forecast": pv, "IC 90% Inferior": max(0.0, pv - interval),
                                      "IC 90% Superior": pv + interval,
                                      "Alpha": params_full["Alpha"], "Beta": params_full["Beta"], "Gamma": params_full["Gamma"]})
            except Exception as exc:
                summaries.append({"Produto": prod, "Modelo": model_name, "Períodos": len(y),
                                   "Treino": len(train), "Teste": len(test), "Regra": split_rule,
                                   "MAPE Período Médio %": np.nan, "MAPE Agregado %": np.nan,
                                   "Alpha": np.nan, "Beta": np.nan, "Gamma": np.nan, "Erro Modelo": str(exc)})
    treated_df = pd.concat(treated_frames, ignore_index=True) if treated_frames else pd.DataFrame()
    forecast_df = pd.DataFrame(forecasts)
    if not forecast_df.empty:
        annual = forecast_df.groupby(["Produto", "Modelo"], as_index=False)["IC 90% Superior"].sum()
        annual = annual.rename(columns={"IC 90% Superior": "Soma 4T - Limite Superior"})
        forecast_df = forecast_df.merge(annual, on=["Produto", "Modelo"], how="left")
    return treated_df, pd.DataFrame(summaries), pd.DataFrame(audits), forecast_df

def to_excel_bytes(sheets):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
            wb, ws = writer.book, writer.sheets[name[:31]]
            hfmt = wb.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": "#1a6b2f", "border": 1})
            nfmt = wb.add_format({"num_format": "#,##0.00"})
            for i, col in enumerate(df.columns):
                ws.write(0, i, col, hfmt)
                w = max(12, min(32, len(str(col)) + 4))
                ws.set_column(i, i, w, nfmt if pd.api.types.is_numeric_dtype(df[col]) else None)
            ws.freeze_panes(1, 0)
    return output.getvalue()


# =========================================================
# Sidebar
# =========================================================
st.sidebar.markdown("### 🌱 LFDA Forecast")
st.sidebar.markdown("<span class='small-muted'>Processamento local • Streamlit</span>", unsafe_allow_html=True)

page = st.sidebar.radio("Menu", ["Início", "Upload", "Validação", "Modelos", "Auditoria", "Forecast", "Exportação"], index=0)

st.sidebar.divider()
st.sidebar.markdown("**Configuração atual**")
st.sidebar.caption("Modelos: SES, Holt e Holt-Winters")
st.sidebar.caption("Parâmetros: otimização automática Python")
st.sidebar.caption("Forecast: próximos 4 trimestres")
st.sidebar.caption("Intervalo de confiança: 90%")

for key in ["raw_df", "monthly", "quarterly", "treated", "summary", "audit", "forecast", "inferred_cols"]:
    if key not in st.session_state:
        st.session_state[key] = None

def process_file(uploaded_file):
    raw = pd.read_excel(uploaded_file) if uploaded_file.name.lower().endswith((".xlsx", ".xls")) else pd.read_csv(uploaded_file, sep=None, engine="python")
    raw = clean_dataframe(raw)
    product_col, date_col, qty_col = infer_columns(raw)
    if not all([product_col, date_col, qty_col]):
        raise ValueError("Não foi possível identificar as colunas de Produto, Data e Quantidade.")
    monthly = prepare_monthly(raw, product_col, date_col, qty_col)
    quarterly = fill_missing_quarters(monthly_to_quarterly(monthly))
    treated, summary, audit, forecast = run_models(quarterly)
    st.session_state.raw_df = raw
    st.session_state.monthly = monthly
    st.session_state.quarterly = quarterly
    st.session_state.treated = treated
    st.session_state.summary = summary
    st.session_state.audit = audit
    st.session_state.forecast = forecast
    st.session_state.inferred_cols = (product_col, date_col, qty_col)

def has_results():
    return st.session_state.summary is not None and not st.session_state.summary.empty


# =========================================================
# INÍCIO
# =========================================================
if page == "Início":

    st.markdown("""
    <div class="capa-hero">
        <div class="capa-title">🌱 LFDA Forecast</div>
        <div class="capa-subtitle">Sistema de Previsão de Demanda de Insumos Agropecuários</div>
        <div class="capa-inst">Trabalho de Conclusão de Curso — Engenharia de Produção · UFRGS</div>
        <span class="capa-badge">📊 Modelos estatísticos</span>
        <span class="capa-badge">🔍 Auditoria completa</span>
        <span class="capa-badge">📁 Exportação Excel</span>
        <span class="capa-badge">🔒 Processamento local</span>
        <span class="capa-badge">🎓 UFRGS · Eng. de Produção</span>
    </div>
    """, unsafe_allow_html=True)

    # --- Sobre ---
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown('<div class="section-title">Sobre o projeto</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card">
            <strong>O que é o LFDA Forecast?</strong><br><br>
            Ferramenta desenvolvida por alunos do curso de <strong>Engenharia de Produção da UFRGS</strong>
            como parte de um Trabalho de Conclusão de Curso (TCC). O objetivo é oferecer à
            <strong>LFDA-RS</strong> uma solução prática e acessível para previsão de demanda de
            insumos agropecuários, sem exigir conhecimento técnico em programação ou estatística.
            <br><br>
            O sistema automatiza todo o processo — da leitura da planilha até a exportação dos
            resultados — rodando diretamente no navegador, sem envio de dados para servidores externos.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card-blue">
            <strong>🎯 Objetivo do sistema</strong><br><br>
            Gerar previsões trimestrais de demanda para cada produto da LFDA-RS, comparando três
            modelos estatísticos clássicos e indicando automaticamente qual deles melhor se ajusta
            ao histórico de cada produto — com base no menor erro de previsão (MAPE).
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">Desenvolvedores</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card-yellow">
            <strong>🎓 Equipe UFRGS</strong><br><br>
            Projeto desenvolvido por alunos de graduação em <strong>Engenharia de Produção</strong>
            da Universidade Federal do Rio Grande do Sul (UFRGS), em parceria com a LFDA-RS.<br><br>
            <strong>Disciplina:</strong> Trabalho de Conclusão de Curso<br>
            <strong>Instituição:</strong> UFRGS — Porto Alegre, RS<br>
            <strong>Parceiro:</strong> LFDA-RS
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <strong>⚙️ Tecnologias utilizadas</strong><br><br>
            Python • Streamlit • Pandas • NumPy • Statsmodels • XlsxWriter<br><br>
            Modelos: SES, Holt e Holt-Winters (suavização exponencial)
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --- Formato da planilha ---
    st.markdown('<div class="section-title">📋 Formato esperado da planilha de entrada</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card-yellow">
        <strong>⚠️ Atenção:</strong> Para que o sistema funcione corretamente, a planilha deve seguir
        exatamente o formato abaixo. O nome das colunas pode variar (o sistema reconhece variações),
        mas os dados de cada coluna precisam estar no padrão indicado.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="planilha-card">
        <div style="font-weight:700; color:var(--green-dark); margin-bottom:14px; font-size:0.95rem;">Estrutura obrigatória — 3 colunas:</div>

        <div class="col-row">
            <div class="col-name">📦 Produto</div>
            <div class="col-pos">Coluna A (col. 1)</div>
            <div class="col-desc">Nome ou código do produto. Deve ser texto. O sistema agrupa todas as linhas com o mesmo nome.</div>
            <div class="col-example">Ex: "Metanol", "Acetonitrila", "Tubo Falcon 15ml"</div>
        </div>

        <div class="col-row">
            <div class="col-name">📅 Data</div>
            <div class="col-pos">Coluna B (col. 2)</div>
            <div class="col-desc">Mês de referência da demanda. Aceita datas em português, barras ou formato padrão.</div>
            <div class="col-example">Ex: "março de 2026", "Mar/2026", "03/2026", "01/03/2026"</div>
        </div>

        <div class="col-row">
            <div class="col-name">🔢 Quantidade</div>
            <div class="col-pos">Coluna C (col. 3)</div>
            <div class="col-desc">Volume ou demanda do produto naquele mês. Deve ser número inteiro ou decimal. Não use texto, símbolos ou unidades.</div>
            <div class="col-example">Ex: 5000, 16000, 1500 — não use "5.000 L" ou "R$ 16.000"</div>
        </div>

        <div style="margin-top:14px; font-size:0.82rem; color:var(--muted);">
            ✅ Colunas extras (como índices ou colunas vazias) são ignoradas automaticamente.<br>
            ✅ A base deve ser <strong>mensal</strong> — o sistema converte para trimestral automaticamente.<br>
            ✅ Cada linha = um produto em um mês. Pode haver múltiplos produtos na mesma planilha.<br>
            ✅ Não é necessário ordenar por data — o sistema ordena internamente.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Exemplo real da planilha utilizada pela LFDA-RS:**")
    st.dataframe(pd.DataFrame({
        "Produto": ["Metanol", "Acetonitrila", "Tubo Falcon 15ml", "Tubo Falcon 50ml", "Vial Vidro"],
        "Data": ["março de 2026", "março de 2026", "março de 2026", "março de 2026", "março de 2026"],
        "Quantidade": [5000, 16000, 1000, 1000, 500],
    }), use_container_width=True)

    st.markdown("---")

    # --- Modelos ---
    st.markdown('<div class="section-title">📐 Como funcionam os modelos estatísticos</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="model-card">
            <div class="model-title">📈 SES — Suavização Exponencial Simples</div>
            <div class="model-desc">
                O modelo mais básico da família de suavização exponencial. Ele calcula uma
                <strong>média ponderada do histórico</strong>, dando mais peso aos períodos
                mais recentes e menos peso aos mais antigos.<br><br>
                O parâmetro <strong>Alpha (α)</strong> controla essa ponderação: valores altos
                de α fazem o modelo reagir rapidamente a mudanças recentes; valores baixos
                produzem previsões mais estáveis.<br><br>
                O SES projeta um valor constante para o futuro — ele não captura tendência de
                crescimento ou queda, nem variação sazonal.
            </div>
            <div class="model-when">✅ Indicado quando: a demanda oscila sem tendência clara e sem padrão sazonal definido.</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="model-card">
            <div class="model-title">📈📈 Holt — Suavização com Tendência</div>
            <div class="model-desc">
                Extensão do SES que adiciona um <strong>componente de tendência</strong> à previsão.
                Enquanto o SES projeta um valor fixo, o Holt projeta uma linha com inclinação —
                capturando se a demanda está crescendo ou caindo ao longo do tempo.<br><br>
                Usa dois parâmetros: <strong>Alpha (α)</strong> para o nível e
                <strong>Beta (β)</strong> para a tendência. Ambos são otimizados automaticamente
                pelo sistema para minimizar o erro histórico.<br><br>
                Não captura sazonalidade — apenas a direção geral da série.
            </div>
            <div class="model-when">✅ Indicado quando: a demanda tem crescimento ou queda consistente ao longo dos trimestres.</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="model-card">
            <div class="model-title">📈📈📈 Holt-Winters — Tendência e Sazonalidade</div>
            <div class="model-desc">
                O modelo mais completo da família. Além do nível e da tendência, adiciona um
                <strong>componente sazonal</strong> que captura padrões que se repetem a cada
                4 trimestres (ciclo anual).<br><br>
                Usa três parâmetros: <strong>Alpha (α)</strong> para nível, <strong>Beta (β)</strong>
                para tendência e <strong>Gamma (γ)</strong> para sazonalidade. Requer pelo menos
                8 trimestres de histórico para funcionar corretamente.<br><br>
                Quando há menos de 8 trimestres disponíveis, o sistema usa o Holt como fallback.
            </div>
            <div class="model-when">✅ Indicado quando: a demanda tem picos e vales que se repetem a cada ano.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --- Passo a passo ---
    st.markdown('<div class="section-title">🗺️ Fluxo completo do sistema — passo a passo</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="step-flow">
        <div class="step-item">
            <div class="step-circle">1</div>
            <div class="step-label">Upload</div>
            <div class="step-desc">Envie o Excel com Produto, Data e Quantidade</div>
        </div>
        <div class="step-arrow">→</div>
        <div class="step-item">
            <div class="step-circle">2</div>
            <div class="step-label">Validação</div>
            <div class="step-desc">Sistema identifica colunas e converte datas automaticamente</div>
        </div>
        <div class="step-arrow">→</div>
        <div class="step-item">
            <div class="step-circle">3</div>
            <div class="step-label">Trimestral</div>
            <div class="step-desc">Base mensal é agregada em trimestres internamente</div>
        </div>
        <div class="step-arrow">→</div>
        <div class="step-item">
            <div class="step-circle">4</div>
            <div class="step-label">Zeros</div>
            <div class="step-desc">Zeros entre períodos são redistribuídos sem alterar o total</div>
        </div>
        <div class="step-arrow">→</div>
        <div class="step-item">
            <div class="step-circle">5</div>
            <div class="step-label">Split</div>
            <div class="step-desc">Histórico é dividido em treino e teste de forma adaptativa</div>
        </div>
        <div class="step-arrow">→</div>
        <div class="step-item">
            <div class="step-circle">6</div>
            <div class="step-label">Modelos</div>
            <div class="step-desc">SES, Holt e Holt-Winters são calibrados automaticamente</div>
        </div>
        <div class="step-arrow">→</div>
        <div class="step-item">
            <div class="step-circle">7</div>
            <div class="step-label">MAPE</div>
            <div class="step-desc">Erro de cada modelo é calculado período a período e no agregado</div>
        </div>
        <div class="step-arrow">→</div>
        <div class="step-item">
            <div class="step-circle">8</div>
            <div class="step-label">Forecast</div>
            <div class="step-desc">Previsão dos próximos 4 trimestres com intervalo de confiança de 90%</div>
        </div>
        <div class="step-arrow">→</div>
        <div class="step-item">
            <div class="step-circle">9</div>
            <div class="step-label">Exportação</div>
            <div class="step-desc">Resultados completos em Excel com todas as abas formatadas</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # --- Split adaptativo ---
    st.markdown('<div class="section-title">✂️ Como funciona o split treino/teste adaptativo</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
        O sistema divide o histórico de cada produto em dois blocos: <strong>treino</strong> (usado para calibrar o modelo)
        e <strong>teste</strong> (usado para medir o erro de previsão). A divisão é adaptativa conforme o histórico disponível:
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(pd.DataFrame({
        "Histórico disponível": ["17 ou mais trimestres", "12 a 16 trimestres", "8 a 11 trimestres",
                                  "5 a 7 trimestres", "3 a 4 trimestres", "Menos de 3 trimestres"],
        "Períodos de treino": ["13", "Restante", "Restante", "Restante", "Restante", "Todos"],
        "Períodos de teste": ["4", "4", "3", "2", "1", "0"],
        "Confiança do MAPE": ["Alta", "Moderada", "Moderada", "Baixa", "Baixa", "Exploratória"],
    }), use_container_width=True)

    st.markdown("""
    <div class="info-card-blue">
        <strong>O que é o MAPE?</strong><br><br>
        MAPE significa <em>Mean Absolute Percentage Error</em> — Erro Percentual Absoluto Médio.
        Ele mede, em porcentagem, o quanto o modelo errou em relação à demanda real no período de teste.<br><br>
        Um MAPE de <strong>10%</strong> significa que o modelo errou, em média, 10% da demanda real.
        Quanto menor o MAPE, melhor o modelo.<br><br>
        O sistema calcula dois tipos: <strong>MAPE por período</strong> (erro em cada trimestre de teste)
        e <strong>MAPE agregado</strong> (erro na soma total do período de teste).
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# UPLOAD
# =========================================================
elif page == "Upload":
    st.subheader("Upload da base")
    st.markdown("<div class='small-muted'>Envie uma base mensal em Excel ou CSV com as colunas Produto, Data e Quantidade.</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Arquivo de entrada", type=["xlsx", "xls", "csv"])
    if uploaded is not None:
        try:
            process_file(uploaded)
            st.success("Base processada com sucesso.")
        except Exception as e:
            st.error(f"Erro ao processar arquivo: {e}")
    if st.session_state.raw_df is not None:
        product_col, date_col, qty_col = st.session_state.inferred_cols
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Linhas originais", f"{len(st.session_state.raw_df):,}".replace(",", "."))
        c2.metric("Produtos", st.session_state.monthly["Produto"].nunique())
        c3.metric("Meses válidos", st.session_state.monthly["Data"].nunique())
        c4.metric("Trimestres", st.session_state.quarterly["Trimestre"].nunique())
        st.markdown("### Colunas identificadas")
        st.dataframe(pd.DataFrame({"Campo": ["Produto", "Data", "Quantidade"], "Coluna identificada": [product_col, date_col, qty_col]}), use_container_width=True)
        st.markdown("### Prévia da base original")
        st.dataframe(st.session_state.raw_df.head(30), use_container_width=True)


# =========================================================
# VALIDAÇÃO
# =========================================================
elif page == "Validação":
    st.subheader("Validação e conversão mensal → trimestral")
    if st.session_state.monthly is None:
        st.info("Faça o upload de uma base para visualizar a validação.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Registros mensais", len(st.session_state.monthly))
        c2.metric("Registros trimestrais", len(st.session_state.quarterly))
        c3.metric("Demanda total", f"{st.session_state.monthly['Quantidade'].sum():,.0f}".replace(",", "."))
        st.markdown("### Base mensal padronizada")
        st.dataframe(st.session_state.monthly, use_container_width=True)
        st.markdown("### Base trimestral construída internamente")
        q_view = st.session_state.quarterly.copy()
        q_view["Trimestre"] = q_view["Trimestre"].apply(quarter_label)
        st.dataframe(q_view, use_container_width=True)
        st.markdown("### Tratamento dos zeros")
        t_view = st.session_state.treated.copy()
        t_view["Trimestre"] = t_view["Trimestre"].apply(quarter_label)
        t_view["Diferença Tratamento"] = t_view["Valor Tratado"] - t_view["Quantidade"]
        st.dataframe(t_view, use_container_width=True)


# =========================================================
# MODELOS
# =========================================================
elif page == "Modelos":
    st.subheader("Backtest dos modelos")
    if not has_results():
        st.info("Faça o upload de uma base para rodar os modelos.")
    else:
        summary = st.session_state.summary.copy()
        best = summary.sort_values(["Produto", "MAPE Agregado %"], na_position="last").groupby("Produto").head(1)
        c1, c2, c3 = st.columns(3)
        c1.metric("Produtos modelados", summary["Produto"].nunique())
        c2.metric("Modelos rodados", len(summary))
        c3.metric("MAPE agregado mediano", f"{summary['MAPE Agregado %'].median():.1f}%" if summary["MAPE Agregado %"].notna().any() else "n/a")
        st.markdown("### Resultado por produto e modelo")
        st.dataframe(summary, use_container_width=True)
        st.markdown("### Melhor modelo por produto — critério: menor MAPE agregado")
        st.dataframe(best, use_container_width=True)


# =========================================================
# AUDITORIA
# =========================================================
elif page == "Auditoria":
    st.subheader("Auditoria estatística")
    if st.session_state.audit is None or st.session_state.audit.empty:
        st.info("Faça o upload de uma base para gerar a auditoria.")
    else:
        audit = st.session_state.audit.copy()
        c1, c2 = st.columns(2)
        prod_filter = c1.selectbox("Produto", ["Todos"] + sorted(audit["Produto"].unique().tolist()))
        model_filter = c2.selectbox("Modelo", ["Todos"] + sorted(audit["Modelo"].unique().tolist()))
        if prod_filter != "Todos": audit = audit[audit["Produto"] == prod_filter]
        if model_filter != "Todos": audit = audit[audit["Modelo"] == model_filter]
        st.dataframe(audit, use_container_width=True)


# =========================================================
# FORECAST
# =========================================================
elif page == "Forecast":
    st.subheader("Forecast — próximos 4 trimestres")
    if st.session_state.forecast is None or st.session_state.forecast.empty:
        st.info("Faça o upload de uma base para gerar o forecast.")
    else:
        fc = st.session_state.forecast.copy()
        c1, c2 = st.columns(2)
        prod_filter = c1.selectbox("Produto", ["Todos"] + sorted(fc["Produto"].unique().tolist()))
        model_filter = c2.selectbox("Modelo", ["Todos"] + sorted(fc["Modelo"].unique().tolist()))
        if prod_filter != "Todos": fc = fc[fc["Produto"] == prod_filter]
        if model_filter != "Todos": fc = fc[fc["Modelo"] == model_filter]
        st.dataframe(fc, use_container_width=True)
        st.markdown("### Soma anual do limite superior por produto e modelo")
        annual = fc.groupby(["Produto", "Modelo"], as_index=False)["IC 90% Superior"].sum()
        annual = annual.rename(columns={"IC 90% Superior": "Soma 4T - Limite Superior"})
        st.dataframe(annual, use_container_width=True)


# =========================================================
# EXPORTAÇÃO
# =========================================================
elif page == "Exportação":
    st.subheader("Exportação Excel")
    if not has_results():
        st.info("Faça o upload de uma base para habilitar a exportação.")
    else:
        sheets = {
            "Base Original": st.session_state.raw_df,
            "Mensal Padronizada": st.session_state.monthly,
            "Trimestral": st.session_state.quarterly.assign(Trimestre=st.session_state.quarterly["Trimestre"].apply(quarter_label)),
            "Tratamento Zeros": st.session_state.treated.assign(Trimestre=st.session_state.treated["Trimestre"].apply(quarter_label)),
            "Modelos MAPE": st.session_state.summary,
            "Auditoria": st.session_state.audit,
            "Forecast": st.session_state.forecast,
        }
        excel = to_excel_bytes(sheets)
        st.download_button(
            label="⬇️ Baixar resultado em Excel",
            data=excel,
            file_name="LFDA_Forecast_Resultados.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.markdown("<div class='small-muted'>Contém: base original, base mensal, trimestral, tratamento dos zeros, MAPE, auditoria e forecast.</div>", unsafe_allow_html=True)
