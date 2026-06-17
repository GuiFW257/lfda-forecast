import re
import io
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing


# =========================================================
# LFDA FORECAST — Sistema de Previsão de Demanda
# =========================================================

st.set_page_config(
    page_title="LFDA Forecast",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# CSS — Campo Moderno + Inter
# -----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --bg: #f4f9f4;
        --panel: #ffffff;
        --panel-2: #e8f5e9;
        --border: #b2d8b2;
        --text: #1a2e1a;
        --muted: #4a7a4a;
        --green: #38b249;
        --green-dark: #1a6b2f;
        --green-light: #e8f5e9;
        --blue: #1e7ec8;
        --blue-dark: #155fa0;
        --blue-light: #e3f2fd;
        --red: #e53935;
        --yellow: #f9a825;
        --dark: #2d3a2e;
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
        padding: 1.25rem 0 1.1rem 0;
        border-bottom: 3px solid var(--green);
        margin-bottom: 1.4rem;
    }

    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.04em;
        color: var(--green-dark);
        margin-bottom: 0.15rem;
    }

    .main-subtitle {
        font-size: 1rem;
        color: var(--muted);
        margin-bottom: 0.45rem;
        font-weight: 400;
    }

    .navline {
        color: var(--muted);
        font-size: 0.87rem;
        font-weight: 400;
    }

    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--green);
        margin-right: 6px;
        box-shadow: 0 0 8px rgba(56, 178, 73, 0.6);
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--border);
        border-top: 3px solid var(--green);
        border-radius: 14px;
        padding: .8rem 1rem;
        box-shadow: 0 2px 8px rgba(26, 107, 47, 0.07);
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    .stButton>button, .stDownloadButton>button {
        border-radius: 10px;
        border: 2px solid var(--green-dark);
        background: var(--green-dark);
        color: #ffffff;
        font-weight: 600;
        font-family: 'Inter', sans-serif !important;
    }

    .stButton>button:hover, .stDownloadButton>button:hover {
        background: var(--green);
        border-color: var(--green);
        color: white;
    }

    h1, h2, h3 {
        font-family: 'Inter', sans-serif !important;
        letter-spacing: -0.03em;
        color: var(--green-dark);
    }

    /* Capa */
    .capa-hero {
        background: linear-gradient(135deg, #1a6b2f 0%, #2d8a45 50%, #1e7ec8 100%);
        border-radius: 20px;
        padding: 3rem 2.5rem;
        color: white;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }

    .capa-title {
        font-size: 2.8rem;
        font-weight: 700;
        letter-spacing: -0.04em;
        margin-bottom: 0.4rem;
        color: white;
    }

    .capa-subtitle {
        font-size: 1.15rem;
        font-weight: 400;
        opacity: 0.88;
        margin-bottom: 1.5rem;
        color: white;
    }

    .capa-badge {
        display: inline-block;
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.35);
        border-radius: 20px;
        padding: 0.3rem 1rem;
        font-size: 0.82rem;
        font-weight: 500;
        color: white;
        margin-right: 8px;
        margin-bottom: 8px;
    }

    .info-card {
        background: #ffffff;
        border: 1px solid var(--border);
        border-left: 4px solid var(--green);
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
    }

    .info-card-blue {
        background: #ffffff;
        border: 1px solid #b3d4ef;
        border-left: 4px solid var(--blue);
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
    }

    .step-flow {
        display: flex;
        align-items: flex-start;
        gap: 0px;
        margin: 1.5rem 0;
        flex-wrap: wrap;
    }

    .step-item {
        display: flex;
        flex-direction: column;
        align-items: center;
        min-width: 100px;
        flex: 1;
    }

    .step-circle {
        width: 48px;
        height: 48px;
        border-radius: 50%;
        background: var(--green-dark);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        font-family: 'Inter', sans-serif;
    }

    .step-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: var(--green-dark);
        text-align: center;
        font-family: 'Inter', sans-serif;
    }

    .step-desc {
        font-size: 0.72rem;
        color: var(--muted);
        text-align: center;
        margin-top: 2px;
        font-family: 'Inter', sans-serif;
    }

    .step-arrow {
        font-size: 1.4rem;
        color: var(--border);
        margin-top: 12px;
        padding: 0 4px;
        flex: 0;
    }

    .model-card {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        height: 100%;
    }

    .model-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: var(--green-dark);
        margin-bottom: 0.3rem;
    }

    .model-desc {
        font-size: 0.82rem;
        color: var(--muted);
        line-height: 1.5;
    }

    .small-muted { color: var(--muted); font-size: .88rem; }
    .green { color: var(--green); }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div class="main-header">
        <div class="main-title">LFDA Forecast</div>
        <div class="main-subtitle">Sistema de Previsão de Demanda</div>
        <div class="navline">Upload • Auditoria • Modelos • Forecast • Resultados&nbsp;&nbsp;&nbsp;&nbsp;<span class="status-dot"></span>Processamento local</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Helpers
# =========================================================

def strip_accents(text: str) -> str:
    text = str(text)
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def normalize_col(col: str) -> str:
    col = strip_accents(str(col)).lower().strip()
    col = re.sub(r"[^a-z0-9]+", "_", col)
    return col.strip("_")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(axis=1, how="all")
    df = df.loc[:, ~df.columns.astype(str).str.lower().str.contains(r"^unnamed")]
    df = df.dropna(axis=0, how="all")
    return df


def infer_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    norm_map = {col: normalize_col(col) for col in df.columns}
    product_keys = ["produto", "product", "item", "sku", "material", "codigo", "descricao", "cliente_produto"]
    date_keys = ["data", "date", "mes", "month", "periodo", "competencia", "ano_mes", "mes_ano"]
    qty_keys = ["quantidade", "quantity", "qty", "qtd", "demanda", "demand", "volume", "vendas", "sales"]
    product_col = next((c for c, n in norm_map.items() if any(k in n for k in product_keys)), None)
    date_col = next((c for c, n in norm_map.items() if any(k in n for k in date_keys)), None)
    qty_col = next((c for c, n in norm_map.items() if any(k in n for k in qty_keys)), None)
    if qty_col is None:
        numeric_scores = []
        for c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            numeric_scores.append((c, s.notna().sum()))
        if numeric_scores:
            qty_col = max(numeric_scores, key=lambda x: x[1])[0]
    return product_col, date_col, qty_col


MONTHS_PT = {
    "janeiro": 1, "jan": 1,
    "fevereiro": 2, "fev": 2,
    "marco": 3, "mar": 3, "março": 3,
    "abril": 4, "abr": 4,
    "maio": 5, "mai": 5,
    "junho": 6, "jun": 6,
    "julho": 7, "jul": 7,
    "agosto": 8, "ago": 8,
    "setembro": 9, "set": 9,
    "outubro": 10, "out": 10,
    "novembro": 11, "nov": 11,
    "dezembro": 12, "dez": 12,
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
        month = int(m.group(1))
        year = int(m.group(2))
        if 1 <= month <= 12:
            return pd.Timestamp(year=year, month=month, day=1)
    parts = raw.split()
    if len(parts) >= 2:
        month_word = parts[0]
        year_match = re.search(r"(20\d{2}|19\d{2})", raw)
        if month_word in MONTHS_PT and year_match:
            return pd.Timestamp(year=int(year_match.group(1)), month=MONTHS_PT[month_word], day=1)
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
            end_zero = i - 1
            left = start_zero - 1
            right = i
            if left >= 0 and right < n and y[left] > 0 and y[right] > 0:
                k = end_zero - start_zero + 1
                avg_extremes = (y[left] + y[right]) / 2.0
                per_zero = avg_extremes / k
                y[start_zero:i] = per_zero
                reduction_total = avg_extremes
                denom = y[left] + y[right]
                if denom > 0:
                    y[left] = max(0.0, y[left] - reduction_total * (y[left] / denom))
                    y[right] = max(0.0, y[right] - reduction_total * (y[right] / denom))
        else:
            i += 1
    return y


def split_train_test(y):
    n = len(y)
    if n >= 17:
        return y.iloc[:-4], y.iloc[-4:], "13+ treino / 4 teste"
    if 12 <= n <= 16:
        return y.iloc[:-4], y.iloc[-4:], "Últimos 4 períodos em teste"
    if 8 <= n <= 11:
        return y.iloc[:-3], y.iloc[-3:], "Últimos 3 períodos em teste"
    if 5 <= n <= 7:
        return y.iloc[:-2], y.iloc[-2:], "Últimos 2 períodos em teste"
    if 3 <= n <= 4:
        return y.iloc[:-1], y.iloc[-1:], "Último período em teste"
    return y, pd.Series(dtype=float), "Forecast exploratório sem teste"


def safe_mape(actual, pred):
    actual = np.array(actual, dtype=float)
    pred = np.array(pred, dtype=float)
    mask = actual != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100)


def agg_mape(actual, pred):
    actual_sum = float(np.sum(actual))
    pred_sum = float(np.sum(pred))
    if actual_sum == 0:
        return np.nan
    return abs(actual_sum - pred_sum) / abs(actual_sum) * 100


def model_params(fit):
    params = getattr(fit, "params", {}) or {}
    return {
        "Alpha": params.get("smoothing_level", np.nan),
        "Beta": params.get("smoothing_trend", np.nan),
        "Gamma": params.get("smoothing_seasonal", np.nan),
    }


def fit_forecast_model(model_name, train, steps):
    y = train.astype(float)
    if len(y) < 2:
        fc = np.repeat(y.iloc[-1] if len(y) else 0.0, steps)
        return None, pd.Series(fc)
    if model_name == "SES":
        fit = SimpleExpSmoothing(y, initialization_method="estimated").fit(optimized=True)
    elif model_name == "Holt":
        fit = Holt(y, initialization_method="estimated").fit(optimized=True)
    elif model_name == "Holt-Winters":
        if len(y) >= 8:
            fit = ExponentialSmoothing(
                y, trend="add", seasonal="add", seasonal_periods=4,
                initialization_method="estimated",
            ).fit(optimized=True)
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
    treated_frames = []
    summaries = []
    audits = []
    forecasts = []

    for prod, g in q.groupby("Produto"):
        g = g.sort_values("Trimestre").copy()
        original = g["Quantidade"].astype(float).values
        treated = treat_zeros_preserve_total(original)
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
                    actual_original = float(g.loc[g["Trimestre"] == dt, "Quantidade"].iloc[0])
                    actual_treated = float(test.iloc[idx])
                    pred = float(pred_test[idx]) if idx < len(pred_test) else np.nan
                    err = actual_treated - pred if pd.notna(pred) else np.nan
                    mape_i = abs(err / actual_treated) * 100 if actual_treated != 0 and pd.notna(err) else np.nan
                    audits.append({
                        "Produto": prod, "Modelo": model_name, "Tipo": "Teste",
                        "Trimestre": quarter_label(dt), "Valor Real": actual_original,
                        "Valor Tratado": actual_treated, "Valor Previsto": pred,
                        "Erro": err, "MAPE %": mape_i,
                        "Alpha": params["Alpha"], "Beta": params["Beta"], "Gamma": params["Gamma"],
                        "Regra Treino/Teste": split_rule,
                    })

                summaries.append({
                    "Produto": prod, "Modelo": model_name, "Períodos": len(y),
                    "Treino": len(train), "Teste": len(test), "Regra": split_rule,
                    "MAPE Período Médio %": period_mape, "MAPE Agregado %": aggregate_mape,
                    "Alpha": params["Alpha"], "Beta": params["Beta"], "Gamma": params["Gamma"],
                })

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
                    pred_val = max(0.0, float(fc[h - 1]))
                    interval = z90 * residual_std * np.sqrt(h)
                    forecasts.append({
                        "Produto": prod, "Modelo": model_name, "Trimestre": quarter_label(dt),
                        "Forecast": pred_val, "IC 90% Inferior": max(0.0, pred_val - interval),
                        "IC 90% Superior": pred_val + interval,
                        "Alpha": params_full["Alpha"], "Beta": params_full["Beta"], "Gamma": params_full["Gamma"],
                    })

            except Exception as exc:
                summaries.append({
                    "Produto": prod, "Modelo": model_name, "Períodos": len(y),
                    "Treino": len(train), "Teste": len(test), "Regra": split_rule,
                    "MAPE Período Médio %": np.nan, "MAPE Agregado %": np.nan,
                    "Alpha": np.nan, "Beta": np.nan, "Gamma": np.nan, "Erro Modelo": str(exc),
                })

    treated_df = pd.concat(treated_frames, ignore_index=True) if treated_frames else pd.DataFrame()
    summary_df = pd.DataFrame(summaries)
    audit_df = pd.DataFrame(audits)
    forecast_df = pd.DataFrame(forecasts)

    if not forecast_df.empty:
        annual_upper = forecast_df.groupby(["Produto", "Modelo"], as_index=False)["IC 90% Superior"].sum()
        annual_upper = annual_upper.rename(columns={"IC 90% Superior": "Soma 4T - Limite Superior"})
        forecast_df = forecast_df.merge(annual_upper, on=["Produto", "Modelo"], how="left")

    return treated_df, summary_df, audit_df, forecast_df


def to_excel_bytes(sheets):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for name, df in sheets.items():
            safe_name = name[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)
            workbook = writer.book
            worksheet = writer.sheets[safe_name]
            header_fmt = workbook.add_format({
                "bold": True, "font_color": "#FFFFFF",
                "bg_color": "#1a6b2f", "border": 1,
            })
            num_fmt = workbook.add_format({"num_format": "#,##0.00"})
            for col_num, col_name in enumerate(df.columns):
                worksheet.write(0, col_num, col_name, header_fmt)
                width = max(12, min(32, len(str(col_name)) + 4))
                worksheet.set_column(col_num, col_num, width)
                if pd.api.types.is_numeric_dtype(df[col_name]):
                    worksheet.set_column(col_num, col_num, width, num_fmt)
            worksheet.freeze_panes(1, 0)
    return output.getvalue()


# =========================================================
# Sidebar
# =========================================================
st.sidebar.markdown("### 🌱 LFDA Forecast")
st.sidebar.markdown("<span class='small-muted'>Ambiente local • Streamlit</span>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Menu",
    ["Início", "Upload", "Validação", "Modelos", "Auditoria", "Forecast", "Exportação"],
    index=0,
)

st.sidebar.divider()
st.sidebar.markdown("**Configuração atual**")
st.sidebar.caption("Modelos: SES, Holt e Holt-Winters")
st.sidebar.caption("Parâmetros: otimização automática Python")
st.sidebar.caption("Forecast: próximos 4 trimestres")
st.sidebar.caption("Intervalo de confiança: 90%")


# =========================================================
# Session State
# =========================================================
for key in ["raw_df", "monthly", "quarterly", "treated", "summary", "audit", "forecast", "inferred_cols"]:
    if key not in st.session_state:
        st.session_state[key] = None


def process_file(uploaded_file):
    if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
        raw = pd.read_excel(uploaded_file)
    else:
        raw = pd.read_csv(uploaded_file, sep=None, engine="python")
    raw = clean_dataframe(raw)
    product_col, date_col, qty_col = infer_columns(raw)
    if product_col is None or date_col is None or qty_col is None:
        raise ValueError("Não foi possível identificar automaticamente as colunas de Produto, Data e Quantidade.")
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
# Pages
# =========================================================

# ----------------------------------------------------------
# INÍCIO — Capa
# ----------------------------------------------------------
if page == "Início":
    st.markdown("""
    <div class="capa-hero">
        <div class="capa-title">🌱 LFDA Forecast</div>
        <div class="capa-subtitle">Sistema de Previsão de Demanda — LFDA-RS</div>
        <span class="capa-badge">📊 Modelos estatísticos</span>
        <span class="capa-badge">🔍 Auditoria completa</span>
        <span class="capa-badge">📁 Exportação Excel</span>
        <span class="capa-badge">🔒 Processamento local</span>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("#### Sobre o sistema")
        st.markdown("""
        <div class="info-card">
            <strong>O que é o LFDA Forecast?</strong><br><br>
            Uma ferramenta desenvolvida internamente pela LFDA-RS para gerar previsões de demanda
            de produtos agropecuários com base em histórico mensal. O sistema automatiza todo o processo
            estatístico — da leitura da planilha até a exportação dos resultados — sem exigir conhecimento
            de programação do usuário.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card-blue">
            <strong>Como usar?</strong><br><br>
            Basta subir uma planilha Excel ou CSV com três colunas: <strong>Produto</strong>, <strong>Data</strong>
            e <strong>Quantidade</strong>. O sistema faz todo o restante automaticamente.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### Modelos disponíveis")
        st.markdown("""
        <div class="model-card" style="margin-bottom:10px">
            <div class="model-title">SES — Simple Exponential Smoothing</div>
            <div class="model-desc">Ideal para séries sem tendência clara. Usa apenas o nível histórico para projetar o futuro.</div>
        </div>
        <div class="model-card" style="margin-bottom:10px">
            <div class="model-title">Holt — Double Exponential Smoothing</div>
            <div class="model-desc">Captura tendência de crescimento ou queda. Indicado quando a demanda tem direção definida.</div>
        </div>
        <div class="model-card">
            <div class="model-title">Holt-Winters — Triple Exponential Smoothing</div>
            <div class="model-desc">Captura tendência e sazonalidade trimestral. Mais completo para produtos com variação sazonal.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Como o sistema funciona — passo a passo")

    st.markdown("""
    <div class="step-flow">
        <div class="step-item">
            <div class="step-circle">1</div>
            <div class="step-label">Upload</div>
            <div class="step-desc">Envie o Excel ou CSV com Produto, Data e Quantidade</div>
        </div>
        <div class="step-arrow">→</div>
        <div class="step-item">
            <div class="step-circle">2</div>
            <div class="step-label">Validação</div>
            <div class="step-desc">O sistema identifica colunas e converte datas automaticamente</div>
        </div>
        <div class="step-arrow">→</div>
        <div class="step-item">
            <div class="step-circle">3</div>
            <div class="step-label">Trimestral</div>
            <div class="step-desc">Converte dados mensais em série trimestral internamente</div>
        </div>
        <div class="step-arrow">→</div>
        <div class="step-item">
            <div class="step-circle">4</div>
            <div class="step-label">Zeros</div>
            <div class="step-desc">Trata zeros entre períodos preservando a demanda total</div>
        </div>
        <div class="step-arrow">→</div>
        <div class="step-item">
            <div class="step-circle">5</div>
            <div class="step-label">Modelos</div>
            <div class="step-desc">Roda SES, Holt e Holt-Winters com split adaptativo</div>
        </div>
        <div class="step-arrow">→</div>
        <div class="step-item">
            <div class="step-circle">6</div>
            <div class="step-label">Auditoria</div>
            <div class="step-desc">Calcula MAPE por período e agregado para cada modelo</div>
        </div>
        <div class="step-arrow">→</div>
        <div class="step-item">
            <div class="step-circle">7</div>
            <div class="step-label">Forecast</div>
            <div class="step-desc">Gera previsão dos próximos 4 trimestres com IC 90%</div>
        </div>
        <div class="step-arrow">→</div>
        <div class="step-item">
            <div class="step-circle">8</div>
            <div class="step-label">Exportação</div>
            <div class="step-desc">Baixe tudo em Excel formatado com todas as abas</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Formato esperado da planilha")
    st.dataframe(pd.DataFrame({
        "Produto": ["Metanol", "Acetonitrila", "Tubo Falcon 15ml"],
        "Data": ["março de 2026", "março de 2026", "março de 2026"],
        "Quantidade": [5000, 16000, 1000],
    }), use_container_width=True)
    st.markdown("<div class='small-muted'>A base pode ser mensal. O sistema converte para trimestral automaticamente. Datas em português são aceitas.</div>", unsafe_allow_html=True)


# ----------------------------------------------------------
# UPLOAD
# ----------------------------------------------------------
elif page == "Upload":
    st.subheader("Upload da base")
    st.markdown("<div class='small-muted'>Envie uma base mensal em Excel ou CSV. O sistema identifica Produto, Data e Quantidade automaticamente.</div>", unsafe_allow_html=True)
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
        st.dataframe(pd.DataFrame({
            "Campo": ["Produto", "Data", "Quantidade"],
            "Coluna identificada": [product_col, date_col, qty_col],
        }), use_container_width=True)

        st.markdown("### Prévia da base original")
        st.dataframe(st.session_state.raw_df.head(30), use_container_width=True)


# ----------------------------------------------------------
# VALIDAÇÃO
# ----------------------------------------------------------
elif page == "Validação":
    st.subheader("Validação e conversão mensal → trimestral")
    if st.session_state.monthly is None:
        st.info("Faça o upload de uma base para visualizar a validação.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Registros mensais tratados", len(st.session_state.monthly))
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


# ----------------------------------------------------------
# MODELOS
# ----------------------------------------------------------
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


# ----------------------------------------------------------
# AUDITORIA
# ----------------------------------------------------------
elif page == "Auditoria":
    st.subheader("Auditoria estatística")
    if st.session_state.audit is None or st.session_state.audit.empty:
        st.info("Faça o upload de uma base para gerar a auditoria.")
    else:
        audit = st.session_state.audit.copy()
        produtos = ["Todos"] + sorted(audit["Produto"].unique().tolist())
        modelos = ["Todos"] + sorted(audit["Modelo"].unique().tolist())
        c1, c2 = st.columns(2)
        prod_filter = c1.selectbox("Produto", produtos)
        model_filter = c2.selectbox("Modelo", modelos)
        if prod_filter != "Todos":
            audit = audit[audit["Produto"] == prod_filter]
        if model_filter != "Todos":
            audit = audit[audit["Modelo"] == model_filter]
        st.dataframe(audit, use_container_width=True)


# ----------------------------------------------------------
# FORECAST
# ----------------------------------------------------------
elif page == "Forecast":
    st.subheader("Forecast — próximos 4 trimestres")
    if st.session_state.forecast is None or st.session_state.forecast.empty:
        st.info("Faça o upload de uma base para gerar o forecast.")
    else:
        fc = st.session_state.forecast.copy()
        produtos = ["Todos"] + sorted(fc["Produto"].unique().tolist())
        modelos = ["Todos"] + sorted(fc["Modelo"].unique().tolist())
        c1, c2 = st.columns(2)
        prod_filter = c1.selectbox("Produto", produtos)
        model_filter = c2.selectbox("Modelo", modelos)
        if prod_filter != "Todos":
            fc = fc[fc["Produto"] == prod_filter]
        if model_filter != "Todos":
            fc = fc[fc["Modelo"] == model_filter]

        st.dataframe(fc, use_container_width=True)

        st.markdown("### Soma anual do limite superior por produto e modelo")
        annual = fc.groupby(["Produto", "Modelo"], as_index=False)["IC 90% Superior"].sum()
        annual = annual.rename(columns={"IC 90% Superior": "Soma 4T - Limite Superior"})
        st.dataframe(annual, use_container_width=True)


# ----------------------------------------------------------
# EXPORTAÇÃO
# ----------------------------------------------------------
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
        st.markdown("<div class='small-muted'>O arquivo contém base original, base mensal, trimestral, tratamento dos zeros, MAPE, auditoria e forecast.</div>", unsafe_allow_html=True)
