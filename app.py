import re
import io
import unicodedata
from typing import Dict, List, Optional, Tuple

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing


st.set_page_config(page_title="LFDA Forecast", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

# Cores de marca usadas também nos gráficos
C_GREEN = "#1a6b2f"
C_GREEN_L = "#38b249"
C_BLUE = "#1e7ec8"
C_AMBER = "#d4a017"
C_GREY = "#9aa8b5"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
        --green: #2e8b57; --green-dark: #1a6b2f; --green-light: #e8f3ec;
        --blue: #1e6fb8; --blue-dark: #155290; --blue-light: #e8f0f8;
        --border: #dce3ea; --border-strong: #c5d0db;
        --muted: #687989; --text: #1f2d3a; --bg: #f5f7fa; --panel: #ffffff;
    }
    * { font-family: 'Inter', sans-serif !important; }
    .stApp { background: var(--bg); color: var(--text); }
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1300px; }
    [data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--border-strong); }
    [data-testid="stSidebar"] * { font-family: 'Inter', sans-serif !important; }
    .status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--green); margin-right: 6px; }
    div[data-testid="stMetric"] { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: .85rem 1rem; box-shadow: none; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: var(--text); }
    div[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 8px; }
    .stDownloadButton>button { border-radius: 6px; border: 1px solid var(--green-dark); background: var(--green-dark); color: #fff; font-weight: 600; }
    .stDownloadButton>button:hover { background: var(--green); border-color: var(--green); }
    h1, h2, h3 { font-family: 'Inter', sans-serif !important; letter-spacing: -0.02em; color: var(--text); }

    .sb-logo { padding: 1.1rem 0.6rem 0.8rem 0.6rem; border-bottom: 1px solid var(--border); margin-bottom: 0.5rem; }
    .sb-logo-row { display: flex; align-items: center; gap: 10px; }
    .sb-logo-mark { width: 30px; height: 30px; border-radius: 7px; background: linear-gradient(135deg, #1a6b2f 0%, #1e6fb8 100%); flex-shrink: 0; }
    .sb-logo-title { font-size: 1.05rem; font-weight: 700; color: var(--text); letter-spacing: -0.02em; }
    .sb-logo-sub { font-size: 0.72rem; color: var(--muted); margin-top: 6px; }
    .sb-group-label { font-size: 0.66rem; font-weight: 700; letter-spacing: 0.12em; color: #9aa8b5; text-transform: uppercase; padding: 0.9rem 0.6rem 0.4rem 0.6rem; display: block; }

    [data-testid="stSidebar"] .stButton > button {
        width: 100%; text-align: left; justify-content: flex-start;
        background: transparent; border: none; color: var(--text);
        font-size: 0.92rem; font-weight: 500; padding: 0.55rem 0.7rem;
        border-radius: 7px; border-left: 3px solid transparent; margin: 1px 0;
        box-shadow: none; transition: background 0.12s;
    }
    [data-testid="stSidebar"] .stButton > button:hover { background: #f0f3f7; color: var(--text); }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: var(--green-light); color: var(--green-dark); font-weight: 600;
        border-left: 3px solid var(--green-dark);
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover { background: #ddeee3; }

    .sb-config { margin: 14px 4px 0 4px; padding: 11px 12px; background: #f7f9fb; border: 1px solid var(--border); border-radius: 8px; font-size: 0.74rem; color: var(--muted); line-height: 1.8; }
    .sb-config-title { font-weight: 700; color: var(--text); font-size: 0.76rem; margin-bottom: 4px; }

    .topbar { display: flex; align-items: center; justify-content: space-between; padding: 0.6rem 1.1rem; margin-bottom: 1.3rem; background: var(--panel); border: 1px solid var(--border); border-radius: 8px; }
    .topbar-left { display: flex; align-items: center; gap: 12px; }
    .topbar-mark { display: inline-block; width: 26px; height: 26px; border-radius: 6px; background: linear-gradient(135deg, #1a6b2f 0%, #1e6fb8 100%); }
    .topbar-brand { font-size: 1.05rem; font-weight: 700; color: var(--text); letter-spacing: -0.02em; }
    .topbar-div { color: var(--border-strong); font-weight: 300; }
    .topbar-sub { font-size: 0.9rem; color: var(--muted); font-weight: 400; }
    .topbar-right { font-size: 0.8rem; color: var(--muted); display: flex; align-items: center; }

    .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 1.4rem 1.6rem; margin-bottom: 1.1rem; }
    .panel-title { font-size: 1.05rem; font-weight: 700; color: var(--text); margin-bottom: 0.3rem; }

    .page-title { font-size: 1.35rem; font-weight: 700; color: var(--text); letter-spacing: -0.02em; margin-bottom: 0.2rem; }
    .page-sub { font-size: 0.9rem; color: var(--muted); margin-bottom: 1.2rem; }
    .section-title { font-size: 1.05rem; font-weight: 700; color: var(--text); margin: 1.4rem 0 0.7rem 0; padding-bottom: 0.4rem; border-bottom: 2px solid var(--green); display: inline-block; }

    .capa-head { background: var(--panel); border: 1px solid var(--border); border-left: 4px solid var(--green); border-radius: 8px; padding: 1.8rem 2rem; margin-bottom: 1.2rem; }
    .capa-title { font-size: 1.9rem; font-weight: 700; letter-spacing: -0.03em; color: var(--text); margin-bottom: 0.3rem; }
    .capa-subtitle { font-size: 1.05rem; font-weight: 500; color: var(--green-dark); margin-bottom: 0.3rem; }
    .capa-inst { font-size: 0.88rem; color: var(--muted); margin-bottom: 1rem; }
    .capa-context { font-size: 0.92rem; color: var(--text); line-height: 1.7; max-width: 880px; }
    .capa-prod { display: inline-block; background: var(--green-light); border: 1px solid #cfe3d6; border-radius: 5px; padding: 0.2rem 0.6rem; font-size: 0.8rem; font-weight: 600; color: var(--green-dark); margin: 5px 5px 0 0; }

    .info-card { background: var(--panel); border: 1px solid var(--border); border-left: 3px solid var(--green); border-radius: 7px; padding: 1.1rem 1.3rem; margin-bottom: 0.9rem; color: var(--text); font-size: 0.9rem; line-height: 1.6; }
    .info-card-blue { background: var(--panel); border: 1px solid var(--border); border-left: 3px solid var(--blue); border-radius: 7px; padding: 1.1rem 1.3rem; margin-bottom: 0.9rem; color: var(--text); font-size: 0.9rem; line-height: 1.6; }
    .info-card-yellow { background: #fffdf6; border: 1px solid #ece2c0; border-left: 3px solid #c79a1e; border-radius: 7px; padding: 1.1rem 1.3rem; margin-bottom: 0.9rem; color: var(--text); font-size: 0.9rem; line-height: 1.6; }

    .model-card { background: var(--panel); border: 1px solid var(--border); border-radius: 7px; padding: 1.1rem 1.3rem; margin-bottom: 10px; height: 100%; }
    .model-title { font-size: 0.92rem; font-weight: 700; color: var(--text); margin-bottom: 0.4rem; }
    .model-desc { font-size: 0.82rem; color: var(--muted); line-height: 1.6; }
    .model-when { font-size: 0.78rem; color: var(--blue-dark); margin-top: 0.4rem; font-weight: 500; }

    .step-flow { display: flex; align-items: flex-start; gap: 0; margin: 1.2rem 0; flex-wrap: wrap; }
    .step-item { display: flex; flex-direction: column; align-items: center; min-width: 90px; flex: 1; }
    .step-circle { width: 42px; height: 42px; border-radius: 8px; background: var(--green-dark); color: white; display: flex; align-items: center; justify-content: center; font-size: 1rem; font-weight: 700; margin-bottom: 0.5rem; }
    .step-label { font-size: 0.76rem; font-weight: 700; color: var(--text); text-align: center; }
    .step-desc { font-size: 0.7rem; color: var(--muted); text-align: center; margin-top: 3px; line-height: 1.4; }
    .step-arrow { font-size: 1.2rem; color: var(--border-strong); margin-top: 10px; padding: 0 2px; flex: 0; }

    .small-muted { color: var(--muted); font-size: .88rem; }

    .exec-card { background: var(--panel); border: 1px solid var(--border); border-top: 3px solid var(--green); border-radius: 8px; padding: 1.2rem 1.4rem; margin-bottom: 12px; }
    .exec-prod { font-size: 1.08rem; font-weight: 700; color: var(--text); margin-bottom: 0.7rem; padding-bottom: 0.5rem; border-bottom: 1px solid var(--border); }
    .exec-row { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.35rem; }
    .exec-label { font-size: 0.82rem; color: var(--muted); }
    .exec-value { font-size: 1.05rem; font-weight: 700; color: var(--text); }
    .exec-model { display: inline-block; background: var(--green-light); color: var(--green-dark); font-size: 0.78rem; font-weight: 600; padding: 0.2rem 0.7rem; border-radius: 5px; }

    .mape-badge { display: inline-block; font-size: 0.72rem; font-weight: 600; padding: 0.15rem 0.6rem; border-radius: 5px; }
    .mape-exc { background: #d6f0de; color: #137a36; }
    .mape-bom { background: #faf0d2; color: #8a6200; }
    .mape-att { background: #f8dcdc; color: #b03030; }
    .gran-badge { display: inline-block; font-size: 0.72rem; font-weight: 600; padding: 0.15rem 0.6rem; border-radius: 5px; }
    .gran-mensal { background: var(--blue-light); color: var(--blue-dark); }
    .gran-trim { background: #efe6f7; color: #6a3da8; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="topbar">
    <div class="topbar-left">
        <span class="topbar-mark"></span>
        <span class="topbar-brand">LFDA&nbsp;Forecast</span>
        <span class="topbar-div">|</span>
        <span class="topbar-sub">Sistema de Previsão de Demanda</span>
    </div>
    <div class="topbar-right">
        <span class="status-dot"></span>Processamento local
    </div>
</div>
""", unsafe_allow_html=True)


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

MONTHS_PT = {"janeiro": 1, "jan": 1, "fevereiro": 2, "fev": 2, "marco": 3, "mar": 3, "março": 3,
             "abril": 4, "abr": 4, "maio": 5, "mai": 5, "junho": 6, "jun": 6, "julho": 7, "jul": 7,
             "agosto": 8, "ago": 8, "setembro": 9, "set": 9, "outubro": 10, "out": 10,
             "novembro": 11, "nov": 11, "dezembro": 12, "dez": 12}

def parse_month_date(value):
    if pd.isna(value): return pd.NaT
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.to_datetime(value).to_period("M").to_timestamp()
    raw = strip_accents(str(value).strip().lower())
    raw = raw.replace(" de ", " ").replace("/", " ").replace("-", " ").replace(".", " ")
    raw = re.sub(r"\s+", " ", raw).strip()
    m = re.match(r"^(\d{1,2})\s+(\d{4})$", raw)
    if m:
        month, year = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12:
            return pd.Timestamp(year=year, month=month, day=1)
    parts = raw.split()
    year_match = re.search(r"(20\d{2}|19\d{2})", raw)
    if len(parts) >= 2 and parts[0] in MONTHS_PT and year_match:
        return pd.Timestamp(year=int(year_match.group(1)), month=MONTHS_PT[parts[0]], day=1)
    parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
    return parsed.to_period("M").to_timestamp() if pd.notna(parsed) else pd.NaT

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
    q["Periodo"] = q["Data"].dt.to_period("Q").dt.to_timestamp()
    q = q.groupby(["Produto", "Periodo"], as_index=False)["Quantidade"].sum()
    return q.sort_values(["Produto", "Periodo"])

def aggregate_monthly(monthly):
    m = monthly.copy()
    m["Periodo"] = m["Data"].dt.to_period("M").dt.to_timestamp()
    m = m.groupby(["Produto", "Periodo"], as_index=False)["Quantidade"].sum()
    return m.sort_values(["Produto", "Periodo"])

def fill_missing_periods(pdf, freq):
    """Preenche períodos faltantes com zero. freq: 'Q' ou 'M'."""
    frames = []
    for prod, g in pdf.groupby("Produto"):
        g = g.sort_values("Periodo")
        full_idx = pd.period_range(g["Periodo"].min(), g["Periodo"].max(), freq=freq).to_timestamp()
        aux = pd.DataFrame({"Periodo": full_idx})
        aux["Produto"] = prod
        aux = aux.merge(g, on=["Produto", "Periodo"], how="left")
        aux["Quantidade"] = aux["Quantidade"].fillna(0.0)
        frames.append(aux)
    return pd.concat(frames, ignore_index=True) if frames else pdf

def treat_zeros_preserve_total(values):
    y = np.array(values, dtype=float).copy()
    n = len(y); i = 0
    while i < n:
        if y[i] == 0:
            start_zero = i
            while i < n and y[i] == 0:
                i += 1
            left, right = start_zero - 1, i
            if left >= 0 and right < n and y[left] > 0 and y[right] > 0:
                k = i - start_zero
                avg = (y[left] + y[right]) / 2.0
                y[start_zero:i] = avg / k
                denom = y[left] + y[right]
                if denom > 0:
                    y[left] = max(0.0, y[left] - avg * (y[left] / denom))
                    y[right] = max(0.0, y[right] - avg * (y[right] / denom))
        else:
            i += 1
    return y

def split_train_test(y, horizon):
    """Split adaptativo. horizon = nº de períodos do teste alvo (4 trim ou 12 meses)."""
    n = len(y)
    h = horizon
    # mínimo de teste razoável conforme histórico
    if n >= 4 * h + 1:           # histórico farto
        return y.iloc[:-h], y.iloc[-h:], f"{n-h} treino / {h} teste"
    if n >= 3 * h:
        return y.iloc[:-h], y.iloc[-h:], f"Últimos {h} períodos em teste"
    test_n = max(1, n // 4)
    if n >= test_n + 2:
        return y.iloc[:-test_n], y.iloc[-test_n:], f"Últimos {test_n} períodos em teste"
    if n >= 3:
        return y.iloc[:-1], y.iloc[-1:], "Último período em teste"
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
    return {"Alpha": params.get("smoothing_level", np.nan), "Beta": params.get("smoothing_trend", np.nan), "Gamma": params.get("smoothing_seasonal", np.nan)}

def fit_forecast_model(model_name, train, steps, seasonal_periods=None):
    """Ajusta o modelo. Para Holt-Winters, seasonal_periods define a sazonalidade."""
    y = train.astype(float)
    if len(y) < 2:
        return None, pd.Series(np.repeat(y.iloc[-1] if len(y) else 0.0, steps))
    if model_name == "SES":
        fit = SimpleExpSmoothing(y, initialization_method="estimated").fit(optimized=True)
    elif model_name == "Holt":
        fit = Holt(y, initialization_method="estimated").fit(optimized=True)
    elif model_name == "Holt-Winters":
        sp = seasonal_periods or 4
        if len(y) >= 2 * sp:
            fit = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=sp,
                                       initialization_method="estimated").fit(optimized=True)
        else:
            fit = Holt(y, initialization_method="estimated").fit(optimized=True)
    else:
        raise ValueError("Modelo inválido")
    return fit, fit.forecast(steps)

def future_periods(last_period, periods, freq):
    start = pd.Period(last_period, freq=freq) + 1
    return list(pd.period_range(start, periods=periods, freq=freq).to_timestamp())

def quarter_label(ts):
    p = pd.Period(ts, freq="Q")
    return f"{p.year}T{p.quarter}"

def month_label(ts):
    ts = pd.Timestamp(ts)
    return f"{ts.year}-{ts.month:02d}"

def fmt_br(value, decimals=0):
    """Formata número no padrão brasileiro: 66939.21 -> 66.939,21"""
    if pd.isna(value): return "—"
    s = f"{value:,.{decimals}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")

def mape_badge_html(mape):
    if pd.isna(mape):
        return '<span class="mape-badge mape-att">Sem teste</span>'
    if mape <= 15:
        return f'<span class="mape-badge mape-exc">Excelente · {mape:.1f}%</span>'
    if mape <= 30:
        return f'<span class="mape-badge mape-bom">Bom · {mape:.1f}%</span>'
    return f'<span class="mape-badge mape-att">Atenção · {mape:.1f}%</span>'

def gran_badge_html(gran):
    cls = "gran-mensal" if gran == "Mensal" else "gran-trim"
    return f'<span class="gran-badge {cls}">{gran}</span>'


# Configurações de cada granularidade testada
# label_func: como rotular o período; freq; horizonte (1 ano); variantes de sazonalidade do HW
GRAN_CONFIGS = {
    "Mensal":     {"freq": "M", "horizon": 12, "label": month_label,   "hw_seasons": [12, 4]},
    "Trimestral": {"freq": "Q", "horizon": 4,  "label": quarter_label, "hw_seasons": [4]},
}


def _run_one_series(prod, periodo_df, gran_name, audits, summaries, forecasts):
    """Roda os 3 modelos para um produto numa granularidade. Preenche audits/summaries/forecasts."""
    cfg = GRAN_CONFIGS[gran_name]
    freq, horizon, label_func = cfg["freq"], cfg["horizon"], cfg["label"]

    g = periodo_df.sort_values("Periodo").copy()
    treated = treat_zeros_preserve_total(g["Quantidade"].astype(float).values)
    g["Valor Tratado"] = treated
    y = pd.Series(treated, index=g["Periodo"])
    train, test, split_rule = split_train_test(y, horizon)
    test_steps = len(test)

    # define as variantes de modelo a rodar (HW pode ter mais de uma sazonalidade)
    variants = [("SES", None), ("Holt", None)]
    for sp in cfg["hw_seasons"]:
        variants.append(("Holt-Winters", sp))

    for model_name, sp in variants:
        # rótulo do modelo distinguindo sazonalidade do HW
        if model_name == "Holt-Winters":
            disp_model = f"Holt-Winters (s={sp})"
        else:
            disp_model = model_name
        try:
            fit_bt, pred_test = fit_forecast_model(model_name, train, max(test_steps, 1), seasonal_periods=sp)
            pred_test = pd.Series(pred_test).iloc[:test_steps].values if test_steps > 0 else np.array([])
            period_mape = safe_mape(test.values, pred_test) if test_steps else np.nan
            aggregate_mape = agg_mape(test.values, pred_test) if test_steps else np.nan
            params = model_params(fit_bt) if fit_bt is not None else {"Alpha": np.nan, "Beta": np.nan, "Gamma": np.nan}

            for idx, dt in enumerate(test.index):
                actual_o = float(g.loc[g["Periodo"] == dt, "Quantidade"].iloc[0])
                actual_t = float(test.iloc[idx])
                pred = float(pred_test[idx]) if idx < len(pred_test) else np.nan
                err = actual_t - pred if pd.notna(pred) else np.nan
                mape_i = abs(err / actual_t) * 100 if actual_t != 0 and pd.notna(err) else np.nan
                audits.append({"Produto": prod, "Granularidade": gran_name, "Modelo": disp_model, "Tipo": "Teste",
                               "Período": label_func(dt), "Valor Real": actual_o, "Valor Tratado": actual_t,
                               "Valor Previsto": pred, "Erro": err, "MAPE %": mape_i,
                               "Alpha": params["Alpha"], "Beta": params["Beta"], "Gamma": params["Gamma"],
                               "Regra Treino/Teste": split_rule})

            summaries.append({"Produto": prod, "Granularidade": gran_name, "Modelo": disp_model,
                               "Períodos": len(y), "Treino": len(train), "Teste": len(test), "Regra": split_rule,
                               "MAPE Período Médio %": period_mape, "MAPE Agregado %": aggregate_mape,
                               "Alpha": params["Alpha"], "Beta": params["Beta"], "Gamma": params["Gamma"]})

            # forecast final com série completa
            fit_full, fc = fit_forecast_model(model_name, y, horizon, seasonal_periods=sp)
            fc = pd.Series(fc).astype(float).values
            fitted_vals = None
            if fit_full is not None:
                fv = np.array(getattr(fit_full, "fittedvalues", []), dtype=float)
                if len(fv) == len(y):
                    fitted_vals = fv
            residual_std = np.nan
            if fit_bt is not None and test_steps > 1:
                residual_std = float(np.nanstd(test.values - pred_test, ddof=1))
            elif fitted_vals is not None and len(y) > 2:
                residual_std = float(np.nanstd(y.values - fitted_vals, ddof=1))
            if pd.isna(residual_std) or residual_std == 0:
                residual_std = float(np.nanstd(y.values, ddof=1)) if len(y) > 1 else 0.0
            z90 = 1.645
            future = future_periods(g["Periodo"].max(), horizon, freq)
            params_full = model_params(fit_full) if fit_full is not None else params
            for h, dt in enumerate(future, start=1):
                pv = max(0.0, float(fc[h - 1]))
                interval = z90 * residual_std * np.sqrt(h)
                forecasts.append({"Produto": prod, "Granularidade": gran_name, "Modelo": disp_model,
                                  "Período": label_func(dt), "PeriodoData": dt, "Forecast": pv,
                                  "IC 90% Inferior": max(0.0, pv - interval), "IC 90% Superior": pv + interval,
                                  "Alpha": params_full["Alpha"], "Beta": params_full["Beta"], "Gamma": params_full["Gamma"]})
        except Exception as exc:
            summaries.append({"Produto": prod, "Granularidade": gran_name, "Modelo": disp_model,
                               "Períodos": len(y), "Treino": len(train), "Teste": len(test), "Regra": split_rule,
                               "MAPE Período Médio %": np.nan, "MAPE Agregado %": np.nan,
                               "Alpha": np.nan, "Beta": np.nan, "Gamma": np.nan, "Erro Modelo": str(exc)})

    return g  # série tratada desta granularidade


def run_models(monthly_clean):
    """Roda todos os produtos em Mensal e Trimestral, todos os modelos."""
    treated_m, treated_q = [], []
    summaries, audits, forecasts = [], [], []

    # bases por granularidade
    base_q = fill_missing_periods(monthly_to_quarterly(monthly_clean), "Q")
    base_m = fill_missing_periods(aggregate_monthly(monthly_clean), "M")

    for prod in sorted(monthly_clean["Produto"].unique()):
        gq = base_q[base_q["Produto"] == prod]
        gm = base_m[base_m["Produto"] == prod]
        if not gq.empty:
            tq = _run_one_series(prod, gq, "Trimestral", audits, summaries, forecasts)
            treated_q.append(tq)
        if not gm.empty:
            tm = _run_one_series(prod, gm, "Mensal", audits, summaries, forecasts)
            treated_m.append(tm)

    treated_q_df = pd.concat(treated_q, ignore_index=True) if treated_q else pd.DataFrame()
    treated_m_df = pd.concat(treated_m, ignore_index=True) if treated_m else pd.DataFrame()
    summary_df = pd.DataFrame(summaries)
    audit_df = pd.DataFrame(audits)
    forecast_df = pd.DataFrame(forecasts)

    if not forecast_df.empty:
        annual = forecast_df.groupby(["Produto", "Granularidade", "Modelo"], as_index=False)["IC 90% Superior"].sum()
        annual = annual.rename(columns={"IC 90% Superior": "Soma Anual - Limite Superior"})
        forecast_df = forecast_df.merge(annual, on=["Produto", "Granularidade", "Modelo"], how="left")

    return treated_q_df, treated_m_df, summary_df, audit_df, forecast_df


def best_combo_for(summary, prod):
    """Retorna (granularidade, modelo, mape) da melhor combinação para o produto."""
    sub = summary[summary["Produto"] == prod].copy()
    sub = sub.dropna(subset=["MAPE Agregado %"])
    if sub.empty:
        # sem teste: pega qualquer um, preferindo trimestral
        any_sub = summary[summary["Produto"] == prod]
        if any_sub.empty:
            return None, None, np.nan
        row = any_sub.iloc[0]
        return row["Granularidade"], row["Modelo"], np.nan
    sub = sub.sort_values("MAPE Agregado %")
    row = sub.iloc[0]
    return row["Granularidade"], row["Modelo"], row["MAPE Agregado %"]

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
# Gráficos
# =========================================================
def _treated_for(prod, gran, treated_q_df, treated_m_df):
    src = treated_m_df if gran == "Mensal" else treated_q_df
    label_func = GRAN_CONFIGS[gran]["label"]
    h = src[src["Produto"] == prod].sort_values("Periodo")
    return h, label_func

def plot_history_forecast(prod, gran, model_name, treated_q_df, treated_m_df, forecast_df):
    """Linha do tempo: histórico tratado + forecast do modelo/granularidade escolhidos + faixa IC90."""
    hist, label_func = _treated_for(prod, gran, treated_q_df, treated_m_df)
    fc = forecast_df[(forecast_df["Produto"] == prod) & (forecast_df["Granularidade"] == gran) &
                     (forecast_df["Modelo"] == model_name)].sort_values("PeriodoData")
    fig = go.Figure()
    if hist.empty:
        return fig
    fig.add_trace(go.Scatter(x=[label_func(t) for t in hist["Periodo"]], y=hist["Valor Tratado"],
                             mode="lines+markers", name="Histórico", line=dict(color=C_GREEN, width=3), marker=dict(size=6)))
    if not fc.empty:
        last_x = label_func(hist["Periodo"].iloc[-1]); last_y = hist["Valor Tratado"].iloc[-1]
        fc_x = [last_x] + [label_func(t) for t in fc["PeriodoData"]]
        fc_y = [last_y] + fc["Forecast"].tolist()
        up = [last_y] + fc["IC 90% Superior"].tolist()
        lo = [last_y] + fc["IC 90% Inferior"].tolist()
        fig.add_trace(go.Scatter(x=fc_x + fc_x[::-1], y=up + lo[::-1], fill="toself",
                                 fillcolor="rgba(30,126,200,0.12)", line=dict(color="rgba(0,0,0,0)"),
                                 name="IC 90%", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=fc_x, y=fc_y, mode="lines+markers", name=f"Forecast ({model_name})",
                                 line=dict(color=C_BLUE, width=3, dash="dash"), marker=dict(size=7, symbol="diamond")))
    eixo = "Mês" if gran == "Mensal" else "Trimestre"
    fig.update_layout(template="plotly_white", height=420, margin=dict(l=10, r=10, t=95, b=10),
                      font=dict(family="Inter, sans-serif", color="#1a2e1a"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                      plot_bgcolor="rgba(255,255,255,0)", paper_bgcolor="rgba(255,255,255,0)",
                      title=dict(text=f"{prod} — histórico e previsão ({gran.lower()})", font=dict(size=15, color=C_GREEN), y=0.97, yanchor="top"),
                      yaxis=dict(title="Quantidade", gridcolor="#e0ece0"), xaxis=dict(title=eixo))
    return fig

def plot_model_comparison(prod, gran, best_model, treated_q_df, treated_m_df, forecast_df):
    """Compara os modelos no forecast dentro da granularidade escolhida, destacando o melhor."""
    hist, label_func = _treated_for(prod, gran, treated_q_df, treated_m_df)
    fig = go.Figure()
    if hist.empty:
        return fig
    fig.add_trace(go.Scatter(x=[label_func(t) for t in hist["Periodo"]], y=hist["Valor Tratado"],
                             mode="lines+markers", name="Histórico", line=dict(color=C_GREEN, width=3), marker=dict(size=5)))
    last_x = label_func(hist["Periodo"].iloc[-1]); last_y = hist["Valor Tratado"].iloc[-1]
    models_here = forecast_df[(forecast_df["Produto"] == prod) & (forecast_df["Granularidade"] == gran)]["Modelo"].unique()
    palette = ["#1e7ec8", "#d4a017", "#9b59b6", "#e67e22", "#16a085"]
    for i, model_name in enumerate(sorted(models_here)):
        fc = forecast_df[(forecast_df["Produto"] == prod) & (forecast_df["Granularidade"] == gran) &
                         (forecast_df["Modelo"] == model_name)].sort_values("PeriodoData")
        if fc.empty: continue
        fc_x = [last_x] + [label_func(t) for t in fc["PeriodoData"]]
        fc_y = [last_y] + fc["Forecast"].tolist()
        is_best = (model_name == best_model)
        fig.add_trace(go.Scatter(x=fc_x, y=fc_y, mode="lines+markers",
                                 name=f"{model_name}" + (" ✓ escolhido" if is_best else ""),
                                 line=dict(color=palette[i % len(palette)], width=4 if is_best else 2,
                                           dash="solid" if is_best else "dot"),
                                 marker=dict(size=8 if is_best else 5), opacity=1.0 if is_best else 0.5))
    eixo = "Mês" if gran == "Mensal" else "Trimestre"
    fig.update_layout(template="plotly_white", height=420, margin=dict(l=10, r=10, t=95, b=10),
                      font=dict(family="Inter, sans-serif", color="#1a2e1a"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                      plot_bgcolor="rgba(255,255,255,0)", paper_bgcolor="rgba(255,255,255,0)",
                      title=dict(text=f"{prod} — comparação entre modelos ({gran.lower()})", font=dict(size=15, color=C_GREEN), y=0.97, yanchor="top"),
                      yaxis=dict(title="Quantidade", gridcolor="#e0ece0"), xaxis=dict(title=eixo))
    return fig


# =========================================================
# SIDEBAR
# =========================================================
# Menu de navegação em grupos. Os títulos (Principal/Metodologia/Resultado)
# são apenas rótulos de markdown — não são clicáveis. A seleção é única:
# clicar num item de um grupo desmarca os demais.
GROUPS = {
    "Principal": ["Início", "Upload", "Exportação"],
    "Previsão de Demanda": ["Resultados mais significativos", "Resultados abrangentes"],
}

if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Início"

# Redirect automático pós-upload: leva direto aos resultados
if st.session_state.get("go_forecast"):
    st.session_state.nav_page = "Resultados mais significativos"
    st.session_state.go_forecast = False

with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
        <div class="sb-logo-row">
            <div class="sb-logo-mark"></div>
            <div class="sb-logo-title">LFDA Forecast</div>
        </div>
        <div class="sb-logo-sub">Sistema de Previsão de Demanda</div>
    </div>
    """, unsafe_allow_html=True)

    for group_name, items in GROUPS.items():
        st.markdown(f'<span class="sb-group-label">{group_name}</span>', unsafe_allow_html=True)
        for item in items:
            is_active = (st.session_state.nav_page == item)
            if st.button(item, key=f"navbtn_{item}", use_container_width=True,
                         type="primary" if is_active else "secondary"):
                st.session_state.nav_page = item
                st.rerun()

    st.markdown("""<div class="sb-config"><div class="sb-config-title">Configuração atual</div>Modelos: SES · Holt · Holt-Winters<br>Parâmetros: otimização automática<br>Granularidade: mensal e trimestral (auto)<br>Horizonte: 1 ano à frente<br>Intervalo de confiança: 90%</div>""", unsafe_allow_html=True)

current = st.session_state.nav_page


# =========================================================
# Session State
# =========================================================
for key in ["raw_df", "monthly", "quarterly", "treated_q", "treated_m", "summary", "audit", "forecast", "inferred_cols"]:
    if key not in st.session_state:
        st.session_state[key] = None

def process_file(uploaded_file):
    raw = pd.read_excel(uploaded_file) if uploaded_file.name.lower().endswith((".xlsx", ".xls")) else pd.read_csv(uploaded_file, sep=None, engine="python")
    raw = clean_dataframe(raw)
    product_col, date_col, qty_col = infer_columns(raw)
    if not all([product_col, date_col, qty_col]):
        raise ValueError("Não foi possível identificar as colunas de Produto, Data e Quantidade.")
    monthly = prepare_monthly(raw, product_col, date_col, qty_col)
    quarterly = fill_missing_periods(monthly_to_quarterly(monthly), "Q")
    treated_q, treated_m, summary, audit, forecast = run_models(monthly)
    st.session_state.raw_df = raw
    st.session_state.monthly = monthly
    st.session_state.quarterly = quarterly
    st.session_state.treated_q = treated_q
    st.session_state.treated_m = treated_m
    st.session_state.summary = summary
    st.session_state.audit = audit
    st.session_state.forecast = forecast
    st.session_state.inferred_cols = (product_col, date_col, qty_col)

def has_results():
    return st.session_state.summary is not None and not st.session_state.summary.empty


# =========================================================
# INÍCIO
# =========================================================
if current == "Início":
    st.markdown("""
    <div class="capa-head">
        <div class="capa-title">LFDA Forecast</div>
        <div class="capa-subtitle">Sistema de Previsão de Demanda de Insumos Agropecuários</div>
        <div class="capa-inst">Desenvolvido por alunos de Engenharia de Produção da UFRGS em parceria com a LFDA-RS</div>
        <div class="capa-context">
            Previsão de demanda — um estudo de caso baseado nos principais produtos do histórico do laboratório:
            <br>
            <span class="capa-prod">Metanol</span>
            <span class="capa-prod">Acetonitrila</span>
            <span class="capa-prod">Tubo Falcon 15ml</span>
            <span class="capa-prod">Tubo Falcon 50ml</span>
            <span class="capa-prod">Vial Vidro</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown('<div class="section-title">Sobre o projeto</div>', unsafe_allow_html=True)
        st.markdown("""<div class="info-card"><strong>O que é o LFDA Forecast?</strong><br><br>
        Ferramenta desenvolvida por alunos de <strong>Engenharia de Produção da UFRGS</strong>
        em parceria com a <strong>LFDA-RS</strong>, com o objetivo de oferecer uma solução prática
        e acessível para previsão de demanda de insumos agropecuários — sem exigir conhecimento
        técnico em programação ou estatística.<br><br>
        O sistema automatiza todo o processo: da leitura da planilha até a exportação dos resultados,
        rodando diretamente no navegador sem envio de dados para servidores externos.</div>""", unsafe_allow_html=True)
        st.markdown("""<div class="info-card-blue"><strong>Objetivo</strong><br><br>
        Gerar previsões de demanda de 1 ano para cada produto. O sistema testa modelos em escala mensal
        e trimestral e indica automaticamente a combinação de modelo e granularidade que melhor se ajusta
        ao histórico de cada produto — com base no menor erro de previsão (MAPE).</div>""", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="section-title">Equipe</div>', unsafe_allow_html=True)
        st.markdown("""<div class="info-card-yellow"><strong>Desenvolvedores</strong><br><br>
        Alunos de graduação em <strong>Engenharia de Produção</strong> da Universidade Federal
        do Rio Grande do Sul (UFRGS), em parceria com a LFDA-RS.<br><br>
        <strong>Instituição:</strong> UFRGS — Porto Alegre, RS<br><strong>Parceiro:</strong> LFDA-RS</div>""", unsafe_allow_html=True)
        st.markdown("""<div class="info-card"><strong>Tecnologias</strong><br><br>
        Python · Streamlit · Pandas · NumPy · Statsmodels · Plotly · XlsxWriter<br><br>
        <strong>Modelos:</strong> SES, Holt e Holt-Winters</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Formato da planilha de entrada</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-card-yellow">A planilha deve ter três colunas preenchidas da seguinte forma:<br><br>
    <strong>Coluna B</strong> — Nome do produto &nbsp;|&nbsp; <strong>Coluna C</strong> — Data (mês de referência) &nbsp;|&nbsp; <strong>Coluna D</strong> — Quantidade</div>""", unsafe_allow_html=True)
    st.markdown("**Exemplo:**")
    st.dataframe(pd.DataFrame({"Produto": ["Metanol", "Acetonitrila", "Tubo Falcon 15ml", "Tubo Falcon 50ml", "Vial Vidro"],
                               "Data": ["março de 2026"] * 5, "Quantidade": [5000, 16000, 1000, 1000, 500]}), use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Como funcionam os modelos estatísticos</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="model-card"><div class="model-title">SES — Suavização Exponencial Simples</div>
        <div class="model-desc">Calcula uma <strong>média ponderada do histórico</strong>, dando mais peso aos períodos
        recentes. O parâmetro <strong>Alpha (α)</strong> controla essa ponderação. Projeta um valor constante —
        não captura tendência nem sazonalidade.</div>
        <div class="model-when">Indicado quando: demanda oscila sem tendência nem padrão sazonal.</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="model-card"><div class="model-title">Holt — Suavização com Tendência</div>
        <div class="model-desc">Adiciona um <strong>componente de tendência</strong> ao SES, projetando uma linha com
        inclinação. Usa <strong>Alpha (α)</strong> para o nível e <strong>Beta (β)</strong> para a tendência.
        Não captura sazonalidade.</div>
        <div class="model-when">Indicado quando: demanda tem crescimento ou queda consistente.</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="model-card"><div class="model-title">Holt-Winters — Tendência e Sazonalidade</div>
        <div class="model-desc">O mais completo. Adiciona um <strong>componente sazonal</strong> (ciclo de 4 trimestres).
        Usa <strong>Alpha (α)</strong>, <strong>Beta (β)</strong> e <strong>Gamma (γ)</strong>. Requer ao menos
        8 trimestres de histórico.</div>
        <div class="model-when">Indicado quando: demanda tem picos e vales anuais.</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Fluxo completo — passo a passo</div>', unsafe_allow_html=True)
    st.markdown("""<div class="step-flow">
        <div class="step-item"><div class="step-circle">1</div><div class="step-label">Upload</div><div class="step-desc">Envie o Excel com Produto, Data e Quantidade</div></div>
        <div class="step-arrow">→</div>
        <div class="step-item"><div class="step-circle">2</div><div class="step-label">Validação</div><div class="step-desc">Sistema identifica colunas e converte datas</div></div>
        <div class="step-arrow">→</div>
        <div class="step-item"><div class="step-circle">3</div><div class="step-label">Trimestral</div><div class="step-desc">Base mensal agregada em trimestres</div></div>
        <div class="step-arrow">→</div>
        <div class="step-item"><div class="step-circle">4</div><div class="step-label">Zeros</div><div class="step-desc">Zeros redistribuídos sem alterar o total</div></div>
        <div class="step-arrow">→</div>
        <div class="step-item"><div class="step-circle">5</div><div class="step-label">Split</div><div class="step-desc">Treino e teste adaptativos</div></div>
        <div class="step-arrow">→</div>
        <div class="step-item"><div class="step-circle">6</div><div class="step-label">Modelos</div><div class="step-desc">SES, Holt e Holt-Winters calibrados</div></div>
        <div class="step-arrow">→</div>
        <div class="step-item"><div class="step-circle">7</div><div class="step-label">MAPE</div><div class="step-desc">Erro por período e agregado</div></div>
        <div class="step-arrow">→</div>
        <div class="step-item"><div class="step-circle">8</div><div class="step-label">Forecast</div><div class="step-desc">4 trimestres com IC 90%</div></div>
        <div class="step-arrow">→</div>
        <div class="step-item"><div class="step-circle">9</div><div class="step-label">Exportação</div><div class="step-desc">Excel formatado completo</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Split treino/teste adaptativo</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({"Histórico disponível": ["17+ trimestres", "12–16 trimestres", "8–11 trimestres", "5–7 trimestres", "3–4 trimestres", "Menos de 3"],
                               "Treino": ["13", "Restante", "Restante", "Restante", "Restante", "Todos"],
                               "Teste": ["4", "4", "3", "2", "1", "0"],
                               "Confiança do MAPE": ["Alta", "Moderada", "Moderada", "Baixa", "Baixa", "Exploratória"]}), use_container_width=True)
    st.markdown("""<div class="info-card-blue"><strong>O que é o MAPE?</strong><br><br>
    MAPE (<em>Mean Absolute Percentage Error</em>) mede em porcentagem o quanto o modelo errou em relação à
    demanda real. Um MAPE de <strong>10%</strong> significa erro médio de 10%. Quanto menor, melhor.<br><br>
    O sistema usa estes patamares de qualidade: <span class="mape-badge mape-exc">Excelente · ≤15%</span>
    <span class="mape-badge mape-bom">Bom · 15–30%</span> <span class="mape-badge mape-att">Atenção · >30%</span></div>""", unsafe_allow_html=True)


# =========================================================
# UPLOAD
# =========================================================
elif current == "Upload":
    st.markdown('<div class="page-title">Upload da base</div><div class="page-sub">Importe a planilha de demanda para processar os modelos.</div>', unsafe_allow_html=True)
    st.markdown("<div class='small-muted'>Envie uma base mensal em Excel ou CSV com as colunas Produto, Data e Quantidade.</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Arquivo de entrada", type=["xlsx", "xls", "csv"])
    if uploaded is not None:
        try:
            process_file(uploaded)
            st.session_state.go_forecast = True
            st.success("Base processada com sucesso! Redirecionando para os resultados...")
            st.rerun()
        except Exception as e:
            st.error(f"Erro ao processar arquivo: {e}")
    if st.session_state.raw_df is not None:
        product_col, date_col, qty_col = st.session_state.inferred_cols
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Linhas originais", fmt_br(len(st.session_state.raw_df)))
        c2.metric("Produtos", st.session_state.monthly["Produto"].nunique())
        c3.metric("Meses válidos", st.session_state.monthly["Data"].nunique())
        c4.metric("Trimestres", st.session_state.quarterly["Periodo"].nunique())
        st.markdown("### Colunas identificadas")
        st.dataframe(pd.DataFrame({"Campo": ["Produto", "Data", "Quantidade"], "Coluna identificada": [product_col, date_col, qty_col]}), use_container_width=True)
        st.markdown("### Prévia da base original")
        st.dataframe(st.session_state.raw_df.head(30), use_container_width=True)
    else:
        st.info("Comece subindo a planilha. Depois acesse a aba Forecast para ver as previsões.")


# =========================================================
# RESULTADOS ABRANGENTES (Validação + Modelos + Auditoria)
# =========================================================
elif current == "Resultados abrangentes":
    st.markdown('<div class="page-title">Resultados abrangentes</div><div class="page-sub">Detalhamento completo do processamento: base e tratamento, comparação de modelos e auditoria estatística período a período.</div>', unsafe_allow_html=True)
    if st.session_state.monthly is None or not has_results():
        st.info("Faça o upload de uma base em **Upload** para visualizar os resultados abrangentes.")
    else:
        # ---------------- SEÇÃO 1: BASE E TRATAMENTO ----------------
        st.markdown('<div class="section-title">1. Base e tratamento dos dados</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Registros mensais", fmt_br(len(st.session_state.monthly)))
        c2.metric("Registros trimestrais", fmt_br(len(st.session_state.quarterly)))
        c3.metric("Demanda total", fmt_br(st.session_state.monthly['Quantidade'].sum()))

        st.markdown("##### Base mensal padronizada")
        st.dataframe(st.session_state.monthly, use_container_width=True)

        st.markdown("##### Base trimestral")
        q_view = st.session_state.quarterly.copy()
        q_view["Trimestre"] = q_view["Periodo"].apply(quarter_label)
        st.dataframe(q_view[["Produto", "Trimestre", "Quantidade"]], use_container_width=True)

        st.markdown("##### Tratamento dos zeros — série trimestral")
        if st.session_state.treated_q is not None and not st.session_state.treated_q.empty:
            t_view = st.session_state.treated_q.copy()
            t_view["Trimestre"] = t_view["Periodo"].apply(quarter_label)
            t_view["Diferença"] = t_view["Valor Tratado"] - t_view["Quantidade"]
            st.dataframe(t_view[["Produto", "Trimestre", "Quantidade", "Valor Tratado", "Diferença"]], use_container_width=True)

        st.markdown("##### Tratamento dos zeros — série mensal")
        if st.session_state.treated_m is not None and not st.session_state.treated_m.empty:
            tm_view = st.session_state.treated_m.copy()
            tm_view["Mês"] = tm_view["Periodo"].apply(month_label)
            tm_view["Diferença"] = tm_view["Valor Tratado"] - tm_view["Quantidade"]
            st.dataframe(tm_view[["Produto", "Mês", "Quantidade", "Valor Tratado", "Diferença"]], use_container_width=True)

        st.markdown("---")

        # ---------------- SEÇÃO 2: COMPARAÇÃO DE MODELOS ----------------
        st.markdown('<div class="section-title">2. Comparação de modelos e granularidades</div>', unsafe_allow_html=True)
        summary = st.session_state.summary.copy()
        best_rows = []
        for prod in sorted(summary["Produto"].unique()):
            gran, model, mape = best_combo_for(summary, prod)
            best_rows.append({"Produto": prod, "Granularidade": gran, "Modelo": model, "MAPE Agregado %": mape})
        best = pd.DataFrame(best_rows)

        c1, c2, c3 = st.columns(3)
        c1.metric("Produtos modelados", summary["Produto"].nunique())
        c2.metric("Combinações testadas", len(summary))
        c3.metric("MAPE agregado mediano", f"{summary['MAPE Agregado %'].median():.1f}%" if summary["MAPE Agregado %"].notna().any() else "n/a")

        st.markdown("##### Resultado por produto, granularidade e modelo")
        best_keys = set(zip(best["Produto"], best["Granularidade"], best["Modelo"]))
        def highlight_best(row):
            is_best = (row["Produto"], row["Granularidade"], row["Modelo"]) in best_keys
            return ['background-color: #d6f5dd; font-weight: 600' if is_best else '' for _ in row]
        styled = summary.style.apply(highlight_best, axis=1).format({
            "MAPE Período Médio %": "{:.1f}", "MAPE Agregado %": "{:.1f}",
            "Alpha": "{:.4f}", "Beta": "{:.4f}", "Gamma": "{:.4f}"}, na_rep="—")
        st.dataframe(styled, use_container_width=True)
        st.caption("Linhas em verde: a melhor combinação de granularidade + modelo de cada produto (menor MAPE agregado).")

        st.markdown("##### Melhor combinação por produto")
        best_show = best.copy()
        best_show["MAPE Agregado %"] = best_show["MAPE Agregado %"].apply(lambda v: f"{v:.1f}" if pd.notna(v) else "—")
        st.dataframe(best_show, use_container_width=True)

        st.markdown("---")

        # ---------------- SEÇÃO 3: AUDITORIA ----------------
        st.markdown('<div class="section-title">3. Auditoria estatística período a período</div>', unsafe_allow_html=True)
        if st.session_state.audit is not None and not st.session_state.audit.empty:
            audit = st.session_state.audit.copy()
            c1, c2, c3 = st.columns(3)
            prod_filter = c1.selectbox("Produto", ["Todos"] + sorted(audit["Produto"].unique().tolist()), key="aud_prod")
            gran_filter = c2.selectbox("Granularidade", ["Todas"] + sorted(audit["Granularidade"].unique().tolist()), key="aud_gran")
            model_filter = c3.selectbox("Modelo", ["Todos"] + sorted(audit["Modelo"].unique().tolist()), key="aud_model")
            if prod_filter != "Todos": audit = audit[audit["Produto"] == prod_filter]
            if gran_filter != "Todas": audit = audit[audit["Granularidade"] == gran_filter]
            if model_filter != "Todos": audit = audit[audit["Modelo"] == model_filter]
            st.dataframe(audit.drop(columns=["PeriodoData"], errors="ignore"), use_container_width=True)


# =========================================================
# FORECAST
# =========================================================
elif current == "Resultados mais significativos":
    st.markdown('<div class="page-title">Resultados mais significativos</div><div class="page-sub">Previsão de demanda por produto. O sistema testa cada produto em escala mensal e trimestral e escolhe automaticamente a melhor combinação de modelo e granularidade.</div>', unsafe_allow_html=True)
    if st.session_state.forecast is None or st.session_state.forecast.empty:
        st.info("Faça o upload de uma base em **Upload** para gerar o forecast.")
    else:
        fc_all = st.session_state.forecast.copy()
        summary = st.session_state.summary.copy()
        treated_q = st.session_state.treated_q
        treated_m = st.session_state.treated_m
        produtos = sorted(fc_all["Produto"].unique().tolist())

        # ---------- RESUMO EXECUTIVO ----------
        st.markdown('<div class="section-title">Resumo executivo</div>', unsafe_allow_html=True)
        st.markdown("<div class='small-muted'>Previsão anual usando a melhor combinação de modelo e granularidade de cada produto.</div>", unsafe_allow_html=True)
        st.write("")

        cols = st.columns(min(len(produtos), 3))
        for i, prod in enumerate(produtos):
            gran, model, mape_v = best_combo_for(summary, prod)
            fc_best = fc_all[(fc_all["Produto"] == prod) & (fc_all["Granularidade"] == gran) & (fc_all["Modelo"] == model)]
            anual = fc_best["Forecast"].sum()
            anual_sup = fc_best["IC 90% Superior"].sum()
            with cols[i % len(cols)]:
                st.markdown(f"""<div class="exec-card">
                    <div class="exec-prod">{prod}</div>
                    <div class="exec-row"><span class="exec-label">Previsão anual</span><span class="exec-value">{fmt_br(anual)}</span></div>
                    <div class="exec-row"><span class="exec-label">Limite superior (IC 90%)</span><span class="exec-value">{fmt_br(anual_sup)}</span></div>
                    <div class="exec-row"><span class="exec-label">Melhor modelo</span><span class="exec-model">{model}</span></div>
                    <div class="exec-row"><span class="exec-label">Granularidade</span>{gran_badge_html(gran)}</div>
                    <div style="margin-top:8px">{mape_badge_html(mape_v)}</div>
                </div>""", unsafe_allow_html=True)

        st.write("")
        st.markdown("##### Previsão anual consolidada (melhor combinação de cada produto)")
        st.markdown("<div class='small-muted'>Como cada produto pode ter sido melhor em mensal ou trimestral, a tabela mostra o total anual previsto e a granularidade escolhida.</div>", unsafe_allow_html=True)

        rows = []
        for prod in produtos:
            gran, model, mape_v = best_combo_for(summary, prod)
            fc_best = fc_all[(fc_all["Produto"] == prod) & (fc_all["Granularidade"] == gran) & (fc_all["Modelo"] == model)]
            rows.append({
                "Produto": prod, "Granularidade": gran, "Modelo": model,
                "Total anual previsto": fc_best["Forecast"].sum(),
                "Limite superior anual": fc_best["IC 90% Superior"].sum(),
                "MAPE Agregado %": mape_v,
            })
        consol = pd.DataFrame(rows)
        st.dataframe(consol.style.format({
            "Total anual previsto": lambda v: fmt_br(v),
            "Limite superior anual": lambda v: fmt_br(v),
            "MAPE Agregado %": lambda v: f"{v:.1f}" if pd.notna(v) else "—"}, na_rep="—"), use_container_width=True)

        st.markdown("---")

        # ---------- VISUALIZAÇÃO POR PRODUTO ----------
        st.markdown('<div class="section-title">Visualização detalhada</div>', unsafe_allow_html=True)
        sel_prod = st.selectbox("Selecione o produto", produtos)
        gran, model, mape_v = best_combo_for(summary, sel_prod)
        st.markdown(f"<div class='small-muted'>Melhor combinação para <strong>{sel_prod}</strong>: modelo <strong>{model}</strong> em escala <strong>{gran.lower()}</strong>.</div>", unsafe_allow_html=True)
        st.write("")

        st.plotly_chart(plot_history_forecast(sel_prod, gran, model, treated_q, treated_m, fc_all), use_container_width=True)
        st.markdown(f"<div class='small-muted'>Histórico + previsão do modelo escolhido (<strong>{model}</strong>, {gran.lower()}) com faixa de confiança de 90%.</div>", unsafe_allow_html=True)

        st.write("")
        st.plotly_chart(plot_model_comparison(sel_prod, gran, model, treated_q, treated_m, fc_all), use_container_width=True)
        st.markdown(f"<div class='small-muted'>Comparação dos modelos na granularidade vencedora. A linha em destaque é o <strong>{model}</strong>, escolhido por ter o menor MAPE agregado.</div>", unsafe_allow_html=True)

        st.markdown("---")

        # ---------- TABELA DETALHADA ----------
        st.markdown('<div class="section-title">Tabela de previsões</div>', unsafe_allow_html=True)
        fc = fc_all.copy()
        c1, c2, c3 = st.columns(3)
        prod_filter = c1.selectbox("Filtrar produto", ["Todos"] + produtos, key="fc_prod")
        gran_filter = c2.selectbox("Filtrar granularidade", ["Todas"] + sorted(fc["Granularidade"].unique().tolist()), key="fc_gran")
        model_filter = c3.selectbox("Filtrar modelo", ["Todos"] + sorted(fc["Modelo"].unique().tolist()), key="fc_model")
        if prod_filter != "Todos": fc = fc[fc["Produto"] == prod_filter]
        if gran_filter != "Todas": fc = fc[fc["Granularidade"] == gran_filter]
        if model_filter != "Todos": fc = fc[fc["Modelo"] == model_filter]
        fc_show = fc.drop(columns=["PeriodoData"], errors="ignore")
        st.dataframe(fc_show.style.format({
            "Forecast": lambda v: fmt_br(v), "IC 90% Inferior": lambda v: fmt_br(v),
            "IC 90% Superior": lambda v: fmt_br(v), "Soma Anual - Limite Superior": lambda v: fmt_br(v),
            "Alpha": "{:.4f}", "Beta": "{:.4f}", "Gamma": "{:.4f}"}, na_rep="—"), use_container_width=True)


# =========================================================
# EXPORTAÇÃO
# =========================================================
elif current == "Exportação":
    st.markdown('<div class="page-title">Exportação</div><div class="page-sub">Baixe todos os resultados consolidados em uma planilha Excel.</div>', unsafe_allow_html=True)
    if not has_results():
        st.info("Faça o upload de uma base em **Upload** para habilitar a exportação.")
    else:
        fc_export = st.session_state.forecast.drop(columns=["PeriodoData"], errors="ignore")
        q_tab = st.session_state.quarterly.copy()
        q_tab["Trimestre"] = q_tab["Periodo"].apply(quarter_label)
        q_tab = q_tab[["Produto", "Trimestre", "Quantidade"]]
        sheets = {
            "Base Original": st.session_state.raw_df,
            "Mensal Padronizada": st.session_state.monthly,
            "Trimestral": q_tab,
            "Modelos MAPE": st.session_state.summary,
            "Auditoria": st.session_state.audit.drop(columns=["PeriodoData"], errors="ignore") if st.session_state.audit is not None else pd.DataFrame(),
            "Forecast": fc_export,
        }
        if st.session_state.treated_q is not None and not st.session_state.treated_q.empty:
            tq = st.session_state.treated_q.copy()
            tq["Trimestre"] = tq["Periodo"].apply(quarter_label)
            sheets["Zeros Trimestral"] = tq[["Produto", "Trimestre", "Quantidade", "Valor Tratado"]]
        if st.session_state.treated_m is not None and not st.session_state.treated_m.empty:
            tm = st.session_state.treated_m.copy()
            tm["Mes"] = tm["Periodo"].apply(month_label)
            sheets["Zeros Mensal"] = tm[["Produto", "Mes", "Quantidade", "Valor Tratado"]]
        excel = to_excel_bytes(sheets)
        st.download_button(label="Baixar resultado em Excel", data=excel,
                           file_name="LFDA_Forecast_Resultados.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.markdown("<div class='small-muted'>Contém: base original, mensal, trimestral, tratamento de zeros (mensal e trimestral), MAPE, auditoria e forecast.</div>", unsafe_allow_html=True)
