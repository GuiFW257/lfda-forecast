import re
import io
import unicodedata
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing


st.set_page_config(page_title="LFDA Forecast", page_icon="🌱", layout="wide", initial_sidebar_state="expanded")

# Cores de marca usadas também nos gráficos
C_GREEN = "#1a6b2f"
C_GREEN_L = "#38b249"
C_BLUE = "#1e7ec8"
C_AMBER = "#d4a017"
C_GREY = "#9aa8b5"
MODEL_COLORS = {"SES": "#1e7ec8", "Holt": "#d4a017", "Holt-Winters": "#9b59b6"}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --green: #38b249; --green-dark: #1a6b2f; --green-light: #e8f5e9;
    --blue: #1e7ec8; --blue-dark: #155fa0; --border: #b2d8b2;
    --muted: #4a7a4a; --text: #1a2e1a;
}
* { font-family: 'Inter', sans-serif !important; }
.stApp { background: linear-gradient(135deg, #f0f7f0 0%, #f4f9f4 60%, #e8f5e9 100%); color: var(--text); }
[data-testid="stSidebar"] { background: #ffffff; border-right: 2px solid var(--border); }
[data-testid="stSidebar"] * { font-family: 'Inter', sans-serif !important; }

.main-header { padding: 1.25rem 0 1rem 0; border-bottom: 3px solid var(--green); margin-bottom: 1.4rem; }
.main-title { font-size: 2.1rem; font-weight: 700; letter-spacing: -0.04em; color: var(--green-dark); margin-bottom: 0.15rem; }
.main-subtitle { font-size: 1rem; color: var(--muted); font-weight: 400; margin-bottom: 0.45rem; }
.navline { color: var(--muted); font-size: 0.87rem; }
.status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--green); margin-right: 6px; box-shadow: 0 0 8px rgba(56,178,73,0.6); }

div[data-testid="stMetric"] { background: #fff; border: 1px solid var(--border); border-top: 3px solid var(--green); border-radius: 14px; padding: .8rem 1rem; box-shadow: 0 2px 8px rgba(26,107,47,0.07); }
div[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 12px; }
.stButton>button, .stDownloadButton>button { border-radius: 10px; border: 2px solid var(--green-dark); background: var(--green-dark); color: #fff; font-weight: 600; }
.stButton>button:hover, .stDownloadButton>button:hover { background: var(--green); border-color: var(--green); }
h1, h2, h3 { font-family: 'Inter', sans-serif !important; letter-spacing: -0.03em; color: var(--green-dark); }

.sb-logo { padding: 1.2rem 1rem 0.5rem 1rem; }
.sb-logo-title { font-size: 1.15rem; font-weight: 700; color: var(--green-dark); letter-spacing: -0.03em; }
.sb-logo-sub { font-size: 0.75rem; color: var(--muted); margin-top: 2px; }
.sb-group-label { font-size: 0.68rem; font-weight: 600; letter-spacing: 0.1em; color: #9ab89a; text-transform: uppercase; padding: 0.9rem 1rem 0.3rem 1rem; display: block; }
.sb-divider { border: none; border-top: 1px solid #ddeedd; margin: 4px 12px; }
.sb-config { margin: 12px 10px 0 10px; padding: 10px 12px; background: #f4faf4; border: 1px solid var(--border); border-radius: 10px; font-size: 0.75rem; color: var(--muted); line-height: 1.8; }
.sb-config-title { font-weight: 700; color: var(--green-dark); font-size: 0.78rem; margin-bottom: 4px; }

.capa-hero { background: linear-gradient(135deg, #1a6b2f 0%, #2d8a45 50%, #1e7ec8 100%); border-radius: 20px; padding: 3.2rem 2.8rem; color: white; margin-bottom: 2rem; }
.capa-title { font-size: 3.2rem; font-weight: 700; letter-spacing: -0.04em; margin-bottom: 0.5rem; color: white; }
.capa-subtitle { font-size: 1.4rem; font-weight: 500; opacity: 0.95; margin-bottom: 0.4rem; color: white; }
.capa-inst { font-size: 0.95rem; opacity: 0.75; margin-bottom: 1.5rem; color: white; }
.capa-context { font-size: 1rem; opacity: 0.92; line-height: 1.7; color: white; max-width: 850px; }
.capa-prod { display: inline-block; background: rgba(255,255,255,0.2); border: 1px solid rgba(255,255,255,0.4); border-radius: 8px; padding: 0.25rem 0.7rem; font-size: 0.85rem; font-weight: 600; color: white; margin: 4px 5px 0 0; }

.info-card { background: #fff; border: 1px solid var(--border); border-left: 4px solid var(--green); border-radius: 12px; padding: 1.2rem 1.4rem; margin-bottom: 1rem; color: var(--text); }
.info-card-blue { background: #fff; border: 1px solid #b3d4ef; border-left: 4px solid var(--blue); border-radius: 12px; padding: 1.2rem 1.4rem; margin-bottom: 1rem; color: var(--text); }
.info-card-yellow { background: #fffef5; border: 1px solid #e8d88a; border-left: 4px solid #d4a017; border-radius: 12px; padding: 1.2rem 1.4rem; margin-bottom: 1rem; color: var(--text); }

.model-card { background: #fff; border: 1px solid var(--border); border-radius: 12px; padding: 1.1rem 1.3rem; margin-bottom: 10px; }
.model-title { font-size: 0.95rem; font-weight: 700; color: var(--green-dark); margin-bottom: 0.4rem; }
.model-desc { font-size: 0.83rem; color: var(--muted); line-height: 1.6; }
.model-when { font-size: 0.78rem; color: var(--blue-dark); margin-top: 0.4rem; font-weight: 500; }

.step-flow { display: flex; align-items: flex-start; gap: 0; margin: 1.5rem 0; flex-wrap: wrap; }
.step-item { display: flex; flex-direction: column; align-items: center; min-width: 90px; flex: 1; }
.step-circle { width: 48px; height: 48px; border-radius: 50%; background: var(--green-dark); color: white; display: flex; align-items: center; justify-content: center; font-size: 1.1rem; font-weight: 700; margin-bottom: 0.5rem; }
.step-label { font-size: 0.77rem; font-weight: 700; color: var(--green-dark); text-align: center; }
.step-desc { font-size: 0.7rem; color: var(--muted); text-align: center; margin-top: 3px; line-height: 1.4; }
.step-arrow { font-size: 1.4rem; color: var(--border); margin-top: 12px; padding: 0 2px; flex: 0; }

.section-title { font-size: 1.15rem; font-weight: 700; color: var(--green-dark); margin: 1.5rem 0 0.8rem 0; border-left: 4px solid var(--green); padding-left: 10px; }
.small-muted { color: var(--muted); font-size: .88rem; }

/* Resumo executivo */
.exec-card { background: #fff; border: 1px solid var(--border); border-radius: 16px; padding: 1.3rem 1.5rem; margin-bottom: 12px; box-shadow: 0 3px 12px rgba(26,107,47,0.08); }
.exec-prod { font-size: 1.15rem; font-weight: 700; color: var(--green-dark); margin-bottom: 0.6rem; }
.exec-row { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.3rem; }
.exec-label { font-size: 0.82rem; color: var(--muted); }
.exec-value { font-size: 1.05rem; font-weight: 600; color: var(--text); }
.exec-model { display: inline-block; background: var(--green-light); color: var(--green-dark); font-size: 0.78rem; font-weight: 600; padding: 0.2rem 0.7rem; border-radius: 20px; }

/* Selo MAPE */
.mape-badge { display: inline-block; font-size: 0.72rem; font-weight: 600; padding: 0.15rem 0.6rem; border-radius: 20px; }
.mape-exc { background: #d6f5dd; color: #137a36; }
.mape-bom { background: #fff4d6; color: #9a6d00; }
.mape-att { background: #fde0e0; color: #c0392b; }

/* Menu de navegação único */
[data-testid="stSidebar"] [role="radiogroup"] label { padding: 2px 0; }
[data-testid="stSidebar"] [role="radiogroup"] label p { font-size: 0.92rem; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <div class="main-title">🌱 LFDA Forecast</div>
    <div class="main-subtitle">Sistema de Previsão de Demanda</div>
    <div class="navline">Upload • Validação • Modelos • Auditoria • Forecast • Exportação &nbsp;&nbsp;<span class="status-dot"></span>Processamento local</div>
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
    return {"Alpha": params.get("smoothing_level", np.nan), "Beta": params.get("smoothing_trend", np.nan), "Gamma": params.get("smoothing_seasonal", np.nan)}

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
            fit = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=4, initialization_method="estimated").fit(optimized=True)
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
                    audits.append({"Produto": prod, "Modelo": model_name, "Tipo": "Teste", "Trimestre": quarter_label(dt),
                                   "Valor Real": actual_o, "Valor Tratado": actual_t, "Valor Previsto": pred,
                                   "Erro": err, "MAPE %": mape_i, "Alpha": params["Alpha"], "Beta": params["Beta"],
                                   "Gamma": params["Gamma"], "Regra Treino/Teste": split_rule})
                summaries.append({"Produto": prod, "Modelo": model_name, "Períodos": len(y), "Treino": len(train),
                                   "Teste": len(test), "Regra": split_rule, "MAPE Período Médio %": period_mape,
                                   "MAPE Agregado %": aggregate_mape, "Alpha": params["Alpha"], "Beta": params["Beta"], "Gamma": params["Gamma"]})
                fit_full, fc = fit_forecast_model(model_name, y, 4)
                fc = pd.Series(fc).astype(float).values
                # fitted values para gráfico
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
                future = future_quarters(g["Trimestre"].max(), 4)
                params_full = model_params(fit_full) if fit_full is not None else params
                for h, dt in enumerate(future, start=1):
                    pv = max(0.0, float(fc[h - 1]))
                    interval = z90 * residual_std * np.sqrt(h)
                    forecasts.append({"Produto": prod, "Modelo": model_name, "Trimestre": quarter_label(dt),
                                      "TrimestreData": dt, "Forecast": pv, "IC 90% Inferior": max(0.0, pv - interval),
                                      "IC 90% Superior": pv + interval, "Alpha": params_full["Alpha"],
                                      "Beta": params_full["Beta"], "Gamma": params_full["Gamma"]})
            except Exception as exc:
                summaries.append({"Produto": prod, "Modelo": model_name, "Períodos": len(y), "Treino": len(train),
                                   "Teste": len(test), "Regra": split_rule, "MAPE Período Médio %": np.nan,
                                   "MAPE Agregado %": np.nan, "Alpha": np.nan, "Beta": np.nan, "Gamma": np.nan, "Erro Modelo": str(exc)})
    treated_df = pd.concat(treated_frames, ignore_index=True) if treated_frames else pd.DataFrame()
    forecast_df = pd.DataFrame(forecasts)
    if not forecast_df.empty:
        annual = forecast_df.groupby(["Produto", "Modelo"], as_index=False)["IC 90% Superior"].sum()
        annual = annual.rename(columns={"IC 90% Superior": "Soma 4T - Limite Superior"})
        forecast_df = forecast_df.merge(annual, on=["Produto", "Modelo"], how="left")
    return treated_df, pd.DataFrame(summaries), pd.DataFrame(audits), forecast_df

def best_model_for(summary, prod):
    sub = summary[summary["Produto"] == prod].copy()
    sub = sub.sort_values("MAPE Agregado %", na_position="last")
    if sub.empty: return None
    return sub.iloc[0]["Modelo"]

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
def plot_history_forecast(prod, treated_df, forecast_df, model_name):
    """Linha do tempo: histórico tratado + forecast do modelo escolhido + faixa IC90."""
    hist = treated_df[treated_df["Produto"] == prod].sort_values("Trimestre")
    fc = forecast_df[(forecast_df["Produto"] == prod) & (forecast_df["Modelo"] == model_name)].sort_values("TrimestreData")
    fig = go.Figure()

    # Histórico
    fig.add_trace(go.Scatter(
        x=[quarter_label(t) for t in hist["Trimestre"]],
        y=hist["Valor Tratado"], mode="lines+markers", name="Histórico",
        line=dict(color=C_GREEN, width=3), marker=dict(size=7)))

    if not fc.empty:
        # ponto de conexão entre histórico e forecast
        last_x = quarter_label(hist["Trimestre"].iloc[-1])
        last_y = hist["Valor Tratado"].iloc[-1]
        fc_x = [last_x] + [quarter_label(t) for t in fc["TrimestreData"]]
        fc_y = [last_y] + fc["Forecast"].tolist()
        up = [last_y] + fc["IC 90% Superior"].tolist()
        lo = [last_y] + fc["IC 90% Inferior"].tolist()

        # Faixa IC
        fig.add_trace(go.Scatter(x=fc_x + fc_x[::-1], y=up + lo[::-1], fill="toself",
                                 fillcolor="rgba(30,126,200,0.12)", line=dict(color="rgba(0,0,0,0)"),
                                 name="IC 90%", hoverinfo="skip"))
        # Forecast
        fig.add_trace(go.Scatter(x=fc_x, y=fc_y, mode="lines+markers", name=f"Forecast ({model_name})",
                                 line=dict(color=C_BLUE, width=3, dash="dash"), marker=dict(size=7, symbol="diamond")))

    fig.update_layout(
        template="plotly_white", height=420, margin=dict(l=10, r=10, t=95, b=10),
        font=dict(family="Inter, sans-serif", color="#1a2e1a"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor="rgba(255,255,255,0)", paper_bgcolor="rgba(255,255,255,0)",
        title=dict(text=f"{prod} — histórico e previsão", font=dict(size=15, color=C_GREEN), y=0.97, yanchor="top"),
        yaxis=dict(title="Quantidade", gridcolor="#e0ece0"), xaxis=dict(title="Trimestre"))
    return fig

def plot_model_comparison(prod, treated_df, forecast_df, best):
    """Compara os 3 modelos no forecast, destacando o escolhido."""
    hist = treated_df[treated_df["Produto"] == prod].sort_values("Trimestre")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[quarter_label(t) for t in hist["Trimestre"]], y=hist["Valor Tratado"],
                             mode="lines+markers", name="Histórico", line=dict(color=C_GREEN, width=3), marker=dict(size=6)))
    last_x = quarter_label(hist["Trimestre"].iloc[-1]); last_y = hist["Valor Tratado"].iloc[-1]
    for model_name in ["SES", "Holt", "Holt-Winters"]:
        fc = forecast_df[(forecast_df["Produto"] == prod) & (forecast_df["Modelo"] == model_name)].sort_values("TrimestreData")
        if fc.empty: continue
        fc_x = [last_x] + [quarter_label(t) for t in fc["TrimestreData"]]
        fc_y = [last_y] + fc["Forecast"].tolist()
        is_best = (model_name == best)
        fig.add_trace(go.Scatter(x=fc_x, y=fc_y, mode="lines+markers",
                                 name=f"{model_name}" + (" ✓ escolhido" if is_best else ""),
                                 line=dict(color=MODEL_COLORS[model_name], width=4 if is_best else 2,
                                           dash="solid" if is_best else "dot"),
                                 marker=dict(size=8 if is_best else 5),
                                 opacity=1.0 if is_best else 0.55))
    fig.update_layout(template="plotly_white", height=420, margin=dict(l=10, r=10, t=95, b=10),
                      font=dict(family="Inter, sans-serif", color="#1a2e1a"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                      plot_bgcolor="rgba(255,255,255,0)", paper_bgcolor="rgba(255,255,255,0)",
                      title=dict(text=f"{prod} — comparação entre os 3 modelos", font=dict(size=15, color=C_GREEN), y=0.97, yanchor="top"),
                      yaxis=dict(title="Quantidade", gridcolor="#e0ece0"), xaxis=dict(title="Trimestre"))
    return fig


# =========================================================
# SIDEBAR
# =========================================================
# Menu de navegação em grupos. Os títulos (Principal/Metodologia/Resultado)
# são apenas rótulos de markdown — não são clicáveis. A seleção é única:
# clicar num item de um grupo desmarca os demais.
GROUPS = {
    "Principal": ["🏠  Início", "⬆️  Upload", "⬇️  Exportação"],
    "Metodologia": ["✅  Validação", "📐  Modelos", "🔍  Auditoria"],
    "Resultado": ["📊  Forecast"],
}

if "nav_page" not in st.session_state:
    st.session_state.nav_page = "🏠  Início"

# Redirect automático pós-upload: leva direto ao Forecast
if st.session_state.get("go_forecast"):
    st.session_state.nav_page = "📊  Forecast"
    st.session_state.go_forecast = False

def _on_group_change(group_name):
    """Quando um grupo recebe seleção, ela vira a página ativa e os demais grupos são limpos."""
    val = st.session_state.get(f"nav_{group_name}")
    if val:
        st.session_state.nav_page = val
        for other in GROUPS:
            if other != group_name:
                st.session_state[f"nav_{other}"] = None

with st.sidebar:
    st.markdown("""<div class="sb-logo"><div class="sb-logo-title">🌱 LFDA Forecast</div><div class="sb-logo-sub">Processamento local · Streamlit</div></div>""", unsafe_allow_html=True)
    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)

    for group_name, items in GROUPS.items():
        st.markdown(f'<span class="sb-group-label">{group_name}</span>', unsafe_allow_html=True)
        idx = items.index(st.session_state.nav_page) if st.session_state.nav_page in items else None
        st.radio(
            f"nav_{group_name}", items,
            index=idx, key=f"nav_{group_name}",
            label_visibility="collapsed",
            on_change=_on_group_change, args=(group_name,),
        )
        st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)

    st.markdown("""<div class="sb-config"><div class="sb-config-title">⚙️ Configuração atual</div>Modelos: SES · Holt · Holt-Winters<br>Parâmetros: otimização automática<br>Forecast: 4 trimestres à frente<br>Intervalo de confiança: 90%</div>""", unsafe_allow_html=True)

def page_name(p):
    return p.split("  ")[-1].strip() if "  " in p else p.strip()
current = page_name(st.session_state.nav_page)


# =========================================================
# Session State
# =========================================================
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
if current == "Início":
    st.markdown("""
    <div class="capa-hero">
        <div class="capa-title">🌱 LFDA Forecast</div>
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
        st.markdown("""<div class="info-card-blue"><strong>🎯 Objetivo</strong><br><br>
        Gerar previsões trimestrais de demanda para cada produto, comparando três modelos estatísticos
        e indicando automaticamente qual deles melhor se ajusta ao histórico de cada produto — com base
        no menor erro de previsão (MAPE).</div>""", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="section-title">Equipe</div>', unsafe_allow_html=True)
        st.markdown("""<div class="info-card-yellow"><strong>🎓 Desenvolvedores</strong><br><br>
        Alunos de graduação em <strong>Engenharia de Produção</strong> da Universidade Federal
        do Rio Grande do Sul (UFRGS), em parceria com a LFDA-RS.<br><br>
        <strong>Instituição:</strong> UFRGS — Porto Alegre, RS<br><strong>Parceiro:</strong> LFDA-RS</div>""", unsafe_allow_html=True)
        st.markdown("""<div class="info-card"><strong>⚙️ Tecnologias</strong><br><br>
        Python · Streamlit · Pandas · NumPy · Statsmodels · Plotly · XlsxWriter<br><br>
        <strong>Modelos:</strong> SES, Holt e Holt-Winters</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">📋 Formato da planilha de entrada</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-card-yellow">A planilha deve ter três colunas preenchidas da seguinte forma:<br><br>
    <strong>Coluna B</strong> — Nome do produto &nbsp;|&nbsp; <strong>Coluna C</strong> — Data (mês de referência) &nbsp;|&nbsp; <strong>Coluna D</strong> — Quantidade</div>""", unsafe_allow_html=True)
    st.markdown("**Exemplo:**")
    st.dataframe(pd.DataFrame({"Produto": ["Metanol", "Acetonitrila", "Tubo Falcon 15ml", "Tubo Falcon 50ml", "Vial Vidro"],
                               "Data": ["março de 2026"] * 5, "Quantidade": [5000, 16000, 1000, 1000, 500]}), use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">📐 Como funcionam os modelos estatísticos</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="model-card"><div class="model-title">SES — Suavização Exponencial Simples</div>
        <div class="model-desc">Calcula uma <strong>média ponderada do histórico</strong>, dando mais peso aos períodos
        recentes. O parâmetro <strong>Alpha (α)</strong> controla essa ponderação. Projeta um valor constante —
        não captura tendência nem sazonalidade.</div>
        <div class="model-when">✅ Use quando: demanda oscila sem tendência nem padrão sazonal.</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="model-card"><div class="model-title">Holt — Suavização com Tendência</div>
        <div class="model-desc">Adiciona um <strong>componente de tendência</strong> ao SES, projetando uma linha com
        inclinação. Usa <strong>Alpha (α)</strong> para o nível e <strong>Beta (β)</strong> para a tendência.
        Não captura sazonalidade.</div>
        <div class="model-when">✅ Use quando: demanda tem crescimento ou queda consistente.</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="model-card"><div class="model-title">Holt-Winters — Tendência e Sazonalidade</div>
        <div class="model-desc">O mais completo. Adiciona um <strong>componente sazonal</strong> (ciclo de 4 trimestres).
        Usa <strong>Alpha (α)</strong>, <strong>Beta (β)</strong> e <strong>Gamma (γ)</strong>. Requer ao menos
        8 trimestres de histórico.</div>
        <div class="model-when">✅ Use quando: demanda tem picos e vales anuais.</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">🗺️ Fluxo completo — passo a passo</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="section-title">✂️ Split treino/teste adaptativo</div>', unsafe_allow_html=True)
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
    st.subheader("⬆️ Upload da base")
    st.markdown("<div class='small-muted'>Envie uma base mensal em Excel ou CSV com as colunas Produto, Data e Quantidade.</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Arquivo de entrada", type=["xlsx", "xls", "csv"])
    if uploaded is not None:
        try:
            process_file(uploaded)
            st.session_state.go_forecast = True
            st.success("Base processada com sucesso! Redirecionando para o Forecast...")
            st.rerun()
        except Exception as e:
            st.error(f"Erro ao processar arquivo: {e}")
    if st.session_state.raw_df is not None:
        product_col, date_col, qty_col = st.session_state.inferred_cols
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Linhas originais", fmt_br(len(st.session_state.raw_df)))
        c2.metric("Produtos", st.session_state.monthly["Produto"].nunique())
        c3.metric("Meses válidos", st.session_state.monthly["Data"].nunique())
        c4.metric("Trimestres", st.session_state.quarterly["Trimestre"].nunique())
        st.markdown("### Colunas identificadas")
        st.dataframe(pd.DataFrame({"Campo": ["Produto", "Data", "Quantidade"], "Coluna identificada": [product_col, date_col, qty_col]}), use_container_width=True)
        st.markdown("### Prévia da base original")
        st.dataframe(st.session_state.raw_df.head(30), use_container_width=True)
    else:
        st.info("👈 Comece subindo a planilha. Depois acesse **Forecast** para ver as previsões.")


# =========================================================
# VALIDAÇÃO
# =========================================================
elif current == "Validação":
    st.subheader("✅ Validação e conversão mensal → trimestral")
    if st.session_state.monthly is None:
        st.info("Faça o upload de uma base em **Upload** para visualizar a validação.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Registros mensais", fmt_br(len(st.session_state.monthly)))
        c2.metric("Registros trimestrais", fmt_br(len(st.session_state.quarterly)))
        c3.metric("Demanda total", fmt_br(st.session_state.monthly['Quantidade'].sum()))
        st.markdown("### Base mensal padronizada")
        st.dataframe(st.session_state.monthly, use_container_width=True)
        st.markdown("### Base trimestral")
        q_view = st.session_state.quarterly.copy()
        q_view["Trimestre"] = q_view["Trimestre"].apply(quarter_label)
        st.dataframe(q_view, use_container_width=True)
        st.markdown("### Tratamento dos zeros")
        t_view = st.session_state.treated.copy()
        t_view["Trimestre"] = t_view["Trimestre"].apply(quarter_label)
        t_view["Diferença"] = t_view["Valor Tratado"] - t_view["Quantidade"]
        st.dataframe(t_view, use_container_width=True)


# =========================================================
# MODELOS
# =========================================================
elif current == "Modelos":
    st.subheader("📐 Backtest dos modelos")
    if not has_results():
        st.info("Faça o upload de uma base em **Upload** para rodar os modelos.")
    else:
        summary = st.session_state.summary.copy()
        best = summary.sort_values(["Produto", "MAPE Agregado %"], na_position="last").groupby("Produto").head(1)
        c1, c2, c3 = st.columns(3)
        c1.metric("Produtos modelados", summary["Produto"].nunique())
        c2.metric("Modelos rodados", len(summary))
        c3.metric("MAPE agregado mediano", f"{summary['MAPE Agregado %'].median():.1f}%" if summary["MAPE Agregado %"].notna().any() else "n/a")

        st.markdown("### Resultado por produto e modelo")
        # destaque condicional do melhor modelo por produto
        best_pairs = set(zip(best["Produto"], best["Modelo"]))
        def highlight_best(row):
            is_best = (row["Produto"], row["Modelo"]) in best_pairs
            return ['background-color: #d6f5dd; font-weight: 600' if is_best else '' for _ in row]
        styled = summary.style.apply(highlight_best, axis=1).format({
            "MAPE Período Médio %": "{:.1f}", "MAPE Agregado %": "{:.1f}",
            "Alpha": "{:.4f}", "Beta": "{:.4f}", "Gamma": "{:.4f}"}, na_rep="—")
        st.dataframe(styled, use_container_width=True)
        st.caption("🟢 Linhas em verde indicam o melhor modelo de cada produto (menor MAPE agregado).")

        st.markdown("### Melhor modelo por produto")
        st.dataframe(best, use_container_width=True)


# =========================================================
# AUDITORIA
# =========================================================
elif current == "Auditoria":
    st.subheader("🔍 Auditoria estatística")
    if st.session_state.audit is None or st.session_state.audit.empty:
        st.info("Faça o upload de uma base em **Upload** para gerar a auditoria.")
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
elif current == "Forecast":
    st.subheader("📊 Forecast — próximos 4 trimestres")
    if st.session_state.forecast is None or st.session_state.forecast.empty:
        st.info("Faça o upload de uma base em **Upload** para gerar o forecast.")
    else:
        fc_all = st.session_state.forecast.copy()
        summary = st.session_state.summary.copy()
        treated = st.session_state.treated.copy()
        produtos = sorted(fc_all["Produto"].unique().tolist())

        # ---------- RESUMO EXECUTIVO ----------
        st.markdown('<div class="section-title">Resumo executivo</div>', unsafe_allow_html=True)
        st.markdown("<div class='small-muted'>Previsão anual (soma dos 4 trimestres) usando o melhor modelo de cada produto.</div>", unsafe_allow_html=True)
        st.write("")

        cols = st.columns(min(len(produtos), 3))
        for i, prod in enumerate(produtos):
            best = best_model_for(summary, prod)
            fc_best = fc_all[(fc_all["Produto"] == prod) & (fc_all["Modelo"] == best)]
            anual = fc_best["Forecast"].sum()
            anual_sup = fc_best["IC 90% Superior"].sum()
            mape = summary[(summary["Produto"] == prod) & (summary["Modelo"] == best)]["MAPE Agregado %"]
            mape_v = mape.iloc[0] if not mape.empty else np.nan
            with cols[i % len(cols)]:
                st.markdown(f"""<div class="exec-card">
                    <div class="exec-prod">{prod}</div>
                    <div class="exec-row"><span class="exec-label">Previsão anual</span><span class="exec-value">{fmt_br(anual)}</span></div>
                    <div class="exec-row"><span class="exec-label">Limite superior (IC 90%)</span><span class="exec-value">{fmt_br(anual_sup)}</span></div>
                    <div class="exec-row"><span class="exec-label">Melhor modelo</span><span class="exec-model">{best}</span></div>
                    <div style="margin-top:8px">{mape_badge_html(mape_v)}</div>
                </div>""", unsafe_allow_html=True)

        st.write("")
        st.markdown("##### 📅 Previsão por trimestre (melhor modelo de cada produto)")
        st.markdown("<div class='small-muted'>Quanto de cada produto está previsto para os próximos 4 trimestres.</div>", unsafe_allow_html=True)

        # Monta tabela pivô: linhas = produto, colunas = trimestres futuros
        rows = []
        for prod in produtos:
            best = best_model_for(summary, prod)
            fc_best = fc_all[(fc_all["Produto"] == prod) & (fc_all["Modelo"] == best)].sort_values("TrimestreData")
            row = {"Produto": prod, "Modelo": best}
            for _, r in fc_best.iterrows():
                row[r["Trimestre"]] = r["Forecast"]
            row["Total anual"] = fc_best["Forecast"].sum()
            rows.append(row)
        pivot = pd.DataFrame(rows)
        # formata números em PT-BR
        num_cols = [c for c in pivot.columns if c not in ("Produto", "Modelo")]
        fmt_dict = {c: (lambda v: fmt_br(v)) for c in num_cols}
        st.dataframe(pivot.style.format(fmt_dict, na_rep="—"), use_container_width=True)

        st.markdown("---")

        # ---------- VISUALIZAÇÃO POR PRODUTO ----------
        st.markdown('<div class="section-title">Visualização detalhada</div>', unsafe_allow_html=True)
        sel_prod = st.selectbox("Selecione o produto", produtos)
        best = best_model_for(summary, sel_prod)

        st.plotly_chart(plot_history_forecast(sel_prod, treated, fc_all, best), use_container_width=True)
        st.markdown(f"<div class='small-muted'>Gráfico acima: histórico + previsão do modelo escolhido (<strong>{best}</strong>) com faixa de confiança de 90%.</div>", unsafe_allow_html=True)

        st.write("")
        st.plotly_chart(plot_model_comparison(sel_prod, treated, fc_all, best), use_container_width=True)
        st.markdown(f"<div class='small-muted'>Comparação dos 3 modelos. A linha em destaque (mais grossa e sólida) é o modelo <strong>{best}</strong>, escolhido por ter o menor MAPE agregado.</div>", unsafe_allow_html=True)

        st.markdown("---")

        # ---------- TABELA DETALHADA ----------
        st.markdown('<div class="section-title">Tabela de previsões</div>', unsafe_allow_html=True)
        fc = fc_all.copy()
        c1, c2 = st.columns(2)
        prod_filter = c1.selectbox("Filtrar produto", ["Todos"] + produtos, key="fc_prod")
        model_filter = c2.selectbox("Filtrar modelo", ["Todos"] + sorted(fc["Modelo"].unique().tolist()), key="fc_model")
        if prod_filter != "Todos": fc = fc[fc["Produto"] == prod_filter]
        if model_filter != "Todos": fc = fc[fc["Modelo"] == model_filter]
        fc_show = fc.drop(columns=["TrimestreData"], errors="ignore")
        st.dataframe(fc_show.style.format({
            "Forecast": lambda v: fmt_br(v), "IC 90% Inferior": lambda v: fmt_br(v),
            "IC 90% Superior": lambda v: fmt_br(v), "Soma 4T - Limite Superior": lambda v: fmt_br(v),
            "Alpha": "{:.4f}", "Beta": "{:.4f}", "Gamma": "{:.4f}"}, na_rep="—"), use_container_width=True)


# =========================================================
# EXPORTAÇÃO
# =========================================================
elif current == "Exportação":
    st.subheader("⬇️ Exportação Excel")
    if not has_results():
        st.info("Faça o upload de uma base em **Upload** para habilitar a exportação.")
    else:
        fc_export = st.session_state.forecast.drop(columns=["TrimestreData"], errors="ignore")
        sheets = {
            "Base Original": st.session_state.raw_df,
            "Mensal Padronizada": st.session_state.monthly,
            "Trimestral": st.session_state.quarterly.assign(Trimestre=st.session_state.quarterly["Trimestre"].apply(quarter_label)),
            "Tratamento Zeros": st.session_state.treated.assign(Trimestre=st.session_state.treated["Trimestre"].apply(quarter_label)),
            "Modelos MAPE": st.session_state.summary,
            "Auditoria": st.session_state.audit,
            "Forecast": fc_export,
        }
        excel = to_excel_bytes(sheets)
        st.download_button(label="⬇️ Baixar resultado em Excel", data=excel,
                           file_name="LFDA_Forecast_Resultados.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.markdown("<div class='small-muted'>Contém: base original, mensal, trimestral, zeros, MAPE, auditoria e forecast.</div>", unsafe_allow_html=True)
