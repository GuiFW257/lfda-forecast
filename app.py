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
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# CSS — visual corporativo escuro
# -----------------------------
st.markdown(
    """
    <style>
    :root {
        --bg: #0b1117;
        --panel: #111a22;
        --panel-2: #151f29;
        --border: #263544;
        --text: #e8edf2;
        --muted: #9aa8b5;
        --green: #2bd17e;
        --green-dark: #138a51;
        --red: #ff6b6b;
        --yellow: #f7c948;
    }

    .stApp {
        background: radial-gradient(circle at top left, #13202a 0%, #0b1117 36%, #070b10 100%);
        color: var(--text);
    }

    [data-testid="stSidebar"] {
        background: #080d12;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] * {
        color: var(--text);
    }

    .main-header {
        padding: 1.25rem 0 1.1rem 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 1.2rem;
    }

    .main-title {
        font-size: 2.05rem;
        font-weight: 700;
        letter-spacing: -0.035em;
        color: var(--text);
        margin-bottom: 0.15rem;
    }

    .main-subtitle {
        font-size: 1.02rem;
        color: var(--muted);
        margin-bottom: 0.45rem;
    }

    .navline {
        color: var(--muted);
        font-size: 0.88rem;
    }

    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--green);
        margin-right: 6px;
        box-shadow: 0 0 12px rgba(43, 209, 126, 0.75);
    }

    .metric-card {
        background: linear-gradient(180deg, rgba(21,31,41,.98), rgba(17,26,34,.98));
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1rem 1.05rem;
        min-height: 105px;
        box-shadow: 0 12px 28px rgba(0,0,0,.18);
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: .06em;
        margin-bottom: .35rem;
    }

    .metric-value {
        color: var(--text);
        font-size: 1.55rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    .soft-card {
        background: rgba(17, 26, 34, 0.92);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .small-muted { color: var(--muted); font-size: .88rem; }
    .green { color: var(--green); }

    h1, h2, h3 { letter-spacing: -0.025em; }
    div[data-testid="stMetric"] {
        background: rgba(17, 26, 34, .90);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: .8rem 1rem;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 12px;
    }
    .stButton>button, .stDownloadButton>button {
        border-radius: 10px;
        border: 1px solid var(--green-dark);
        background: #0e2d1d;
        color: #eafff3;
        font-weight: 600;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        border: 1px solid var(--green);
        color: white;
    }
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
        <div class="navline">Upload • Auditoria • Modelos • Forecast • Resultados&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="status-dot"></span>Processamento local</div>
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

    # fallback para quantidade: coluna numérica com maior preenchimento
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

    # formatos tipo 03 2026 ou 3 2026
    m = re.match(r"^(\d{1,2})\s+(\d{4})$", raw)
    if m:
        month = int(m.group(1))
        year = int(m.group(2))
        if 1 <= month <= 12:
            return pd.Timestamp(year=year, month=month, day=1)

    # formatos tipo marco 2026
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


def prepare_monthly(df: pd.DataFrame, product_col: str, date_col: str, qty_col: str) -> pd.DataFrame:
    out = df[[product_col, date_col, qty_col]].copy()
    out.columns = ["Produto", "Data", "Quantidade"]
    out["Produto"] = out["Produto"].astype(str).str.strip()
    out["Data"] = out["Data"].apply(parse_month_date)
    out["Quantidade"] = pd.to_numeric(out["Quantidade"], errors="coerce").fillna(0.0)
    out = out.dropna(subset=["Data"])
    out = out[out["Produto"].notna() & (out["Produto"].astype(str).str.len() > 0)]
    out = out.groupby(["Produto", "Data"], as_index=False)["Quantidade"].sum()
    return out.sort_values(["Produto", "Data"])


def monthly_to_quarterly(monthly: pd.DataFrame) -> pd.DataFrame:
    q = monthly.copy()
    q["Trimestre"] = q["Data"].dt.to_period("Q").dt.to_timestamp()
    q = q.groupby(["Produto", "Trimestre"], as_index=False)["Quantidade"].sum()
    return q.sort_values(["Produto", "Trimestre"])


def fill_missing_quarters(qdf: pd.DataFrame) -> pd.DataFrame:
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


def treat_zeros_preserve_total(values: np.ndarray) -> np.ndarray:
    """
    Suaviza sequências internas de zeros entre dois valores positivos,
    preservando a soma total do bloco.

    Exemplo: 120, 0, 0, 120 -> 60, 60, 60, 60
    Os zeros recebem média dos extremos dividida pelos espaços;
    os extremos cedem metade dessa carga proporcionalmente.
    """
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
                total_to_zeros = avg_extremes
                per_zero = total_to_zeros / k
                y[start_zero:i] = per_zero
                reduction_total = total_to_zeros
                denom = y[left] + y[right]
                if denom > 0:
                    y[left] = max(0.0, y[left] - reduction_total * (y[left] / denom))
                    y[right] = max(0.0, y[right] - reduction_total * (y[right] / denom))
        else:
            i += 1
    return y


def split_train_test(y: pd.Series) -> Tuple[pd.Series, pd.Series, str]:
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


def safe_mape(actual, pred) -> float:
    actual = np.array(actual, dtype=float)
    pred = np.array(pred, dtype=float)
    mask = actual != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100)


def agg_mape(actual, pred) -> float:
    actual_sum = float(np.sum(actual))
    pred_sum = float(np.sum(pred))
    if actual_sum == 0:
        return np.nan
    return abs(actual_sum - pred_sum) / abs(actual_sum) * 100


def model_params(fit) -> Dict[str, Optional[float]]:
    params = getattr(fit, "params", {}) or {}
    return {
        "Alpha": params.get("smoothing_level", np.nan),
        "Beta": params.get("smoothing_trend", np.nan),
        "Gamma": params.get("smoothing_seasonal", np.nan),
    }


def fit_forecast_model(model_name: str, train: pd.Series, steps: int):
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
                y,
                trend="add",
                seasonal="add",
                seasonal_periods=4,
                initialization_method="estimated",
            ).fit(optimized=True)
        else:
            # fallback: sem sazonalidade quando há poucos trimestres
            fit = Holt(y, initialization_method="estimated").fit(optimized=True)
    else:
        raise ValueError("Modelo inválido")

    return fit, fit.forecast(steps)


def future_quarters(last_quarter: pd.Timestamp, periods: int = 4) -> List[pd.Timestamp]:
    start = pd.Period(last_quarter, freq="Q") + 1
    return list(pd.period_range(start, periods=periods, freq="Q").to_timestamp())


def quarter_label(ts) -> str:
    p = pd.Period(ts, freq="Q")
    return f"{p.year}T{p.quarter}"


@dataclass
class ProductResult:
    model_summary: pd.DataFrame
    audit: pd.DataFrame
    forecast: pd.DataFrame


def run_models(q: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

        models = ["SES", "Holt", "Holt-Winters"]
        for model_name in models:
            test_steps = len(test)
            try:
                fit_bt, pred_test = fit_forecast_model(model_name, train, max(test_steps, 1))
                pred_test = pd.Series(pred_test).iloc[:test_steps].values if test_steps > 0 else np.array([])

                period_mape = safe_mape(test.values, pred_test) if test_steps else np.nan
                aggregate_mape = agg_mape(test.values, pred_test) if test_steps else np.nan
                params = model_params(fit_bt) if fit_bt is not None else {"Alpha": np.nan, "Beta": np.nan, "Gamma": np.nan}

                # Auditoria do teste
                for idx, dt in enumerate(test.index):
                    actual_original = float(g.loc[g["Trimestre"] == dt, "Quantidade"].iloc[0])
                    actual_treated = float(test.iloc[idx])
                    pred = float(pred_test[idx]) if idx < len(pred_test) else np.nan
                    err = actual_treated - pred if pd.notna(pred) else np.nan
                    mape_i = abs(err / actual_treated) * 100 if actual_treated != 0 and pd.notna(err) else np.nan
                    audits.append({
                        "Produto": prod,
                        "Modelo": model_name,
                        "Tipo": "Teste",
                        "Trimestre": quarter_label(dt),
                        "Valor Real": actual_original,
                        "Valor Tratado": actual_treated,
                        "Valor Previsto": pred,
                        "Erro": err,
                        "MAPE %": mape_i,
                        "Alpha": params["Alpha"],
                        "Beta": params["Beta"],
                        "Gamma": params["Gamma"],
                        "Regra Treino/Teste": split_rule,
                    })

                summaries.append({
                    "Produto": prod,
                    "Modelo": model_name,
                    "Períodos": len(y),
                    "Treino": len(train),
                    "Teste": len(test),
                    "Regra": split_rule,
                    "MAPE Período Médio %": period_mape,
                    "MAPE Agregado %": aggregate_mape,
                    "Alpha": params["Alpha"],
                    "Beta": params["Beta"],
                    "Gamma": params["Gamma"],
                })

                # Forecast final com série completa tratada
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
                    pred = max(0.0, float(fc[h - 1]))
                    interval = z90 * residual_std * np.sqrt(h)
                    forecasts.append({
                        "Produto": prod,
                        "Modelo": model_name,
                        "Trimestre": quarter_label(dt),
                        "Forecast": pred,
                        "IC 90% Inferior": max(0.0, pred - interval),
                        "IC 90% Superior": pred + interval,
                        "Alpha": params_full["Alpha"],
                        "Beta": params_full["Beta"],
                        "Gamma": params_full["Gamma"],
                    })

            except Exception as exc:
                summaries.append({
                    "Produto": prod,
                    "Modelo": model_name,
                    "Períodos": len(y),
                    "Treino": len(train),
                    "Teste": len(test),
                    "Regra": split_rule,
                    "MAPE Período Médio %": np.nan,
                    "MAPE Agregado %": np.nan,
                    "Alpha": np.nan,
                    "Beta": np.nan,
                    "Gamma": np.nan,
                    "Erro Modelo": str(exc),
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


def to_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for name, df in sheets.items():
            safe_name = name[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)
            workbook = writer.book
            worksheet = writer.sheets[safe_name]
            header_fmt = workbook.add_format({
                "bold": True,
                "font_color": "#FFFFFF",
                "bg_color": "#111A22",
                "border": 1,
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
st.sidebar.markdown("### LFDA Forecast")
st.sidebar.markdown("<span class='small-muted'>Ambiente local • Streamlit</span>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Menu",
    ["Upload", "Validação", "Modelos", "Auditoria", "Forecast", "Exportação"],
    index=0,
)

st.sidebar.divider()
st.sidebar.markdown("**Configuração atual**")
st.sidebar.caption("Modelos: SES, Holt e Holt-Winters")
st.sidebar.caption("Parâmetros: otimização automática Python")
st.sidebar.caption("Forecast: próximos 4 trimestres")
st.sidebar.caption("Intervalo de confiança: 90%")


# =========================================================
# Upload + processamento global
# =========================================================
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "monthly" not in st.session_state:
    st.session_state.monthly = None
if "quarterly" not in st.session_state:
    st.session_state.quarterly = None
if "treated" not in st.session_state:
    st.session_state.treated = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "audit" not in st.session_state:
    st.session_state.audit = None
if "forecast" not in st.session_state:
    st.session_state.forecast = None
if "inferred_cols" not in st.session_state:
    st.session_state.inferred_cols = None


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


def has_results() -> bool:
    return st.session_state.summary is not None and not st.session_state.summary.empty


# =========================================================
# Pages
# =========================================================
if page == "Upload":
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

        st.markdown("### Melhor modelo por produto — critério atual: menor MAPE agregado")
        st.dataframe(best, use_container_width=True)

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
            label="Baixar resultado em Excel",
            data=excel,
            file_name="LFDA_Forecast_Resultados.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.markdown("<div class='small-muted'>O arquivo contém base original, base mensal, trimestral, tratamento dos zeros, MAPE, auditoria e forecast.</div>", unsafe_allow_html=True)
