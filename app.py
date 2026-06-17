# -----------------------------
# CSS — visual claro verde + azul elétrico
# -----------------------------
st.markdown(
    """
    <style>
    :root {
        --bg: #f4f7fb;
        --panel: #ffffff;
        --panel-2: #eef3f9;
        --border: #c3d4e8;
        --text: #0d1f2d;
        --muted: #5a7a96;
        --green: #00c853;
        --green-dark: #00953e;
        --blue: #0057ff;
        --blue-dark: #0040cc;
        --blue-light: #e8f0ff;
        --red: #e53935;
        --yellow: #f9a825;
    }

    .stApp {
        background: linear-gradient(135deg, #eef3f9 0%, #f4f7fb 60%, #e8f0ff 100%);
        color: var(--text);
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 2px solid var(--border);
    }

    [data-testid="stSidebar"] * {
        color: var(--text);
    }

    .main-header {
        padding: 1.25rem 0 1.1rem 0;
        border-bottom: 2px solid var(--blue);
        margin-bottom: 1.2rem;
    }

    .main-title {
        font-size: 2.05rem;
        font-weight: 700;
        letter-spacing: -0.035em;
        color: var(--blue);
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
        box-shadow: 0 0 10px rgba(0, 200, 83, 0.7);
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid var(--border);
        border-top: 3px solid var(--blue);
        border-radius: 14px;
        padding: 1rem 1.05rem;
        min-height: 105px;
        box-shadow: 0 4px 16px rgba(0, 87, 255, 0.08);
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: .06em;
        margin-bottom: .35rem;
    }

    .metric-value {
        color: var(--blue);
        font-size: 1.55rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    .soft-card {
        background: #ffffff;
        border: 1px solid var(--border);
        border-left: 4px solid var(--green);
        border-radius: 14px;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .small-muted { color: var(--muted); font-size: .88rem; }
    .green { color: var(--green); }

    h1, h2, h3 {
        letter-spacing: -0.025em;
        color: var(--blue);
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--border);
        border-top: 3px solid var(--blue);
        border-radius: 14px;
        padding: .8rem 1rem;
        box-shadow: 0 2px 8px rgba(0, 87, 255, 0.07);
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    .stButton>button, .stDownloadButton>button {
        border-radius: 10px;
        border: 2px solid var(--blue);
        background: var(--blue);
        color: #ffffff;
        font-weight: 600;
    }

    .stButton>button:hover, .stDownloadButton>button:hover {
        background: var(--blue-dark);
        border-color: var(--blue-dark);
        color: white;
    }

    [data-testid="stSidebar"] .stRadio label {
        color: var(--text) !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: var(--border);
    }
    </style>
    """,
    unsafe_allow_html=True,
)
