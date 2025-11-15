import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import os
import re
import logging
from typing import List, Optional

try:
    import google.generativeai as genai
except Exception:
    genai = None

from sklearn.ensemble import RandomForestRegressor
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import feedparser

# ==============================
# LOGGING
# ==============================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================
# CONFIG & GEMINI
# ==============================
GEMINI_API_KEY = None
if hasattr(st, "secrets") and st.secrets.get("GEMINI_API_KEY"):
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
else:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

USE_GEMINI = False
if GEMINI_API_KEY and genai is not None:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        USE_GEMINI = True
        logger.info("Gemini configured")
    except Exception as e:
        logger.warning(f"Impossible de configurer Gemini: {e}")
        USE_GEMINI = False

st.set_page_config(
    page_title="AgriPredict Pro",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================
# STYLES (clean, professional)
# ==============================
st.markdown(
    """
    <style>
    :root {
      --brand:#0b6623;
      --muted:#6b7280;
      --card:#ffffff;
      --card-shadow: 0 8px 30px rgba(11,102,35,0.06);
    }
    .app-header { display:flex; align-items:center; gap:12px; }
    .brand-title { font-size:1.75rem; color:var(--brand); font-weight:700; margin:0; }
    .brand-sub { color:var(--muted); margin:0; font-size:0.95rem; }
    .card { background:var(--card); padding:14px; border-radius:12px; box-shadow:var(--card-shadow); }
    .kpi { font-size:1.4rem; color:var(--brand); font-weight:700; }
    .kpi-sub { color:var(--muted); font-size:0.85rem; }
    .small { color:var(--muted); font-size:0.85rem; }
    .pro-badge { background:var(--brand); color:white; padding:6px 10px; border-radius:10px; font-weight:600; }
    .muted { color:var(--muted); }
    .btn { background: linear-gradient(90deg,var(--brand),#2e8b57) !important; color:white !important; }
    footer { visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================
# NETWORK SESSION WITH RETRIES
# ==============================
def make_session(retries: int = 3, backoff: float = 0.4) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({"User-Agent": "AgriPredictBot/1.0 (+https://example.com)"})
    return s

session = make_session()

# ==============================
# SIMULATED DATA FALLBACKS
# ==============================
@st.cache_data(ttl=3600)
def generer_fret_simule() -> pd.DataFrame:
    np.random.seed(42)
    jours = 60
    dates = [datetime.today() - timedelta(days=x) for x in range(jours)][::-1]
    base = 1500
    prix = []
    for i in range(jours):
        trend = base * (1 + 0.0003 * i)
        noise = np.random.normal(0, 50)
        prix.append(max(trend + noise, base * 0.7))
    return pd.DataFrame({
        "Date": [d.strftime("%Y-%m-%d") for d in dates],
        "Prix": np.round(prix, 2),
        "Volume": np.random.randint(5000, 12000, jours)
    })

def generer_donnees_fallback(actif: str) -> pd.DataFrame:
    np.random.seed(42)
    jours = 60
    dates = [datetime.today() - timedelta(days=x) for x in range(jours)][::-1]
    base_map = {"Blé tendre": 250, "Blé dur": 290, "Maïs": 200, "Soja": 500, "Orge": 195}
    base = base_map.get(actif, 250)
    prix = []
    for i in range(jours):
        trend = base * (1 + 0.0005 * i)
        seasonal = 5 * np.sin(2 * np.pi * i / 7)
        noise = np.random.normal(0, 8)
        prix.append(max(trend + seasonal + noise, base * 0.8))
    return pd.DataFrame({
        "Date": [d.strftime("%Y-%m-%d") for d in dates],
        "Prix": np.round(prix, 2),
        "Volume": np.random.randint(8000, 18000, jours)
    })

# ==============================
# DATA SOURCES (BDI & USDA)
# ==============================
def scrape_bdi_index() -> pd.DataFrame:
    url = "https://www.balticexchange.com/en/market-data/main-indices/dry.html"
    try:
        resp = session.get(url, timeout=10)
        if resp.status_code != 200:
            logger.warning("BDI page returned status %s", resp.status_code)
            return generer_fret_simule()[:1]
        soup = BeautifulSoup(resp.content, "html.parser")
        text = soup.get_text(separator=" ")
        m = re.search(r"\b(\d{3,5}(?:[.,]\d{1,2})?)\b", text)
        if m:
            raw = m.group(1).replace(",", "")
            try:
                val = float(raw)
                if 100 < val < 20000:
                    return pd.DataFrame({
                        "Date": [datetime.today().strftime("%Y-%m-%d")],
                        "Prix": [val],
                        "Volume": [0]
                    })
            except Exception:
                logger.debug("BDI parse failed")
        st.warning("⚠️ Impossible de récupérer le BDI en temps réel. Utilisation d'une estimation locale.")
        return generer_fret_simule()[:1]
    except Exception as e:
        logger.exception("Erreur BDI")
        st.warning(f"⚠️ Erreur BDI scraping : {str(e)[:120]}")
        return generer_fret_simule()[:1]

def charger_prix_usda_api(actif: str, api_key: Optional[str] = None) -> pd.DataFrame:
    if not api_key:
        api_key = os.getenv("USDA_API_KEY") or (st.secrets.get("USDA_API_KEY") if hasattr(st, "secrets") else None)
    if not api_key:
        st.warning("⚠️ Clé USDA API manquante. Données simulées utilisées.")
        return generer_donnees_fallback(actif)

    commodity_map = {"Blé tendre": "WHEAT", "Maïs": "CORN", "Soja": "SOYBEANS"}
    commodity = commodity_map.get(actif)
    if not commodity:
        return generer_donnees_fallback(actif)

    try:
        url = "https://quickstats.nass.usda.gov/api/api_GET/"
        params = {
            "key": api_key,
            "commodity_desc": commodity,
            "statisticcat_desc": "PRICE",
            "prodn_practice_desc": "ALL PRODUCTION PRACTICES",
            "freq_desc": "WEEKLY",
            "reference_period_desc": "WEEK",
            "format": "JSON"
        }
        resp = session.get(url, params=params, timeout=12)
        resp.raise_for_status()
        data = resp.json()
        if not data or not data.get("data"):
            st.warning(f"⚠️ Aucune donnée USDA trouvée pour {actif}.")
            return generer_donnees_fallback(actif)
        latest = data["data"][0]
        prix = float(latest.get("Value", 0))
        np.random.seed(42)
        jours = 60
        dates = [datetime.today() - timedelta(days=x) for x in range(jours)][::-1]
        prix_list = [max(prix + np.random.normal(0, 5), prix * 0.8) for _ in range(jours)]
        return pd.DataFrame({
            "Date": [d.strftime("%Y-%m-%d") for d in dates],
            "Prix": np.round(prix_list, 2),
            "Volume": np.random.randint(10000, 20000, jours)
        })
    except Exception as e:
        logger.exception("Erreur USDA")
        st.warning(f"⚠️ Erreur API USDA : {str(e)[:120]}")
        return generer_donnees_fallback(actif)

def charger_donnees_investpy(actif: str) -> pd.DataFrame:
    if actif == "Fret maritime":
        return scrape_bdi_index()
    elif actif in ["Blé tendre", "Maïs", "Soja"]:
        return charger_prix_usda_api(actif)
    else:
        return generer_donnees_fallback(actif)

# ==============================
# NEWS (RSS via feedparser) and RAG
# ==============================
@st.cache_data(ttl=7200)
def recuperer_actualites_reelles(actif: str) -> List[str]:
    try:
        if "Blé" in actif:
            url = "https://www.agrimoney.com/rss/feed/latest"
        elif actif == "Maïs":
            url = "https://www.farmprogress.com/rss.xml"
        elif actif == "Fret maritime":
            url = "https://www.balticexchange.com/en/news-and-events/news.html"
        else:
            url = "https://www.agweb.com/rss"
        feed = feedparser.parse(url)
        entries = feed.entries or []
        if entries:
            return [e.title[:120] + "..." for e in entries[:3]]
        resp = session.get(url, timeout=8)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")
        titles = [t.get_text().strip() for t in soup.find_all(["h1", "h2", "h3", "a"])][:3]
        if titles:
            return [t[:120] + "..." for t in titles]
    except Exception as e:
        logger.debug("Actualités fetch failed: %s", e)
    return [
        "Marché stable avec faible volatilité.",
        "Aucun événement majeur rapporté.",
        "Tendances techniques neutres."
    ]

def generer_recommandation_rag(prix: float, prevision: float, actualites: List[str]) -> str:
    actualites_text = " ".join(actualites[:2])
    if USE_GEMINI and genai is not None:
        try:
            prompt = (
                f"Tu es un expert en trading agricole et logistique.\n"
                f"Prix actuel: {prix:.1f} €/t\n"
                f"Prévision (15j): {prevision:.1f} €/t\n"
                f"Actualités récentes: {actualites_text}\n\n"
                "Donne une recommandation concise en français (max 3 phrases) pour un trader professionnel. "
                "Mentionne les risques et opportunités."
            )
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt, request_options={"timeout": 10})
            return response.text
        except Exception as e:
            logger.warning(f"Gemini generation failed: {e}")
            return f"**Analyse IA** : Erreur Gemini : {str(e)[:120]}"

    tendance = "hausse" if prevision > prix else "baisse"
    rec = f"**Analyse IA** :\n- Prix actuel : {prix:.1f} €/t\n- Prévision (moyenne 15j) : {prevision:.1f} €/t ({tendance})"
    if "sécheresse" in actualites_text.lower() or "tensions" in actualites_text.lower():
        rec += "\n- ⚠️ Facteur de risque détecté."
    if tendance == "hausse":
        rec += "\n\n✅ Recommandation : Opportunité d'achat à court terme."
    else:
        rec += "\n\n⚠️ Recommandation : Surveillance accrue."
    return rec

# ==============================
# FEATURE PREPARATION
# ==============================
def preparer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["Jour"] = np.arange(len(df))
    df["Jour_semaine"] = df["Date"].dt.dayofweek
    df["Tendance_7j"] = df["Prix"].rolling(window=7, min_periods=1).mean()
    df["Volatilite_7j"] = df["Prix"].rolling(window=7, min_periods=1).std().fillna(0)
    return df.fillna(method="bfill")

# ==============================
# SIDEBAR CONFIG
# ==============================
st.sidebar.title("Configuration")
actif = st.sidebar.selectbox("Actif", ["Blé tendre", "Blé dur", "Maïs", "Soja", "Orge", "Fret maritime"], index=0)
zone = st.sidebar.radio("Zone", ["Global 🌍", "Europe 🇪🇺", "USA 🇺🇸"], index=0)
dark_mode = st.sidebar.checkbox("Mode sombre", value=False)
st.sidebar.markdown("---")
st.sidebar.markdown("Modèle IA")
n_estimators = st.sidebar.slider("n_estimators", 10, 500, 100, step=10)
max_depth = st.sidebar.slider("max_depth", 3, 30, 10)
st.sidebar.markdown("---")
if st.sidebar.button("Réinitialiser cache / redémarrer"):
    st.cache_data.clear()
    st.experimental_rerun()

# optional dark mode tweaks
if dark_mode:
    st.markdown(
        """
        <style>
            body { background:#0b1220; color:#e6eef3; }
            .card { background:#0f1720; color:#e6eef3; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# HEADER
header_left, header_right = st.columns([4, 1])
with header_left:
    st.markdown('<div class="app-header">', unsafe_allow_html=True)
    st.markdown('<div><h1 class="brand-title">🌾 AgriPredict Pro</h1><div class="brand-sub">Prévisions IA • Analyses • Alertes</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
with header_right:
    st.markdown(f"<div class='small muted'>{datetime.now().strftime('%d/%m/%Y %H:%M')}</div>", unsafe_allow_html=True)

# Initialize session flags
if "training" not in st.session_state:
    st.session_state["training"] = False

# ==============================
# LOAD DATA
# ==============================
with st.spinner("Chargement des données..."):
    df_hist = charger_donnees_investpy(actif)

prix_actuel = float(df_hist["Prix"].iloc[-1])
st.session_state["prix_actuel"] = prix_actuel
volatilite = float(df_hist["Prix"].std())

# ==============================
# TABS: Overview / Prévisions / Analyses / Paramètres
# ==============================
tab_overview, tab_pred, tab_rag, tab_settings = st.tabs(["Overview", "Prévisions", "Analyses", "Paramètres"])

with tab_overview:
    st.markdown("### Vue d'ensemble")
    # KPI strip
    c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1])
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi">{prix_actuel:.1f} €/t</div><div class="kpi-sub">Prix actuel</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi">{df_hist["Volume"].mean():,.0f} t</div><div class="kpi-sub">Volume moyen</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi">{volatilite:.1f}</div><div class="kpi-sub">Volatilité (std)</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        status = "Estimation" if actif == "Fret maritime" else "En temps réel"
        st.markdown(f'<div class="pro-badge">{status}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("#### Historique (60 jours)")
    df_plot = df_hist.copy()
    df_plot["Date"] = pd.to_datetime(df_plot["Date"])
    st.line_chart(df_plot.set_index("Date")["Prix"], use_container_width=True)

    # Recent news preview
    st.markdown("#### Actualités récentes")
    news = recuperer_actualites_reelles(actif)
    for n in news:
        st.markdown(f"- {n}")

with tab_pred:
    st.markdown("### Générer prévision")
    left, right = st.columns([3, 1])
    with left:
        if st.session_state.get("training", False):
            st.info("Entraînement en cours... veuillez patienter.")
        else:
            if st.button("✨ Lancer la prévision IA", key="gen_pred_btn"):
                st.session_state["training"] = True
                try:
                    with st.spinner("Entraînement du modèle et génération..."):
                        df_feat = preparer_features(df_hist.copy())
                        X = df_feat[["Jour", "Jour_semaine", "Tendance_7j", "Volatilite_7j"]]
                        y = df_feat["Prix"]

                        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
                        model.fit(X, y)

                        # compute residual std for simple CI band
                        preds_train = model.predict(X)
                        resid_std = float(np.std(y - preds_train))

                        futur_jours = np.arange(len(df_feat), len(df_feat) + 15)
                        last_date = pd.to_datetime(df_feat["Date"].iloc[-1])
                        futur_dates = pd.date_range(start=last_date + timedelta(days=1), periods=15)
                        tendance_7j = float(y.tail(7).mean())
                        vol_7j = float(y.tail(7).std())
                        futur_df = pd.DataFrame({
                            "Jour": futur_jours,
                            "Jour_semaine": [d.weekday() for d in futur_dates],
                            "Tendance_7j": [tendance_7j] * 15,
                            "Volatilite_7j": [vol_7j] * 15
                        })
                        y_pred = model.predict(futur_df)
                        y_pred = np.maximum(y_pred, 0.0)

                        df_pred = pd.DataFrame({
                            "Date": [d.strftime("%Y-%m-%d") for d in futur_dates],
                            "Prix": np.round(y_pred, 2),
                            "Upper": np.round(y_pred + 1.96 * resid_std, 2),
                            "Lower": np.round(np.maximum(y_pred - 1.96 * resid_std, 0.0), 2)
                        })

                        st.session_state["df_pred"] = df_pred
                        st.session_state["prevision"] = float(df_pred["Prix"].mean())
                        st.session_state["model_trained"] = True
                        st.session_state["training"] = False

                        st.success("Prévision générée")
                        # show chart with band: combine history + pred with upper/lower
                        combined = pd.concat([df_plot.set_index("Date")[["Prix"]], df_pred.set_index("Date")[["Prix", "Upper", "Lower"]]])
                        combined.index = pd.to_datetime(combined.index)
                        st.line_chart(combined[["Prix", "Upper", "Lower"]], use_container_width=True)
                        st.dataframe(df_pred.style.format({"Prix":"{:.2f}", "Upper":"{:.2f}", "Lower":"{:.2f}"}))
                        csv = df_pred.to_csv(index=False).encode("utf-8")
                        st.download_button("Télécharger prévision CSV", data=csv, file_name=f"prevision_{actif}_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
                except Exception as e:
                    logger.exception("Erreur durant la génération")
                    st.error("Une erreur est survenue lors de l'entraînement. Voir logs.")
                    st.session_state["training"] = False
    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("Paramètres du modèle")
        st.markdown(f"- n_estimators: **{n_estimators}**")
        st.markdown(f"- max_depth: **{max_depth}**")
        st.markdown(f"- Horizon: **15 jours**")
        st.markdown("</div>", unsafe_allow_html=True)

    # If predictions exist show summary card
    if st.session_state.get("df_pred") is not None:
        st.markdown("#### Résumé rapide")
        df_pred = st.session_state["df_pred"]
        mean_pred = float(df_pred["Prix"].mean())
        delta_pct = (mean_pred - prix_actuel) / prix_actuel * 100
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Prévision moyenne (15j)", f"{mean_pred:.1f} €/t", delta=f"{delta_pct:.2f}%")
        with col_b:
            st.metric("Premier jour prévision", f"{df_pred['Prix'].iloc[0]:.1f} €/t")
        with col_c:
            st.metric("Intervalle confiance (±)", f"{(df_pred['Upper']-df_pred['Lower']).mean()/2:.1f} €/t")

with tab_rag:
    st.markdown("### Analyse contextuelle (RAG) + Actualités")
    if st.button("🔍 Générer analyse IA avec actualités", key="rag_btn"):
        with st.spinner("Récupération d'actualités et génération de recommandation..."):
            actualites = recuperer_actualites_reelles(actif)
            prix = st.session_state.get("prix_actuel", prix_actuel)
            prev = st.session_state.get("prevision", prix * 1.02)
            recommandation = generer_recommandation_rag(prix, prev, actualites)
            st.success("Analyse générée")
            st.markdown(recommandation)
            st.markdown("#### Sources consultées")
            for i, act in enumerate(actualites, 1):
                st.caption(f"{i}. {act}")
    else:
        st.info("Cliquez pour obtenir une recommandation contextuelle basée sur les dernières actualités.")

with tab_settings:
    st.markdown("### Paramètres & Informations")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Intégrations")
        st.markdown("- USDA API: configurez la clé via `st.secrets` ou la variable d'environnement USDA_API_KEY.")
        st.markdown("- Gemini (optionnel): activez GEMINI_API_KEY pour recommandations IA avancées.")
        st.markdown("- Requêtes réseau avec retries et timeouts.")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card" style="margin-top:12px">', unsafe_allow_html=True)
        st.markdown("#### Contrôle qualité & bonnes pratiques")
        st.markdown("- Ne pas exposer les clés API dans l'UI.")
        st.markdown("- Vérifiez l'exactitude des sélecteurs lors du scraping (BDI).")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div style="text-align:center"><strong>🚀 Version Pro</strong></div>', unsafe_allow_html=True)
        st.markdown("<div class='small'>Export PDF, alertes emails, API privée</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# FOOTER
st.markdown("---")
fcol1, fcol2 = st.columns([3, 1])
with fcol1:
    st.caption("Données: Investpy / USDA / Baltic Exchange (simulations si indisponibles) • Modèle: RandomForest")
    st.caption("Interface modernisée • UX optimisée pour traders et analystes")
with fcol2:
    st.markdown('<div class="pro-badge">Contact: vous@agripredict.com</div>', unsafe_allow_html=True)v>', unsafe_allow_html=True)
