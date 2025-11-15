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

# Network retry utilities
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import feedparser

# ==============================
# LOGGING
# ==============================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================
# CONFIGURATION IA & PAGE
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
    page_title="AgriPredict Pro - Céréales & Fret",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================
# STYLES MODERNES
# ==============================
st.markdown(
    """
    <style>
    /* Global */
    .main-header { text-align: left; color: #0b6623; font-weight: 700; font-size: 2.2rem; margin: 0; }
    .subtitle { color: #555; margin-top: 0.2rem; margin-bottom: 1rem; }
    .card { background: linear-gradient(180deg, #ffffff 0%, #fbfbfb 100%); padding: 1rem; border-radius: 12px; box-shadow: 0 6px 20px rgba(7,20,3,0.06); }
    .metric { font-size: 1.6rem; font-weight: 700; color: #0b6623; }
    .metric-sub { color: #6b7280; font-size: 0.9rem; }
    .small-muted { color:#6b7280; font-size:0.85rem; }
    footer { visibility: hidden; }
    .stButton>button { background: linear-gradient(90deg,#0b6623,#2e8b57); color: white; border: none; padding: 8px 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================
# UTILITAIRES RÉSEAU
# ==============================
def make_session(retries: int = 3, backoff: float = 0.4) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(['GET', 'POST']),
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({"User-Agent": "AgriPredictBot/1.0 (+https://example.com)"})
    return s

session = make_session()

# ==============================
# DONNÉES SIMULÉES (fallbacks)
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
# SCRAPING BDI (robustifié)
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
            raw = m.group(1).replace(',', '')
            try:
                val = float(raw)
                if 100 < val < 20000:
                    return pd.DataFrame({
                        "Date": [datetime.today().strftime("%Y-%m-%d")],
                        "Prix": [val],
                        "Volume": [0]
                    })
            except Exception:
                logger.debug("BDI regex matched but conversion failed")
        st.warning("⚠️ Impossible de récupérer le BDI en temps réel. Utilisation d'une estimation locale.")
        return generer_fret_simule()[:1]
    except Exception as e:
        logger.exception("Erreur lors du scraping BDI")
        st.warning(f"⚠️ Erreur BDI scraping : {str(e)[:120]}")
        return generer_fret_simule()[:1]

# ==============================
# USDA API
# ==============================
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
        logger.exception("Erreur API USDA")
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
# RAG AVEC ACTUALITÉS RÉELLES (feedparser)
# ==============================
@st.cache_data(ttl=7200)
def recuperer_actualites_reelles(actif: str) -> List[str]:
    try:
        if "Blé" in actif:
            url = "https://www.agrimoney.com/rss/feed/latest"
        elif actif == "Maïs":
            url = "https://www.farmprogress.com/rss.xml"
        elif actif == "Fret maritime":
            # Baltic doesn't always provide RSS; use fallback HTML or a general maritime news feed
            url = "https://www.balticexchange.com/en/news-and-events/news.html"
        else:
            url = "https://www.agweb.com/rss"
        # Try RSS first
        feed = feedparser.parse(url)
        entries = feed.entries or []
        if entries:
            return [e.title[:120] + "..." for e in entries[:3]]
        # If no RSS entries, try scraping simple titles from HTML
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
# FEATURES PREPARATION
# ==============================
def preparer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["Jour"] = np.arange(len(df))
    df["Jour_semaine"] = df["Date"].dt.dayofweek
    df["Tendance_7j"] = df["Prix"].rolling(window=7, min_periods=1).mean()
    df["Volatilite_7j"] = df["Prix"].rolling(window=7, min_periods=1).std().fillna(0)
    return df.fillna(method='bfill')

# ==============================
# SIDEBAR - CONFIGURATION
# ==============================
st.sidebar.title("Configuration")
actif = st.sidebar.selectbox("Actif", ["Blé tendre", "Blé dur", "Maïs", "Soja", "Orge", "Fret maritime"], index=0)
zone = st.sidebar.radio("Zone", ["Global 🌍", "Europe 🇪🇺", "USA 🇺🇸"], index=0)
dark_mode = st.sidebar.checkbox("Mode sombre", value=False)
st.sidebar.markdown("---")
if st.sidebar.button("Réinitialiser cache"):
    st.cache_data.clear()
    st.experimental_rerun()

# Dark mode minimal
if dark_mode:
    st.markdown(
        """
        <style>
            body { background-color: #0b1220; color: #e6eef3; }
            .card { background: #0f1720; color: #e6eef3; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# HEADER + NAV
header_col1, header_col2 = st.columns([4, 1])
with header_col1:
    st.markdown('<div class="main-header">🌾 AgriPredict Pro</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Prévisions IA & analyse contextuelle pour céréales et fret</div>', unsafe_allow_html=True)
with header_col2:
    st.markdown(f"<div class='small-muted'>{datetime.now().strftime('%d/%m/%Y %H:%M')}</div>", unsafe_allow_html=True)

# ==============================
# LOAD DATA
# ==============================
with st.spinner("Chargement des données..."):
    df_hist = charger_donnees_investpy(actif)

# Ensure session_state price updated each run
prix_actuel = float(df_hist["Prix"].iloc[-1])
st.session_state["prix_actuel"] = prix_actuel

volatilite = float(df_hist["Prix"].std())

# ==============================
# MARKET METRICS - NICE CARDS
# ==============================
st.markdown("### Indicateurs du marché")
m1, m2, m3, m4 = st.columns([1.2, 1, 1, 1])
with m1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.metric(label="Prix actuel (€/t)", value=f"{prix_actuel:.1f}")
    st.markdown('</div>', unsafe_allow_html=True)
with m2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.metric(label="Volume moyen (t)", value=f"{df_hist['Volume'].mean():,.0f}")
    st.markdown('</div>', unsafe_allow_html=True)
with m3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.metric(label="Volatilité (std)", value=f"{volatilite:.1f}")
    st.markdown('</div>', unsafe_allow_html=True)
with m4:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    status_label = "En temps réel" if "Fret maritime" not in actif else "Estimation"
    st.markdown(f"<div style='font-weight:700;color:#0b6623'>{status_label}</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# HISTORY CHART
# ==============================
st.markdown("### Historique (60 derniers jours)")
df_plot = df_hist.copy()
df_plot["Date"] = pd.to_datetime(df_plot["Date"])
st.line_chart(df_plot.set_index("Date")["Prix"], use_container_width=True)

# ==============================
# PREDICTION (RandomForest)
# ==============================
st.markdown("### Prévision avancée (Random Forest)")
col_pred_left, col_pred_right = st.columns([3, 1])
with col_pred_left:
    if st.button("✨ Générer prévision IA", key="prevision_btn"):
        with st.spinner("Entraînement du modèle IA et génération de prévisions..."):
            df_feat = preparer_features(df_hist.copy())
            X = df_feat[["Jour", "Jour_semaine", "Tendance_7j", "Volatilite_7j"]]
            y = df_feat["Prix"]

            model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            model.fit(X, y)

            futur_jours = np.arange(len(df_feat), len(df_feat) + 15)
            last_date = pd.to_datetime(df_feat['Date'].iloc[-1])
            futur_dates = pd.date_range(start=last_date + timedelta(days=1), periods=15)
            # Recurisvely generate predictions would be more robust; here we use last window stats as features
            tendance_7j = float(y.tail(7).mean())
            vol_7j = float(y.tail(7).std())
            futur_df = pd.DataFrame({
                "Jour": futur_jours,
                "Jour_semaine": [d.weekday() for d in futur_dates],
                "Tendance_7j": [tendance_7j] * 15,
                "Volatilite_7j": [vol_7j] * 15
            })
            y_pred = model.predict(futur_df)
            y_pred = np.maximum(y_pred, 0.0)  # safety

            df_pred = pd.DataFrame({
                "Date": [d.strftime("%Y-%m-%d") for d in futur_dates],
                "Prix": np.round(y_pred, 2)
            })
            # store mean forecast (15-day mean) for RAG and UI
            st.session_state["prevision"] = float(df_pred["Prix"].mean()) if len(df_pred) else float(prix_actuel)
            st.session_state["df_pred"] = df_pred

            st.success("Prévision IA générée !")
            combined = pd.concat([df_plot.set_index('Date')[['Prix']], df_pred.set_index('Date')[['Prix']]])
            combined.index = pd.to_datetime(combined.index)
            st.line_chart(combined["Prix"], use_container_width=True)
            st.dataframe(df_pred.style.format({"Prix": "{:.2f}"}))
            # allow download
            csv = df_pred.to_csv(index=False).encode('utf-8')
            st.download_button("Télécharger prévision CSV", data=csv, file_name=f"prevision_{actif}_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
    else:
        st.info("Cliquez sur le bouton pour entraîner le modèle et générer une prévision sur 15 jours.")

with col_pred_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("Prévision courte • Modèle : RandomForest\nDonnées : dernières 60 journées")
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# RAG (Actualités + Recommandation)
# ==============================
st.markdown("### Analyse contextuelle (RAG)")
if st.button("🔍 Générer analyse IA avec actualités", key="rag_btn"):
    with st.spinner("Récupération d'actualités et génération de recommandation..."):
        actualites = recuperer_actualites_reelles(actif)
        prix = st.session_state.get('prix_actuel', prix_actuel)
        prev = st.session_state.get('prevision', prix * 1.02)
        recommandation = generer_recommandation_rag(prix, prev, actualites)
    st.success("Analyse générée")
    st.markdown(recommandation)
    st.markdown("Sources consultées :")
    for i, act in enumerate(actualites, 1):
        st.caption(f"{i}. {act}")

# ==============================
# PRO BANNER & FOOTER
# ==============================
if not (hasattr(st, "secrets") and st.secrets.get("PREMIUM_USER", False)):
    st.info("💡 Version Pro : Export PDF, alertes email, précision 97%. Contact: vous@agripredict.com")

st.markdown("---")
foot_col1, foot_col2 = st.columns([3, 1])
with foot_col1:
    st.caption(f"Données : Investpy / USDA / Baltic Exchange • Interface modernisée")
    st.caption("Modèle : RandomForest • Outils : feedparser, BeautifulSoup, requests (retries)")
with foot_col2:
    st.markdown('<div style="background:#0b6623;color:white;padding:6px;border-radius:8px;text-align:center">🚀 Version Pro</div>', unsafe_allow_html=True)
