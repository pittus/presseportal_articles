# app_judge_only.py
# Streamlit Mini-Interface (ohne deterministische Checks): Polizei-Meldung -> Kurzartikel (express.de & ksta.de)
# - Liest optional 'input.txt' (gleiches Verzeichnis) oder Text aus Textarea
# - Generiert je Site (express.de, ksta.de) Artikel via LLM mit 3 Few-Shot-Beispielen
# - Prüft NUR via LLM-Judge (Fakten/Stil/Struktur/Länge/Safety)
# - Eine optionale Korrekturrunde basierend auf den Judge-Ergebnissen
# - Zeigt Artikel & Scores nebeneinander, bietet JSON-Download an
#
# Setup
#   pip install streamlit openai python-dotenv
#   export OPENAI_API_KEY=sk-...
# Start
#   streamlit run app_judge_only.py

import os
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
# --- Simple Password Login (ENV) ---
ADMIN_USER = os.getenv("ADMIN_USER", "")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")

def require_login():
    # Falls keine Server-Creds gesetzt sind, klar kommunizieren:
    if not ADMIN_USER or not ADMIN_PASSWORD:
        st.error("Login ist aktiviert, aber ADMIN_USER/ADMIN_PASSWORD fehlen als Env-Variablen.")
        st.stop()

    # Bereits eingeloggt?
    if st.session_state.get("authenticated", False):
        return

    st.title("🔐 Login")
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Benutzername")
        password = st.text_input("Passwort", type="password")
        remember = st.checkbox("Eingeloggt bleiben", value=True)
        submitted = st.form_submit_button("Anmelden")

    if submitted:
        if username == ADMIN_USER and password == ADMIN_PASSWORD:
            st.session_state["authenticated"] = True
            # Optionale Merk-Flag – hier nur als Info genutzt
            st.session_state["remember_me"] = remember
            st.success("Erfolgreich angemeldet.")
            st.experimental_rerun()
        else:
            st.error("Ungültiger Benutzername oder Passwort.")
            # harter Stop, damit der Rest der App nicht rendert
            st.stop()
    else:
        # Login-Form zeigen und Rendering der App stoppen
        st.stop()

# Login jetzt erzwingen (vor allem anderen App-Content)
require_login()

# ---- OpenAI Client ----
try:
    from openai import OpenAI
except Exception as e:
    raise RuntimeError("Bitte 'pip install openai' ausführen.") from e

client = OpenAI()

# Default-Modelle (per Sidebar änderbar)
ARTICLE_MODEL_DEFAULT = os.getenv("ARTICLE_MODEL", "gpt-4o-mini")
JUDGE_MODEL_DEFAULT   = os.getenv("JUDGE_MODEL",   "o3-mini")  

# -------- Style-Profile --------
STYLE_EXPRESS = {
  "site": "express.de",
  "tone": "aktiv, zugespitzt, boulevardesk aber faktentreu; kurze Sätze, klare Verben",
  "headline": { "max_chars": 60, "allow_exclamation": True },
  "length_words": { "min": 50, "max": 160 },
  "structure": ["headline", "lead", "body_paragraphs", "callout_optional?"],
  "facts_policy": "keine neuen Fakten, nur Quelle",
}


STYLE_KSTA = {
  "site": "ksta.de",
  "tone": "nüchtern, sachlich, informativ; neutrale Wortwahl, keine Zuspitzung",
  "headline": { "max_chars": 70, "allow_exclamation": False },
  "length_words": { "min": 100, "max": 220 },
  "structure": ["headline", "teaser", "body_paragraphs", "context_optional?"],
  "facts_policy": "keine neuen Fakten; ausschließlich Inhalte aus der vorliegenden Meldung; behördliche Angaben kenntlich machen",

}

# -------- Few-shot Beispiele (3 je Site) --------
FEWSHOT_EXPRESS = [
  {
    "headline": "Messer-Drama in Bedburg: Heftiger Fund – Haftbefehl",
    "teaser_or_lead": "Nachbarschaftsstreit mit Folgen: 34-Jähriger festgenommen, Messerteil im Körper des Opfers entdeckt.",
    "body_paragraphs": [
      "Die Kölner Polizei nahm am Freitagnachmittag (17. Oktober) in Bedburg-Kirchherten einen 34-jährigen Mann fest. Ihm wird ein versuchtes Tötungsdelikt vorgeworfen.",
      "Der Beschuldigte soll am Dienstag (14. Oktober) seinen 43-jährigen Nachbarn mit einem Messer in den Oberkörper gestochen und schwer verletzt haben. Ein Haftbefehl wurde vollstreckt.",
      "Bewohner eines Mehrfamilienhauses alarmierten gegen 20 Uhr die Polizei. Bei einer Operation entdeckten Ärzte später eine abgebrochene Messerspitze im Oberkörper des Opfers."
    ]
  },
  {
    "headline": "8400 Personen kontrolliert: Mega-Einsatz in Düsseldorf",
    "teaser_or_lead": "Schwerpunktkontrollen bis in die Nacht – Altstadt und Süden im Fokus.",
    "body_paragraphs": [
      "Die Düsseldorfer Polizei nahm am Freitag (10. Oktober 2025) die Gewaltkriminalität ins Visier. Bei einem Großeinsatz wurden rund 8400 Personen kontrolliert.",
      "Besonderes Augenmerk lag auf der Waffenverbotszone in der Altstadt. Ab 20 Uhr kontrollierten Einsatzkräfte an den U-Bahn-Aufgängen am Bolker Stern und tasteten Taschen sowie Rucksäcke ab.",
      "Für genauere Durchsuchungen standen Zelte bereit. Die Maßnahmen dauerten bis in die späte Nacht."
    ]
  },
  {
    "headline": "Vor WM-Quali: Polizei wollte Nationalspieler verhaften",
    "teaser_or_lead": "Beamte stürmen Kabine Nicaraguas in San José kurz vor dem Anpfiff.",
    "body_paragraphs": [
      "Unmittelbar vor dem WM-Qualifikationsspiel zwischen Costa Rica und Nicaragua kam es in San José zu einem Polizeieinsatz in der Umkleide der Gäste.",
      "Ziel war die Verhaftung eines Nationalspielers. Laut Polizei lag ein Gerichtsbeschluss wegen Unterhaltsforderungen vor.",
      "Mehrere Medien berichteten übereinstimmend über den Vorfall, bestätigte Angaben machte Polizeidirektor Marlon Cubillo."
    ]
  }
]


FEWSHOT_KSTA = [
  {
    "headline": "Sieben Verletzte bei Unfall im A57-Herkulestunnel – mutmaßliches Autorennen",
    "teaser_or_lead": "Der Herkulestunnel in Köln war am Samstagabend für mehrere Stunden gesperrt. Sieben Menschen wurden leicht verletzt.",
    "body_paragraphs": [
      "Bei einem mutmaßlichen Autorennen auf der Bundesautobahn 57 in Köln sind am Samstagabend (18. Oktober) sieben Personen leicht verletzt worden. Für Rettungs- und Aufräumarbeiten wurde der Herkulestunnel mehrere Stunden gesperrt.",
      "Nach Angaben von Zeuginnen und Zeugen soll der 22-jährige Fahrer eines grauen Audi RS 3 gegen 23 Uhr zeitweise mit mehr als 170 km/h in Richtung Innenstadt unterwegs gewesen sein. Im Tunnel verlor er demnach die Kontrolle und kollidierte mit einem Renault Captur, den ein 44-Jähriger steuerte.",
      "Im Audi saßen zwei weitere Männer (17 und 21 Jahre), im Renault vier Personen (15, 52, 71 und 77 Jahre). Alle Beteiligten – mit Ausnahme des 17-jährigen Beifahrers – erlitten leichte Verletzungen. Die Unfallaufnahme und Spurensicherung dauerten bis Sonntagmorgen 7.30 Uhr an. Gegen den Audi-Fahrer wird wegen des Verdachts eines verbotenen Kraftfahrzeugrennens ermittelt."
    ]
  },
  {
    "headline": "Steinwurf von Autobahnbrücke: Ehepaar aus Köln auf der A61 unverletzt – Polizei sucht Zeugen",
    "teaser_or_lead": "Ein von einer Brücke geworfener Stein traf am Sonntagnachmittag die Windschutzscheibe eines Pkw. Verletzt wurde niemand.",
    "body_paragraphs": [
      "Ein Kölner Ehepaar ist am Sonntag (19. Oktober) gegen 14.45 Uhr auf der A61 bei Armsheim von einem Stein getroffen worden, der von einer Autobahnbrücke geworfen wurde. Die Windschutzscheibe des Fahrzeugs wurde getroffen; Personen- oder Sachschaden entstand nicht.",
      "Nach Angaben der Polizei standen zwei Kinder auf der Brücke und warfen einen kleinen Stein auf die Fahrbahn. Die Polizei ermittelt wegen des gefährlichen Eingriffs in den Straßenverkehr und bittet Zeuginnen und Zeugen um Hinweise.",
      "Relevante Beobachtungen zur Tatzeit oder Angaben zur Identität der Kinder nimmt jede Polizeidienststelle entgegen. Ermittlungen dauern an."
    ]
  },
  {
    "headline": "Schwerpunktkontrollen in Kölner Altstadt – Waffenverbotszone im Fokus",
    "teaser_or_lead": "Die Polizei führte in den Abendstunden Kontrollen in der Altstadt durch. Ziel war die Überprüfung der Einhaltung der Waffenverbotszone.",
    "body_paragraphs": [
      "Die Polizei Köln hat in den Abendstunden Schwerpunktkontrollen in der Altstadt durchgeführt. Im Mittelpunkt stand die Einhaltung der bestehenden Waffenverbotszone. Laut Polizei wurden Personen stichprobenartig überprüft und mitgeführte Taschen kontrolliert.",
      "Ziel der Maßnahme war die Prävention von Gewaltdelikten und die Stärkung des Sicherheitsempfindens. Die Ergebnisse der Kontrollen wertet die Polizei derzeit aus; zu möglichen Verstößen lagen zunächst keine weiteren Angaben vor.",
      "Die Polizei kündigte an, vergleichbare Kontrollen fortzusetzen. Hinweise aus der Bevölkerung werden entgegengenommen."
    ]
  }
]


# -------- Datenklassen --------
@dataclass
class Article:
    site: str
    headline: str
    teaser_or_lead: str
    body_paragraphs: List[str]
    callout_optional: Optional[str]
    seo_title: str
    meta_description: str
    tags: List[str]
    attribution: Dict[str, Any]
    fact_table: Optional[Dict[str, Any]] = None

@dataclass
class QCResult:
    scores: Dict[str, Any]
    violations: List[str]
    suggested_fixes: List[str]
    decision: str

# -------- Prompting Helpers --------
def chat_completion(model: str, system: str, user: str, temperature: float = None) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]
    # o*-Reasoning-Modelle (o3, o3-mini, o1, etc.) akzeptieren kein temperature
    is_reasoning = model.lower().startswith(("o1", "o3", "o4-mini"))  # ggf. Liste erweitern
    kwargs = {"model": model, "messages": messages}
    if (temperature is not None) and (not is_reasoning):
        kwargs["temperature"] = temperature
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content


def writer_prompt(site_profile: dict, fewshots: List[dict], source_text: str, source_url: str = "") -> Tuple[str, str]:
    system = (
        "Du bist Redakteur für die angegebene Website und hältst dich strikt an das Style-Profile. "
        "Verfasse sachlich korrekte Kurzmeldungen und erfinde keine Fakten.\n\n"
        "STYLE_PROFILE_JSON:\n" + json.dumps(site_profile, ensure_ascii=False, indent=2)
    )

    schema = {
        "site": site_profile["site"],
        "headline": "...",
        "teaser_or_lead": "...",
        "body_paragraphs": ["...","..."],
        "callout_optional": None,
        "seo_title": "...",
        "meta_description": "...",
        "tags": [],
        "attribution": {"source": "Polizei", "source_url": source_url},
        "fact_table": None
    }

    min_w = site_profile["length_words"]["min"]
    max_w = site_profile["length_words"]["max"]
    max_head = site_profile["headline"]["max_chars"]
    allow_excl = site_profile["headline"]["allow_exclamation"]

    user = (
        "Erzeuge aus dem folgenden Polizeitext eine Kurzmeldung NUR als valides JSON im vorgegebenen Schema.\n"
        f"- Wortanzahl gesamt (ohne SEO/Meta): {min_w}-{max_w} Wörter.\n"
        f"- Headline max. {max_head} Zeichen; Ausrufezeichen erlaubt: {str(allow_excl).lower()}.\n"
        "- Nutze ausschließlich bestätigte Inhalte aus der Quelle; keine neuen Fakten, keine Spekulation.\n"
        "- Verwende neutrale, klare Sprache gemäß Style-Profile.\n\n"
        "SCHEMA:\n" + json.dumps(schema, ensure_ascii=False, indent=2) + "\n\n"
        "BEISPIELE (Format & Ton, nicht den Inhalt kopieren):\n"
        + json.dumps(fewshots, ensure_ascii=False, indent=2) + "\n\n"
        "POLIZEITEXT:\n<<<\n" + source_text + "\n>>>\n"
        "Antworte NUR mit JSON, keine Erklärungen."
    )
    return system, user


def judge_prompt(site_profile: dict, article_json: Dict[str, Any], source_text: str) -> Tuple[str, str]:
    system = (
        "Du bist QA-Redakteur:in. Prüfe streng, kurz und binär. "
        "Gib ausschließlich valides JSON gemäß Vorgabe zurück."
    )

    user = (
        "Prüfe den Artikel gegen (1) das site-spezifische STYLE_PROFILE_JSON und (2) den QUELLE_TEXT.\n"
        "Arbeite in zwei Stufen: zuerst MUST-PASS (harte K.O.-Kriterien), dann Scores/Booleans.\n\n"
        "MUST-PASS (harte K.O.-Kriterien – bei Verstoß => decision='human_review'):\n"
        "1) Nur-Quelle-Fakten: Jeder inhaltliche Satz ist im QUELLE_TEXT belegbar (keine neuen Details, keine Spekulationen).\n"
        "2) Unschuldsvermutung: Formulierungen wie 'tatverdächtig', 'laut Polizei', 'nach Angaben der Polizei'.\n"
        "3) Opfer-/Minderjährigenschutz: Keine identifizierenden Details (Namen, exakte Adressen, Kennzeichen, Schulen).\n"
        "4) Attribution vorhanden: Quelle Polizei/Behörde klar benannt (inkl. source_url, falls übergeben).\n"
        "5) Safety: keine Beleidigung/Doxing/Aufruf zu Gewalt/sonstige Moderationsverstöße.\n"
        "6) Headline & Länge regelkonform: Headline ≤ max_chars; wenn im Style verboten, kein '!'; Wortanzahl im Range.\n"
        "7) Pflichtstruktur vollständig: headline, teaser_or_lead, ≥2 body_paragraphs.\n"
        "8) Zahlen/Ort/Zeit konsistent zur Quelle (keine Abweichungen).\n\n"
        "SCORES/Booleans (kompakt halten):\n"
        "- factual_consistency (0..1): 1.0 nur wenn keinerlei Abweichung zur Quelle.\n"
        "- style_match (0..1): Ton/Headline-Regeln/Längenvorgaben der Site erkennbar eingehalten?\n"
        "- length_ok (bool)\n"
        "- structure_ok (bool)\n"
        "- safety_ok (bool)\n\n"
        "STYLE_PROFILE_JSON:\n" + json.dumps(site_profile, ensure_ascii=False, indent=2) + "\n\n"
        "ARTIKEL_JSON:\n" + json.dumps(article_json, ensure_ascii=False, indent=2) + "\n\n"
        "QUELLE_TEXT:\n<<<\n" + source_text + "\n>>>\n\n"
        "Gib NUR dieses JSON zurück (ohne weitere Erklärungen):\n"
        "{\n"
        '  "metrics": {\n'
        '    "headline_length_chars": 0,\n'
        '    "body_word_count": 0\n'
        "  },\n"
        '  "scores": {\n'
        '    "factual_consistency": 0.0,\n'
        '    "style_match": 0.0,\n'
        '    "length_ok": true,\n'
        '    "structure_ok": true,\n'
        '    "safety_ok": true\n'
        "  },\n"
        '  "violations": ["..."],\n'
        '  "suggested_fixes": ["..."],\n'
        '  "decision": "auto_ok" | "revise" | "human_review"\n'
        "}"
    )
    return system, user




# -------- High-level Pipeline ohne deterministische Regeln --------
def generate_article(site_profile: dict, fewshots: List[dict], source_text: str, source_url: str = "") -> Article:
    sys, usr = writer_prompt(site_profile, fewshots, source_text, source_url)
    raw = chat_completion(st.session_state.get("ARTICLE_MODEL", ARTICLE_MODEL_DEFAULT), sys, usr, temperature=0.35)
    try:
        data = json.loads(raw)
    except Exception as e:
        raise ValueError(f"Writer lieferte kein valides JSON:\n{raw}") from e

    # Minimal-Validierung auf Schema-Felder – nur um UI nicht zu crashen
    required = ["site","headline","teaser_or_lead","body_paragraphs","seo_title","meta_description","attribution"]
    for k in required:
        if k not in data:
            raise ValueError(f"Feld '{k}' fehlt im Writer-Output")

    return Article(
        site=data["site"],
        headline=data["headline"].strip(),
        teaser_or_lead=data["teaser_or_lead"].strip(),
        body_paragraphs=[p.strip() for p in data["body_paragraphs"]],
        callout_optional=data.get("callout_optional"),
        seo_title=data["seo_title"].strip(),
        meta_description=data["meta_description"].strip(),
        tags=data.get("tags", []),
        attribution=data.get("attribution", {}),
        fact_table=data.get("fact_table")
    )

def judge_article(site_profile: dict, article: Article, source_text: str) -> QCResult:
    article_json = asdict(article)
    sys, usr = judge_prompt(site_profile, article_json, source_text)
    raw = chat_completion(st.session_state.get("JUDGE_MODEL", JUDGE_MODEL_DEFAULT), sys, usr, temperature=0.0)
    try:
        data = json.loads(raw)
    except Exception as e:
        raise ValueError(f"Judge lieferte kein valides JSON:\n{raw}") from e

    expected_keys = {"scores","violations","suggested_fixes","decision"}
    if not expected_keys.issubset(set(data.keys())):
        raise ValueError(f"Judge-JSON unvollständig: {data.keys()}")

    return QCResult(
        scores=data["scores"],
        violations=data.get("violations", []),
        suggested_fixes=data.get("suggested_fixes", []),
        decision=data["decision"]
    )

# -------- Streamlit UI --------
st.set_page_config(page_title="AI News POC (Judge-only)", layout="wide")
st.title("📰 Polizei-Meldung → Kurzartikel (express.de & ksta.de) – Judge-only")

with st.sidebar:
    st.header("🔧 Einstellungen")
    st.session_state["ARTICLE_MODEL"] = st.text_input("ARTICLE_MODEL", ARTICLE_MODEL_DEFAULT)
    st.session_state["JUDGE_MODEL"]   = st.text_input("JUDGE_MODEL", JUDGE_MODEL_DEFAULT)
    source_url = st.text_input("Quelle (optional)", "")
    load_sample = st.button("Text aus input.txt laden")

if load_sample and Path("input.txt").exists():
    default_text = Path("input.txt").read_text(encoding="utf-8")
else:
    default_text = ""

st.markdown("Kopiere den Originaltext unten hinein und klicke **Generieren**. Es werden **keine** deterministischen Checks ausgeführt – nur der LLM-Judge entscheidet.")

text = st.text_area("Original-Polizeitext", value=default_text, height=300, placeholder="Füge hier die Polizeimeldung ein…")

colA, colB = st.columns([1,1])
with colA:
    gen_btn = st.button("🚀 Generieren")
with colB:
    clear_btn = st.button("🧹 Leeren")

if clear_btn:
    st.session_state.clear()
    st.experimental_rerun()

THRESHOLDS = {
    "factual_consistency": 0.98,
    "style_match": 0.90
}

if gen_btn:
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY fehlt. Bitte als Env-Var setzen oder .env verwenden.")
    elif not text.strip():
        st.warning("Bitte zuerst einen Polizeitext einfügen.")
    else:
        with st.spinner("Erzeuge Artikel & Judge bewertet…"):
            try:
                # 1) Schreiben
                art_ex = generate_article(STYLE_EXPRESS, FEWSHOT_EXPRESS, text, source_url)
                art_ks = generate_article(STYLE_KSTA,    FEWSHOT_KSTA,    text, source_url)

                # 2) Bewerten
                qc_ex  = judge_article(STYLE_EXPRESS, art_ex, text)
                qc_ks  = judge_article(STYLE_KSTA,    art_ks, text)

                # 3) Optional: eine Korrektur-Runde, nur basierend auf Judge-Signalen
                def maybe_revise(site_profile: dict, article: Article, qc: QCResult, source_text: str) -> Tuple[Article, QCResult]:
                    reasons = []
                    s = qc.scores
                    if s.get("factual_consistency", 0) < THRESHOLDS["factual_consistency"]:
                        reasons.append("Faktenkonsistenz erhöhen, ausschließlich bestätigte Inhalte nutzen.")
                    if s.get("style_match", 0) < THRESHOLDS["style_match"]:
                        reasons.append("Stil konsequent an das Style-Profile anpassen (Ton, Länge, Struktur, Headline-Vorgaben).")
                    for k in ["length_ok", "structure_ok", "safety_ok"]:
                        if not s.get(k, True):
                            reasons.append(f"{k} == false korrigieren.")
                    reasons += qc.violations
                    reasons += qc.suggested_fixes

                    if not reasons:
                        return article, qc

                    # Re-Prompt
                    system = (
                        "Du bist Redakteur. Korrigiere den vorhandenen Artikel minimal. "
                        "Erfinde KEINE neuen Fakten. Antworte NUR mit gültigem JSON gemäß Artikelschema.\n\n"
                        "STYLE_PROFILE_JSON:\n" + json.dumps(site_profile, ensure_ascii=False, indent=2)
                    )
                    user = (
                        "AKTUELLER ARTIKEL (JSON):\n" + json.dumps(asdict(article), ensure_ascii=False, indent=2) + "\n\n"
                        "QUELLE:\n<<<\n" + source_text + "\n>>>\n\n"
                        "BEHEBE FOLGENDE PUNKTE (ohne neue Fakten):\n- " + "\n- ".join(reasons) + "\n\n"
                        "Antworte NUR mit gültigem JSON (gleiches Schema)."
                    )
                    raw = chat_completion(st.session_state.get("ARTICLE_MODEL", ARTICLE_MODEL_DEFAULT), system, user, temperature=0.2)
                    try:
                        data = json.loads(raw)
                        fixed = Article(
                            site=data["site"],
                            headline=data["headline"].strip(),
                            teaser_or_lead=data["teaser_or_lead"].strip(),
                            body_paragraphs=[p.strip() for p in data["body_paragraphs"]],
                            callout_optional=data.get("callout_optional"),
                            seo_title=data["seo_title"].strip(),
                            meta_description=data["meta_description"].strip(),
                            tags=data.get("tags", []),
                            attribution=data.get("attribution", {}),
                            fact_table=data.get("fact_table")
                        )
                    except Exception as e:
                        # Wenn das Fixing-JSON nicht valide ist, behalten wir das Original
                        return article, qc

                    qc2 = judge_article(site_profile, fixed, source_text)
                    return fixed, qc2

                art_ex, qc_ex = maybe_revise(STYLE_EXPRESS, art_ex, qc_ex, text)
                art_ks, qc_ks = maybe_revise(STYLE_KSTA,    art_ks, qc_ks, text)

                st.success("Fertig.")

                left, right = st.columns(2)

                def render_block(container, site_label: str, article: Article, qc: QCResult):
                    container.subheader(site_label)
                    with container.expander("⚙️ LLM-Judge Scores", expanded=True):
                        container.write(qc.scores)
                        if qc.violations:
                            container.markdown("**Violations:** " + ", ".join(qc.violations))
                        if qc.suggested_fixes:
                            container.markdown("**Suggested fixes:** " + ", ".join(qc.suggested_fixes))
                        container.markdown(f"**Decision:** `{qc.decision}`")

                    container.markdown(f"### {article.headline}")
                    lead_label = "Lead" if site_label == "express.de" else "Teaser"
                    container.markdown(f"**{lead_label}:** {article.teaser_or_lead}")
                    for p in article.body_paragraphs:
                        container.markdown(p)
                    if article.callout_optional:
                        container.info(article.callout_optional)

                    art_json = json.dumps(asdict(article), ensure_ascii=False, indent=2)
                    container.download_button(
                        label="⬇️ Artikel JSON",
                        file_name=f"article_{site_label.replace('.', '_')}.json",
                        mime="application/json",
                        data=art_json
                    )

                render_block(left,  "express.de", art_ex, qc_ex)
                render_block(right, "ksta.de",    art_ks, qc_ks)

            except Exception as e:
                st.error(f"Fehler: {e}")
