#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# AI News POC (Chat Completions + JSON-mode, Claim-Checking Judge)
# - Writer & Judge via Chat Completions API + response_format=json_object
# - Evidenzbasierter Judge mit Claim-Abgleich & Coverage-Ratio
# - Optional 1× Auto-Revise (Sidebar)
# - Workflow-Tab mit Graphviz (optional)

import os
import json
import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

# ---------------- Auth (simple) ----------------
ADMIN_USER = os.getenv("ADMIN_USER", "")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")

def require_login():
    if not ADMIN_USER or not ADMIN_PASSWORD:
        st.error("Login ist aktiviert, aber ADMIN_USER/ADMIN_PASSWORD fehlen als Env-Variablen.")
        st.stop()
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
            st.session_state["remember_me"] = remember
            st.success("Erfolgreich angemeldet.")
            st.experimental_rerun()
        else:
            st.error("Ungültiger Benutzername oder Passwort.")
            st.stop()
    else:
        st.stop()

require_login()

# ---------------- Optional: Graphviz ----------------
try:
    from graphviz import Digraph
    HAS_GRAPHVIZ = True
except Exception:
    HAS_GRAPHVIZ = False

def build_flow_graph():
    if not HAS_GRAPHVIZ:
        return None
    dot = Digraph(comment="POC Flow")
    dot.attr(rankdir="LR", fontsize="10")
    dot.node("A", "Textarea-Text", shape="box")
    dot.node("B", "generate_article(EXPRESS)", shape="box")
    dot.node("C", "Article_ex", shape="oval")
    dot.node("D", "generate_article(KSTA)", shape="box")
    dot.node("E", "Article_ks", shape="oval")
    dot.node("F", "judge_article(EXPRESS, Article_ex)", shape="box")
    dot.node("G", "judge_article(KSTA, Article_ks)", shape="box")
    dot.node("H", "maybe_revise(EXPRESS, Article_ex, QC_ex)", shape="diamond")
    dot.node("I", "maybe_revise(KSTA, Article_ks, QC_ks)", shape="diamond")
    dot.node("J", "(Article_ex2?, QC_ex2?)", shape="oval")
    dot.node("K", "(Article_ks2?, QC_ks2?)", shape="oval")
    dot.node("L", "Render UI + Downloads", shape="box")
    dot.edges(["AB", "BC"])
    dot.edges(["AD", "DE"])
    dot.edge("C", "F")
    dot.edge("E", "G")
    dot.edge("F", "H")
    dot.edge("G", "I")
    dot.edge("H", "J", label="optional")
    dot.edge("I", "K", label="optional")
    for src in ("F", "G", "J", "K"):
        dot.edge(src, "L")
    return dot

# ---------------- OpenAI Client (Chat Completions) ----------------
try:
    from openai import OpenAI
except Exception as e:
    raise RuntimeError("Bitte 'pip install openai' ausführen.") from e

client = OpenAI()

# Models (per Sidebar änderbar)
ARTICLE_MODEL_DEFAULT = os.getenv("ARTICLE_MODEL", "gpt-4.1-mini")
JUDGE_MODEL_DEFAULT   = os.getenv("JUDGE_MODEL",   "o4-mini")  # reasoning: keine temperature

# ---------------- Styles & Few-shots ----------------
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
      "Im Audi saßen zwei weitere Männer (17 und 21 Jahre), im Renault vier Personen (15, 52, 71 und 77 Jahre). Alle Beteiligten – mit Ausnahme des 17-jährigen Beifahrers – erlitten leichte Verletzungen."
    ]
  },
  {
    "headline": "Steinwurf von Autobahnbrücke: Ehepaar aus Köln auf der A61 unverletzt – Polizei sucht Zeugen",
    "teaser_or_lead": "Ein von einer Brücke geworfener Stein traf am Sonntagnachmittag die Windschutzscheibe eines Pkw. Verletzt wurde niemand.",
    "body_paragraphs": [
      "Ein Kölner Ehepaar ist am Sonntag (19. Oktober) gegen 14.45 Uhr auf der A61 bei Armsheim von einem Stein getroffen worden, der von einer Autobahnbrücke geworfen wurde.",
      "Nach Angaben der Polizei standen zwei Kinder auf der Brücke und warfen einen kleinen Stein auf die Fahrbahn. Die Polizei bittet Zeuginnen und Zeugen um Hinweise."
    ]
  }
]

# ---------------- Data classes ----------------
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
    metrics: Optional[Dict[str, Any]] = None

# ---------------- Helpers ----------------
JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}\s*$")

def coerce_json(text: str) -> Dict[str, Any]:
    """Nimmt den letzten JSON-Block und ersetzt typ. Anführungszeichen."""
    candidate = (text or "").strip()
    m = JSON_BLOCK_RE.search(candidate)
    if m:
        candidate = m.group(0)
    candidate = candidate.replace("“", "\"").replace("”", "\"").replace("’", "'")
    return json.loads(candidate)

def is_reasoning_model(name: str) -> bool:
    n = (name or "").lower()
    return n.startswith(("o1", "o3", "o4"))

def chat_call(model: str, system: str, user: str, force_json: bool = True) -> str:
    """
    Chat Completions API mit optionalem JSON-Mode (response_format=json_object).
    Hinweis: Structured Outputs für Chat sind offiziell dokumentiert. (OpenAI Docs)
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    kwargs = {"model": model, "messages": messages}
    if force_json:
        # JSON mode (Structured Outputs)
        kwargs["response_format"] = {"type": "json_object"}
    # keine temperature für o*-Reasoning-Modelle
    if not is_reasoning_model(model):
        kwargs["temperature"] = 0
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content

# ---------------- Prompts ----------------
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
        "Du bist QA-Redakteur:in. Prüfe streng, evidenzbasiert und konservativ. "
        "Antworte ausschließlich mit VALIDE(M) JSON nach Vorgabe."
    )
    user = (
        "Prüfe den ARTIKEL_JSON gegen (1) das STYLE_PROFILE_JSON und (2) den QUELLE_TEXT.\n"
        "Arbeite evidenzbasiert und protokollierbar:\n"
        "1) Extrahiere prüfbare Kernaussagen (claims) aus Headline/Teaser/Body (keine SEO/Meta).\n"
        "2) Weise JEDEM Claim ein Beleg-Zitat (quote) aus QUELLE_TEXT zu ODER markiere 'unbelegt' / 'abweichung'.\n"
        "3) Berechne coverage_ratio = belegte_claims / gesamt_claims (auf 2 Dezimalstellen runden).\n\n"
        "MUST-PASS (→ decision='human_review' bei Verstoß):\n"
        "- Nur-Quelle-Fakten & korrekte Attribution (z. B. 'laut Polizei').\n"
        "- Unschuldsvermutung; keine Vorverurteilung.\n"
        "- Schutz Persönlichkeitsrechte/Minderjährige (keine identifizierenden Details).\n"
        "- Keine Diskriminierung; sensible Merkmale nur bei Erforderlichkeit.\n"
        "- Kein entwürdigender Sensationsstil; Boulevardton bei express.de ok, aber respektvoll.\n"
        "- Formale Regeln: Headline ≤ max_chars; '!' nur wenn erlaubt; Wortanzahl im Range; Pflichtstruktur; Attribution inkl. source/source_url; Zahlen/Ort/Zeit konsistent.\n\n"
        "Scoring:\n"
        "- factual_consistency: 1.0 nur bei coverage_ratio=1.00 und keinen Abweichungen/Verstößen; sonst strikte Abzüge.\n"
        "- style_match: 1.0 nur wenn alle Stil-/Headline-Regeln exakt eingehalten wurden.\n\n"
        "STYLE_PROFILE_JSON:\n" + json.dumps(site_profile, ensure_ascii=False, indent=2) + "\n\n"
        "ARTIKEL_JSON:\n" + json.dumps(article_json, ensure_ascii=False, indent=2) + "\n\n"
        "QUELLE_TEXT:\n<<<\n" + source_text + "\n>>>\n\n"
        "Gib NUR folgendes JSON zurück:\n"
        "{\n"
        '  "metrics": {\n'
        '    "headline_length_chars": 0,\n'
        '    "body_word_count": 0,\n'
        '    "coverage_ratio": 0.00,\n'
        '    "checked_claims": [\n'
        '      {"claim":"...", "status":"belegt|unbelegt|abweichung", "quote":"", "note":""}\n'
        '    ]\n'
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

# ---------------- High-level funcs ----------------
def generate_article(site_profile: dict, fewshots: List[dict], source_text: str, source_url: str = "") -> Article:
    sys, usr = writer_prompt(site_profile, fewshots, source_text, source_url)
    raw = chat_call(st.session_state.get("ARTICLE_MODEL", ARTICLE_MODEL_DEFAULT), sys, usr, force_json=True)
    try:
        data = coerce_json(raw)
    except Exception as e:
        raise ValueError(f"Writer lieferte kein valides JSON:\n{raw}") from e

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
    raw = chat_call(st.session_state.get("JUDGE_MODEL", JUDGE_MODEL_DEFAULT), sys, usr, force_json=True)
    try:
        data = coerce_json(raw)
    except Exception as e:
        raise ValueError(f"Judge lieferte kein valides JSON:\n{raw}") from e

    return QCResult(
        scores=data.get("scores", {}),
        violations=data.get("violations", []),
        suggested_fixes=data.get("suggested_fixes", []),
        decision=data.get("decision", "human_review"),
        metrics=data.get("metrics")
    )

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="AI News POC (Chat Completions + JSON)", layout="wide")
st.title("📰 Polizei-Meldung → Kurzartikel (express.de & ksta.de)")

tab_app, tab_flow = st.tabs(["✍️ Generator", "🧭 Workflow"])

with tab_app:
    with st.sidebar:
        st.header("🔧 Einstellungen")
        st.session_state["ARTICLE_MODEL"] = st.text_input("ARTICLE_MODEL", ARTICLE_MODEL_DEFAULT)
        st.session_state["JUDGE_MODEL"]   = st.text_input("JUDGE_MODEL", JUDGE_MODEL_DEFAULT)
        enable_revise = st.checkbox("Auto-Revise 1×", value=False)
        load_sample = st.button("Text aus input.txt laden")

    if load_sample and Path("input.txt").exists():
        default_text = Path("input.txt").read_text(encoding="utf-8")
    else:
        default_text = ""

    st.markdown("Füge unten die Polizeimeldung ein und klicke **Generieren**. QS läuft über den LLM-Judge (Claim-Abgleich).")
    text = st.text_area("Original-Polizeitext", value=default_text, height=300, placeholder="…")

    colA, colB = st.columns([1,1])
    with colA:
        gen_btn = st.button("🚀 Generieren")

    THRESHOLDS = {"factual_consistency": 0.98, "style_match": 0.90}

    if gen_btn:
        if not os.getenv("OPENAI_API_KEY"):
            st.error("OPENAI_API_KEY fehlt. Bitte als Env-Var setzen oder .env verwenden.")
        elif not text.strip():
            st.warning("Bitte zuerst einen Polizeitext einfügen.")
        else:
            with st.spinner("Erzeuge Artikel & Judge bewertet…"):
                try:
                    # 1) Schreiben
                    art_ex = generate_article(STYLE_EXPRESS, FEWSHOT_EXPRESS, text)
                    art_ks = generate_article(STYLE_KSTA,    FEWSHOT_KSTA,    text)

                    # 2) Bewerten
                    qc_ex  = judge_article(STYLE_EXPRESS, art_ex, text)
                    qc_ks  = judge_article(STYLE_KSTA,    art_ks, text)

                    # 3) Optional: 1× Revise
                    def maybe_revise(site_profile: dict, article: Article, qc: QCResult, source_text: str) -> Tuple[Article, QCResult]:
                        reasons = []
                        s = qc.scores or {}
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
                        raw = chat_call(st.session_state.get("ARTICLE_MODEL", ARTICLE_MODEL_DEFAULT), system, user, force_json=True)
                        try:
                            data = coerce_json(raw)
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
                        except Exception:
                            return article, qc
                        qc2 = judge_article(site_profile, fixed, source_text)
                        return fixed, qc2

                    if enable_revise:
                        art_ex, qc_ex = maybe_revise(STYLE_EXPRESS, art_ex, qc_ex, text)
                        art_ks, qc_ks = maybe_revise(STYLE_KSTA,    art_ks, qc_ks, text)

                    st.success("Fertig.")

                    left, right = st.columns(2)

                    def pct(x: Optional[float]) -> str:
                        try:
                            return f"{x*100:.0f}%"
                        except Exception:
                            return "-"

                    def ratio_to_pct(v) -> str:
                        try:
                            f = float(v)
                            return f"{f*100:.0f}%"
                        except Exception:
                            return "-"

                    def render_block(container, site_label: str, article: Article, qc: QCResult):
                        container.subheader(site_label)
                        exp = container.expander("⚙️ LLM-Judge (Scores & Details)", expanded=False)
                        with exp:
                            s = qc.scores or {}
                            c1, c2, c3, c4, c5 = exp.columns(5)
                            c1.metric("Factual",  pct(s.get("factual_consistency")))
                            c2.metric("Style",    pct(s.get("style_match")))
                            c3.metric("LengthOK", "✅" if s.get("length_ok") else "❌")
                            c4.metric("StructOK", "✅" if s.get("structure_ok") else "❌")
                            c5.metric("SafetyOK", "✅" if s.get("safety_ok") else "❌")

                            if qc.metrics:
                                hl = qc.metrics.get("headline_length_chars")
                                bw = qc.metrics.get("body_word_count")
                                cr = qc.metrics.get("coverage_ratio")
                                if (hl is not None) or (bw is not None) or (cr is not None):
                                    colm1, colm2, colm3 = exp.columns(3)
                                    colm1.metric("Headline-Zeichen", f"{hl}" if hl is not None else "-")
                                    colm2.metric("Wörter (Body)", f"{bw}" if bw is not None else "-")
                                    colm3.metric("Coverage", ratio_to_pct(cr))
                                claims = qc.metrics.get("checked_claims") if isinstance(qc.metrics, dict) else None
                                if claims:
                                    exp.markdown("**Claim-Check (Auszug)**")
                                    for c in claims[:5]:
                                        badge = {"belegt":"✅","unbelegt":"⚠️","abweichung":"❌"}.get(c.get("status"), "•")
                                        exp.write(f"{badge} *{c.get('claim','')}*")
                                        if c.get("quote"):
                                            exp.caption(f"Zitat: „{c['quote']}”")
                                        if c.get("note"):
                                            exp.caption(f"Notiz: {c['note']}")

                            if qc.violations:
                                exp.markdown("**Violations:** " + ", ".join(qc.violations))
                            if qc.suggested_fixes:
                                exp.markdown("**Suggested fixes:** " + ", ".join(qc.suggested_fixes))
                            exp.markdown(f"**Decision:** `{qc.decision}`")

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

with tab_flow:
    st.subheader("End-to-End Workflow")
    dot = build_flow_graph()
    if dot:
        st.graphviz_chart(dot, use_container_width=True)
        st.caption("Ablauf von Text → Writer → Judge → optionaler Fix → Rendering/Download")
        st.download_button(
            "⬇️ Flow als DOT",
            data=dot.source.encode("utf-8"),
            file_name="poc_flow.dot",
            mime="text/vnd.graphviz"
        )
    else:
        st.warning("Graphviz ist nicht installiert. Zeige Fallback.")
        st.code(
            """Textarea-Text
  → generate_article(EXPRESS) → Article_ex
  → generate_article(KSTA)    → Article_ks
  → judge_article(EXPRESS, Article_ex) → QC_ex
  → judge_article(KSTA, Article_ks)    → QC_ks
  → maybe_revise(EXPRESS, Article_ex, QC_ex) → (Article_ex2?, QC_ex2?)
  → maybe_revise(KSTA, Article_ks, QC_ks)    → (Article_ks2?, QC_ks2?)
  → Render UI + Downloads""",
            language="text",
        )
