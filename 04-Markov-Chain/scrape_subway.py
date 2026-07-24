#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bygger et Subway Surfers-tekstkorpus til Markov-kaeder og sprogmodeller.

Kilder:
  1. Subway Surfers Fandom-wikien (subwaysurf.fandom.com) — al broedtekst fra
     alle ~1700 artikelsider (karakterer, hoverboards, byer, opdateringer ...).
  2. Wikipedia-artiklen "Subway Surfers" paa dansk og engelsk.

Metoden er den samme som de oevrige korpora i DataTextFolder/ (Star_Wars_Wiki.txt,
minecraft_wiki.txt ...): hent renderet HTML pr. side via MediaWiki-API'et og traek
den rene broedtekst ud med BeautifulSoup (infobokse, tabeller, referencer og
navigation fjernes). Resultatet er ét stort .txt-korpus.

Koer:   python3 scrape_subway.py
Output: DataTextFolder/subway_surfers.txt
"""

import json
import os
import re
import subprocess
import sys
import time
import urllib.parse

from bs4 import BeautifulSoup

# --- indstillinger ---------------------------------------------------------
UA = "UNF-KIC26 teaching-material corpus builder"
FANDOM = "https://subwaysurf.fandom.com/api.php"
WIKIPEDIA = {
    "dansk":   "https://da.wikipedia.org/w/api.php",
    "engelsk": "https://en.wikipedia.org/w/api.php",
}
HER = os.path.dirname(os.path.abspath(__file__))
UD_FIL = os.path.join(HER, "DataTextFolder", "subway_surfers.txt")
PAUSE = 0.2   # sekunders pause mellem kald (vær venlig ved serveren)


# --- hjaelpere -------------------------------------------------------------
def hent_json(base, params):
    """Ét GET-kald mod et MediaWiki-API med simpel retry."""
    url = base + "?" + urllib.parse.urlencode(params)
    for forsoeg in range(4):
        svar = subprocess.run(
            ["curl", "-s", "--max-time", "40", "-A", UA, url],
            capture_output=True, text=True,
        )
        try:
            return json.loads(svar.stdout)
        except json.JSONDecodeError:
            time.sleep(1.5 * (forsoeg + 1))
    return {}


def alle_artikelsider(base):
    """Titler paa alle rigtige artikelsider (ns=0, ingen redirects/gallerier)."""
    titler, cont = [], {}
    while True:
        params = {
            "action": "query", "list": "allpages", "apnamespace": "0",
            "apfilterredir": "nonredirects", "aplimit": "500", "format": "json",
        }
        params.update(cont)
        data = hent_json(base, params)
        titler += [a["title"] for a in data.get("query", {}).get("allpages", [])]
        if "continue" in data:
            cont = data["continue"]
        else:
            break
    # "/"-subsider er billed-gallerier uden broedtekst — dem springer vi over
    return [t for t in titler if "/" not in t]


def html_til_tekst(html):
    """Renderet artikel-HTML -> ren broedtekst (afsnit og punkter, én pr. linje)."""
    suppe = BeautifulSoup(html, "html.parser")
    # fjern stoej: infobokse, tabeller, referencer, gallerier, navigation, redigér
    stoej = ["table", "sup", "style", "script", "figure",
             ".navbox", ".reference", ".mw-editsection", ".gallery",
             ".portable-infobox", ".toc", ".mw-references-wrap", ".hatnote"]
    for vaelger in stoej:
        for element in suppe.select(vaelger):
            element.decompose()
    # tag broedtekst fra afsnit, punkter og overskrifter — hver som sin egen linje
    linjer = []
    for blok in suppe.select("p, li, h2, h3, h4"):
        linjer.append(blok.get_text(separator=" "))
    return "\n".join(linjer)


def side_broedtekst(base, titel):
    """Hent én sides renderede HTML og udtraek broedteksten."""
    data = hent_json(base, {
        "action": "parse", "page": titel, "prop": "text",
        "formatversion": "2", "format": "json",
    })
    html = data.get("parse", {}).get("text")
    return html_til_tekst(html) if html else ""


def wikipedia_artikel(base, titel="Subway Surfers"):
    """Ren plaintext fra en Wikipedia-artikel (TextExtracts-udvidelsen)."""
    data = hent_json(base, {
        "action": "query", "prop": "extracts", "explaintext": "1",
        "exsectionformat": "plain", "redirects": "1",
        "titles": titel, "format": "json",
    })
    for side in data.get("query", {}).get("pages", {}).values():
        return side.get("extract", "") or ""
    return ""


def ryd(tekst):
    """Fjern reference-mærker og normaliser mellemrum/tomme linjer."""
    tekst = re.sub(r"\[\s*\d+\s*\]", "", tekst)      # [1], [ 2 ]
    tekst = re.sub(r"\[\s*(edit|redigér)\s*\]", "", tekst, flags=re.I)
    rene = []
    for linje in tekst.splitlines():
        linje = re.sub(r"[ \t]+", " ", linje).strip()
        if len(linje) > 1:                            # drop tomme/enkelttegns-linjer
            rene.append(linje)
    return "\n".join(rene)


# --- hovedprogram ----------------------------------------------------------
def main():
    dele = []

    # 1) Wikipedia (dansk + engelsk) — lille, men det brugeren bad om
    for sprog, api in WIKIPEDIA.items():
        tekst = ryd(wikipedia_artikel(api))
        if tekst:
            dele.append(tekst)
            print(f"[wikipedia/{sprog}] {len(tekst):>7} tegn", file=sys.stderr)
        time.sleep(PAUSE)

    # 2) Fandom-wikien — hele broedtekst-korpuset
    titler = alle_artikelsider(FANDOM)
    print(f"[fandom] {len(titler)} artikelsider at hente ...", file=sys.stderr)
    for i, titel in enumerate(titler, 1):
        tekst = ryd(side_broedtekst(FANDOM, titel))
        if len(tekst) > 40:                           # spring naesten-tomme sider over
            dele.append(f"{titel}\n{tekst}")
        if i % 100 == 0:
            samlet = sum(len(d) for d in dele)
            print(f"[fandom] {i}/{len(titler)} sider — {samlet:,} tegn indtil nu",
                  file=sys.stderr)
        time.sleep(PAUSE)

    korpus = "\n\n".join(dele) + "\n"
    os.makedirs(os.path.dirname(UD_FIL), exist_ok=True)
    with open(UD_FIL, "w", encoding="utf-8") as f:
        f.write(korpus)

    ord = len(korpus.split())
    print(f"\nFAERDIG: {UD_FIL}", file=sys.stderr)
    print(f"  {len(korpus):,} tegn | {ord:,} ord | {len(set(korpus))} unikke tegn",
          file=sys.stderr)


if __name__ == "__main__":
    main()
