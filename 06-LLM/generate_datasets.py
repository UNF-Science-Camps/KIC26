"""generate_datasets.py
=======================
Genererer de tre traeningsdatasaet til transformer-workshoppen.

Temaet er fantasy/eventyr (riddere, drager, magi, quests). Alle linjer foelger
manuskript-formatet:

    Navn: replik          (en person SIGER noget)
    Navn: *handling*      (en person GOER noget, markeret med stjerner)

Scener kan starte med en kontekst-linje:

    [Kontekst: ... kort baggrund for scenen ...]

Der laves tre saet:
  * data_base.txt     - relativt serioest fantasy. Den gode "grund"-model traenes paa dette.
  * data_finetune.txt - samme univers, men med genkendelige jokes/referencer (gaming, internet, pop).
  * data_cooked.txt   - totalt "kogt" brainrot. Modellen bliver helt blaest. Ren komik.

Vil du have MERE variation? Tilfoej bare flere linjer i listerne nedenfor (fx BASE_SAYS,
BASE_DOES) eller flere navne/steder i pools'ene. Jo flere skabeloner, jo mere varierede historier.

VIGTIGT - fast vokabular:
  Vi bruger et FAST tegnsaet (VOCAB_CHARS). Det sikrer at modellens stoerrelse er den
  samme uanset hvilket data man traener paa - saa et faerdigtraenet checkpoint altid kan
  loades igen. Generatoren bruger kun tegn fra dette saet (det tjekkes til sidst med en assert).

Koer som script for at skrive de tre filer:
    python generate_datasets.py
"""

import random


# ---------------------------------------------------------------------------
# Det FASTE vokabular. Alt data skal kunne skrives med praecis disse tegn.
# (smaa + store bogstaver, danske bogstaver, tal, mellemrum/linjeskift og lidt tegnsaetning)
# ---------------------------------------------------------------------------
VOCAB_CHARS = (
    "\n "                            # linjeskift og mellemrum
    "abcdefghijklmnopqrstuvwxyz"     # smaa bogstaver
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"     # store bogstaver
    "æøåÆØÅ"                          # danske bogstaver
    "0123456789"                     # tal
    ".,!?:;-'\"*()[]"                # tegnsaetning, * (handling) og [] (kontekst)
)

# Hurtigt opslag til at tjekke om et tegn er tilladt.
VOCAB_SET = set(VOCAB_CHARS)


def filter_to_vocab(text):
    """Fjerner alle tegn der ikke er i det faste vokabular.

    Returnerer (renset_tekst, antal_fjernede).
    """
    kept = [ch for ch in text if ch in VOCAB_SET]
    removed = len(text) - len(kept)
    return "".join(kept), removed


# ---------------------------------------------------------------------------
# Faelles "byggeklodser" (pools) som skabelonerne traekker tilfaeldigt fra.
# Tilfoej gerne flere for endnu mere variation.
# ---------------------------------------------------------------------------
HERO_NAMES = [
    "Sir Aldric", "Lady Brunhild", "Troldmanden Mira", "Dronning Sigrid",
    "Kong Halfdan", "Væbneren Tobias", "Krigeren Greta", "Eremitten Osvald",
    "Tyven Freja", "Bueskytten Kasper", "Præsten Ingmar", "Smeden Bjørn",
    "Heksen Yrsa", "Ridderen Valdemar", "Spejderen Liv", "Munken Anselm",
    "Jægeren Sten", "Sangeren Alvilda", "Vandreren Rurik", "Alkymisten Dagny",
    "Kaptajnen Estrid", "Lejesvenden Knud", "Druiden Siv", "Stifinderen Aksel",
    "Healeren Tove", "Barden Frode", "Vogteren Hagen", "Novicen Ebba",
]

CREATURES = [
    "dragen", "trolden", "goblinen", "skelettet", "skovheksen", "skyggeulven",
    "spøgelset", "kæmpen", "slangen", "gravrøveren", "natravnen", "isbjørnen",
    "basilisken", "grottebjørnen", "lindormen", "vætten", "mosekonen",
    "ravnehæren", "stenmonsteret", "flagermusene",
]

PLACES = [
    "borgen", "Skyggeskoven", "bjergpasset", "den gamle kro", "landsbyen",
    "fangekælderen", "det høje tårn", "ruinerne", "den tågede sump",
    "havnen", "tronsalen", "krypten", "markedspladsen", "klosteret",
    "vindebroen", "dværgenes miner", "den frosne fjord", "alfernes lund",
    "den forladte mølle", "kongens have", "vagttårnet", "smedjen",
    "biblioteket", "offerlunden",
]

OBJECTS = [
    "det forsvundne sværd", "den magiske amulet", "den gyldne nøgle",
    "den forbandede ring", "kortet over kongeriget", "den hellige graal",
    "troldmandens stav", "den glemte bog", "kronens juvel", "elixiren",
    "det sorte segl", "drageægget", "runestenen", "sølvpilen",
    "den knuste krone", "trylledrikken", "det gamle banner", "ulvefangen",
]

TIMES = [
    "solnedgang", "midnat", "daggry", "den første sne", "fuldmåne",
    "stormen kommer", "vagtskiftet", "festen begynder", "den tredje dag",
    "klokkerne ringer", "tågen letter", "det bliver mørkt", "høsten", "vintersolhverv",
]

ADJ = [
    "modige", "snedige", "trætte", "sårede", "ivrige", "forsigtige",
    "stolte", "rastløse", "loyale", "frygtløse", "sultne", "håbefulde",
    "vagtsomme", "beslutsomme", "udmattede", "tavse", "kampklare", "nysgerrige",
]

FEELINGS = [
    "roligt", "alvorligt", "med et grin", "hviskende", "uden tøven",
    "med frygt i stemmen", "forpustet", "bestemt",
]

DIRECTIONS = [
    "mod nord", "ind i skoven", "op ad bjerget", "ud i tågen",
    "ned i krypten", "tilbage mod lejren", "mod porten", "langs floden",
]


def _slots(rng):
    """Vaelger en frisk tilfaeldig vaerdi til hver pladsholder en skabelon kan bruge."""
    return {
        "creature": rng.choice(CREATURES),
        "place": rng.choice(PLACES),
        "object": rng.choice(OBJECTS),
        "other": rng.choice(HERO_NAMES),
        "time": rng.choice(TIMES),
        "adj": rng.choice(ADJ),
        "feeling": rng.choice(FEELINGS),
        "direction": rng.choice(DIRECTIONS),
        "number": rng.choice(["tre", "syv", "tolv", "hundrede", "to", "fem", "ni"]),
    }


# ---------------------------------------------------------------------------
# SAET 1 - basis (serioest fantasy). Mange skabeloner = varierede historier.
# ---------------------------------------------------------------------------
BASE_SAYS = [
    "Vi må nå frem til {place} før {time}.",
    "Pas på - {creature} lurer et sted i {place}.",
    "{object} er skjult dybt inde i {place}.",
    "Jeg har hørt at {creature} vogter {object}.",
    "Vi kan ikke stole på {other}, ikke endnu.",
    "Følg mig, og hold jer tæt sammen.",
    "Stien gennem {place} er farlig efter {time}.",
    "Vi har brug for {object}, ellers er alt tabt.",
    "Lad os hvile her til {time} og fortsætte så.",
    "Jeg så {creature} bevæge sig mod {place}.",
    "Kongeriget regner med os nu.",
    "Tag {object} og løb, jeg holder dem tilbage.",
    "Der er noget galt med {place}, kan I mærke det?",
    "Mit sværd er klar, når I er.",
    "Vi har rejst i {number} dage for at nå hertil.",
    "Sig til {other} at vi venter ved {place}.",
    "Frygt ikke - vi er {adj} nok til det her.",
    "Legenden siger at {object} kun virker ved {time}.",
    "Hold øje med skyggerne i {place}.",
    "Vi deler os: halvdelen mod {place}, resten bliver.",
    "Det er en fælde, vend om mens vi kan.",
    "Jeg sværger at beskytte {object} med mit liv.",
    "Hold jer {adj} - vi er der næsten.",
    "{other}, du tager teten {direction}.",
    "Jeg stoler på dig, {other}.",
    "Vent her, jeg spejder {direction}.",
    "Hører I det? Noget bevæger sig i {place}.",
    "Vi har ikke meget tid før {time}.",
    "Lad os finde læ i {place} til natten.",
    "Det her sted giver mig en dårlig fornemmelse.",
    "{creature} kan ikke være langt væk nu.",
    "Saml jer om bålet, jeg har en plan.",
    "Tag dit våben frem, {other}.",
    "Vi kæmper kun hvis vi bliver tvunget til det.",
    "Profetien talte om denne nat.",
    "Find {object}, så er rejsen forbi.",
    "Jeg går først - dæk min ryg.",
    "Stille nu, vagterne er tæt på.",
    "Mit hjerte siger vi skal {direction}.",
    "Vi klarede det sidste, vi klarer også {place}.",
    "Kongen vil høre om vores bedrift.",
    "Ingen helt bliver født uden frygt.",
    "Drik og spis, i morgen rider vi mod {place}.",
    "Sværg på at vi følges ad til enden.",
]

BASE_DOES = [
    "trækker sit sværd og spejder mod {place}",
    "tænder en fakkel i den mørke {place}",
    "knæler og undersøger sporene i mudderet",
    "spænder buen og sigter mod {creature}",
    "lister lydløst forbi {creature}",
    "løfter {object} forsigtigt op fra alteret",
    "tegner et kort over {place} i sandet",
    "lytter ved døren ind til {place}",
    "binder sit sår og rejser sig {adj}",
    "kalder de andre {adj} sammen ved bålet",
    "kigger ud over {place} fra tårnets top",
    "skjuler {object} under sin kappe",
    "blokerer angrebet fra {creature} med skjoldet",
    "hvisker en besværgelse og lukker øjnene",
    "nikker {feeling} til {other}",
    "skuer ud over {place} {feeling}",
    "trækker kappen tættere om sig",
    "drager sit sværd og går {direction}",
    "samler troppen og peger {direction}",
    "lytter efter lyde fra {creature}",
    "tænder en lanterne og kigger {direction}",
    "knuger {object} mod brystet",
    "rejser sig {feeling} og griber sit våben",
    "studerer kortet over {place}",
    "deler brød ud til de {adj} rejsende",
    "spejder efter {creature} fra en klippe",
    "binder hesten ved {place}",
    "tegner en rune i støvet {feeling}",
]

BASE_CONTEXT = [
    "[Kontekst: {other} leder et lille følge gennem {place} for at finde {object}.]",
    "[Kontekst: Efter {time} samles de {adj} helte i {place} for at lægge en plan.]",
    "[Kontekst: {creature} har taget {object}, og kun {other} ved hvor det er gemt.]",
    "[Kontekst: En gammel profeti siger at {object} skal bringes til {place} før {time}.]",
    "[Kontekst: {other} og et lille følge søger ly i {place}, mens {creature} jager dem.]",
    "[Kontekst: Kongen har sendt {other} ud for at hente {object} inden {time}.]",
    "[Kontekst: Rygtet siger at {object} kan vække {creature} til live i {place}.]",
    "[Kontekst: To {adj} venner, {other} og deres følge, raster i {place} ved {time}.]",
    "[Kontekst: {creature} bevogter {object} dybt i {place}, og kun de modige tør derind.]",
    "[Kontekst: Efter et slag samles de {adj} overlevende i {place} for at planlægge videre.]",
]


# ---------------------------------------------------------------------------
# SAET 2 - finetune (samme univers, men med genkendelige jokes/referencer:
# gaming, internet og pop/musik). Stadig laesbart som en historie.
# ---------------------------------------------------------------------------
JOKE_SAYS = [
    "Den her quest er ren speedrun, kom nu folkens.",
    "{creature} dropper garanteret legendary loot, lad os farme den.",
    "Jeg er level {number}, den boss er piece of cake.",
    "Kongen streamer fra tronsalen i aften, glem ikke at like.",
    "Vi mangler bare {object} for at complete the main quest.",
    "Hold da op, {place} har vild respawn-rate på fjender.",
    "Min mana er på nul, jeg skal lige loote en potion.",
    "{other} gik AFK midt i kampen mod {creature}, klassisk.",
    "Skal vi rushe {place} eller tage den stealth?",
    "Den drage-fight er pure endgame content, fr.",
    "Jeg har grindet xp i {place} hele natten.",
    "Pro tip: bloker {creature} og counter når den lagger.",
    "Vi skal have {object} før serveren lukker ved {time}.",
    "Lyt til den nye banger mens vi rider mod {place}.",
    "Det loot er straight up cracked, no kidding.",
    "GG til holdet, vi clearede {place} uden at dø.",
    "Jeg fulgte en guide, men {creature} er stadig svær.",
    "Kan nogen heale? Jeg er på {number} hp mod {creature}.",
    "Den boss har for meget HP, nerf den.",
    "Jeg har lige unlocked en ny skill, watch this.",
    "Vi wipede på {creature} igen, seriøst?",
    "Hvem har aggroet {creature}?! Ikke mig.",
    "Loot-tabellen for {place} er straight bugged.",
    "Jeg buffer holdet, så pusher vi {place}.",
    "Den daily quest giver dobbelt xp i dag.",
    "Skiftede til min crit-build før {creature}-fighten.",
    "Brb, jeg skal lige crafte {object}.",
    "Vi tager den på hardcore, ingen respawns.",
]

JOKE_DOES = [
    "caster fireball og råber GG",
    "laver et sick 360 no-scope med buen mod {creature}",
    "tjekker sin loadout før raidet på {place}",
    "emoter foran {creature} for at flexe",
    "spammer heal-knappen og survival mod {creature}",
    "looter hele {place} for skrald og sælger det",
    "tager en quick selfie med {object} til sit feed",
    "rage-quitter da {creature} one-shotter ham",
    "laver en TikTok-dans midt i {place}",
    "drikker en energidrik og pusher mod {place}",
    "kalder sit team sammen for en pep-talk",
    "alt-tabber for at google hvordan man slår {creature}",
    "kiter {creature} rundt om {place}",
    "spammer combo-tasterne mod {creature}",
    "tjekker minimap'et over {place}",
    "saver sit game lige før bossen",
    "trash-talker {other} i voice chat",
    "looter en chest og finder {object}",
]

JOKE_CONTEXT = [
    "[Kontekst: {other} samler et squad til et raid på {place} for at få {object}.]",
    "[Kontekst: Det er sidste dag i seasonen, og alle vil nå {object} før {time}.]",
    "[Kontekst: {creature} er den nye boss, og chatten håber på legendary loot.]",
    "[Kontekst: Holdet grinder dailies i {place} for at nå max level før {time}.]",
    "[Kontekst: Det er guild raid-aften, og {creature} er den sidste boss.]",
    "[Kontekst: {other} streamer hele runet gennem {place} live til chatten.]",
]


# ---------------------------------------------------------------------------
# SAET 3 - kogt/brainrot (maksimalt kaos). Curated og PG-13: kun teen-venlig slang.
# ---------------------------------------------------------------------------
COOKED_NAMES = [
    "Skibidi-ridderen", "Sigma-dragen", "Gigachad-kongen", "Rizz-troldmanden",
    "Ohio-goblinen", "NPC-væbneren", "Based-heksen", "Mid-skelettet",
    "W-ridderen", "Cringe-trolden", "Sheesh-barden", "Goofy-vagten",
]

COOKED_SAYS = [
    "kongen har mad rizz no cap fr fr",
    "skibidi {place} go BRRR, only in Ohio",
    "L plus ratio plus du faldt i {place} bro",
    "{creature} er straight up mid, touch grass",
    "vi farmer {object} fr fr sheeesh sigma grindset",
    "han er en NPC, kig på hans goofy ahh walk",
    "rizz level over 9000 i {place} no cap",
    "based king W W W det er bare facts",
    "{creature} prøvede at rizze dronningen, sus",
    "skibidi toilet boss i {place} igen, brainrot",
    "real ones looter {object} ved {time} only",
    "ohio rizz dragen sigma alpha gigachad sheesh",
    "bro {other} er så cringe det er crazy fr",
    "we move we move, ingen cap, ren W energi",
    "han mewing mens {creature} angriber, goofy",
    "skibidi sigma rizz {object} sheeesh naah jk",
    "{place} er straight up Ohio, ingen lyver",
    "han droppede en hard W i {place} sheesh",
    "skibidi rizz party i tronsalen lets gooo",
    "{other} er en NPC fr fr touch grass",
    "gigachad energi only, vi farmer {object}",
    "no cap den drage er goofy ahh mid",
    "we cooking i {place} skibidi style",
    "ratio plus L plus {creature} er cringe",
    "sigma sigma boy farmer xp i {place}",
    "based king dropper {object} sheeesh",
]

COOKED_DOES = [
    "laver en Ohio backflip SHEEESH",
    "hitter en sigma pose foran {creature}",
    "råber SKIBIDI så hele {place} ryster",
    "dabber på {creature} efter en clean W",
    "mewer intenst mens {place} brænder",
    "spammer L plus ratio i hele {place}",
    "rizzer {creature} med ren sigma energi",
    "gør en goofy ahh dans på kongens bord",
    "tager {object} og råber ONLY IN OHIO",
    "looker direkte i kameraet som en chad",
    "hitter en griddy på {creature}s lig",
    "skibidi-danser midt i {place}",
    "råber W RIZZ så {place} ryster",
    "mewer mens han looter {object}",
    "gør en sigma staredown med {creature}",
    "dabber på {other} efter en clean ratio",
]

COOKED_CONTEXT = [
    "[Kontekst: {other} og squaddet farmer skibidi-rizz i {place}, ingen cap.]",
    "[Kontekst: Det er Ohio-finalen, og {creature} har mad rizz men er mid.]",
    "[Kontekst: Sigma-grindset er real, og {object} venter i {place} fr fr.]",
    "[Kontekst: Skibidi-squaddet rusher {place} for at få max rizz inden {time}.]",
    "[Kontekst: Det er Ohio o'clock, og {creature} har null rizz men max cringe.]",
    "[Kontekst: Sigma-grindset fortsætter i {place}, kun real ones er tilbage.]",
]


# ---------------------------------------------------------------------------
# Scene-bygger - faelles for alle tre saet.
# ---------------------------------------------------------------------------
def _fill(template, rng):
    """Indsaetter tilfaeldige vaerdier i en skabelon."""
    return template.format_map(_slots(rng))


def make_scene(rng, says, does, contexts, names, context_prob=0.35, action_prob=0.3):
    """Bygger en enkelt lille scene med 2-4 personer og 4-13 linjer."""
    lines = []
    # Eventuel kontekst-linje i toppen.
    if rng.random() < context_prob:
        lines.append(_fill(rng.choice(contexts), rng))
    # Et lille fast "cast" giver lokal sammenhaeng i scenen.
    cast = rng.sample(names, rng.randint(2, 4))
    n_lines = rng.randint(4, 13)
    for _ in range(n_lines):
        speaker = rng.choice(cast)
        if rng.random() < action_prob:
            line = "{}: *{}*".format(speaker, _fill(rng.choice(does), rng))
        else:
            line = "{}: {}".format(speaker, _fill(rng.choice(says), rng))
        lines.append(line)
    return "\n".join(lines) + "\n\n"


def build_dataset(target_chars, says, does, contexts, names, seed,
                  context_prob=0.35, action_prob=0.3):
    """Bygger scener indtil teksten er ca. target_chars tegn lang."""
    rng = random.Random(seed)
    chunks = []
    total = 0
    while total < target_chars:
        scene = make_scene(rng, says, does, contexts, names,
                           context_prob=context_prob, action_prob=action_prob)
        chunks.append(scene)
        total += len(scene)
    return "".join(chunks)


def main():
    # Maal-stoerrelser (i tegn). Basis er stoerst, saa modellen kan lave ordentlige historier.
    datasets = {
        "data_base.txt": dict(
            target_chars=900_000, says=BASE_SAYS, does=BASE_DOES,
            contexts=BASE_CONTEXT, names=HERO_NAMES, seed=1337,
        ),
        "data_finetune.txt": dict(
            target_chars=300_000, says=JOKE_SAYS, does=JOKE_DOES,
            contexts=JOKE_CONTEXT, names=HERO_NAMES, seed=2024,
        ),
        "data_cooked.txt": dict(
            target_chars=300_000, says=COOKED_SAYS, does=COOKED_DOES,
            contexts=COOKED_CONTEXT, names=COOKED_NAMES, seed=9001,
            context_prob=0.3, action_prob=0.35,
        ),
    }

    for filename, kwargs in datasets.items():
        text = build_dataset(**kwargs)
        # Sikkerhedstjek: alt skal kunne skrives med det faste vokabular.
        cleaned, removed = filter_to_vocab(text)
        assert removed == 0, (
            "{} indeholdt {} tegn udenfor VOCAB_CHARS!".format(filename, removed)
        )
        with open(filename, "w", encoding="utf-8") as f:
            f.write(cleaned)
        print("Skrev {:>18}  ({:>9,} tegn)".format(filename, len(cleaned)))

    print("Faerdig. Vokabularstoerrelse:", len(sorted(set(VOCAB_CHARS))))


if __name__ == "__main__":
    main()
