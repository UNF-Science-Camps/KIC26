"""generate_datasets.py
=======================
Genererer de tre traeningsdatasaet til transformer-workshoppen.

Temaet er fantasy/eventyr (riddere, drager, magi, quests). Alle linjer foelger
manuskript-formatet:

    Navn: replik          (en person SIGER noget)
    Navn: *handling*      (en person GOER noget, markeret med stjerner)

Scener kan starte med en kontekst-linje:

    [Kontekst: ... kort baggrund for scenen ...]

Der laves syv saet plus EN samlet fil med det hele, hvert med mindst
10.000 FORSKELLIGE linjer (et minimum, ikke et loft - se nedenfor):
  * data_base.txt        - relativt serioest fantasy. Den gode "grund"-model traenes paa dette.
  * data_finetune.txt    - samme univers + gaming, Minecraft, Roblox, kantine og skole-humor.
  * data_cooked.txt      - totalt "kogt" brainrot, isaer italiensk. Modellen bliver helt blaest.
  * data_dad_jokes.txt   - danske far-jokes rundt om spisebordet.
  * data_news.txt        - nyhedsstudiet daekker fantasy-universet, helt toert.
  * data_shakespeare.txt - gammeldags teater-dansk med dolke og drama.
  * data_fairytales.txt  - eventyr i H.C. Andersen-stil.
  * data_all.txt         - ALT det ovenstaaende i en fil, blandet godt.

Vil du have LAENGERE filer? Giv en faktor med paa kommandolinjen:

    python generate_datasets.py 3      (alle filer bliver ca. 3x saa lange,
                                        og antallet af forskellige saetninger stiger med)

Vil du have MERE variation? Tilfoej flere linjer i listerne nedenfor (fx BASE_SAYS,
BASE_DOES) eller flere navne/steder i pools'ene. Jo flere skabeloner, jo mere varierede historier.
Vil du hente, blande eller bygge dine EGNE datasaet, saa brug det lille script dataset_tools.py.

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
    "Skjoldmøen Thyra", "Riddersvenden Palle", "Grevinden Mathilde",
    "Kortmageren Niels", "Falkoneren Runa", "Skriveren Otto",
    "Urtekonen Gudrun", "Fanebæreren Erik", "Spionen Vigga",
    "Søfareren Leif", "Kokken Magda", "Staldknægten Jeppe",
    "Stjernetyderen Vera", "Runemesteren Gorm", "Skjalden Helga",
    "Portneren Aage", "Feltskæren Sune", "Budbringeren Kirsten",
    "Dragetæmmeren Ulf", "Bibliotekaren Petra",
    "Ridderen Astrid", "Væbneren Malte", "Troldkvinden Signe", "Kong Regnar",
    "Dronning Alfhild", "Bersærkeren Torsten", "Skovfogeden Villum",
    "Fiskeren Espen", "Grev Mogens", "Baronessen Ellen", "Ildtæmmeren Saxo",
    "Isheksen Hedvig", "Stormesteren Arnulf", "Lærlingen Pil",
    "Krystalmageren Rane", "Natvægteren Alf", "Perlefiskeren Mette",
    "Bjergkongen Harald", "Sværddanseren Idun", "Fortælleren Ravn",
    "Urmageren Cornelius", "Duepasseren Lærke", "Brobyggeren Steen",
    "Måneridderen Selma", "Tårnvogteren Oluf", "Kortlæseren Vibe",
    "Ulvetæmmeren Bjarke", "Havheksen Marina", "Skyggejægeren Njord",
    "Falkedronningen Gry",
]

CREATURES = [
    "dragen", "trolden", "goblinen", "skelettet", "skovheksen", "skyggeulven",
    "spøgelset", "kæmpen", "slangen", "gravrøveren", "natravnen", "isbjørnen",
    "basilisken", "grottebjørnen", "lindormen", "vætten", "mosekonen",
    "ravnehæren", "stenmonsteret", "flagermusene",
    "kæmpeedderkoppen", "sumpuhyret", "ildfuglen", "stentrolden",
    "havuhyret", "varulven", "tågedragen", "krystalgolemmen",
    "skyggekatten", "hulebjørnen", "sneharpyen", "gravormen",
    "nissehæren", "elverdronningen", "søslangen", "askedæmonen",
    "griffen", "enhjørningen", "kæmpeuglen", "sandormen", "tågehunden",
    "ildsalamanderen", "frostjætten", "stormørnen", "hulepadden",
    "krystaledderkoppen", "søtrolden", "sfinksen", "mareridtshesten",
    "skovånden", "gravkatten", "kloakdragen", "spindelheksen",
    "dværgdragen", "kæmpekrabben", "skyggeharen", "sumptrolden",
    "jernravnen", "månevargen", "askeslangen",
]

PLACES = [
    "borgen", "Skyggeskoven", "bjergpasset", "den gamle kro", "landsbyen",
    "fangekælderen", "det høje tårn", "ruinerne", "den tågede sump",
    "havnen", "tronsalen", "krypten", "markedspladsen", "klosteret",
    "vindebroen", "dværgenes miner", "den frosne fjord", "alfernes lund",
    "den forladte mølle", "kongens have", "vagttårnet", "smedjen",
    "biblioteket", "offerlunden",
    "den sunkne by", "krystalgrotten", "fyrtårnet", "den glemte dal",
    "røverkroen", "iskløften", "slottets køkken", "den store bro",
    "månesøen", "gletsjeren", "ørkenfortet", "de hængende haver",
    "underjorden", "tempelruinen", "grænselandet", "den brændte mark",
    "elverhøjen", "dybhavnen",
    "drageborgen", "sølvminen", "tågehavnen", "gudetemplet",
    "hvidtjørneskoven", "røverborgen", "saltstepperne", "månetårnet",
    "det gamle amfiteater", "perlebugten", "vulkanens rand",
    "stjernekammeret", "de syv søer", "kongsgården", "glasbroen",
    "skyggedalen", "midnatsmarkedet", "flodmundingen", "bjørnehulen",
    "solurspladsen", "verdens ende", "den hemmelige have",
]

OBJECTS = [
    "det forsvundne sværd", "den magiske amulet", "den gyldne nøgle",
    "den forbandede ring", "kortet over kongeriget", "den hellige graal",
    "troldmandens stav", "den glemte bog", "kronens juvel", "elixiren",
    "det sorte segl", "drageægget", "runestenen", "sølvpilen",
    "den knuste krone", "trylledrikken", "det gamle banner", "ulvefangen",
    "månespejlet", "det syngende horn", "isdolken", "stormklokken",
    "den evige fakkel", "dragehandsken", "søkortet", "gudetræets frø",
    "den sorte kappe", "krystalkuglen", "heltekongens hjelm", "solstenen",
    "tordenhammeren", "sølvfløjten", "livets vand", "spejlskjoldet",
    "ravkæden", "vinterkronen", "den syvende nøgle", "sandhedens fjer",
    "skyggelygten", "havfruens kam", "jættens tand", "stormflasken",
    "guldharpen", "dragens skæl", "profetiens rulle", "nattens diadem",
    "kongens signetring", "det utrættelige reb", "frøen af jade",
    "kompasset der aldrig lyver",
]

FOODS = [
    "chokoladekiks", "kanelsnegle", "rugbrødsmadder", "æblegrød",
    "honningkager", "saltkringler", "svampesuppe", "brombærsyltetøj",
    "friskbagt brød", "gedeost", "æblemost", "krydderboller",
    "havregrød", "røget skinke", "pandekager", "hindbærsnitter",
    "kartoffelsuppe", "flæskesteg", "risengrød", "lakridsbånd",
    "æbleskiver", "rødgrød med fløde", "smørrebrød", "medisterpølse",
    "kringle", "vaniljekranse", "boller i karry", "leverpostej",
    "frikadeller", "citronmåne", "drømmekage", "koldskål",
    "spandauer", "hakkebøf", "brunede kartofler", "jordbærgrød",
]

CITIES = [
    "København", "Aarhus", "Odense", "Aalborg", "Esbjerg", "Randers",
    "Kolding", "Horsens", "Vejle", "Roskilde", "Herning", "Silkeborg",
    "Sønderborg", "Hillerød", "Næstved", "Viborg",
    "Skagen", "Ribe", "Helsingør", "Fredericia", "Nyborg", "Svendborg",
    "Holstebro", "Slagelse", "Køge", "Frederikshavn",
]

TIMES = [
    "solnedgang", "midnat", "daggry", "den første sne", "fuldmåne",
    "stormen kommer", "vagtskiftet", "festen begynder", "den tredje dag",
    "klokkerne ringer", "tågen letter", "det bliver mørkt", "høsten", "vintersolhverv",
    "morgengry", "den syvende time", "nymåne", "tordenvejret", "frokosttid",
    "den lange nat", "markedsdagen", "sankthans", "den første frost",
    "den anden vagt", "solformørkelsen", "aftensang", "det første hanegal",
    "tidevandsskiftet", "midsommer", "den niende time", "regntiden",
    "lysfesten", "torvedagen", "månens opgang", "det sidste bål",
]

ADJ = [
    "modige", "snedige", "trætte", "sårede", "ivrige", "forsigtige",
    "stolte", "rastløse", "loyale", "frygtløse", "sultne", "håbefulde",
    "vagtsomme", "beslutsomme", "udmattede", "tavse", "kampklare", "nysgerrige",
    "søvnige", "genstridige", "muntre", "skeptiske", "standhaftige",
    "forfrosne", "opstemte", "årvågne", "tålmodige", "drillesyge",
    "listige", "gavmilde", "hidsige", "sindige", "rådsnare",
    "storsindede", "tvivlende", "grublende", "letbenede", "skarpsynede",
    "jernviljede", "blødhjertede", "stædige", "skælmske", "højrøstede",
    "lavmælte", "solbrændte", "forblæste", "veludhvilede", "morgenfriske",
    "sejrssikre", "eventyrlystne", "gnavne", "fnisende", "kampglade",
    "letskræmte",
]

FEELINGS = [
    "roligt", "alvorligt", "med et grin", "hviskende", "uden tøven",
    "med frygt i stemmen", "forpustet", "bestemt",
    "eftertænksomt", "med et skævt smil", "højtideligt", "gabende",
    "med tårer i øjnene", "syngende", "tørt", "med løftet pegefinger",
    "med sammenbidte tænder", "lettet", "drømmende", "med hænderne i siden",
    "halvt i søvne", "med glimt i øjet", "mut", "andægtigt",
    "med rynket pande", "ivrigt", "beklemt", "med høj røst",
]

DIRECTIONS = [
    "mod nord", "ind i skoven", "op ad bjerget", "ud i tågen",
    "ned i krypten", "tilbage mod lejren", "mod porten", "langs floden",
    "mod syd", "over engen", "gennem kløften", "ad bagtrappen",
    "ud på isen", "ind i mørket", "over vindebroen", "ned til stranden",
    "mod øst", "mod vest", "gennem sivene", "op ad klippestien",
    "ind gennem sideporten", "ud over vidderne", "ned ad vindeltrappen",
    "hen over torvet", "ud i regnen", "hjemad",
]


def _slots(rng, names):
    """Vaelger en frisk tilfaeldig vaerdi til hver pladsholder en skabelon kan bruge.

    'names' er det aktuelle saets persongalleri, saa {other} passer til universet
    (fx familien i far-jokes og brainrot-figurerne i det kogte saet).
    """
    return {
        "creature": rng.choice(CREATURES),
        "creature2": rng.choice(CREATURES),
        "place": rng.choice(PLACES),
        "object": rng.choice(OBJECTS),
        "other": rng.choice(names),
        "other2": rng.choice(names),
        "time": rng.choice(TIMES),
        "adj": rng.choice(ADJ),
        "feeling": rng.choice(FEELINGS),
        "direction": rng.choice(DIRECTIONS),
        "number": rng.choice(["tre", "syv", "tolv", "hundrede", "to", "fem", "ni"]),
        "food": rng.choice(FOODS),
        "food2": rng.choice(FOODS),
        "city": rng.choice(CITIES),
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
    "Følg mig gennem {place}, {other}, og hold jer tæt sammen.",
    "Stien gennem {place} er farlig efter {time}.",
    "Vi har brug for {object}, ellers er alt tabt.",
    "Lad os hvile her til {time} og fortsætte så.",
    "Jeg så {creature} bevæge sig mod {place}.",
    "Kongeriget regner med os, {other} - især efter {time}.",
    "Tag {object} og løb, jeg holder dem tilbage.",
    "Der er noget galt med {place}, kan I mærke det?",
    "Mit sværd er klar, {other} - vi mødes ved {place}.",
    "Vi har rejst i {number} dage for at nå hertil.",
    "Sig til {other} at vi venter ved {place}.",
    "Frygt ikke - vi er {adj} nok til det her.",
    "Legenden siger at {object} kun virker ved {time}.",
    "Hold øje med skyggerne i {place}.",
    "Vi deler os: halvdelen mod {place}, resten bliver.",
    "Det er en fælde, {other} - {creature} venter i {place}!",
    "Jeg sværger at beskytte {object} med mit liv.",
    "Hold jer {adj} - vi er der næsten.",
    "{other}, du tager teten {direction}.",
    "Jeg stoler på dig, {other}.",
    "Vent her, jeg spejder {direction}.",
    "Hører I det? Noget bevæger sig i {place}.",
    "Vi har ikke meget tid før {time}.",
    "Lad os finde læ i {place} til natten.",
    "{place} giver mig en dårlig fornemmelse, især ved {time}.",
    "{creature} kan ikke være langt væk nu.",
    "Saml jer om bålet, {other}, jeg har en plan mod {creature}.",
    "Tag dit våben frem, {other}.",
    "Vi kæmper kun, hvis {creature} tvinger os til det, {other}.",
    "Profetien talte om {time} og om {object}.",
    "Find {object}, så er rejsen forbi.",
    "Jeg går først ind i {place} - dæk min ryg, {other}.",
    "Stille nu, {other} - vagterne fra {place} er tæt på.",
    "Mit hjerte siger vi skal {direction}.",
    "Vi klarede det sidste, vi klarer også {place}.",
    "Kongen i {place} vil høre om vores bedrift, {other}.",
    "Ingen helt bliver født uden frygt, {other} - heller ikke i {place}.",
    "Drik og spis, i morgen rider vi mod {place}.",
    "Sværg ved {object}, {other}, at vi følges ad til enden.",
    # korte raab og svar
    "Stille, {other}!",
    "Løb mod {place}!",
    "Bag dig, {other}!",
    "Nu sker det, {other}.",
    "Denne vej, {other} - {direction}!",
    "Vent på mig ved {place}!",
    "Er alle med, {other}?",
    "Ikke en lyd nu, {other}.",
    "Se derovre - ved {place}!",
    "Hold linjen, {other}!",
    "Det var tæt på, {other}.",
    "Godt klaret, {other}.",
    "Vi ses ved {place}.",
    "Aldrig i livet, {other}.",
    "Så er det nu, {other}.",
    # laengere replikker
    "Da jeg var ung, sagde min mester altid: den der frygter {creature}, har allerede tabt.",
    "Hvis vi følger floden forbi {place} og drejer af ved {time}, kan vi nå frem uset.",
    "Der findes et gammelt kort i {place}, og på det kort er {object} tegnet med rødt blæk.",
    "Jeg har set {creature} en enkelt gang før, og jeg lover jer: den er større, end sagnene siger.",
    "Min bedstemor fortalte, at {object} blev smedet i {place}, længe før kongeriget fik sit navn.",
    "Hvis jeg ikke er tilbage inden {time}, så tag {object} og rid mod {place} uden mig.",
    "Der er to veje til {place}: den korte, hvor {creature} venter, og den lange, hvor vinteren venter.",
    "Enhver kan svinge et sværd, men det kræver mod at række hånden frem mod {other}.",
    "Jeg drømte i nat om {place} - og i drømmen stod {object} og lyste som en lille sol.",
    "{food} og håb, {other} - mere har en helt aldrig haft brug for.",
    "Lad os dele det sidste {food} - ingen kæmper godt på tom mave.",
    "De kalder {place} for forbandet, men jeg tror kun, det er glemt.",
    "Engang var {other} den bedste i hele riget til at spore {creature}.",
    "Vinden vender ved {time}, og så sejler vi, uanset hvad rådet siger.",
    "Skriv det i krøniken, {other}: ved {place} veg vi ikke tilbage.",
    "Hvor mange gange har jeg reddet dig nu, {other}? Det var {number} gange, mindst.",
    "Der er mere mellem himmel og jord end {creature} og gamle sagn.",
    "Først finder vi {object}, så finder vi hjem, og så sover jeg i {number} dage.",
    "Rygtet i {place} siger, at {other} har set {creature} i live.",
    "Uden {object} kommer ingen af os levende gennem {place}.",
    "Jeg bytter gerne min plads i sagnene for et fad {food} og en varm seng.",
    "Hør efter: Vi går ind ved {time}, vi går {direction}, og vi går sammen.",
    "Det siges at {creature} kun kan besejres, mens {time} står på.",
    "En dag skal børnene i {place} synge om det, vi gør i nat.",
    "Tro mig, {other}: modet kommer først, når man har brug for det.",
    "Skjoldene op! {creature} angriber fra {direction}!",
    "Har nogen set min hest? Den stod her ved {time}.",
    "Det her kort er ældre end {other}s bedstemor.",
    "Vi bytter: min {food} for din plads tættest på bålet.",
    "Porten til {place} åbner kun for den, der bærer {object}.",
    "Jeg fandt friske spor af {creature} nede ved {place}.",
    "Hvem tog den sidste fakkel? Vi skal bruge lys før {time}.",
    "En sang, {other}! Vejen til {place} er lang uden musik.",
    "Rolig nu. Ingen rører {object}, før jeg siger til.",
    "Alt for riget og for {place}!",
    "Bag skjoldet, {other}, nu!",
    "Der, {other}! I tågen ved {place}!",
    "Fremad, venner - mod {place}!",
    "Ikke flere løfter, {other} - kun handling før {time}.",
    "Min fars sværd har aldrig svigtet mod {creature}. Endnu.",
    "Vagten ved {place} sover altid ved {time}. Det er vores chance.",
    "Hvis {creature} kan bløde, kan den besejres.",
    "Tag min hånd, {other}! Jeg slipper dig ikke.",
    "Regnen over {place} sletter vores spor, {other}. Held i uheld.",
    "Tre ting kræver et helteliv, {other}: mod, venner og {food}.",
    "Jeg giver {object} videre til dig, {other}. Vogt det bedre, end jeg gjorde.",
    "Sagnet om {place} nævner en skjult dør {direction}.",
    "Ingen forlader {place}, før {object} er fundet!",
    "Du skylder mig stadig for den gang med {creature}, {other}.",
    "Ved {time} angriber vi {place}. Sov, mens I kan.",
    "Hør vinden... {creature} er tættere på, end vi troede.",
    "Mit skjold er dit skjold, {other}.",
    "Selv den længste vagt slutter ved {time}.",
]

BASE_DOES = [
    "trækker sit sværd og spejder mod {place}",
    "tænder en fakkel i den mørke {place} {feeling}",
    "knæler og undersøger sporene efter {creature} i mudderet",
    "spænder buen og sigter mod {creature} {feeling}",
    "lister lydløst forbi {creature}",
    "løfter {object} forsigtigt op fra alteret",
    "tegner et kort over {place} i sandet {feeling}",
    "lytter ved døren ind til {place} {feeling}",
    "binder sit sår og rejser sig {adj}",
    "kalder de andre {adj} sammen ved bålet",
    "kigger ud over {place} fra tårnets top {feeling}",
    "skjuler {object} under sin kappe {feeling}",
    "blokerer angrebet fra {creature} med skjoldet {feeling}",
    "hvisker en besværgelse over {object} og lukker øjnene",
    "nikker {feeling} til {other}",
    "skuer ud over {place} {feeling}",
    "trækker kappen tættere om sig i vinden fra {place}",
    "drager sit sværd og går {direction}",
    "samler troppen og peger {direction}",
    "lytter efter lyde fra {creature}",
    "tænder en lanterne og kigger {direction}",
    "knuger {object} mod brystet",
    "rejser sig {feeling} og griber sit våben",
    "studerer kortet over {place} {feeling}",
    "deler brød ud til de {adj} rejsende",
    "spejder efter {creature} fra en klippe",
    "binder hesten ved {place} {feeling}",
    "tegner en rune i støvet {feeling}",
    "deler {food} ud til hele følget",
    "pudser sit skjold {feeling}, til det skinner",
    "slår lejr i udkanten af {place}",
    "klatrer op i det højeste træ for at spejde efter {creature}",
    "vander hestene ved floden nær {place} {feeling}",
    "gemmer sig for {creature} bag en væltet vogn",
    "affyrer en advarselspil mod {creature}",
    "synger en gammel sang om {place}",
    "lapper sin kappe med sytråd fra {place}",
    "smager {feeling} på den kolde {food}",
    "øver sig i at bruge {object}",
    "bærer den sårede {other} {direction}",
    "kaster en sten efter {creature} og rammer",
    "ruller kortet over {place} sammen og rejser sig {feeling}",
    "holder vagt ved {place}, mens {other} sover",
    "skriver et brev til {other} i lyset fra bålet",
    "fodrer sin falk med rester af {food}",
    "sliber sit sværd {feeling}",
    "kigger op mod stjernerne over {place}",
    "vækker de andre lige før {time}",
    "stiller sig i vejen for {creature} med hævet skjold",
    "prøver {feeling} at tænde bål i silende regn",
    "leder efter spiselige bær langs stien til {place}",
    "hugger brænde til natten {feeling}",
    "kaster en line over kløften ved {place} {feeling}",
    "studerer runerne på {object}",
    "lytter med øret mod jorden efter {creature} {feeling}",
    "gemmer kortet over {place} i støvlen",
    "spejder fra bakketoppen {direction}",
    "reparerer sit skjold efter slaget mod {creature}",
    "øver et nyt hug med træsværdet sammen med {other}",
    "hilser vagterne ved {place} med et kort nik",
    "trækker {other} i skjul bag muren i {place}",
    "deler den sidste vandsæk med {other}",
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
    "[Kontekst: Sneen falder over {place}, mens {other} deler den sidste {food} ud.]",
    "[Kontekst: Et bud er ankommet til {place} med nyt om {creature}.]",
    "[Kontekst: {other} har mistet {object} i {place} og tør ikke sige det til de andre.]",
    "[Kontekst: Ved {time} skal {object} lægges tilbage i {place}, ellers vågner {creature}.]",
    "[Kontekst: Følget fejrer en sejr i {place} med sang og {food}.]",
    "[Kontekst: {other} vogter broen ved {place}, og ingen kommer forbi uden et godt svar.]",
    "[Kontekst: Kortet er brændt, og kun {other} kan huske vejen gennem {place}.]",
    "[Kontekst: En storm har drevet følget i ly i {place}, hvor noget lurer i mørket.]",
    "[Kontekst: {other} træner de unge i {place}, mens {creature} nærmer sig.]",
    "[Kontekst: Det er {time}, og vagterne i {place} har set lys ude i tågen.]",
    "[Kontekst: Vinteren er kommet tidligt, og følget må gennem {place} før {time}.]",
    "[Kontekst: {other} har udlovet en dusør for {object}, og alle i {place} taler om det.]",
    "[Kontekst: Broen ved {place} er styrtet sammen, og kun {creature} kender den anden vej.]",
    "[Kontekst: En maskeret fremmed har efterladt {object} ved porten til {place}.]",
    "[Kontekst: Følget har sejlet i {number} dage og øjner nu {place} i horisonten.]",
    "[Kontekst: Freden i {place} afhænger af, at {object} kommer tilbage før {time}.]",
]


# ---------------------------------------------------------------------------
# SAET 2 - finetune (samme univers, men med genkendelige jokes/referencer:
# gaming, internet og pop/musik). Stadig laesbart som en historie.
# ---------------------------------------------------------------------------
JOKE_SAYS = [
    "Questen i {place} er ren speedrun, kom nu, {other}.",
    "{creature} dropper garanteret legendary loot, lad os farme den.",
    "Jeg er level {number}, den boss er piece of cake.",
    "Kongen streamer fra {place} ved {time}, glem ikke at like.",
    "Vi mangler bare {object} for at complete the main quest.",
    "Hold da op, {place} har vild respawn-rate på fjender.",
    "Min mana er på nul efter {creature}, jeg looter lige en potion i {place}.",
    "{other} gik AFK midt i kampen mod {creature}, klassisk.",
    "Skal vi rushe {place} eller tage den stealth?",
    "Fighten mod {creature} er pure endgame content, fr.",
    "Jeg har grindet xp i {place} hele natten.",
    "Pro tip: bloker {creature} og counter når den lagger.",
    "Vi skal have {object} før serveren lukker ved {time}.",
    "Lyt til den nye banger mens vi rider mod {place}.",
    "Lootet fra {place} er straight up cracked, {other}.",
    "GG til holdet, vi clearede {place} uden at dø.",
    "Jeg fulgte en guide, men {creature} er stadig svær.",
    "Kan nogen heale? Jeg er på {number} hp mod {creature}.",
    "{creature} har for meget HP, nerf den.",
    "Jeg har lige unlocked en ny skill mod {creature} - watch this, {other}.",
    "Vi wipede på {creature} igen, seriøst?",
    "Hvem har aggroet {creature}?! Ikke mig.",
    "Loot-tabellen for {place} er straight bugged.",
    "Jeg buffer holdet, så pusher vi {place}.",
    "Daily questen i {place} giver dobbelt xp indtil {time}.",
    "Skiftede til min crit-build før {creature}-fighten.",
    "Brb, jeg skal lige crafte {object}.",
    "Vi tager {place} på hardcore, {other} - ingen respawns.",
    # Minecraft
    "Hørte I det? Det lød som en creeper bag {place}.",
    "Jeg har fundet diamanter under {place}, sig det ikke til {other}.",
    "Byg en mur af cobblestone rundt om {place} før {time}.",
    "Min hakke knækkede midt i {place}, klassisk.",
    "Vi skal bruge mere redstone til fælden mod {creature}.",
    "Ingen graver lige ned i {place}! Det er reglen, {other}!",
    "Ender-dragen er ingenting mod {creature}, tro mig.",
    "Jeg har fuld diamond armor, lad {creature} bare komme.",
    "Nogen har set min hest? Jeg parkerede den ved {place}.",
    "Det er ikke lava, sagde {other}. Det var lava.",
    "Vi bygger en nether portal i {place}, hvem har obsidian?",
    "En landsbyboer snød mig: {number} smaragder for en {food}!",
    # Roblox
    "Den her obby gennem {place} er umulig, jeg er dead seriøs.",
    "{other} fik banket hele serveren i {place}, W gamer.",
    "Min avatar har federe drip end {other}, bare vent og se.",
    "Prøv det nye map i {place}, det er faktisk sygt godt.",
    "Nej, jeg giver dig ikke mine robux, {other}.",
    # skole og kantine
    "Kantinen har {food} i dag, LØB, ikke gå.",
    "Nogen har taget den sidste {food} fra min taske, jeg kræver retfærdighed.",
    "Læreren tegnede {creature} på tavlen, og kridtet knækkede af bar drama.",
    "Vi har prøve i {place}-historie ved {time}, har nogen læst?",
    "Min madpakke er kun {food} igen, wanna trade?",
    "Sidder vi bagerst, kan {creature} ikke se os, det er matematik.",
    "Gruppearbejdet om {object} endte som altid: jeg lavede det hele.",
    "Fem minutter mere, sagde {other} - for {number} timer siden.",
    "Deler I en {food}, er I venner for livet, det er reglen.",
    "Nogen har skrevet SKAK PÅ FREDAG med kridt på døren til {place}.",
    "Energidrik plus {food} er morgenmad for champions.",
    "Hvem har opvasken efter raidet på {place}? Ikke mig, {other}, jeg kaldte det først.",
    "Pausen er hellig - selv {creature} venter til efter frokost.",
    "Patch notes: {creature} er blevet nerfet, {food} er blevet buffet.",
    "Min ping er over {number} hundrede - spiller jeg fra {place} eller hvad?",
    "Tutorial-bossen i {place} tog mig {number} forsøg. Ingen kommentarer.",
    "Free loot i {place}! Det er helt sikkert ikke en fælde.",
    "Jeg spiller healer mod {creature}, så I siger ALLE pænt tak - også dig, {other}.",
    "Ny skin til {creature}? Instant køb.",
    "Serverens økonomi er ødelagt: {number} guld for en {food}?!",
    "Achievement unlocked: overlevede {place} uden at miste madpakken.",
    "Er {creature} elite-mob eller boss? Svar hurtigt, den kigger på mig.",
    "Min build er glaskanon: et hit fra {creature}, og jeg respawner i {place}.",
    "AFK 2 min, mor kalder, der er {food}.",
    "Ingen spoilers! Jeg er ikke nået til {place} endnu.",
    "Sidequest: find {other}s madpakke. Belønning: venskab.",
    "Hvem har inviteret {creature} med i guildet?!",
    "Lobbyen er fuld af NPC'er, vi rykker til {place}.",
    "Det er ikke camping, det er strategisk placering ved {place}.",
    "Daily reward i kantinen: {food}. Streak: {number} dage.",
    "Speedrun-rekorden til {place} er slået - af pedellen.",
    "GG go next, {creature} var bugged alligevel.",
    "Jeg har en teori: {place} er bare {city} med bedre grafik.",
    "Skolens wifi lagger mere end kampen mod {creature}.",
    "Byttehandel: to {food} for en plads på holdet mod {creature}.",
    "Minimap'et siger, vi er fremme ved {place}. Minimap'et lyver, {other}.",
    "Respekt til {other}, der clutchede mod {creature} med 1 hp.",
    "Husk hydrering, siger {other} og åbner sin tredje energidrik.",
]

JOKE_DOES = [
    "caster fireball mod {creature} og råber GG",
    "laver et sick 360 no-scope med buen mod {creature}",
    "tjekker sin loadout før raidet på {place} {feeling}",
    "emoter foran {creature} for at flexe",
    "spammer heal-knappen og survival mod {creature}",
    "looter hele {place} for skrald og sælger det {feeling}",
    "tager en quick selfie med {object} til sit feed",
    "rage-quitter da {creature} one-shotter ham",
    "laver en TikTok-dans midt i {place}",
    "drikker en energidrik og pusher mod {place} {feeling}",
    "kalder sit team sammen i {place} for en pep-talk",
    "alt-tabber for at google hvordan man slår {creature} {feeling}",
    "kiter {creature} rundt om {place}",
    "spammer combo-tasterne mod {creature} {feeling}",
    "tjekker minimap'et over {place} {feeling}",
    "saver sit game lige før {creature} {feeling}",
    "trash-talker {other} i voice chat",
    "looter en chest og finder {object}",
    "placerer en crafting table midt i {place} {feeling}",
    "graver en hemmelig base under {place} {feeling}",
    "løber fra en creeper gennem hele {place}",
    "bygger et tårn af dirt over {place} for at flexe for {other}",
    "spiser en gylden {food} for ekstra hearts",
    "AFK-farmer xp ved {place} med god samvittighed",
    "viser sit nye Roblox-skin frem for {other}",
    "falder gennem map'et i {place} og griner",
    "bytter sin {food} for {other}s {food} i kantinen",
    "gemmer en {food} i penalhuset til {time}",
    "skriver GG til {other} på tavlen med tre farver kridt",
    "øver sin sejrsdans til efter {creature}-fighten",
    "sætter en skål {food} frem som bait til {creature}",
    "opdaterer sin tier-list over {place}-bosser",
    "planlægger weekendens LAN-party i {place}",
    "napper en chokoladekiks mens {other} ser den anden vej",
    "opdaterer sin loadout foran porten til {place}",
    "prøver at parere {creature} med en bakke {food}",
    "farmer likes med et billede af {object}",
    "sætter sit flag ved {place} og claimer området",
    "tjekker droprates for {object} på sin telefon",
    "bytter sig til {food} på klassens sorte marked",
    "laver en sejrsdans ved {place}",
    "quicksaver før den svære samtale med {other}",
    "nørkler med sin build til kampen mod {creature}",
    "streamer sin frokost: dagens {food}",
]

JOKE_CONTEXT = [
    "[Kontekst: {other} samler et squad til et raid på {place} for at få {object}.]",
    "[Kontekst: Det er sidste dag i seasonen, og alle vil nå {object} før {time}.]",
    "[Kontekst: {creature} er den nye boss, og chatten håber på legendary loot.]",
    "[Kontekst: Holdet grinder dailies i {place} for at nå max level før {time}.]",
    "[Kontekst: Det er guild raid-aften, og {creature} er den sidste boss.]",
    "[Kontekst: {other} streamer hele runet gennem {place} live til chatten.]",
    "[Kontekst: Klassen er på lejrtur i {place}, og {other} har glemt sin madpakke.]",
    "[Kontekst: Serveren i {place} er nede, og alle giver {other} skylden.]",
    "[Kontekst: Kantinen har {food} i dag, og køen går helt om til {place}.]",
    "[Kontekst: Squaddet bygger en base i {place}, mens {creature} spawner udenfor.]",
    "[Kontekst: Det er fredag, og alle vil hjem og game i stedet for at finde {object}.]",
    "[Kontekst: {other} har væddet en {food} på at kunne besejre {creature} solo.]",
    "[Kontekst: Ny season er landet, og alle grinder det nye map i {place}.]",
    "[Kontekst: Klassens turnering afgøres i {place} - vinderen får {food} i en uge.]",
    "[Kontekst: {other} har lovet chatten at klare {creature} solo inden {time}.]",
    "[Kontekst: LAN-party i {place}: {number} skærme og en enkelt forlængerledning.]",
]


# ---------------------------------------------------------------------------
# SAET 3 - kogt/brainrot (maksimalt kaos). Curated og PG-13: kun teen-venlig slang.
# ---------------------------------------------------------------------------
COOKED_NAMES = [
    "Skibidi-ridderen", "Sigma-dragen", "Gigachad-kongen", "Rizz-troldmanden",
    "Ohio-goblinen", "NPC-væbneren", "Based-heksen", "Mid-skelettet",
    "W-ridderen", "Cringe-trolden", "Sheesh-barden", "Goofy-vagten",
    # italiensk brainrot-kanon
    "Tralalero Tralala", "Bombardiro Crocodilo", "Ballerina Cappuccina",
    "Cappuccino Assassino", "Bombombini Gusini", "Chimpanzini Bananini",
    "Brr Brr Patapim", "Lirili Larila", "Tung Tung Tung Sahur",
    "Boneca Ambalabu", "La Vaca Saturno Saturnita", "Trippi Troppi",
    "Frigo Camelo", "Glorbo Fruttodrillo",
]

COOKED_SAYS = [
    "kongen af {place} har mad rizz no cap fr fr",
    "skibidi {place} go BRRR, only in Ohio",
    "L plus ratio plus du faldt i {place} bro",
    "{creature} er straight up mid, touch grass",
    "vi farmer {object} fr fr sheeesh sigma grindset",
    "{other} er en NPC, kig på den goofy ahh walk gennem {place}",
    "rizz level over 9000 i {place} no cap",
    "based king {other}, W W W, det er bare facts",
    "{creature} prøvede at rizze dronningen, sus",
    "skibidi toilet boss i {place} igen, brainrot",
    "real ones looter {object} ved {time} only",
    "ohio rizz {creature} sigma alpha gigachad sheesh",
    "bro {other} er så cringe det er crazy fr",
    "we move mod {place}, ingen cap, ren W energi",
    "han mewing mens {creature} angriber, goofy",
    "skibidi sigma rizz {object} sheeesh naah jk",
    "{place} er straight up Ohio, ingen lyver",
    "han droppede en hard W i {place} sheesh",
    "skibidi rizz party i {place} ved {time} lets gooo",
    "{other} er en NPC fr fr touch grass",
    "gigachad energi only, vi farmer {object}",
    "no cap {creature} er goofy ahh mid, sig det, {other}",
    "we cooking i {place} skibidi style",
    "ratio plus L plus {creature} er cringe",
    "sigma sigma boy farmer xp i {place}",
    "based king dropper {object} sheeesh",
    # italiensk brainrot
    "tralalero tralala, hajen med Nike-sko tager en W i {place}",
    "bombardiro crocodilo flyver lavt over {place}, mamma mia",
    "ballerina cappuccina snurrer gennem {place} med kaffekoppen højt",
    "cappuccino assassino er for hurtig for {creature}, ingen ser espresso-klingen",
    "tung tung tung sahur banker på din dør ved {time}, pas på",
    "boneca ambalabu siger frøens visdom hviler over {place}",
    "brr brr patapim vogter sin jungle i {place}, capisce",
    "chimpanzini bananini splitter en banan og hele {place} klapper",
    "lirili larila går baglæns gennem {place} i sine sandaler",
    "la vaca saturno saturnita kredser om {place}, muuu fra rummet",
    "trippi troppi troppa trippa, spørg bare {other}, det er lore",
    "mamma mia, {creature} har ingen chance mod bombardiro",
    "assassino spiller det italienske theme mens {place} brænder",
    "tralalero siger til {other}: tre sko er bedre end to, capisce",
    "sahur sahur sahur, klokken er {time} og trommen lyder",
    "bombombini gusini letter fra {place} med fuld last, ciao",
    "frigo camelo fragter kold {food} gennem ørkenen, respekt",
    "glorbo fruttodrillo gemmer sig i frugtkurven i {place}",
    "italiensk brainrot er ikke en fase, {other}, det er en livsstil fr",
    "kun real ones kender loren om {other} og hajen",
    "skibidi og tralalero i samme rum? {place} er ikke klar",
    "ballerina cappuccina elsker cappuccino assassino, det er kanon i {place}",
    "tung tung sahur og boneca ambalabu tog {place} ved {time}, duoen ingen bad om",
    "vaca saturno ser alt fra oven, selv dig, {other}",
    "patapim patapim, junglen kalder og {creature} svarer",
    "bananini uden banan er stadig bananini, forklar det for {other}",
    "tralalero har {number} Nike-sko nu, loren udvikler sig",
    "bombardiro og bombombini flyver i formation over {place}, cinema",
    "cappuccina mia, dansen i {place} var ren kunst",
    "tung tung tung sahur glemte at banke hos {other} ved {time}, utilgiveligt",
    "boneca ambalabu har taget {object}, og kun frøen ved hvorfor",
    "lirili larila trådte ud af tiden og ind i {place}",
    "vaca saturno drak alt i {place}? nej vent, det var mælk",
    "patapim kalder mødet med {creature} jungle-diplomati, jeg kalder det kaos",
    "assassino afleverede en espresso til {other} og forsvandt mod {place}, klassisk",
    "brainrot-rådet har stemt: {place} er officielt Ohio nu",
    "chimpanzini fandt en gylden banan i {place}, det er lore-tungt",
    "det er ikke støj, det er tralalero-remixet fra {place}",
    "sigma-opgøret i {place} endte uafgjort, alle mewede",
    "W-alliancen holder, så længe der er {food} i {place}",
    "trippi troppi byttede havet ud med {place}, vildt vibe-skifte",
    "frigo camelo krydsede {place} med is i maven og frost i ryggen",
    "glorbo siger frugtkurven i {place} er hellig grund",
    "der er ingen aura tilbage i {place}, {other} tog det hele",
    "skibidi-alarmen lød ved {time}, alle i dækning",
    "giga-lore: {creature} og tralalero var venner engang",
    "rizz-økonomien i {place} er kollapset, sell sell sell",
    "det her er ikke engang min endelige brainrot-form, {other}",
]

COOKED_DOES = [
    "laver en Ohio backflip over {place} SHEEESH",
    "hitter en sigma pose foran {creature} {feeling}",
    "råber SKIBIDI så hele {place} ryster",
    "dabber på {creature} efter en clean W {feeling}",
    "mewer intenst mens {place} brænder",
    "spammer L plus ratio i hele {place} {feeling}",
    "rizzer {creature} med ren sigma energi {feeling}",
    "gør en goofy ahh dans på bordet i {place}",
    "tager {object} og råber ONLY IN OHIO",
    "kigger direkte i kameraet som en chad, mens {creature} nærmer sig",
    "hitter en griddy på {creature}s lig",
    "skibidi-danser midt i {place} {feeling}",
    "råber W RIZZ så {place} ryster",
    "mewer mens han looter {object}",
    "gør en sigma staredown med {creature}",
    "dabber på {other} efter en clean ratio",
    "flyver en bombetur over {place} som bombardiro",
    "snurrer en piruette med espressokop for {other}",
    "banker tre gange på alle døre i {place} som tung tung sahur",
    "lister gennem {place} med espresso-klingen som assassino",
    "peger mod himlen over {place}, hvor la vaca saturno kredser",
    "reciterer hele loren om {other} udenad",
    "går baglæns som lirili larila og forsvinder i tågen over {place}",
    "trommer sahur-rytmen på sit skjold ved {time} {feeling}",
    "deler en banan med chimpanzini bananini og {other}",
    "råber MAMMA MIA så det giver ekko i {place}",
    "tegner brainrot-slægtstræet med kridt på muren i {place}",
    "laver lydeffekten tralalero tralala helt perfekt for {other}",
    "holder et foredrag om hvorfor {creature} er mid {feeling}",
    "bygger et alter af {food} til vaca saturno",
    "øver bombardiro-lyden ved {place} til perfektion",
    "sorterer sine {number} Nike-sko efter aura",
    "holder aura-auktion midt i {place}",
    "afviser {creature} med et enkelt nonchalant nik",
    "opdaterer lore-tavlen i {place} med dagens kanon",
    "gemmer sin espresso for assassino bag {object}",
    "opkræver fanum tax af {other}s {food}",
    "kalder på vaca saturno med et spejl fra {place}",
    "sætter {object} på tronen og erklærer det based",
    "prøver at forklare loren for {creature} og {other}, forgæves",
]

COOKED_CONTEXT = [
    "[Kontekst: {other} og squaddet farmer skibidi-rizz i {place}, ingen cap.]",
    "[Kontekst: Det er Ohio-finalen, og {creature} har mad rizz men er mid.]",
    "[Kontekst: Sigma-grindset er real, og {object} venter i {place} fr fr.]",
    "[Kontekst: Skibidi-squaddet rusher {place} for at få max rizz inden {time}.]",
    "[Kontekst: Det er Ohio o'clock, og {creature} har null rizz men max cringe.]",
    "[Kontekst: Sigma-grindset fortsætter i {place}, kun real ones er tilbage.]",
    "[Kontekst: Tralalero Tralala og Bombardiro Crocodilo mødes til det store opgør i {place}.]",
    "[Kontekst: Ballerina Cappuccina holder bal i {place}, og alle brainrots er inviteret.]",
    "[Kontekst: Tung Tung Tung Sahur vandrer gennem {place} ved {time} og banker tre gange.]",
    "[Kontekst: La Vaca Saturno Saturnita svæver over {place}, og ingen ved hvorfor.]",
    "[Kontekst: Cappuccino Assassino er set i {place}, og alle gemmer deres kaffe.]",
    "[Kontekst: Brr Brr Patapim har indkaldt hele junglen til krisemøde om {object}.]",
    "[Kontekst: Loren siger at {other} engang gav {creature} en high five i {place}.]",
    "[Kontekst: Det italienske brainrot-råd samles i {place} for at kåre den mest based.]",
    "[Kontekst: Aura-målingen i {place} slår ud, og alle mistænker {other}.]",
    "[Kontekst: Den store lore-konference afholdes i {place} ved {time}.]",
    "[Kontekst: Tralalero Tralala har mistet en Nike-sko i {place}, og nationen sørger.]",
    "[Kontekst: Fanum tax er indført i {place}, og ingen {food} er længere sikker.]",
]


# ---------------------------------------------------------------------------
# SAET 4 - danske far-jokes. Familien er samlet, og Far er i topform.
# ---------------------------------------------------------------------------
DAD_NAMES = [
    "Far", "Mor", "Sønnen", "Datteren", "Onkel Bent", "Tante Oda",
    "Bedstefar", "Bedstemor", "Naboen Preben", "Fætter Karl", "Kusine Sofie",
    "Farfar", "Morfar", "Moster Ulla", "Lillebror", "Storesøster",
]

DAD_SAYS = [
    "Hvorfor gik {creature} over vejen? For at komme over til {place}.",
    "Jeg ville fortælle en joke om {object}, men den er for lang.",
    "Banke banke på! Hvem der? Det er {creature} fra {place}!",
    "Tjener, tjener, der er {creature} i min suppe! Bare rolig, den spiser ikke ret meget.",
    "Hvad er forskellen på {food} og {food2}? Det finder vi ud af til aftensmaden.",
    "Min læge siger, jeg skal droppe {food}. Så nu spiser jeg det, hvor lægen ikke kan se det.",
    "Hvad får man, hvis man krydser {creature} med {creature2}? Et meget forvirret dyr.",
    "Hvad siger {creature}, når den kommer for sent? Undskyld, der var kø i {place}.",
    "Da jeg var på din alder, gik vi {number} kilometer gennem {place} - hver vej, i sne.",
    "Skal vi køre en tur til {city}? Nej? Godt, for der var alligevel ikke benzin på bilen.",
    "Jeg har lavet {food} på en ny måde. Den nye måde er, at jeg brændte det på.",
    "Hvorfor tog {creature} til {city}? Den ville se noget andet end {place}.",
    "Der er kun to ting, jeg ikke kan lave i køkkenet: {food} og alt det andet.",
    "Jeg googlede lige {creature}. Nu tror internettet, at vi skal have en.",
    "Hvis I rydder op på værelserne, får vi {food} i weekenden. Det er ikke en trussel, det er et tilbud.",
    "Kald mig bare kok: min {food} har fået {number} stjerner - af mig selv.",
    "Jeg faldt lige over {object} ude i garagen. Bogstaveligt talt.",
    "Vidste I, at {creature} kan høre {food} blive åbnet på {number} kilometers afstand? Det kan jeg også.",
    "I gamle dage hed {city} noget andet. Hvad? Det ved jeg ikke, men det var billigere dengang.",
    "Nu skal I høre om dengang, jeg næsten fangede {creature} i {place}. Igen? JA, igen.",
    "Sidste mand ud af {place} lukker og slukker!",
    "Hvem vil have {food}? Forkert svar: ALLE vil have {food}.",
    "Jeg siger det kun {number} gange mere: vi skal IKKE have {creature} som kæledyr.",
    "Turen til {city} tager tyve minutter. Med far bag rattet: ti.",
    "Ved I hvad {creature} siger til {creature2}? Ingenting, de har ikke hilst på hinanden.",
    "Min ryg lyder som en skål {food}, når jeg rejser mig. Knas, knas.",
    "Vi er der næsten. Hvor langt er næsten? Cirka {number} timer.",
    "Hvorfor tog myren til {city}? Den skulle besøge sin myre-moster.",
    "Hvad siger man til en trist {food}? Ingenting. Man spiser den, så den ikke er ked af det længere.",
    "Hvad kalder man {creature} i en elevator? Et pladsproblem.",
    "Hvordan ved man, at {creature} har været i køleskabet? Der er poteaftryk i smørret.",
    "Hvad siger {creature} på ferie i {city}? Endelig fred for {other}.",
    "Min GPS siger {city}, min mave siger {food}. Maven vinder.",
    "Der er to slags familier: dem der elsker {food}, og dem der tager fejl.",
    "Hvorfor må {creature} ikke spille kort? Den gemmer esser i pelsen.",
    "Jeg fandt {object} i sofaen. Ingen spørgsmål, tak.",
    "Sov godt, og lad ikke {creature} bide - det er dens job at prøve.",
    "Da jeg var barn, kostede {food} en krone. Og {city} lå tættere på.",
    "Hvis {other} spørger, så var det {creature}, der spiste den sidste {food}.",
    "Regel nummer {number}: Far bestemmer over grillen og fjernbetjeningen.",
    "Vi kører IKKE forkert. Vejen til {city} har bare flyttet sig.",
    "Hvad laver {creature} i haven? Forhåbentlig lugearbejdet.",
    "Halvdelen af {city} griner stadig af min joke om {creature}.",
    "Man takker ikke nej til {food}. Det står i loven. Paragraf {number}.",
    "Jeg er på diæt: kun {food} i hverdagene. Og i weekenderne.",
    "Kan {creature} flyve? Nej? Så er det ikke den, der har taget min {food}.",
    "Nyt fra garagen: {object} virker igen. Spørg ikke hvordan.",
]

# Faste ordspil (danske + engelske fra dad_jokes.csv). Et ordspil kan ikke faa
# pladsholdere uden at miste pointen - saa hver joke pakkes i en VARIABEL ramme
# nedenfor, og alle linjer ender alligevel med 1-2 pladsholdere.
DAD_FIXED_JOKES = [
    "Hvad hedder en bjørn uden tænder? En gummibjørn.",
    "Hvad kalder man en fisk uden øjne? En fsk.",
    "Hvad er det bedste ved Schweiz? Aner det ikke, men flaget er et stort plus.",
    "Hvorfor kan en cykel ikke stå selv? Fordi den er træt.",
    "Hvad sagde den ene væg til den anden? Vi mødes i hjørnet.",
    "Hvorfor gik skelettet ikke til fest? Det havde ingen krop at følges med.",
    "Jeg læste en bog om anti-tyngdekraft. Den var umulig at lægge fra sig.",
    "Hvad kalder man en hund der kan trylle? En labracadabrador.",
    "Hvorfor er spøgelser så dårlige til at lyve? Man kan se lige igennem dem.",
    "Hvad sagde nul til otte? Fedt bælte.",
    "Jeg ville fortælle en kemi-joke, men jeg får nok ingen reaktion.",
    "Hvorfor tog computeren til lægen? Den havde en virus.",
    "Hvad kalder man en snemand i juli? En vandpyt.",
    "Hvorfor gik bageren konkurs? Han manglede dej.",
    "Hvad kalder man hvaler der spiller musik? Et orka-ster.",
    "Man skal aldrig stole på trapper. De er altid ude på noget.",
    "Hvad sagde bonden, da han mistede sin traktor? Hvor er min traktor?",
    "Er du træt? Hej Træt, jeg er Far.",
    "Far, kan du lave mig en pandekage? Bum, nu er du en pandekage.",
    "Hvad er brunt og klæbrigt? En pind.",
    "Hvorfor skal man aldrig skændes med DNA? Det har altid et godt gen-svar.",
    "Hvornår er en dør ikke en dør? Når den står på klem.",
    "Hvad sagde den store skorsten til den lille? Du er for ung til at ryge.",
    "To muffins sad i ovnen. Den ene sagde: puha, her er varmt. Den anden: AAAH, en talende muffin!",
    "Min ven faldt ned fra en 15 meter høj stige. Heldigvis stod han på nederste trin.",
    "Jeg har en joke om papir, men den er lidt tynd.",
    "Jeg har også en joke om elastik, men den trækker ud.",
    "Hvad laver en løve i teatret? Brøler.",
    "Hvad sagde tryllekunstneren til fiskeren? Vælg en torsk, en helt tilfældig torsk.",
    "Elektrikeren kom hjem midt om natten. Konen spurgte: hvor har du været? Han svarede: Watt?",
    "Hvad er gult og kan ikke svømme? En gravko.",
    "Hvad er rødt og dårligt for tænderne? En mursten.",
    "Hvad er usynligt og lugter af gulerødder? Kaninprutter.",
    "Der findes tre slags mennesker: dem der kan tælle, og dem der ikke kan.",
    "Hvorfor tager skeletter aldrig i svømmehallen? Vandet går lige igennem dem.",
    "Hvad hedder en ko uden ben? Hakkebøf.",
    "Man kan ikke trække en slange i benet.",
    "Hvad er sort, hvidt og rødt over det hele? En pingvin med solskoldning.",
    "Hvorfor fik uret ballade i skolen? Det talte i timen.",
    "Hvad siger en snegl på ryggen af en skildpadde? Wiiii!",
    "Hvad sagde det ene øje til det andet? Der er noget mellem os, og det lugter.",
    "Hvad hedder en snemand med solbriller? En vandpyt med attitude.",
    "Jeg ville lave en joke om pizza, men den er for cheesy.",
    "Hvad får man, når man krydser en snemand med en vampyr? Forfrysninger.",
    "Min søn spurgte, hvor solen bliver af om natten. Det gik op for ham i morges.",
    "Hvad er forskellen på en giraf? Jo længere op, jo mere hals.",
    "To høns gik over vejen. Så var der ingen høns på den anden side... vent, jo, to.",
    "Hvorfor gik champignonen til fest? Den var en sjov svamp. En FUN-gi. Forstår I?",
    "Hvad er høfligt og bor i skoven? Please-svampen... nej vent, den hedder pigsvampen.",
    "Hvorfor må man ikke fortælle hemmeligheder på en mark? Kartoflerne har øjne, og majsen har ører.",
    "What do you call a twig that won't stop looking in the mirror? A narcissistick.",
    "What time did the man go to the dentist Tooth hurt-y.",
    "What has ears but cannot hear? A cornfield!",
    "What did the 0 say to the 8? Nice belt!",
    "Why do peppers make such good archers? Because they habanero.",
    "What lights up a soccer stadium? A soccer match.",
    "What's the best way to watch a fly fishing tournament? Live stream.",
    "The glass eyeball manufacturer is having a promotion An eye for an eye.",
    "When one door closes, another door opens. You are being ejected through the air lock.",
    "A few weeks ago I ordered a box to store my money and a set of speakers online. They arrived today, safe and sound.",
    "If Drake owned a breakfast cereal franchise, what would it be called? OV O's!",
    "We've been trying to organize a Fear of Commitment workshop. But we just can't seem to nail down a date.",
    "What happens when you go to the bathroom in France? European.",
    "I got fired from my job of making leaf blowers... because they all sucked.",
    "Why is Windows software so predictable? You can see right through it.",
    "Why couldn't the bicycle stand up by itself? It was two tired.",
    "Mrs. Goat : Honey, we're going to have a baby! Mr. Goat : You're kidding.",
    "My New Year's Resolution is to be more humble... Which should be easy as I'm already *really* good at it!",
    "Do you know the joke of little Jef in the bathroom? Me neither, the door was locked.",
    "Took a job in a fertiliser factory... It's my first daytime job where they give me nitrates.",
    "Q: What's the difference between a hippo and a Zippo? A: One is very heavy, the other is a little lighter.",
    "Chun Li: Can I ask you a question? Ken: SURE-YOU-CAN!!!",
    "What do you call a potato that makes videos for the internet? A YouTUBER.",
    "I went to buy some camouflage trousers the other day but I couldn't find any.",
    "I asked the librarian if she knew who authored any books on dinosaurs. She said, \"Try Sarah Topps.\"",
    "Two cannibals are eating a clown. One says to the other: \"Does this taste funny to you?\"",
    "I saw a guy at the beach yelling, \"Help! Shark, help!\" I just laughed because I knew the shark wasn't going to help him.",
    "My dad has the heart of a lion... And a lifetime ban at the zoo.",
    "I needed some change in my life So I decided to start a coin collection. I know it seems odd but it makes cents to me.",
    "Whenever I do crown molding I can never get the corners to match up quite right I guess I have coping issues.",
    "Who was the dad's favorite comic book hero? The Pun-isher.",
    "It's good thing tall people like me don't grow on trees friend: too bad shorter people could use the shade!",
    "Carp is about to hit the fan. That's right, I'm going fishing in a helicopter!",
    "The photophobiac's power just went out. He is delighted.",
    "What did the duck say when it bought chapstick? \"Put it on my bill!\"",
    "Did you hear the one about the kid who started a business tying shoelaces on the playground? It was a knot-for-profit.",
    "What do you call someone with no body and no nose Nobody knows.",
    "Why did the tie not laugh at the other tie's jokes? They were knot funny.",
    "How was the first digital sound created? Someone snapped their fingers.",
    "How can you tell the difference between a horse and a pig? The horse is the one that doesn't look like a pig.",
    "What was the inventor of suspenders awarded for their discovery? The no-belt prize.",
    "Can of coke fell on a mans head from a high building Fortunately he survived because it was just a soft drink!!!",
    "What's the motto of the American Writers Guild? YOU ESSAY! YOU ESSAY!",
    "What do you call a locomotive carrying bubble gum? A chew chew train.",
    "I go nuts for washers You know what I'm talking a bolt?",
    "What do you call corn that joins the army? Kernel.",
    "If I'm being subjective, the greatest sci-fi show of all time is Dr. Who. If I'm being objective, it's Dr. Whom.",
    "What do you call a row of dolls burning on a grill? Barbie Queue.",
    "Where do monsters like to party? At the g-rave-yard.",
    "I visited the National Air and Space Museum. I believe the title is misleading because it's actually full of stuff.",
    "What do you call a slightly injured Cow? Ow.",
    "What kind of egg did the evil chicken lay? A deviled egg.",
    "Recent cyber security breaches are discovered due to their rapid deployment. The hackers are always Russian.",
    "My parents raised me as an only child. Which really annoyed my younger brother.",
    "Have you guys tried lighting pure oxygen on fire? You'll have a blast.",
    "My friends secretly downloaded a 700MB exe file into my laptop. I think it's a huge setup.",
    "Why are fish so smart? They live in schools!",
    "Why do golfers need two pairs of pants? In case they get a hole-in-one.",
    "What T.V. Channel will never air the sitcom Scrubs? TLC; Because, they don't want, no scrubs.",
    "Why was the T-Rex Cafe always hiring? No matter what, they always seemed a bit \"short handed\".",
    "Why isn't every man in a red suit with a beard Santa? Because correlation doesn't imply Claus-ality.",
    "Spring is here! I got so excited that I wet my plants.",
    "I hate Velcro. It's a rip off.",
    "What's black, and white, and OBVIOUSLY belongs in NeverLand? Pan, duh.",
    "I never buy pre-shredded cheese. Because doing it yourself is grate.",
    "What did the knight say to the trainee who broke the blade off his sword? You need to get a handle on that.",
    "I decided not to keep the skunk I bought and returned it to the vendor because... it didn't make scents.",
    "I asked my dad for his best dad joke and he said, 'You.'\"",
    "What kind of math does an Owl like to solve? Owl-gebra.",
    "Why do the Ninja Turtles attack Shredder 4 on 1? Because their master is a rat.",
    "Wanna hear a great cat joke? Just kitten. I don't have one.",
    "I totally understand why people work at fragrance factories... Makes scents...",
    "What's the longest sentence in the English language? 'I do'.",
    "What is DJ Khaled's favorite number? Eleven. Because it's a 1. And another 1.",
    "You should get a candle. If you have a smelly room I would recommend getting a candle. It just makes scents.",
    "I have an inferiority complex, but it's not a very good one.",
    "I was that bad in geography That I couldn't even find the class.",
    "What is the name of Daniel Craig's last movie? Probably, \"Bond Voyage.\"",
    "Why did the man ask his boss for more salad? He thought he was due a celery increase.",
    "Imagine you're stuck in the ocean, surrounded by sharks. What do you do to save your life? Stop imagining.",
    "What's a rappers favourite toy? A yo yo!",
    "Why don't skeletons ever go trick or treating? Because they have no body to go with.",
    "Why was the robot so tired after his road trip? He had a hard drive.",
    "I can't remember if I had a Dalmatian or leopard when I was a kid. Either way, my memory of my pet is kinda spotty.",
    "Did you hear the dull story about the Japanese policeman's hatchet? It was an anti-crime axe.",
    "Today I learnt koala bears aren't actual bears. They're marsupials. I guess they don't meet the koalafications.",
    "What do you get when you cross a pig and a dinosaur? Jurassic Pork.",
    "Without a doubt, my favorite Robin Williams movie is Mrs. Fire.",
    "Why can't any of the others elements ever get in touch with Sodium? Because it's always NA.",
    "A cop started crying while he was writing me a ticket. I asked him why and he said, \"It's a moving violation.\"",
    "What kind of instrument can you make with a gherkin? A piccolo.",
    "What did Babe Ruth name his pet pig? The Great Hambeano...",
    "Fun fact! Irish soups only use 239 beans If they used one more, it would be two-fahrty...",
    "What do lazy farmers grow? Couch potatoes!",
    "Why do squirrels swim on their backs? To keep their nuts dry.",
    "[joke about Minecraft] Why can't the Ender Dragon read a book? Because she always starts at the End.",
    "I used to be a motorcycle courier... Man those things are heavy..",
    "What do you call a boomerang that won't come back? A stick.",
    "What do you call a Nose that has a job as a Health Inspector? A Scenter for Disease Control.",
    "In the 5th month of every year, my aunt let's her pigs in the field.... It's mayham!",
    "What's orange and sounds like a parrot? A carrot!",
    "What does the dentist of the year get? A little plaque.",
    "Don't forget to tip your server, that's what they always say... But then I got fired from the Google Datacenter.",
    "What happens when you attach stew to a blimp? Soup rise!",
    "What do you get from a pampered cow? Spoiled milk.",
    "The wedding was so beautiful, even the cake was in tiers.",
    "What invention allows us to see through walls? Windows.",
    "Did you know: If you stacked every elephant on earth on top of each other... ...most of them would fall.",
    "I'd like to return this. It's unused. Clerk: Sir, this is your diploma. Me: Cash is fine.",
    "How did the hipster burn his mouth? He ate the pizza before it was cool.",
    "The Flat-Earth Society is now recruiting new members... We have chapters all around the world.",
    "Why are soldiers so tired at the beginning of April? They just had a 31 day March.",
    "What does a clock do when it's hungry? It goes back for seconds.",
    "How do you measure the mass of a red hot chili pepper. Give it a weigh. Give it a weigh. Give it a weigh now.",
    "How much weight do you lose after having a wisdom tooth taken out? A molar mass.",
    "What's your New Years resolution? Mine is 3120x1440. I got a new phone.",
    "Why did the traffic light turn red? You would too if you had to change in the middle of the street.",
    "What did the nose tell the finger? Stop picking on me!",
    "What's Rectangle, red and bad for your teeth? A Brick!",
    "How do the trees get on the internet? They log on.",
    "I don't suffer from insanity-I enjoy every minute of it.",
    "Why do those with open wounds deserve love the most? Because they're going to need a suture soon!",
    "Why are pediatricians always so angry? Because they have little patients.",
    "A guy walked into a bar, and lost the limbo contest.",
    "What kind of bird works on a construction site? A crane.",
    "Why are fish easy to weigh? Because they have their own scales.",
    "Where do books hide when they're afraid? Under their covers.",
    "If two vegetarians get in an argument, is it still called beef?",
    "Soaking a twig in coke is nice, but soaking a twig in fanta... Fanta stick.",
    "What did the shy pebble wish for? That he was a little boulder.",
    "I don't know what animal the year 2020 is in the Chinese calendar but I'm pretty sure it has rabies.",
    "So this dude is like, \"Where are you going with all that Element 83?\" and I was like \"None of your bismuth\"",
    "What's it called when a crab is walking to it's part time job? A side hustle.",
    "I was going to go on an expensive vacation with a classical pianist, but he was too baroque.",
    "How is working at McDonald's like being an archaeologist in Athens? Either way, you end up smelling like ancient grease.",
    "What's Harry Potter's favorite way to get down a hill? Walking. JK, Rolling.",
    "A lot of people don't like Mondays But 48 hours ago was a sadder day.",
    "Where do boats go when they're sick? To the boat doc.",
    "What happened to the cat after she swallowed a ball of yarn? She had mittens.",
    "Her : I'm leaving . I'm sick of you wearing a different t-shirt every hour . Me : Wait . I can change .",
    "How does Darth Vader like his bagels? On the dark side.",
    "My neighbor failed the entrance exam for butcher's school. He didn't make the cut.",
    "I have a fear of speed bumps. I'm slowly getting over it.",
    "A kind man gave me some gold for free But I am a fool, for upon closer inspection I discovered it was pyrite.",
    "What do you get when you cross a llama and a sweet potato? You get a Yyama!",
    "Why do bananas wear sunscreen? Because they peel.",
    "What do you call a pear thats a dad? I don't really know but it should be apparent.",
    "Why does a Moon-rock taste better than an Earth-rock? Because it's a little meteor.",
    "What did one elevator say to the other elevator? I think I'm coming down with something.",
    "How do you get two whales in a car? Start in England and drive West.",
    "What do witches ask for at a hotel? Broom service.",
    "My boss told me to have a good day, so I went home!",
    "Why is it better to smell roses and lemons than poop? It's just plain common scents.",
    "Do you know the story about the chicken that crossed the road? Me neither, I couldn't follow it.",
    "Did you hear the one about the roofer with a perfect safety record? He never had a shingle accident.",
    "Do not use \"beef stew\" as a computer password. It is not stroganoff.",
    "My Friend Told Me Twelve is a Significant Number. I disagreed. I said it dozen't matter.",
    "Why does Darth Vader prefer coarse-grain pepper? He hates it when it's high ground.",
    "I'm tired of seeing Frozen merchandise everywhere... Why can't everyone just let it go?",
    "What do you call a bear caught in a rain shower. A drizzly bear!",
    "Why did the deer go to the dentist? It had buck teeth.",
    "What sound does a witch's motorcycle make? BROOM, BROOOOOM!",
    "When you have a bladder infection, urine trouble.",
    "What runs around a baseball field but never moves? A fence.",
    "Roses are crimson, violets are violet I have an art degree...you want fries with that?",
    "How much money does a skunk have? Just one scent.",
    "Why did the invisible man turn down his job? He couldn't see himself doing it.",
    "Have you heard about those new corduroy pillows? They're making headlines.",
    "Why did the Tiger run away from the lion? The lion invaded the golf-course.",
    "Why do ambulances require two drivers at all times? Because they're a pair o' medics.",
    "Why couldn't the sailor learn his alphabet? He kept getting lost at C.",
    "I sat at the cafe today. No cellphone.No tablet.No laptop.I just sat there.Drinking coffee.Like a Psychopath.",
    "I tried to get a smart car the other day but they sold out too fast. Why? I guess I'm just a bit slow.",
    "What do you call a 6 feet tall circle that recently got his diploma from college? A Graduated cylinder.",
    "Why did the baker have brown fingers? Because he kneaded a poo.",
    "I can read any language in the world! If it is written in English.",
    "Where do pigs hear their favorite songs? On the Ham Radio!",
    "I wonder what turtle tastes like? It tastes like plastic.",
    "I took photo of my flower. Now it can photosynthesize.",
    "How does a man on the moon cut his hair? Eclipse it.",
    "I heard Miley Cyrus is in the new Silence of the Lambs reboot She plays Hannibal Montannibal.",
    "I'd love to you a joke about Edward Elric... but it will cost an arm and a leg.",
    "I'm glad the Chicago Cubs finally won the World Series. 108 years of hibernation just doesn't seem healthy.",
    "What's the most logical building in the USA? The US Mint.. it makes a lot of cents.",
    "When I was a kid we would get some big tires, then get inside and roll down the hill. Those were the Goodyears.",
    "I was recently the victim of a drone attack. I forgot to wear my beekeepers veil.",
    "What state do crayons go to on vacation? Color-ado.",
    "I have a joke about drilling, but it's boring.",
    "Did you hear the one about the guy with the broken hearing aid? Neither did he.",
    "I bought your book \"How to scam people on Internet\"... ...and I still haven't received it.",
    "What did the cop say to the criminal salad? Lettuce see your hands! You have the right to romaine silent.",
    "How many blood hungry vampires does it take to dress a wound? The answer's irrelevant as they all suck at it anyway.",
    "I'll call you later.' Don't call me later, call me Dad.",
    "Eve eating the apple marked.. .. the first time when Artificial Intelligence got out of its creator's control.",
    "Sundays are always a little sad, but the day before is a sadder day.",
    "People are usually shocked that I have a Police record. But I love their greatest hits!",
    "What do you call an English man at a world cup final? A referee.",
    "Why should you always knock on a refrigerator door before opening it? In case there's a salad dressing.",
    "I know this might make me sound big-headed.... I can't get my jumper off.",
    "How does a penguin build a house? Igloos it together.",
    "It is really tough being a dairy farmer. You make money by the skim of your teat!",
    "What did the dryer say to the boring duvet cover that just got out of the washer? \"Don't be such a wet blanket.\"",
    "What do you call it when a bunch of anti-maskers are kicked out of a store? A coronal mass ejection.",
    "Two peanuts went walking down the street. One was assaulted.",
    "I'd hate to be a knight They take L's left and right.",
    "What is the trigonometry teacher's favorite food? COS Law!",
    "Why did the cookie go to the doctor? It was feeling crumby.",
    "Did you hear about the guy whose whole left side was cut off? Hes all right now.",
    "Pam: \"We're hoping our interview seals the deal.\" Jim: \"If not, there's always the army...the infantry.\"",
    "If you see a robbery at an Apple Store does that make you an iWitness?",
    "How do you blow up a dinosaur? With Dinomite.",
    "When Drake gets cocky, he calls me so I can hit him with a one-liner insult to keep him humble... I'm his Hotline Zing!",
    "I was interrogated over the theft of a cheese toastie. Man, they really grilled me.",
    "Bad at golf? Join the club.",
    "If fire and water are both elements, what is steam? Better than Epic.",
    "Today's forecast is going to be.... Partially sunny......",
    "In today's Criminology class we will learn about cannibalism. It's my Hannibal Lecture.",
    "I turned over a brand new leaf today... the folks at the Nissan dealership were not very happy with me.",
    "Did you hear about the lumberjack who got a promotion? Now he's a branch manager.",
    "What did the juicer say to the orange during self-quarantine? Can't wait to squeeze you!",
    "Why are cats afraid of cucumbers? They dont like anything cooler than they are.",
    "If skeletons could be any ruler from history, who would they be? Napoleon Bone-a-Part.",
    "I have a great joke about nepotism. But I'll only tell it to my kids.",
    "What do you call a belt made out of watches? A waist of time.",
    "Why are writers really good at coding? Because they are really into Pro grammar.",
    "Why didn't the vampire attack Taylor Swift? She had bad blood.",
    "What's a judge's favourite drink ? Guil-tea.",
    "To whoever stole my copy of Microsoft Office, I will find you. You have my Word.",
    "Ever wondered why bees hum? It's because they don't know the words.",
    "What's the easiest way to burn 1000 calories? Leave the pizza in the oven.",
    "It wasn't much fun breaking my neck and being in a cast.. But now I can look back and laugh.",
    "Why did the doctor put a flesh-eating snail on the burn wound? To make the Eschar go!",
    "Why was the burglar so sensitive? He takes things personally.",
    "They locked down and instituted a curfew in the capital of Switzerland. It's a controlled Bern.",
    "They developed a toilet for the space station for two reasons: Number one, and, of course, number two.",
    "I tell dad jokes, but I don't have any kids. I'm a faux pa.",
    "What is the least spoken language in the world? Sign language.",
    "I saw a bunch of baby kittens by a dumpster... Didn't anyone tell their mother not to litter?",
    "What do you call a bull that is always felling sleepy? A bulldozer.",
    "Where did Noah keep the bees during the flood? In the Ark Hives.",
    "A single zombie is scary, but a row of zombies forms something even scarier... A deadline.",
    "What happens when you pinch a grape? It lets out a little whine.",
    "When your date shows up in a white suit that's covered in honey... You know she's gonna be a keeper.",
    "I'm done making self deprecating jokes! I'm not funny enough anyway.",
    "How do flowers whistle? Through their tulips.",
    "I like fried chick peas.... But I dont think it agrees with me. Everytime I eat them I Falafel.",
    "I can't believe I got fired from the calendar factory. All I did was take a day off.",
    "The owl asked the most introspect question ever. Who are you???",
    "I formed a rock group called the elastics, things aren't going so well so far though, We have one song and it's band.",
    "Where does Fonzie like to go for lunch? Chick-Fil-Eyyyyyyyy.",
    "Did you hear about the Frenchman that got baked into a loaf of bread? He's in a lot of pain.",
    "What do sea monsters eat for lunch? Fish and ships.",
    "My plan was to skip shoveling and just let the snow melt. It wasnt well thawed out.",
    "Waitress: Do you have any questions about the menu? Me: What font is this?",
    "Why can't you hear a psychiatrist using the bathroom? Because the 'P' is silent.",
    "Why did the math book look so sad Because of all of its problems!",
]

_DAD_FRAMES = [
    "{other} og {other2}, hør lige den her: {joke}",
    "Joke nummer {number} til dig, {other}: {joke}",
    "Den her lærte jeg i {city}, {other}: {joke}",
    "{joke} Den er til dig, {other} - lige fra {city}.",
    "{other} og {other2}, hør her: {joke}",
    "Direkte fra {city}, {other}: {joke}",
    "Nummer {number} fra jokebogen, {other}: {joke}",
    "{joke} Bedøm den, {other}: 1 til {number}.",
]
DAD_SAYS = DAD_SAYS + [
    _DAD_FRAMES[i % len(_DAD_FRAMES)].replace("{joke}", joke)
    for i, joke in enumerate(DAD_FIXED_JOKES)
]

DAD_DOES = [
    "griner længere af sin egen joke end {other}",
    "venter {feeling} på, at {other} griner",
    "tørrer en stolt tåre væk, mens {other} sukker",
    "ser sig om efter anerkendelse fra {other}",
    "skriver joken om {creature} ned i den lille notesbog {feeling}",
    "gentager pointen for {other}, bare lidt højere",
    "sukker dybt og henter mere {food} {feeling}",
    "ryster på hovedet ad {other}s joke med et lille smil",
    "klapper sig selv på skulderen foran {other}",
    "lover {other} højtideligt, at det var den sidste joke før {time}",
    "peger dobbelt med begge pegefingre mod {other} {feeling}",
    "tager en tallerken {food} som trøst",
    "kigger rundt om bordet, fra {other} til {other2}",
    "skriver et postkort fra {city} med tre jokes på",
    "laver sin egen trommelyd for {other}: ba-dum-tss",
    "forklarer joken for {other} igen, bare langsommere",
    "nikker til {other} som medskyldig",
    "noterer publikums bedømmelse: {number} ud af 10",
    "trækker vejret dybt - der er {number} jokes mere på lager",
    "giver {other} high five som den eneste, der grinede",
    "putter ekstra {food} på {other}s tallerken",
    "fortæller naboen fra {city} om sin nye joke",
    "tegner {creature} på indkøbslisten {feeling}",
    "øver sin {city}-dialekt til familiefesten",
    "gemmer den sidste {food} bag mælken {feeling}",
    "opfinder familieregel nummer {number}",
]

DAD_CONTEXT = [
    "[Kontekst: Familien er samlet til {food}, og Far har fået en ny jokebog i {city}.]",
    "[Kontekst: Det er søndag i sofaen, og {other} har gemt fjernbetjeningen.]",
    "[Kontekst: Bilturen til {city} er lang, og Far har mikrofonen, altså rattet.]",
    "[Kontekst: Der er {food} til dessert, men først skal der grines - siger Far.]",
    "[Kontekst: Grillen ryger i haven, og naboerne er kommet forbi til {food}.]",
    "[Kontekst: Det er jul hos {other}, og Far har gemt årets værste joke til gåsen.]",
    "[Kontekst: Det er morgenbord med {food}, og Far har sovet på {number} nye jokes.]",
    "[Kontekst: Familien venter på {food} i ovnen, og Far udnytter det svage øjeblik.]",
    "[Kontekst: {other} har taget en ven med hjem, og Far ser sit publikum vokse.]",
]


# ---------------------------------------------------------------------------
# SAET 5 - nyhedsstudiet. Breaking news fra fantasy-universet, fortalt helt toert.
# ---------------------------------------------------------------------------
NEWS_NAMES = [
    "Studieværten", "Reporteren", "Vejrværten", "Eksperten", "Korrespondenten",
    "Sportsværten", "Gæsten", "Øjenvidnet", "Pressechefen", "Praktikanten",
]

NEWS_SAYS = [
    "Godaften fra studiet i {city}, og velkommen til udsendelsen om {creature}.",
    "Vi afbryder programmet med seneste nyt fra {city}.",
    "{creature} er i morges set nær {place}, politiet følger sagen tæt.",
    "Trafikken snegler sig gennem {city} efter et uheld med en vogn fuld af {food}.",
    "Eksperter advarer: prisen på {food} kan stige med {number} procent inden {time}.",
    "Vores korrespondent står klar i {place}. Hvad kan du fortælle os?",
    "Jeg står her i {place}, hvor stemningen er trykket, men rolig.",
    "Vejret i morgen: tåge over {place} og enkelte byger over {city}.",
    "Sporten kort: {city} vandt igen, og {other} scorede {number} gange.",
    "Kongehuset oplyser, at {object} er fundet i god behold.",
    "Ingen kommentarer, siger borgmesteren i {city} om sagen om {object}.",
    "Øjenvidner beskriver {creature} som stor, men egentlig ret høflig.",
    "Vi får lige et billede op... ja, det er {creature} på taget af {place}.",
    "Breaking: {object} er forsvundet fra museet i {city}.",
    "Seerne spørger: er {creature} farlig? Vores ekspert svarer efter pausen.",
    "Tak, fordi du kom i studiet med så kort varsel, {other}.",
    "Det er for tidligt at konkludere, men alt peger i retning af {place}.",
    "Myndighederne beder alle i {city} om at holde sig indendørs ved {time}.",
    "Og nu til noget helt andet: en mand i {city} har bygget {object} af {food}.",
    "Vi følger sagen fra {place} minut for minut her på kanalen.",
    "Kilder tæt på slottet siger, at kongen personligt leder efter {object}.",
    "Efter pausen: stort interview med {other} om livet med {creature}.",
    "En ny måling viser, at {number} ud af 10 danskere foretrækker {food}.",
    "Vores reporter har fulgt {other} i {number} dage. Se dokumentaren i aften.",
    "Der er stadig ingen forklaring på lyset over {place}.",
    "Politiet i {city} efterlyser vidner, der har set {creature} ved {time}.",
    "Det var alt fra {place} for nu. Vi er tilbage efter vejret over {city}.",
    "En sidste nyhed: {other} har slået rekorden i at spise {food}.",
    "Rolig, helt rolig, siger beredskabet i {city}, mens {creature} nærmer sig {place}.",
    "Dagens gæst mener, at {object} er stærkt overvurderet. Debat efter pausen.",
    "Vi har netop fået bekræftet: {creature} og {creature2} er set SAMMEN nær {place}.",
    "Regeringen indkalder til pressemøde om {object} klokken {number}.",
    "Analytikere kalder situationen i {place} for usædvanlig, men stabil.",
    "Seertallene er i top, siden {creature} flyttede ind i {place}.",
    "Husk at sende jeres billeder af {creature} til redaktionen.",
    "Vinderen af aftenens konkurrence er {other} fra {city}. Stort tillykke!",
    "Landbruget melder om rekordhøst af {food} trods {creature}-plagen.",
    "Kort nyt: broen ved {place} genåbner inden {time}.",
    "Flere detaljer fra {city}, når vi ved mere. Nu til vejret over {place}.",
    "Er {object} det nye store dille? Vi har spurgt de unge i {city}.",
    "Vidnet hævder, at {creature} vinkede tilbage. Det undersøges nu.",
    "Redaktionen har modtaget {number} opkald om lyden fra {place}.",
    "Vi går nu direkte til pressemødet i {city}.",
    "Undskyld, tekniske problemer fra {place}... nå, der kom billedet af {creature}.",
    "Aftenens gæst er ekspert i {creature} og har skrevet {number} bøger om emnet.",
    "Kan du beskrive lyden, du hørte fra {place}?",
    "Ifølge vores oplysninger forlod {creature} {place} i ophidset tilstand.",
    "Vi advarer sarte seere: de næste billeder viser {creature} i {place}.",
    "Kommunen lover, at hullet i vejen ved {place} lukkes inden {time}.",
    "Årets navn i {city} er netop kåret: det blev {other}.",
    "Til alle, der lige er stået op: ja, det er rigtigt, {object} er væk.",
    "Eksperten vurderer, at {creature} blot leder efter {food}.",
    "Vores meteorolog følger skyen over {place} minut for minut.",
    "Der er lang kø ved {place}, efter rygtet om gratis {food} spredte sig.",
    "Statsministeren udtaler: situationen med {creature} er under kontrol.",
    "Nu til kulturen: {other} åbner udstilling om {object} i {city}.",
    "Vi gentager: {place} er lukket for besøgende indtil {time}.",
    "Sporten: {other} vandt løbet rundt om {place} i rekordtid.",
    "Tip os: har du set {creature}? Ring til studiet.",
    "Og med det smukke billede af {creature} i solnedgangen siger vi godnat.",
]

NEWS_DOES = [
    "blader alvorligt i papirerne om {creature}",
    "retter på sit slips før interviewet med {other}",
    "peger på vejrkortet over {place} {feeling}",
    "lytter koncentreret til nyt fra {city} i øresneglen",
    "skifter til den alvorlige stemme om {place}",
    "holder mikrofonen op mod {creature} {feeling}",
    "vender sig mod kamera 2 og mod {other}",
    "drikker en tår vand under indslaget fra {city} {feeling}",
    "vinker til seerne derhjemme i {city} {feeling}",
    "kæmper med papirerne i blæsten foran {place}",
    "interviewer et øjenvidne foran {place} {feeling}",
    "tegner en stor rød pil på skærmen mod {place}",
    "prøver at holde masken efter indslaget om {creature}",
    "får overrakt et telegram fra {city} midt i udsendelsen",
    "zoomer ind på kortet over {place} {feeling}",
    "afbryder {other} så høfligt som muligt",
    "viser seernes billeder af {creature} et efter et {feeling}",
    "retter i sidste sekund {other}s navn i rulleteksten",
    "lover seerne i {city} svar om {object} efter pausen",
]

NEWS_CONTEXT = [
    "[Kontekst: Aftenens nyhedsudsendelse er i gang, og der er breaking news fra {place}.]",
    "[Kontekst: Studiet er i direkte kontakt med korrespondenten i {city}.]",
    "[Kontekst: {creature} er set i {city}, og redaktionen er i alarmberedskab.]",
    "[Kontekst: Vejrudsigten lover storm over {place} omkring {time}.]",
    "[Kontekst: Museet i {city} melder {object} savnet, og pressen er mødt talstærkt op.]",
    "[Kontekst: Hele landet ser med, mens {other} udtaler sig om {creature} for første gang.]",
    "[Kontekst: Det er valgaften, og alle tal peger mod {city}.]",
    "[Kontekst: En sky formet som {creature} hænger over {city}, og telefonerne kimer.]",
    "[Kontekst: Studiet sender live fra {place}, hvor {other} netop er ankommet.]",
]


# ---------------------------------------------------------------------------
# SAET 6 - Shakespeare-pastiche paa gammeldags teater-dansk. Drama, dolke, drømme.
# ---------------------------------------------------------------------------
SHAKES_NAMES = [
    "Hamlet", "Ophelia", "Romeo", "Julie", "Macbeth", "Lady Macbeth",
    "Kong Lear", "Narren", "Ånden", "Puck", "Portia", "Othello",
]

SHAKES_SAYS = [
    "At være eller ikke være i {place}, det er sandelig spørgsmålet, {other}.",
    "Mit kongerige for en hest, {other}!",
    "Godnat, godnat, {other}! At skilles før {time} er en så sød sorg.",
    "Bort, forbandede plet fra {object}! Bort, siger jeg!",
    "Er dette en daggert, jeg ser for mig i {place}, med skæftet vendt mod min hånd?",
    "Verden er en scene, {other}, og selv {creature} er kun en spiller.",
    "Galskab, ja, {other} - og dog er der metode i den.",
    "Ak, stakkels Yorick! Jeg kendte ham så vel fra {place}.",
    "Thi den, der stjæler min pung, stjæler kun tant - men rør ej {object}!",
    "O ve, o skændsel, {object} er borte!",
    "Hvad lys bryder frem bag hint vindue i {place}?",
    "Tal, jeg besværger dig! Hvad så du i {place}?",
    "Der er noget råddent i {place}, det siger jeg eder.",
    "En rose ved et andet navn dufter lige så sødt som {food}.",
    "Vogt eder for {creature}, thi den smiler og smiler og er dog en skurk.",
    "I nat ved {time} mødes vi ved {place}. Sværg det!",
    "Himlens stjerner blegner mod din pande, skønne {other}.",
    "Forræderi! {other} har taget {object} og er flygtet {direction}!",
    "Min samvittighed nager mig som {number} sultne ulve.",
    "Hvil dig nu, ædle {other}, thi dagen var lang og blodig.",
    "Sværdet er draget, og jeg viger ikke for {creature}.",
    "O grumme skæbne, hvi sendte du {creature} til {place}?",
    "Et bud, et bud! Hvad nyt bringer du fra {place}?",
    "Narren taler sandt i {place}, når kongen tier, {other}.",
    "Kom, giftige nat, og skjul vor færd mod {place}.",
    "Jeg drømte om {object} i nat, og drømme lyver sjældent.",
    "Ti stille, hjerte, og bank ej så vildt for {other}.",
    "Ædle herrer, læg eders sværd, thi blod løser intet i {place} i nat.",
    "Hvad er vel {object} mod et trofast hjerte?",
    "Min dolk tørster efter {creature}, men min sjæl tøver, {other}.",
    "Sig frem, {other}, eller ti for evigt.",
    "Kys mig farvel, thi ved {time} rider jeg mod {place}.",
    "Se, hvor {creature} lurer bag forhænget som en tyv i natten!",
    "Lad trompeterne gjalde, thi {other} er vendt hjem fra {place}.",
    "Skål, ædle venner, for sejren ved {place}!",
    "Ve den dag, jeg lod {object} ude af syne!",
    "Kom, hvad komme vil, {other} - tiden løber selv gennem {time}.",
    "Elsker du mig, {other}? Så sig det, om end kun i hvisken.",
    "To slægter, begge lige fine, i skønne {place}, hvor scenen står.",
    "Noget ondt kommer mod {place} - jeg mærker det på mine tommelfingre.",
    "Sov ej mere! {other} myrder søvnen, den uskyldige søvn.",
    "Er hele verden da en løgn, og {place} dens hovedstad?",
    "Hvad er et navn? {object} ved et andet navn skinner lige klart.",
    "Tiden er af lave i {place} - o forbandede pligt, at jeg blev født til at rette den!",
    "Giv mig mit sværd, {other}, og lad natten over {place} dømme mellem os.",
    "Eders tunge er skarpere end nogen klinge i {place}.",
    "Jeg bærer en storm i brystet og kalder den {other}.",
    "Den, der ler ad ar, {other}, har aldrig mødt {creature}.",
    "Kom, nar, og syng om {place}! Thi sorgen tåler ikke stilhed.",
    "I morgen og i morgen og i morgen - dagene kryber mod {time}.",
    "Mit rige gav jeg bort for mindre end {object}.",
    "Løgnen har hurtige ben, men sandheden rider ved nat fra {place}.",
    "Frygt ikke storhed: nogle fødes store, andre kaster sig over {food}.",
    "En hest! En hest! Mit kongerige for en hest - eller blot et æsel fra {place}!",
    "Dolken ser jeg stadig - og nu peger den mod {place}.",
    "Farvel, farvel! Husk mig, når klokkerne ringer over {place}.",
    "Alt guld, der glimter i {place}, er ikke guld.",
    "Vejen til {place} er brolagt med brudte løfter.",
]

SHAKES_DOES = [
    "hæver sit sværd mod himlen over {place} {feeling}",
    "falder på knæ midt i {place} {feeling}",
    "deklamerer {feeling} mod månen",
    "svøber sig i sin kappe {feeling}",
    "stirrer længe og mørkt på {object}",
    "vandrer hvileløst gennem {place}",
    "læser brevet fra {other} med rystende hænder",
    "peger anklagende på {other}",
    "lytter i smug bag forhænget til {other}",
    "skjuler {object} i sit ærme {feeling}",
    "bukker dybt og bittert for {other}",
    "taler {feeling} til et kranium",
    "river sit hår og råber mod stormen over {place} {feeling}",
    "skåler med {other} med et hemmeligt smil",
    "øver sin monolog for et tomt {place} {feeling}",
    "kaster handsken for fødderne af {other} {feeling}",
    "gemmer giftflasken bag ryggen for {other} {feeling}",
    "kroner sig selv med en krans af efeu fra {place} {feeling}",
    "vender sig mod publikum med en hvisken om {other}",
    "falder teatralsk for {other} - {number} gange, for effektens skyld",
]

SHAKES_CONTEXT = [
    "[Kontekst: Natten er sort over {place}, og en ånd viser sig ved {time}.]",
    "[Kontekst: Hoffet er samlet i {place}, men to slægter hader hinanden.]",
    "[Kontekst: {other} har mistet {object}, og hævnen kalder.]",
    "[Kontekst: En hemmelig kærlighed blomstrer i {place} trods alle advarsler.]",
    "[Kontekst: Kronen vakler, og alle øjne hviler på {other}.]",
    "[Kontekst: Narren samler hoffet i {place} til et skuespil i skuespillet.]",
    "[Kontekst: Stormen raser over heden, og {other} vandrer alene med sin nar.]",
    "[Kontekst: Et brev er blevet forbyttet i {place}, og misforståelsen vokser ved {time}.]",
    "[Kontekst: Skuespillerne gør klar i {place}, men en af dem bærer en ægte dolk.]",
]


# ---------------------------------------------------------------------------
# SAET 7 - eventyr i H.C. Andersen-stil. Der var engang...
# ---------------------------------------------------------------------------
FAIRY_NAMES = [
    "Fortælleren", "Den lille pige", "Tinsoldaten", "Nattergalen", "Kejseren",
    "Heksen", "Den grimme ælling", "Prinsessen", "Klods-Hans", "Snedronningen",
    "Havfruen", "Nissen",
]

FAIRY_SAYS = [
    "Der var engang, for længe, længe siden, i et rige bag {place}.",
    "Det er ganske vist! Alle i {place} taler om det.",
    "Men kejseren af {place} har jo ikke noget tøj på!",
    "Jeg mærkede en ært gennem {number} madrasser. Tænk engang!",
    "Ude er godt, men hjemme i {place} er bedst.",
    "At rejse til {place} er at leve, sagde den gamle digter fra {city}.",
    "Den grimme ælling blev den smukkeste svane i hele {place}.",
    "Hør nattergalen i {place}! Den ægte synger skønnere end den af guld.",
    "Tolv hvide svaner fløj hen over {place} ved {time}.",
    "Heksen bor dybt inde i {place}, hvor træerne hvisker.",
    "Kun den, der ejer {object}, kan vække prinsessen.",
    "Klods-Hans red på sin gedebuk hele vejen til {place}.",
    "Snedronningen kyssede {other} to gange. Den tredje gang ville være hans død.",
    "Havfruen gav sin stemme bort for at danse i {place} ved {time}.",
    "Fyrtøjet kalder på {number} hunde med øjne så store som møllehjul.",
    "Og de levede lykkeligt i {place} til deres dages ende.",
    "Skyggen bukkede for {other} og sagde: nu er det mig, der er herren.",
    "Grantræet ønskede sig altid hen til {place}, det stakkels lille træ.",
    "Tommelise sov i en valnøddeskal i {place} med et rosenblad som dyne.",
    "Den standhaftige tinsoldat så aldrig væk fra danserinden i {place}.",
    "Der kom en soldat marcherende fra {place}: en, to! en, to!",
    "Kejserens nye klæder var vævet af den fineste luft fra {place}.",
    "Hyrdinden og skorstensfejeren så hele {place} fra skorstenens rand.",
    "I {place} bor en konge, der ejer {object}, siger folk.",
    "Vinden fortæller så mange historier, når den blæser fra {place}.",
    "Hver aften ved {time} fløj de vilde svaner hjem over {place}.",
    "Ællingen frøs fast i isen på søen ved {place}.",
    "Guldskatten lå på bunden af {place}, vogtet af {creature}.",
    "Rosen på graven blomstrede allersmukkest ved {time}.",
    "Stoppenålen fra {place} troede, hun var en synål, så fin følte hun sig.",
    "I den store sal i {place} sad Snedronningen på sin trone af is.",
    "Den flyvende kuffert landede midt i {place} ved {time}.",
    "Nissen flytter med, for hvor {food} er, der er hjemmet.",
    "Alting på sin rette plads, sagde de gamle i {place}.",
    "Historien om {other} og {creature} fortælles endnu i {place}.",
    "Prinsessen kunne mærke {object} gennem tyve dyner.",
    "Man er vel fra {city}, sagde han og rettede på hatten.",
    "Bare en svovlstik mere, så bliver her varmt og lyst som i {place}.",
    "Der sad en lille havfrue på stenen og så efter skibet ved {time}.",
    "Eventyret begynder først rigtigt, når man tør gå ind i {place}.",
    "Langt ude i skoven lå {place}, og derinde boede {creature}.",
    "Den mindste af de {number} brødre var den klogeste.",
    "Åh, det var kun en drøm om {place}, sagde hun - men sneen lå endnu på gulvet.",
    "Solen skinnede over {place}, og alle klokker ringede.",
    "Gamle lamper for nye, råbte manden i gården i {city}.",
    "Den tapre lille tinsoldat sejlede gennem {place} i sin papirbåd.",
    "Ingen roser uden torne, sagde gartneren i {place}.",
    "Da klokken slog tolv, blev {object} forvandlet tilbage.",
    "Historien om {other} er ikke forbi, før nattergalen har sunget den i {place}.",
    "Og siden boede de i {place} med {food} hver søndag.",
    "Gid jeg havde en skygge så flot som {other}s.",
    "Der er intet så stille som sneen over {place} ved {time}.",
    "Pas på, hvad du ønsker dig ved {object} - det plejer at gå i opfyldelse.",
    "Kun et ægte kongebarn kan mærke en ært gennem {number} dyner.",
]

FAIRY_DOES = [
    "slår den store bog op og fortæller om {creature} i {place}",
    "tænder en svovlstik i mørket ved {place} {feeling}",
    "danser på tåspidser gennem {place} som papirdanserinden",
    "pakker sin lille kuffert til rejsen mod {place} {feeling}",
    "lytter til nattergalens sang i {place} {feeling}",
    "graver fyrtøjet frem fra det hule træ nær {place} {feeling}",
    "sejler ned ad rendestenen mod {place} i en papirbåd {feeling}",
    "prøver glasskoen med bankende hjerte foran {other}",
    "strør brødkrummer på stien gennem {place} {feeling}",
    "ser sit spejlbillede i søen ved {place} og undrer sig",
    "vinker farvel til {other} fra den flyvende kuffert",
    "syr en dyne af rosenblade fra {place} til Tommelise",
    "følger de hvide svaner {direction}",
    "deler lidt {food} med den fattige soldat",
    "ånder et hul i den frosne rude og ser {creature}",
    "tæller kirkeurets slag ved {time} {feeling}",
    "varmer hænderne på en varm kartoffel fra {place} {feeling}",
    "følger lygtemanden over mosen ved {place}",
    "lægger en ært under {number} madrasser {feeling}",
    "klipper gækkebreve til {other} af det fineste papir",
]

FAIRY_CONTEXT = [
    "[Kontekst: Der var engang et rige bag {place}, hvor {other} boede i et lille hus.]",
    "[Kontekst: Fortælleren samler børnene om ilden i {place} ved {time}.]",
    "[Kontekst: En fattig soldat finder {object} i et hult træ nær {place}.]",
    "[Kontekst: Kejseren har hørt om nattergalen, der synger i {place}.]",
    "[Kontekst: Sneen daler over {place}, og Snedronningens slæde er set ved {time}.]",
    "[Kontekst: Den grimme ælling vandrer alene fra {place} mod {city}.]",
    "[Kontekst: Hele {place} holder vejret, mens prinsessen prøver glasskoen.]",
    "[Kontekst: Julesneen daler over {place}, og alle vinduer lyser undtagen et.]",
    "[Kontekst: {other} har fået tre ønsker af {creature} og har allerede fortrudt det første.]",
    "[Kontekst: Måneskinnet forvandler {place}, og legetøjet begynder at røre på sig.]",
]


# ---------------------------------------------------------------------------
# Scene-bygger - faelles for alle saet.
# ---------------------------------------------------------------------------
def _fill(template, rng, names):
    """Indsaetter tilfaeldige vaerdier i en skabelon."""
    return template.format_map(_slots(rng, names))


def make_scene(rng, says, does, contexts, names, context_prob=0.35, action_prob=0.3):
    """Bygger en enkelt lille scene med 2-4 personer og 4-13 linjer."""
    lines = []
    # Eventuel kontekst-linje i toppen.
    if rng.random() < context_prob:
        lines.append(_fill(rng.choice(contexts), rng, names))
    # Et lille fast "cast" giver lokal sammenhaeng i scenen.
    cast = rng.sample(names, min(len(names), rng.randint(2, 4)))
    n_lines = rng.randint(4, 13)
    for _ in range(n_lines):
        speaker = rng.choice(cast)
        if rng.random() < action_prob:
            line = "{}: *{}*".format(speaker, _fill(rng.choice(does), rng, names))
        else:
            line = "{}: {}".format(speaker, _fill(rng.choice(says), rng, names))
        lines.append(line)
    return "\n".join(lines) + "\n\n"


def build_dataset(target_chars, says, does, contexts, names, seed,
                  context_prob=0.35, action_prob=0.3, min_unique_lines=10_000):
    """Bygger scener indtil teksten er mindst target_chars tegn lang OG
    indeholder mindst min_unique_lines FORSKELLIGE linjer (saa modellen
    ikke bare ser de samme saetninger igen og igen)."""
    rng = random.Random(seed)
    chunks = []
    unique_lines = set()
    total = 0
    # Sikkerhedsstop, hvis skabelon-puljen skulle vaere for lille til maalet.
    max_chars = max(6 * target_chars, 6_000_000)
    while (total < target_chars or len(unique_lines) < min_unique_lines) and total < max_chars:
        scene = make_scene(rng, says, does, contexts, names,
                           context_prob=context_prob, action_prob=action_prob)
        chunks.append(scene)
        total += len(scene)
        unique_lines.update(l for l in scene.split("\n") if l.strip())
    return "".join(chunks)


def main(size_factor=1.0):
    """Skriver alle datasaet. size_factor goer ALLE filer laengere/kortere:
    fx main(3) giver ca. 3x saa lange filer - og dermed langt flere end de
    minimum 10.000 forskellige saetninger."""
    # Maal-stoerrelser (i tegn). Basis er stoerst, saa modellen kan lave ordentlige historier.
    datasets = {
        "data_base.txt": dict(
            target_chars=1_200_000, says=BASE_SAYS, does=BASE_DOES,
            contexts=BASE_CONTEXT, names=HERO_NAMES, seed=1337,
        ),
        "data_finetune.txt": dict(
            target_chars=800_000, says=JOKE_SAYS, does=JOKE_DOES,
            contexts=JOKE_CONTEXT, names=HERO_NAMES, seed=2024,
        ),
        "data_cooked.txt": dict(
            target_chars=800_000, says=COOKED_SAYS, does=COOKED_DOES,
            contexts=COOKED_CONTEXT, names=COOKED_NAMES, seed=9001,
            context_prob=0.3, action_prob=0.35,
        ),
        "data_dad_jokes.txt": dict(
            target_chars=700_000, says=DAD_SAYS, does=DAD_DOES,
            contexts=DAD_CONTEXT, names=DAD_NAMES, seed=4242,
            context_prob=0.25, action_prob=0.25,
        ),
        "data_news.txt": dict(
            target_chars=700_000, says=NEWS_SAYS, does=NEWS_DOES,
            contexts=NEWS_CONTEXT, names=NEWS_NAMES, seed=1864,
            context_prob=0.3, action_prob=0.2,
        ),
        "data_shakespeare.txt": dict(
            target_chars=700_000, says=SHAKES_SAYS, does=SHAKES_DOES,
            contexts=SHAKES_CONTEXT, names=SHAKES_NAMES, seed=1616,
            context_prob=0.3, action_prob=0.3,
        ),
        "data_fairytales.txt": dict(
            target_chars=700_000, says=FAIRY_SAYS, does=FAIRY_DOES,
            contexts=FAIRY_CONTEXT, names=FAIRY_NAMES, seed=1805,
            context_prob=0.35, action_prob=0.25,
        ),
    }

    all_scenes = []
    for filename, kwargs in datasets.items():
        kwargs = dict(kwargs)
        kwargs["target_chars"] = int(kwargs["target_chars"] * size_factor)
        text = build_dataset(**kwargs)
        _write_and_check(filename, text)
        all_scenes.extend(s for s in text.split("\n\n") if s.strip())

    # Til sidst: EN fil med ALT - alle scener fra alle syv saet, blandet godt.
    random.Random(777).shuffle(all_scenes)
    _write_and_check("data_all.txt", "\n\n".join(all_scenes) + "\n\n")

    print("Faerdig. Vokabularstoerrelse:", len(sorted(set(VOCAB_CHARS))))


def _write_and_check(filename, text):
    """Renser, tjekker og skriver en datafil + printer statistik."""
    # Sikkerhedstjek: alt skal kunne skrives med det faste vokabular.
    cleaned, removed = filter_to_vocab(text)
    assert removed == 0, (
        "{} indeholdt {} tegn udenfor VOCAB_CHARS!".format(filename, removed)
    )
    n_unique = len(set(l for l in cleaned.split("\n") if l.strip()))
    assert n_unique >= 10_000, (
        "{} har kun {} forskellige linjer!".format(filename, n_unique)
    )
    with open(filename, "w", encoding="utf-8") as f:
        f.write(cleaned)
    print("Skrev {:>22}  ({:>10,} tegn, {:>6,} forskellige linjer)".format(
        filename, len(cleaned), n_unique))


if __name__ == "__main__":
    import sys
    # Vil du have laengere filer? Giv en faktor med:  python generate_datasets.py 3
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 1.0)
