"""Hjælpefunktioner til Intro-ML — KUN plotte-hjælp, ingen magi.

I behøver ikke læse koden her for at følge med i emnet. Men I er meget
velkomne til at kigge: det er helt almindelig matplotlib, som I selv har
brugt — bare med lidt ekstra pynt (meshgrid, pile og subplots-gymnastik).
Alt det, der er selve læringsmålet (data, modeller og træningsloops),
står synligt i notebooks'ene.
"""

from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from IPython.display import HTML
import matplotlib as mpl
import io
import matplotlib.pyplot as plt
import numpy as np
import torch
import re


def _til_numpy(x):
    """Konverterer en tensor til numpy — og lader numpy-arrays være i fred."""
    if torch.is_tensor(x):
        return x.detach().numpy()
    return np.asarray(x)


def plot_gd_skridt(f, punkter, titel="Gradient descent"):
    """Tegner funktionen f og de skridt, gradient descent har taget.

    f:       en funktion der tager et tal (eller numpy-array) og returnerer f(x)
    punkter: liste af x-værdier, ét tal pr. skridt (samlet op i jeres loop)
    """
    punkter = np.array([float(p) for p in punkter])
    bredde = max(punkter.max() - punkter.min(), 1.0)
    x = np.linspace(punkter.min() - 0.5 * bredde, punkter.max() + 0.5 * bredde, 300)

    plt.figure(figsize=(8, 5))
    plt.plot(x, f(x), color="steelblue", linewidth=2, label="f(x)")
    plt.scatter(punkter, f(punkter), color="crimson", zorder=3, label="skridt")
    plt.scatter(punkter[0], f(punkter[0]), color="orange", s=120, zorder=4,
                label="start", edgecolors="black")
    for i in range(len(punkter) - 1):
        plt.annotate("",
                     xy=(punkter[i + 1], f(punkter[i + 1])),
                     xytext=(punkter[i], f(punkter[i])),
                     arrowprops=dict(arrowstyle="->", color="crimson", alpha=0.6))
    plt.title(titel)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.show()


def plot_beslutningsgraense(model, X, y, titel=""):
    """Farver planen efter modellens forudsigelser og tegner datapunkterne ovenpå.

    model: en (trænet) PyTorch-model der tager præcis 2 features som input
    X:     (N, 2) tensor eller numpy-array med de to features
    y:     (N,) labels — 0/1 eller klassenumre
    """
    X = _til_numpy(X).astype(np.float32)
    y = _til_numpy(y).ravel()

    marg = 0.5
    x1 = np.linspace(X[:, 0].min() - marg, X[:, 0].max() + marg, 200)
    x2 = np.linspace(X[:, 1].min() - marg, X[:, 1].max() + marg, 200)
    g1, g2 = np.meshgrid(x1, x2)
    gitter = torch.tensor(np.column_stack([g1.ravel(), g2.ravel()]),
                          dtype=torch.float32)

    with torch.no_grad():
        ud = model(gitter).reshape(len(gitter), -1)

    plt.figure(figsize=(7, 6))
    if ud.shape[1] == 1:
        # Binær klassifikation: farv efter sandsynligheden for klasse 1
        z = ud.numpy().reshape(g1.shape)
        plt.contourf(g1, g2, z, levels=20, cmap="RdBu_r", alpha=0.6,
                     vmin=0.0, vmax=1.0)
        plt.colorbar(label="modellens sandsynlighed for klasse 1")
        plt.contour(g1, g2, z, levels=[0.5], colors="black", linewidths=2)
    else:
        # Flere klasser: farv efter den klasse, modellen tror mest på
        z = ud.argmax(dim=1).numpy().reshape(g1.shape)
        antal_klasser = ud.shape[1]
        plt.contourf(g1, g2, z, levels=np.arange(antal_klasser + 1) - 0.5,
                     cmap="Pastel1", alpha=0.9)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r",
                edgecolors="black", s=25, zorder=3)
    plt.title(titel)
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.show()


def vis_mnist_billeder(billeder, labels, forudsigelser=None, n=10):
    """Viser n MNIST-billeder i ét grid.

    billeder:       (N, 784) eller (N, 28, 28) tensor/array med pixelværdier
    labels:         (N,) de rigtige cifre
    forudsigelser:  (N,) modellens gæt (valgfri) — titlen bliver RØD ved fejlgæt
    """
    billeder = _til_numpy(billeder)
    labels = _til_numpy(labels).ravel()
    if billeder.ndim == 2:
        billeder = billeder.reshape(-1, 28, 28)
    if forudsigelser is not None:
        forudsigelser = _til_numpy(forudsigelser).ravel()

    n = min(n, len(billeder))
    kolonner = min(n, 5)
    raekker = int(np.ceil(n / kolonner))
    fig, akser = plt.subplots(raekker, kolonner,
                              figsize=(2 * kolonner, 2.4 * raekker))
    for i, akse in enumerate(np.atleast_1d(akser).ravel()):
        akse.axis("off")
        if i >= n:
            continue
        akse.imshow(billeder[i], cmap="gray")
        if forudsigelser is None:
            akse.set_title(f"label: {int(labels[i])}")
        else:
            gaet = int(forudsigelser[i])
            rigtigt = gaet == int(labels[i])
            akse.set_title(f"gæt: {gaet} (label: {int(labels[i])})",
                           color="black" if rigtigt else "red")
    plt.tight_layout()
    plt.show()


def _som_2d_billeder(billeder):
    """Accepterer (N, H*W), (N, H, W) eller (N, 1, H, W) og giver (N, H, W)."""
    billeder = _til_numpy(billeder)
    if billeder.ndim == 4:                       # (N, 1, H, W)
        billeder = billeder[:, 0]
    if billeder.ndim == 2:                       # (N, H*W) — antag kvadratisk
        side = int(np.sqrt(billeder.shape[1]))
        billeder = billeder.reshape(-1, side, side)
    return billeder


def vis_billeder(billeder, labels=None, forudsigelser=None, n=10, klassenavne=None):
    """Viser n billeder i ét grid.

    billeder:       (N, H*W), (N, H, W) eller (N, 1, H, W) — tensor eller array
    labels:         (N,) de rigtige klasser (valgfri)
    forudsigelser:  (N,) modellens gæt (valgfri) — titlen bliver RØD ved fejlgæt
    klassenavne:    liste der oversætter klassenumre til navne (fx dansk tøj)
    """
    billeder = _som_2d_billeder(billeder)
    if labels is not None:
        labels = _til_numpy(labels).ravel()
    if forudsigelser is not None:
        forudsigelser = _til_numpy(forudsigelser).ravel()

    def navn(k):
        k = int(k)
        return klassenavne[k] if klassenavne is not None else str(k)

    n = min(n, len(billeder))
    kolonner = min(n, 5)
    raekker = int(np.ceil(n / kolonner))
    fig, akser = plt.subplots(raekker, kolonner,
                              figsize=(2.2 * kolonner, 2.6 * raekker))
    for i, akse in enumerate(np.atleast_1d(akser).ravel()):
        akse.axis("off")
        if i >= n:
            continue
        akse.imshow(billeder[i], cmap="gray")
        if forudsigelser is not None and labels is not None:
            rigtigt = int(forudsigelser[i]) == int(labels[i])
            akse.set_title(f"gæt: {navn(forudsigelser[i])}\n(label: {navn(labels[i])})",
                           color="black" if rigtigt else "red", fontsize=9)
        elif labels is not None:
            akse.set_title(navn(labels[i]), fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_forvirringsmatrix(y_sand, y_gaet, klassenavne=None, titel="Forvirringsmatrix"):
    """Tegner en forvirringsmatrix: rækker = det rigtige svar, kolonner = modellens gæt.

    Diagonalen er de korrekte gæt — alt udenfor er forvekslinger.
    """
    y_sand = _til_numpy(y_sand).ravel().astype(int)
    y_gaet = _til_numpy(y_gaet).ravel().astype(int)
    antal_klasser = int(max(y_sand.max(), y_gaet.max())) + 1

    matrix = np.zeros((antal_klasser, antal_klasser), dtype=int)
    for sand, gaet in zip(y_sand, y_gaet):
        matrix[sand, gaet] += 1

    plt.figure(figsize=(7.5, 6.5))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar(label="antal")
    navne = klassenavne if klassenavne is not None else [str(i) for i in range(antal_klasser)]
    plt.xticks(range(antal_klasser), navne, rotation=45, ha="right")
    plt.yticks(range(antal_klasser), navne)
    plt.xlabel("modellens gæt")
    plt.ylabel("rigtigt svar")
    plt.title(titel)
    # skriv tallene i cellerne (hvid tekst på mørk baggrund, sort på lys)
    graense = matrix.max() / 2 if matrix.max() > 0 else 0.5
    for i in range(antal_klasser):
        for j in range(antal_klasser):
            plt.text(j, i, matrix[i, j], ha="center", va="center", fontsize=8,
                     color="white" if matrix[i, j] > graense else "black")
    plt.tight_layout()
    plt.show()


def plot_kmeans_skridt(X, centre_historik, tildelinger_historik):
    """Tegner k-means-algoritmens iterationer som et grid af scatterplots.

    X:                    (N, 2) datapunkter
    centre_historik:      liste af (k, 2)-arrays — centrene efter hver iteration
    tildelinger_historik: liste af (N,)-arrays — hvert punkts klynge-nummer pr. iteration
    """
    X = _til_numpy(X)
    antal = len(centre_historik)
    kolonner = min(antal, 3)
    raekker = int(np.ceil(antal / kolonner))
    fig, akser = plt.subplots(raekker, kolonner,
                              figsize=(4.5 * kolonner, 4 * raekker))
    for i, akse in enumerate(np.atleast_1d(akser).ravel()):
        if i >= antal:
            akse.axis("off")
            continue
        centre = _til_numpy(centre_historik[i])
        tildeling = _til_numpy(tildelinger_historik[i])
        akse.scatter(X[:, 0], X[:, 1], c=tildeling, cmap="tab10", s=15, alpha=0.7)
        akse.scatter(centre[:, 0], centre[:, 1], marker="X", s=250,
                     color="black", edgecolors="white", linewidths=1.5, zorder=3)
        akse.set_title(f"iteration {i}")
    plt.tight_layout()
    plt.show()


def vis_rekonstruktioner(originaler, rekonstruktioner, n=8, titel=""):
    """Viser originale billeder (øverste række) og rekonstruktioner (nederste)."""
    originaler = _som_2d_billeder(originaler)
    rekonstruktioner = _som_2d_billeder(rekonstruktioner)
    n = min(n, len(originaler))

    fig, akser = plt.subplots(2, n, figsize=(1.6 * n, 3.6))
    akser = np.array(akser).reshape(2, n)     # også når n = 1
    for i in range(n):
        akser[0, i].imshow(originaler[i], cmap="gray")
        akser[1, i].imshow(rekonstruktioner[i], cmap="gray")
        akser[0, i].axis("off")
        akser[1, i].axis("off")
    akser[0, 0].set_title("original", loc="left", fontsize=10)
    akser[1, 0].set_title("rekonstruktion", loc="left", fontsize=10)
    if titel:
        fig.suptitle(titel)
    plt.tight_layout()
    plt.show()



def read_file_into_tokens(file_path):
    """
    Læser en given fil, og omdanner det til en rå liste af tokens (altså individuelle ord)
    """

    # Her læser vi filens indhold
    file_contents = ""
    file = open(file_path, "r", encoding="utf-8")
    for line in file:
        file_contents += line
    file.close()
    
    # Vi laver alt indhold lower-case
    file_contents = file_contents.lower()
    
    # Her bruger vi et værktøj der hedder regex.
    # Det er bare en smart måde at vi hurtigt kan få omdannet vores fil-indhold til en liste af strings ved at opdele den ved specifikke tegn
    raw_tokens = re.split(r"(?![:,\.;\"][0-9])[:,\.;?\" \r\n]+", file_contents)
    
    # Behold kun de tokens som har en værdi (altså ingen tokens som er tom tekst)
    tokens = []
    for raw_token in raw_tokens:
        if raw_token:
            tokens.append(raw_token)
    return tokens

def create_states(data, amount_of_words=1):
    """
    Skaber individuelle states til vores data ud fra mængden af ord der betragtes på en gang  
    """
    list_of_states = []
    
    # Vi gå gennem vores data ud fra index idet vi skal kombinere dele af listen af gangen
    for i in range(len(data) + 1 - amount_of_words):
        # Kombiner fremliggende naboer op til mængden af betragtede ord (eg. amount_of_words = 2 medfører at hvert token bliver 2 ord lang)
        list_of_states.append(" ".join(data[i:i + amount_of_words]))

    return list_of_states


def generate_markov_chain(states):
    markov_chain={}
    """
    Genererer en dictionary som angiver mængden af forekomster og naboliggende ord (og deres hyppigheder) i vores datasæt som en 
    """
    for index, word in enumerate(states):
        # Positionen til det næste ord i vores liste
        nabo_index = index + 1

        # Hvis det næste ord i sekvensen ligger udenfor listen, er vi færdige.
        if nabo_index >= len(states):
            # Hvis det allersidste ord ikke findes andre steder end i slutningen, kan det som state i markov kæden ikke føre til andre states.
            # I dette tilfælde, siger vi derfor blot at dens næste ord er sig selv, således at vi ikke har en slags 'blind vej' i vores endelige markov kæde.
            if word not in markov_chain:
                markov_chain[word] = [1,{}]
                #markov_chain[word] = [1,{}]
                markov_chain[word][1][word] = 1
                markov_chain[word][1][next_word] = 1
            break
            
        next_word = states[nabo_index].split(" ")[-1]
        
        # Hvis vi har registreret vores ord i dictionary i forvejen, så tilføj 1 til hyppigheden og forekomsterne
        if word in markov_chain:
            # Ved 0 indekset ligger totale mængde forekomster i teksten
            markov_chain[word][0] += 1
            
            # Ved 1 indekset ligger en dictionary som viser hyppigheden for hvert naboliggende ord.
            # Opdater hyppigheden for det naboliggende ord 
            if next_word in markov_chain[word][1]:
                markov_chain[word][1][next_word] += 1
            else:
                markov_chain[word][1][next_word] = 1

        # Hvis ordet ikke er registreret endnu, registrerer vi den til vores data dictionary
        else:
            markov_chain[word] = [1,{}]
            markov_chain[word][1][next_word] = 1
    
    # Ved alle states sorterer vi de næste states ud fra deres hyppighed. 
    # Dette gør vi for at gøre det nemmere at beregne sandsynlighed
    for state in markov_chain.keys():
        markov_chain[state][1] = dict(sorted(markov_chain[state][1].items(), key = lambda item: item[1], reverse=True))

    return markov_chain


def calculate_state_chances(temperature, total_occurences, possible_next_states):
    """
    Beregner chancerne for at gå til hvert mulige state
    """
    return [calculate_single_state_chance(temperature, int(total_occurences), len(possible_next_states), state) for state in possible_next_states.values()]


def calculate_single_state_chance(temperature, total_occurances, possible_next_state_count, next_state_frequency):
    """
    Beregner en chance for at gå til hvert mulige state
    """
    
    if temperature<=1:
        # Denne beregning gør at temperatur på 0 medfører en determinisk generation, samt at ved 1 så generering chancen direkte ækvivalent med frekvensen / totale hyppighed.
        # Imellem disse 2 ender, er en glat overgang.
        return  next_state_frequency / ((1-temperature) * next_state_frequency + total_occurances * temperature)
    
    else:
        
        # Hvis temperature = 1, bliver den flippet til 0. Derefter beregnes
        """
        Først temperature = temperature-1 så temperature = 0
        I slutning skal det være next_state_frequency/totalt_occurences
        så next_state_frequency*(1-temperature)+temperature = next_state_frequency*(1-0)+0 = next_state_frequency
        så temperature = total_occurrences

        q= dif_amount*r+total_amount*(1-r) = dif_amount*0+total_amount*(1-0) =total_amount
        q = total_amount

        t/q =n/total_amount

        """
        
        
        # Hvis r=2 så bliver den 1 og det her er, hvad der skal ske med den.
        """
        Først r = r-1 så r = 1
        I slutning skal det være 1/dif_amount
        så n*(1-r)+r =n*(1-1)+1 = n*0+1 = 1
        så t = 1

        q= dif_amount*r+total_amount*(1-r) = dif_amount*1+total_amount*(1-1) =dif_amount
        q = dif_amount
        t/q= 1/dif_amount 
        """
        
        temperature = temperature-1
        t = next_state_frequency*(1-temperature)+temperature

        q = possible_next_state_count*temperature+total_occurances*(1-temperature) 
        return t/q
     

def generate_text(markov_chain, amount_to_generate = 10, starting_state = None, temperature = 1):
    """
    Genererer tekst ud fra den givne markov kæde. 
    Bemærk at hvis starting_state er None, bliver en tilfældig starting state valgt fra markov kæden.
    """
    
    # Hvis vi ikke har givet et start state, så vælg et tilfældigt
    if starting_state is None:
        starting_state = np.random.choice(markov_chain.keys())
    
    # Gør vores start til lower-case
    starting_state = starting_state.lower()

    # Tjek vores input
    if starting_state not in markov_chain.keys():
        raise ValueError("Start staten findes ikke som et state i markov kæden")
    if (amount_to_generate <= 0):
        raise ValueError("Du kan ikke generere nul eller en negativ mængde tekst")
    if (temperature < 0 or temperature > 2):
        raise ValueError("Temperatur skal være mellem 0 og 2")
    
    # Vi bestemmer mængden af ord betragtet på en gang ud fra det givne start ordet. 
    # Dette bør altid være den samme mængde som mængden af ord betragtet i selve markov kæden.
    amount_of_words = len(starting_state.split(" "))
    
    # Denne liste vil indeholde vores genererede tekst. Den starter derfor med at indeholde vores startord
    generated_text = starting_state.split(" ")
    
    # Vi bruger denne til at holde styr på hvad vi gerne vil generere det næste ud fra
    prev_state = starting_state
    
    # Generer tekst
    for i in range(amount_to_generate):
        total_occurances = markov_chain[prev_state][0]
        next_possible_states = markov_chain[prev_state][1]
        next_possible_state_chances = calculate_state_chances(temperature, total_occurances, next_possible_states)
        
        # Vi opsummere løbende chancerne for hvert state. Når den overgår vores tilfældigt-satte threshold, er det den state vi går til næst.
        # Dette er den simpleste måde at programmatisk vælge et element fra en liste tilfældigt ud fra en chance.
        random_threshold = np.random.rand(1,1)[0][0]
        sum_chance = 0

        # Variablen er, hvor meget chance der er tilbage før der har noget på, som bliver anvendt senere til en form for normalisering
        amount_left = 1

        # Variablen er hvad der skal ganges med, så det er normaliset på en måde for, at sikre at det altid bliver summed up til 1 i slutning
        multiply_to_norm = 1
        
        # Vi gennemgår hvert næste mulige state med et indeks, idet vi bruger seperate lister til chancerne og selve states.
        for next_state_index, next_state_text in enumerate(next_possible_states.keys()):
            
            # Tjek om vores tilfældige threshold er nået- hvis ja, er det nuværende element det tilfældigt valgte. Ellers, fortsæt gennem vores muligheder.
            sum_chance += next_possible_state_chances[next_state_index]*multiply_to_norm
            if sum_chance <= random_threshold:
                # Opdater mængden der er left baseret på sum_chancen
                amount_left = 1 - sum_chance 
                # dette bliver til hvad man skal gange alle værdierne med, så deres sum giver mængden der er tilbage for at sikre at der altid vil 1.
                multiply_to_norm = 1/(sum(next_possible_state_chances[next_state_index:])*(1/(amount_left)))
                continue

            # Vi har hermed vores næste state.
            generated_text.append(next_state_text)
            
            # Opdater det nuværende state til at være de sidste ord i vores genererede tekst
            prev_state = " ".join(generated_text[-amount_of_words:])
            break

    # Vores genererede tekst består af individuelle ord. Vi samler dem for at få et sammenhængende stykke tekst
    return " ".join(generated_text)



mpl.rcParams['animation.embed_limit'] = 100


def gradient_linje(a, b, punkter):
    """Gradienten af squared loss for linjen y = a·x + b, mht. a og b. Bruges til opg5's gradient descent."""
    grad_a, grad_b = 0.0, 0.0
    for x, y in punkter:
        err    = 2 * (a * x + b - y)
        grad_a += err * x
        grad_b += err
    return grad_a, grad_b


def closed_form_linje(punkter):
    """
    Analytisk mindste-kvadraters løsning for y = a·x + b — "de normale ligninger",
    w = (XᵀX)⁻¹Xᵀy, samme lineære algebra som opgave 1_4. Bruges til at sammenligne
    med den linje jeres egen gradient descent finder.
    """
    xs = np.array([p[0] for p in punkter], dtype=float)
    ys = np.array([p[1] for p in punkter], dtype=float)
    A = np.column_stack([xs, np.ones_like(xs)])
    w = np.linalg.inv(A.T @ A) @ A.T @ ys
    return float(w[0]), float(w[1])


# ── Delte hjælpefunktioner til loss-flade og fejl-kvadrater ────────────────

def _kurve_label(params, prefix=""):
    """Formatterer 'y = a·x + b' eller 'y = a·x² + b·x + c' afhængig af antal parametre."""
    try:
        if len(params) == 2:
            a, b = params
            sign = '+' if b >= 0 else '-'
            return f"{prefix}: y = {a:.3f}·x {sign} {abs(b):.3f}"
        a, b, c = params
        return f"{prefix}: y = {a:.3f}·x² + {b:.3f}·x + {c:.3f}"
    except Exception:
        return f"{prefix}: {params}"


def _eval_kurve(params, x):
    """Evaluerer linjen (2 params) eller parablen (3 params) i x — x må gerne være et numpy-array."""
    if len(params) == 2:
        a, b = params
        return a * x + b
    a, b, c = params
    return a * x**2 + b * x + c


def _tegn_matrix_bracket(ax, x, y_top, y_bot, retning=1, tick=0.15, lw=1.6):
    """Tegner en kantet-parentes-side ('[' hvis retning=1, ']' hvis retning=-1) — ren matplotlib,
    ingen afhængighed af LaTeX/usetex."""
    ax.plot([x, x], [y_top, y_bot], color='black', linewidth=lw, solid_capstyle='butt')
    ax.plot([x, x + retning * tick], [y_top, y_top], color='black', linewidth=lw, solid_capstyle='butt')
    ax.plot([x, x + retning * tick], [y_bot, y_bot], color='black', linewidth=lw, solid_capstyle='butt')


def _matrix_hojde(M, cell_h=0.55):
    """Hvor høj matricen M bliver når den tegnes med _tegn_matrix (bruges til at placere rækker)."""
    M = np.array(M, dtype=object)
    rows = M.shape[0] if M.ndim > 1 else len(M)
    return rows * cell_h


def _tegn_matrix(ax, x, y, label, M, fmt="{:.2f}", cell_w=1.05, cell_h=0.55, fontsize=11):
    """
    Tegner matricen M (1D vises som søjlevektor) centreret om (x, y) som venstre kant.
    fmt=None betyder symbolske entries (fx bogstaver som 'a'/'b') — de vises som de er,
    uden talformattering. Returnerer x-positionen lige efter matricen, så flere matricer
    og symboler kan sættes op i forlængelse af hinanden — vandret i samme kald, lodret
    via y (se linalg_losning).
    """
    M = np.array(M, dtype=object) if fmt is None else np.array(M, dtype=float)
    if M.ndim == 1:
        M = M.reshape(-1, 1)
    rows, cols = M.shape
    height = rows * cell_h
    width  = cols * cell_w
    pad    = cell_w * 0.35
    y_top, y_bot = y + height / 2, y - height / 2

    for i in range(rows):
        for j in range(cols):
            cx = x + pad + j * cell_w + cell_w / 2
            cy = y_top - i * cell_h - cell_h / 2
            tekst = str(M[i, j]) if fmt is None else fmt.format(M[i, j])
            ax.text(cx, cy, tekst, ha='center', va='center', fontsize=fontsize)

    x_right = x + width + 2 * pad
    _tegn_matrix_bracket(ax, x,       y_top, y_bot, retning=1)
    _tegn_matrix_bracket(ax, x_right, y_top, y_bot, retning=-1)
    ax.text((x + x_right) / 2, y_top + 0.35, label, ha='center', fontsize=fontsize + 1)
    return x_right


def _tegn_symbol(ax, x, y, symbol, bredde=0.7, fontsize=18):
    """Tegner et regneoperator-symbol ('·', '=', ...) mellem to matricer, se linalg_losning."""
    ax.text(x + bredde / 2, y, symbol, ha='center', va='center', fontsize=fontsize)
    return x + bredde


def _loss_grid(loss_fn, a_range, b_range, resolution=60):
    """Bygger (A, B, L) meshgrid ved at evaluere loss_fn(a, b) over gridet."""
    a_vals = np.linspace(a_range[0], a_range[1], resolution)
    b_vals = np.linspace(b_range[0], b_range[1], resolution)
    A, B = np.meshgrid(a_vals, b_vals)
    L = np.empty_like(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            L[i, j] = loss_fn(A[i, j], B[i, j])
    return A, B, L


_STANDARD_ØVRE_PERCENTIL = 90  # se _tegn_kontur/loss_range — SAMME klippe-regel bruges overalt,
                                # både når et panel autoskalerer til sine egne (lokale) data og
                                # når flere paneler eksplicit deler samme farve_range, netop
                                # for at farverne skal være sammenlignelige på tværs af BÅDE
                                # paneler OG celler i det hele taget.


def _tegn_kontur(ax, A, B, L, colorbar=True, titel='Loss (kontur)', farve_range=None):
    if farve_range is not None:
        # DELT farveskala på tværs af flere paneler (fx et zoomet og et fuldt udzoomet) — så
        # samme farve altid betyder samme loss-værdi, uanset hvilket panel man kigger på.
        lo, hi = farve_range
    else:
        # STANDARD (intet farve_range givet): autoskalér til dette panels EGNE data, men klip
        # stadig de øverste ekstremværdier (samme percentil som ovenfor) i stedet for det
        # bogstavelige maksimum — ellers ville fx rosenbrocks få ekstreme hjørner sluge det
        # meste af farveskalaen og efterlade næsten ingen kontrast omkring de lave (interessante)
        # loss-værdier, i ETHVERT panel der viser den slags landskab, ikke kun dem der eksplicit
        # beder om det.
        lo, hi = float(np.min(L)), float(np.percentile(L, _STANDARD_ØVRE_PERCENTIL))
    cs = ax.contourf(A, B, L, levels=np.linspace(lo, max(hi, lo + 1e-12), 31), cmap='viridis', extend='max')
    if colorbar:
        # Bemærk: hvert kald skrumper ax'et lidt for at gøre plads til farveskalaen,
        # så ved gentagne redraws af samme ax (fx i animate()) bør colorbar=False bruges.
        plt.colorbar(cs, ax=ax, shrink=0.8)
    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_title(titel)
    return cs


def loss_range(loss_fn, a_range, b_range, resolution=60, øvre_percentil=_STANDARD_ØVRE_PERCENTIL):
    """Regner (min, max) af loss_fn hen over et grid — brug den til at give flere paneler
    samme farve_range (se loss_kontur), så farverne kan sammenlignes på tværs af paneler,
    fx et zoomet og et fuldt udzoomet panel af samme landskab.
    øvre_percentil: default 90 (SAMME percentil som loss_kontur selv klipper til, når intet
    farve_range er givet — se _STANDARD_ØVRE_PERCENTIL) — brug denne percentil af
    loss-værdierne som 'max' i stedet for den bogstavelige maksimumsværdi. Nyttigt for
    landskaber som rosenbrock, hvor et par ekstreme hjørner ellers ville sluge det meste af
    farveskalaen og efterlade næsten ingen kontrast omkring de lave (interessante)
    loss-værdier. Sæt til 100 for den bogstavelige maksimumsværdi (ingen klipning). Værdier
    over percentilen klippes til samme farve som percentilen selv (loss_kontur bruger
    extend='max')."""
    _, _, L = _loss_grid(loss_fn, a_range, b_range, resolution)
    hi = float(np.percentile(L, øvre_percentil)) if øvre_percentil is not None else float(np.max(L))
    return float(np.min(L)), hi


def _tegn_3d(ax, A, B, L):
    # matplotlib 3D-akser regner selv ud i hvilken rækkefølge artists tegnes (baseret på
    # kamera-afstand), så et scatter-punkt kan ende "begravet" i fladen selvom det har
    # højere zorder. computed_zorder=False slår det fra, så vi selv styrer rækkefølgen.
    ax.computed_zorder = False
    ax.plot_surface(A, B, L, cmap='viridis', linewidth=0, antialiased=True, alpha=0.9, zorder=1)
    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_zlabel('loss')
    ax.set_title('Loss (3D-flade)')


def _tegn_kvadrater_panel(ax, xs, ys, params, x_min=None, x_max=None, y_min=None, y_max=None, label_prefix="model"):
    """
    Tegner datapunkter, modellen (linje hvis params=(a,b), parabel hvis params=(a,b,c)),
    og et kvadrat per fejl. Kvadratets areal = fejlens bidrag til squared loss.
    Returnerer fejlene (y_pred - y) så den kaldende kode kan udregne loss.
    Angiv x_min/x_max/y_min/y_max for at låse aksernes skala på tværs af flere kald
    (ellers deformeres kvadraterne visuelt når fejlens størrelse ændrer sig).
    """
    y_preds = _eval_kurve(params, xs)
    errors  = y_preds - ys

    max_e = float(np.max(np.abs(errors))) if len(errors) else 1.0
    if x_min is None:
        x_min = xs.min() - 0.5
    if x_max is None:
        x_max = xs.max() + max_e + 0.5
    x_line = np.linspace(x_min, x_max, 300)

    ax.plot(x_line, _eval_kurve(params, x_line), color='royalblue', linewidth=2,
            label=_kurve_label(params, prefix=label_prefix), zorder=2)

    for x, y, y_pred, e in zip(xs, ys, y_preds, errors):
        side  = abs(float(e))
        y_bot = min(float(y), float(y_pred))
        if side > 1e-10:
            ax.add_patch(Rectangle(
                (x, y_bot), side, side,
                linewidth=1, edgecolor='tomato', facecolor='tomato', alpha=0.25,
                zorder=3,
            ))
            ax.text(x + side / 2, y_bot + side / 2, f'{e**2:.2f}',
                    ha='center', va='center', fontsize=8, color='darkred', zorder=5, clip_on=True)
        ax.plot([x, x], [y, y_pred], color='tomato', linewidth=1.5, zorder=4)

    ax.scatter(xs, ys, color='black', s=60, zorder=6, label='punkter')
    ax.set_xlim(x_min, x_max)
    if y_min is None or y_max is None:
        y_all = np.concatenate([ys, y_preds])
        pad   = 0.5
        y_min = y_all.min() - pad if y_min is None else y_min
        y_max = y_all.max() + pad if y_max is None else y_max
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    return errors


# ── Komponerbare paneler: virker alene, eller samlet via display() ────────
# (samme idé som Maples plot(...) / display(p1, p2) — et panel viser sig selv
# hvis det står alene i en celle, eller kan lægges sammen med andre paneler.)

class _Panel:
    def __init__(self, tegn_fn, figsize=(6, 5), projection=None):
        self._tegn      = tegn_fn
        self.figsize    = figsize
        self.projection = projection

    def _repr_png_(self):
        fig = plt.figure(figsize=self.figsize)
        self._tegn(fig.add_subplot(1, 1, 1, projection=self.projection))
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        return buf.getvalue()


def display(*paneler, figsize=None):
    """Vis flere paneler side om side, fx: sesy_viz.display(sesy_viz.loss_kontur(...), sesy_viz.loss_3d(...))"""
    n = len(paneler)
    fig = plt.figure(figsize=figsize or (6 * n, 5))
    for i, panel in enumerate(paneler):
        panel._tegn(fig.add_subplot(1, n, i + 1, projection=panel.projection))
    plt.tight_layout()
    plt.show()


def display_grid(paneler, cols, row_labels=None, figsize=None):
    """
    Som display(), men i et helt grid i stedet for kun én række — paneler er en FLAD liste,
    der ombrydes til en ny række for hver `cols` paneler. Bruges fx til
    træningsloop-konkurrencens 24 paneler (4 landskaber × 6 startpunkter, cols=6).
    row_labels: valgfri liste af navne, ét pr. række, vist som y-akse-label på rækkens første panel.
    """
    rows = -(-len(paneler) // cols)  # rundet op
    fig = plt.figure(figsize=figsize or (3 * cols, 2.8 * rows))
    for i, panel in enumerate(paneler):
        raekke, soejle = divmod(i, cols)
        ax = fig.add_subplot(rows, cols, i + 1, projection=panel.projection)
        panel._tegn(ax)
        if soejle == 0 and row_labels:
            ax.set_ylabel(f"{row_labels[raekke]}\n\n{ax.get_ylabel()}", fontsize=9)
    plt.tight_layout()
    plt.show()


def bar_sammenligning(navne, kørsler, farver, titel=None):
    """
    Sammenlign flere metoder på AFSTAND TIL BEDSTE MULIGE loss (0 = perfekt, altid ≥ 0 — så
    'lavest er bedst' altid betyder 'tættest på bunden', uanset om landskabets egen loss-skala
    er negativ, positiv, eller begge dele).
    navne/kørsler/farver: samme rækkefølge. kørsler[i] er EN LISTE af individuelle afstande for
    metode navne[i] — typisk én pr. udvalgt startpunkt, samme rækkefølge som start-panelerne
    ved siden af dette panel. En kørsel på None betyder at den REELT eksploderede (landede uden
    for landskabet, eller gav et ikke-endeligt loss) — det er den ENESTE ting der gør en søjle
    skraveret/"tæller ikke med". Et endeligt, men bare stort tal (fx et gyldigt punkt på
    rosenbrock, der bare stadig har høj loss der) er IKKE det samme som eksploderet, og skal
    ikke se ud som det — sådan en søjle tegnes i stedet afskåret ved panelets loft, markeret med
    en lille trekant, med sin egen farve intakt (en "der er mere, men det er stadig et rigtigt
    resultat"-søjle, ikke en "her skete der ingenting"-søjle).
    Hver metode vises som en lille klynge af søjler: én (fuld farve, sort kant) for medianen af
    kørslerne, derefter én smal, halvgennemsigtig søjle pr. individuel kørsel.
    """
    def tegn(ax):
        medianer = []
        for løb in kørsler:
            synlige = [v for v in løb if v is not None]
            medianer.append(float(np.median(synlige)) if synlige else None)

        alle_synlige = [v for løb in kørsler for v in løb if v is not None] + \
                       [m for m in medianer if m is not None]
        loft = max((np.percentile(alle_synlige, 90) if alle_synlige else 1.0) * 1.35, 1e-9)

        n_søjler = 5  # median + 4 individuelle kørsler
        bredde = 0.8 / n_søjler
        offsets = [-0.4 + bredde * (i + 0.5) for i in range(n_søjler)]

        def tegn_søjle(x, v, farve, alpha, kant, linjebredde):
            # kant/linjebredde bevares i ALLE 3 tilfælde — ellers mister median-søjlen (sort
            # kant) sin eneste visuelle forskel fra de 4 tynde kørsel-søjler, uanset hvilket
            # af de 3 tilfælde den rammer.
            if v is None:
                # reelt eksploderet — ingen søjle at vise, kun skravering
                ax.bar(x, loft, width=bredde, color='none', edgecolor=kant, hatch='///', linewidth=linjebredde)
            elif v > loft:
                # gyldigt, endeligt resultat — bare større end denne rækkes skala. Vis en RIGTIG
                # (fyldt) søjle afskåret ved loftet, ikke en skraveret — søjlens egen farve
                # signalerer "dette TALTE med", trekanten "men tallet er større end vist".
                ax.bar(x, loft, width=bredde, color=farve, alpha=alpha, edgecolor=kant, linewidth=linjebredde)
                ax.plot(x, loft, marker='^', color=(kant if kant != 'none' else farve), markersize=5,
                        zorder=5, clip_on=False)
            else:
                ax.bar(x, max(v, 0.0), width=bredde, color=farve, alpha=alpha, edgecolor=kant, linewidth=linjebredde)

        for m, (farve, løb, median) in enumerate(zip(farver, kørsler, medianer)):
            tegn_søjle(m + offsets[0], median, farve, alpha=1.0, kant='black', linjebredde=1.4)
            for k, v in enumerate(løb):
                tegn_søjle(m + offsets[k + 1], v, farve, alpha=0.5, kant=farve, linjebredde=0.6)

        ax.set_xticks(range(len(navne)))
        ax.set_xticklabels(navne, fontsize=9, rotation=20, ha='right')
        # lidt luft over loftet, ellers klippes den lille trekant-markør (clip_on=False) af
        # figurkanten i stedet for at ses ovenpå den afskårne søjle.
        ax.set_ylim(0, loft * 1.06)
        ax.set_ylabel('afstand til bedste mulige')
        ax.set_title(titel or 'sammenligning')
    return _Panel(tegn, figsize=(4.5, 5))


def loss_kontur(loss_fn, a_range, b_range, resolution=60, punkt=None, path=None, gradient=None, gradient_farve='white', colorbar=True, titel=None, ekstra_stier=None, ekstra_pile=None, equal_aspect=False, farve_range=None, skala=None):
    """
    Loss-flade som kontur-plot over (a, b). loss_fn skal være en funktion af (a, b) —
    typisk en lambda der wrapper en af jeres egne opg3/opg4-funktioner, fx:
        loss_fn = lambda a, b: opg3_2(a, b, punkter)
    Har din opgave 3 parametre (fx opg3_3/opg3_4), så fastlås den tredje i lambdaen:
        loss_fn = lambda a, b: opg3_3(a, b, c_fast, punkter)

    punkt: valgfri (a, b)-markør, fx den man selv har valgt at prøve.
    path: valgfri liste af (a, b)-punkter der tegnes som en sti med en markør ved det sidste
    (fx den vej gradient descent har gået indtil videre) — bruges i stedet for punkt.
    gradient: valgfri (grad_a, grad_b) — tegner en pil fra punkt/sti's sidste sted som
    den fulde vektor (grad_a, grad_b), altså i skala med (a, b)-akserne. Den rå gradient
    er ofte for stor til at se noget fornuftigt — skalér den selv (fx med en fast
    visningsskala, uafhængig af jeres learning rate) før I sender den ind.
    colorbar: sæt til False når panelet skal redraws gentagne gange på samme akse (fx i
    animate()) — ellers skrumper aksen lidt for hvert kald.
    titel: valgfri overskrift i stedet for standard-teksten 'Loss (kontur)' — praktisk i et
    display_grid() med mange paneler, hvor hvert panel skal vise sit eget resultat.
    gradient_farve: farven på 'gradient'-pilen (default hvid) — brug fx til at farvekode hvad
    pilen viser (fx orange for 'det endelige skridt der rent faktisk blev taget', for at
    matche samme farve brugt i et andet panel).
    ekstra_stier: valgfri liste af (punkter, farve) eller (punkter, farve, label) — tegner flere
    stier oveni den primære 'path', hver med en lille cirkel ved sit sidste punkt. Bruges fx til
    at vise to metoders stier på samme landskab, eller (i træningsloop-konkurrencen) loss-kald og
    gradient-kald som to forskellige stier. Giver mindst én sti et 'label', tegnes der automatisk
    en legend (til fx at sammenligne flere metoders stier fra samme startpunkt).
    ekstra_pile: valgfri liste af (a, b, grad_a, grad_b, farve, alpha) — tegner FLERE pile,
    hver med sit EGET udgangspunkt (a, b) (i stedet for kun fra punkt/sti's sidste sted, som
    'gradient' gør). Bruges fx til at vise et helt forløbs historiske gradienter, hver tegnet
    der hvor den faktisk blev målt.
    equal_aspect: sæt til True for at give a- og b-aksen samme antal pixels pr. data-enhed —
    ellers strækkes panelet altid til et kvadrat, uanset hvor bredt/smalt a_range/b_range
    reelt er. Brug det til at ZOOME ind med et asymmetrisk, lille a_range/b_range (fx et
    lille vindue omkring jeres nuværende punkt) — så bliver selve panelets BREDDE/HØJDE (ikke
    tal på akserne) den ting man kan SE hvor skævt landskabet er skaleret i de to retninger,
    fx til at vise RMSprops idé (skalér hver akse med dens egen gradient-størrelse).
    farve_range: valgfrit (min, max) — bruges til at give FLERE paneler samme (lineære)
    farveskala, så samme farve betyder samme loss-værdi uanset panel — samme princip som
    standard-farveskalaen andre steder i notebooken, bare eksplicit delt mellem paneler i
    stedet for at hvert panel selv autoskalerer til sine egne data. Brug fx
    sesy_viz.loss_range(...) på det fulde landskab, og send samme farve_range til både et
    udzoomet og et zoomet panel. NB: et lille, zoomet vindue dækker typisk kun en brøkdel af
    hele landskabets loss-interval, så farven derinde kan ende med at se ret ensartet ud —
    det er prisen for at kunne sammenligne på tværs af paneler.
    skala: valgfri (skala_a, skala_b) — tegner loss-FLADEN i et ÆGTE omskaleret koordinatsystem
    (u = a·skala_a, v = b·skala_b) i stedet for de rå (a, b)-akser, dvs. konturen ÆNDRER FORM
    (en aflang dal kan blive rund) — det er ikke det samme som blot at zoome ind (equal_aspect),
    som viser den SAMME, urørte form, bare tættere på. Bruges til RMSprops/Adams idé: skalér
    hver akse med dens egen (rod af) gradient-størrelse. punkt/path/ekstra_stier/ekstra_pile's
    UDGANGSPUNKTER omregnes automatisk (ganges med skala) — men selve vektorerne I sender via
    'gradient'/'ekstra_pile' gør IKKE, da en forskydning og en rå gradient transformerer MODSAT
    af hinanden under omskaleringen (en forskydning ganges med skala, en gradient DELES med
    skala) — funktionen kan ikke gætte hvilken af de to I mener, så send selv vektoren i de
    rigtige, allerede-omregnede enheder. Sætter selv aksernes forhold til 'equal' (ellers kan
    man ikke SE om omskaleringen faktisk gjorde noget). Giv selv et lille, lokalt a_range/b_range
    (centreret om jeres nuværende punkt) for stadig at kunne zoome ind OVEN PÅ transformationen
    — det fulde landskabs range bliver typisk enormt (og uinformativt) i det omskalerede rum.
    Kør alene for at se plottet, eller giv den til display()/animate() sammen med andre paneler.
    """
    def tegn(ax):
        if skala is not None:
            skala_a, skala_b = skala
            grid_loss_fn = lambda aa, bb: loss_fn(aa / skala_a, bb / skala_b)
            grid_a_range = (a_range[0] * skala_a, a_range[1] * skala_a)
            grid_b_range = (b_range[0] * skala_b, b_range[1] * skala_b)
        else:
            skala_a, skala_b = 1.0, 1.0
            grid_loss_fn, grid_a_range, grid_b_range = loss_fn, a_range, b_range

        def omskaler(p):
            return (p[0] * skala_a, p[1] * skala_b)

        A, B, L = _loss_grid(grid_loss_fn, grid_a_range, grid_b_range, resolution)
        _tegn_kontur(ax, A, B, L, colorbar=colorbar, titel=titel or 'Loss (kontur)', farve_range=farve_range)
        # eksplicit sat (i stedet for at stole på contourf's egen autoscale) — nødvendigt i
        # animate(), som genbruger samme ax hvert frame: når autoscale først er slået fra
        # (linjen herunder), opdaterer et senere contourf-kald IKKE længere ax'ets grænser af
        # sig selv, så uden dette ville et panel hvor a_range/b_range ÆNDRER sig fra frame til
        # frame (fx et zoomet/omskaleret panel med et lokalt, bevægeligt vindue) fryse fast på
        # det første frames grænser, og resten af animationen ville se tom ud.
        ax.set_xlim(grid_a_range)
        ax.set_ylim(grid_b_range)
        if equal_aspect or skala is not None:
            ax.set_aspect('equal', adjustable='box')
        ax.autoscale(False)  # markør/sti må ikke flytte akserne yderligere
        if ekstra_stier:
            har_labels = False
            for item in ekstra_stier:
                punkter, farve = item[0], item[1]
                label = item[2] if len(item) > 2 else None
                har_labels = har_labels or bool(label)
                if punkter:
                    punkter_skaleret = [omskaler(p) for p in punkter]
                    ax.plot([p[0] for p in punkter_skaleret], [p[1] for p in punkter_skaleret],
                            color=farve, linewidth=1.5, alpha=0.85, label=label)
                    ax.plot([punkter_skaleret[-1][0]], [punkter_skaleret[-1][1]], 'o', color=farve, markersize=6, zorder=5)
            if har_labels:
                ax.legend(fontsize=8, loc='best')
        sted = None
        if path:
            path_skaleret = [omskaler(p) for p in path]
            a_vals = [p[0] for p in path_skaleret]
            b_vals = [p[1] for p in path_skaleret]
            ax.plot(a_vals, b_vals, color='white', linewidth=1.5, alpha=0.8)
            ax.plot([a_vals[-1]], [b_vals[-1]], 'o', color='tomato', markersize=10, zorder=5)
            sted = (a_vals[-1], b_vals[-1])
        elif punkt is not None:
            punkt_skaleret = omskaler(punkt)
            ax.plot([punkt_skaleret[0]], [punkt_skaleret[1]], 'o', color='tomato', markersize=10, zorder=5)
            sted = punkt_skaleret

        def tegn_pil(oprindelse, grad_a, grad_b, farve, alpha=1.0, zorder=7):
            # tolerance i stedet for præcis 0-tjek: ved den optimale løsning er gradienten
            # matematisk 0, men flydende-komma-regning giver typisk noget i stil med 1e-14
            if abs(grad_a) <= 1e-9 and abs(grad_b) <= 1e-9:
                return
            spids = (oprindelse[0] + grad_a, oprindelse[1] + grad_b)
            ann = ax.annotate('', xy=spids, xytext=oprindelse,
                         annotation_clip=False,  # ellers tegnes pilen slet ikke hvis den rager uden for a/b_range
                         arrowprops=dict(facecolor=farve, edgecolor='black', alpha=alpha,
                                          linewidth=1.5, width=3, headwidth=10, headlength=10),
                         zorder=zorder)
            # skær pilen af ved panelets egen kant — ellers kan en stor gradient
            # (fx tidligt i gradient descent) tegnes hen over nabo-panelerne i display()/animate()
            ann.arrow_patch.set_clip_path(ax.patch)

        if gradient is not None and sted is not None:
            tegn_pil(sted, gradient[0], gradient[1], gradient_farve)
        if ekstra_pile:
            # zorder=8 (over den primære 'gradient's 7): ekstra-pile er typisk den mere
            # specifikke/interessante af de to (fx en lille, præcis pil oveni en stor,
            # generel baggrunds-pil) og skal kunne ses selvom den ligger inden i den store.
            for a_p, b_p, grad_a, grad_b, farve, alpha in ekstra_pile:
                tegn_pil(omskaler((a_p, b_p)), grad_a, grad_b, farve, alpha=alpha, zorder=8)
    return _Panel(tegn, figsize=(6, 5))


def vektor_vifte(par, sum_vektor=None, titel=None, original_farve='gray', skaleret_farve='white', sum_farve='orange', margin_andel=0.3):
    """
    Tegner en 'vifte' af vektor-PAR fra samme udgangspunkt (0, 0) — på sit EGET, tomme
    koordinatsystem der selv skalerer til at passe vektorernes egen størrelse, IKKE oven på
    et loss-landskab. Bruges når selve landskabets akser ville gøre vektorerne alt for små
    til at se noget (fx momentums tidligere gradient-bidrag, som typisk er MEGET mindre end
    landskabets egen skala).
    par: liste af ((dx_reference, dy_reference), (dx_nu, dy_nu)) — begge tegnes oveni hinanden
    for HVERT historisk punkt: reference i baggrunden (fast værdi, ÆNDRER SIG IKKE fra frame til
    frame — fx hvor stort bidraget var lige da det først kom med), nu i forgrunden (den aktuelle,
    vægtede størrelse). Forskellen i længde viser direkte hvor meget hvert bidrag er skrumpet
    siden det først dukkede op.
    sum_vektor: valgfri (dx, dy) — SUMMEN af alle 'nu'-vektorerne, i sin egen farve. Modsat
    reference/nu er denne IKKE historisk akkumuleret: I giver den forfra hvert frame, så kun
    det aktuelle billedes sum vises (den forrige forsvinder, ligesom gradient-pilene i de
    andre paneler).
    Zoom bestemmes KUN af 'nu'-vektorerne (+ sum) — dem er vi interesseret i at se tydeligt —
    plus en margin, der er en FAST brøkdel (margin_andel) af den STØRSTE reference-vektor
    nogensinde (ikke af 'nu'-vektorernes egen, ofte meget lille, udstrækning — en margin
    relativt til DEM ville kollapse mod ~0 lige efter et fald i gradientstørrelse). Reference-
    pilene tegnes stadig fuldt ud og må gerne rage ud over kanten (klippes pænt af der) — de er
    baggrund, ikke det zoomen skal følge. Det giver et stabilt, forudsigeligt zoom-niveau selv
    når gradientstørrelsen vokser og falder undervejs (fx når man slipper ud af et lokalt minimum).
    """
    def tegn(ax):
        nu_punkter = [skal for _, skal in par] + [(0.0, 0.0)]
        if sum_vektor is not None:
            nu_punkter.append(sum_vektor)
        xs = [p[0] for p in nu_punkter]
        ys = [p[1] for p in nu_punkter]
        største_reference = max((max(abs(dx_o), abs(dy_o)) for (dx_o, dy_o), _ in par), default=1.0)
        margin = margin_andel * største_reference

        x_midt, y_midt = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        halv_side = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 + margin
        ax.set_xlim(x_midt - halv_side, x_midt + halv_side)
        ax.set_ylim(y_midt - halv_side, y_midt + halv_side)

        def tegn_pil(dx, dy, farve, kant, alpha, bredde, hovedbredde, hovedlængde, zorder):
            if abs(dx) <= 1e-12 and abs(dy) <= 1e-12:
                return
            ann = ax.annotate('', xy=(dx, dy), xytext=(0, 0), annotation_clip=False,
                         arrowprops=dict(facecolor=farve, edgecolor=kant, alpha=alpha,
                                          linewidth=bredde, width=bredde * 2.5,
                                          headwidth=hovedbredde, headlength=hovedlængde),
                         zorder=zorder)
            ann.arrow_patch.set_clip_path(ax.patch)  # klip pænt ved kanten i stedet for at forsvinde/flyde ud

        for (dx_o, dy_o), (dx_s, dy_s) in par:
            tegn_pil(dx_o, dy_o, original_farve, original_farve, 0.5, 1.0, 6, 6, zorder=3)
            tegn_pil(dx_s, dy_s, skaleret_farve, 'black', 0.9, 1.2, 8, 8, zorder=5)
        if sum_vektor is not None:
            tegn_pil(sum_vektor[0], sum_vektor[1], sum_farve, 'black', 1.0, 1.5, 10, 10, zorder=6)

        ax.axhline(0, color='gray', linewidth=0.6, zorder=0)
        ax.axvline(0, color='gray', linewidth=0.6, zorder=0)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(titel or 'vektorer')
        ax.set_xlabel('Δa')
        ax.set_ylabel('Δb')
    return _Panel(tegn, figsize=(5, 5))


def loss_3d(loss_fn, a_range, b_range, resolution=60, punkt=None):
    """Samme loss-flade som loss_kontur, men som en 3D-flade. Se loss_kontur for detaljer om loss_fn."""
    def tegn(ax):
        A, B, L = _loss_grid(loss_fn, a_range, b_range, resolution)
        _tegn_3d(ax, A, B, L)
        if punkt is not None:
            a, b = punkt
            ax.scatter([a], [b], [loss_fn(a, b)], color='tomato', s=80, zorder=10,
                       depthshade=False, edgecolor='black', linewidth=0.5)
    return _Panel(tegn, figsize=(6, 5), projection='3d')


def modelfit(*args, x_range=None, y_range=None):
    """
    Vis modellen sammen med punkterne og fejl-kvadraterne — linjen y=a·x+b hvis I giver
    2 parametre, parablen y=a·x²+b·x+c hvis I giver 3. Kald fx som:
        sesy_viz.modelfit(a, b, punkter)
        sesy_viz.modelfit(a, b, c, punkter)
    Angiv x_range/y_range for at låse aksernes skala på tværs af flere kald med
    forskellige parametre — ellers deformeres kvadraterne når fejlens størrelse ændrer sig.
    Kør alene for at se plottet, eller giv den til display() sammen med andre paneler.
    """
    *params, punkter = args
    xs = np.array([p[0] for p in punkter], dtype=float)
    ys = np.array([p[1] for p in punkter], dtype=float)
    # uden eksplicit x_range/y_range: lad _tegn_kvadrater_panel selv beregne plads til
    # fejl-kvadraterne (ellers bliver de skåret af når fejlen er stor)
    x_min, x_max = x_range if x_range else (None, None)
    y_min, y_max = y_range if y_range else (None, None)

    def tegn(ax):
        errors = _tegn_kvadrater_panel(ax, xs, ys, params, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
        ax.set_aspect('equal', adjustable='box')
        loss = float(np.sum(errors**2))
        label = '  '.join(f'{navn}={v:.2f}' for navn, v in zip('abc', params))
        ax.set_title(f'{label}   loss={loss:.2f}')
        ax.legend(fontsize=9)
    return _Panel(tegn, figsize=(6, 5))


def linalg_losning(A, y, AtA, Aty, AtA_inv, w, fmt="{:.2f}"):
    """
    Visualiserer regnestykket bag den analytiske løsning, byggesten for byggesten —
    I bygger selv A, y, AᵀA, Aᵀy og w i cellen (se opgave 1_4/closed_form_linje for
    samme lineære algebra), denne funktion tegner dem bare op som matricer:
        Aᵀ · A = AᵀA
        Aᵀ · y = Aᵀy
        (AᵀA)⁻¹ · Aᵀy = w
    Alle entries i A, Aᵀ og y er synlige — det eneste der er en "sort boks" er selve
    matrix-inversionen (AᵀA)⁻¹, resten er bare matrix-multiplikation I kan tjekke tal for tal.
    Kør alene for at se plottet, eller giv den til display() sammen med andre paneler.
    """
    At = np.array(A, dtype=float).T
    rows_spec = [
        [("matrix", "Aᵀ", At), ("symbol", "·"), ("matrix", "A", A), ("symbol", "="), ("matrix", "AᵀA", AtA)],
        [("matrix", "Aᵀ", At), ("symbol", "·"), ("matrix", "y", y), ("symbol", "="), ("matrix", "Aᵀy", Aty)],
        [("matrix", "(AᵀA)⁻¹", AtA_inv), ("symbol", "·"), ("matrix", "Aᵀy", Aty), ("symbol", "="), ("matrix", "w", w),
         ("symbol", "="), ("matrix", "", ["a", "b"], None)],
    ]

    def tegn(ax):
        ax.axis('off')
        margin  = 0.85
        højder  = [max(_matrix_hojde(item[2]) for item in row if item[0] == "matrix")
                   for row in rows_spec]
        y_cursor = sum(højder) / 2 + margin * (len(rows_spec) - 1) / 2
        centre = []
        for h in højder:
            y_cursor -= h / 2
            centre.append(y_cursor)
            y_cursor -= h / 2 + margin

        bredde = 0.0
        for row, yc in zip(rows_spec, centre):
            x = 0.0
            for item in row:
                if item[0] == "matrix":
                    item_fmt = item[3] if len(item) > 3 else fmt
                    x = _tegn_matrix(ax, x, yc, item[1], item[2], item_fmt)
                else:
                    x = _tegn_symbol(ax, x, yc, item[1])
            bredde = max(bredde, x)

        ax.set_xlim(-0.3, bredde + 0.3)
        ax.set_ylim(centre[-1] - højder[-1] / 2 - 0.4, centre[0] + højder[0] / 2 + 0.4)
    return _Panel(tegn, figsize=(11, 6.5))


def loss_over_tid(losses):
    """Viser loss pr. skridt som en kurve, med en markør ved det seneste skridt."""
    def tegn(ax):
        ax.plot(range(len(losses)), losses, color='lightgray', linewidth=1.5)
        ax.plot([len(losses) - 1], [losses[-1]], 'o', color='tomato', markersize=8, zorder=5)
        ax.set_xlabel('skridt')
        ax.set_ylabel('loss')
        ax.set_title('Squared loss over tid')
    return _Panel(tegn, figsize=(6, 4.5))


def animate(paneler_per_frame, interval=120, figsize=None, max_frames=25):
    """
    Animér en liste af panel-tupler, én tuple pr. frame — genbruger de samme
    panel-funktioner som display(), fx:
        paneler_per_frame = [
            (modelfit(a0, b0, punkter, x_range=(0,5), y_range=(0,5)),
             loss_kontur(loss_fn, a_range=(-3,4), b_range=(-2,6), path=[(a0,b0)])),
            (modelfit(a1, b1, punkter, x_range=(0,5), y_range=(0,5)),
             loss_kontur(loss_fn, a_range=(-3,4), b_range=(-2,6), path=[(a0,b0),(a1,b1)])),
            ...
        ]
        sesy_viz.animate(paneler_per_frame)
    Hvert frame tegnes forfra med de(t) panel(er) man har bygget til lige netop det skridt.

    max_frames: hvert frame kræver en fuld figur-redraw (akser, tekst, det hele) for at
    kunne gemmes som sit eget billede — det er langt den dyreste del af denne funktion,
    ca. konstant tid pr. frame uanset hvor simpelt panelet er. Har I bygget flere frames
    end dette, vises kun hvert n'te (altid inkl. det sidste), og interval skaleres op så
    den samlede afspilningstid stort set er den samme. Selve beregningen af alle jeres
    paneler er stadig billig — sæt max_frames=None for at tegne dem alle alligevel.
    """
    if max_frames and len(paneler_per_frame) > max_frames:
        skridt = -(-len(paneler_per_frame) // max_frames)  # rundet op
        udvalgte = list(paneler_per_frame[::skridt])
        if udvalgte[-1] is not paneler_per_frame[-1]:
            udvalgte.append(paneler_per_frame[-1])
        interval = interval * skridt
        paneler_per_frame = udvalgte

    n = len(paneler_per_frame[0])
    fig  = plt.figure(figsize=figsize or (6 * n, 5))
    axes = [fig.add_subplot(1, n, i + 1, projection=paneler_per_frame[0][i].projection) for i in range(n)]
    # oprindelig placering af hver ax — skal genskabes hvert frame, ellers skrumper fx
    # colorbars aksen lidt mere for hvert kald og den ender som en tynd streg
    positioner = [ax.get_position() for ax in axes]
    # ax.clear() er markant dyrere end det ser ud til: det tvinger matplotlib til at
    # genopbygge hele akse-layoutet (tick-labels, skriftstørrelser osv.) fra bunden hvert
    # frame — det er det der gør 50 frames langsomme. i stedet fjerner vi kun de artists
    # forrige frame selv tilføjede (linjer, konturflade, pil), og lader akse/titel/labels
    # stå urørt, siden de alligevel sættes til det samme hver gang.
    forrige_artists = [None] * n

    def _animate(i):
        for ax in list(fig.axes):       # ryd ekstra akser (fx colorbars) fra forrige frame
            if ax not in axes:
                fig.delaxes(ax)
        for idx, (ax, panel, pos) in enumerate(zip(axes, paneler_per_frame[i], positioner)):
            if forrige_artists[idx] is not None:
                for artist in forrige_artists[idx]:
                    if artist in ax.get_children():
                        artist.remove()
            før = set(ax.get_children())
            ax.set_position(pos)
            panel._tegn(ax)
            forrige_artists[idx] = [a for a in ax.get_children() if a not in før]
        return axes

    anim = FuncAnimation(fig, _animate, frames=len(paneler_per_frame), interval=interval, blit=False)
    plt.tight_layout()
    html = HTML(anim.to_jshtml())
    plt.close(fig)
    return html


# ── Feedback-plots til regression_test.py ─────────────────────────────────

def feedback_kurve(punkter, got_params, exp_params):
    """
    Feedback for opg1/opg2: viser datapunkter, studentens kurve (rød stiplet)
    og facit-kurven (grøn).
    Bruges ved fejl i linje- og parabeL-opgaver.
    """
    xs = [p[0] for p in punkter]
    ys = [p[1] for p in punkter]
    pad    = max((max(xs) - min(xs)) * 0.2, 0.5)
    x_min  = min(xs) - pad
    x_max  = max(xs) + pad
    x_line = np.linspace(x_min, x_max, 300)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(xs, ys, color='black', s=60, zorder=5, label='punkter')

    try:
        ax.plot(x_line, _eval_kurve(got_params, x_line), 'r--', linewidth=2,
                label=_kurve_label(got_params, prefix="din"))
    except Exception:
        pass

    ax.plot(x_line, _eval_kurve(exp_params, x_line), color='green', linewidth=2,
            label=_kurve_label(exp_params, prefix="korrekt"))

    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


def feedback_haeldning(f, x, got, exp, span=None):
    """
    Feedback for opg4's differentiations-opgaver: viser grafen for f (forskriften),
    punktet (x, f(x)), og hældningen som en pil igennem punktet — jeres svar (rød)
    mod facit (grøn). En forkert hældning ses direkte som en forkert vinkel på pilen.
    """
    x = float(x)
    y = float(f(x))
    if span is None:
        span = max(abs(x), 1.0) * 2.0 + 2.0
    xs = np.linspace(x - span, x + span, 300)
    ys = [float(f(xv)) for xv in xs]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(xs, ys, color='royalblue', linewidth=2, label='f(x)', zorder=2)
    ax.scatter([x], [y], color='black', s=60, zorder=5, label=f'punkt: x={x:.2g}')

    pad = (max(ys) - min(ys)) * 0.3 + 0.5
    ax.set_xlim(x - span, x + span)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.autoscale(False)  # pilene må ikke skubbe grafen ud af syne, selv hvis hældningen er meget forkert

    dx = span * 0.3
    for hældning, farve, navn in [(exp, 'green', 'korrekt'), (got, 'tomato', 'din')]:
        ax.annotate('', xy=(x + dx, y + hældning * dx), xytext=(x, y),
                     arrowprops=dict(facecolor=farve, edgecolor=farve, linewidth=2,
                                      width=2.5, headwidth=9, headlength=10), zorder=6)
        ax.plot([], [], color=farve, linewidth=2.5, label=f"{navn}: f'({x:.2g}) = {hældning:.3g}")

    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


def feedback_gradient_3d(f, x, y, got, exp, span=None, resolution=40):
    """
    Feedback for opg5's gradient-opgaver: viser fladen z=f(x,y) i 3D, punktet
    (x, y, f(x,y)), og gradienten som en pil i tangentplanen — jeres svar (rød)
    mod facit (grøn). Den korrekte pil følger fladen tæt omkring punktet;
    en forkert gradient peger i en forkert retning/hældning oven på fladen.
    got og exp er hver (df/dx, df/dy).
    """
    x, y = float(x), float(y)
    z = float(f(x, y))
    if span is None:
        span = max(abs(x), abs(y), 1.0) * 1.5 + 1.5

    xs = np.linspace(x - span, x + span, resolution)
    ys = np.linspace(y - span, y + span, resolution)
    X, Y = np.meshgrid(xs, ys)
    Z = np.empty_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f(X[i, j], Y[i, j])

    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.computed_zorder = False
    ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=True, alpha=0.75, zorder=1)
    ax.scatter([x], [y], [z], color='black', s=50, zorder=10, depthshade=False, edgecolor='black')

    d = span * 0.35
    for (gx, gy), farve, navn in [(exp, 'green', 'korrekt'), (got, 'tomato', 'din')]:
        dz = gx * d + gy * d  # stigningen langs tangentplanen i den retning
        ax.quiver(x, y, z, d, d, dz, color=farve, linewidth=2.5, arrow_length_ratio=0.2, zorder=11,
                   label=f"{navn}: ∇f = ({gx:.2g}, {gy:.2g})")

    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('f(x,y)')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def feedback_loss(model_params, punkter, got_loss, exp_loss):
    """
    Feedback for opg3/opg4: viser kurven, punkterne og et kvadrat per fejl.
    Kvadratets areal = fejlens bidrag til squared loss.
    Bruges ved fejl i loss-beregnings-opgaver. model_params er (a,b) eller (a,b,c).
    """
    xs = np.array([p[0] for p in punkter], dtype=float)
    ys = np.array([p[1] for p in punkter], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    _tegn_kvadrater_panel(ax, xs, ys, model_params)
    ax.set_title(
        f'Kvadraternes arealer summer til: {got_loss:.4f}\n'
        f'Korrekt loss = {exp_loss:.4f}'
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


def feedback_metode(loss_fn, start, got, exp, margin=1.4, resolution=60):
    """
    Feedback for træningsloop-blokkens gd/momentum/rmsprop/adam-opgaver: viser loss-landskabet
    I faktisk blev testet på, med start (sort), jeres endepunkt (rød) og facit-endepunktet
    (grøn) tegnet oveni som hver sin sti. Zoomer automatisk til at indeholde alle tre punkter.
    got/exp må gerne have flere elementer end (a,b) (fx momentum/rmsprop/adams interne
    va,vb/sa,sb) — kun de 2 første (positionen) bruges.
    """
    a0, b0 = float(start[0]), float(start[1])
    ga, gb = float(got[0]), float(got[1])
    ea, eb = float(exp[0]), float(exp[1])
    a_vals, b_vals = [a0, ga, ea], [b0, gb, eb]
    a_span = max(max(a_vals) - min(a_vals), 1e-6) * margin
    b_span = max(max(b_vals) - min(b_vals), 1e-6) * margin
    a_mid, b_mid = (max(a_vals) + min(a_vals)) / 2, (max(b_vals) + min(b_vals)) / 2
    a_range = (a_mid - a_span / 2, a_mid + a_span / 2)
    b_range = (b_mid - b_span / 2, b_mid + b_span / 2)

    fig, ax = plt.subplots(figsize=(6, 5))
    A, B, L = _loss_grid(loss_fn, a_range, b_range, resolution)
    _tegn_kontur(ax, A, B, L, colorbar=True, titel='Loss (kontur)')
    ax.set_xlim(a_range); ax.set_ylim(b_range)
    ax.plot([a0, ga], [b0, gb], color='tomato', linewidth=2, marker='o', markersize=6, label='jeres svar')
    ax.plot([a0, ea], [b0, eb], color='limegreen', linewidth=2, marker='o', markersize=6, label='facit')
    ax.plot([a0], [b0], 'o', color='black', markersize=7, zorder=6, label='start')
    ax.set_xlabel('a'); ax.set_ylabel('b')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


# ── ReLU/torch-specifikke paneler ──────────────────────────────────────────
# (bruger modelfit/loss_kontur/loss_3d osv. ovenfor, men modellen har her ikke
# et fast antal navngivne parametre (a,b,c) — den er bygget af en liste af
# ReLU'er, eller er en torch-model, så den tages ind som en vilkårlig model_fn.)

def curve_fit(model_fn, punkter, model_label="model", x_range=None, y_range=None):
    """
    Ligesom modelfit ovenfor, men virker med EN VILKÅRLIG model_fn(x) -> y i stedet
    for kun linjer/parabler med et fast antal navngivne parametre. Bruges når
    modellen er bygget af flere ReLU'er (opgave 2), eller er en torch/nn.Module-model.
    """
    xs = np.array([p[0] for p in punkter], dtype=float)
    ys = np.array([p[1] for p in punkter], dtype=float)
    y_preds = np.array([float(model_fn(x)) for x in xs])
    errors = y_preds - ys

    x_min, x_max = x_range if x_range else (float(xs.min()) - 0.5, float(xs.max()) + 0.5)
    x_line = np.linspace(x_min, x_max, 300)
    y_line = np.array([float(model_fn(x)) for x in x_line])

    def tegn(ax):
        ax.plot(x_line, y_line, color='royalblue', linewidth=2, label=model_label, zorder=2)
        for x, y, y_pred, e in zip(xs, ys, y_preds, errors):
            side = abs(float(e))
            y_bot = min(float(y), float(y_pred))
            if side > 1e-10:
                ax.add_patch(Rectangle(
                    (x, y_bot), side, side,
                    linewidth=1, edgecolor='tomato', facecolor='tomato', alpha=0.25, zorder=3,
                ))
            ax.plot([x, x], [y, y_pred], color='tomato', linewidth=1.5, zorder=4)
        ax.scatter(xs, ys, color='black', s=40, zorder=6, label='punkter')
        if x_range:
            ax.set_xlim(*x_range)
        if y_range:
            ax.set_ylim(*y_range)
        loss = float(np.sum(errors ** 2))
        ax.set_title(f'{model_label}   SSE={loss:.3f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(fontsize=9)
    return _Panel(tegn, figsize=(6, 5))


def feedback_relu_fn(punkter, got_fn, exp_fn, x_range=None):
    """
    Feedback for opgave 1's "forbind punkterne"-opgaver: viser datapunkterne,
    studentens kurve (rød stiplet) og facit-kurven (grøn) — samme idé som
    feedback_kurve. got_fn/exp_fn er kaldbare funktioner af x.
    """
    xs = [p[0] for p in punkter]
    ys = [p[1] for p in punkter]
    if x_range:
        x_min, x_max = x_range
    else:
        pad = max((max(xs) - min(xs)) * 0.3, 1.0)
        x_min, x_max = min(xs) - pad, max(xs) + pad
    x_line = np.linspace(x_min, x_max, 300)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(xs, ys, color='black', s=60, zorder=5, label='punkter')

    try:
        y_got = [float(got_fn(x)) for x in x_line]
        ax.plot(x_line, y_got, 'r--', linewidth=2, label='jeres model')
    except Exception:
        pass

    y_exp = [float(exp_fn(x)) for x in x_line]
    ax.plot(x_line, y_exp, color='green', linewidth=2, label='korrekt model')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


def relu_komponenter(weights, biases, x_range=(-2, 4), c=0.0):
    """
    Viser hver vægtet ReLU-komponent w_i·relu(x−b_i) for sig (stiplet, tynd),
    og deres sum c + Σ w_i·relu(x−b_i) (fuld, sort linje) — samme model som opgave 2_1.
    """
    x = np.linspace(*x_range, 300)

    def tegn(ax):
        total = np.full_like(x, float(c))
        for i, (w, b) in enumerate(zip(weights, biases)):
            komponent = w * np.maximum(0, x - b)
            total = total + komponent
            ax.plot(x, komponent, '--', linewidth=1.2, alpha=0.7, label=f'w{i}·relu(x−b{i})')
        ax.plot(x, total, color='black', linewidth=2.5, label='sum (modellen)')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(fontsize=8)
        ax.set_title('ReLU-komponenter og deres sum')
    return _Panel(tegn, figsize=(6.5, 5))


"""
Landskaber + evalueringsharness til træningsloop-blokkens gradient-descent-konkurrence.
Eleverne skriver deres egen gd_metode(loss, gradient, start, ...) -> (a,b) — typisk versioneret
som egen_gd_metode_v1, v2, ... i takt med at de bygger flere idéer ind (momentum, RMSprop, ...)
— og sammenligner en HEL LISTE af metoder ad gangen med evaluer_gd_metode([("gd", gd_metode), ...]),
kørt på tværs af 4 landskaber × 4 udvalgte startpunkter.
"""

class BudgetOverskredet(Exception):
    """Rejses når en gd_metode-funktion har brugt alle sine kald til loss/gradient."""
    pass


# ── Landskabernes matematik ──────────────────────────────────────────────
# hvert landskab er loss(a,b) -> tal (ren aritmetik, virker på tal OG torch-tensorer, så
# eleverne selv kan kalde .backward() på det). De analytiske gradienter herunder er PRIVATE:
# de bruges kun til at finde landskabets minimum ved import — eleverne får kun loss.

def _generer_linje_punkter(n=100, a_sand=1.2, b_sand=1.5, x_range=(0, 10), stoej=0.6, seed=42):
    """~100 punkter normalfordelt omkring en linje — landskabet til minibatch/SGD-afsnittet,
    og konkurrencens 'linjefitting'-landskab."""
    rng = np.random.default_rng(seed)
    xs = rng.uniform(*x_range, n)
    ys = a_sand * xs + b_sand + rng.normal(0, stoej, n)
    return list(zip(xs.tolist(), ys.tolist()))


LINJE_PUNKTER = _generer_linje_punkter()


def _linje_loss(a, b, punkter=LINJE_PUNKTER):
    n = len(punkter)
    return sum((a * x + b - y) ** 2 for x, y in punkter) / n


def _linje_gradient(a, b, punkter=LINJE_PUNKTER):
    n = len(punkter)
    da, db = 0.0, 0.0
    for x, y in punkter:
        fejl = a * x + b - y
        da += 2 * fejl * x
        db += 2 * fejl
    return da / n, db / n


def _linje_minimum(punkter=LINJE_PUNKTER):
    """Normalligningerne — samme lukkede løsning som regression_facit."""
    xs = np.array([p[0] for p in punkter])
    ys = np.array([p[1] for p in punkter])
    A = np.column_stack([xs, np.ones_like(xs)])
    a, b = np.linalg.inv(A.T @ A) @ A.T @ ys
    return float(a), float(b)


# offentlige navne (uden understreg) til minibatch-afsnittet — "givet" infrastruktur, ligesom
# sesy_viz's egne funktioner: I skal ikke selv udlede MSE-gradienten igen, den kender I allerede
# fra regression — brug den bare på et TILFÆLDIGT UDPLUK af punkter i stedet for alle.
linje_loss = _linje_loss
linje_gradient = _linje_gradient


def _plateau_led(t, k):
    """Ét led i plateau-familien (jf. Refference/SOP/Gradient Decent's PlatauExample)
    — typisk 2 lokale minima langs denne akse."""
    return t ** 3 + k * t + 0.5 * t ** 4


def _plateau_led_grad(t, k):
    return 3 * t ** 2 + k + 2 * t ** 3


def _platau1d_loss(a, b):
    # "1D": den eneste interessante (svære) akse er a — b-aksen er en simpel skål (b²) der
    # trækker mod midterlinjen b=0, så landskabet ikke bare er en uendelig flad rygrad langs b.
    return _plateau_led(a, -0.25) + b ** 2


def _platau1d_gradient(a, b):
    return _plateau_led_grad(a, -0.25), 2 * b


def _platau2d_loss(a, b):
    return _plateau_led(a, -0.25) + _plateau_led(b, -0.8)


def _platau2d_gradient(a, b):
    return _plateau_led_grad(a, -0.25), _plateau_led_grad(b, -0.8)


def _rosenbrock_loss(a, b):
    return (1 - a) ** 2 + 100 * (b - a ** 2) ** 2


def _rosenbrock_gradient(a, b):
    da = -2 + 2 * a - 400 * (-a ** 2 + b) * a
    db = -200 * a ** 2 + 200 * b
    return da, db


# ── Registrering: her tilføjes nye landskaber ────────────────────────────

def _multistart_minimum(loss, gradient, view, forsoeg=40, skridt=3000, lr=0.01, seed=0):
    """
    Finder (tilnærmet) det globale minimum ved almindelig gradient descent fra mange
    tilfældige startpunkter, og beholder det bedste resultat. Køres ÉN gang ved import,
    ikke en del af elevernes kald-budget — bruges kun når 'minimum' ikke er kendt analytisk.
    """
    rng = np.random.default_rng(seed)
    a_min, a_max, b_min, b_max = view
    bedste = (None, None, np.inf)
    for _ in range(forsoeg):
        a = rng.uniform(a_min, a_max)
        b = rng.uniform(b_min, b_max)
        for _ in range(skridt):
            da, db = gradient(a, b)
            a -= lr * da
            b -= lr * db
            if not (np.isfinite(a) and np.isfinite(b)):
                break
        l = loss(a, b)
        if np.isfinite(l) and l < bedste[2]:
            bedste = (a, b, l)
    return bedste[0], bedste[1]


def _byg_landskab(navn, loss, gradient, a_range, b_range, minimum=None):
    """minimum: giv (a*,b*) hvis den kendes analytisk — ellers findes den selv (se
    _multistart_minimum). Sådan tilføjes et nyt landskab: bare loss+gradient+view."""
    if minimum is None:
        minimum = _multistart_minimum(loss, gradient, (*a_range, *b_range))
    a_star, b_star = minimum
    return {
        "navn": navn, "loss": loss,
        "a_range": a_range, "b_range": b_range,
        "minimum": (a_star, b_star), "min_loss": float(loss(a_star, b_star)),
    }


_platau1d_a_star, _ = _multistart_minimum(_platau1d_loss, _platau1d_gradient, (-2.1, 1.1, -2.1, 1.1))

LANDSKABER = [
    _byg_landskab("linjefitting", _linje_loss, _linje_gradient, (-1, 3), (-3, 3), minimum=_linje_minimum()),
    # b-view centreret om 0 (b²-skålens midterlinje) — anderledes range end a, som er valgt til
    # at ramme a-aksens 2 lokale minima. b-aksens minimum er PRÆCIST 0.0, låst eksplicit i
    # stedet for at stole på multistart-søgningens numeriske tilnærmelse for den akse.
    _byg_landskab("plateau (1D)", _platau1d_loss, _platau1d_gradient, (-2.1, 1.1), (-1, 1), minimum=(_platau1d_a_star, 0.0)),
    _byg_landskab("plateau (2D)", _platau2d_loss, _platau2d_gradient, (-2.1, 1.1), (-2.1, 1.1)),
    _byg_landskab("rosenbrock", _rosenbrock_loss, _rosenbrock_gradient, (-1.5, 1.5), (-1, 2), minimum=(1.0, 1.0)),
]


# ── 4 udvalgte startpunkter pr. landskab ──────────────────────────────────
# HÅNDPLUKKEDE (ikke et generisk hjørne-mønster) — valgt ved faktisk at køre gd/momentum/
# rmsprop/adam fra mange kandidat-punkter og beholde de 4 pr. landskab der viser mest
# forskellig opførsel: fx et sted alle metoder er enige, et sted de er dybt uenige (en metode
# vinder klart), osv. Se intro-ml-plan-memoen for den fulde udforskning bag disse tal.

INTERESSANTE_STARTPUNKTER = {
    "linjefitting":  [(-0.6, -2.4), (1.0, 0.0), (2.2, 1.8), (1.6, 0.9)],
    "plateau (1D)":  [(-1.78, -0.8), (1.0, 0.6), (-0.5, 0.0), (-0.3, -0.8)],
    "plateau (2D)":  [(-1.78, 0.78), (0.78, -1.78), (-0.5, -0.5), (-0.5, -1.78)],
    "rosenbrock":    [(-1.0, -0.5), (0.0, 0.0), (1.2, 1.7), (0.0, -0.9)],
}


# ── Instrumentering: tæller og logger kald til loss ─────────────────────

def _sikker_loss(fn, a, b):
    """Regner fn(a,b) med numpy-tal, så en eksploderet metode (fx Rosenbrock med for stort lr)
    giver et meget dårligt tal (inf/nan) i stedet for at vælte hele konkurrencen med en exception."""
    with np.errstate(all="ignore"):
        try:
            return float(fn(np.float64(a), np.float64(b)))
        except (OverflowError, ValueError):
            return float("nan")


_AFSTAND_GRAENSE = 1000  # et kald om et punkt længere væk end dette fra landskabets midte ansés som useriøst


def instrumenter(landskab, budget=100, afstand_graense=_AFSTAND_GRAENSE):
    """
    Pakker landskabets loss ind, så hvert kald tælles og huskes (positioner undersøgt).
    Eleverne får KUN loss(a,b) — gradienten finder de selv med .backward(). Hvert loss-kald
    (ét forward-pass før .backward()) tæller ét mod budgettet på i alt `budget` kald.

    Stopper kørslen (rejser BudgetOverskredet, med en forklarende print) i 2 tilfælde:
    - I har brugt alle jeres `budget` kald — jeres svar bliver automatisk det sidste punkt I
      nåede at undersøge (så I godt må stoppe FØR budgettet er brugt, ved selv at returnere —
      det er kun hvis I IKKE selv stopper i tide, at dette griber ind).
    - I beder om et punkt mere end `afstand_graense` fra landskabets midte — det regnes
      slet ikke på (undgår at meningsløst store tal vælter jeres egen udregning), og jeres
      svar bliver det sidste GYLDIGE punkt I undersøgte før det.

    loss returnerer landskabets loss uden om _sikker_loss, så a/b-tensorer med requires_grad
    beholder deres graf og .backward() virker. Positionerne huskes som almindelige tal.
    """
    a0, a1 = landskab["a_range"]
    b0, b1 = landskab["b_range"]
    centrum_a, centrum_b = (a0 + a1) / 2, (b0 + b1) / 2

    loss_punkter = []
    tilstand = {"antal": 0, "sidste": None, "for_langt_vaek": False}

    def _brug_et_kald(a, b):
        afstand = ((a - centrum_a) ** 2 + (b - centrum_b) ** 2) ** 0.5
        if afstand > afstand_graense:
            print(f"⚠ punktet ({a:.1f}, {b:.1f}) er {afstand:.0f} fra landskabets midte — "
                  f"det virker useriøst langt væk, stopper her.")
            tilstand["for_langt_vaek"] = True
            raise BudgetOverskredet("punkt for langt fra landskabet")
        tilstand["antal"] += 1
        if tilstand["antal"] > budget:
            print(f"⚠ I har brugt alle jeres {budget} loss-kald — "
                  f"bruger punkt nr. {budget} som jeres svar.")
            raise BudgetOverskredet("budget opbrugt")

    def _tal(x):
        return x.detach().item() if hasattr(x, "detach") else float(x)

    def loss(a, b):
        av, bv = _tal(a), _tal(b)            # almindelige tal til tælling/logning
        _brug_et_kald(av, bv)
        v = landskab["loss"](a, b)           # tensoren sendes igennem, så .backward() virker
        loss_punkter.append((av, bv, _tal(v)))
        tilstand["sidste"] = (av, bv)
        return v

    return loss, loss_punkter, tilstand


def koer_en_gang(gd_metode, landskab, start, budget=100):
    """
    Kør gd_metode(loss, start) ÉN gang på ét landskab, fra ét startpunkt. Returnerer et
    resultat-dict (landskab, start, slut, slut_loss, bedste_loss, loss_punkter) — selve
    visualiseringen af det bygger I i notebook'en.
    """
    loss, loss_punkter, tilstand = instrumenter(landskab, budget)
    try:
        slut = gd_metode(loss, start)
    except BudgetOverskredet:
        slut = tilstand["sidste"] or start
    a, b = slut
    slut_loss = _sikker_loss(landskab["loss"], a, b)
    return {
        "landskab": landskab["navn"],
        "a_range": landskab["a_range"], "b_range": landskab["b_range"],
        "start": start, "slut": (float(a), float(b)),
        "slut_loss": slut_loss,
        "bedste_loss": landskab["min_loss"],
        # True hvis metoden endte langt uden for landskabet (uanset om slut_loss selv er et
        # stort-men-endeligt tal eller nan/inf) — se instrumenter()'s afstands-tjek.
        "eksploderet": bool(tilstand["for_langt_vaek"]) or not np.isfinite(slut_loss),
        "loss_punkter": loss_punkter,
    }


# ── Farver: fast pr. kendt metode, så gd/momentum/rmsprop/adam altid ser ens ud ──────────

_PALET = {
    "blaa": "#2a78d6", "groen": "#008300", "magenta": "#e87ba4", "gul": "#eda100",
    "cyan": "#1baf7a", "orange": "#eb6834", "lilla": "#4a3aa7", "roed": "#e34948",
}
_KENDTE_METODE_FARVER = {
    "gd": _PALET["blaa"], "momentum": _PALET["groen"],
    "rmsprop": _PALET["magenta"], "adam": _PALET["gul"],
}
_EKSTRA_FARVE_REKKEFOLGE = [_PALET["cyan"], _PALET["orange"], _PALET["lilla"], _PALET["roed"]]


def _tildel_farver(navne):
    """Fast farve for gd/momentum/rmsprop/adam (samme farve overalt i notebooken) — egne,
    ukendte metode-navne får en farve fra en fast rækkefølge, i den rækkefølge de optræder."""
    farver, næste = {}, 0
    for navn in navne:
        if navn in farver:
            continue
        if navn in _KENDTE_METODE_FARVER:
            farver[navn] = _KENDTE_METODE_FARVER[navn]
        else:
            farver[navn] = _EKSTRA_FARVE_REKKEFOLGE[næste % len(_EKSTRA_FARVE_REKKEFOLGE)]
            næste += 1
    return farver


# ── Sammenlign en liste af metoder på hele konkurrencen, i ét kald ───────

def evaluer_gd_metode(metoder, landskaber=LANDSKABER, budget=100, resolution=30):
    """
    Kør en eller flere metoder på tværs af alle landskaber og deres 4 udvalgte startpunkter,
    og vis resultatet som ét grid — én række pr. landskab. Første søjle i hver række er et
    søjlediagram (median slut-loss pr. metode + bedste mulige loss som stiplet linje), de næste
    4 paneler viser alle metoders sti overlejret på samme landskab, ét panel pr. startpunkt.

    metoder: liste af (navn, funktion)-par, fx [("gd", gd_metode), ("momentum", momentum_metode)]
    — skriv hele listen igen (med alt I vil sammenligne) hver gang I kalder denne, i stedet for
    at bygge videre på en gammel liste.
    """
    farver = _tildel_farver([navn for navn, _ in metoder])
    paneler, row_labels = [], []

    for landskab in landskaber:
        row_labels.append(landskab["navn"])
        starts = INTERESSANTE_STARTPUNKTER[landskab["navn"]]
        resultater = {navn: [koer_en_gang(fn, landskab, start, budget) for start in starts]
                      for navn, fn in metoder}

        # afstand til bedste mulige loss, én liste pr. metode — én værdi pr. udvalgt startpunkt
        # (samme rækkefølge som sti-panelerne herunder), None hvis den kørsel eksploderede.
        afstande = [
            [None if r["eksploderet"] else r["slut_loss"] - landskab["min_loss"] for r in resultater[navn]]
            for navn, _ in metoder
        ]

        # bar-panelet er den eneste farve/navn-forklaring i rækken (labeled x-akse) — de 4
        # sti-paneler til højre for det holder sig derfor bevidst UDEN egen legend, for ikke at
        # gentage samme 4 navne i alle 20 paneler i gridet.
        paneler.append(bar_sammenligning(
            [navn for navn, _ in metoder], afstande, [farver[navn] for navn, _ in metoder],
        ))

        for i, start in enumerate(starts):
            stier = []
            for navn, _ in metoder:
                r = resultater[navn][i]
                sti = [(a, b) for (a, b, _) in r["loss_punkter"]] or [start]
                stier.append((sti, farver[navn]))
            paneler.append(loss_kontur(
                landskab["loss"], landskab["a_range"], landskab["b_range"], resolution=resolution,
                ekstra_stier=stier, colorbar=False, titel=f"start=({start[0]:.2g}, {start[1]:.2g})",
            ))

    display_grid(paneler, cols=5, row_labels=row_labels)
