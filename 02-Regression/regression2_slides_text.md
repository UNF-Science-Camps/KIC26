# Regression 2 — slide-tekst

(ingen ny titel/om-mig/big-picture-åbning — dette er en fortsættelse af regression1-blokken, samme dag)

---

**slide**

recap: samme model, loss, gradient og skridt som i regression1 — men nu med ~100 punkter i stedet for 4

[animation fra worksheet]

**slide**

hvorfor regne gradienten ud fra alle punkter for hvert skridt? dyrt når man har mange punkter (millioner, for rigtige datasæt)

løsning: brug et tilfældigt udpluk (en batch) i stedet — cirka samme gradient, meget billigere

**slide**

Opgaver i at løse denne type opgaver:
- opg1_1
- opg1_2

**slide**

[animation fra worksheet]

prøv batch_størrelse = 1, 5, 20, 50, 100 — beskriv med egne ord hvad der ændrer sig (opg1_3, ingen facit, bare skriv jeres observation)

---

**slide**

kan vi komme hurtigere/mere direkte til minimum, uden nødvendigvis at bruge flere punkter pr. skridt? det er det metoder som Adam prøver at gøre

vi bygger 4 metoder, hver i to lag: selve opdateringen og løkken udenom, alle med samme form (loss, gradient, start, ...)

constraints: loss/gradient må kaldes maks 100 gange tilsammen pr. kørsel — testes på 4 landskaber (linjefitting + 3 ukendte), hver fra 4 udvalgte startpunkter

**almindelig gradient descent (opg2)**

samme gradient+skridt fra recap, bare omskrevet til (loss, gradient, start)-formen

**slide**

Opgaver i at løse denne type opgaver:
- opg2_1
- opg2_2

ekstra:
- opg2_3 (opg2_2 klarer sig især dårligt på rosenbrock — prøv at forbedre den)

**momentum (opg3)**

**ikke** en fysisk "bold der ruller ned ad bakken" — et vægtet, aftagende gennemsnit af alle gradienter man har set indtil videre

v_t = β\cdot v_{t-1} + (1-β)\cdot ∇L(a_t,b_t)
(a,b)_{t+1} = (a,b)_t - lr\cdot v_t

β (typisk ~0.9) styrer hvor meget vægt ældre gradienter stadig har

**slide**

Opgaver i at løse denne type opgaver:
- opg3_1
- opg3_2
- opg3_3

ekstra:
- opg3_4 (egen momentum-variant — andet β, eller en anden idé)

**slide**

[animation fra worksheet]

**rmsprop (opg4)**

skalér hver akse med sin egen (nyligt sete) gradient-størrelse, så der tages lige store skridt i alle retninger — uanset om aksen er stejl eller flad

s_t = β\cdot s_{t-1}+(1-β)\cdot ∇L(a_t,b_t)^2
(a,b)_{t+1} = (a,b)_t - lr\cdot ∇L(a_t,b_t)/(√s_t+ε)

(s regnes elementvis pr. parameter — a og b får hver deres egen skalering)

**slide**

Opgaver i at løse denne type opgaver:
- opg4_1
- opg4_2
- opg4_3

ekstra:
- opg4_4 (egen rmsprop-variant — andet β/lr, eller kombinér med en idé fra momentum)

**slide**

[animation fra worksheet]

**adam (opg5)**

kombinerer momentum (v) og RMSprop (s), plus en bias-korrektion af begge — vigtig i de første skridt, hvor v_0=s_0=0 ellers trækker gennemsnittet kunstigt mod nul

v̂_t = v_t/(1-β_1^t),  ŝ_t = s_t/(1-β_2^t)
(a,b)_{t+1} = (a,b)_t - lr\cdot v̂_t/(√ŝ_t+ε)

**slide**

Opgaver i at løse denne type opgaver:
- opg5_1
- opg5_2

**slide**

[animation fra worksheet]

overblik: gd vs. momentum vs. rmsprop vs. adam, samlet i ét sammenligningsgrid

**slide**

ekstra:
- opg5_3 (byg jeres bedste metode — kombinér gerne idéer fra gd/momentum/rmsprop/adam, eller prøv noget nyt)
