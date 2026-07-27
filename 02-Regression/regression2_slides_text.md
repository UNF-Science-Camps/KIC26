# Regression 2 — slide-tekst

(ingen ny titel/om-mig/big-picture-åbning — dette er en fortsættelse af regression1-blokken, samme dag)

---

**slide**

i regression 1 fandt I selv gradienten ved at differentiere i hånden — det virker fint for simple modeller, men bliver upraktisk (og nogle gange umuligt) med tusindvis af parametre og en kompliceret loss-funktion

løsning: autograd — PyTorch følger med i alle udregninger og finder den afledte for jer bagefter, uanset hvor kompliceret udregningen er

**slide**

gør en variabel til en **tensor** med `requires_grad=True`, byg en udregning med den, og kald `.backward()` — gradienten ligger nu i `.grad`

[eksempler fra worksheet: f(x)=5x² (potensreglen), g(x)=(2x+1)² (kædereglen), mse(a,b,punkter) (gradient mht. a og b på én gang)]

**slide**

nu bruger vi det samme værktøj på selve loss-funktionen fra regression 1 (MSE for linjen y=a·x+b) — bare med 100 punkter i stedet for 4

[kode fra worksheet: loss(a,b,punkter), gradient(a,b,punkter) via .backward(), step(a,b,punkter,lr)]

Opgaver i at løse denne type opgaver (refleksion, ingen facit):
- opg1_1
- opg1_2

**slide**

[animation fra worksheet — gradient descent med autograd, alle 100 punkter]

---

**slide**

hvorfor regne gradienten ud fra alle punkter for hvert skridt? dyrt når man har mange punkter (millioner, for rigtige datasæt)

løsning: brug et tilfældigt udpluk (en batch) i stedet — cirka samme gradient, meget billigere. kaldes minibatch eller stokastisk gradient descent (SGD)

**slide**

`random.sample(punkter, k)` er givet: tager en liste og et tal k, vælger tilfældigt k af dem, giver det som output

[animation fra worksheet — batch-punkterne fremhævet (orange) blandt alle punkterne i modelfit-panelet, resten dæmpet]

prøv batch_størrelse = 1, 5, 20, 50, 100 — beskriv med egne ord hvad der ændrer sig

Opgaver i at løse denne type opgaver:
- opg2_1
- opg2_2 (refleksion, ingen facit)

---

**slide**

kan vi komme hurtigere/mere direkte til minimum, uden nødvendigvis at bruge flere punkter pr. skridt? det er det bl.a. Adam — en af de mest populære optimizers — prøver at gøre

vi bygger 4 metoder, hver i to lag: selve opdateringen og løkken udenom, alle med samme form (loss, start, ...) — gradienten finder de selv med autograd (.backward())

constraints: kørslen stopper automatisk (og bruger seneste punkt) hvis loss kaldes mere end 100 gange, eller hvis man ender mere end 1000 væk fra landskabet (nok en fejl, ikke en strategi) — testes på 4 landskaber (linjefitting + 3 ukendte), hver fra 4 udvalgte startpunkter

**almindelig gradient descent (opg3)**

samme gradient+skridt fra recap, bare omskrevet til (loss, start)-formen

(a,b)_{t+1} = (a,b)_t - lr\cdot ∇L(a_t,b_t)

**slide**

Opgaver i at løse denne type opgaver:
- opg3_1
- opg3_2

ekstra:
- opg3_3 (opg3_2 klarer sig især dårligt på rosenbrock, hvor hældningerne er ekstremt store — brainstorm hvordan man undgår for store skridt, fx en anden lr, begræns størrelsen af skridtet, eller en helt anden idé)

**momentum (opg4)**

**ikke** en fysisk "bold der ruller ned ad bakken" — et vægtet, aftagende gennemsnit af alle gradienter man har set indtil videre

v_t = β\cdot v_{t-1} + (1-β)\cdot ∇L(a_t,b_t)
(a,b)_{t+1} = (a,b)_t - lr\cdot v_t

β (typisk ~0.9) er en hyperparameter der styrer hvor meget vi vægter gamle værdier vs. en ny observation — en gradient fra k skridt tilbage tæller med vægten (1-β)β^k, altså mindre og mindre jo længere tilbage den er

**slide**

Opgaver i at løse denne type opgaver:
- opg4_1
- opg4_2
- opg4_3

ekstra:
- opg4_4 (egen variant — tag inspiration fra momentum, find på noget helt andet, eller byg videre på opg3_3)

**slide**

[animation fra worksheet]

**rmsprop (opg5)**

skalér hver akse med sin egen (nyligt sete) gradient-størrelse, så der tages lige store skridt i alle retninger — uanset om aksen er stejl eller flad

s_t = β\cdot s_{t-1}+(1-β)\cdot ∇L(a_t,b_t)^2
(a,b)_{t+1} = (a,b)_t - lr\cdot ∇L(a_t,b_t)/(√s_t+ε)

(s regnes elementvis pr. parameter — a og b får hver deres egen skalering)

**slide**

Opgaver i at løse denne type opgaver:
- opg5_1
- opg5_2
- opg5_3

ekstra:
- opg5_4 (egen variant — tag inspiration fra momentum/rmsprop, find på noget helt andet, eller byg videre på en tidligere ekstra opgave)

**slide**

[animation fra worksheet]

**adam (opg6)**

kombinerer momentum (v) og RMSprop (s) — genbruger jeres opg4_1/opg5_1 — plus en bias-korrektion af begge, vigtig i de første skridt, hvor v_0=s_0=0 ellers trækker gennemsnittet kunstigt mod nul

v̂_t = v_t/(1-β_1^t),  ŝ_t = s_t/(1-β_2^t)
(a,b)_{t+1} = (a,b)_t - lr\cdot v̂_t/(√ŝ_t+ε)

**slide**

Opgaver i at løse denne type opgaver:
- opg6_1
- opg6_2

**slide**

[animation fra worksheet]

overblik: gd vs. momentum vs. rmsprop vs. adam, samlet i ét sammenligningsgrid

**slide**

ekstra:
- opg6_3 (byg jeres bedste metode — kombinér gerne idéer fra gd/momentum/rmsprop/adam, eller prøv noget nyt)
