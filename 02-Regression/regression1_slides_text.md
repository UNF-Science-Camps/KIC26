# Regression 1 — slide-tekst

Formål: slidene skal duplikere fortællingen fra notebooken (så eleverne ikke behøver læse md-cellerne selv), tilføje lidt ekstra stillads undervejs, og efter hver opgaveblok highlighte "det store billede" i fælles gennemgang. Struktur: præsentation → *arbejd på worksheet* → fælles gennemgang → gentag.

Tre slide-typer i denne fil:
- **Præsentation**: rigtigt indhold — det der reelt skal stå på skærmen.
- **Arbejd på worksheet**: ren transition-slide, ingen indhold — bare et fikspunkt der viser hvilke opgaver eleverne skal i gang med nu, og evt. hvor lang tid de har.
- **Fælles gennemgang**: også en let slide — selve gennemgangen er bare at scrolle igennem facit-notebooken live og køre visualiseringerne. Slidet er kun en overskrift/markør.

---

## 0. Åbning

**Slide 0.1 — Titel** *(allerede i template)*
Kunstig Intelligens Camp — Regression

**Slide 0.2 — Om mig**
navn: Sebastian Strøyer - SESY i unf regi
Dette er min første camp som arrengør/faglig,
Har været på 3 camps som deltager: CSC 2023, MLC 2024, MLC 2025

Studere Kognition og Datavidenskab på KU, så er nok stærkere end den gennemsnitlige arrengør ift hvordan hjernen fungere.
Jeg vil mene jeg er ret stærk inden for ml, men det meste er ikke kommet fra mit studie. Min intuition inden for ai kommer primært fra youtube, tidligere camps, og programmeringsprojekter jeg har lavet i min fritid.


**Slide 0.3 — Sådan kører blokken**
Format: 
jeg har prøvet at bygge et så godt og selvforklarende worksheet som muligt, så formattet bliver:
01. mini oplæg
02. arbejd med en del af worksheet + stil spørgsmål undervejs
03. fælles opsamling 
04. return to 01.

**Slide 0.4**
big picture:
1. vi får en intuition for hvad en matematisk ml model er
2. hvordan kvantificere hvor god en model er og får en visuel intuition
3. vi finder ud af hvordan vi skal modificere vægte i modellen for at forbedre den
4. og så finder vi ud af hvordan vi træner vores simple model (på en måde der generalisere op til kæmpe netværk)

---


**slide**

ai model = funktion = f_{vægte}(input)
eg. 
f_{a}(x) = a\cdot x
f_{a,b}(x) = a\cdot x + b
f_{a,b,c}(x) = a\cdot x^2 +b \cdot x + c


vi ved nok hvordan vi løser ligninger med ubekendte
vi kender x_n og y_n, og vil gerne isolere a og b

ax=y

ax_1+b=y_1
ax_2+b=y_2

...

**slide**

vi ved nok allerede hvordan vi fitter disse modeller til 1 eller 2 punkter:

[alle 3 modeller hvor parameterne bevæger sig, men der er fastsatte punkter som skal rammes]



**slide**

Opgaver i at løse denne type opgaver:
- opg1_1
- opg1_2
- opg2_1
- opg2_2

ekstra:
- opg1_3 # svær
- opg2_3


**flere punkter en antal parametre**

Nu har vi flere punkter end vi har parametre, modellere generalisere ikke:
[figur fra worksheet] 

**kvantificer fejl**

*mat fra worksheet*



**slide**

Opgaver i at løse denne type opgaver:
- opg3_1
- opg3_2
- opg4_1

ekstra:
- opg3_3
- opg3_4
- opg4_2
- opg4_3


**slide**
nu har vi i regnet loss, ud fra det kan vi definere en hel flade og se hvordan den ser ud

[figur fra worksheet]



**slide**
vi vil gerne finde det laveste punkt på denne flade, da det repræsentere de bedste a og b værdier.

vi skal nu bruge et nyt redskab der peger os i retningen af bunden: gradienten

først skal vi lære at finde hældningen af funktioner

**slide**
potensregl
sumregl
kæderegl

**slide**

Potensregl:
- opg5_1
- opg5_2
- opg5_3
- opg5_4

Potensregl + Sumregl:
- opg5_5
- opg5_6

Potensregl + Sumregl + Kæderegl:
- opg5_7
- opg5_8


**slide**

Nu har vi gjort det med funktioner der afhænger af en variabel.

men loss afhænger af flere parametre på samme tid (a og b, evt c)

partielt afledte: differentier med hensyn til én variabel ad gangen, behandl resten som konstanter

man kan forstille sig som at man fastlåser alle parametre undtagen 1, således at man får en traditionel funktion af 1 variabel.
Hældningen af denne funktion er lig den partielt afledte 

gradient = vektor af alle partielt afledte, peger i den retning funktionen vokser hurtigst
∇f = (∂f/∂x, ∂f/∂y)

**slide**

Opgaver i at løse denne type opgaver:
- opg6_1
- opg6_2
- opg6_3
- opg6_4
- opg6_5
- opg6_6

**slide**

lad os samle det vi har lært: model, fejl, SSE

model_{a,b}(x) = a\cdot x+b
fejl_{a,b}(x,y) = model_{a,b}(x) - y
SSE_{a,b}(punkter) = fejl_{a,b}(x_1,y_1)^2 + fejl_{a,b}(x_2,y_2)^2 + ...


Vi vil gerne beregne gradienten gradienten af vores tideligere flade.
ie vi skal beregne gradienten af SSE med respekt til a og b:
∇SSE_{a,b} = (∂SSE/∂a, ∂SSE/∂b)

**slide**

Opgaver i at løse denne type opgaver:
- opg7_1
- opg7_2
- opg7_3

ekstra:
- opg7_4
- opg7_5
- opg7_6
- opg7_7

**slide**

lad os se hvad gradienten faktisk betyder: en pil der peger i den retning loss vokser hurtigst

[figur fra worksheet]

**slide**

MSE = SSE/n, en konstant faktor går lige igennem gradienten:
∇MSE = ∇SSE / n

Opgaver i at løse denne type opgaver:
- opg8_1

ekstra:
- opg8_2
- opg8_3

**gradient descent**

gradienten peger opad, for at komme ned ad fladen går vi den modsatte vej

vi starter med vilkårlige parametre og tager gentagne små skridt imod gradienten:
1. opdater vægtene — træk skridtet fra
2. et helt skridt — gradient (opg8) * learning rate
3. træningsloop — gentag skridtet n gange

**slide**

Opgaver i at løse denne type opgaver:
- opg9_1
- opg9_2
- opg9_3

ekstra:
- opg9_4
- opg9_5
- opg9_6

**slide**

[animation fra worksheet]

leg med learning rate: 0.02, 0.05, 0.1, 0.115, 0.12 — hvad sker der når lr bliver for stor?

**bonus, kun hvis tid**

for simple modeller som vores findes der en "closed form solution", det gør der ikke for neurale netværk

closed-form-løsningen er netop der hvor gradienten er 0

[figur fra worksheet]
