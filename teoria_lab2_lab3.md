# Teoria do Laboratoriów 2 i 3 - Sieci Neuronowe
## Przewodnik z wyjaśnieniami

## Spis treści
1. [Laboratorium 2 - Regresja Logistyczna](#laboratorium-2---regresja-logistyczna)
2. [Laboratorium 3 - Wielowarstwowa Sieć Neuronowa](#laboratorium-3---wielowarstwowa-sieć-neuronowa)

---

## Laboratorium 2 - Regresja Logistyczna

### 1. Co to jest regresja logistyczna i po co nam to?

Wyobraź sobie, że masz dane pacjenta (wiek, ciśnienie, cholesterol itp.) i chcesz przewidzieć: czy ten człowiek jest chory na serce, czy nie? To jest **klasyczny problem klasyfikacji binarnej** - odpowiedź to TAK (1) lub NIE (0).

Regresja logistyczna to najprostszy model, który to robi. Działa jak prosta "bramka decyzyjna" - bierze wszystkie cechy pacjenta, łączy je razem i mówi: "Jest 73% szans, że ta osoba jest chora". 

**Dlaczego "logistyczna"?** Bo używa funkcji logistycznej (sigmoid), która zamienia dowolną liczbę na prawdopodobieństwo.

**Dlaczego zaczynamy od tego?** Bo to najprostszy "neuron" - podstawowy budulec większych sieci. Jeśli zrozumiesz to, zrozumiesz całą resztę!

### 2. Funkcja sigmoid - "przekształcacz na prawdopodobieństwa"

**Problem:** Model liczy nam jakąś wartość, powiedzmy -3.7 lub 12.4. Ale my chcemy prawdopodobieństwa (coś między 0 a 1)!

**Rozwiązanie:** Funkcja sigmoid! To matematyczna "zjeżdżalnia" która:
- Duże liczby dodatnie (np. 10) → przekształca na ~1 (prawie pewne)
- Duże liczby ujemne (np. -10) → przekształca na ~0 (prawie niemożliwe)  
- Zero → przekształca na dokładnie 0.5 (totalny rzut monetą)

**Wzór:**
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Intuicja wizualna:** Wyobraź sobie gładką literę "S" leżącą na boku. Gdy idziesz w prawo (większe z), zbliżasz się do 1. Gdy w lewo (mniejsze z), spadasz do 0.

**Pochodna (dlaczego jest ważna?):**
$$\sigma'(z) = \sigma(z) \cdot (1 - \sigma(z))$$

Ta pochodna jest SUPER wygodna w obliczeniach! Jeśli już obliczyłeś sigmoid(z), to jego pochodna to po prostu wynik pomnożony przez (1 - wynik). Eleganckie!

#### 🔬 Skąd się bierze ten wzór na pochodną?

**Wyprowadzenie:**
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Zapiszmy to jako: $\sigma(z) = (1 + e^{-z})^{-1}$

Używamy **reguły łańcuchowej**: jeśli $f(z) = [g(z)]^n$, to $f'(z) = n[g(z)]^{n-1} \cdot g'(z)$

$$\sigma'(z) = -1 \cdot (1 + e^{-z})^{-2} \cdot \frac{d}{dz}(1 + e^{-z})$$

Pochodna wykładniczej: $\frac{d}{dz}(e^{-z}) = -e^{-z}$

$$\sigma'(z) = -1 \cdot (1 + e^{-z})^{-2} \cdot (-e^{-z}) = \frac{e^{-z}}{(1 + e^{-z})^2}$$

Teraz sprytna sztuczka! Rozbijmy $(1 + e^{-z})^2$ w mianowniku:

$$\sigma'(z) = \frac{e^{-z}}{(1 + e^{-z})^2} = \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}}$$

Zauważ, że:
- $\frac{1}{1 + e^{-z}} = \sigma(z)$ 
- $\frac{e^{-z}}{1 + e^{-z}} = \frac{e^{-z} + 1 - 1}{1 + e^{-z}} = 1 - \frac{1}{1 + e^{-z}} = 1 - \sigma(z)$

Zatem:
$$\boxed{\sigma'(z) = \sigma(z) \cdot (1 - \sigma(z))}$$

**Piękno tego wzoru:** Nie musisz liczyć wykładniczych od nowa - używasz już obliczonej wartości sigmoid!

### 3. Jak działa model? (od środka)

Model to w zasadzie prosta formuła w dwóch krokach:

**Krok 1: Ważona suma**
Weź wszystkie cechy pacjenta (np. wiek=50, ciśnienie=140, cholesterol=200), przemnóż każdą przez "wagę" i dodaj wszystko razem:
$$z = w_1 \cdot wiek + w_2 \cdot ciśnienie + w_3 \cdot cholesterol + b$$

Można to zapisać krócej jako: $z = w^T x + b$

**Co to są wagi?** To "ważności" - jeśli waga jest duża, ta cecha ma duży wpływ. Ujemna waga = cecha obniża ryzyko.

**Co to jest bias (b)?** To "punkt startowy" - ogólne przesunięcie całego modelu.

**Krok 2: Przekształcenie na prawdopodobieństwo**
$$p(chory) = \sigma(z) = \frac{1}{1 + e^{-z}}$$

Teraz masz liczbę od 0 do 1 - gotowe prawdopodobieństwo!


### 4. Funkcja kosztu - jak mierzymy, czy model jest dobry?

**Problem:** Mamy model, który daje jakieś przewidywania. Ale jak sprawdzić, czy są dobre?

Potrzebujemy "karnej" funkcji, która powie: "Hej, źle przewidziałeś, dostajesz wysoką karę!" albo "Dobra robota, niska kara!".

#### Binary Cross-Entropy (BCE) - funkcja straty

To jest nasza "kara" za złe przewidywania. Działa sprytnie:

**Dla jednej próbki:**
$$L(y, \hat{y}) = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$$

**Jak to czytać?**
- Jeśli prawdziwa etykieta $y = 1$ (pacjent chory):
  - Liczy się tylko część: $-\log(\hat{y})$
  - Jeśli przewidziałeś $\hat{y} = 0.9$ (wysoka pewność) → kara mała ✓
  - Jeśli przewidziałeś $\hat{y} = 0.1$ (niska pewność) → kara DUŻA ✗
  
- Jeśli prawdziwa etykieta $y = 0$ (pacjent zdrowy):
  - Liczy się tylko część: $-\log(1-\hat{y})$
  - Jeśli przewidziałeś $\hat{y} = 0.1$ (pewność choroby niska) → kara mała ✓
  - Jeśli przewidziałeś $\hat{y} = 0.9$ (pewność choroby wysoka) → kara DUŻA ✗

**Dla całego zbioru danych:**
$$J(w, b) = \frac{1}{m} \sum_{i=1}^{m} L(y^{(i)}, \hat{y}^{(i)})$$

Po prostu uśredniamy kary ze wszystkich próbek!

**Sztuczka programistyczna:** W kodzie dodajemy mikroskopijne `epsilon` (np. $10^{-15}$) do logarytmu, żeby uniknąć `log(0)` = `-infinity` i nie zepsuć obliczeń.

#### 🔬 Skąd się bierze wzór na BCE?

**Intuicja z teorii informacji:**

Binary Cross-Entropy pochodzi z **teorii informacji** i mierzy "zaskoczenie" modelu.

**Dla jednej próbki:**

1. Jeśli prawdziwa etykieta to $y=1$ (pacjent chory):
   - Model przewiduje $\hat{y} = 0.99$ → "niskie zaskoczenie" = mała kara
   - Model przewiduje $\hat{y} = 0.01$ → "MEGA zaskoczenie!" = duża kara
   
2. "Zaskoczenie" mierzymy jako $-\log(\hat{y})$
   - $-\log(0.99) = 0.01$ (mała kara)
   - $-\log(0.01) = 4.6$ (duża kara)

**Pełny wzór dla obu przypadków:**

Chcemy jednej formuły, która działa zarówno dla $y=0$ jak i $y=1$:

- Gdy $y=1$: kara = $-\log(\hat{y})$
- Gdy $y=0$: kara = $-\log(1-\hat{y})$

Sprytna sztuczka matematyczna - połączmy to:
$$L(y, \hat{y}) = -[y \cdot \log(\hat{y}) + (1-y) \cdot \log(1-\hat{y})]$$

**Dlaczego to działa?**
- Gdy $y=1$: $(1-y)=0$, więc drugi człon znika → zostaje $-\log(\hat{y})$ ✓
- Gdy $y=0$: $y=0$, więc pierwszy człon znika → zostaje $-\log(1-\hat{y})$ ✓

**Dlaczego akurat logarytm?**
- Ma odpowiednie właściwości matematyczne (różniczkowalny, wypukły)
- Mocno karze pewne błędy (gdy model jest bardzo pewny, a się myli)
- Wynika z maksymalizacji prawdopodobieństwa (Maximum Likelihood Estimation)

#### 🔬 Związek z prawdopodobieństwem (dla ciekawskich)

Model daje prawdopodobieństwo: $P(y=1|x) = \hat{y}$

Dla pojedynczej próbki prawdopodobieństwo poprawnej predykcji to:
$$P(y|x) = \hat{y}^y \cdot (1-\hat{y})^{1-y}$$

(Gdy y=1, zostaje $\hat{y}$; gdy y=0, zostaje $1-\hat{y}$)

Chcemy **maksymalizować** to prawdopodobieństwo. Bierzemy logarytm (łatwiej matematycznie):
$$\log P(y|x) = y \log(\hat{y}) + (1-y) \log(1-\hat{y})$$

**Maksymalizacja** = **minimalizacja z minusem**, stąd:
$$L = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$$

To właśnie Binary Cross-Entropy!

### 5. Spadek gradientu - jak model się uczy?

**Wielka idea:** Wyobraź sobie, że stoisz w górach w gęstej mgle i chcesz zejść do doliny (tam gdzie funkcja kosztu jest najmniejsza). Nie widzisz, gdzie jest dolina, ale czujesz nachylenie terenu pod stopami. **Idziesz w dół!**

To właśnie robi gradient descent:
1. Sprawdza "nachylenie" funkcji kosztu (gradient)
2. Robi krok w kierunku, gdzie koszt maleje
3. Powtarza, aż znajdzie minimum (lub się zmęczy - max. liczba kroków)

**Algorytm krok po kroku:**

```
1. START: Ustaw losowe wagi w i bias b
2. PĘTLA (dla każdej epoki):
   a) Policz predykcje dla wszystkich danych
   b) Policz funkcję kosztu (jak bardzo się mylisz)
   c) Policz gradient (w którą stronę iść, żeby poprawić wynik)
   d) AKTUALIZUJ parametry:
      w = w - α × gradient_w
      b = b - α × gradient_b
3. STOP: gdy koszt przestanie maleć lub osiągniesz max. epok
```

**Co to jest α (learning rate)?**
To "długość kroku". 
- Za duże α → przeskakujesz minimum, model skacze jak szalony 🤪
- Za małe α → uczysz się meeeeeedlennie, ale bezpiecznie 🐌
- W samą porę α → szybko i dokładnie dochodzisz do celu! 🎯

### 6. Gradienty - matematyka, która napędza uczenie

Gradient to po prostu "nachylenie" funkcji - mówi, jak szybko rośnie funkcja w danym kierunku.

**Dla wag:**
$$\frac{\partial J}{\partial w} = \frac{1}{m} X^T (\hat{y} - y)$$

**Po ludzku:** 
- $(\hat{y} - y)$ to "błąd" - jak bardzo się pomyliłeś
- $X^T$ to cechy twojego pacjenta
- Mnożysz je razem i dostajesz: "w którą stronę i jak mocno zmienić wagi"

**Dla biasu:**
$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})$$

**Po ludzku:** To po prostu średni błąd ze wszystkich próbek.

**Magia:** Zauważ, że gradient zależy od błędu $(\hat{y} - y)$. Im większy błąd, tym większa korekta!

#### 🔬 Skąd się biorą te wzory na gradienty?

**Zacznijmy od funkcji kosztu:**
$$J(w, b) = \frac{1}{m} \sum_{i=1}^{m} L(y^{(i)}, \hat{y}^{(i)})$$

gdzie: $\hat{y} = \sigma(w^T x + b)$

**Gradient względem WAG (∂J/∂w):**

Użyjmy **reguły łańcuchowej**:
$$\frac{\partial J}{\partial w} = \frac{\partial J}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w}$$

gdzie $z = w^T x + b$

**Krok 1:** Pochodna BCE względem $\hat{y}$:
$$\frac{\partial L}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}$$

Upraszczamy do wspólnego mianownika:
$$\frac{\partial L}{\partial \hat{y}} = \frac{-y(1-\hat{y}) + (1-y)\hat{y}}{\hat{y}(1-\hat{y})} = \frac{\hat{y} - y}{\hat{y}(1-\hat{y})}$$

**Krok 2:** Pochodna sigmoid względem $z$:
$$\frac{\partial \hat{y}}{\partial z} = \sigma'(z) = \sigma(z)(1-\sigma(z)) = \hat{y}(1-\hat{y})$$

**Krok 3:** Pochodna $z$ względem $w$:
$$\frac{\partial z}{\partial w} = \frac{\partial (w^T x + b)}{\partial w} = x$$

**Łączymy wszystko:**
$$\frac{\partial L}{\partial w} = \frac{\hat{y} - y}{\hat{y}(1-\hat{y})} \cdot \hat{y}(1-\hat{y}) \cdot x = (\hat{y} - y) \cdot x$$

**MAGIA:** Środkowe człony się skracają! $\hat{y}(1-\hat{y})$ w liczniku i mianowniku znika!

**Dla wszystkich próbek (macierzowo):**
$$\frac{\partial J}{\partial w} = \frac{1}{m} X^T (\hat{y} - y)$$

**Gradient względem BIASU (∂J/∂b):**

Analogicznie:
$$\frac{\partial z}{\partial b} = 1$$

Więc:
$$\frac{\partial L}{\partial b} = (\hat{y} - y)$$

Dla wszystkich próbek:
$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})$$

**Elegancja tego wyniku:** Gradient to po prostu błąd razy wejście! Sigmoid + BCE dają super proste gradienty.

### 7. Normalizacja - dlaczego to jest MEGA ważne?

**Problem:** Masz cechy o różnych skalach:
- Wiek: 20-80 (zakres ~60)
- Cholesterol: 150-300 (zakres ~150)
- Jakiś wskaźnik: 0.001-0.01 (zakres ~0.009)

Model będzie miał trudności! Wagi dla cholesterolu będą mikroskopijne, a dla wskaźnika ogromne. Uczenie będzie wolne i niestabilne.

**Rozwiązanie: Standaryzacja**
$$x_{znormalizowane} = \frac{x - \text{średnia}}{\text{odchylenie standardowe}}$$

**Co to robi?**
Przekształca każdą cechę tak, że:
- Ma średnią = 0 (jest "wycentrowana")
- Ma odchylenie standardowe = 1 (ma "standardowy zakres")

Teraz wszystkie cechy są porównywalne!

**⚠️ SUPER WAŻNE:**
1. Oblicz średnią i odchylenie **TYLKO** na danych treningowych
2. Użyj TYCH SAMYCH wartości do normalizacji danych testowych
3. Dlaczego? Bo w życiu codziennym model będzie widział nowe dane - nie może "podglądać" ich statystyk z góry!


### 8. Metryki - jak ocenić jakość modelu?

Po wytrenowaniu modelu trzeba go sprawdzić. "Accuracy to za mało!" - usłyszysz często. Dlaczego?

**Przykład problemu:**
Masz 100 pacjentów, 95 zdrowych, 5 chorych. Model-idiota mówi zawsze "zdrowy".
- Accuracy = 95%! Wow! 🎉
- Ale... nie złapał ANI JEDNEGO chorego! 😱

Dlatego patrzymy na więcej metryk:

#### Accuracy (dokładność)
$$\text{Accuracy} = \frac{\text{ile dobrze przewidział}}{\text{ile było wszystkich}}$$

**Intuicja:** Ogólna "celność" modelu. Dobra jako pierwszy sprawdzian, ale nie mów całej prawdy.

#### Precision (precyzja)
$$\text{Precision} = \frac{TP}{TP + FP}$$

**Po ludzku:** "Spośród tych, których model oznaczył jako CHORYCH, ilu rzeczywiście jest chorych?"

- Wysoka precyzja = jak model mówi "chory", to raczej ma rację
- Niska precyzja = dużo fałszywych alarmów

**Kiedy ważna?** Gdy fałszywy alarm jest kosztowny (np. niepotrzebna operacja).

#### Recall / Czułość
$$\text{Recall} = \frac{TP}{TP + FN}$$

**Po ludzku:** "Spośród wszystkich RZECZYWIŚCIE CHORYCH, ilu model złapał?"

- Wysoki recall = model łapie większość chorych
- Niski recall = model przegapia chorych ludzi

**Kiedy ważna?** Gdy przeoczenie chorego jest niebezpieczne (np. wykrywanie raka).

#### F1-score (kompromis)
$$\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Po ludzku:** Średnia harmoniczna precision i recall. Łączy obie metryki w jedną liczbę.

Wysoki F1 = model jest jednocześnie dokładny I łapie większość przypadków. To dobra równowaga!

**Przypomnienie:**
- TP (True Positive) = poprawnie przewidziany chory
- FP (False Positive) = pomyłka - zdrowy oznaczony jako chory
- FN (False Negative) = pomyłka - chory oznaczony jako zdrowy
- TN (True Negative) = poprawnie przewidziany zdrowy

### 9. Próg decyzyjny - kiedy mówić TAK?

Model daje ci prawdopodobieństwo, np. 0.73. Ale potrzebujesz decyzji: chory czy zdrowy?

**Standardowo:** Próg = 0.5
- p ≥ 0.5 → CHORY (klasa 1)
- p < 0.5 → ZDROWY (klasa 0)

**Ale możesz to zmieniać!**
- Próg = 0.3 → Model chętniej mówi "chory" (więcej fałszywych alarmów, ale mniej przeoczonych chorych)
- Próg = 0.7 → Model ostrożniejszy (mniej fałszywych alarmów, ale więcej przeoczonych chorych)

**Jak wybrać próg?** Zależy od problemu:
- Wykrywanie raka? Lepiej fałszywy alarm niż przeoczona choroba → niższy próg
- Spam w mailu? Lepiej czasem przepuścić spam niż usunąć ważny mail → wyższy próg

---

## Laboratorium 3 - Wielowarstwowa Sieć Neuronowa

### 1. Czym różni się sieć wielowarstwowa od regresji logistycznej?

**Regresja logistyczna (Lab 2):** Prosty "neuron"
```
wejście → [wagi i bias] → sigmoid → wyjście
```
**Umie:** Tylko proste, liniowe decyzje (np. prosta linia dzieląca zdrowych od chorych)

**Sieć wielowarstwowa (MLP - Lab 3):** Stos neuronów ułożonych w warstwy!
```
wejście → [warstwa 1] → [warstwa 2] → [warstwa 3] → wyjście
```
**Umie:** Złożone, nieliniowe wzorce (może rysować zakrzywione granice, wykrywać skomplikowane zależności)

**Analogia:** 
- Regresja logistyczna = kalkulator
- MLP = komputer

### 2. Anatomia sieci wielowarstwowej

**Warstwy:**
1. **Warstwa wejściowa** - po prostu twoje dane (np. wiek, ciśnienie, cholesterol)
2. **Warstwy ukryte** - tu dzieje się magia! To "myślące" warstwy, które wykrywają wzorce
3. **Warstwa wyjściowa** - finalna predykcja

**Co dzieje się w każdej warstwie?**
```
[poprzednia warstwa] 
    ↓
1. Mnożenie przez wagi + dodanie biasu
    ↓
2. Funkcja aktywacji (nieliniowość!)
    ↓
[aktualna warstwa]
```

**Dlaczego wiele warstw = większa moc?**
- Pierwsza warstwa ukryta: wykrywa proste wzorce ("tutaj jest krawędź", "tu coś się zmienia")
- Druga warstwa: łączy proste wzorce w bardziej złożone ("to wygląda jak nos", "to przypomina krzywą")
- Trzecia warstwa: jeszcze bardziej abstrakcyjne koncepty
- Wyjście: finalna decyzja oparta na wszystkich wykrytych wzorcach

### 3. Matematyka pojedynczej warstwy (łatwa!)

**Krok 1: Transformacja liniowa**
$$z = xW + b$$

- $x$ - to co wchodzi (np. [wiek, ciśnienie, cholesterol])
- $W$ - wagi (macierz "połączeń" między wejściami a neuronami)
- $b$ - biasy (osobne "przesunięcie" dla każdego neuronu)
- $z$ - wynik przed aktywacją

**Krok 2: Funkcja aktywacji**
$$a = f(z)$$

Ta funkcja wprowadza "nieliniowość" - bez niej cała sieć zachowywałaby się jak jedna duża regresja logistyczna!

### 4. Funkcje aktywacji - kluczowe składniki

#### 4.1 Sigmoid (znasz już z Lab 2!)
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Kiedy używać?** 
- Warstwa wyjściowa w klasyfikacji binarnej (daje prawdopodobieństwo 0-1)

**Problem w warstwach ukrytych:**
Dla bardzo dużych lub małych z, pochodna jest bliska 0. Gradient "zanika" - sieć uczy się mega wolno lub wcale!

#### 4.2 ReLU (Rectified Linear Unit) - gwiazda głębokich sieci!
$$\text{ReLU}(z) = \max(0, z) = \begin{cases} z & \text{jeśli } z > 0 \\ 0 & \text{jeśli } z \leq 0 \end{cases}$$

**Intuicja:** "Przepuszczasz tylko wartości dodatnie, resztę zeruj"

**Pochodna:**
$$\text{ReLU}'(z) = \begin{cases} 1 & \text{jeśli } z > 0 \\ 0 & \text{w przeciwnym razie} \end{cases}$$

#### 🔬 Skąd się bierze wzór na pochodną ReLU?

ReLU to najprostrza funkcja w historii głębokich sieci!

$$\text{ReLU}(z) = \max(0, z) = \begin{cases} z & \text{jeśli } z > 0 \\ 0 & \text{jeśli } z \leq 0 \end{cases}$$

**Pochodna to po prostu nachylenie:**

**Dla z > 0:** Funkcja to po prostu $f(z) = z$, więc $f'(z) = 1$
**Dla z < 0:** Funkcja to stała $f(z) = 0$, więc $f'(z) = 0$
**Dla z = 0:** Technicznie pochodna nie istnieje (załamanie), ale w praktyce przyjmujemy 0 lub 1 (zazwyczaj 0)

$$\boxed{\text{ReLU}'(z) = \begin{cases} 1 & \text{jeśli } z > 0 \\ 0 & \text{w przeciwnym razie} \end{cases}}$$

**Dlaczego to jest genialne?**
- Obliczeniowo darmowe - tylko porównanie z zerem!
- Gradient = 1 dla aktywnych neuronów (nie zanika!)
- Gradient = 0 dla nieaktywnych (neuron "wyłączony")

To jest najprostsza możliwa nieliniowość, która działa!

**Dlaczego jest super?**
✅ Prosta jak budowa cepa
✅ Szybka do obliczenia
✅ Nie ma problemu zanikającego gradientu (dla z > 0)
✅ Empirycznie działa świetnie w głębokich sieciach

**Jeden problem: "Dying ReLU"**
Jeśli neuron wpadnie w rejon z < 0 i tam zostanie, jego pochodna = 0. "Umiera" - przestaje się uczyć.

**Rozwiązanie:** Leaky ReLU, ale to już dla zaawansowanych 😎

**Kiedy używać?**
- Warstwy ukryte - praktycznie zawsze!
- Chyba że masz dobry powód, żeby użyć czegoś innego


### 5. Forward Propagation - "przepychanie" danych przez sieć

**To jest prosta część!** Bierzesz dane i przepuszczasz je przez kolejne warstwy. Każda warstwa robi swoje: mnoży, dodaje, aktywuje.

**Przykład z 3 warstwami:**

```
Dane wejściowe: [wiek, ciśnienie, cholesterol, ...]
    ↓
WARSTWA 1 (ukryta, 32 neurony, ReLU):
    z¹ = x·W¹ + b¹
    a¹ = ReLU(z¹)
    ↓
WARSTWA 2 (ukryta, 16 neuronów, ReLU):
    z² = a¹·W² + b²
    a² = ReLU(z²)
    ↓
WARSTWA 3 (wyjście, 1 neuron, Sigmoid):
    z³ = a²·W³ + b³
    ŷ = Sigmoid(z³)
    ↓
Wynik: 0.73 (73% prawdopodobieństwa choroby)
```

**Klucz:** Wyjście z jednej warstwy (a) staje się wejściem do następnej!

**W kodzie:**
```python
def forward(X):
    out = X
    for warstwa in warstwy:
        out = warstwa.forward(out)
    return out  # finalna predykcja
```

### 6. Backpropagation - MAGIA uczenia sieci!

**To jest trudniejsza część, ale najważniejsza!**

OK, mamy sieć. Robi predykcje. Ale **jak ją nauczyć?** Musimy zaktualizować wagi we WSZYSTKICH warstwach. Problem: jak gradient z wyjścia dotrzeć do pierwszych warstw?

**Odpowiedź: Backpropagation** = propagacja wstecz = puszczanie gradientu od końca do początku sieci.

#### Intuicja: gra w "głuchy telefon" z gradientami

Wyobraź sobie:
1. Na końcu (wyjście) obliczamy: "o ile się pomyliłem?"
2. Pytamy ostatnią warstwę: "o ile TY powinna zmienić swoje wagi?"
3. Ta warstwa mówi poprzedniej: "hej, twoja wina była taka-a-taka"
4. I tak dalej, aż dotrzemy do początku

Każda warstwa:
- Dostaje "winę" z kolejnej warstwy (gradient)
- Oblicza, jak zmienić swoje wagi
- Przekazuje część "winy" do warstwy przed sobą

#### Matematyka (uproszczona)

**Dla warstwy $l$:**

**Krok 1:** Masz gradient względem aktywacji: $\frac{\partial L}{\partial a^{[l]}}$ 
(to jest "wina" przekazana przez następną warstwę)

**Krok 2:** Oblicz gradient względem z (przed aktywacją):
$$\frac{\partial L}{\partial z^{[l]}} = \frac{\partial L}{\partial a^{[l]}} \odot f'(z^{[l]})$$

$\odot$ = mnożenie element po elemencie (każdy z każdym z osobna)

**Dlaczego pochodna aktywacji?** Bo to "bramka" - kontroluje, jak mocno sygnał przeszedł. Jeśli $f'(z)$ jest mała, gradient słabnie (problem zanikającego gradientu!).

**Krok 3:** Oblicz, jak zmienić WAGI tej warstwy:
$$\frac{\partial L}{\partial W^{[l]}} = \frac{1}{m} (a^{[l-1]})^T \cdot \frac{\partial L}{\partial z^{[l]}}$$

**Intuicja:** "Które wagi były najbardziej odpowiedzialne za błąd?"
- To zależy od aktywacji wejściowej ($a^{[l-1]}$) i błędu ($\frac{\partial L}{\partial z^{[l]}}$)
- Duże wejście × duży błąd = ta waga potrzebuje dużej korekty!

**Krok 4:** Oblicz gradient dla biasów:
$$\frac{\partial L}{\partial b^{[l]}} = \frac{1}{m} \sum \frac{\partial L}{\partial z^{[l]}}$$

Po prostu średni błąd z wszystkich próbek.

**Krok 5:** Przekaż "winę" do poprzedniej warstwy:
$$\frac{\partial L}{\partial a^{[l-1]}} = \frac{\partial L}{\partial z^{[l]}} \cdot (W^{[l]})^T$$

To będzie gradient dla warstwy $l-1$. I tak w kółko, aż dojdziesz do początku!

#### 🔬 Skąd się biorą wzory w backpropagation?

To wygląda na czarną magię, ale to tylko **reguła łańcuchowa** zastosowana wielokrotnie!

**Przypomnijmy regułę łańcuchową:**
Jeśli $y = f(g(x))$, to: $\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}$

**W naszej sieci:**
```
x → [Warstwa 1] → a¹ → [Warstwa 2] → a² → ... → ŷ → [Loss] → L
```

Chcemy: $\frac{\partial L}{\partial W^{[l]}}$ (jak zmiana wag wpływa na loss)

**Wyprowadzenie dla warstwy $l$:**

Oznaczenia:
- $z^{[l]} = a^{[l-1]} W^{[l]} + b^{[l]}$ (przed aktywacją)
- $a^{[l]} = f(z^{[l]})$ (po aktywacji)

**1) Gradient względem z (pre-activation):**

Wiemy, że aktywacja działa na każdy element z osobna:
$$a^{[l]}_i = f(z^{[l]}_i)$$

Reguła łańcuchowa:
$$\frac{\partial L}{\partial z^{[l]}} = \frac{\partial L}{\partial a^{[l]}} \cdot \frac{\partial a^{[l]}}{\partial z^{[l]}}$$

To $\frac{\partial a^{[l]}}{\partial z^{[l]}}$ to po prostu $f'(z^{[l]})$ - pochodna funkcji aktywacji!

$$\boxed{\frac{\partial L}{\partial z^{[l]}} = \frac{\partial L}{\partial a^{[l]}} \odot f'(z^{[l]})}$$

($\odot$ = mnożenie element-wise, bo każdy element z ma swoją pochodną)

**2) Gradient względem WAG:**

$$z^{[l]} = a^{[l-1]} W^{[l]} + b^{[l]}$$

To mnożenie macierzy! Dla pojedynczego elementu:
$$z^{[l]}_{ij} = \sum_k a^{[l-1]}_{ik} W^{[l]}_{kj}$$

Pochodna względem wagi $W_{kj}$:
$$\frac{\partial z^{[l]}_{ij}}{\partial W^{[l]}_{kj}} = a^{[l-1]}_{ik}$$

W notacji macierzowej (używając właściwości iloczynu macierzy):
$$\boxed{\frac{\partial L}{\partial W^{[l]}} = \frac{1}{m} (a^{[l-1]})^T \frac{\partial L}{\partial z^{[l]}}}$$

**Intuicja:** 
- $(a^{[l-1]})^T$ - co weszło do warstwy
- $\frac{\partial L}{\partial z^{[l]}}$ - jak bardzo się pomyliła
- Mnożenie: które połączenia (wagi) były odpowiedzialne za błąd

**3) Gradient względem BIASU:**

$$\frac{\partial z^{[l]}}{\partial b^{[l]}} = 1$$

Bo bias dodajemy bezpośrednio. Więc:
$$\boxed{\frac{\partial L}{\partial b^{[l]}} = \frac{1}{m} \sum \frac{\partial L}{\partial z^{[l]}}}$$

Po prostu sumujemy gradienty po wszystkich przykładach (bo ten sam bias jest używany wszędzie).

**4) Gradient przekazywany wstecz:**

Potrzebujemy $\frac{\partial L}{\partial a^{[l-1]}}$ żeby móc policzyć gradienty dla poprzedniej warstwy.

Z równania: $z^{[l]} = a^{[l-1]} W^{[l]} + b^{[l]}$

Reguła łańcuchowa:
$$\frac{\partial L}{\partial a^{[l-1]}} = \frac{\partial L}{\partial z^{[l]}} \cdot \frac{\partial z^{[l]}}{\partial a^{[l-1]}}$$

Pochodna względem $a^{[l-1]}$:
$$\frac{\partial z^{[l]}}{\partial a^{[l-1]}} = W^{[l]}$$

W notacji macierzowej:
$$\boxed{\frac{\partial L}{\partial a^{[l-1]}} = \frac{\partial L}{\partial z^{[l]}} (W^{[l]})^T}$$

**Intuicja:** "Wina" rozprzestrzenia się wstecz przez te same połączenia (wagi), którymi szedł sygnał do przodu!

#### 🔬 Dlaczego dla Sigmoid + BCE wychodzi tak ładnie?

**Twierdzenie:** Dla ostatniej warstwy z sigmoid i BCE:
$$\frac{\partial L}{\partial z^{[last]}} = \hat{y} - y$$

**Dowód:**

Funkcja kosztu: $L = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$

gdzie $\hat{y} = \sigma(z)$

**Krok 1:** Pochodna L względem $\hat{y}$:
$$\frac{\partial L}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}} = \frac{\hat{y} - y}{\hat{y}(1-\hat{y})}$$

**Krok 2:** Pochodna sigmoid:
$$\frac{\partial \hat{y}}{\partial z} = \sigma(z)(1-\sigma(z)) = \hat{y}(1-\hat{y})$$

**Krok 3:** Łączymy (reguła łańcuchowa):
$$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} = \frac{\hat{y} - y}{\hat{y}(1-\hat{y})} \cdot \hat{y}(1-\hat{y})$$

**MAGIA:** $\hat{y}(1-\hat{y})$ się skraca!

$$\boxed{\frac{\partial L}{\partial z} = \hat{y} - y}$$

**To nie przypadek!** Sigmoid i BCE zostały "stworzone dla siebie". Kombinacja daje najprostszy możliwy gradient.

To samo dzieje się dla:
- Softmax + Categorical Cross-Entropy (dla wielu klas)
- MSE + Identity (dla regresji)

#### Specjalny przypadek: ostatnia warstwa z BCE i Sigmoid

Normalna jest tam straszna matematyka z reguły łańcuchowej. ALE! Jest piękne uproszczenie:

$$\frac{\partial L}{\partial z^{[last]}} = \hat{y} - y$$

**TAK! Po prostu błąd predykcji!** 

To wynika z magicznych właściwości sigmoid + BCE. Pochodne się upraszczają i zostaje czysta różnica.

### 7. Aktualizacja parametrów

Gdy masz już wszystkie gradienty (z backprop), aktualizujesz wagi:

$$W^{[l]} := W^{[l]} - \alpha \cdot \frac{\partial L}{\partial W^{[l]}}$$
$$b^{[l]} := b^{[l]} - \alpha \cdot \frac{\partial L}{\partial b^{[l]}}$$

Znak minus, bo idziesz PRZECIWNIE do gradientu (w dół, nie w górę!)

### 8. Pełny cykl treningu - wszystko razem

```
SETUP: Zainicjuj losowe wagi dla wszystkich warstw

PĘTLA TRENINGOWA (epochs):
    
    1. FORWARD PASS:
       - Przepuść dane przez sieć (warstwa po warstwie)
       - Zapisz wszystkie pośrednie wartości (z, a)
       - Oblicz predykcję ŷ
       - Oblicz funkcję kosztu L(y, ŷ)
    
    2. BACKWARD PASS:
       - Zacznij od końca: policz gradient wyjścia
       - Idź wstecz przez warstwy:
         * Policz gradienty dla wag i biasów
         * Przekaż gradient do poprzedniej warstwy
    
    3. UPDATE:
       - Zaktualizuj wszystkie wagi: W -= α × gradient
       - Zaktualizuj wszystkie biasy: b -= α × gradient
    
    4. SPRAWDŹ:
       - Czy koszt maleje? ✓ Dobrze!
       - Czy przestał maleć? → STOP, już nauczony

KONIEC: Masz wytrenowaną sieć!
```

### 9. Inicjalizacja wag - zacznij dobrze!

**Nie możesz zainicjować wszystkiego zerami!** Bo wtedy wszystkie neurony w warstwie będą robić to samo i uczą się jednakowo. Symetria = marnowanie neuronów!

**Losowe małe wartości:**
- Najprościej: `W = np.random.randn(n_in, n_out) * 0.01`

**He initialization (dla ReLU):**
$$W \sim \mathcal{N}(0, \sqrt{\frac{2}{n_{in}}})$$

Bierze pod uwagę liczbę wejść - im więcej, tym mniejsze wagi (żeby nie wybuchnęły).

**Xavier/Glorot (dla Sigmoid/Tanh):**
$$W \sim \mathcal{N}(0, \sqrt{\frac{1}{n_{in}}})$$

**Biasy:**
Zazwyczaj zera są OK. Czasem małe wartości dodatnie dla ReLU.

**Dlaczego to ważne?**
- Za duże wagi → aktywacje eksplodują, gradient eksploduje, uczenie się wypalam
- Za małe wagi → aktywacje umierają, gradient zanika, sieć nie uczy się nic
- W sam raz → uczenie przebiega gładko 🎯

#### 🔬 Skąd się biorą te konkretne wartości inicjalizacji?

**Problem do rozwiązania:** Chcemy, żeby wariancja aktywacji była podobna w każdej warstwie. Bez tego:
- Głębokie warstwy mogą mieć gigantyczne wartości (exploding)
- Lub mikroskopijne wartości bliskie zeru (vanishing)

**He Initialization (dla ReLU):**

Załóżmy, że:
- Wejście $x$ ma wariancję $Var(x) = 1$ (po normalizacji)
- Mamy $n_{in}$ wejść do neuronu
- Wagi: $w_i \sim \mathcal{N}(0, \sigma_w^2)$

Wyjście neuronu (przed aktywacją):
$$z = w_1 x_1 + w_2 x_2 + ... + w_{n_{in}} x_{n_{in}}$$

**Wariancja sumy niezależnych zmiennych losowych:**
$$Var(z) = Var(w_1 x_1) + Var(w_2 x_2) + ... + Var(w_{n_{in}} x_{n_{in}})$$

Dla każdego składnika (zakładając niezależność):
$$Var(w_i x_i) = E[w_i]^2 Var(x_i) + E[x_i]^2 Var(w_i) + Var(w_i)Var(x_i)$$

Ale $E[w_i] = 0$ i $E[x_i] = 0$ (bo wycentrowane), więc:
$$Var(w_i x_i) = Var(w_i) \cdot Var(x_i) = \sigma_w^2 \cdot 1 = \sigma_w^2$$

Zatem:
$$Var(z) = n_{in} \cdot \sigma_w^2$$

**Chcemy:** $Var(z) \approx 1$ (żeby nie eksplodowało ani nie zanikło)

Więc:
$$n_{in} \cdot \sigma_w^2 = 1$$
$$\sigma_w^2 = \frac{1}{n_{in}}$$
$$\sigma_w = \sqrt{\frac{1}{n_{in}}}$$

**Ale to dla liniowej aktywacji!** ReLU zeruje połowę neuronów (z < 0), więc:
- Efektywna liczba aktywnych wejść to $\frac{n_{in}}{2}$
- Żeby skompensować, mnożymy wariancję przez 2

$$\boxed{\sigma_w = \sqrt{\frac{2}{n_{in}}}}$$

To jest **He initialization**!

**Xavier/Glorot Initialization (dla Sigmoid/Tanh):**

Dla sigmoid/tanh bierzemy pod uwagę zarówno forward jak i backward pass:
$$\sigma_w = \sqrt{\frac{1}{n_{in}}}$$

Lub uśrednioną wersję:
$$\sigma_w = \sqrt{\frac{2}{n_{in} + n_{out}}}$$

**W praktyce:** Używaj He dla ReLU, Xavier dla sigmoid/tanh!

### 10. Hiperparametry - pokrętła do kręcenia

**Learning rate (α):**
- Za mały (np. 0.0001): Uczenie meeeedlennie, ale stabilnie
- Za duży (np. 10): Chaos! Sieć nie może się ustabilizować
- W sam raz (np. 0.01, 0.001): Szybko i sprawnie
- **Trick:** Zacznij od większego, potem zmniejszaj (learning rate decay)

**Architektura (warstwy i neurony):**
- Więcej warstw = głębsza sieć = więcej pojemności = może uczyć się bardziej złożonych rzeczy
- ALE: Trudniejsza w trenowaniu, łatwiej przeuczyć
- **Reguła kciuka:** Zacznij od prostej (2-3 warstwy ukryte), zwiększaj jeśli potrzeba

**Liczba neuronów:**
- Więcej neuronów = większa pojemność warstwy
- ALE: Więcej obliczeń, ryzyko przeuczenia
- **Reguła kciuka:** Zaczynaj od coś między liczbą cech wejściowych a wyjściowych

**Liczba epok:**
- Za mało: Nie nauczy się dobrze (underfitting)
- Za dużo: Przepamięta dane treningowe (overfitting)
- **Reguła:** Trenuj, aż loss przestanie maleć na zbiorze walidacyjnym (early stopping)

**Funkcje aktywacji:**
- Warstwy ukryte: ReLU (lub jego warianty)
- Wyjście binarne: Sigmoid
- Wyjście multi-class: Softmax
- Regresja: Brak aktywacji (liniowe wyjście)

### 11. Problem zanikającego gradientu - dlaczego sigmoid w warstwach ukrytych to zły pomysł

**Co się dzieje?**

W backpropagation mnożysz gradienty przez pochodne funkcji aktywacji.

Dla sigmoid: $\sigma'(z) = \sigma(z)(1-\sigma(z))$
- Maksimum w z=0: $\sigma'(0) = 0.25$
- Dla dużych |z|: $\sigma'(z) \approx 0$

**Problem:**
```
Warstwa 5: gradient × 0.1
Warstwa 4: gradient × 0.1 × 0.2 = gradient × 0.02
Warstwa 3: gradient × 0.02 × 0.15 = gradient × 0.003
Warstwa 2: gradient × 0.003 × 0.1 = gradient × 0.0003
Warstwa 1: gradient × 0.0003 × 0.2 = gradient × 0.00006
```

Gradient ZANIKA! Pierwsze warstwy uczą się mega wolno lub wcale.

**Rozwiązanie: ReLU!**
- Dla z > 0: pochodna = 1 (nie zmniejsza gradientu!)
- Gradient nie zanika tak łatwo
- Dlatego ReLU = standard w głębokich sieciach

#### 🔬 Matematyczne wyjaśnienie zanikającego gradientu

**Przypomnijmy backpropagation:**

Dla warstwy $l$:
$$\frac{\partial L}{\partial z^{[l]}} = \frac{\partial L}{\partial a^{[l]}} \odot f'(z^{[l]})$$

Propagacja do poprzedniej warstwy:
$$\frac{\partial L}{\partial a^{[l-1]}} = \frac{\partial L}{\partial z^{[l]}} (W^{[l]})^T$$

**W głębokiej sieci (L warstw):**

Gradient dla pierwszej warstwy to iloczyn pochodnych ze WSZYSTKICH warstw:
$$\frac{\partial L}{\partial W^{[1]}} \propto f'^{[L]}(z^{[L]}) \cdot W^{[L]} \cdot f'^{[L-1]}(z^{[L-1]}) \cdot W^{[L-1]} \cdot ... \cdot f'^{[2]}(z^{[2]}) \cdot W^{[2]}$$

**Problem z sigmoid:**

Maksymalna wartość pochodnej sigmoid: $\sigma'(z) = \sigma(z)(1-\sigma(z)) \leq 0.25$

Dla 5 warstw:
$$\text{gradient} \propto 0.25 \times W \times 0.25 \times W \times 0.25 \times W \times ...$$

Nawet jeśli wagi są OK (bliskie 1), mnożysz przez $0.25^5 = 0.00098$!

**Gradient maleje wykładniczo z głębokością sieci!**

**Dlaczego ReLU rozwiązuje problem:**

Dla $z > 0$: $\text{ReLU}'(z) = 1$

$$\text{gradient} \propto 1 \times W \times 1 \times W \times 1 \times W \times ...$$

Gradient NIE zanika automatycznie! (Oczywiście nadal może zanikać przez wagi, ale nie przez aktywację)

**Dodatkowe sposoby walki:**
- Batch Normalization (normalizuje aktywacje między warstwami)
- Residual connections / Skip connections (gradient może "ominąć" warstwy)
- Gradient clipping (obcina za duże gradienty)
- Lepsze inicjalizacje wag (He/Xavier)

### 12. Porównanie: Regresja logistyczna vs MLP

| **Co?** | **Regresja logistyczna** | **MLP (sieć wielowarstwowa)** |
|---------|-------------------------|-------------------------------|
| **Struktura** | Pojedynczy "neuron" | Wiele warstw neuronów |
| **Złożoność** | Prosta, liniowa decyzja | Może uczyć się nieliniowych wzorców |
| **Granica decyzyjna** | Prosta linia (lub płaszczyzna) | Zakrzywione, złożone kształty |
| **Parametry** | Niewiele (n+1) | Dużo (zależy od architektury) |
| **Szybkość treningu** | Szybka | Wolniejsza |
| **Ryzyko overfittingu** | Małe | Większe (potrzeba regularyzacji) |
| **Kiedy używać?** | Problemy prostsze, liniowo-separowalne | Problemy złożone, nieliniowe |
| **Łatwość interpretacji** | Łatwa (wagi = ważność cech) | Trudna (czarna skrzynka) |

**Analogia:**
- Regresja logistyczna = prosta: "Jeśli cholesterol > 200 i wiek > 50, to chory"
- MLP = zawiłe reguły: "Jeśli ((cholesterol wysoki I młody) LUB (cholesterol średni I stary I ciśnienie...)) TO..."

---

## Podsumowanie - klucze do zrozumienia

### Lab 2 - Regresja logistyczna
🔑 **Kluczowa idea:** Pojedynczy neuron, który uczy się prostej granicy decyzyjnej
🔧 **Narzędzia:** Sigmoid (przekształca na prawdopodobieństwo), BCE (funkcja kosztu), gradient descent (uczenie)
💡 **Zastosowanie:** Proste problemy klasyfikacji binarnej

### Lab 3 - MLP
🔑 **Kluczowa idea:** Stos warstw, które uczą się coraz bardziej abstrakcyjnych wzorców
🔧 **Narzędzia:** ReLU (aktywacja dla ukrytych), backpropagation (uczenie całej sieci), forward/backward pass
💡 **Zastosowanie:** Złożone problemy, nieliniowe zależności

### Najważniejsze "aha!" momenty

1. **Funkcje aktywacji wprowadzają nieliniowość** - bez nich cała sieć = jedna wielka regresja liniowa!

2. **Backpropagation to "głuchy telefon" z gradientami** - każda warstwa przekazuje "winę" do poprzedniej

3. **Learning rate kontroluje tempo uczenia** - za mały = wolno, za duży = chaos

4. **Normalizacja danych jest KRYTYCZNA** - bez niej uczenie będzie wolne i niestabilne

5. **ReLU > Sigmoid w warstwach ukrytych** - prostszy, szybszy, bez zanikającego gradientu

6. **Więcej warstw ≠ zawsze lepiej** - potrzebujesz więcej danych i czasu na trening

7. **Metryki poza accuracy są ważne** - zwłaszcza dla niezbalansowanych klas

Powodzenia w laboratoriach! 🚀

---

## 📚 BONUS: Cheat sheet - "Skąd się biorą wzory?"

### Kluczowe wyprowadzenia w pigułce

#### 1️⃣ Pochodna Sigmoid: $\sigma'(z) = \sigma(z)(1-\sigma(z))$
**Metoda:** Reguła łańcuchowa na $\frac{1}{1+e^{-z}}$
**Klucz:** Rozbij na $(1+e^{-z})^{-1}$ i różniczkuj złożenie

#### 2️⃣ Binary Cross-Entropy: $L = -[y\log(\hat{y}) + (1-y)\log(1-\hat{y})]$
**Źródło:** Maksymalizacja prawdopodobieństwa (Maximum Likelihood)
**Klucz:** $P(y|x) = \hat{y}^y(1-\hat{y})^{1-y}$ → weź $-\log$

#### 3️⃣ Gradient dla regresji logistycznej: $\frac{\partial J}{\partial w} = \frac{1}{m}X^T(\hat{y}-y)$
**Metoda:** Reguła łańcuchowa: $\frac{\partial J}{\partial w} = \frac{\partial J}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w}$
**Klucz:** Pochodne sigmoid i BCE się skracają, zostaje błąd × wejście!

#### 4️⃣ Backpropagation: $\frac{\partial L}{\partial W^{[l]}} = (a^{[l-1]})^T \frac{\partial L}{\partial z^{[l]}}$
**Metoda:** Reguła łańcuchowa + algebra macierzy
**Klucz:** Gradient = "co weszło" × "jak się pomyliło"

#### 5️⃣ Gradient dla ostatniej warstwy: $\frac{\partial L}{\partial z^{[last]}} = \hat{y} - y$
**Metoda:** Sigmoid + BCE, pochodne się upraszczają
**Klucz:** $\hat{y}(1-\hat{y})$ z sigmoid skraca się z BCE!

#### 6️⃣ He initialization: $W \sim \mathcal{N}(0, \sqrt{2/n_{in}})$
**Źródło:** Analiza wariancji aktywacji
**Klucz:** Chcemy $Var(z) \approx 1$, ReLU zeruje połowę → mnożnik 2

#### 7️⃣ Pochodna ReLU: $\text{ReLU}'(z) = \mathbb{1}_{z>0}$
**Metoda:** Nachylenie prostych odcinków
**Klucz:** Dla $z>0$ nachylenie=1, dla $z \leq 0$ nachylenie=0

### Uniwersalna strategia wyprowadzania

**Dla każdego wzoru:**

1. **Zidentyfikuj złożenie funkcji** 
   - Co zależy od czego? $L(y, \hat{y}(\sigma(z(W, x))))$

2. **Reguła łańcuchowa!**
   - Różniczkuj od zewnątrz do wewnątrz

3. **Szukaj skróceń**
   - Często elementy się skracają (to nie przypadek - te funkcje dobrano właśnie dlatego!)

4. **Sprawdź wymiary**
   - Gradient musi mieć ten sam kształt co zmienna po której różniczkujesz

5. **Test zdroworozsądkowy**
   - Większy błąd → większa korekta? ✓
   - Większe wejście → większy wpływ na gradient? ✓

### Najważniejsze narzędzia matematyczne

**Reguła łańcuchowa:**
$$\frac{df(g(x))}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}$$

**Pochodna iloczynu:**
$$\frac{d(fg)}{dx} = f'g + fg'$$

**Pochodna ilorazu:**
$$\frac{d(f/g)}{dx} = \frac{f'g - fg'}{g^2}$$

**Pochodna wykładniczej:**
$$\frac{d(e^x)}{dx} = e^x$$

**Pochodna logarytmu:**
$$\frac{d(\ln x)}{dx} = \frac{1}{x}$$

**Algebra macierzy:**
- $(AB)^T = B^T A^T$
- Gradient względem macierzy często wymaga transpozycji

---

**Pamiętaj:** Te wzory NIE spadły z nieba! Każdy ma logiczne wyprowadzenie. Jeśli nie rozumiesz wzoru - wróć do wyprowadzenia krok po kroku! 📐✨
