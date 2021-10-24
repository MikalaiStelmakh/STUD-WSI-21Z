##### `Mikalai Stelmakh 316951`

# WSI Lab 1 - przeszukiwanie przestrzen

## Zadanie

`Zaimplementowanie metody najszybszego spadku gradientu i metody Newtona dla następującej finkcji celu:`

```
f(x,y) = (1-x)^2 + 100(y-x^2)^2 ,  -5 <= x <= 5 , -5 <= y <= 5
```

---

## Uruchomenie programu

```
python3 minimize_rosenbrock.py [--method METHOD] [--graph] X Y step iterations epsilon
```

Żeby dowiedzieć się więcej na temat argumentów należy użyć flagi `"-h"`.

---

## Wyniki

### Metoda najszybszego spadku gradientu

- `Działa poprawnie tylko dla stosunkowo małego współczynniku β (< 0,0001).` </br>
  Dzieje się tak dlatego, że dla dłuższego kroku algorytm "przeskakuje" właściwy punkt i nigdy nie trafia w minimum funkcji.
- `Z powodu na małą długość kroku algorytm potrzebuje stosunkowo wielu iteracji dla znalezienia włąściwego punktu.` </br>
  Kilka przykładów dla różnych punktów początkowych (β = 0,0001):
  | Heading 1 | Heading 2 | Heading 3 |
  |-----------|:-----------:|-----------|
  | Cell 1 | Cell 2 | Cell 3 |
