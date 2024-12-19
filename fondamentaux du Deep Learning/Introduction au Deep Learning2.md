# Étape suivante : Travailler avec un jeu de données plus complexe

## Création du dataset

Nous allons maintenant étudier un cas où notre dataset contient davantage de données, rendant la tâche d’ajustement d’un modèle plus difficile.

### Dataset étendu :

```python
data = pd.DataFrame([
    {"n_chambres": 2, "prix": 10},
    {"n_chambres": 4, "prix": 20},
    {"n_chambres": 3, "prix": 15},
    {"n_chambres": 3, "prix": 20},
    {"n_chambres": 1, "prix": 3.5},
    {"n_chambres": 6, "prix": 31},
    {"n_chambres": 5, "prix": 33},
    {"n_chambres": 8, "prix": 36},
    {"n_chambres": 7, "prix": 50},
    {"n_chambres": 9, "prix": 42}
])

```

### Représentation tabulaire :

| Index | n_chambres | prix |
| --- | --- | --- |
| 0 | 2 | 10 |
| 1 | 4 | 20 |
| 2 | 3 | 15 |
| 3 | 3 | 20 |
| 4 | 1 | 3.5 |
| 5 | 6 | 31 |
| 6 | 5 | 33 |
| 7 | 8 | 36 |
| 8 | 7 | 50 |
| 9 | 9 | 42 |

## Visualisation des données

Lorsque nous représentons ces données sur un graphique, les points sont dispersés et ne suivent pas une relation simple comme dans le cas précédent.

```python
plt.scatter(data['n_chambres'], data['prix'], color='red')
plt.xlabel("Nombre de chambres")
plt.ylabel("Prix (en milliers)")
plt.title("Distribution des données")
plt.show()

```

### Résultat attendu :

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/9ac25fab-e20b-40de-a0e3-afe5bec21adc/7934c911-9f1a-427b-aa8d-22cfdd3342ea/image.png)

## Limites du modèle précédent

Si nous utilisons le modèle $y = 5x$  trouvé précédemment, il devient évident qu’il ne s’ajuste pas bien à ces nouvelles données. Cela montre que trouver une droite qui passe exactement par tous les points est non seulement difficile, mais aussi **non idéal**. Nous avons besoin d’une méthode pour trouver une **droite d’ajustement** qui minimise les erreurs globales.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/9ac25fab-e20b-40de-a0e3-afe5bec21adc/6b4ae0a1-6256-4e63-ad09-da59cfd4a814/image.png)

# Objectif : Trouver une droite qui minimise les erreurs

## Définir le problème

L’objectif est de trouver une droite $y = wx$ qui se rapproche **au mieux** des points, en minimisant les écarts (erreurs) entre les **valeurs observées** $( y )$ et les **valeurs prédites** par le modèle $( y_{\text{modèle}} )$.

## Calcul de l'erreur

### Qu’est-ce qu’une erreur ?

L’erreur pour un point donné est la différence entre la valeur réelle $( y )$ et la valeur prédit $( y_{\text{modèle}} )$ par la droite :

$$

\text{Erreur} = y - y_{\text{modèle}}

$$

Pour que notre modèle prenne en compte toutes les erreurs et accorde plus de poids aux grandes erreurs, nous utilisons la somme des carrés des erreurs :

$$

\text{Erreur totale} = \sum \left( y - y_{\text{modèle}} \right)^2

$$

### Pourquoi utiliser le carré des erreurs ?

- Les grandes erreurs sont amplifiées, ce qui incite le modèle à les réduire.
- Les petites erreurs ont moins d’impact.
- Cela évite que les erreurs positives et négatives s’annulent.

## Implémentation en Python

### Définir le modèle  $y = wx$

```python
def model(x, w):
    return w * x

```

Cette fonction retourne la prédiction $( y_{\text{modèle}} )$ pour une valeur donnée de $x$  et un poids $w$ (le coefficient de la droite).

### Visualiser différentes droites

Pour comprendre comment différents $w$ affectent la droite, traçons plusieurs droites sur le graphique.

```python
import numpy as np

# Tracer les données
plt.scatter(data['n_chambres'], data['prix'], color='red', label="Données")

# Essayer différentes valeurs de w
for w in [3, 4, 5, 6]:
    plt.plot(data['n_chambres'], model(data['n_chambres'], w), label=f"w = {w}")

plt.xlabel("Nombre de chambres")
plt.ylabel("Prix (en milliers)")
plt.title("Différentes droites d'ajustement")
plt.legend()
plt.show()

```

### Résultat attendu :

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/9ac25fab-e20b-40de-a0e3-afe5bec21adc/16050fcb-00a4-4c69-a854-1c3569e88dfd/image.png)

## Fonction d’erreur

Pour évaluer quelle droite est la meilleure, définissons une fonction d’erreur qui calcule la somme des carrés des écarts pour tous les points.

```python
def erreur(x, w, y):
    # Calculer les prédictions
    y_model = model(x, w)
    # Calculer la distance (erreur quadratique)
    distance = (y - y_model) ** 2
    # Retourner la somme des distances
    return np.sum(distance)

```

## Exemple : Calcul de l’erreur

Essayons différentes valeurs de $w$ pour trouver celle qui minimise l’erreur.

```python
# Tester différentes valeurs de w
for w in [3, 4, 5, 6]:
    err = erreur(data['n_chambres'], w, data['prix'])
    print(f"Erreur pour w = {w}: {err}")

```

Cela nous montre l’erreur totale pour chaque valeur de $w$. Le $w$ avec la plus petite erreur correspond à la meilleure droite d’ajustement.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/9ac25fab-e20b-40de-a0e3-afe5bec21adc/da79ab80-3fcf-48d5-9564-9da7925516e1/image.png)

## Optimisation de notre modèle : Approche par recherche exhaustive

## Étape 1 : Exploration fine autour de la meilleure valeur de $w$

Nous avons constaté que pour $w = 5$, l’erreur est minimisée par rapport aux autres valeurs initiales testées $( w = 3, 4, 6$ , etc.). Pour aller plus loin, nous testons des valeurs légèrement supérieures et inférieures, comme $w = 4.9$ et $w = 5.1$, et observons une amélioration de l'erreur avec $w = 5.1$. Cela suggère que nous pouvons affiner davantage notre recherche en explorant des valeurs intermédiaires plus précises.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/9ac25fab-e20b-40de-a0e3-afe5bec21adc/d8e0f6fb-3657-461a-93c1-e7b20a58e005/image.png)

## Étape 2 : Tester une gamme étendue de $w$ avec un pas fin

Pour automatiser ce processus et trouver la valeur optimale de $w$, nous pouvons tester une large gamme de valeurs, par exemple de $-20$ à $20$, avec un **pas de 0,001**. Cela nous permettra de calculer l’erreur pour chaque valeur de $w$ et de trouver celle qui produit la plus petite erreur.

### Générer les valeurs possibles de $w$

Nous utilisons la fonction `np.arange` pour créer une liste de valeurs possibles :

```python
possible_w = np.arange(-20, 20, 0.001)

```

Cette commande génère une liste contenant des milliers de valeurs de $w$, toutes espacées de 0,001. 

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/9ac25fab-e20b-40de-a0e3-afe5bec21adc/a9678a79-2cb2-4572-9b78-71df665cb5c5/image.png)

## Étape 3 : Calcul des erreurs pour chaque $w$

Nous utilisons une boucle pour tester chaque valeur de  $w$ sur notre dataset et enregistrer l’erreur associée dans une liste appelée `errors`.

```python
errors = []

for w in possible_w:
    e = erreur(x=data['n_chambres'], w=w, y=data['prix'])
    errors.append(e)

```

Ici :

- `data['n_chambres']` représente les $x$ (les nombres de chambres),
- `data['prix']` représente les $y$ (les prix réels des maisons),
- La fonction `erreur` calcule la somme des carrés des erreurs pour une valeur donnée de $w$.

## Étape 4 : Visualisation des erreurs en fonction de $w$

Nous traçons un graphique pour observer comment l’erreur varie en fonction de $w$.

```python
plt.plot(possible_w, errors)
plt.xlabel('Valeurs possibles de w')
plt.ylabel('Erreur totale')
plt.title('Erreur en fonction des valeurs de w')
plt.show()

```

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/9ac25fab-e20b-40de-a0e3-afe5bec21adc/d146e2ca-eb93-4fec-94d4-6face7ea3f78/image.png)

### Interprétation du diagramme :

Le graphique montre une **courbe en U**, où l’axe des $x$ correspond aux valeurs possibles de $w$ et l’axe des $y$ représente les erreurs totales.

- La partie basse de la courbe représente les valeurs de $w$ qui minimisent l’erreur.
- Le point le plus bas de cette courbe correspond à la meilleure valeur de $w$, c’est-à-dire celle qui ajuste le mieux notre modèle aux données.

## Étape 5 : Trouver le $w$ optimal

Pour identifier la valeur optimale de $w$, nous recherchons l’indice de l’erreur minimale dans la liste `errors` :

```python
min_error = np.min(errors)
optimal_index = np.argmin(errors)
optimal_w = possible_w[optimal_index]

```

### Interprétation :

- `np.min(errors)` : retourne la valeur minimale d’erreur.
- `np.argmin(errors)` : retourne la position (indice) de cette erreur minimale dans la liste.
- `possible_w[optimal_index]` : retourne la valeur de $w$ correspondante.

Par exemple, si `optimal_index = 25359`, cela signifie que  $w \approx 5.36$  est la meilleure valeur testée, minimisant l’erreur.

donc notre model sera:

```python
plt.scatter(data['n_chambres'], data['prix'], color='red')
plt.plot(data['n_chambres'], data['n_chambres'] * 5.359000000030992, color='blue', label="Modèle : y=5x")
plt.xlabel("Nombre de chambres")
plt.ylabel("Prix (en milliers)")
plt.title("Distribution des données")
plt.show()
```

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/9ac25fab-e20b-40de-a0e3-afe5bec21adc/ac45536a-8a9b-4d67-abc1-c2344150384c/image.png)

# Liens avec le Machine Learning

Reprenons les termes vus jusqu’ici et expliquons-les en langage Machine Learning :

1. **Modèle** :
    - En Machine Learning, un modèle est une fonction qui établit une relation entre les entrées $( x )$ et les sorties $(y)$.
    - Ici, notre modèle est une droite $y = wx$.
2. **Paramètres** :
    - Les paramètres sont les valeurs internes du modèle que nous ajustons pour améliorer ses performances.
    - Dans notre cas, $w$ est le paramètre de la droite.
3. **Erreur (ou coût)** :
    - L’erreur mesure l’écart entre les prédictions $( y_{\text{modèle}} )$ et les vraies valeurs $(y)$.
    - Nous utilisons la somme des carrés des erreurs comme fonction d’erreur :
    
    $$
    
    \text{Erreur totale} = \sum (y - y_{\text{modèle}})^2
    
    $$
    
4. **Minimisation de l’erreur** :
    - L’objectif du Machine Learning est de trouver les paramètres $( w )$ qui minimisent la fonction d’erreur.
    - Dans notre exemple, nous testons différentes valeurs de $w$ pour minimiser cette erreur.
5. **Recherche exhaustive** :
    - Tester toutes les valeurs possibles dans une plage définie est une méthode simple mais coûteuse en calcul.
    - Dans des modèles plus complexes, on utilise des algorithmes comme la descente de gradient pour optimiser efficacement.
6. **Visualisation des erreurs** :
    - Le graphique des erreurs en fonction des $w$ montre comment le choix des paramètres affecte la performance du modèle.
7. **Validation du modèle** :
    - Une fois le meilleur $w$ trouvé, il est important de vérifier que le modèle s’ajuste correctement aux données tout en évitant le surajustement (overfitting).

# Ce que nous avons appris jusqu’ici

1. La construction d’un modèle de base $( y = wx )$.
2. Le calcul des erreurs pour évaluer les performances du modèle.
3. La recherche exhaustive comme première approche pour trouver le paramètre optimal.
4. L’importance de la visualisation pour comprendre le comportement de l’erreur.

Dans les prochaines étapes, nous pourrons explorer des approches plus avancées, comme la **descente de gradient**, pour optimiser automatiquement les paramètres.