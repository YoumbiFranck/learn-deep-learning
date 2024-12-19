# Introduction au Deep Learning : Les fondamentaux à travers un exemple simple

Pour comprendre les bases du **Deep Learning**, il est essentiel de commencer par un exemple simple. Supposons que nous souhaitons créer un modèle qui permet de **prédire le prix des maisons** en fonction de leurs caractéristiques. Dans cet exemple, nous allons nous concentrer sur une seule caractéristique : **le nombre de chambres**. Cela nous aidera à comprendre les concepts fondamentaux que nous approfondirons plus tard.

## Création du dataset

Voici notre jeu de données (dataset) de départ :

```python
import pandas as pd

data = pd.DataFrame([
    {"n_chambres": 2, "prix": 10},
    {"n_chambres": 4, "prix": 20}
])

```

### Représentation tabulaire des données :

| Index | n_chambres | prix |
| --- | --- | --- |
| 0 | 2 | 10 |
| 1 | 4 | 20 |

### Description des colonnes :

1. **n_chambres** : Le nombre de chambres dans une maison.
2. **prix** : Le prix de la maison (en milliers d'euros, par exemple).

Ces données montrent clairement une relation entre le nombre de chambres et le prix. Mais comment exprimer cette relation mathématiquement ?

## Visualisation des données

Pour mieux comprendre la relation entre ces deux colonnes, nous pouvons les **visualiser sur un graphique**. Utilisons un scatter plot :

```python
import matplotlib.pyplot as plt

plt.scatter(data['n_chambres'], data['prix'], color='red')
plt.xlabel("Nombre de chambres")
plt.ylabel("Prix (en milliers)")
plt.title("Relation entre le nombre de chambres et le prix")
plt.show()

```

### Résultat attendu :

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/9ac25fab-e20b-40de-a0e3-afe5bec21adc/044f6c7c-2119-4a2b-8781-82a4b8e108cd/image.png)

## Trouver la relation entre les données

En observant le tableau et le graphique, nous remarquons que :

- Pour 2 chambres, le prix est 10.
- Pour 4 chambres, le prix est 20.

Il semble que le **prix** est **proportionnel au nombre de chambres**, avec un facteur multiplicatif de 5. En d'autres termes :

$Prix=5×Nombre de chambres$

ou encore :

 $y = 5x$ 

Ici :

- $x$  représente le nombre de chambres $( n\_chambres )$.
- $y$ représente le prix.

Cette relation est une **droite**. En Deep Learning, on appelle cette relation un **modèle**.

## Tracer la droite de régression

Nous pouvons maintenant afficher cette relation sur le graphique en traçant la droite $y = 5x$ 

```python
plt.scatter(data['n_chambres'], data['prix'], color='red', label="Données")
plt.plot(data['n_chambres'], data['n_chambres'] * 5, color='blue', label="Modèle : y=5x")
plt.xlabel("Nombre de chambres")
plt.ylabel("Prix (en milliers)")
plt.title("Droite de régression : y = 5x")
plt.legend()
plt.show()

```

### Résultat attendu :

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/9ac25fab-e20b-40de-a0e3-afe5bec21adc/25c7afab-1f25-44c6-a8e5-ba70f92446f4/image.png)

## Prédiction avec le modèle

Maintenant que nous avons trouvé une relation $y = 5x$, nous pouvons l'utiliser pour **prédire le prix de maisons** qui ne sont pas dans notre dataset.

### Exemple :

- Combien coûterait une maison avec **3 chambres** ?
    - En utilisant notre modèle : $y = 5 \times 3 = 15$ .
    - Le prix estimé est **15 milliers d'euros**.

Graphiquement, le point correspondant à 3 chambres et 15 de prix se situera exactement sur la droite $y = 5x$.

```python
# Ajout d'une prédiction au graphique
plt.scatter(data['n_chambres'], data['prix'], color='red', label="Données")
plt.plot(data['n_chambres'], data['n_chambres'] * 5, color='blue', label="Modèle : y=5x")
plt.scatter(3, 15, color='green', label="Prédiction (3 chambres : 15)")
plt.xlabel("Nombre de chambres")
plt.ylabel("Prix (en milliers)")
plt.title("Prédiction avec le modèle")
plt.legend()
plt.show()

```

### Résultat attendu :

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/9ac25fab-e20b-40de-a0e3-afe5bec21adc/6edf6439-bdaa-4dbc-b928-d50a6640d53a/image.png)

## Ce que nous avons appris

1. **Relation entre les données :** Nous avons découvert une relation linéaire entre le nombre de chambres et le prix.
2. **Modèle :** Le modèle $y = 5x$ est une représentation mathématique de cette relation.
3. **Prédiction :** Grâce au modèle, nous pouvons estimer des valeurs futures.
4. **Fondement du Deep Learning :** Trouver des relations (souvent complexes) dans des données pour faire des prédictions est l'objectif principal du Deep Learning.

## Conclusion

Bien que cet exemple soit simple et linéaire, il pose les bases du Deep Learning. Dans des scénarios réels, les données et les relations peuvent être bien plus complexes, nécessitant des outils avancés comme les **réseaux de neurones profonds** pour les analyser. C’est ce que nous explorerons dans les prochaines étapes.