# TP sur les Iris

Chaque apprenant doit me rendre ce travail avec le QCM rempli, créez un dépôt pour mettre votre code en ligne. Vous devez rédiger chacune des questions.

##  Régression Linéaire sur le Jeu de Données des Iris

Utilisez sklearn, module utilisé dans le ML, dans ce TP pour la droite de régression, il permet également de créer le modèle.

```python
from sklearn.linear_model import LinearRegression
# Création du modèle de régression linéaire
model = LinearRegression()

# Entraînement du modèle
model.fit(X, y) # X et Y sont à déterminées en fonction de vos données

# Coefficient (pente)
coefficient = model.coef_[0]

# Interception
interception = model.intercept_

print(f"Coefficient : {coefficient}")
print(f"Interception : {interception}")

# Prédictions du modèle == la droite de régression
y_pred = model.predict(X)

# Evaluation du modèle
r_squared = model.score(X, y)
print(f"Coefficient de détermination R² : {r_squared}")
```

- **Coefficient** : Le **coefficient**, si on s'intéresse au iris, selon les variables X et Y choisies, il peut indiquer l'effet de la longueur des sépales sur la longueur des pétales. Si le coefficient est positif, cela signifie que la longueur des pétales augmente avec la longueur des sépales. Coefficient directeur de la droite dans votre modèle.
- **Interception** : L'**interception** est la valeur prédite de la longueur des pétales lorsque la longueur des sépales est zéro (l'ordonnée à l'origine), valeur purement théorique, elle sert à placer la droite dans le graphique.
- **Coefficient de détermination r_squared** : Cette valeur varie entre 0 et 1 et indique la proportion de la variance de la variable cible expliquée (dépendante y) par le modèle. Plus la valeur de r_squared est proche de 1, meilleur est le modèle !

```txt
Y = C*X + I, on a une valeur de X longueur de sépale et Y serait la longueur des sépales
```

###  Préparation du Jeu de Données

1. **Chargement et exploration des données**
   
```python
import pandas as pd
from sklearn.datasets import load_iris

# Chargement du jeu de données des iris
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

# Affichage des premières lignes du DataFrame
print(iris_df.head())
```

2. **Sélection des variables explicatives et cible**
    - Sélectionnez la longueur des sépales comme variable explicative (X).
    - Sélectionnez la longueur des pétales comme variable cible (y).


### Modélisation - regression linéaire 

1. **Création et entraînement du modèle de régression linéaire**
    - Importez la classe `LinearRegression` de `sklearn.linear_model`.
    - Créez une instance de `LinearRegression` et entraînez le modèle sur les données.


2. **Affichage des coefficients appris par le modèle**
    - Affichez le coefficient et l'interception du modèle.


3. **Analyse des coefficients et interprétation**
    - Expliquez ce que représentent le coefficient et l'interception dans le contexte du jeu de données des iris.
    - Réponse : 
      - le coefficient montre comment les pétales réagir en fonction des sépales.
      - l'interception est l'estimation de la longueur des pétales lorsque la longueur des sépales est nulle.
   

4. Interprétez les résultats obtenus en termes de relation entre la longueur des sépales et la longueur des pétales.
   - Réponse : 
        - Coefficient (1.858) : Si la longueur des sépales (une partie de la fleur) augmente de 1 cm, en moyenne, la longueur des pétales (une autre partie de la fleur) augmente de presque 1.86 cm.
        - Interception (-7.10) : Si la longueur des sépales était de 0 cm (ce qui n'est pas possible dans la réalité des fleurs), notre modèle prédit que la longueur des pétales serait d'environ -7.10 cm.
          - valeur :   
            - Coefficient : [1.85843298]
            - Interception : -7.101443369602455

5. Que représente le coefficient dans une régression linéaire ?
    - Réponse :
      - Coefficient (ou pente) : Il indique de combien la variable cible (Y) change en moyenne lorsque la variable explicative (X) augmente d'une unité. 

6. En utilisant les coefficients obtenus, écrivez l'équation de la droite de régression.
    (P_length = petal length (cm), s_length = sepal length (cm))
    - P_length=Coefficient×s_length−Interception


7. **Visualisation**
    - Tracez un graphique de la longueur des sépales en fonction de la longueur des pétales.
    - Ajoutez la droite de régression au graphique pour visualiser l'ajustement du modèle.

8. **Évaluation du modèle**
    - Calculez le coefficient de détermination R² pour évaluer la performance du modèle.
      - Coefficient de détermination R² : 0.759954645772515 



Jupyter :
import pandas as pd
from sklearn.datasets import load_iris


# Chargement du jeu de données des iris
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

# Affichage des premières lignes du DataFrame
print(iris_df.head())x = pd.DataFrame(iris_df['sepal length (cm)'])
y = pd.DataFrame(iris_df['petal length (cm)'])

print(x)
print(y)from sklearn.linear_model import LinearRegression
# Création du modèle de régression linéaire
model = LinearRegression()

# Entraînement du modèle
model.fit(x, y) # X et Y sont à déterminées en fonction de vos données

# Prédictions du modèle == la droite de régression
y_pred = model.predict(x)#2 
coefficient = model.coef_[0][0]
interception = model.intercept_[0]
print(f"Coefficient : {coefficient}")
print(f"Interception : {interception}")r_squared = model.score(x, y)
print(f"Coefficient de détermination R² : {r_squared}")import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', label='Données réelles')

plt.plot(x, y_pred, color='red', linewidth=2, label='Droite de régression')

plt.xlabel('Longueur des sépales (cm)')
plt.ylabel('Longueur des pétales (cm)')
plt.title('Régression linéaire : Longueur des sépales vs Longueur des pétales')
plt.legend()

plt.grid(True)
plt.show()