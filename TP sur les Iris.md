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