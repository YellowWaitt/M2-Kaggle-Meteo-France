# M2 Kaggle Meteo France

La participation de mon groupe de M2 à un défi sur [Kaggle](https://www.kaggle.com/c/defi-ia-2022/overview).
Il n'y a rien de révolutionnaire ici mais des choses qui fonctionnent.

Voilà ce que l'on peux retrouver éparpillé dans les différents fichiers :

#### `datamanip.py`

Contient deux outils permettant de compléter les trous dans les données que l'on a utilisées :
- `check_nb_hours` qui permet de rajouter une ligne de `nan` pour les journées dont les 24 heures n'étaient pas présentes.
- `KNNFiller` qui pour chacune des heures prises séparemment effectue une interpolation linéaires des k plus proches stations voisines pour remplacer les valeurs `nan` dans les données.

#### `plots.py`

Contient une fonction qui permet de créer des animations. En lui fournissant les données et un paramètre, l'affiche sur une carte de France pour la période souhaîtée où la valeur sur chaque station est indiquée. Pratique pour avoir un aperçu visuel sur comment les datas varient en fonction du temps et de la répartition spatiale.

#### `windowgenerator.py`

Une classe reprise de [la documentation de tensorflow](https://www.tensorflow.org/tutorials/structured_data/time_series#data_windowing) qui génère des fenêtres d'entraînement pour les réseaux de neuronnes. Sont rôle est de générée des batchs (X, Y) qui contiennent chacun des paramètres précisés sur un nombre d'heure également à précisé.

Dans notre cas on a utilisé des fenêtres de 24 heures pour prédires des quantités de pluies par heure sur 24 heures.

La classe a été modifié afin de pouvoir travailler sur plusieurs stations à la fois. Les données sur chaques stations sont regroupées, les fenêtres générées et enfin les batchs mélangés. C'est la pièce important qui fait le lien entre les données et le modèle.

Le fichier contient également la fonction `make_pred` qui s'assure de bien formater les données d'entrées pour le réseau et la prédiction pour la soumition sur Kaggle.

#### `make_rnn.py`

Ce fichier résume comment est utlisé le reste du code et illustre comment définir le modèle tensorflow et l'entraîner. Les étapes sont simples :
- chargement des données.
- définition d'une fenêtre.
- définition d'un modèle tensorflow.
- entraînement du réseau.
- prédictions du modèle à l'aide de la fenêtre.

#### `dataloader.py`

Contient quelques fonctions pour charger les données dans des dataframes.

#### `model.py`

Un wrapper pour utiliser des modèles fournis par `sklearn`. Le modèle très simpliste utilise toutes les données sur 24 heures pour prédire la quantité de pluie accumulée du lendemain.
