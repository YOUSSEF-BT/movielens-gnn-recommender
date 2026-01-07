# TP — Recommender System avec GNN (Mini-LightGCN) sur MovieLens (Colab)

Ce TP (réalisé pendant le cours) montre comment construire un **système de recommandation** basé sur un **graphe user–movie** en utilisant **PyTorch Geometric**, puis comment générer des **recommandations Top-N** en filtrant les films déjà vus.

Notebook : **`GNN_Recommender.ipynb`**  
Environnement : **Google Colab**

---

## Objectifs

- Installer **PyTorch** + **PyTorch Geometric** sur Colab
- Charger le dataset **MovieLens** (fichiers `u.data` et `u.item`)
- Construire le graphe biparti **Users ↔ Movies** (`edge_index`)
- Implémenter un modèle simple type **Mini-LightGCN**
- Produire des **Top-N recommandations** (sans films déjà regardés)
- Visualiser un sous-graphe (échantillon) avec **NetworkX**

---

## Dataset (MovieLens 100K)

Le notebook utilise :
- `u.data` : interactions (user_id, item_id, rating, timestamp)
- `u.item` : métadonnées des films (titre, date de sortie, …)

### Option A — Upload manuel (simple)
1. Télécharge le dataset MovieLens 100K sur ton PC.
2. Dans Colab : **Files > Upload** et ajoute `u.data` et `u.item` à la racine du runtime.

### Option B — Téléchargement direct (si tu veux automatiser)
Tu peux ajouter une cellule (optionnelle) au début :
```
# Exemple (à adapter si besoin)
!wget -q -O ml-100k.zip https://files.grouplens.org/datasets/movielens/ml-100k.zip
!unzip -q ml-100k.zip
!cp ml-100k/u.data .
!cp ml-100k/u.item .
Installation (Colab)
```

Le notebook installe les dépendances principales :

torch, torchvision, torchaudio

torch-geometric

pandas, networkx, matplotlib (souvent déjà disponibles sur Colab)

Dans Colab : Runtime > Run all après l’installation.

Étapes principales du notebook

Installation des librairies

Importation du dataset (u.data + u.item)

Préprocessing

Re-indexation des user_id et item_id à partir de 0

Décalage des IDs movies (movie_offset) pour éviter le chevauchement avec les users

Construction de edge_index pour PyTorch Geometric

Mini-LightGCN

Embeddings pour chaque nœud (users + movies)

Propagation simple sur le graphe pour produire une représentation (node_repr)

Recommandations Top-N

Score(user, movie) via produit scalaire

Filtrage des films déjà vus par l’utilisateur

Affichage des films recommandés

Visualisation

Échantillon de quelques interactions (ex: 400)

Affichage du mini-graphe avec NetworkX

Résultat attendu

À la fin, tu dois obtenir :

Un tableau/listing des Top-N films recommandés pour un USER_ID

Une visualisation d’un sous-graphe user–movie

Exécution rapide (checklist)

 Ouvrir GNN_Recommender.ipynb dans Colab

 Avoir u.data et u.item disponibles dans l’espace de fichiers Colab

 Exécuter toutes les cellules dans l’ordre

 Tester avec différents USER_ID

Remarques

Le modèle “Mini-LightGCN” est une version simplifiée pour le TP (pédagogique).

Les recommandations dépendent de l’étape de représentation (embeddings + propagation) telle qu’implémentée dans le notebook.

Auteur

TP réalisé pendant le cours — implémentation et exécution sur Google Colab.
