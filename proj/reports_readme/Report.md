#  Projet de Prédiction d’Attribution Auteur-Publication via Graph Embedding

Ce projet explore l’application de **cinq algorithmes de graph embedding** — **Node2Vec**, **DeepWalk**, **GraphSAGE**, **GCN**, et **LINE** — pour la **prédiction de liens auteur-publication** dans la base de données bibliographiques **DBLP**. L’objectif principal est d’identifier la méthode la plus performante pour **attribuer des publications à leurs auteurs manquants** en utilisant des représentations vectorielles apprises à partir de la structure du graphe.

Le système génère des **embeddings de nœuds**.Les résultats sont visualisés dans une **interface interactive JavaFX** permettant l’exploration comparative, l’analyse de similarité cosinus, et la visualisation .

---

## Table des matières

1. [Qu’est-ce qu’un Graph Embedding ?](#-quest-ce-quun-graph-embedding-)
2. [Base de données DBLP](#-base-de-données-dblp)
3. [Détail des Algorithmes](#-détail-des-algorithmes)
   - [3.1 DeepWalk](#31-deepwalk)
   - [3.2 Node2Vec](#32-node2vec)
   - [3.3 LINE](#33-line)
   - [3.4 GCN (Graph Convolutional Network)](#34-gcn-graph-convolutional-network)
   - [3.5 GraphSAGE](#35-graphsage)
4. [Comparaison détaillée](#-comparaison-détaillée)

---

##  Qu’est-ce qu’un Graph Embedding ?

Un **graph embedding** (ou plongement de graphe) est une technique d’apprentissage automatique qui vise à représenter les **nœuds d’un graphe** sous forme de **vecteurs denses et continus** dans un espace de faible dimension (ex. ℝ^d, d ≪ |V|). Ces vecteurs, appelés **embeddings**, capturent la **structure topologique** du graphe et/ou les **attributs des nœuds**, de sorte que des nœuds similaires dans le graphe (selon une notion définie) soient proches dans l’espace vectoriel.

### Objectifs principaux :
- **Préserver la proximité** : nœuds connectés → embeddings proches.
- **Généraliser à de nouveaux nœuds** (inductive vs transductive).
- **Être efficace** sur de grands graphes (scalabilité).
- **Être compatible** avec des tâches downstream (classification, lien prediction, clustering).

Formellement, soit un graphe \( G = (V, E) \), un embedding est une fonction :
\[
f: V \rightarrow \mathbb{R}^d, \quad \text{avec } d \ll |V|
\]
où \( f(v) \) est le vecteur associé au nœud \( v \).

---

##  Base de données DBLP

**DBLP** (*Digital Bibliography & Library Project*) est une base de données bibliographiques informatique contenant des **publications (articles, conférences)** et leurs **auteurs**. Dans ce projet, nous modélisons DBLP comme un **graphe biparti** :

- **Nœuds** : auteurs (\( A \)) et publications (\( P \))
- **Arêtes** : relations « auteur-publie » → \( E \subseteq A \times P \)

Le graphe est donc non orienté et biparti : \( G = (A \cup P, E) \).

**Tâche** : prédire des arêtes manquantes (ex. : auteur inconnu pour une publication).  
**Évaluation** :  entraîner les embeddings, puis prédire les liens manquants via similarité cosinus .

---

## Détail des Algorithmes

### 3.1 DeepWalk

**Publication** : Perozzi et al., *KDD 2014*

####  Concept
DeepWalk traite un graphe comme un **corpus de texte** en générant des **marches aléatoires (random walks)** à partir de chaque nœud. Ces séquences sont ensuite traitées comme des "phrases", et les nœuds comme des "mots".

####  Étapes
1. **Générer des marches aléatoires** :
   Pour chaque nœud \( v_i \), effectuer \( \gamma \) marches de longueur \( t \) :
   \[
   W_{v_i} = (v_i, v_{i+1}, ..., v_{i+t})
   \]
   où \( v_{i+1} \sim \text{Uniform}(\mathcal{N}(v_i)) \)

2. **Apprentissage via Skip-gram** :
   Maximiser la probabilité de prédire les voisins contextuels :
   \[
   \max_{\Phi} \sum_{v \in V} \sum_{w \in \mathcal{N}_R(v)} \log P(w | \Phi(v))
   \]
   où \( \mathcal{N}_R(v) \) est l’ensemble des voisins dans les random walks, et :
   \[
   P(w | \Phi(v)) = \frac{\exp(\Phi(w)^\top \Phi(v))}{\sum_{u \in V} \exp(\Phi(u)^\top \Phi(v))}
   \]

3. **Approximation** : utilisée **Negative Sampling** ou **Hierarchical Softmax** pour réduire la complexité.

#### Avantages
- Simple, efficace sur graphes homogènes.
- Transductif (ne généralise pas à de nouveaux nœuds).

#### Limites
- Marches purement aléatoires → ne capture pas la structure locale/globale de façon contrôlée.

---

### 3.2 Node2Vec

**Publication** : Grover & Leskovec, *KDD 2016*

#### Concept
Généralisation de DeepWalk avec un **biais contrôlé** dans les marches aléatoires via deux hyperparamètres :  
- **p** (return parameter)  
- **q** (in-out parameter)

Cela permet d’interpoler entre **BFS** (exploration locale) et **DFS** (exploration globale).

####  Étapes
1. **Marche aléatoire biaisée** :
   Soit un chemin \( (t, v, x) \). La probabilité de passer de \( v \) à \( x \) est :
   \[
   P(x | t, v) = \frac{\alpha_{t,x}}{Z}, \quad \text{où } \alpha_{t,x} =
   \begin{cases}
   \frac{1}{p} & \text{si } d_{tx} = 0 \text{ (retour)} \\
   1 & \text{si } d_{tx} = 1 \text{ (voisin commun)} \\
   \frac{1}{q} & \text{si } d_{tx} = 2 \text{ (loin)}
   \end{cases}
   \]
   avec \( d_{tx} \) = distance entre \( t \) et \( x \).

2. **Skip-gram** (identique à DeepWalk).

####  Avantages
- Flexible : ajuste la nature des similarités.
- Meilleure capture de la structure hiérarchique.

####  Limites
- Transductif.
- Coût de réglage de (p, q).

---

### 3.3 LINE (Large-scale Information Network Embedding)

**Publication** : Tang et al., *WWW 2015*

####  Concept
Optimise directement deux types de **proximités** :
1. **Premier ordre** : similarité directe (arêtes).
2. **Second ordre** : similarité via voisins communs.

####  Étapes
1. **Proximité du 1er ordre** :
   Minimiser :
   \[
   \mathcal{L}_1 = -\sum_{(i,j) \in E} w_{ij} \log \sigma(\mathbf{u}_i^\top \mathbf{u}_j)
   \]
   où \( w_{ij} \) est le poids de l’arête.

2. **Proximité du 2nd ordre** :
   Chaque nœud \( i \) a un vecteur **contextuel** \( \mathbf{u}'_i \). Minimiser :
   \[
   \mathcal{L}_2 = -\sum_{(i,j) \in E} w_{ij} \log \sigma(\mathbf{u}'_i^\top \mathbf{u}_j) + \text{régularisation}
   \]
   L’objectif global : \( \mathcal{L} = \mathcal{L}_1 + \mathcal{L}_2 \)

3. **Optimisation** : Edge sampling pour gérer les graphes pondérés.

####  Avantages
- Conçu pour **grands graphes**.
- Prend en compte les poids.
- Espace d’embedding explicite.

####  Limites
- Ne capture pas les structures de haut niveau (>2-hop).
- Transductif.

---

### 3.4 GCN (Graph Convolutional Network)

**Publication** : Kipf & Welling, *ICLR 2017*

####  Concept
Utilise un **réseau de neurones** avec des **convolutions de graphe** pour agréger les informations des voisins.

####  Étapes
1. **Couche de convolution** :
   \[
   H^{(l+1)} = \sigma\left( \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)} \right)
   \]
   avec :
   - \( \tilde{A} = A + I \) (graphe + auto-boucles)
   - \( \tilde{D}_{ii} = \sum_j \tilde{A}_{ij} \)
   - \( H^{(0)} = X \) (attributs des nœuds, ou one-hot si non disponibles)

2. **Sortie** :
   Après \( L \) couches, \( Z = H^{(L)} \) est l’embedding.

3. **Apprentissage** : supervision partielle (ex. : prédire labels ou liens).

####  Avantages
- **Inductif** si attributs fournis.
- Propage l’information de façon hiérarchique.
- Interprétable via propagation de messages.

####  Limites
- Transductif si pas d’attributs (dépend de la matrice d’adjacence entière).
- Coût O(|V|²) en mémoire pour grands graphes.
- Pas de prise en charge native des graphes bipartis (nécessite adaptation).

---

### 3.5 GraphSAGE (Graph Sample and AggregatE)

**Publication** : Hamilton et al., *NIPS 2017*

####  Concept
Méthode **inductive** : génère des embeddings **sans accès à l’ensemble du graphe** à l’inférence. Utilise un **échantillonnage de voisins** et un **agrégateur paramétrique**.

####  Étapes
Pour chaque nœud \( v \) :
1. **Échantillonner** \( k \) voisins à chaque couche.
2. **Agréger** les embeddings des voisins :
   \[
   h_{\mathcal{N}(v)}^{(k)} = \text{AGGREGATE}^{(k)}\left( \{ h_u^{(k-1)}, \forall u \in \mathcal{N}(v) \} \right)
   \]
   Options : mean, LSTM, pooling.

3. **Combiner** avec l’embedding du nœud :
   \[
   h_v^{(k)} = \sigma\left( W^{(k)} \cdot \text{CONCAT}(h_v^{(k-1)}, h_{\mathcal{N}(v)}^{(k)}) \right)
   \]

4. **Normaliser** : \( h_v^{(k)} \leftarrow h_v^{(k)} / \|h_v^{(k)}\| \)

####  Avantages
- **Inductif** par nature.
- Scalable : ne charge pas tout le graphe.
- Flexible (choix d’agrégateur).

####  Limites
- Moins précis que GCN sur petits graphes.
- Nécessite un échantillonnage soigneux.

---

##  Comparaison détaillée

| Critère              | DeepWalk | Node2Vec | LINE     | GCN       | GraphSAGE |
|----------------------|----------|----------|----------|-----------|-----------|
| **Nature**           | Transductif | Transductif | Transductif | Transductif* | **Inductif** |
| **Utilise attributs ?** | Non      | Non      | Non      | **Oui**   | **Oui**   |
| **Scalabilité**      | Moyenne  | Moyenne  | **Haute**| Faible    | **Haute** |
| **Contrôle du voisinage** | Aléatoire | **Paramétrable** | 1/2-hop | Fixe (K-hop) | **Échantillonné** |
| **Optimisation**     | Skip-gram| Skip-gram| Proximité | Backprop  | Backprop  |
| **Meilleur pour**    | Graphes denses, homogènes | Graphes avec structure hiérarchique | Grands graphes bipartis | Graphes avec attributs, semi-supervisé | **Nouveaux nœuds**, grands graphes |
| **Complexité**       | O(|E|tγ) | O(|E|tγ) | O(|E|)   | O(|V|²)   | O(L·k·|V|) |

> \* GCN peut être inductif si les attributs sont disponibles, mais classiquement transductif dans DBLP (pas d’attributs → one-hot → dépend de |V|).

### Quand utiliser quel algorithme ?

- **DBLP biparti sans attributs** → **LINE** ou **Node2Vec** (bons compromis précision/temps).
- **Grands graphes + nouveaux auteurs/publications** → **GraphSAGE**.
- **Précision maximale + petit graphe** → **GCN** (si on peut ajouter des attributs comme mots-clés).
- **Analyse exploratoire rapide** → **DeepWalk**.
- **Structure communautaire forte** → **Node2Vec** avec (p < q).

---

##  Méthodologie expérimentale

1. **Prétraitement DBLP** :
   - Extraction d’un sous-graphe (ex. : 10k auteurs, 20k pubs).
   - Conversion en graphe biparti non orienté.

2. **Génération d’embeddings** :
   - Dimension fixe : d = 64.
   - Paramètres optimisés via validation (ex. : p=1, q=2 pour Node2Vec).

3. **Prédiction de liens** :
   - Pour chaque publication masquée, calculer la similarité cosinus avec tous les auteurs :
     \[
     \text{sim}(a, p) = \frac{\Phi(a) \cdot \Phi(p)}{\|\Phi(a)\| \cdot \|\Phi(p)\|}
     \]
   - Évaluer via **AUC**, **Precision@10**, **Recall@50**.

4. **Mesures de performance** :
   - Temps d’entraînement (CPU/GPU).
   - Consommation mémoire.
   - Score de prédiction.

---
...

##  Interface JavaFX

L’interface permet :
- Chargement des résultats (dplp.xml).
- Visualisation du bipartite graph.
- Visualisation 2D des embeddings par différents algorithmes .


###  Contraintes techniques et optimisations

La **visualisation de graphes à grande échelle** pose des défis majeurs pour les interfaces graphiques traditionnelles comme **JavaFX**, particulièrement lorsqu’elles sont exécutées sur des machines standard (sans GPU dédié). Le graphe complet de DBLP peut contenir **des centaines de milliers, voire des millions de nœuds**, ce qui rend toute tentative de visualisation intégrale **prohibitivement coûteuse en mémoire et en temps de rendu**. En pratique, JavaFX **devient extrêmement lent** ou **crash** (OutOfMemoryError) lorsqu’on tente d’afficher plus de quelques milliers de nœuds simultanément.

Pour contourner cette limitation, l’interface adopte une stratégie de **visualisation échantillonnée intelligente** :

- **Les embeddings sont calculés sur l’ensemble complet du graphe** (tous les auteurs et publications). Cela garantit que les représentations vectorielles capturent fidèlement la structure globale du réseau, même si seuls quelques nœuds sont affichés.
- **Seul un sous-ensemble aléatoire ou ciblé de nœuds  est chargé dans JavaFX à la fois**, rendant l’interface fluide et réactive.
- **L’espace d’embedding (initialement en 64 dimensions) est projeté en 2D** à l’aide de **PCA** (pour la rapidité) ou **t-SNE** (pour la préservation locale), permettant une représentation visuelle intuitive sans perte critique de proximité sémantique.

###  Fonctionnalités interactives de la visualisation

L’utilisateur dispose de plusieurs outils pour explorer les résultats de manière ciblée :

1. **Sélection aléatoire ou manuelle** :
   - Possibilité de **choisir aléatoirement un petit ensemble de nœuds** (auteurs et/ou publications) à visualiser.
   - Ou de **saisir des identifiants spécifiques** (ex. : "Publication#12345") pour les afficher avec leurs voisins prédits.

2. **Affichage enrichi des informations** :
   - Au survol ou au clic sur un nœud, l’interface affiche :
     - Le **type** du nœud (auteur ou publication).
     - Son **identifiant** et ses métadonnées (nom, titre, année).
     - Les **vecteurs d’embedding** (tronqués ou projetés).
     - Pour les paires (auteur, publication) : la **similarité cosinus** calculée à partir des embeddings.
     - La **probabilité prédite de lien** (si un modèle de classification a été entraîné en aval des embeddings).

3. **Comparaison dynamique** :
   - L’utilisateur peut visualiser directement la distance entre des vecteurs, ainsi que la probabilité que le lien existe.

Cette approche garantit une **expérience interactive fluide**, tout en exploitant la **puissance prédictive des embeddings appris sur le graphe entier**, même si la visualisation reste nécessairement partielle.

...

