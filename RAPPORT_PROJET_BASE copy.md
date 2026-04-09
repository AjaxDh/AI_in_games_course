# Rapport Projet RL - Agent Rolling Ball (Template)

Auteur: [Nom Prenom]
Date: [JJ/MM/AAAA]
Projet: Rolling Ball DQN
Matiere: [Nom du cours]

---

## Introduction

### Contexte
Ce projet porte sur l'entrainement d'un agent de Reinforcement Learning dans un environnement Unity (ML-Agents), avec un controle discret des actions pour deplacer une balle vers une cible (cube).

L'agent est entraine avec un algorithme Deep Q-Network (DQN), implante en Python, qui interagit avec Unity via un wrapper Gym.

### Objectifs
- Concevoir et completer un agent DQN fonctionnel.
- Etudier l'impact des hyperparametres et des recompenses sur l'apprentissage.
- Comparer 4 a 5 experiences avec configurations differentes.
- Interpreter les resultats (performance, stabilite, limites).

---

## Methodologie

### Rappel du cadre algorithmique
- Paradigme: Reinforcement Learning.
- Algorithme: DQN (Q-network + target network + replay memory).
- Fonction objectif: approximation de la Q-function.

Equation de Bellman (forme utilisee en DQN):

$$
Q(s_t, a_t) \leftarrow r_t + \gamma \max_a Q_{target}(s_{t+1}, a)
$$

### Concepts importants a mobiliser
- **Reinforcement Learning**: apprentissage par interaction agent-environnement.
- **Equation de Bellman**: met a jour la valeur d'une action a partir du reward immediat + futur estime.
- **Gradient / descente de gradient**: ajustement des poids du reseau pour reduire la loss.

### Parametres modifiables et hypotheses
- **Gamma ($\gamma$)**: poids des rewards futures.
- **N (nombre d'episodes)**: duree d'entrainement (risque sous-apprentissage si trop faible).
- **Learning rate (lr)**: vitesse de mise a jour des poids.
- **Epsilon (exploration/exploitation)**: comportement aleatoire vs politique apprise.
- **F (frequence de mise a jour target network)**.
- **B (batch size)**: taille d'echantillon replay pour chaque update.
- **M (buffer memory)**: capacite de replay memory.
- **NN (taille reseau)**: capacite de representation, cout de calcul, risque d'instabilite.
- **Rewards**: signal d'apprentissage (shaping + terminal rewards).

---

## Experiences

> Consigne: lancer **4 a 5 experiences** avec des parametres differents.

### Tableau recapitulatif des experiences

| Experience | Gamma | N episodes | lr | Epsilon (start/end/decay) | F | B | M | NN | Rewards | Attente principale |
|---|---:|---:|---:|---|---:|---:|---:|---|---|---|
| E1 (baseline) | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| E2 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| E3 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| E4 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| E5 (optionnel) | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |

### Detail de chaque experience (a dupliquer)

#### Experience E1 - [Titre]
- **Choix des parametres**:
  - [Expliquer ce qui est modifie et pourquoi]
- **Methodologie / reflexion / approche**:
  - [Comment le test a ete concu, variable controlee, comparaison]
- **Attentes avant execution**:
  - [Effet attendu sur reward, duree episode, stabilite]
- **Resultats observes**:
  - [Courbes, valeurs, comportement agent]
- **Interpretation rapide**:
  - [Est-ce coherent avec l'attente?]

#### Experience E2 - [Titre]
- **Choix des parametres**:
- **Methodologie / reflexion / approche**:
- **Attentes avant execution**:
- **Resultats observes**:
- **Interpretation rapide**:

#### Experience E3 - [Titre]
- **Choix des parametres**:
- **Methodologie / reflexion / approche**:
- **Attentes avant execution**:
- **Resultats observes**:
- **Interpretation rapide**:

#### Experience E4 - [Titre]
- **Choix des parametres**:
- **Methodologie / reflexion / approche**:
- **Attentes avant execution**:
- **Resultats observes**:
- **Interpretation rapide**:

#### Experience E5 (optionnelle) - [Titre]
- **Choix des parametres**:
- **Methodologie / reflexion / approche**:
- **Attentes avant execution**:
- **Resultats observes**:
- **Interpretation rapide**:

---

## Resultats

### Resultats bruts
- Figure 1: [courbe reward par episode]
- Figure 2: [courbe duree par episode]
- Figure 3 (optionnel): [moving average ou taux de succes]

### Resultats attendus vs observes
- **Ce qui etait attendu**:
  - [ ]
- **Ce qui a ete observe**:
  - [ ]
- **Ecarts et surprises**:
  - [ ]

### Tableau de comparaison finale

| Critere | E1 | E2 | E3 | E4 | E5 |
|---|---:|---:|---:|---:|---:|
| Reward moyenne (fin de run) | [ ] | [ ] | [ ] | [ ] | [ ] |
| Reward moyenne lissee | [ ] | [ ] | [ ] | [ ] | [ ] |
| Duree moyenne episode | [ ] | [ ] | [ ] | [ ] | [ ] |
| Stabilite (spikes) | [ ] | [ ] | [ ] | [ ] | [ ] |
| Taux de succes (si mesure) | [ ] | [ ] | [ ] | [ ] | [ ] |

---

## Analyse

### Tendances identifiees
- [Ex: gamma eleve -> meilleure vision long terme mais convergence plus lente]
- [Ex: lr trop grand -> instabilite des courbes]
- [Ex: eps_decay trop rapide -> blocage en politique sous-optimale]
- [Ex: batch plus grand -> gradient plus stable mais apprentissage plus lent]

### Limites
- Taille de l'echantillon (nombre de runs) [ ]
- Variabilite due a l'alea (seed non fixee) [ ]
- Sensibilite aux rewards shaping [ ]
- Temps de calcul / vitesse simulation [ ]

### Menaces a la validite
- Difference entre entrainement en Editor et build headless.
- Parametres non isoles si plusieurs changent a la fois.
- Dependance a l'initialisation aleatoire de l'environnement.

---

## Conclusion

- **Synthese des resultats**:
  - [Resume en 4-6 lignes]
- **Reponse aux objectifs initiaux**:
  - [Objectif 1 atteint? Objectif 2? Objectif 3?]
- **Configuration recommandee**:
  - [Parametres finaux retenus]
- **Ameliorations futures**:
  - [Ex: Double DQN, prioritized replay, recompenses mieux faconnees, evaluation sur plusieurs seeds]

---

## Annexes (optionnel)

### A. Configuration logicielle
- Version Unity: [ ]
- Version Python: [ ]
- Librairies: `torch`, `mlagents_envs`, `numpy`, `matplotlib`

### B. Commandes de lancement
```bash
# Unity: lancer la scene rolling_ball_gym puis Play
# Python:
python Rolling_ball_Python/rolling_ball_gym.py
```

### C. Journal de runs (court)
| Date | Experience | Duree run | Observation cle |
|---|---|---|---|
| [ ] | [ ] | [ ] | [ ] |
