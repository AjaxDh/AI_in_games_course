# Rapport Projet RL - Agent Rolling Ball

Auteur: Ajax DESHAYES--HUET
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
- Faire evoluer les objectifs au fil des experiences a partir des resultats observes.

### Logique generale du rapport
Le rapport doit montrer une demarche iterative et non une suite de tests isoles.
Chaque experience sert a:
1. observer un comportement,
2. l'interpreter,
3. identifier une limite,
4. ajuster les parametres,
5. verifier si la nouvelle hypothese ameliore la situation.

Autrement dit, les resultats d'une experience modifient l'objectif de la suivante.

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

### Comment raisonner sur les modifications
- Si Unity devient lent au fil des episodes, reduire `N` ou la frequence de tests longs pour accelerer le cycle d'analyse.
- Si la courbe reward montre beaucoup de spikes, verifier en premier les rewards et `lr`.
- Si l'agent semble sous-apprendre, on peut augmenter la duree d'entrainement ou reduire plus lentement l'exploration.
- Si l'agent apprend mais reste instable, on peut reduire `lr`, lisser les rewards, ou ajuster `F`.
- Si le calcul devient trop couteux, garder les changements qui ont le plus d'impact sur l'apprentissage et limiter le reste.

### Libertes techniques ajoutees (hors consigne minimale)
Des fonctionnalites pratiques ont ete ajoutees pour mieux conduire les experiences, sans changer le coeur de l'algorithme DQN:
- timer global et temps par episode,
- pause/reprise pendant l'entrainement (touche espace) et arret propre (touche `Q`),
- journal automatique des episodes (CSV),
- resume de run (JSON) et journal lisible (Markdown),
- checkpoints periodiques du modele.

Ces ajouts servent surtout a la reproductibilite et au confort experimental (suivi, comparaison, reprise), pas a "booster" artificiellement les performances de l'agent.

---

## Experiences

> Consigne: lancer **4 a 5 experiences** avec des parametres differents.

### Fil conducteur experimental
Le rapport peut etre raconte comme une suite d'iterations:
- Experience 1: base initiale apres completion des exercices.
- Experience 2: ajustement pour reduire le lag et accelerer l'experience.
- Experience 3: correction d'un sous-apprentissage ou d'une instabilite observee.
- Experience 4: affinement pour rendre la courbe plus reguliere.
- Experience 5 (optionnelle): validation finale de la configuration retenue.

### Tableau recapitulatif des experiences

| Experience | Gamma | N episodes | lr | Epsilon (start/end/decay) | F | B | M | NN | Rewards | Attente principale |
|---|---:|---:|---:|---|---:|---:|---:|---|---|---|
| E1 (baseline) | 0.99 | 300 | 1e-4 | 0.9 / 0.05 / 3000 | 500 | 128 | 100000 | [9,512,512,5] | shaping +/-0.02, terminal +1/-1, timeout -0.5 | Baseline stable a lancer |
| E2 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| E3 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| E4 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| E5 (optionnel) | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |

### Detail de chaque experience (a dupliquer)

#### Experience E1 - [Titre]
- **Choix des parametres**:
  - Configuration actuelle du projet (baseline): `gamma=0.99`, `N=300`, `lr=1e-4`, `epsilon=0.9/0.05/3000`, `F=500`, `batch_size=128`, `memory=100000`, reseau `[9,512,512,5]`.
  - Reward design: shaping `+0.02/-0.02`, terminal `+1/-1`, timeout `-0.5`.
- **Methodologie / reflexion / approche**:
  - Cette experience sert de point de reference pour comparer toutes les experiences suivantes.
- **Attentes avant execution**:
  - Verifier la stabilite globale de la courbe reward et la tendance de duree moyenne des episodes.
- **Resultats observes**:
  - [A completer avec le plot]
- **Interpretation rapide**:
  - [ ]

#### Experience E2 - [Titre]
- **Choix des parametres**:
  - [ ]
- **Methodologie / reflexion / approche**:
  - [ ]
- **Attentes avant execution**:
  - [ ]
- **Resultats observes**:
  - [ ]
- **Interpretation rapide**:
  - [ ]

#### Experience E3 - [Titre]
- **Choix des parametres**:
  - [ ]
- **Methodologie / reflexion / approche**:
  - [ ]
- **Attentes avant execution**:
  - [ ]
- **Resultats observes**:
  - [ ]
- **Interpretation rapide**:
  - [ ]

#### Experience E4 - [Titre]
- **Choix des parametres**:
  - [ ]
- **Methodologie / reflexion / approche**:
  - [ ]
- **Attentes avant execution**:
  - [ ]
- **Resultats observes**:
  - [ ]
- **Interpretation rapide**:
  - [ ]

#### Experience E5 (optionnelle) - [Titre]
- **Choix des parametres**:
  - [ ]
- **Methodologie / reflexion / approche**:
  - [ ]
- **Attentes avant execution**:
  - [ ]
- **Resultats observes**:
  - [ ]
- **Interpretation rapide**:
  - [ ]

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
- Certaines ameliorations de vitesse peuvent detruire de la stabilite, et inversement.

### Lecture des spikes
- Des spikes ne signifient pas automatiquement un bug.
- Ils peuvent venir de l'exploration encore trop forte, d'un shaping trop agressif, ou d'une politique pas encore stabilisee.
- Il faut lire ensemble reward moyenne, duree moyenne, et taux de succes.

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

### Phrase de synthese type
L'analyse doit montrer comment chaque experience modifie la suivante: le projet est donc un processus d'optimisation iterative, pas seulement une comparaison de chiffres.

---

