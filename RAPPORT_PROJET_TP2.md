# Rapport Projet RL - TP2 AI Driver Unity (PPO)

Auteur: [Nom Prenom]
Date: [JJ/MM/AAAA]
Projet: Autonomous Car - PPO
Matiere: [Nom du cours]

---

## Introduction

Ce TP2 consiste a entrainer un agent de voiture dans Unity avec ML-Agents et PPO.
Le but est simple: faire rouler la voiture sans crash.

Le rapport peut rester leger. L'important est surtout de montrer:
1. la configuration de depart,
2. les changements faits entre les experiences,
3. ce qu'on observe sur les courbes,
4. la configuration retenue a la fin.

---

## Rappel du projet

### Fichiers principaux
- `results/configuration_example.yaml`: configuration PPO.
- `AI_in_games_unity/Assets/Scripts/car_agents/car_controller.cs`: mouvement de la voiture.
- `AI_in_games_unity/Assets/Scripts/car_agents/car_agent.cs`: classe de base de l'agent.
- `AI_in_games_unity/Assets/Scripts/car_agents/car_agent_template_track1.cs`: script a completer pour le TP2.

### Scripts a citer dans le rapport
- `results/configuration_example.yaml`: fichier de configuration du PPO.
- `AI_in_games_unity/Assets/Scripts/car_agents/car_controller.cs`: controle les roues, l'acceleration, le freinage et la direction.
- `AI_in_games_unity/Assets/Scripts/car_agents/car_agent.cs`: gere l'initialisation commune de l'agent et les episodes.
- `AI_in_games_unity/Assets/Scripts/car_agents/car_agent_template_track1.cs`: classe a completer pour connecter l'agent a l'environnement.

### Ce qu'il faut faire dans Unity
- Completer `_getTarget()` et `_getStart()`.
- Completer `CollectObservations()`.
- Completer les fonctions de reward.
- Completer `OnActionReceived()`.
- Regler les parametres de l'agent dans l'inspecteur.

### Parametres Unity a mentionner
- Behavior Name.
- Vector Observation space size.
- Stacked Vectors.
- Discrete branches.
- Model / Inference Device / Deterministic Inference si un modele est charge.

---

## Methodologie

### Point important sur PPO
PPO est une methode policy-based. On peut simplement dire que l'algorithme apprend une politique stable grace a des mises a jour controlees.

### Parametres utiles a tester
- Batch size.
- Buffer size.
- Learning rate.
- Epsilon.
- Beta.
- Lambda.
- Num epoch.
- Gamma.

### Remarque sur le batch size
Le prof a surtout insiste sur le batch size.
Si le batch size augmente, l'optimisation utilise plus d'observations, donc les mises a jour sont souvent plus stables mais aussi plus lourdes.

### E1 a lancer en premier
Le premier run sert de baseline.
Il faut garder la configuration par defaut, lancer l'entrainement sur la scene track1, puis commenter les courbes obtenues sans modifier trop de choses.

---

## Experiences

> Consigne: faire 4 experiences maximum, pas besoin d'aller trop loin.

### Tableau rapide

| Experience | Batch size | lr | Epsilon | Beta | Lambda | Num epoch | Attente principale |
|---|---:|---:|---:|---:|---:|---:|---|
| E1 | 512 | 3e-4 | 0.2 | 0.005 | 0.95 | 3 | Baseline |
| E2 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | Tester un changement simple |
| E3 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | Corriger le point faible observe |
| E4 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | Valider la meilleure config |

### Structure courte pour chaque experience

#### Experience E1
- **Choix des parametres**: batch size 512, learning rate 3e-4, epsilon 0.2, beta 0.005, lambda 0.95, num epoch 3.
- **Ce qu'on teste**: la baseline PPO sur track1.
- **Resultats observes**: [reward, episode length, entropy, policy loss, value loss]
- **Interpretation**: [resume court sur la stabilite et le comportement du car agent]

#### Experience E2
- **Choix des parametres**: [ ]
- **Ce qu'on teste**: [ ]
- **Resultats observes**: [ ]
- **Interpretation**: [ ]

#### Experience E3
- **Choix des parametres**: [ ]
- **Ce qu'on teste**: [ ]
- **Resultats observes**: [ ]
- **Interpretation**: [ ]

#### Experience E4
- **Choix des parametres**: [ ]
- **Ce qu'on teste**: [ ]
- **Resultats observes**: [ ]
- **Interpretation**: [ ]

---

## Resultats

### Figures a mettre dans le rapport
- Figure 1: courbe reward.
- Figure 2: courbe longueur d'episode.
- Figure 3: entropy.
- Figure 4: policy loss et value loss.

### Images / captures a commenter
- Figure A: voiture au depart de la track1.
- Figure B: voiture qui atteint la cible.
- Figure C: voiture qui crash ou sort de la route.
- Figure D: courbe reward de E1.
- Figure E: courbe episode length de E1.
- Figure F: courbes entropy, policy loss et value loss si elles sont lisibles.

### Comparaison finale

| Critere | E1 | E2 | E3 | E4 |
|---|---:|---:|---:|---:|
| Reward moyenne | [ ] | [ ] | [ ] | [ ] |
| Episode length moyenne | [ ] | [ ] | [ ] | [ ] |
| Entropy | [ ] | [ ] | [ ] | [ ] |
| Policy loss | [ ] | [ ] | [ ] | [ ] |
| Value loss | [ ] | [ ] | [ ] | [ ] |
| Taux de succes | [ ] | [ ] | [ ] | [ ] |

---

## Analyse

### A commenter simplement
- Le batch size plus grand ou plus petit.
- L'effet du learning rate.
- La stabilite des courbes.
- La difference entre performance et vitesse de calcul.

### Limites
- Un seul run par configuration.
- Alea dans l'environnement.
- Temps de calcul sur la machine.
- Influence possible du reward shaping.

### Bilan court
- Ce qui marche.
- Ce qui reste instable.
- Ce qu'on garde comme meilleure configuration.

---

## Conclusion

- **Synthese**: [3 a 5 lignes sur le comportement de l'agent]
- **Config retenue**: [parametres finaux]
- **Amelioration future possible**: [si besoin, une seule phrase]

---

## Annexe

### Lancement
```bash
mlagents-learn results/configuration_example.yaml --run-id=my_agent --force
```

### Rappel
Dans la console, la commande peut aussi produire `results/my_agent/configuration.yaml`.
