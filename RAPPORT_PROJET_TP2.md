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
- **Choix des parametres**: [baseline]
- **Ce qu'on teste**: [reference]
- **Resultats observes**: [reward, episode length, losses]
- **Interpretation**: [resume court]

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
- Une capture de la voiture en train de rouler correctement.
- Une capture d'un crash ou d'une sortie de route.
- Une capture d'un passage stable vers la fin de l'entrainement.
- Une capture des courbes si elles sont assez lisibles.

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
