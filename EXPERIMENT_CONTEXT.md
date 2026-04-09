# Experiment Context - Rolling Ball DQN

Last updated: 2026-04-09

## 1) Objectif
Entrainer un agent DQN (actions discretes) pour atteindre le cube dans la scene Unity de rolling ball.

## 1bis) Fil conducteur experimental
Le projet doit etre presente comme une boucle d'amelioration:
1. lancer une experience de reference,
2. observer les courbes,
3. interpreter les spikes, le lag, ou le sous-apprentissage,
4. ajuster les parametres,
5. relancer une experience pour valider l'hypothese suivante.

Chaque experience modifie donc l'objectif de la suivante.

## 2) Cadre de l'exercice (version prof)
1. Completer les blancs dans `Rolling_ball_Python/rolling_ball_DQN.py`.
2. Lancer l'entrainement via `Rolling_ball_Python/rolling_ball_gym.py` avec la scene Unity active.
3. Modifier les hyperparametres (Python) et les rewards (Unity).

## 3) Etat actuel du code
### DQN: points completes
- Architecture reseau: `[input, 512, 512, output]`.
- `forward()` implemente (ReLU -> ReLU -> output).
- Replay memory mise a jour dans `update()`.
- Cible Bellman a `t+1` calculee dans `learn()`.
- Synchronisation periodique du target network active.

### Etat experimental actuel
- Aucune experience n'a encore ete lancee apres la derniere remise a plat.
- Le rapport doit donc rester un template vierge pour les sections E1 a E5.
- Les prochaines analyses devront partir du premier run reel, pas d'un resultat pre-rempli.

### Hyperparametres actifs
Source: `Rolling_ball_Python/rolling_ball_gym.py` et `Rolling_ball_Python/rolling_ball_DQN.py`

| Parametre | Valeur |
|---|---|
| `input_size` | `9` |
| `output_size` | `5` |
| `batch_size` | `128` |
| `gamma` | `0.99` |
| `F` | `500` |
| `lr` | `1e-4` |
| `eps_start` | `0.9` |
| `eps_end` | `0.05` |
| `eps_decay` | `3000` |
| `n_episode` | `300` |

### Reward design active (agent discret)
Source: `AI_in_games_unity/Assets/Scripts/rolling_ball/rollerAgentDiscrete.cs`

- Rapprochement de la cible: `AddReward(+0.02)`
- Eloignement de la cible: `AddReward(-0.02)`
- Cible atteinte: `SetReward(+1.0)` puis `EndEpisode()`
- Chute: `SetReward(-1.0)` puis `EndEpisode()`
- Timeout (500 steps): `SetReward(-0.5)` puis `EndEpisode()`

## 4) Justification du barème reward
- Echelle normalisee proche de `[-1, 1]` pour la stabilite du DQN.
- Signal terminal clair (`+1.0`) pour l'objectif final.
- Reward shaping faible (`+/-0.02`) pour guider sans dominer le terminal.

## 5) Procedure de lancement
1. Ouvrir Unity et charger la scene d'entrainement rolling ball.
2. Appuyer sur Play dans l'Editor Unity.
3. Lancer le script Python `Rolling_ball_Python/rolling_ball_gym.py`.
4. Surveiller:
   - logs episode dans le terminal
   - image `rolling_ball_reward.png`

## 6) Comment juger une amelioration
- La moyenne lissee des rewards monte.
- La moyenne lissee des durees baisse.
- Moins de gros spikes negatifs au fil des episodes.
- Plus d'episodes reussis de maniere reguliere.

## 7) Probleme connus / diagnostic rapide
- Si imports non resolus (`torch`, `numpy`, `matplotlib`, `mlagents_envs`): verifier l'environnement Python et les packages installes.
- Si courbes tres bruitees: commencer par retoucher rewards et `eps_decay`, puis seulement `lr`/`batch_size`.
- Si Unity rame de plus en plus au fil des episodes: reduire `n_episode` pendant les tests, puis remonter pour la version finale.
- Si reward moyenne monte mais spikes restent grands: reduire l'aggressivite des rewards shaping avant de toucher au reseau.

## 7bis) Hypotheses de prochaine iteration
- Experience suivante potentielle: garder la base actuelle, puis tester une seule variable a la fois.
- Priorite d'analyse: rewards et epsilon avant d'augmenter la complexite du reseau.
- Lecture principale: une bonne courbe est une courbe qui monte sans spikes extremes trop frequents.

## 8) Journal des decisions recentes
- Retour volontaire vers une base proche du prof.
- Re-implementation stricte des TODO de l'exercice DQN.
- Modifications legeres seulement pour l'etape 3.
- Recompenses finalisees en echelle normalisee (`+1/-1`, timeout `-0.5`, shaping `+/-0.02`).
- Interpretation actuelle: l'agent apprend, mais le rapport doit montrer comment chaque test corrige un probleme observe sur la courbe ou sur le temps de simulation.
- Aucun resultat experimental final a consigner pour l'instant.

## 9) Checklist de reprise rapide
- [ ] Unity ouvert, scene correcte, mode Play actif
- [ ] Environnement Python selectionne avec dependances
- [ ] Parametres confirmes dans `rolling_ball_gym.py`
- [ ] Parametres confirmes dans `rolling_ball_DQN.py`
- [ ] Rewards confirmees dans `rollerAgentDiscrete.cs`
- [ ] Lancer un run complet et sauvegarder le plot
