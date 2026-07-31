# Rapport de fermeture — intégration PDHG canonique V1

Date : 2026-07-30

## 1. Statut

Le backend `lambert_prox_torch_v1.py` est désormais la seule source des prox dans
`pdhg_canonical_v1.py`. Le chemin Torch est fermé sur CPU. Aucun ancien module
`solve_u_log_u.py` ou `torch_lambert_prox_v3.py` n'est importé.

**Statut Torch : VALIDÉ SUR CPU.**

**Statut Triton/CUDA : NOT_RUN.** L'environnement d'exécution contient PyTorch
`2.10.0+cpu` sans CUDA et sans Triton. Ce statut n'est pas un PASS.

## 2. Contrats obtenus

- opérateur périodique `K=grad` et `K^T=-div` ;
- norme exacte utilisée : `sqrt(8)` ;
- projection TV isotrope point par point ;
- sept prox routés vers le backend canonique ;
- sélection de backend explicite ;
- aucune retombée silencieuse de Triton vers Torch ;
- arrêt sur changements relatifs primal **et** dual ;
- journal indiquant le backend demandé et réellement exécuté ;
- vues non contiguës testées sur le chemin Torch.

## 3. Tests exécutés

- `.........................                                                [100%]
25 passed in 1.60s`
- gap d'adjoint relatif FP64 : `2.402350e-15`

| Modèle bien posé | Itérations | Convergence | Objectif initial | Objectif final | Baisse |
|---|---:|:---:|---:|---:|---:|
| gaussian | 151 | oui | 230.192391 | 51.0978822 | 179.094509 |
| poisson_intensity | 555 | oui | 2073.5594 | 2059.95745 | 13.6019495 |
| poisson_log | 149 | oui | 2086.50113 | 2066.65032 | 19.8508092 |
| kl | 361 | oui | 133.284433 | 52.911435 | 80.3729978 |
| xlogx | 40 | oui | 0 | -779.904415 | 779.904415 |

Les cinq exécutions sont finies, déterministes, utilisent effectivement le backend
`torch` et diminuent l'objectif entre l'initialisation et la sortie.

## 4. Correction de statut mathématique des anciennes démonstrations

Un prox valide ne garantit pas que le problème global `F + lambda TV` possède un
minimiseur.

- `exp(x) + lambda TV(x)` : l'infimum vaut 0 le long des constantes tendant vers
  `-infinity`, mais il n'est pas atteint.
- `-log(x) + lambda TV(x)` : la fonctionnelle tend vers `-infinity` le long des
  constantes positives tendant vers `+infinity`.
- `exp(x)-y x + lambda TV(x)` : la version V1 exige `y>0` pour le modèle PDHG.

Le solveur canonique refuse donc par défaut les deux premiers modèles autonomes.
Les prox restent disponibles comme briques dans des objectifs comportant d'autres
termes coercifs.

## 5. Audit statique des candidats Triton hérités

L'audit ne remplace pas une exécution CUDA. Il identifie néanmoins quatre écarts :

1. Halley à quatre passages en coordonnée valeur, contre six passages
   bi-coordonnées dans le canon ;
2. reconstruction `v-u` / `d-u` pour `exp` et Poisson log, au lieu de
   `q-log(lambda)` ;
3. KL avec plancher artificiel de `y=0`, au lieu de la frontière exacte ;
4. absence du mode `poisson_log` dans le stencil fusionné.

Le passage explicite des strides et l'interdiction de l'in-place sont visibles dans
le candidat stencil, mais ne sont pas validés ici faute de CUDA.

## 6. Interprétation

Cette fermeture valide l'architecture mathématique et logicielle du chemin Torch.
Elle ne valide pas :

- les noyaux Triton ;
- CUDA ;
- les performances ;
- un speedup ;
- AMP, FP16 ou BF16 ;
- un claim SOTA.

La prochaine étape est une exécution du fichier
`triton_oracle_validation_v1.py` sur CUDA/Triton. Tout écart doit être corrigé
avant de reconnecter un noyau fusionné au solveur de production.
