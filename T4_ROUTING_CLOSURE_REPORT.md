# Fermeture du routage automatique T4 — V3

Statut : **PASS**

## Résultat

Les dix couples modèle–taille testés sélectionnent `triton_stencil`. La décision est fondée sur des mesures appariées et une borne basse bootstrap à 95 % supérieure à 1,10×.

| Modèle | Taille | Gain médian | IC95 bas | IC95 haut | Pic incrémental Torch / stencil |
|---|---:|---:|---:|---:|---:|
| gaussian | 128×192 | 2.33× | 2.27× | 2.64× | 1.00× |
| poisson_intensity | 128×192 | 2.85× | 2.33× | 3.05× | 1.19× |
| poisson_log | 128×192 | 8.19× | 7.56× | 8.46× | 3.56× |
| kl | 128×192 | 7.97× | 6.80× | 8.39× | 3.58× |
| xlogx | 128×192 | 8.15× | 7.72× | 8.38× | 3.56× |
| gaussian | 512×512 | 2.33× | 2.22× | 2.38× | 1.00× |
| poisson_intensity | 512×512 | 2.68× | 2.50× | 2.78× | 1.19× |
| poisson_log | 512×512 | 7.18× | 6.89× | 7.51× | 3.56× |
| kl | 512×512 | 7.92× | 7.47× | 8.49× | 3.58× |
| xlogx | 512×512 | 7.31× | 6.60× | 7.67× | 3.56× |

## Claim autorisé

**Sur Tesla T4, PyTorch 2.11.0+cu128, CUDA 12.8 et Triton 3.6.0, pour les deux géométries testées et les cinq modèles considérés, le backend stencil fusionné est plus rapide que le backend Torch avec une borne basse bootstrap à 95 % supérieure à 1,10×, tout en reproduisant l’oracle dans la tolérance validée.**

## Non-claims

- Aucun claim SOTA ou inter-GPU.
- Aucune garantie pour FP64, AMP, autograd Triton, batch/canaux différents ou conditions de bord non périodiques.
- Les pics mémoire sont des pics incrémentaux de l’allocateur PyTorch, pas la mémoire totale du procédé.
- Les tailles intermédiaires sont routées par proximité de surface avec gardes de distance et d’aspect ; hors domaine, Torch est choisi.

## Politique de production

- environnement logiciel/GPU exact ;
- CUDA FP32 seulement ;
- surface au plus 4× de la géométrie testée la plus proche ;
- facteur d’aspect au plus 2× ;
- preuve bootstrap toujours au-dessus du seuil de 1,10× ;
- fallback Torch explicite et enregistré en mode `auto` ;
- une demande Triton explicite ne fallback jamais.
