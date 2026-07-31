# Lecture du pilote T4 — non statistique

Ce bilan reprend la régression multi-itérations déjà exécutée sur Tesla T4. Il contient une seule durée par backend et par modèle ; il ne satisfait donc pas le protocole statistique V1 et ne doit pas alimenter directement le routeur.

| Modèle | Torch (s) | Triton élémentaire (s) | Gain élém. | Triton stencil (s) | Gain stencil |
|---|---:|---:|---:|---:|---:|
| Gaussian | 0,2635 | 0,2684 | 0,98× | 0,8726 | 0,30× |
| Poisson intensité | 0,7131 | 1,0209 | 0,70× | 0,6615 | 1,08× |
| Poisson log | 0,9912 | 0,3947 | 2,51× | 0,4276 | 2,32× |
| KL | 0,8829 | 0,4051 | 2,18× | 0,4391 | 2,01× |
| xlogx | 0,4580 | 0,3128 | 1,46× | 0,3781 | 1,21× |

## Hypothèses à tester

- Gaussian devrait rester routé vers Torch.
- Poisson intensité pourrait basculer vers stencil seulement au-delà d'une taille seuil.
- Poisson log et KL montrent les signaux les plus nets en faveur de Triton.
- xlogx pourrait préférer le chemin élémentaire.

Ces phrases sont des hypothèses de campagne, pas des décisions de production. La table définitive exige les répétitions appariées, le bootstrap et les tailles multiples du protocole V1.
