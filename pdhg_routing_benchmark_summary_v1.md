# Lambert-Prox PDHG — benchmark statistique et routage V1

Statut : **PASS**

Ce document décrit des mesures sur l'environnement indiqué. Il ne constitue pas un claim SOTA ni une garantie inter-GPU.

## Environnement

- GPU : `Tesla T4`
- Compute capability : `7.5`
- PyTorch : `2.11.0+cu128`
- CUDA : `12.8`
- Triton : `3.6.0`

## Décisions

| Modèle | Taille | Torch ms | Élémentaire ms | Stencil ms | Route |
|---|---:|---:|---:|---:|---|
| gaussian | 128×192 | 126.838 | 99.712 | 54.016 | `triton_stencil` |
| poisson_intensity | 128×192 | 88.114 | 57.670 | 26.581 | `triton_stencil` |
| poisson_log | 128×192 | 98.115 | 22.041 | 12.162 | `triton_stencil` |
| kl | 128×192 | 96.196 | 21.432 | 12.050 | `triton_stencil` |
| xlogx | 128×192 | 101.033 | 22.810 | 12.646 | `triton_stencil` |
| gaussian | 512×512 | 91.502 | 74.754 | 39.474 | `triton_stencil` |
| poisson_intensity | 512×512 | 105.553 | 66.796 | 38.387 | `triton_stencil` |
| poisson_log | 512×512 | 76.566 | 18.753 | 10.670 | `triton_stencil` |
| kl | 512×512 | 104.333 | 23.142 | 13.163 | `triton_stencil` |
| xlogx | 512×512 | 99.807 | 22.850 | 13.419 | `triton_stencil` |

## Règle de promotion performance

Un backend accéléré est retenu seulement si la borne basse bootstrap à 95 % du gain apparié est au moins `1.10×`. Sinon, la route est `torch`.
