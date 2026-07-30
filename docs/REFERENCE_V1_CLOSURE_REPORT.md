# Rapport de fermeture - Référence FP64 canonique V1

- Statut global : **PASS**
- Points du balayage inverse : 30004
- Résidu coordonnée max : 1.137e-13
- Résidu coordonnée p99 : 1.776e-15
- Itérations max / moyenne : 4 / 2.738
- Statuts log-only : 1285
- Erreur max q contre mpmath : 2.220e-16
- Erreur relative max u contre mpmath : 5.421e-20

## Résidus KKT normalisés maximaux

- `exp` : 1.717e-15
- `xlogx` : 6.267e-14
- `kl` : 9.577e-14
- `poisson_log` : 1.035e-15
- `poisson_intensity` : 3.132e-16
- `neglog` : 2.947e-16
- `gaussian` : 5.930e-15

## Statut exact de cette étape

Démontré dans la spécification : unicité de l'inverse et réductions proximales.

Vérifié ici : oracle NumPy FP64, contrat bi-coordonnées, cas limites et KKT sur les domaines consignés.

Non vérifié ici : backend Torch, double backward, Triton, CUDA, performance et originalité bibliographique.
