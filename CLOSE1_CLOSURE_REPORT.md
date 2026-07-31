# LAMBERT-PROX-CLOSE1 - rapport de fermeture finale T4

Date : 31 juillet 2026

## Verdict

CLOSE1 ferme les quatre défauts identifiés sans réécrire les preuves historiques. Les corrections mathématiques, le pack autonome et la relance Tesla T4 avec provenance pré-run/post-run sont réalisés. L'attestation retournée est liée octet pour octet aux huit sources figées du pack.

## Corrections

1. La norme de l'opérateur périodique est désormais calculée sur la grille finie :

   `||K||^2 = 4 sin^2(pi floor(W/2)/W) + 4 sin^2(pi floor(H/2)/H)`.

   `sqrt(8)` reste une borne uniforme sûre, pas une égalité générale.

2. Le solveur CLOSE1 sépare trois statuts :

   - `stabilized` : les changements relatifs des itérés ont franchi le seuil d'arrêt ;
   - `certified` : le gap et les résidus KKT de point fixe satisfont les tolérances numériques ;
   - convergence mathématique : théorème du PDHG exact, qui n'est pas identifié à lui seul à une exécution FP32 avec prox approchés.

3. Le diagnostic final calcule le primal, le dual, le gap de Fenchel, la violation de la boule duale et deux résidus de résolvante KKT normalisés.

4. Le domaine fermé de `x log x` est `x >= 0`, avec `0 log 0 = 0`.

5. Les modules Triton bas niveau restent des artefacts internes prévalidés. L'API publique passe par le solveur, qui valide les observations avant exécution ; aucun clamp silencieux n'est promu comme politique de domaine publique.

6. Le pack contient l'oracle NumPy, Torch, le PDHG canonique utilisé par la validation, les validateurs, les tests, les manifestes, les rapports et les données brutes de routage.

7. Le nouveau runner T4 écrit les SHA-256 avant l'import du validateur et les contrôle après le run. L'exécution du 31 juillet 2026 retourne `PASS`, 47/47 comparaisons et des cartes avant/après identiques.

8. Le digest de release T4 est `7e0757258621520664c48653650efdf5e13b6ed95773537dce40133172d2417f`. Il est recalculé localement à partir des mêmes huit sources.

## Statuts séparés

| Élément | Statut CLOSE1 |
|---|---|
| Correction de norme | Fermée et testée statiquement |
| Gap et résidus KKT | Implémentés avec tests dédiés |
| Vocabulaire de convergence | Corrigé dans l'API CLOSE1 |
| Pack autonome | Assemblé et vérifiable sans import GPU |
| Preuve T4 historique | Préservée, non réécrite |
| Relance T4 pré-run hashée | PASS, 47/47 sur Tesla T4 |
| Claim inter-GPU/SOTA | Interdit |

## Commandes normatives

```bash
python verify_close1_package.py
python -m pytest -q
python run_close1_validation.py
```

Sur un Colab Tesla T4 :

```python
%run run_cuda_validation_close1_colab.py
```

## Portée exacte de la fermeture

Le projet peut revendiquer une fermeture CPU/statique de CLOSE1 et une attestation d'exécution des 47 comparaisons CUDA sur la Tesla T4 enregistrée. Il ne peut pas extrapoler ce résultat à d'autres GPU, dtypes ou piles logicielles, ni le transformer en claim SOTA ou en preuve globale de convergence d'un PDHG inexact en arithmétique flottante.
