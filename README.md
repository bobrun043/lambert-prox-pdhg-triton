# Lambert-Prox PDHG — Auto-Routed V3

Ce pack ajoute un backend `auto` au solveur PDHG canonique promu.

## Contrat

- Le routeur accéléré ne s'active que pour une clé exacte GPU/logiciels présente dans la table.
- Le pack actuel contient uniquement la preuve statistique Tesla T4.
- CUDA FP32 est obligatoire pour Triton.
- Hors domaine de taille, d'aspect, de modèle, de dtype ou d'environnement, la route est Torch.
- Un fallback du mode `auto` est toujours visible dans `AutoRoutingReport`.
- Une demande Triton explicite ne fallback jamais.

## Utilisation

```python
from pdhg_auto_routed_v3 import pdhg_auto

x, dual, info, route = pdhg_auto(
    "kl", x0, y_obs, lam_tv=0.03,
    backend="auto",
    max_iter=500,
)
print(route.to_dict())
```

## Tests

```bash
python test_routing_policy_v2.py
python test_pdhg_auto_routed_v3_cpu.py
```

Dans Colab T4 :

```python
%run run_auto_routing_smoke_colab.py
```
