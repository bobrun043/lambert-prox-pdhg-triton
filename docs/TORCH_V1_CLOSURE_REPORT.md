# Rapport de fermeture — backend Torch canonique V1

## Verdict

**PASS sur CPU : 12 tests sur 12.**

Le backend Torch unique reproduit l’oracle NumPy FP64, fournit un autograd natif de premier et second ordre, et remplace les formules proximales dupliquées par une seule source numérique.

## Environnement exécuté

- Python : `3.13.5`
- PyTorch : `2.10.0+cpu`
- CUDA disponible : `False`
- Device effectivement testé : `cpu`

## Primitive bi-coordonnées

### FP64

- points : 110008
- convergence : True
- cas `OK_LOG_ONLY` : 2
- résidu absolu maximal : 9.095e-13
- résidu normalisé maximal : 3.331e-16
- erreur relative maximale de `q` contre l’oracle : 4.432e-16

### FP32

- points : 110008
- convergence : True
- cas `OK_LOG_ONLY` : 2
- résidu absolu maximal : 1.953e-03
- résidu normalisé maximal : 1.788e-07
- erreur relative maximale de `q` contre l’oracle arrondi FP32 : 1.541e-07

Le résidu absolu FP32 maximal apparaît aux grandes échelles de `R`; le résidu normalisé reste au niveau de l’arrondi FP32.

## KKT normalisés maximaux

| Prox | FP64 | FP32 |
|---|---:|---:|
| `exp` | 8.257e-16 | 6.549e-07 |
| `xlogx` | 5.934e-14 | 3.710e-05 |
| `kl` | 5.040e-14 | 4.220e-05 |
| `poisson_log` | 7.928e-16 | 4.839e-07 |
| `poisson_intensity` | 3.195e-16 | 1.830e-07 |
| `neglog` | 3.730e-16 | 1.914e-07 |
| `gaussian` | 4.930e-15 | 2.930e-06 |


Les deux plus grands résidus FP32 concernent `xlogx` et KL. Ils sont évalués avec les coordonnées logarithmiques canoniques, sans transformer un sous-débordement de valeur en faux échec.

## Autograd

Vérifié :

- `gradcheck` de la primitive en coordonnées `u` et `q` ;
- `gradgradcheck` de la primitive ;
- `gradcheck` et `gradgradcheck` des sept prox par rapport à `v` ;
- dérivées implicites exactes :
  
  \[
  rac{{du}}{{dR}}=rac{{u}}{{1+u}},\qquad
  rac{{dq}}{{dR}}=rac{{1}}{{1+u}};
  \]
- dérivées secondes :
  
  \[
  rac{{d^2u}}{{dR^2}}=rac{{u}}{{(1+u)^3}},\qquad
  rac{{d^2q}}{{dR^2}}=-rac{{u}}{{(1+u)^3}}.
  \]

Aucun `torch.autograd.Function` opaque n’est utilisé : le graphe reste accessible aux dérivées secondes.

## Correction structurante ajoutée pendant la validation

Les prox `xlogx` et KL exposent désormais aussi :

- `prox_xlogx_log` ;
- `prox_kl_log`.

Ces fonctions conservent `log(x)` lorsque la valeur positive `x` sous-déborde. Les KKT extrêmes doivent utiliser cette coordonnée plutôt que `torch.log(x)`.

## Statut exact

### Fermé

- backend Torch numérique unique ;
- équivalence FP64 avec l’oracle ;
- contrat FP32 mesuré ;
- KKT des sept prox ;
- premier et second ordre autograd ;
- cas limites de domaine et racines quadratiques stables ;
- adaptateur de migration sans duplication de formules.

### Non fermé

- CUDA ;
- Triton ;
- AMP, FP16 et bfloat16 ;
- `torch.compile` ;
- performance ;
- intégration effective dans le solveur PDHG ;
- suppression définitive des anciens modules.

Aucun claim de vitesse, de SOTA ou d’originalité n’est attaché à cette étape.
