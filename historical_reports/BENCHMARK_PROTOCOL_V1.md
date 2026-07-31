# Protocole de performance Lambert-Prox PDHG V1

## Objet

Mesurer, sur un environnement GPU explicitement identifié, les trois chemins :

- `torch` ;
- `triton_elementwise` ;
- `triton_stencil`.

Le résultat attendu est une table locale

\[
(\text{GPU},\text{logiciels},\text{modèle},H,W)\mapsto\text{backend}.
\]

Il ne s'agit ni d'un benchmark SOTA, ni d'une extrapolation inter-GPU.

## Périmètre

Cinq modèles PDHG bien posés : Gaussian, Poisson intensité, Poisson log-intensité, KL et \(x\log x\). Les modèles autonomes `exp + TV` et `-log + TV` restent exclus.

## Contrôles avant chronométrage

1. Vérification SHA-256 du pack promu.
2. Comparaison des sorties primales et duales contre Torch après 20 itérations.
3. Seuil : erreur maximale mise à l'échelle \(\le 8\times10^{-5}\).
4. Premier appel JIT observé et enregistré, mais exclu des échantillons mesurés.

## Mesure

- synchronisation CUDA avant et après chaque échantillon ;
- nombre d'itérations fixe pour les trois backends d'un même cas ;
- calibration sur Torch pour viser environ 150 ms par échantillon ;
- 2 warmups par défaut ;
- 15 répétitions appariées par défaut ;
- ordre des trois backends randomisé à chaque répétition ;
- médiane, MAD, percentiles 10/25/75/90 ;
- latence par itération et par pixel-itération ;
- mémoire GPU incrémentale maximale.

## Décision de routage

Un backend accéléré est sélectionné seulement lorsque :

1. il passe le contrôle de correction ;
2. sa médiane est inférieure à Torch ;
3. la borne basse bootstrap à 95 % du gain apparié est au moins `1.10x`.

Dans tous les autres cas, la route est `torch`.

Si les deux chemins Triton sont à moins de 3 %, le protocole préfère le plus faible pic mémoire, puis `triton_elementwise` en cas d'égalité.

## Interpolation

La politique d'exécution exige une correspondance exacte de GPU, compute capability, CUDA et PyTorch. Pour une taille non mesurée, elle prend la taille testée la plus proche en nombre de pixels seulement si le rapport d'aires est inférieur ou égal à 4. Sinon, elle revient à Torch.

## Limites

- fréquence, température et charge du GPU restent des variables expérimentales ;
- la compilation froide est enregistrée comme « premier appel observé », sans prétendre mesurer un cache vierge ;
- les résultats ne valent que pour FP32 forward et le code lié par les empreintes du rapport ;
- AMP, backward Triton et autres GPU demandent une campagne distincte.
