# Rapport de fermeture — préparation Benchmark & Routing V1

## Statut

**Infrastructure préparée et testée statiquement. Mesures CUDA statistiques : NOT_RUN dans l'environnement de construction.**

## Éléments livrés

- benchmark sans argument, avec checkpoints après chaque cas ;
- contrôle de correction avant chronométrage ;
- exclusion des premiers appels JIT des échantillons ;
- calibration automatique du nombre d'itérations ;
- répétitions appariées et ordre randomisé ;
- bootstrap du gain contre Torch ;
- mesure de mémoire GPU ;
- production automatique d'une table de routage ;
- politique runtime conservatrice ;
- fusion de rapports T4, A100 et RTX 3060.

## Règle normative

Aucun chemin Triton n'est routé par défaut sur la seule base d'une médiane ponctuelle. La borne basse bootstrap à 95 % doit établir un gain d'au moins 10 % contre Torch. À défaut, Torch reste la route.

## Statut des données antérieures

Les temps de la régression T4 déjà obtenus constituent un signal exploratoire. Ils ne satisfont pas ce nouveau protocole statistique, car ils proviennent d'une seule mesure par backend et par modèle.

## Claims autorisés après exécution

- « Sur le GPU et les versions logicielles enregistrés, le backend sélectionné minimise la médiane observée selon le protocole V1. »
- « La décision accélérée dépasse Torch d'au moins 10 % avec une borne basse bootstrap à 95 % supérieure ou égale à 1,10. »

## Claims non autorisés

- SOTA ;
- backend universellement plus rapide ;
- extrapolation à un autre GPU ou à une autre version logicielle ;
- efficacité énergétique ;
- performance AMP ou backward.
