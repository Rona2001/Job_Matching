# Job_Matching
This project is build on Machine Learning to Match profiles.
2 Jeu de Donn ́ees
Pour ce projet, nous avons utilis ́e deux fichiers CSV extraits de la collec- tion de Firebase jobs :
— companies job.csv : Contient les informations sur les offres d’emploi des entreprises.
— candidates job.csv : Contient les informations sur les candidats `a la recherche d’emploi.
Ces jeux de donn ́ees incluent des champs tels que les comp ́etences tech- niques et non techniques, les valeurs, le type de contrat, l’exp ́erience, le niveau d’ ́etudes, et bien d’autres.
3 M ́ethodologie
4
La m ́ethodologie suivie dans ce projet comprend plusieurs  ́etapes cl ́es :
1. Encodage des attributs : Les caract ́eristiques des candidats et des offres sont encod ́ees en utilisant des mappings pr ́ed ́efinis pour les va- riables cat ́egorielles telles que le type de contrat, l’exp ́erience, le niveau d’ ́etudes, etc.
2. Remplissage des valeurs manquantes : Pour les entr ́ees man- quantes, nous avons attribu ́e des valeurs par d ́efaut afin de garantir l’int ́egrit ́e des donn ́ees.
3. Encodage des comp ́etences et des valeurs : Les comp ́etences non techniques et les valeurs sont encod ́ees sous forme de matrices binaires.
4. Vectorisation TF-IDF : Les comp ́etences techniques sont vecto- ris ́ees en utilisant la m ́ethode TF-IDF (Term Frequency-Inverse Do- cument Frequency).
5. Combinaison des caract ́eristiques : Toutes les caract ́eristiques en- cod ́ees sont combin ́ees en un seul DataFrame pour les candidats et les offres.
6. Calcul de la similarit ́e cosinus : La similarit ́e cosinus entre les ca- ract ́eristiques des candidats et des offres est calcul ́ee pour d ́eterminer les meilleures correspondances.

Vectorisation des Comp ́etences Techniques
Les comp ́etences techniques sont vectoris ́ees en utilisant la m ́ethode TF- IDF, qui permet de convertir des textes en vecteurs de caract ́eristiques num ́eriques
3
en tenant compte de l’importance relative des termes.
7 Calcul de la Similarit ́e Cosinus
Le calcul de la similarit ́e cosinus entre les vecteurs de caract ́eristiques des candidats et des offres permet d’ ́evaluer la pertinence des correspondances. La similarit ́e cosinus est d ́efinie comme le produit scalaire des vecteurs, normalis ́e par les normes des vecteurs.
Similarite Cosinus = A · B ∥A∥∥B∥
