# Artificial_Intelligence
Here, I will present my work in artificial intelligence.


Le maching Learning est un domaine de l'intelligence artificielle qui consiste a programmer une machine pour que celle ci apprenne a realiser des taches en etudiant des donnees de ces dernieres.Generalement la machine utilise ces donnees pour creer des modeles,par exemple une fonction du type f(x)=ax+ b.Le but est souvent de trouver le modele qui donne les meilleur parametres a et b pour avoir  le meilleur modele possible.Pour cela , on programme dans la machine un algorithme d'optimisation qui va venir tester differentes valeurs de a et b jusqu'a obtenir la combinaison qui minimise la distance entre le modele et les points.
Comme modele nous avons : Modele Lineaires (Descente de Gradient ),Abres de decision (Algorithme de CART) ,Support vecteur machines (Marge Maximun).

Le Deep Leaning est un domaine du machine leaning dans lequel au lieu de devellopper un des modeles cites plus haut , on devellope a la place de CNN : Reseaux de Neuronnes Artificieles.Ici on a plus juste une fonction du type f(x)=ax+b ,  ici on a une ensemble de fonction connecter les uns au autres.Plus ces reseaux sont profonds(c'est a dire plus ils contiennent des fonction a l'interieur) plus la machine peux apprendre les donnees complexes.
Intelligence artificielle >machinne learning>deep learning.

# Histoire:
- Les inventions des premiers neurones artificiels on ete invente en 1943 par les  mathematiciens  WARREN McCULLOCH et WALTER Pitts.Le systemes des neurrons convolutionelle est calque sur le fonctionnement des neuronnes du cerveau humain.(Se sont des cellules excitables conncetees les unes aux autres et ayant pour roles de transmettre des informations dans notre systeme nerveux).
Le but etais en fait de modeliser les neurrons huamines en considerant que une neuronnes pouvais etre considere comme fonction de transfert f qui prends en entree de signaux (x1,x2,...xn) et retourne  une sortir y.
A l'interieur de cette fonction on retrouve deux grande etape:
# L'etape d'agregation :on fait la sommes de tout les entrees du neuronnes en mutipliant chaque entree par un coeficient w (agregation f= w1*x1 + w2*x2+...wn*xn)
# L'etape d'activation : ou on regarde le resultat du calcul obtenu en agregation  si ce resultat est superieur a 0 alors le neuronnes s'active  et retourne une sortir y si non il reste a 0 .

- 1957 invention de Perceptron par Farnk Rosenblatt .La particularite de perceptron c'est qu'il dispos∈d'un algorithme d'apprentissage  qui permet d'obtenir les valeurs des parametres  w1,w2,...w3 afin de trouver la valeurs des sorties y qui nous convienne .En d'autres termes, le perceptrons consiste a entrainer un neurone artificiel sur des donnees de references (x,y) pour que celui ci renforce ses parametres W a chaque fois qu'une entre X est activee en meme temps que la sortir y presente dans ces donnees.En gros on a W=W+α(y'-y)X

- 1969 : XOR problem Misky et Paprt et de 1974  a 1980 c'est la periode de l'hiver de l'IA 

- 1980 : Invention du Perceptrons Multicouche par Geoffrey Hinton.Etant donnee que le perceptrons a lui seul est un modele lineaire,et que la plus part de probleme qui nouys entoure ne le sont pas le perceptrons c'est heurter a plusieurs difficultes.Les neuronnes multicouches regroupes plusieurs couche par exemple avec 3 neurones on peut mettre sur pieds deux couches,une couche d'entree constituer de 2 neurrones et une couche de sortir constituer d'un seul neurrone.
Pour trouver de facon optimale les valeurs de a et b afin d'obtenir un bon model , on utilise generalement la notion de Back-Propagation :consiste  a determiner comment la sortie du reseaux varie en fonction des parametres (W , b )presents dans chaque couche du model.Pour cela on utilise la formule de la descente de gradient.W=W-α(∂Erreur/∂W).
Pour developper des Reseaux de Neurones Artificiels , on a besoins de 4 etapes:
    - Forward propagation :on fais circule les donnees de la premieres couche jusqu'a la derniere afin de produire un dortir y.
    - Cost Function : calculer l'erreur entre cette sortir et la sortir de reference y que l'on desir avoir 
    - Back Propagation : On mesure comment cette fonction cout varie par rapport a chaque couche de notre modele a partant de la derniere vers la premiere.
    - Gradient Descent : qui consiste a corriger chaque parametres du modele grace a l'algorithme de la descente de gradient. 
    - Reboucle vers l'etape 1


# Le perceptron
C'est l'unite de base des reseaux de neurones.Il s'agit d'un modele de classification binaire ncapable de separer lineairement 2 classes de points.
si par exemple on veut classer deux type de plante,les plantes toxique et non toxique:En utilisant le perceptron simple , on obtient une courbe linaire qui represente la frontiere de decision.Si une pante est eloingnee de la frontiere de decision alors il est tres probable qu'elle soit en effect toxique.
Cette plante est proche de la frontiere de decision ,elle est a mis - chemin entre plante toxique et non toxique .

# La fonction d'activation sigmoide ou fonction logistique
son expression est a(z)=1/1-exp(-z):permet de convertir la sortir z en une probabilite a(z) celle que une plante appartient a la classe 1
Ex:si z=1.4,a(z)=0.8 donc notre plante a 80% d'appartenir a la classe 1.
# La fonction cout :
Est une fonction aui permet de quantifier les erreurs effectues par un modele.Nous utilisons la fonction Log Loss
# Descente de Gradient 
consiste a ajuster les parametres W et b de facon a minimiser les erreurs du  modele c'est a dire a minimiser la fonction cout (Log Loss).Pour cela on dois voir comment cette fonction varie en fonction des differents parametres.
Le gradient est alors la derive de la fonction cout .NB : L'algorithme de la descente de gradient n'est applicable que sur des fonction convexe qui ne contient qu'un suel minimum

# 03-07-2026
# La vectorisation
Cela consiste a mettre nos donnees dans les vecteurs des matrices ou des tableaux a N-dimension afin d'effectuer des operations  mathematique sur l'ensemble de ces donnees.Etant donnee que en machine learning on travaille sur de grosse quantite de donnes , sans la vectorisation on devrais travailler avec les iteration ce qui a un enjeux tres important sur la memoire.La vectorisation nous permet de gagner du temps memoire.
# La Normalisation 
C'est le fait de mettre tous les variables sur une meme echelle ,cela permet a la fonction cout d'evoluer de facon similaire sur tous ses parametres ce qui garantir un bonne convergence de l'algoritheme de Descente de Gradients.Mais si une variable est plus importante que l'autre (voir evolue 5 fois plus vite) alors la fonction de cout est compressee  car w2 a un grand impact sur la sortie A(Z).Cela complique la ocnvergence de la Descente de Gradients.
# Normalisation MinMax:
Mettre toutes nos variables dans la même échelle de valeurs, généralement entre 0 et 1.Cela permet d'éviter que certaines caractéristiques dominent le processus d'apprentissage en raison de leur échelle plus grande.
# Le Overfithing
C'est le fait pour un modele au lieu de generaliser l'apprentissage se met juste a memoriser.Cela se caracterise par de bonne performence au niveua des donnees du train mais des performance tres mauvaise au niveau des donnees test.Pour pallier a ce probleme , on dois : Fournir plus de donnees a la machine lors de l'entrenement pour garantir une etude appronfondir des caracteristique des object present dans le dataset train.
Lorsque on a un grand decallage entre le nombre de photos et le nombre de variable , on obtient un  phenomene appeler le fleaux de la dimension qui fait que l'espace dans lequel se trouve nos donnees est un espace principalement remplir de vide dans lequel le model peut se balader comme il veut et trouver la configuration qu'il veut et obtenir un bon score sur le train au detriment sur le test.
# NB 
Le fléau de la dimension (en anglais Curse of Dimensionality) est l'un des concepts les plus importants en Machine Learning, Data Science et Intelligence Artificielle. Il désigne l'ensemble des problèmes qui apparaissent lorsque le nombre de variables (dimensions ou features) d'un jeu de données devient très élevé.Avec peu d'observations et beaucoup de variables, les exemples sont très éloignés les uns des autres.
Le modèle dispose donc de peu d'informations locales pour apprendre.Lorsque le nombre de variables dépasse largement le nombre d'exemples, le modèle peut apprendre les particularités du jeu d'entraînement au lieu des relations générales.

# Reseaux de neurones convolutionnel
Dans une couche , on peut ajouter autant de neurones possible,etant donner que chaque neurone a ces parametres,leurs fonctionnemet serais donc independant des un des autres ce qui fait que plus on a des neuronnes plus le model est performent mais aussi lent pour entrainer.
1. On note Wij le parametre associe au neurone i et provenant de l'entree j 
2. On note [c] le numero de la couche C sur laquelle on travaille :Z[1],Z[2]...
3. Pour calculer les valeurs z d'une couche [c] on utilise les activations a de la couche [c-1]
Z1[2]=W11[2]a1[1]+W12[2]a2[1]+b[2]

# Back propagation 
Consiste a retracer comment la fonction Cout evolue de la derniere couche du reseau jusqu'a la toute premiere.

# Not :
Il est toujours important lorsque l'on travaille avec les images de verifier si les dimension des images test sont les memes que les dimensions des donnees train.

# Broadcasting
est un mécanisme de NumPy qui permet d'effectuer des opérations entre des tableaux (ndarray) de tailles différentes, sans avoir à les redimensionner manuellement.