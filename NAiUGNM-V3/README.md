# V3 - Nasprotniške nevronske mreže

Na vajah boste spoznali nasprotniške nevronske mreže (ang. generative adversarial network - GAN) in njihove različne uporabe. Implementirali boste globoko konvolucijsko generativno nasprotniško nevronsko mrežo (DCGAN), njeno različico CycleGAN in pristop za prenos stila ene slike na drugo (ang. neural style transfer). Zasnovali boste generator in diskriminator, ki ju boste učili z nasprotniškim učenjem.

Postopek učenja skozi čas boste prikazali z animacijo GIF. Vajo implementirajte s programskim jezikom Python in orodjem Jupyter Notebook.

Vaja je sestavljena iz 5 sklopov, ki se ocenjujejo posebej in se med seboj dopolnjujejo.

Točkovanje:

Sklop V3.1 - 50 točk
Sklop V3.2 - 10 točk
Sklop V3.3 - 10 točk
Sklop V3.4 - 15 točk
Sklop V3.5 - 15 točk

# V3.1

Implementirajte globoko konvolucijsko generativno nasprotniško nevronsko mrežo (DCGAN), s katero boste generirali človeške obraze. Uporabite priloženo podatkovno zbirko obrazov, ki je bila ustvarjena iz slik generiranih obrazov (https://thispersondoesnotexist.com). Vaša implementacija naj zajema generator in diskriminator, ki ju boste učili z nasprotniškim učenjem. Generator naj bo sestavljen iz več blokov različnih plasti, kot je prikazano na sliki arhitekture spodaj.

Vhodni šum naj bo dimenzije 100, osnovno število filtrov za konvolucijske plasti pa naj bo 64. Slike so barvne, zato bo število kanalov enako 3. Posameznim blokom generatorja ustrezno nastavite število vhodnih in izhodnih kanalov in ostalih parametrov konvolucijske plasti (kernel, stride, padding).

Diskriminator bo klasifikator, ki bo odločal o tem ali je slika generirana ali ne. Njegova struktura naj bo prav tako sestavljena iz več blokov različnih plasti, kot je prikazano na sliki spodaj.

Podobno kot pri generatorju, ustrezno nastavite dimenzije posameznih plasti v blokih.

Učenje naj poteka v več paketih (ang. batches) tako, da najprej z generatorjem ustvarite sličice. Nato paket iz učne množice sličic iz podatkovne zbirke pošljite v diskriminator. Takoj zatem v diskriminator pošljite še paket generiranih sličic. Na koncu izračunajte funkciji izgube generatorja in diskriminatorja. Funkcija izgube generatorja naj bo binarna križna entropija (ang. binary crossentropy - BCE). Za funkcijo izgube diskriminatorja prav tako uporabite binarno križno entropijo, ki je seštevek vrednosti funkcije izgube za resnične in generirane primere. Učenje naj traja vsaj 50 epoh, priporočljivo pa je učiti vsaj 150 epoh. Dlje bo trajalo učenje, boljša bo kvaliteta generiranih sličic. Pri učenju uporabite vektor šuma, ki bo obsegal 64 sličic. Po vsaki epohi shanite sliko 64 sličic, da boste na koncu lahko prikazali postopek učenja z animacijo GIF. Prikažite tudi grafa funkcij izgub za generator in diskriminator.

# V3.2



# V3.3



# V3.4



# V3.5