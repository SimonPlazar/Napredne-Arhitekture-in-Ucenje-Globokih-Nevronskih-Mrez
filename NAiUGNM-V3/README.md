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

Implementirajte CycleGAN, ki omogoča prenos lastnosti ene slike na drugo sliko. Uporabite podatkovno zbirko Apple2orange. V postopku nasprotniškega učenja boste potrebovali:

Generator, ki preslika jabolka v pomaranče (G_X2Y)
Generator, ki preslika pomaranče v jabolka (G_Y2X)
Diskriminator, ki razločuje med resničnimi in generiranimi jabolki (D_X)
Diskriminator, ki razločuje med resničnimi in generiranimi pomarančami (D_Y)
Med učenjem uporabljajte več funkcij izgube:

Funkcije izgube GAN, ki pripomorejo k temu, da generirane slike izgledajo bolj resnične
Funkciji izgube cikličnosti (ang. cycle loss), ki pripomoreta k obratnim preslikavam (X2Y in Y2X)
Funkciji izgube identitete (ang. identity loss), ki pripomoreta k ohranjanju stabilnosti vsebine slike (npr. barv)
Funkcija izgube GAN naj bo povprečna kvadratna napaka (MSE), ostali dve funkciji izgube pa naj bosta L1 (oz. MAE). Skupna funkcija izgube za generator naj bo seštevek funkcij izgub obeh generatorjev, obeh funkcij izgub cikličnosti in obeh funkcij izgub identitete. Skupna funkcija izgube za diskriminator naj bo podobna kot pri DCGAN. Učenje naj traja vsaj 15 epoh. Po vsaki epohi shanite sliko parov slik, da boste na koncu lahko prikazali postopek učenja z animacijo GIF. Prikažite tudi grafa funkcij izgub za generator in diskriminator.

# V3.3

Implementacijo CycleGAN iz V3.2 naučite na podatkovni zbirki Summer2Winter Yosemite. Učenje naj traja vsaj 15 epoh. Po vsaki epohi shanite sliko parov slik, da boste na koncu lahko prikazali postopek učenja z animacijo GIF. Prikažite tudi grafa funkcij izgub za generator in diskriminator. Primerjajte rezultate s slikami jabolk in pomaranč iz V3.2 in podajte komentar ter ugotovitve.

# V3.4

Implementirajte model za prenos stila iz ene slike na drugo (ang. neural style transfer). Za zgled uporabite članek.  Uporabite obstoječi model za slike VGG19 in ustrezno implementirajte funkciji izgube vsebine (ang. content loss) in stila (ang. style loss). Pri optimizaciji parametrov uporabite gradientni spust, lahko pa preizkusite tudi L-BFGS za boljše rezultate. Uporabite priložene slike in pokažite postopen prenos stila po 50, 100, 150, 200, 250 in 300 korakih.

# V3.5

Preizkusite in pokažite rezultate prenosa stila na generiranih slikah, ki ste jih ustvarili v V3.1, V3.2 in V3.3. Uporabite priložene slike za stil in pokažite postopen prenos stila po 50, 100, 150, 200, 250 in 300 korakih na vsaj 3 slikah iz vsakega izmed sklopov V3.1, V3.2 in V3.3.