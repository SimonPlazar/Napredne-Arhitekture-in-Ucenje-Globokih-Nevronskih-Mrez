# V2 - Samokodirniki

Na vajah boste spoznali in implementirali različne arhitekture samokodirnikov. Skozi sklope boste implementirali različne tipe samokodirnikov in jih preizkusili na podatkovnih zbirkah MNIST, FashionMNIST in CIFAR10. Rezultate pokažite na primerih rekonstrukcij slik s primerjavo z izvornimi slikami. Vaše ugotovitve podajte na pregleden in jasen način (uporabite slike, tabele, po potrebi anotirajte slike z oznakami, ipd.). Vajo implementirajte s programskim jezikom Python in orodjem Jupyter Notebook.

Vaja je sestavljena iz 5 sklopov, ki se ocenjujejo posebej in se med seboj dopolnjujejo.

Točkovanje:

Sklop V2.1 - 50 točk
Sklop V2.2 - 10 točk
Sklop V2.3 - 10 točk
Sklop V2.4 - 15 točk
Sklop V2.5 - 15 točk

Oddajte datoteko .zip z vso programsko kodo (datoteki .ipynb in .py). Datoteko poimenujte po predlogi: V2.1-[ime]-[priimek].zip (primer: V2.1-Janez-Novak.zip).

## V2.1

Implementirajte osnovni samokodirnik in ga naučite s ciljem rekonstrukcije slik v podatkovni zbirki MNIST. Plast kodirnika naj preslika vhod dimenzije 28x28 na dimenzijo 256. Latentni prostor naj bo dimenzije 32. Za funkcijo izgube uporabite povprečno kvadratno napako (ang. mean squared error). Podajte primerjavo nekaj primerov rekonstruiranih slik in njihovih izvornikov. Priložite graf funkcije izgube med učenjem za učno in validacijsko množico.

## V2.2

Implementirajte globoki samokodirnik (ang. deep autodencoder) in ga naučite s ciljem rekonstrukcije slik v podatkovni zbirki MNIST. Kodirnik in dekodirnik naj imata dve plasti. Pri kodirniku naj bo prva plast dimenzije 512, druga pa 256, pri dekodirniku pa obratno. Latentni prostor naj bo dimenzije 32. Za funkcijo izgube uporabite povprečno kvadratno napako (ang. mean squared error). Podajte primerjavo nekaj primerov rekonstruiranih slik in njihovih izvornikov. Dodajte tudi primere rekonstruiranih slik iz sklopa V2.1. Priložite graf funkcije izgube med učenjem za učno in validacijsko množico.

## V2.3

Implementirajte ß-variacijski samokodirnik (ang. ß-variational autoencoder) in ga naučite s ciljem rekonstrukcije slik v podatkovni zbirki MNIST. Ustrezno implementirajte plasti s srednjimi vrednostmi mu in standardnih odklonov sigma. Vhod se naj najprej preslika v dimenzijo 512, latentni prostor pa naj bo dimenzije 32. Za funkcijo izgube uporabite posebno funkcijo, ki združuje funkcijo izgube za rekonstrukcijo (povprečna kvadratna napaka) in Kullback-Lieblerjevo divergenco. V izračun funkcije izgube vključite parameter ß, ki ga privzeto nastavite na vrednost 1.0. Podajte primerjavo nekaj primerov rekonstruiranih slik in njihovih izvornikov. Dodajte tudi primere rekonstruiranih slik iz sklopov V2.1 in V2.2. Priložite graf funkcije izgube med učenjem za učno in validacijsko množico.

## V2.4

Obstoječe implementacije sklopov V2.1, V2.2 in V2.3 dopolnite z logiko predčasnega ustavljanja učenja (ang. early-stopping), kjer se učenje prekine, če se v zadnjih 5 epohah vrednost funkcije izgube za validacijo ne spremeni. Zaženite vse implementirane samokodirnike na podatkovnih zbirkah FashionMNIST in CIFAR10 ter podajte primerjavo rekonstruiranih slik in njihovih izvornikov na nekaj primerih. Bodite pozorni na dimenzije vhoda pri podatkovni zbirki CIFAR10! Latentni prostor naj bo v tem primeru dimenzije 64, pri dimenziji vhoda pa upoštevajte dimenzije slik v tej podatkovni zbirki. Podajte komentar na dobljene rezultate in jih utemeljite.

## V2.5

Na podatkovnih zbirkah MNIST, FashionMNIST in CIFAR10 preizkusite različne vrednosti parametra ß za ß-variacijski samokodirnik. Podajte primere rekonstruiranih slik in komentar ter ugotovitve kako parameter ß vpliva na rekonstrukcijo. Priložite grafe funkcije izgube med učenjem za učno in validacijsko množico.