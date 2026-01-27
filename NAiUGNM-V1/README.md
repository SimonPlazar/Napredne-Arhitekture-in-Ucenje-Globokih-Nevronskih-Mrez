# V1 - Prekomerno prileganje in regularizacija

Na vajah se boste spoznali z načrtovanjem globokih nevronskih mrež in pogostim nezaželenim pojavom - prekomernim prileganjem. Začetno nevronsko mrežo boste z različnimi pristopi poskušali izboljšati tako, da se boste izognili prekomernemu prileganju. Vaša nevronska mreža bo izvedla analizo sentimenta (ang. sentiment analysis) nad filmskimi recenzijami IMDB. Podatkovna zbirka je na voljo v ogrodjih Tensorflow/Keras in PyTorch. Rezultat boste prikazali z izrisom grafov funkcije izgube med učenjem in validacijo kot tudi z izrisom grafa metrike točnosti. Na podlagi teh grafov boste komentirali vsak poskus in poročali o vaših ugotovitvah. Vaše ugotovitve podajte na pregleden in jasen način (uporabite slike, tabele, po potrebi anotirajte slike z oznakami, ipd.). Vajo implementirajte s programskim jezikom Python in orodjem Jupyter Notebook.

Vaja je sestavljena iz 5 sklopov, ki se ocenjujejo posebej in se med seboj dopolnjujejo.

Točkovanje:
Sklop V1.1 - 50 točk
Sklop V1.2 - 10 točk
Sklop V1.3 - 10 točk
Sklop V1.4 - 15 točk
Sklop V1.5 - 15 točk

Oddajte datoteko .zip z vso programsko kodo (datoteki .ipynb in .py). Datoteko poimenujte po predlogi: V1.1-[ime]-[priimek].zip (primer: V1.1-Janez-Novak.zip).

## V1.1

Naložite in izvedite predobdelavo podatkovne zbirke IMDB, ki jo najdete v ogrodju Tensorflow/Keras (keras.datasets) in je že razdeljena na učno in testno množico. Predobdelavo izvedite z vektorizacijo besed tako, da pripravite matriko z enicami, na ustreznih indeksih za pripadajočo besedo. Upoštevajte najpogostejših 3000 besed. Implementirajte začetno nevronsko mrežo, ki ima na vhodni in skriti plasti 16 nevronov in aktivacijsko funkcijo ReLU. Izhodna plast naj ima en nevron in sigmoidno aktivacijsko funkcijo. Uporabite optimizacijski algoritem Adam in kot funkcijo izgube uporabite binarno križno entropijo (ang. binary cross entropy - BCE). Naučite model in izrišite grafa funkcije izgube med učenjem in validacijo. Izrišite tudi graf metrike točnosti, ki ga uporabite pri odgovarjanju naslednjih vprašanj. Dobro poglejte kaj se dogaja na grafih in komentirajte stanje. Ali gre za prekomerno prileganje ali ne? Če gre za prekomerno prileganje, pri kateri epohi se to začne?

## V1.2

Ponovite učenje z nevronsko mrežo z manj nevroni (4 namesto 16). Na enak način kot prej izrišite grafe funkcij izgub in točnosti ter podajte komentar in ugotovitve. Kaj se je spremenilo s tem, ko ste zmanjšali število nevronov? Ali je to pripomoglo k reševanju prekomernega prileganja?

## V1.3

Ponovite učenje z nevronsko mrežo z več nevroni (512 namesto 16). Na enak način kot prej izrišite grafe funkcij izgub in točnosti ter podajte komentar in ugotovitve. Kaj se je spremenilo s tem, ko ste povečali število nevronov? Ali je to pripomoglo k reševanju prekomernega prileganja?

## V1.4

Ponovite učenje vseh treh arhitektur nevronskih mrež (4, 16 in 512 nevronov) z dodano regularizacijo L2. Na enak način kot prej izrišite grafe funkcij izgub in točnosti, ter podajte komentar in ugotovitve. Nato ponovite učenje vseh treh arhitektur nevronskih mrež z dodano plastjo izpusta. Primerjajte grafe funkcij izgub in točnosti s tistimi iz vseh prejšnjih sklopov in s tistimi, kjer ste dodali regularizacijo L2 (4, 16 in 15 nevronov, brez in z regularizacije L2, brez in s plastjo izpusta). Podajte komentar in ugotovitve glede prekomernega prilagajanja. Katera arhitektura nevronske mreže je vodila v najboljše rezultate po metriki točnosti? Pri kateri nevronski mreži je bilo učenje najbolj stabilno?

## V1.5

Ponovite učenje vseh treh arhitektur nevronskih mrež (4, 16 in 512 nevronov) s kombinacijo regularizacije L2 in plasti izpusta. Na enak način kot prej izrišite grafe funkcij izgub in točnosti, ter podajte komentar in ugotovitve. Nato ponovite učenje arhitekture nevronske mreže tega sklopa in vseh arhitektur prejšnjih sklopov z več podatki. Podatkovno zbirko razširite na najpogostejših 5000 besed. Ali se je s povečanjem količine podatkov kaj spremenilo? Za vse kombinacije arhitektur povečajte in pomanjšajte stopnjo učenja. Izrišite grafe funkcij izgub in točnosti, ter podajte komentar in ugotovitve. Ali spreminjanje stopnje učenja vpliva na prekomerno prileganje?
