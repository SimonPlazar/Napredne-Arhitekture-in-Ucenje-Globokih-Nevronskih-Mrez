# V4 - Globoko okrepitveno učenje

Na vajah boste spoznali globoko okrepitveno učenje (ang. deep reinforcement learning) in globoke Q-mreže (ang. deep Q-networks - DQN). Implementirali boste model in agenta za reševanje problemov Lunar Lander, CartPole, Acrobot in Pong v okolju OpenAI Gym/Gymnasium. Implementirali boste modela DQN in DDQN.

Rezultate boste prikazali z izrisom grafov ocen skozi epizode in z animacijami GIF, ki prikazujejo delovanje modelov. Vajo implementirajte s programskim jezikom Python in orodjem Jupyter Notebook.

Vaja je sestavljena iz 5 sklopov, ki se ocenjujejo posebej in se med seboj dopolnjujejo.

Točkovanje:

Sklop V4.1 - 50 točk
Sklop V4.2 - 10 točk
Sklop V4.3 - 10 točk
Sklop V4.4 - 15 točk
Sklop V4.5 - 15 točk

Oddajte datoteko .zip z vso programsko kodo (datoteki .ipynb in .py). Datoteko poimenujte po predlogi: V4.1-[ime]-[priimek].zip (primer: V4.1-Janez-Novak.zip).

## V4.1

Implementirajte agenta za reševanje problema Lunar Lander, ki uporablja DQN (ang. deep Q-network). Pri implementaciji boste implementirali strukturo Q-mreže (ang. Q-network), ki naj ima na vhodni plasti toliko nevronov, kot je število stanj problema Lunar Lander. Na skriti plasti naj bo 128 nevronov, na izhodni plasti pa toliko nevronov, kot je število možnih akcij pri problemu Lunar Lander. Uporabite aktivacijsko funkcijo ReLU.

Za potrebe Q-učenja implementirajte spomin (ang. memory, replay buffer), kamor boste med učenjem shranjevali agentove izkušnje v okolju Lunar Lander. Implementirajte tudi agenta, ki bo vseboval vse potrebne parametre, spomin in Q-mrežo. Pri učenju za funkcijo izgube uporabite povprečno kvadratno napako (ang. mean squared error - MSE). Pri učenju lahko implementirate tudi zaustavitveni pogoj, kadar je povprečna ocena v zadnjih 10 korakih večja ali enaka 250. Izrišite graf ocen skozi epizode učenja in animacijo GIF, ki prikazuje delovanje modela.

## V4.2

Razširite implementacijo sklopa V4.1 z implementacijo DDQN (ang. double deep Q-network) na problemu Lunar Lander. Tokrat imejte v implementaciji agenta dve Q-mreži. Prva Q-mreža bo izbirala akcije (ang. online network), druga pa bo vrednotila akcije (ang. target network). Ideja je, da se izbere tista akcija, ki ima maksimalno vrednost v prvi mreži, vendar se ovrednoti z drugo mrežo. Periodično se vrednosti prve Q-mreže skopirajo v drugo Q-mrežo. Pri učenju za funkcijo izgube uporabite povprečno kvadratno napako (ang. mean squared error - MSE).

Pri učenju lahko implementirate tudi zaustavitveni pogoj, kadar je povprečna ocena v zadnjih 10 korakih večja ali enaka 200. Izrišite graf ocen skozi epizode učenja in animacijo GIF, ki prikazuje delovanje modela.

## V4.3

Implementaciji DQN in DDQN iz sklopov V4.1 in V4.2 zaženite na problemih CartPole in Acrobot. Poiščite ustrezne parametre za učenje implementirate pa lahko tudi smiselne zaustavitvene pogoje. Izrišite graf ocen skozi epizode učenja in animacijo GIF, ki prikazuje delovanje modela za oba problema (priložen je primer za problem Acrobot).

## V4.4

Implementaciji DQN in DDQN iz sklopov V4.1 in V4.2 zaženite na problemu Pong. Poiščite ustrezne parametre za učenje implementirate pa lahko tudi smiselne zaustavitvene pogoje. Izrišite graf ocen skozi epizode učenja in animacijo GIF, ki prikazuje delovanje modela.

## V4.5

Implementacijo DQN in DDQN iz sklopa V4.4 (problem Pong) dopolnite z zmožnostjo učenja na več grafičnih procesnih enotah. Uporabite PyTorch DDP (Distributed Data Parallel) ali PyTorch Lightning Fabric (priporočljivo). Uporabite Kaggle Notebook, ki vam po verifikaciji računa omogoča dostop do 2x NVIDIA Tesla T4 grafičnih procesnih enot. Primerjajte čas učenja in kvaliteto modela na problemu Pong. Izrišite graf ocen skozi epizode učenja in animacijo GIF, ki prikazuje delovanje modela.
