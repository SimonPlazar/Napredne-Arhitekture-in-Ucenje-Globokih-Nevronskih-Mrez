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



## V4.3



## V4.4



## V4.5
