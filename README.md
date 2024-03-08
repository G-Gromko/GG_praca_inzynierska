Wersja Pythona na której został stworzony program: 3.11.0

================================================

Plikiem do uczenia modelu jest notatnik E2E_pianoform_model.ipynb, i jest on przeznaczony do działania w środowisku Google Colab.
Model do uczenia wymaga ~18GB RAM oraz ~9GB VRAM

Zbiór danych GrandStaff do uczenia modelu znajduje się na stronie: https://sites.google.com/view/multiscore-project/datasets

Z racji braku pików partycjonowania danych do uczenia i testowania zawartych w zbiorze, a niebędnych do działania, zbiór po pobraniu należy rozpakować lokalnie na komputerze do wybranego katalogu, po czym uruchomić skrypt make_dataset_partitions.py, podając mu ścieżkę do katalogu rozpakowanego zbioru danych, którego struktura wewnętrzna powinna się prezentować następująco:

├ beethoven<br />
├ chopin<br />
├ hummel<br />
├ joplin<br />
├ mozart<br />
└ scarlatti-d<br />

Gdy skrypt make_dataset_partitions.py zakończy swoje działanie, do powyższych katalogów powinien dołączyć katalog partitions z wewnętrzną strukturą:

partitions<br />
&ensp;├ test.txt<br />
&ensp;├ train.txt<br />
&ensp;└ val.txt<br />

Następnie katalogi beethoven, chopin, hummel, joplin, mozart, partitions oraz scrlatti-d należy zapakować do pliku grandstaff.zip. Utworzone archiwum należy umieścić na Dysku Google konta połączonego z Google Colab.

W momencie gdy notatnik E2E_pianoform_model.ipynb jest załadowany w środowisku Colab, a zapakowany zbiór danych Grandstaff z dodanymi plikami partycjonowania znajduje się na Dysku, by rozpocząć uczenie modelu należy uruchomić cały kod z notatnika i podążać za poleceniami, które mogą się wyświetlić, jak autoryzacja dostępu Google Colab do Dysku, czy prośba o podanie klucza API WandB.


================================================

Z racji swojej wielkości, punkty zapisu znajdują się w folderze na moim Dysku Google, do którego link znajduje się poniżej:

https://drive.google.com/drive/folders/13xe0VCklnhYtFbxWt8kX5HAJeksz4DVO?usp=sharing


Do działania programu należy pobrać jeden z nich i umieścić w folderze /code/model_checkpoints

================================================

Plik hum2mid jest niezbędny dla zapewnienia założonej funkcjonalności i został skompilowany ze zbioru dodatkowych narzędzi Humdrum, dostępnym na repozytorium:

https://github.com/craigsapp/humextra

================================================

Usuwanie wypaczenia jest przyciętą i lekko dostosowaną wersją programu stworzonego przez Matta Zucker'a. Link do źródła znajduje się w sources/websites.txt


