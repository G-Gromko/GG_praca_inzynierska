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

W momencie gdy notatnik E2E_pianoform_model.ipynb jest załadowany w środowisku Colab, a zapakowany zbiór danych Grandstaff z dodanymi plikami partycjonowania znajduje się na Dysku, by rozpocząć uczenie modelu należy uruchomić cały kod z notatnika i podążać za poleceniami, które mogą się wyświetlić, jak autoryzacja dostępu Google Colab do Dysku, czy prośba o podanie klucza API WanDB.


================================================

W folderze code znajduje się kod źródłowy tego co działa aktualnie w projekcie, czyli na tę chwilę wykrywanie pozycji pięciolinii, usuwanie wypaczenia oraz segmentacja obrazu. Wraz z tym jak będzie postępowała praca będę tam dodawał kolejne elementy.

Usuwanie wypaczenia jest przyciętą i lekko dostosowaną wersją programu stworzonego przez Matta Zucker'a. Link do źródła znajduje się w sources/websites.txt


