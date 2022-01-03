# Science Fiction Books Classifier
Cas Kaggle. 2021/2022

### Nom: Pablo Ruiz de la Hermosa Jimenez de los Galanes
### Dataset: Science Fiction Books
### URL: https://www.kaggle.com/tanguypledel/science-fiction-books-subgenres

## Resum
El *dataset* recopila dades del web https://www.goodreads.com i consta de 12 fitxers. Cada fitxer conté mostres de llibres del gènere de ciència-ficció que pertanyen a un dels següents subgèneres:
- Aliens
- Història alternativa
- Univers alternatiu
- Apocalíptic
- Ciberpunk
- Distopia
- Hard
- Militar
- Robots
- Space Opera
- Steampunk
- Viatge en el temps

Les mostres estan etiquetades amb els subgèneres als quals pertanyen. Poden pertànyer a més subgèneres a part del principal del fitxer.

### Objectiu
Predir els subgèneres d'un llibre a partir de la seva descripció (*multi-label text classification*).

## Experiments
### Preprocessat
He ajuntat els 12 fitxers en un únic conjunt de dades. He eliminat les categories que no interessaven per a l'objectiu i m'he quedat amb el títol, autor, any i descripció de cada llibre. He creat una columna per cada subgènere que pot valdre 1 o 0 segons la pertinença del llibre al subgènere.

### Model
He dissenyat i entrenat dos models capaços de resoldre el problema de *multi-label text classification* englobat dins del camp del *Natural Language Processing* (NLP).

Primer, un model basat en la classe `OneVsRestClassifier` amb l'estimador `LinearSVC` de la llibreria *sklearn*. Aquest model permet calcular quina *accuracy* mínima s'obté d'un model simple i ràpid d'entrenar.

Després, un model usant la llibreria PyTorch Lightning i el model preentrenat  BERT de https://huggingface.co/. Fa servir el model BERT (bert-base-uncased), una capa lineal, la funció d'activació sigmoide, la funció d'error `BCELoss()` i l'optimitzador `AdamW`.

|              Model             | Accuracy | Temps |
|:------------------------------:|:--------:|:-----:|
| OneVsRestClassifier(LinearSVC) |    40%   |  20 s  |
|    PyTorch Lightning + BERT    |    92%   |  1,5 h |

## Demo
El model ocupa 1,3 GB. No es pot penjar al repositori.

## Conclusions
És evident que el model que fa servir BERT resol millor l'objectiu. En comparació amb el model de la llibreria *sklearn* té un rendiment molt superior. En comparació amb els models més punters que utilitzen bilions de paràmetres, es queda una mica enrere. Però no es pot aconseguir més disposant únicament de la GPU al núvol de *Google Colab*.

## En un futur...
Es podrien investigar més configuracions del model BERT fent servir GPU i TPU més potents.
