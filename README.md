# `pke` - python keyphrase extraction

`pke` is an **open source** python-based **keyphrase extraction** toolkit. It
provides an end-to-end keyphrase extraction pipeline in which each component can
be easily modified or extended to develop new models. `pke` also allows for 
easy benchmarking of state-of-the-art keyphrase extraction models, and 
ships with supervised models trained on the
[SemEval-2010 dataset](http://aclweb.org/anthology/S10-1004).

[![Build Status](https://travis-ci.org/boudinfl/pke.svg?branch=master)](https://travis-ci.org/boudinfl/pke)

## Table of Contents

* [Installation](#installation)
* [Minimal example](#minimal-example)
* [Getting started](#getting-started)
* [Implemented models](#implemented-models)
* [Citing pke](#citing-pke)

## Installation

`spacy` is required for document preprocessing and could be installed using the following lines.
More details about installation and models for other languages can be found on [`spacy website`](https://spacy.io/usage).

```bash
pip install -U spacy
python -m spacy download en_core_web_sm # here to download the english model
```


To pip install `pke` from github:

```bash
pip install git+https://github.com/boudinfl/pke.git
```

## Minimal example

`pke` provides a standardized API for extracting keyphrases from a document.
Start by typing the 5 lines below. For using another model, simply replace
`pke.unsupervised.TopicRank` with another model ([list of implemented models](#implemented-models)).

```python
import pke
import spacy

# preprocess an input file with spacy
nlp = spacy.load("en_core_web_sm")
with open('path/to/input.txt') as f:
    doc = nlp(f.read())

# initialize keyphrase extraction model, here TopicRank
extractor = pke.unsupervised.TopicRank()

# load the content of the document from a spacy processed doc object
extractor.load_document(input=doc, language='en')

# keyphrase candidate selection, in the case of TopicRank: sequences of nouns
# and adjectives (i.e. `(Noun|Adj)*`)
extractor.candidate_selection()

# candidate weighting, in the case of TopicRank: using a random walk algorithm
extractor.candidate_weighting()

# N-best selection, keyphrases contains the 10 highest scored candidates as
# (keyphrase, score) tuples
keyphrases = extractor.get_n_best(n=10)
```

A detailed example is provided in the [`examples/`](examples/) directory.

## Getting started

Tutorials and code documentation are available at
[https://boudinfl.github.io/pke/](https://boudinfl.github.io/pke/).

## Implemented models

`pke` currently implements the following keyphrase extraction models:

* Unsupervised models
  * Statistical models
    * TfIdf [[documentation](https://boudinfl.github.io/pke/build/html/unsupervised.html#tfidf)]
    * KPMiner [[documentation](https://boudinfl.github.io/pke/build/html/unsupervised.html#kpminer), [article by (El-Beltagy and Rafea, 2010)](http://www.aclweb.org/anthology/S10-1041.pdf)]
    * YAKE [[documentation](https://boudinfl.github.io/pke/build/html/unsupervised.html#yake), [article by (Campos et al., 2020)](https://doi.org/10.1016/j.ins.2019.09.013)]
  * Graph-based models
    * TextRank [[documentation](https://boudinfl.github.io/pke/build/html/unsupervised.html#textrank), [article by (Mihalcea and Tarau, 2004)](http://www.aclweb.org/anthology/W04-3252.pdf)]
    * SingleRank  [[documentation](https://boudinfl.github.io/pke/build/html/unsupervised.html#singlerank), [article by (Wan and Xiao, 2008)](http://www.aclweb.org/anthology/C08-1122.pdf)]
    * TopicRank [[documentation](https://boudinfl.github.io/pke/build/html/unsupervised.html#topicrank), [article by (Bougouin et al., 2013)](http://aclweb.org/anthology/I13-1062.pdf)]
    * TopicalPageRank [[documentation](https://boudinfl.github.io/pke/build/html/unsupervised.html#topicalpagerank), [article by (Sterckx et al., 2015)](http://users.intec.ugent.be/cdvelder/papers/2015/sterckx2015wwwb.pdf)]
    * PositionRank [[documentation](https://boudinfl.github.io/pke/build/html/unsupervised.html#positionrank), [article by (Florescu and Caragea, 2017)](http://www.aclweb.org/anthology/P17-1102.pdf)]
    * MultipartiteRank [[documentation](https://boudinfl.github.io/pke/build/html/unsupervised.html#multipartiterank), [article by (Boudin, 2018)](https://arxiv.org/abs/1803.08721)]
* Supervised models
  * Feature-based models
    * Kea [[documentation](https://boudinfl.github.io/pke/build/html/supervised.html#kea), [article by (Witten et al., 2005)](https://www.cs.waikato.ac.nz/ml/publications/2005/chap_Witten-et-al_Windows.pdf)]
    * WINGNUS [[documentation](https://boudinfl.github.io/pke/build/html/supervised.html#wingnus), [article by (Nguyen and Luong, 2010)](http://www.aclweb.org/anthology/S10-1035.pdf)]

## Citing pke

If you use `pke`, please cite the following paper:

```
@InProceedings{boudin:2016:COLINGDEMO,
  author    = {Boudin, Florian},
  title     = {pke: an open source python-based keyphrase extraction toolkit},
  booktitle = {Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: System Demonstrations},
  month     = {December},
  year      = {2016},
  address   = {Osaka, Japan},
  pages     = {69--73},
  url       = {http://aclweb.org/anthology/C16-2015}
}
```
