#!/usr/bin/env python
# -*- coding: utf-8 -*-

from spacy.tokens import Doc
from spacy.vocab import Vocab

import os
import pke
import spacy

model = pke.unsupervised.TopicRank

data_path = os.path.join('tests', 'data')
doc_test_file = data_path + os.sep + '1939.doc'
raw_test_file = data_path + os.sep + '1939.txt'


def test_reading():

    # loading Doc input
    doc = Doc(Vocab()).from_disk(doc_test_file)
    extr1 = model()
    extr1.load_document(input=doc)

    # loading txt input
    # preprocess an input file with spacy
    nlp = spacy.load("en_core_web_sm")
    with open(raw_test_file) as f:
        doc = nlp(f.read())
    extr2 = model()
    extr2.load_document(input=doc)

    assert len(extr1.sentences) == 4 and extr1.sentences == extr2.sentences


if __name__ == '__main__':
    test_reading()
