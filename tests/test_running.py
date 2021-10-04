#!/usr/bin/env python
# -*- coding: utf-8 -*-

from spacy.tokens import Doc
from spacy.vocab import Vocab

import os
from pke.unsupervised import (
    TopicRank, SingleRank,
    MultipartiteRank, PositionRank,
    TopicalPageRank, ExpandRank,
    TextRank, TfIdf, KPMiner,
    YAKE, FirstPhrases
)
from pke.supervised import Kea, WINGNUS

test_file = os.path.join('tests', 'data', '1939.doc')
doc = Doc(Vocab()).from_disk(test_file)

def test_unsupervised_run():
    def test(model):
        extractor = model()
        extractor.load_document(input=doc)
        extractor.candidate_selection()
        extractor.candidate_weighting()

    models = [
        TopicRank, SingleRank,
        MultipartiteRank, PositionRank,
        TopicalPageRank, ExpandRank,
        TextRank, TfIdf, KPMiner,
        YAKE, FirstPhrases
    ]
    for m in models:
        print("testing {}".format(m))
        test(m)


def test_supervised_run():
    def test(model):
        extractor = model()
        extractor.load_document(input=doc)
        extractor.candidate_selection()
        extractor.candidate_weighting()

    models = [
        Kea, WINGNUS
    ]
    for m in models:
        print("testing {}".format(m))
        test(m)


if __name__ == '__main__':
    test_unsupervised_run()
    test_supervised_run()
