#!/usr/bin/env python
# -*- coding: utf-8 -*-

from spacy.tokens import Doc
from spacy.vocab import Vocab

import os
import pke

test_file = os.path.join('tests', 'data', '1939.doc')
pos = {'NOUN', 'PROPN', 'ADJ'}
doc = Doc(Vocab()).from_disk(test_file)


def test_singlerank_candidate_selection():
    """Test SingleRank candidate selection method."""

    extractor = pke.unsupervised.SingleRank()
    extractor.load_document(input=doc)
    extractor.candidate_selection(pos=pos)
    assert len(extractor.candidates) == 20


def test_singlerank_candidate_weighting():
    """Test SingleRank candidate weighting method."""

    extractor = pke.unsupervised.SingleRank()
    extractor.load_document(input=doc)
    extractor.candidate_selection(pos=pos)
    extractor.candidate_weighting(window=10, pos=pos)
    keyphrases = [k for k, s in extractor.get_n_best(n=3)]
    assert keyphrases == ['minimal generating sets',
                          'minimal set',
                          'types systems']


if __name__ == '__main__':
    test_singlerank_candidate_selection(doc)
    test_singlerank_candidate_weighting(doc)
