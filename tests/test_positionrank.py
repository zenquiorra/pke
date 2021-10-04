#!/usr/bin/env python
# -*- coding: utf-8 -*-

from spacy.tokens import Doc
from spacy.vocab import Vocab

import os
import pke

test_file = os.path.join('tests', 'data', '1939.doc')
doc = Doc(Vocab()).from_disk(test_file)

grammar = "NP: {<ADJ>*<NOUN|PROPN>+}"
pos = {'NOUN', 'PROPN', 'ADJ'}


def test_positionrank_candidate_selection():
    """Test PositionRank candidate selection method."""

    extractor = pke.unsupervised.PositionRank()
    extractor.load_document(input=doc)
    extractor.candidate_selection(grammar=grammar)
    assert len(extractor.candidates) == 18


def test_positionrank_candidate_weighting():
    """Test PositionRank candidate weighting method."""

    extractor = pke.unsupervised.PositionRank()
    extractor.load_document(input=doc)
    extractor.candidate_selection(grammar=grammar)
    extractor.candidate_weighting(window=10, pos=pos)
    keyphrases = [k for k, s in extractor.get_n_best(n=3)]
    assert keyphrases == ['minimal generating sets', 'types systems', 'minimal set']


if __name__ == '__main__':
    test_positionrank_candidate_selection(doc)
    test_positionrank_candidate_weighting(doc)
