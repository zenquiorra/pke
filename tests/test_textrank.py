#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from spacy.tokens import Doc
from spacy.vocab import Vocab

import os
import pke

test_file = os.path.join('tests', 'data', '1939.doc')
pos = {'NOUN', 'PROPN', 'ADJ'}
doc = Doc(Vocab()).from_disk(test_file)


def test_textrank():
  """Test TextRank for keyword extraction using original paper's example."""

  extractor = pke.unsupervised.TextRank()
  extractor.load_document(input=doc)
  extractor.candidate_weighting(top_percent=.33, pos=pos)
  keyphrases = [k for k, s in extractor.get_n_best(n=3)]
  assert keyphrases == ['linear diophantine',
                        'natural numbers',
                        'types']


def test_textrank_with_candidate_selection():
  """Test TextRank with longest-POS-sequences candidate selection."""

  extractor = pke.unsupervised.TextRank()
  extractor.load_document(input=doc)
  extractor.candidate_selection(pos=pos)
  extractor.candidate_weighting(pos=pos)
  keyphrases = [k
                for k, s in extractor.get_n_best(n=3)]
  assert keyphrases == ['linear diophantine equations',
                        'minimal generating sets',
                        'mixed types']


if __name__ == '__main__':
    test_textrank()
    test_textrank_with_candidate_selection()
