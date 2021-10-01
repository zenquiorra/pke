#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Readers for the pke module."""

from pke.data_structures import Document


class Reader(object):
    def read(self, path):
        raise NotImplementedError


class SpacyDocReader(Reader):
    """Minimal Spacy Doc Reader."""

    def read(self, spacy_doc, **kwargs):
        sentences = []
        for sentence_id, sentence in enumerate(spacy_doc.sents):
            sentences.append({
                "words": [token.text for token in sentence],
                "lemmas": [token.lemma_ for token in sentence],
                "POS": [token.pos_ or token.tag_ for token in sentence],
                "char_offsets": [(token.idx, token.idx + len(token.text)) for token in sentence]
            })
        return Document.from_sentences(sentences, **kwargs)
