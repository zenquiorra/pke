# -*- coding: utf-8 -*-
# Author: Florian Boudin
# Date: 19-08-2021

"""Unified graph-based keyphrase generation model.

    WIP...
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import networkx as nx
import numpy as np
import six
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import CountVectorizer

import pke.utils
from pke.unsupervised import TopicRank


class UnifiedRank(TopicRank):
    """Unified graph-based keyphrase generation model.

        WIP...
    """

    def __init__(self):
        """Redefining initializer for UnifiedRank."""

        super(UnifiedRank, self).__init__()

        self.domain_to_integer = {}

        self.topic_to_integer = {}
