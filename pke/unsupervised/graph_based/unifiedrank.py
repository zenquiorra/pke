# -*- coding: utf-8 -*-
# Author: Florian Boudin
# Date: 19-08-2021

"""Unified graph-based keyphrase generation model.

    WIP...
"""

import os
import logging
import networkx as nx

from pke.unsupervised import MultipartiteRank
from itertools import combinations
from collections import defaultdict
# import matplotlib.pyplot as plt


class UnifiedRank(MultipartiteRank):
    """Unified graph-based keyphrase generation model.

        WIP...
    """

    def __init__(self):
        """Redefining initializer for UnifiedRank."""

        super(UnifiedRank, self).__init__()


    def unify_with_phrasebank(self, phrasebank):
        """Unify the phrasebank graph with the topic graph."""

        # self.graph = nx.compose(self.graph, phrasebank)

        for node_i in phrasebank.nodes():
            # add node if missing
            if node_i not in self.graph:
                self.graph.add_node(node_i, src="phrasebank")

        for node_i, node_j in phrasebank.edges():
            if not self.graph.has_edge(node_i, node_j):

                if "src" in self.graph[node_i] and "src" in self.graph[node_j]:
                    self.graph.add_edge(node_i, node_j,
                        weight=phrasebank[node_i][node_j]['weight'])
                else:
                    self.graph.add_edge(node_i, node_j, weight=0.0)
            else:
                self.graph[node_i][node_j]['weight'] += phrasebank[node_i][node_j]['weight']



    def candidate_weighting(self,
                            threshold=0.74,
                            method='average',
                            alpha=1.1,
                            phrasebank=None):
        """ Candidate weight calculation using random walk.

            Args:
                threshold (float): the minimum similarity for clustering,
                    defaults to 0.25.
                method (str): the linkage method, defaults to average.
                alpha (float): hyper-parameter that controls the strength of the
                    weight adjustment, defaults to 1.1.
                phrasebank (nx.Graph): phrasebank graph in network format.
        """
        if not self.candidates:
            return

        # cluster the candidates
        self.topic_clustering(threshold=threshold, method=method)

        # build the topic graph
        self.build_topic_graph()

        # print(len(self.graph.nodes))

        # unify with phrasebank graph
        if phrasebank is not None:
            self.unify_with_phrasebank(phrasebank)

        # print(len(self.graph.nodes))

        # adjust weights if needed
        if alpha > 0.0:
            self.weight_adjustment(alpha)

        # compute the word scores using random walk
        self.weights = nx.pagerank_scipy(self.graph)

