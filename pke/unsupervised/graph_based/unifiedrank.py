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

    def unify_with_phrasebank(self, phrasebank, prune_unreachable):
        """Unify the phrasebank graphs with the topic graph."""

        # self.graph = nx.compose(self.graph, phrasebank)

        # for each graph from the phrasebank
        for G in phrasebank:
            doc_id = G.graph["src"]
            doc_weight = G.graph["weight"]
            logging.debug("unifying with {}".format(doc_id))

            # compose with the document graph
            self.graph = nx.compose(self.graph, G)

            # add the connections
            for node in G.nodes():
                _, node_str = node.split("___")
                if node_str in self.graph:
                    self.graph.add_edges_from([(node, node_str),
                                               (node_str, node)],
                                                weight=1.0)

        # prune not reachable domain nodes
        if prune_unreachable:

            # find all descendants
            descendants = set(self.candidates)
            for candidate in self.candidates:
                descendants.update(nx.algorithms.descendants(self.graph,
                                                             candidate))

            # remove unreachable nodes
            nb_nodes = len(self.graph.nodes)
            self.graph.remove_nodes_from(set(self.graph.nodes) - descendants)
            logging.debug("pruning graph from |{}| to |{}|".format(nb_nodes,
                                                         len(self.graph.nodes)))

            if len(self.graph.nodes) < len(self.candidates):
                logging.warning("Big Issue with node pruning :(")

    def candidate_weighting(self,
                            threshold=0.74,
                            method='average',
                            alpha=1.1,
                            phrasebank=None,
                            prune_unreachable=False):
        """ Candidate weight calculation using random walk.

            Args:
                threshold (float): the minimum similarity for clustering,
                    defaults to 0.25.
                method (str): the linkage method, defaults to average.
                alpha (float): hyper-parameter that controls the strength of the
                    weight adjustment, defaults to 1.1.
                phrasebank (nx.Graph): phrasebank graph in network format.
                prune_unreachable (boolean): prune phrasebank nodes that are
                    unreachable from the document nodes, defaults to False.

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
            self.unify_with_phrasebank(phrasebank, prune_unreachable)

        # print(len(self.graph.nodes))

        # adjust weights if needed
        if alpha > 0.0:
            self.weight_adjustment(alpha)

        # compute the word scores using random walk
        self.weights = nx.pagerank_scipy(self.graph)

        # remove duplicates from connected graphs
        candidates = list(self.weights.keys())
        for candidate in candidates:
            if "___" in candidate:
                _, _candidate = candidate.split("___")
                if _candidate in self.weights:
                    self.weights[_candidate] = max(self.weights[_candidate],
                                                   self.weights[candidate])
                else:
                    self.weights[_candidate] = self.weights[candidate]
                del self.weights[candidate]


