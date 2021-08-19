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

from pke.unsupervised import TopicRank
from itertools import combinations
from collections import defaultdict
import math


class UnifiedRank(TopicRank):
    """Unified graph-based keyphrase generation model.

        WIP...
    """

    def __init__(self):
        """Redefining initializer for UnifiedRank."""

        super(UnifiedRank, self).__init__()

        self.domain_to_integer = {}

        self.topic_to_integer = {}


    def build_topic_graph(self):
        """Re-define the topic graph construction method.

        Build the topic graph by connecting topics if their candidates
        co-occur in the same sentence. Edges are weighted by the number of
        co-occurrences.
        """

        # adding the nodes to the graph
        self.graph.add_nodes_from(range(len(self.topics)), src="topic")

        # loop through the topics to connect the nodes
        for i, j in combinations(range(len(self.topics)), 2):

            # for each candidate in topic i
            for c_i in self.topics[i]:

                # for each candidate in topic j
                for c_j in self.topics[j]:

                    weight = len(
                        set(self.candidates[c_i].sentence_ids).intersection(
                            self.candidates[c_j].sentence_ids))

                    if weight > 0:
                        if not self.graph.has_edge(i, j):
                            self.graph.add_edge(i, j, weight=0, type="in")
                        self.graph[i][j]['weight'] += weight

    def unify_with_phrasebank(self,
                              phrasebank,
                              prune_unreachable_nodes=True):
        """Unify the phrasebank graph with the topic graph, built from a 
        document.

        Args:
            input_file (str): path to the reference file.
            phrasebank (nx.Graph): phrasebank graph in network format.
            prune_unreachable_nodes (bool): prune nodes from the domain graph
                that are not reachable from the document nodes, defaults to
                True.
        """

        # initialize the topic_to_integer map
        for i, topic in enumerate(self.topics):
            for candidate in topic:
                self.topic_to_integer[candidate] = i

        offset = len(self.topics)

        # add src attribute to phrasebank nodes
        nx.set_node_attributes(phrasebank, ["phrasebank"]*len(phrasebank.nodes), 'src')

        self.graph = nx.compose(self.graph, phrasebank)

        # connect document an phrasebank entries
        for i in range(len(self.topics)):
            for candidate in self.topics[i]:
                if candidate in self.graph:
                    self.graph.add_edge(i, candidate, weight=1, type="out")

        # prune not reachable domain nodes
        if prune_unreachable_nodes:

            # find all descendants
            descendants = set()
            for i in range(len(self.topics)):
                descendants.update(nx.algorithms.descendants(self.graph, i))

            # remove unreachable nodes
            self.graph.remove_nodes_from(set(self.graph.nodes) - descendants)


    def candidate_weighting(self,
                            phrasebank,
                            prune_unreachable_nodes=True,
                            lambda_parameter=0.5,
                            nb_iter=100,
                            convergence_threshold=0.001):
        """Weight candidates using the co-ranking formulae.

        Args:
            phrasebank (nx.Graph): phrasebank graph in network format.
            prune_unreachable_nodes (bool): prune nodes from the domain graph
                that are not reachable from the document nodes, defaults to
                True.
            lambda_parameter(float): lambda parameter in the ranking formulae,
                defaults to 0.5.
            nb_iter (int): maximum number of iterations, defaults to 100.
            convergence_threshold (float): early stop threshold, defaults to
                0.001.
        """

        # compute topics
        self.topic_clustering()

        # build graph
        self.build_topic_graph()

        # unify with domain graph
        self.unify_with_phrasebank(phrasebank=phrasebank,
                                prune_unreachable_nodes=prune_unreachable_nodes)

        nb_nodes = len(self.graph.nodes)

        logging.info("resulting graph is {} nodes".format(nb_nodes))

        # weights = [1.0] * nb_nodes
        weights = defaultdict(lambda:1.0)

        # pre-compute the inner/outer normalizations
        # inner_norms = [0.0] * nb_nodes
        # outer_norms = [0.0] * nb_nodes
        inner_norms = defaultdict(lambda:0.0)
        outer_norms = defaultdict(lambda:0.0)

        for j in self.graph.nodes():
            inner_norm = 0
            outer_norm = 0
            for k in self.graph.neighbors(j):
                if self.graph[j][k]['type'] == "in":
                    inner_norm += self.graph[j][k]["weight"]
                else:
                    outer_norm += 1
            inner_norms[j] = inner_norm
            outer_norms[j] = outer_norm

        # ranking nodes in the graph using co-ranking
        converged = False
        while nb_iter > 0 and not converged:

            converged = True

            #logging.info("{} iter left".format(nb_iter))

            # save the weights
            w = weights.copy()

            for i in self.graph.nodes():

                # compute inner/outer recommendations
                r_in = 0.0
                r_out = 0.0
                for j in self.graph.neighbors(i):

                    # inner recommendation
                    if self.graph[i][j]['type'] == "in":
                        r_in += (self.graph[i][j]["weight"] * w[j]) / \
                                inner_norms[j]

                    # outer recommendation
                    else:
                        r_out += w[j] / outer_norms[j]

                # compute the new weight
                if self.graph.nodes[i]["src"] == "topic":
                    weights[i] = (1 - lambda_parameter) * r_out
                    weights[i] += lambda_parameter * r_in
                else:
                    weights[i] = (1 - lambda_parameter) * r_out
                    weights[i] += lambda_parameter * r_in

                # check for non convergence
                if math.fabs(weights[i] - w[i]) > convergence_threshold:
                    converged = False

            nb_iter -= 1

        # get the final ranking
        for i in self.graph.nodes():

            # if it is a topic candidate
            if self.graph.nodes[i]["src"] == "topic":

                # get the candidates from the topic
                topic = self.topics[i]

                # get the offsets of the topic candidates
                offsets = [self.candidates[t].offsets[0] for t in topic]

                first = offsets.index(min(offsets))
                self.weights[topic[first]] = weights[i]

            # otherwise it is a keyphrase from the domain
            else:

                # if the gold keyphrase occurs in the document and is
                # already weighted
                if i in self.weights:
                    self.weights[i] = max(self.weights[i], weights[i])
                else:
                    self.weights[i] = weights[i]

