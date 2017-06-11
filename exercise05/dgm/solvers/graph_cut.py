from __future__ import print_function, division, absolute_import

import numpy
import networkx

class GraphCut(object):
    def __init__(self, model, tolerance=0.00001):
        self.model = model

        if self.model.max_arity > 2:
            raise RuntimeError("graph cut can only be used for second order models")

        self._tolerance = tolerance
        self._best_labels = numpy.zeros(model.n_variables, dtype='uint32')
        self._best_energy = self.model.evaluate(self._best_labels)

    @property
    def best_labels(self):
        return self._best_labels

    @property
    def best_energy(self):
        return self._best_energy

    @property
    def current_labels(self):
        return self._best_labels

    @property
    def current_energy(self):
        return self._best_energy


    

    def optimize(self, starting_point=None, visitor=None):

        # inform visitor that inference has been started
        if visitor is not None:
            visitor.start(self)


        networkx_graph = networkx.DiGraph()
        networkx_graph.add_nodes_from(range(self.model.n_variables+2))

        def add_edge(source, target, capacity):
            edge_dict = networkx_graph.get_edge_data(source, target)
            if edge_dict is None:
                networkx_graph.add_edge(source, target, capacity=capacity)
            else:
                capacity += edge_dict['capacity']
                networkx_graph[source][target]['capacity'] = capacity



        s_node = self.model.n_variables
        t_node = s_node + 1

        for factor in self.model.factors:

            if factor.arity == 1:
                variable = factor.variables[0]
                e0 = factor.evaluate((0,))
                e1 = factor.evaluate((1,))

                if e0 <= e1:
                    c = e1-e0
                    add_edge(s_node, variable, capacity=c)
                else:
                    c = e0 - e1
                    add_edge(variable, t_node, capacity=c)

            elif factor.arity == 2:

                variable0 = factor.variables[0]
                variable1 = factor.variables[1]

                e00 = factor.evaluate((0,0))
                e01 = factor.evaluate((0,1))
                e10 = factor.evaluate((1,0))
                e11 = factor.evaluate((1,1))

                if(e10 > e00):
                    add_edge(s_node, variable0, capacity=e10 - e00);
                elif(e10 < e00):
                    add_edge(variable0, t_node, capacity=e00 - e10);
                if(e11 > e10):
                    add_edge(s_node, variable1, capacity=e11 - e10);
                elif(e11 < e10):
                    add_edge(variable1, t_node, capacity=e10 - e11);

                c = e01 + e10 - e00 - e11
                if c < 0.0 and c >= -1.0*self._tolerance:
                    c = 0.0
                elif c<0:
                    print("e00",e00)
                    print("e01",e01)
                    print("e10",e10)
                    print("e11",e11)
                    raise RuntimeError("pairwise factor must be submodular")

                add_edge(variable0, variable1, capacity=c);

        # get min cut
        cut_value, partition = networkx.minimum_cut(networkx_graph, s=s_node, t=t_node)
        partition0, partition1 = partition


        self._best_labels[:] = 0
        

         
        for variables in partition1:
            if variables < self.model.n_variables:
                self._best_labels[variables] = 1
        self._best_energy = self.model.evaluate(self._best_labels)
        
        if visitor is not None:
            visitor.end(self)

        return self._best_labels


