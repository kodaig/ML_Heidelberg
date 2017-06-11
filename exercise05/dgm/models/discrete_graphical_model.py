from __future__ import print_function, division, absolute_import
import numpy
import numbers






class Factor(object): 
    def __init__(self, variables, value_table):
        self._variables = variables
        self._value_table = value_table


    def find_var_pos(self, variable):
        """Find the position of a variable w.r.t. this factor
        
        Find the x such that factor.variables[x] = variable.
        If the variable is not in factors.variable
        None is returned
        
        Arguments:
            variable {int} -- [description]
        
        Returns:
            int/None -- the position of the variable
                        iff found, else None is returned
        """
        try:
            index = self._variables.index(variable)
            return index
        except ValueError:
            return None

    @property
    def variables(self):
        return self._variables

    @property
    def arity(self):
        return len(self.variables)

    @property
    def shape(self):
        return self._value_table.shape

    @property
    def size(self):
        s = 1
        shape = self.shape
        for i in range(self.arity):
            s *= shape[i]
        return s

    def evaluate(self, labels):
        """Add a new value.

       Args:
           value (str): the value to add.
        """
        return self._value_table[labels]


    # for weighted gms

    @property
    def n_weights(self):
        return self._value_table.n_weights

    @property
    def weight_ids(self):
        return self._value_table.weight_ids

    def change_weights(self, weights):
        self._value_table.change_weights(weights=weights)

    def gradient(self, weight_number, labels):
        return self._value_table.gradient(weight_number=weight_number, labels=labels)




class DiscreteGraphicalModel(object):
    """[summary]
        
        [description]
        
        Arguments:
            variable_space {list,1d-ndarray,...} -- a sequence which is as long as the number
            of variables. Each entry encodes the number of labels for the particular variable

        Usage:
            >>> from __future__ import print_function
            >>> from dgm.models import DiscreteGraphicalModel
            >>> model = DiscreteGraphicalModel([3,3,3,3,2])
            >>> print(model.n_variables)
            5
            >>> print(model.n_labels(0))
            3
            >>> print(model.n_labels(4))
            2
    """
    def __init__(self, variable_space):
        self._max_arity = 0
        self._variable_space = numpy.require(variable_space,dtype='uint32')
        self._variable_space .flags.writeable = False
        self.factors = []
        self.factors_of_variables = [[] for vi in range(self.n_variables)]


    def min_max_number_of_labels(self):
        """ Min and max number of labels
        
        Find the min number of labels and max number of labels
        variables can take in this graphical model

        Usage:
            >>> from __future__ import print_function
            >>> from dgm.models import DiscreteGraphicalModel
            >>> model = DiscreteGraphicalModel([3,3,3,3,2])
            >>> print(model.min_max_number_of_labels)
            (2, 3)
        """
        return int(self._variable_space.min()),int(self._variable_space.min())

    @property
    def variable_space(self):
        return self._variable_space

    @property
    def n_variables(self):

        return len(self._variable_space)

    @property
    def n_factors(self):
        return len(self.factors)

    def n_labels(self, variable):
        return self._variable_space[variable]

    @property
    def max_arity(self):
        return self._max_arity


    

    def evaluate(self, labels):
        """
        @brief      Evaluate the graphical model
            for a certain labeling.
        @param      self    The object
        @param      labels  The labeling 
        @return     The energy for this labeling
        """
        val = 0.0
        for factor in self.factors:

            # make the labeling for the factor
            l = [None]*factor.arity
            for i,var in enumerate(factor.variables):
                l[i] = int(labels[var])
            #print(l)
            # evaluate the factors
            val += factor.evaluate(l)
        return val


    def factors_of_variable(self, variable):
        """aefrear
        """
        return self.factors_of_variables[variable]

    def add_factor(self, variables, value_table):
        if isinstance(variables, numbers.Integral):
            variables = [variables]

        factor_index = self.n_factors   
        fac = Factor(variables=variables, value_table=value_table)

        self._max_arity = max(fac.arity, self._max_arity)

        # add the factor
        self.factors.append(fac)
        
        # mapping from variables to factors
        for var in variables:
            self.factors_of_variables[var].append(fac)



class WeightedDiscreteGraphicalModel(DiscreteGraphicalModel):
    def __init__(self, variable_space, weights):
        super(WeightedDiscreteGraphicalModel, self).__init__(variable_space=variable_space)

        self.weights = weights
    @property
    def n_weights(self):
        return self.weights.shape[0]

    def change_weights(self, weights):
        self.weights = weights
        for factor in self.factors:
            if factor.n_weights > 0:
                factor.change_weights(weights)


    def phi(self, labels):

        ret = numpy.zeros(self.n_weights)
        for factor in self.factors:
            if factor.n_weights > 0:

                factor_labels = labels[factor.variables]
                weight_ids = factor.weight_ids

                for w_nr, w_id in enumerate(weight_ids):
                    g = factor.gradient(w_nr, factor_labels)
                    ret[w_id] += g

        return ret