from __future__ import print_function, division, absolute_import
import numpy
import numbers
import itertools


def conf_gen(shape):
    """
    @brief      yield all configurations
    @param      shape  The shape
    """
    for x in itertools.product(*tuple(range(s) for s in shape)):
        yield x


class ValueTableBase(object):

    def argmin(self):

        min_val = float('inf')
        min_conf = None

        for conf in conf_gen(self.shape):
            val = self[conf]
            if val < min_val:
                min_val = val
                min_conf = conf
        return conf

    def min(self):
        min_val = float('inf')
        for conf in conf_gen(self.shape):
            val = self[conf]
            if val < min_val:
                min_val = val
        return min_val


    def gradient(self, weight_number, labels):
        return 0.0

    @property
    def weight_ids(self):
        return None

    @property
    def n_weights(self):
        return 0

    def change_weights(self, weights):
        pass



class DenseValueTable(ValueTableBase):
    def __init__(self, values):
        super(DenseValueTable, self).__init__()
        self._values = numpy.require(values,dtype='float32',requirements=['C'])

    @property
    def shape(self):
        return self._values.shape

    @property
    def ndim(self):
        return self._values.ndim

    def __getitem__(self, labels):
        if self._values.ndim == 1:
            if isinstance(labels, numbers.Integral):
                return float(self._values[labels])
            else:
                return float(self._values[labels[0]])
        else:
            return float(self._values[tuple(labels)])



class PottsFunction(ValueTableBase):
    def __init__(self, shape, beta):
        super(PottsFunction, self).__init__()
        self.beta = beta
        self._shape = shape

    def __getitem__(self, labels):
        assert len(labels) == 2
        return [0.0,self.beta][labels[0]!=labels[1]]

    @property
    def ndim(self):
        return 2
        
    @property
    def shape(self):
        return self._shape




class WeightedTwoClassUnary(object):
    def __init__(self, features, weight_ids, weights, const_terms=None):
        self.features = features.copy()
        self._weight_ids = weight_ids
        self._values = (0,0)
        self.const_terms = const_terms

    @property
    def shape(self):
        return (2,)

    @property
    def ndim(self):
        return 1

    def __getitem__(self, labels):
        return float(self._values[labels[0]])


    @property
    def weight_ids(self):
        return self._weight_ids

    @property
    def n_weights(self):
        return len(self.weight_ids)

    def change_weights(self, weights):
        self.weights = weights
        w = self.weights[self.weight_ids]
        v1 = numpy.dot(self.features, w)
        if self.const_terms is None:
            self._values = (0.0, v1)
        else:
            self._values = (self.const_terms[0], v1 + self.const_terms[1])

    def gradient(self, weight_number, labels):
        assert weight_number < len(self.weight_ids)
        if labels[0] == 1:
            return self.features[weight_number]
        else:
            return 0.0


 
        
class WeightedPottsFunction(ValueTableBase):
    def __init__(self, shape, features, weight_ids, weights=None):
        super(WeightedPottsFunction, self).__init__()
        self.weights = None
        self._weight_ids = weight_ids.copy()
        self.features = features.copy()
        self.beta = None
        self._shape = shape
        self.change_weights(weights)

    def gradient(self, weight_number, labels):
        assert weight_number < len(self.weight_ids)
        if labels[0] != labels[1]:
            return self.features[weight_number]
        else:
            return 0.0

    @property
    def weight_ids(self):
        return self._weight_ids

    @property
    def n_weights(self):
        return len(self._weight_ids)

    def change_weights(self, weights):
        self.weights = weights
        w = self.weights[self.weight_ids]
        self.beta = numpy.dot(self.features, w)


    def __getitem__(self, labels):
        assert self.beta is not None
        assert len(labels) == 2
        return [0.0,self.beta][labels[0]!=labels[1]]

    @property
    def ndim(self):
        return 2
        
    @property
    def shape(self):
        return self._shape



    # specializations  => they are not needed but nice to have
    def argmin(self):
        return (0,0)

    def min(self):
        return 0.0