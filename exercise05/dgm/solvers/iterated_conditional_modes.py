from __future__ import print_function, division, absolute_import

import numpy

class IteratedConditionalModes(object):
    def __init__(self, model, verbose=1):

        self.model = model
        self.verbose = verbose
        self._best_labels = numpy.zeros(model.n_variables, dtype='uint32')
        self._best_energy = self.model.evaluate(self._best_labels)

    @property
    def best_labels(self):
        return self._best_labels

    @property
    def best_engery(self):
        return self._best_engery

    @property
    def current_labels(self):
        return self._best_labels

    @property
    def current_energy(self):
        return self._best_energy


    @property
    def best_energy(self):
        return self._best_energy
    
    def optimize(self, starting_point=None, visitor=None):

        # inform visitor that inference has been started
        if visitor is not None:
            visitor.start(self)

        # shortcuts
        model = self.model

        # the current best configuration
        _best_labels = self._best_labels
        if starting_point is not None:
            _best_labels[:] = starting_point



        best_energy = model.evaluate(_best_labels)
        is_local_opt = numpy.zeros(model.n_variables, dtype='bool')

        continue_search = True
        while(continue_search):

            # if at least one variable is changing its label during one iteration
            # continue search will be set to true 
            continue_search = False

            # loop over all variables
            # and select the best label for each variable
            # conditioned on the labels of the rest of the variables
            for variable in range(model.n_variables):
                
                # avoid unnecessary computations:
                # variables which are already locally
                # optimal can be skipped 
                if is_local_opt[variable] == False:

                    # get the number of labels for this variable
                    n_labels = model.n_labels(variable)

                    # the current label
                    current_label = _best_labels[variable]

                    # get all factors which
                    # are connected to this variable
                    factors = model.factors_of_variable(variable)

                    # get the position of the variable
                    # w.r.t. the factors variables
                    # => for each factor we find x 
                    #     such that factor.variables[x] = variable
                    var_positions = [factor.find_var_pos(variable) for factor in factors]

                    # get the current configuration of the 
                    # individual factors
                    factor_confs = []
                    for factor in factors:
                        labels = [ _best_labels[v] for v in factor.variables]
                        factor_confs.append(labels)

                    

                    # iterate over all states of the variable 
                    # and evaluate all factors connected to this
                    # variable. Doing so we find the label
                    # with the lowest energy.
                    factors_current_energy = None
                    factors_best_energy = float('inf')
                    best_label  = None

                    # actual iteration over all labels
                    for label in range(n_labels):

                        # evaluate all factors of the variable
                        # with the new label
                        factors_energy = 0.0
                        for factor, pos, conf in zip(factors, var_positions, factor_confs):
                            
                            # set the label to variable the new  label
                            conf[pos] = label

                            # evaluate the energy of the factor
                            factors_energy += factor.evaluate(conf)


                        # is the energy the current best?
                        if factors_energy < factors_best_energy:
                            factors_best_energy = factors_energy
                            best_label = label

                        # if the label is current label
                        # we rember the factors energy 
                        # to compute the total energy
                        # later on.
                        if label == current_label:
                            factors_current_energy = factors_energy


                    # this variable is now locally optimal
                    # => if this variable should be re-optimized
                    # later, and no variable in the direct
                    # neighborhood has changed we can skip the
                    # optimization
                    is_local_opt[variable] = True


                    # Did we improved the energy / did we changed the label?
                    if best_label != current_label:

                        # a label changed => we can continue to search
                        continue_search = True

                        # write the new label into the
                        # current best configuration
                        _best_labels[variable] = best_label

                        # update the the energy of _best_labels:
                        self._best_energy -= factors_current_energy
                        self._best_energy += factors_best_energy

                        # inform all neighbor variables that
                        # they are not locally optimal anymore
                        for factor in factors:
                            for other_var in factor.variables:
                                if other_var != variable:
                                    is_local_opt[other_var] = False

                        
                        # call the visitor
                        if visitor is not None:
                            # call visitor and check if
                            # we should continue
                            continue_search = visitor.visit(self)

            

            


        if visitor is not None:
            visitor.end(self)

        return _best_labels


