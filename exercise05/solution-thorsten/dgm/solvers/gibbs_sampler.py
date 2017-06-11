from __future__ import print_function, division, absolute_import

import numpy
import random
import math

class GibbsSampler(object):
    def __init__(self, model, n_iterations=1000000, temp=0.05):

        self.model = model
        self.n_iterations = n_iterations
        self.temp = temp

        self._best_labels = numpy.zeros(model.n_variables, dtype='uint32')
        self._best_energy = self.model.evaluate(self._best_labels)

        self._current_labels = self._best_labels.copy()
        self._current_energy = self._best_energy

    @property
    def best_labels(self):
        return self._best_labels

    @property
    def best_energy(self):
        return self._best_energy

    @property
    def current_labels(self):
        return self._current_labels

    @property
    def current_energy(self):
        return self._current_energy

    def _current_temp(self, iteration):
        return self.temp

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

       


        for iteration in range(self.n_iterations):

            # get a random variable (uniform distribution)
            variable = random.randint(0,model.n_variables - 1)
            

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

            
            # get a random label
            random_label = int(random.randint(0, model.n_labels(variable)-1))


            if random_label != current_label:
              
                # compute the factors energies 
                # for the current and the random label
                energies = [ ]
                for label in [current_label, random_label]:
                    factors_energy = 0.0
                    for factor, pos, conf in zip(factors, var_positions, factor_confs):
                        conf[pos] = label
                        #print("conf",conf,conf[0],type(conf[0]),"\n\n\n\n")
                        ##print("")
                        #import sys
                        #sys.exit()
                        factors_energy += factor.evaluate(conf)
                    energies.append(factors_energy)

                # compute the total energy
                energy_random_label = self._current_energy - energies[0] + energies[1]

                #print("energy_random_label",energy_random_label, self._current_energy)


                # if the energy of the random label
                # is better we accept this 
                if(energy_random_label < self._current_energy):

                    self._current_energy = energy_random_label
                    self._current_labels[variable] = random_label

                    # check if this is the very best 
                    if self._current_energy < self._best_energy:
                        self._best_energy = self._current_energy
                        self._best_labels[:] = self._current_labels

                # if the energy is worse we accept it with
                # a certain probability
                else:
                    de =  energy_random_label - self._current_energy
                    de = numpy.float64(de)
                    temp = self._current_temp(iteration)
                    p_accept = numpy.exp(-1.0*de/temp)#**(1.0/temp)
                    #print("p_acceot",p_accept)
                    if random.random() < p_accept:
                        self._current_energy = energy_random_label
                        self._current_labels[variable] = random_label

                    
            # call the visitor
            if visitor is not None:
                # call visitor and check if
                # we should continue
                continue_search = visitor.visit(self)
                if not continue_search:
                    break

                

                


        if visitor is not None:
            visitor.end(self)

        return _best_labels


