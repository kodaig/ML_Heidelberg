from __future__ import print_function, division, absolute_import

import numpy
import networkx

import scipy.optimize

try:
    from scipy.optimize  import linprog
except:
    from scipy.optimize  import _linprog 
    from _linprog import linprog


class LpSolver(object):
    def __init__(self, model):
        self.model = model

        if self.model.max_arity > 2:
            raise RuntimeError("currently only implemented for second order models")



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

        # set up the lp
        model = self.model


        # we will set up the number of lp variables incrementally
        n_lp_var = 0
        # mapping from model variables to lp variables
        model_var_to_lp_var_begin = dict()
        # mapping from second order factors to lp variables
        second_order_factor_to_lp_var_begin = dict()


        #################################################
        # SET UP VARIABLES
        #################################################

        # ``first order'' indicator variables
        for var in range(model.n_variables):
            model_var_to_lp_var_begin[var] = n_lp_var
            n_lp_var += model.n_labels(var)

        # ``second order'' indicator variables
        for fac in model.factors:
            if fac.arity == 2:
                second_order_factor_to_lp_var_begin[fac] = n_lp_var
                n_lp_var += fac.size

        #################################################
        # SET UP COEFFICIENTS TO BE MINIMIZED
        #################################################

        # coefficients to be minimized
        coefficients = numpy.zeros(n_lp_var)

        for factor in model.factors:

            # set up coefficients for unary factors
            if factor.arity == 1:
                var = factor.variables[0]
                lp_var_begin = model_var_to_lp_var_begin[var]

                # cost for each label
                for l in range(model.n_labels(var)):
                    c =  factor.evaluate([l])
                    coefficients[lp_var_begin+l] += c


            # set up coefficients for 2-order factors
            elif factor.arity == 2:
                
                lp_var = second_order_factor_to_lp_var_begin[factor]

                var0 = factor.variables[0]
                var1 = factor.variables[1]

                # cost for each label
                for l0 in range(model.n_labels(var0)):
                    for l1 in range(model.n_labels(var1)):

                        c =  factor.evaluate((l0, l1))
                        coefficients[lp_var] = c
                        lp_var += 1

        #################################################
        # SET UP CONSTRAINTS
        #################################################

        # list of all constraints
        A = []
        eq = []
    
        # sum constraints
        for var in range(model.n_variables):
            single_constraint = numpy.zeros(n_lp_var, 'float32')
            lp_var = model_var_to_lp_var_begin[var]
            for l in range(model.n_labels(var)):
                single_constraint[lp_var+l] = 1.0
            A.append(single_constraint)
            eq.append(1.0)

        # consistency constraints
        for factor in model.factors:

            if factor.arity == 2:

                var0 = factor.variables[0]
                var1 = factor.variables[1]

                lp_var_begin = second_order_factor_to_lp_var_begin[factor]

                # remember all states which 

                lp_vars_per_label = [ 
                    [[] for n_labels in range(model.n_labels(var1)) ],
                    [[] for n_labels in range(model.n_labels(var1)) ]
                ]

                
                for l0 in range(model.n_labels(var0)):
                    for l1 in range(model.n_labels(var1)):

                        lp_vars_per_label[0][l0].append(lp_var_begin)
                        lp_vars_per_label[1][l1].append(lp_var_begin)

                        lp_var_begin += 1

                for i in range(2):
                    for label,lp_vars in enumerate(lp_vars_per_label[i]):


                        single_constraint = numpy.zeros(n_lp_var, 'float32')
                        for lp_var in lp_vars:
                            single_constraint[lp_var] = 1.0

                        # get the first order indicator variable
                        lp_var = model_var_to_lp_var_begin[factor.variables[i]]
                        single_constraint[lp_var+label] = -1.0

                        A.append(single_constraint)
                        eq.append(0)

        A = numpy.array(A)
        eq = numpy.array(eq)


        # lower and upper bounds for the variables
        var_lb_ub = numpy.zeros([n_lp_var,2])
        var_lb_ub[:,1] = 1

        optimize_result = linprog(
            c=coefficients,
            A_eq=A,
            b_eq=eq,
            bounds=var_lb_ub
        )

        lp_var_values = optimize_result.x
        print(optimize_result.success)
        print(lp_var_values.shape)
        success = optimize_result.success

        # translate the lp_var_labels to model labels
        for var in range(model.n_variables):

            lp_var_begin = model_var_to_lp_var_begin[var]
            
            max_lp_var_value = -1.0
            max_label = None

            for l in range(model.n_labels(var)):
                lp_var_value = lp_var_values[lp_var_begin+l] 
                if lp_var_value > max_lp_var_value:
                    max_lp_var_value = lp_var_value
                    max_label = l 

                self._best_labels[var] = max_label

        print(lp_var_values)

        if visitor is not None:
            visitor.end(self)

        return self._best_labels


