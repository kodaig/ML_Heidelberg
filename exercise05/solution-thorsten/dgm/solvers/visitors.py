from __future__ import print_function, division, absolute_import

import time

class Visitor(object):    
    def __init__(self, visit_nth=1, name=None, verbose=1, time_limit=None):
        self._iter = 0
        self.t = None
        self.dt = 0.0
        self.visit_nth = visit_nth
        self.name = name
        if name is None:
            self.name = ""
        self.verbose = int(verbose)
        self.time_limit = time_limit
        if self.time_limit is None:
            self.time_limit = float('inf')
    def start(self, solver):
        self.t = time.time()
        if self.verbose:
            print("Start %s:"%self.name)
    def end(self, solver):
        t = time.time()
        self.dt += t - self.t
        dt = round(self.dt, 3)
        if self.verbose:
            print("Finished %s:"%self.name)
            print(dt,'sec',"E",solver.best_energy)
    def visit(self, solver):
        continue_inference = True
        if (self._iter + 1) % self.visit_nth == 0:
            t = time.time()
            self.dt += t - self.t
            if self.dt >= self.time_limit:
                continue_inference = False
            if self.verbose:
                dt = round(self.dt, 3)
                print(dt,'sec','iter',self._iter,"E",solver.best_energy, solver.current_energy)
            self.t = time.time()

        self._iter += 1
        return continue_inference
