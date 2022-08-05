import numpy as np
from sko.base import SkoBase
from abc import ABCMeta, abstractmethod
from sko.operators import crossover, mutation, ranking, selection
from sko.GA import GeneticAlgorithmBase, GA


class Evolution(GeneticAlgorithmBase):
    def __init__(self, func, n_dim, F=0.5,
                 size_pop=50, max_iter=200, prob_mut=0.3,
                 lb=-1, ub=1,X_input=None,create_ratio=0.2,  #### 新加的参数 X_input, create_ratio
                 constraint_eq=tuple(), constraint_ueq=tuple()):
        super().__init__(func, n_dim, size_pop, max_iter, prob_mut,
                         constraint_eq=constraint_eq, constraint_ueq=constraint_ueq)
        self.create_ratio = create_ratio
        self.X_input = X_input
        self.F = F
        self.size_pop=size_pop
        self.V, self.U = None, None
        self.lb, self.ub = np.array(lb) * np.ones(self.n_dim), np.array(ub) * np.ones(self.n_dim)
        self.crtbp()

    def crtbp(self):
        '''
          创建初始population: self.X     
          create the population: self.X
        '''
        pop_    = int(self.size_pop*self.create_ratio)
        len_    = len(self.X_input)

        if len_+pop_<self.size_pop:        # 传入的X_input可能比期待的少
           pop_ = self.size_pop - len_
        elif len_+pop_>self.size_pop: 
           pop_ = self.size_pop - len_

        if pop_>0:
           X_      = np.random.uniform(low=self.lb, high=self.ub, size=(pop_, self.n_dim))
           merge_  = np.vstack((self.X_input,X_))
           self.X  = merge_
        elif pop_==0:
           self.X = self.X_input
        else:
           raise RuntimeError('The current population is larger than max defination!')

        print('The length of current population:',len(self.X))
        return self.X

    def chrom2x(self, Chrom):
        pass

    def ranking(self):
        pass

    def mutation(self):
        '''
        V[i]=X[r1]+F(X[r2]-X[r3]),
        where i, r1, r2, r3 are randomly generated
        '''
        X = self.X
        # i is not needed,
        # and TODO: r1, r2, r3 should not be equal
        random_idx = np.random.randint(0, self.size_pop, size=(self.size_pop, 3))

        r1, r2, r3 = random_idx[:, 0], random_idx[:, 1], random_idx[:, 2]

        # 这里F用固定值，为了防止早熟，可以换成自适应值
        self.V = X[r1, :] + self.F * (X[r2, :] - X[r3, :])

        # the lower & upper bound still works in mutation
        mask = np.random.uniform(low=self.lb, high=self.ub, size=(self.size_pop, self.n_dim))
        self.V = np.where(self.V < self.lb, mask, self.V)
        self.V = np.where(self.V > self.ub, mask, self.V)
        return self.V

    def crossover(self):
        '''
        if rand < prob_crossover, use V, else use X
        '''
        mask = np.random.rand(self.size_pop, self.n_dim) < self.prob_mut
        self.U = np.where(mask, self.V, self.X)
        return self.U

    def selection(self):
        '''
        greedy selection
        '''
        X = self.X.copy()
        f_X = self.x2y().copy()
        self.X = U = self.U
        f_U = self.x2y()

        self.X = np.where((f_X < f_U).reshape(-1, 1), X, U)
        return self.X

    def run(self,logfile=None):
        #self.max_iter = max_iter or self.max_iter
        print('',file=logfile)
        print('                  The Scores of the candidates     ',file=logfile)
        print('----------------------------------------------------------------',file=logfile)
        for i in range(self.max_iter):
            self.mutation()
            self.crossover()
            self.selection()

            # record the best ones
            generation_best_index = self.Y.argmin()
            self.generation_best_X.append(self.X[generation_best_index, :].copy())
            self.generation_best_Y.append(self.Y[generation_best_index])
            self.all_history_Y.append(self.Y)
            if i%5==0:
               print(' ',file=logfile)
            print(' {:9.4f} '.format(self.generation_best_Y[-1]),end='',file=logfile)

        print('\n----------------------------------------------------------------',file=logfile)
        print('',file=logfile)
        global_best_index = np.array(self.generation_best_Y).argmin()
        global_best_X = self.generation_best_X[global_best_index]
        global_best_Y = self.func(np.array([global_best_X]))
        return global_best_X, global_best_Y
