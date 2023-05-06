import random
import numpy as np
import types
from sklearn.cluster import KMeans


class Evolution:
    def __init__(self, func, n_dim, F=0.5,
                 size_pop=50, max_iter=200, prob_mut=0.5,
                 X_input=None,scale=None,n_clusters=1,
                 constraint_eq=tuple(), constraint_ueq=tuple()):
        self.func = func # func_transformer(func)
        assert size_pop % 2 == 0, 'size_pop must be even integer'
        self.size_pop       = size_pop               # size of population
        self.max_iter       = max_iter
        self.prob_mut       = prob_mut               # probability of mutation
        self.n_dim          = n_dim

        # constraint:
        self.has_constraint = len(constraint_eq) > 0 or len(constraint_ueq) > 0
        self.constraint_eq  = list(constraint_eq)    # a list of equal functions with ceq[i] = 0
        self.constraint_ueq = list(constraint_ueq)   # a list of unequal constraint functions with c[i] <= 0

        self.Chrom   = None
        self.X       = None # shape = (size_pop, n_dim)
        self.Y_raw   = None # shape = (size_pop,) , value is f(x)
        self.Y       = None # shape = (size_pop,) , value is f(x) + penalty for constraint
        self.FitV    = None # shape = (size_pop,)

        # self.FitV_history = []
        self.generation_best_X = []
        self.generation_best_Y = []

        self.all_history_Y = []
        self.all_history_FitV = []

        self.best_x, self.best_y = None, None

        # self.create_ratio = create_ratio
        self.X_input = X_input
        self.F = F
        self.size_pop=size_pop
        self.V, self.U = None, None
        # self.lb, self.ub = np.array(lb) * np.ones(self.n_dim), np.array(ub) * np.ones(self.n_dim)
        self.scale = scale
        self.n_clusters = n_clusters
        self.crtbp()

    def crtbp(self):
        '''
          创建初始                  族群: self.X     
          create the inital population: self.X
        '''
        len_       = len(self.X_input)
        n_clusters = self.n_clusters if len_>self.n_clusters else len_
        pop_       = self.size_pop - len_
        self.X     = self.X_input
        Y_         = self.x2y()
        min_y_id   = np.argmin(Y_)
        X_template = self.X_input[min_y_id]          # generate data with Gaussian distribution at X_template 

        self.generation_best_X.append(X_template)
        self.generation_best_Y.append(Y_[min_y_id])

        if pop_>0:
           # X_    = np.random.uniform(low=self.lb, high=self.ub, size=(pop_, self.n_dim))  
           # Using a Gaussian Distribution instead of uniform distrution 使用高斯分布代替均匀分布
           XS = [self.X_input]
           size_pop = int(pop_/n_clusters)

           if n_clusters>1:
              random.seed()
              kmeans = KMeans(n_clusters=n_clusters, random_state=random.randint(0,10)).fit(self.X)

              for i in range(n_clusters):
                  size   = size_pop  if i != n_clusters-1 else pop_-size_pop*i
                  index_ = np.squeeze(np.where(kmeans.labels_==i))
                  i_     = index_  if len(index_.shape)==0 or isinstance(index_,int) else index_[0] # 
                  X_     = np.random.normal(loc=self.X[i_], scale=self.scale, size=(size, self.n_dim))
                  XS.append(X_)
           else:
              X_  = np.random.normal(loc=X_template, scale=self.scale, size=(pop_, self.n_dim))
              XS.append(X_)
           self.X  = np.vstack(XS)

        elif pop_==0:
           self.X = self.X_input
        else:
           raise RuntimeError('The current population is larger than max defination!')
        # print('The length of current population:',len(self.X))
        return self.X

    def register(self, operator_name, operator, *args, **kwargs):
        '''
        regeister udf to the class
        :param operator_name: string
        :param operator: a function, operator itself
        :param args: arg of operator
        :param kwargs: kwargs of operator
        :return:
        '''

        def operator_wapper(*wrapper_args):
            return operator(*(wrapper_args + args), **kwargs)

        setattr(self, operator_name, types.MethodType(operator_wapper, self))
        return self

    def x2y(self):
        self.Y_raw = self.func(self.X)
        if not self.has_constraint:
            self.Y = self.Y_raw
        else:
            # constraint
            penalty_eq = np.array([np.sum(np.abs([c_i(x) for c_i in self.constraint_eq])) for x in self.X])
            penalty_ueq = np.array([np.sum(np.abs([max(0, c_i(x)) for c_i in self.constraint_ueq])) for x in self.X])
            self.Y = self.Y_raw + 1e5 * penalty_eq + 1e5 * penalty_ueq
        return self.Y

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
        # mask = np.random.uniform(low=self.lb, high=self.ub, size=(self.size_pop, self.n_dim))
        # self.V = np.where(self.V < self.lb, mask, self.V)
        # self.V = np.where(self.V > self.ub, mask, self.V)
        # Using a Gaussian Distribution instead of uniform distrution        使用高斯分布代替均匀分布
        self.V = np.random.normal(loc=self.generation_best_X[-1], scale=self.scale, size=(self.size_pop, self.n_dim))
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
        # print('',file=logfile)
        # print('                  The Scores of the candidates     ',file=logfile)
        # print('----------------------------------------------------------------',file=logfile)
        for i in range(self.max_iter):
            self.mutation()
            self.crossover()
            self.selection()

            # record the best ones
            generation_best_index = self.Y.argmin()
            self.generation_best_X.append(self.X[generation_best_index, :].copy())
            self.generation_best_Y.append(self.Y[generation_best_index])
            self.all_history_Y.append(self.Y)
            # if i%5==0:
            #    print(' ',file=logfile)
            # print(' {:9.6f} '.format(self.generation_best_Y[-1]),end='',file=logfile)

        # print('\n----------------------------------------------------------------',file=logfile)
        # print('',file=logfile)
        global_best_index = np.array(self.generation_best_Y).argmin()
        global_best_X = self.generation_best_X[global_best_index]
        global_best_Y = self.func(np.array([global_best_X]))
        return global_best_X, global_best_Y


