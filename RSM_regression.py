import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.stats import f
from scipy.optimize import minimize
import itertools

class RSM_regression(object):

    def __init__(self, codedUnits = None, centroids = None):
        self.codedUnits = codedUnits
        self.centroids = centroids

    def variable_engineer(self,X, modelType = 'SO'):

        self.N,self.D = X.shape
        self.modelType = modelType

        df_X = pd.DataFrame(X, columns = ['X' + str(i) for i in range(self.D)])

        if self.modelType == 'FO':
            self.df_X = df_X.copy(deep = True)
            self.df_X['ones'] = 1

        elif self.modelType == 'SO':
            self.df_X = df_X.copy(deep = True)
            col_combs_all = itertools.combinations_with_replacement(self.df_X.columns,2)
            for col_comb in col_combs_all:
                self.df_X[str(col_comb[0]) + str(col_comb[1])] = self.df_X[col_comb[0]]*self.df_X[col_comb[1]]
            self.df_X['ones'] = 1

        elif self.modelType == 'INT':
            self.df_X = df_X.copy(deep = True)
            col_combs = itertools.combinations(self.df_X.columns,2)
            for col_comb in col_combs_inter:
                self.df_X[str(col_comb[0]) + str(col_comb[1])] = self.df_X[col_comb[0]]*self.df_X[col_comb[1]]
            self.df_X['ones'] = 1

        elif self.modelType == 'MIX':
            self.df_X = df_X.copy(deep = True)
            col_combs = itertools.combinations(self.df_X.columns,2)
            for col_comb in col_combs:
                self.df_X[str(col_comb[0]) + str(col_comb[1])] = self.df_X[col_comb[0]]*self.df_X[col_comb[1]]
            self.df_X['X0X1X2'] = self.df_X['X0']*self.df_X['X1']*self.df_X['X2']

        #self.X_decoded_FO = self.decodeUnits(self.X_first_order)
        #print(self.df_X_second_order)
        #print(self.decoded_FO_X)
        self.X = self.df_X.values
        return self.df_X, self.X


    def decodeUnits(self, X):
        X[:,0] = X[:,0]*25.0 + 67.5
        X[:,1] = X[:,1]*14.0 + 49.0
        return X

    def weights(self,X,Y,l1=None,learning_rate=0.001,l2=0):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if l1 == None:
            return np.linalg.solve(l2*np.eye(X.shape[1]) + (X.T).dot(X),(X.T).dot(Y))
        else:
            W = np.random.random(X.shape[1])/np.sqrt(X.shape[1])
            error = []
            for i in range(10000):
                Yhat = X.dot(W)
                delta = Yhat - Y
                W = W - learning_rate*(X.T.dot(delta) + l1*np.sign(W))
                mse = delta.dot(delta)
                error.append(mse)
            plt.plot(error)
            plt.show()
            return W

    def forward(self,X,W):
        if isinstance(X, pd.DataFrame):
            X = X.values
        #print('W: ',W)
        return X.dot(W)

    def get_stats(self,df_X,l1=None,learning_rate=0.001,l2=0):

        W = self.weights(df_X,self.Y,l1,learning_rate,l2)
        Yhat = self.forward(df_X, W)

        df_X['Y'] = self.Y
        groupmeans = df_X.groupby(['X' + str(i) for i in range(self.D)]).transform('mean')

        df_X['Y_groupmean'] = groupmeans['Y']
        Ygroupmean = df_X['Y_groupmean'].values

        SSRes = (self.Y - Yhat).dot(self.Y - Yhat) # Variation due to residual error
        SSTotal = (self.Y - self.Y.mean()).dot(self.Y - self.Y.mean()) #variation in the data without any predictor variables
        SSReg = (Yhat - self.Y.mean()).dot(Yhat - self.Y.mean()) #Variation due to regression
        SSLof = (Ygroupmean - Yhat).dot(Ygroupmean - Yhat)
        SSPE = (self.Y - Ygroupmean).dot(self.Y - Ygroupmean)

        print('SSLof: ', SSLof, 'SSPE: ', SSPE)

        self.residuals = self.Y - Yhat

        #DF (degrees of freedom)
        p = df_X.values.shape[1] - 2 # need to take away the Y and the groupmean columns that were added on
        m = len(df_X.groupby(['X' + str(i) for i in range(self.D)]).mean().index)
        print('m: ', m, 'p: ', p)
        degFree_Reg = (p - 1)
        degFree_Res = (self.N - p)

        #mean squares
        MSReg = SSReg/degFree_Reg
        MSRes = SSRes/degFree_Res

        #Statistics
        F_reg = MSReg/MSRes
        p_value_reg = f.sf(F_reg, p-1,self.N - p)
        r2 = 1 - SSRes/SSTotal

        #Lack of fit
        MSLof = SSLof/(m - p)
        MSPE = SSPE/(self.N - m)
        F_Lof = MSLof/MSPE # Lack of fit F statistic
        p_value_Lof = f.sf(F_Lof,m - p,self.N - m)

        # F-test for weights (compare full model to reduced model without each weight)
        if p > 2:
            df_X_reduced =  self.df_X.iloc[:,:-2]
            #W_reduced = W
            reduced_F = []
            reduced_p = []
            SS_Sq_all = []
            for i in range(p):
                #holder = W[i] #remember the weight that is about to be set to 0
                #W_reduced[i] = 0 #set the weight to 0
                df_X_reduced = df_X_reduced.drop(df_X_reduced.columns[i], axis = 1)

                W_reduced = self.weights(df_X_reduced,self.Y,l1,learning_rate,l2)
                Yhat_reduced = self.forward(df_X_reduced, W_reduced)

                #Yhat_reduced = self.forward(df_X.iloc[:,:-2], W_reduced) # predict on reduced model
                #W_reduced[i] = holder

                SSRes_reduced = (self.Y - Yhat_reduced).dot(self.Y - Yhat_reduced)
                SSReg_reduced = (Yhat_reduced - self.Y.mean()).dot(Yhat_reduced - self.Y.mean()) #Variation due to regression (how far the model is away from the no relationship situation (the mean))

                MSRes_reduced = SSRes_reduced/(self.N - p - 1) # one of the weights is set to 0 so there is one less degrees of freedom
                MSReg_reduced = SSReg_reduced/(p - 2) # one of the weights is set to 0 so there is one less degrees of freedom

                #F_reduced = MSReg_reduced/MSRes
                degFree_Res_reduced = degFree_Res + 1 #one less parameter
                SS_Sq = SSRes_reduced - SSRes
                F_reduced = ( ((SS_Sq)/(degFree_Res_reduced - degFree_Res))/(SSRes/degFree_Res) )
                p_value_weight = f.sf(F_reduced, (degFree_Res_reduced - degFree_Res), degFree_Res)
                reduced_F.append(F_reduced)
                reduced_p.append(p_value_weight)
                SS_Sq_all.append(SS_Sq)

                df_X_reduced =  self.df_X.iloc[:,:-2]

        if p > 2:
            df_results_weightStats = pd.DataFrame({'weight': df_X.iloc[:,:-2].columns, 'SS Sq': SS_Sq_all, 'F-statistic': reduced_F, 'p-value': reduced_p})
        else:
            df_results_weightStats = 'None'
        df_results = pd.DataFrame({'type':['Regression','Lack of fit'],'F-statistic': [F_reg,F_Lof], 'p-value': [p_value_reg, p_value_Lof], 'r-squared':[r2,'-'], 'SS':[SSReg,SSLof] })
        df_results.set_index('type',inplace = True)

        print(df_results_weightStats)
        print(df_results)
        print('SSTotal: ', SSTotal)
        return df_results_weightStats, df_results

    def getResiduals(self):
        return self.residuals

    def getWeights(self):
        return dict(zip(self.df_X.columns, self.W))

    def evaluateModel(self,X):
        X = self.variable_engineer(X)[1]
        return X.dot(self.W)

    def process(self,Y,l1=None,learning_rate=0.001,l2=0):
        self.Y = Y
        
        self.W = self.weights(self.X,self.Y,l1,learning_rate,l2)

        self.Yhat = self.forward(self.df_X,self.W)
        
        print('==========================================================')
        print('                 Regression stats')
        print('==========================================================')
        print('Model type: ', str(self.modelType))
        self.get_stats(self.df_X)
        print('==========================================================')

        return self.W

    def forwardPlot(self,X,W):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return W[0]*((X[0] - 67.5)/25) + W[1]*((X[1] - 49)/14) + W[2]*((X[0] - 67.5)/25)**2 + W[3]*((X[0] - 67.5)/25)*((X[1] - 49)/14) \
         + W[4]*((X[1] - 49)/14)**2 + W[5]

    def plot3D(self, n_factor = 2, Max = [1.41, 1.41], Min = [-1.41,-1.41], a = 0, x_label = '', y_label = '', z_label = '', title = ''):

        fig = plt.figure()
        ax = fig.add_subplot(111,projection = '3d')
        ax.scatter(self.X[:,0], self.X[:,1], self.Y,color = 'r')

        if self.modelType == 'MIX':
            x0 = np.linspace(0,1,50)
            x1 = np.linspace(1,0,50)
            XX0,XX1 = np.meshgrid(x0,x1)

        else:
            x0 = np.linspace(Min[0],Max[0],50)
            x1 = np.linspace(Min[1],Max[1],50)
            XX0,XX1 = np.meshgrid(x0,x1)

        if self.modelType == 'SO':
            if n_factor == 2:
                ys = np.array([self.forwardPlot(np.array([X0,X1,X0*X0,X0*X1,X1*X1,1]),self.W) for X0,X1 in zip(np.ravel(XX0),np.ravel(XX1))])
                ys_colour_restraint = np.array([(-2.52926*((x - 49)/14) + 0.01168709*((x - 49)/14)**2 + 185.52329) for x in np.linspace(Min[1],Max[1],50)])
            if n_factor == 3:
                ys = np.array([self.forwardPlot(np.array([X0,X1,a,X0*X0,X0*X1,X0*a,X1*X1,X1*a,a*a,1]),self.W) for X0,X1 in zip(np.ravel(XX0),np.ravel(XX1))])
        elif self.modelType == 'INT':
            ys = np.array([self.forward(np.array([X0,X1,X0*X1,1]),self.W) for X0,X1 in zip(np.ravel(XX0),np.ravel(XX1))])
        elif self.modelType =='FO':
            ys = np.array([self.forward(np.array([X0,X1,1]),self.W) for X0,X1 in zip(np.ravel(XX0),np.ravel(XX1))])

        elif self.modelType == 'MIX':
            ys = np.array([self.forward(np.array([X0, X1, (1 - X0 -X1), X0*X1, X0*(1 - X0 -X1), X1*(1 - X0 - X1), X0*X1*(1 - X0 -X1)]),self.W) for X0,X1 in zip(np.ravel(XX0),np.ravel(XX1))])

        Z = np.array([4.92561585 for X0,X1 in zip(np.ravel(XX0),np.ravel(XX1))])


        ys = ys.reshape(XX1.shape)
        if self.modelType == 'MIX':
            ys[XX0 + XX1 >= 1] = np.nan

        ax.plot_surface(XX0,XX1,ys, cmap = cm.winter,alpha = 0.6)
        #ax.plot(x1,ys_colour_restraint, color = 'black')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
        plt.title(title)

        plt.show()


    def maxObjective_n2(self,X):
        X2 = np.array([X[0],X[1],X[0]*X[0],X[0]*X[1],X[1]*X[1],1])
        return -(X2.dot(self.W_SO))

    def minObjective_n2(self,X):
        X2 = np.array([X[0],X[1],X[0]*X[0],X[0]*X[1],X[1]*X[1],1])
        return (X2.dot(self.W_SO))

    def optimise_n2(self, minimise = False):
        b0 = (1,1)
        b1 = (-5,5)

        bnds = (b1,b1)
        X_init = np.random.randn(2,1)
        if minimise == True:
            objective = self.minObjective_n2
        else:
            objective = self.maxObjective_n2
        sol = minimize(objective,X_init,method = 'SLSQP',bounds = bnds)
        print(sol)
        return sol

    def objective_n3(self,X):
        X2 = np.array([X[0],X[1],X[2],X[0]*X[0],X[0]*X[1],X[0]*X[2],X[1]*X[1],X[1]*X[2],X[2]*X[2],1])
        return -(X2.dot(self.W_SO))

    def optimise_n3(self):
        b0 = (1,1)
        b1 = (-1.5,1.5)

        bnds = (b1,b1,b1)
        X_init = np.random.randn(3,1)
        sol = minimize(self.objective_n3,X_init,method = 'SLSQP',bounds = bnds)
        print(sol)
        return sol
