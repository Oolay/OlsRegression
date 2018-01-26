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

    def variable_engineer(self,X):

        self.N,self.D = X.shape

        df_X = pd.DataFrame(X, columns = ['X' + str(i) for i in range(self.D)])
        self.df_X_first_order = df_X.copy(deep = True)
        self.df_X_first_order['ones'] = 1
        self.X_first_order = self.df_X_first_order.values

        self.df_X_second_order = df_X.copy(deep = True)
        col_combs_all = itertools.combinations_with_replacement(self.df_X_second_order.columns,2)
        for col_comb in col_combs_all:
            self.df_X_second_order[str(col_comb[0]) + str(col_comb[1])] = self.df_X_second_order[col_comb[0]]*self.df_X_second_order[col_comb[1]]
        self.df_X_second_order['ones'] = 1
        self.X_second_order = self.df_X_second_order.values

        self.df_X_inter = df_X.copy(deep = True)
        col_combs_inter = itertools.combinations(self.df_X_inter.columns,2)
        for col_comb in col_combs_inter:
            self.df_X_inter[str(col_comb[0]) + str(col_comb[1])] = self.df_X_inter[col_comb[0]]*self.df_X_inter[col_comb[1]]
        self.df_X_inter['ones'] = 1
        self.X_inter = self.df_X_inter.values

        '''self.df_X_mixture = df_X.copy(deep = True)
        col_combs_mixture = itertools.combinations(self.df_X_mixture.columns,2)
        for col_comb in col_combs_mixture:
            self.df_X_mixture[str(col_comb[0]) + str(col_comb[1])] = self.df_X_mixture[col_comb[0]]*self.df_X_mixture[col_comb[1]]
        self.df_X_mixture['X0X1X2'] = self.df_X_mixture['X0']*self.df_X_mixture['X1']*self.df_X_mixture['X2']'''

        self.X_decoded_FO = self.decodeUnits(self.X_first_order)
        #print(self.df_X_second_order)
        #print(self.decoded_FO_X)
        return self.df_X_second_order,self.df_X_inter,self.df_X_first_order, self.X_second_order,self.X_inter,self.X_first_order, self.X_decoded_FO


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
        df_X_second_order_reduced =  self.df_X_second_order.iloc[:,:-2]
        #W_reduced = W
        reduced_F = []
        reduced_p = []
        SS_Sq_all = []
        for i in range(p):
            #holder = W[i] #remember the weight that is about to be set to 0
            #W_reduced[i] = 0 #set the weight to 0
            df_X_second_order_reduced = df_X_second_order_reduced.drop(df_X_second_order_reduced.columns[i], axis = 1)

            W_reduced = self.weights(df_X_second_order_reduced,self.Y,l1,learning_rate,l2)
            Yhat_reduced = self.forward(df_X_second_order_reduced, W_reduced)

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

            df_X_second_order_reduced =  self.df_X_second_order.iloc[:,:-2]

        df_results_weightStats = pd.DataFrame({'weight': df_X.iloc[:,:-2].columns, 'SS Sq': SS_Sq_all, 'F-statistic': reduced_F, 'p-value': reduced_p})
        df_results = pd.DataFrame({'type':['Regression','Lack of fit'],'F-statistic': [F_reg,F_Lof], 'p-value': [p_value_reg, p_value_Lof], 'r-squared':[r2,'-'], 'SS':[SSReg,SSLof] })
        df_results.set_index('type',inplace = True)

        print(df_results_weightStats)
        print(df_results)
        print('SSTotal: ', SSTotal)
        return df_results_weightStats, df_results

    def getResiduals(self):
        return self.residuals

    def getWeights(self):
        return dict(zip(self.df_X_second_order.columns, self.W_SO)), self.W_SO

    def solveSecondOrder(self,X):
        X = self.variable_engineer(X)[3]
        return X.dot(self.W_SO)

    def solvedeCodedSO(self,X):
        X = np.array([X[0],X[1],X[0]*X[0],X[0]*X[1],X[1]*X[1],1])
        return X.dot(self.WdeCoded)

    def getDeCodedWeights(self):
        WdeCoded0 = ( (self.W_SO[0]/self.codedUnits[0]) - ((self.W_SO[2]*(2)*self.centroids[0])/(self.codedUnits[0])**2) - (self.W_SO[3])*(self.centroids[1])/(self.codedUnits[0]*self.codedUnits[1]) )
        WdeCoded1 = ( (self.W_SO[1]/self.codedUnits[1]) - ((self.W_SO[4]*(2)*self.centroids[1])/(self.codedUnits[1])**2) - (self.W_SO[3])*(self.centroids[0])/(self.codedUnits[0]*self.codedUnits[1]) )
        WdeCoded2 = (self.W_SO[2]/(self.codedUnits[0])**2)
        WdeCoded3 = (self.W_SO[3]/(self.codedUnits[0]*self.codedUnits[1])) #x0x1 term
        WdeCoded4 = (self.W_SO[4]/(self.codedUnits[1])**2)
        WdeCoded5 = ( self.W_SO[5] - (self.W_SO[0]*self.centroids[0]/self.codedUnits[0]) - (self.W_SO[1]*self.centroids[1]/self.codedUnits[1]) + (self.W_SO[2]*(self.centroids[0]**2)/(self.codedUnits[0]**2)) \
        + (self.W_SO[4]*(self.centroids[1]**2)/(self.codedUnits[1]**2)) + (self.W_SO[3]*self.centroids[0]*self.centroids[1]/(self.codedUnits[0]*self.codedUnits[1])) )

        self.WdeCoded = np.array([WdeCoded0,WdeCoded1,WdeCoded2,WdeCoded3,WdeCoded4,WdeCoded5])

        return self.WdeCoded

    def process(self,Y,l1=None,learning_rate=0.001,l2=0):
        self.Y = Y
        #self.Y = np.reshape(self.Y,(Y.shape[0],1))

        self.W_FO = self.weights(self.X_first_order,self.Y,l1,learning_rate,l2)
        self.W_int = self.weights(self.X_inter,self.Y,l1,learning_rate,l2)
        self.W_SO = self.weights(self.X_second_order,self.Y,l1,learning_rate,l2)

        self.Yhat_FO = self.forward(self.df_X_first_order,self.W_FO)
        self.Yhat_int = self.forward(self.df_X_inter,self.W_int)
        self.Yhat_SO = self.forward(self.df_X_second_order,self.W_SO)
        #print(self.Yhat_FO)'''

        #print('First order','\n',self.get_stats(self.df_X_first_order),'\n')
        #print('Interactions', '\n',self.get_stats(self.df_X_inter),'\n')
        #print('Second order', '\n',self.get_stats(self.df_X_second_order),'\n')
        print('==========================================================')
        print('                 Regression stats')
        print('==========================================================')
        print('First order')
        self.get_stats(self.df_X_first_order)
        print('==========================================================')
        print('Interactions')
        self.get_stats(self.df_X_inter)
        print('==========================================================')
        print('Second order')
        self.get_stats(self.df_X_second_order)
        print('==========================================================')

    def forwardPlot(self,X,W):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return W[0]*((X[0] - 67.5)/25) + W[1]*((X[1] - 49)/14) + W[2]*((X[0] - 67.5)/25)**2 + W[3]*((X[0] - 67.5)/25)*((X[1] - 49)/14) \
         + W[4]*((X[1] - 49)/14)**2 + W[5]

    def plot3D(self,model = 'SO', n_factor = 2, Max = [1.41, 1.41], Min = [-1.41,-1.41], a = 0, x_label = '', y_label = '', z_label = '', title = ''):

        fig = plt.figure()
        ax = fig.add_subplot(111,projection = '3d')
        #ax.scatter(self.X_decoded_FO[:,0], self.X_decoded_FO[:,1], self.Y,color = 'r')

        x0 = np.linspace(Min[0],Max[0],50)
        x1 = np.linspace(Min[1],Max[1],50)
        XX0,XX1 = np.meshgrid(x0,x1)

        if model == 'SO':
            if n_factor == 2:
                ys = np.array([self.forwardPlot(np.array([X0,X1,X0*X0,X0*X1,X1*X1,1]),self.W_SO) for X0,X1 in zip(np.ravel(XX0),np.ravel(XX1))])
            if n_factor == 3:
                ys = np.array([self.forwardPlot(np.array([X0,X1,a,X0*X0,X0*X1,X0*a,X1*X1,X1*a,a*a,1]),self.W_SO) for X0,X1 in zip(np.ravel(XX0),np.ravel(XX1))])
        elif model == 'int':
            ys = np.array([self.forward(np.array([X0,X1,X0*X1,1]),self.W_int) for X0,X1 in zip(np.ravel(XX0),np.ravel(XX1))])
        elif model =='FO':
            ys = np.array([self.forward(np.array([X0,X1,1]),self.W_FO) for X0,X1 in zip(np.ravel(XX0),np.ravel(XX1))])

        Y_plot = ys.reshape(XX1.shape)

        ax.plot_surface(XX0,XX1,Y_plot, cmap = cm.autumn,alpha = 0.6)
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
