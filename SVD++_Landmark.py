# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 13:38:12 2021

@author: mairu
"""

from scipy.sparse import csr_matrix
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import matplotlib.pyplot as plt 

class SVDPlusPlus:
    def __init__(self,train,test,l, simFunc = cosine_similarity):
        self.data = train.values
        self.test=test
        self.l = l
        self.simFunc = simFunc        
    def item_by_users(self,train):    
        item_by_users = {}
        for line in train.itertuples():    
            item_by_users.setdefault(line[1], []).append(line[2])
    def landmarkselection(self,data):
        users, count_user= np.unique(self.data[:,0], return_counts=True)
        zip_dic = zip(users, count_user)
        dic = dict(zip_dic)
        top_user=dict(Counter(dic).most_common(self.l))
        landmark_users=list(top_user.keys())
        items, count_item= np.unique(self.data[:,1], return_counts=True)
        zip_dic = zip(items, count_item)
        dic = dict(zip_dic)
        top_item=dict(Counter(dic).most_common(self.l))
        landmark_items=list(top_item.keys())
        return landmark_users,landmark_items
    def simlandmark_user(self):
        nUsers = int(np.max(self.data[:,0])) + 1        
        nItems = int(np.max(self.data[:,1])) + 1
        landmark_users,landmark_items=self.landmarkselection(self.data)
        masks = np.isin( self.data[:,0],landmark_users)
        self.landmarks = self.data[masks]
        self.ybar = self.data.copy()
        self.xbar = self.landmarks.copy()
        ranking=list(enumerate(np.unique(self.xbar[:,0])))
        #print(ranking)
        for i in range(len(np.unique(self.xbar[:,0]))):
           self.xbar[:,0]=np.where(self.xbar[:,0]==ranking[i][1], ranking[i][0],self.xbar[:,0]) 
        
        nUserslm = int(len(np.unique(self.landmarks[:,0])))       
        self.ybar = csr_matrix( (self.ybar[:,2], (self.ybar[:,1], self.ybar[:,0])),shape = (nItems, nUsers))
        self.xbar = csr_matrix( (self.xbar[:,2], (self.xbar[:,1], self.xbar[:,0])), shape = (nItems, nUserslm))
        self.landmark_sim_user = self.simFunc(self.ybar.T, self.xbar.T)
        return self.landmark_sim_user
    def simlandmark_item(self):
        nUsers = int(np.max(self.data[:,0])) + 1        
        nItems = int(np.max(self.data[:,1])) + 1
        landmark_users,landmark_items=self.landmarkselection(self.data)
        masks = np.isin( self.data[:,1],landmark_items)
        self.landmarks = self.data[masks]
        self.ybar = self.data.copy()
        self.xbar = self.landmarks.copy()
        ranking=list(enumerate(np.unique(self.xbar[:,1])))        
        for i in range(len(np.unique(self.xbar[:,1]))):
           self.xbar[:,1]=np.where(self.xbar[:,1]==ranking[i][1], ranking[i][0],self.xbar[:,1])         
        ranking=list(enumerate(np.unique(self.xbar[:,1])))        
        nItemslm = int(len(np.unique(self.landmarks[:,1])))        
        self.ybar = csr_matrix( (self.ybar[:,2], (self.ybar[:,1], self.ybar[:,0])),shape = (nItems, nUsers))        
        self.xbar = csr_matrix( (self.xbar[:,2], (self.xbar[:,1], self.xbar[:,0])), shape = (nItemslm, nUsers))        
        self.landmark_sim_item = self.simFunc(self.ybar, self.xbar)
        self.landmark_sim_item=self.landmark_sim_item.T        
        return self.landmark_sim_item
    def fit(self, epochs=20, alpha = 0.005, lamdaV = 0.005, lamdaH = 0.005,lamdaA = 0.005, lamdaO = 0.005, lamdaP = 0.005):
        
        csr = csr_matrix( (self.data[:,2], (self.data[:,0], self.data[:,1])))
        R = csr.tocoo()
        m, n = R.shape
        self.H = self.simlandmark_user(self.data)
        self.V = self.simlandmark_item(self.data)   
        self.A = np.random.rand(n, self.l)
        self.o = np.random.rand(m)
        self.p = np.random.rand(n)
        item_by_users=self.item_by_users(self.train)
        ratings = R.data   
        users, items = R.nonzero()
        self.muy = sum(ratings)/ len(ratings)
        rmse_train_summary=[]
        rmse_test_summary=[]
        self.hPlusA = {}
        for epoch in range(epochs):
            s_err = 0
            ae=0
            for u, i in zip(users, items): 
                rating = R.tocsr()[u,i]
                n_u = len(users[users == u])
                h=np.zeros(self.l)
                for j in item_by_users[u]:          
                    h = np.add(h,self.A[j,:])
                self.hPlusA[u] = np.add(h / np.sqrt(n_u), self.H[u,:])  
                err = rating - self.o[u] - self.p[i] - self.muy - np.dot(self.hPlusA[u], self.V[:,i].T)
                Ht = self.H[u,:] - alpha * (lamdaH * self.H[u,:] - err * self.V[:,i])
                Vt = self.V[:,i] - alpha * (lamdaV * self.V[:,i] - err * self.hPlusA[u])
                for j in item_by_users[u]:
                    self.A[j, :] = self.A[j, :]+ alpha * (err * 1 / np.sqrt(n_u) * self.V[:, i] - lamdaA * self.A[j,:])
                self.o[u] -= alpha * (lamdaO * self.o[u] - err)
                self.p[i] -= alpha * (lamdaP * self.p[i] - err)
                self.H[u,:] = Ht
                self.V[:,i] = Vt
                s_err += err ** 2
                ae += abs(err)
            rmse_train=np.sqrt(s_err/(len(ratings)))
            rmse_train_summary.append(rmse_train)
            print('Error Epoch',epoch,':', s_err,'        RMSE Epoch',epoch,':', rmse_train,'        MAE Epoch',epoch,':', ae/(len(ratings)))           
            mse_test = 0
            ae_test=0
            for u_test, m_test, r_test in zip(self.test['userId'], self.test['movieId'], self.test['rating']):
                pred = self.o[u_test] + self.p[m_test] + self.muy + np.dot(self.hPlusA[u_test], self.V[:,m_test].T)
                mse_test += (pred - r_test) ** 2
                ae_test += abs(pred - r_test)
            rmse_test = np.sqrt(mse_test / self.test.shape[0])
            mae_test = ae_test / self.test.shape[0]
            rmse_test_summary.append(rmse_test)
            print('Error testing',epoch,':', mse_test,'        RMSE testing', rmse_test,'        MAE testing', mae_test)           
        f,(ax1) = plt.subplots(1,1,sharex=True,figsize =(10,4) )
        ax1.plot(rmse_train_summary)#blue 
        ax1.plot(rmse_test_summary)#orange 
        ax1.set_title('Root mean square error')
        plt.xlabel('Epochs(x1)') 
        plt.show()  
    #return self.H,self.V,self.o,self.p
    def predict(self, u, i):
        return self.o[u] + self.p[i] + self.muy + np.dot(self.hPlusA[u], self.V[:,i].T)