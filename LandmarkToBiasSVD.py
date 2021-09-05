# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 13:38:12 2021

@author: mairu
"""

from pandas import read_csv
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import timeit
import matplotlib.pyplot as plt 

class LandmarkToBiasSVD:
    def __init__(self,l,k, simFunc = cosine_similarity):
        self.k = k
        self.l = l
        self.simFunc = simFunc
    def landmarkselection (self,data):
        self.data=data
        self.users, count_user= np.unique(data[:,0], return_counts=True)
        zip_dic = zip(self.users, count_user)
        dic = dict(zip_dic)
        top_user=dict(Counter(dic).most_common(self.l))
        landmark_users=list(top_user.keys())
        self.items, count_item= np.unique(data[:,1], return_counts=True)
        zip_dic = zip(self.items, count_item)
        dic = dict(zip_dic)
        top_item=dict(Counter(dic).most_common(self.l))
        landmark_items=list(top_item.keys())
        return landmark_users,landmark_items
    def simlandmark_user(self, data):
        nUsers = int(np.max(data[:,0]))  +1      
        nItems = int(np.max(data[:,1])) +1
        self.data = data        
        landmark_users,landmark_items=self.landmarkselection(self.data)
        masks = np.isin( self.data[:,0],landmark_users)
        self.landmarks = self.data[masks]
        self.ybar = data.copy()
        self.xbar = self.landmarks.copy()
        ranking=list(enumerate(np.unique(self.xbar[:,0])))
        for i in range(len(np.unique(self.xbar[:,0]))):
           self.xbar[:,0]=np.where(self.xbar[:,0]==ranking[i][1], ranking[i][0],self.xbar[:,0]) 
        
        nUserslm = int(len(np.unique(self.landmarks[:,0])))       
        self.ybar = csr_matrix( (self.ybar[:,2], (self.ybar[:,1], self.ybar[:,0])),shape = (nItems, nUsers))
        self.xbar = csr_matrix( (self.xbar[:,2], (self.xbar[:,1], self.xbar[:,0])), shape = (nItems, nUserslm))
        self.landmark_sim_user = self.simFunc(self.ybar.T, self.xbar.T)
        return self.landmark_sim_user
    def simlandmark_item(self, data):
        nUsers = int(np.max(data[:,0])) +1     
        nItems = int(np.max(data[:,1]))+1
        self.data = data        
        landmark_users,landmark_items=self.landmarkselection(self.data)
        masks = np.isin( self.data[:,1],landmark_items)
        self.landmarks = self.data[masks]
        self.ybar = data.copy()
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
    def item_by_items(self, data):
        self.landmark_matrix=self.simlandmark_item(data).T
        self.similarity=self.simFunc(self.landmark_matrix, self.landmark_matrix)
        self.item_by_items = {}
        for item in range(int(max(self.items))+1):             
            ids = np.where(self.data[:,1] == item)[0].astype(int)
            itemRated = (self.data[ids, 1]).astype(int)
            sim = self.similarity[item, itemRated]
            nns = np.argsort(sim)[-self.k:].astype(int)
            for i in nns:
                self.item_by_items.setdefault(item, []).append(i)
        
        return self.item_by_items
        
    def fit(self, data,test,  epochs, alpha = 0.005, lamdaV = 0.005, lamdaH = 0.005,lamdaB = 0.005, lamdaO = 0.005, lamdaP = 0.005):
        
        csr = csr_matrix( (data[:,2], (data[:,0], data[:,1])))
        R = csr.tocoo()
        m, n = R.shape
        self.H = np.random.rand(m, self.l)
        self.V = np.random.rand(self.l, n)     
        self.B = np.random.rand( self.l,n)
        self.o = np.random.rand(m)
        self.p = np.random.rand(n)
        item_by_items=self.item_by_items(data) 
        ratings = R.data   
        users, items = R.nonzero()
        self.muy = sum(ratings)/ len(ratings)
        rmse_train_summary=[]
        rmse_test_summary=[]
        self.vPlusB = {}

        for epoch in range(epochs):
            s_err = 0
            ae=0
            start = timeit.default_timer()
            for u, i in zip(users, items): 
                rating = R.tocsr()[u,i]                
                h=np.zeros(self.l)
                for j in item_by_items[i]: 
                    h = np.add(h,self.B[:,j].T)
                self.vPlusB[i] = np.add(self.V[:,i],h / np.sqrt(self.k) )  
                err = rating - self.o[u] - self.p[i] - self.muy - np.dot(self.H[u,:],self.vPlusB[i].T)
               
                Ht = self.H[u,:] - alpha * (lamdaH * self.H[u,:] - err * self.vPlusB[i])
                Vt = self.V[:,i] - alpha * (lamdaV * self.V[:,i] - err * self.H[u,:])
                for j in item_by_items[i]:
                    self.B[:,j] = self.B[:,j]- alpha * (lamdaB * self.B[:,j]-err * 1 / np.sqrt(5) * self.H[u,:] )
                self.o[u] -= alpha * (lamdaO * self.o[u] - err)
                self.p[i] -= alpha * (lamdaP * self.p[i] - err)
                self.H[u,:] = Ht
                self.V[:,i] = Vt
                s_err += err ** 2
                ae += abs(err)
            rmse_train=np.sqrt(s_err/(len(ratings)))
            rmse_train_summary.append(rmse_train)
            print('Error Epoch',epoch,':', s_err,'        RMSE Epoch',epoch,':', rmse_train,'        MAE Epoch',epoch,':', ae/(len(ratings)))
            end = timeit.default_timer()
            print('RunTime',end - start)
            mse_test = 0
            ae_test=0
            for u_test, m_test, r_test in zip(test['userId'], test['movieId'], test['rating']):
                pred = self.o[u_test] + self.p[m_test] + self.muy + np.dot(self.H[u_test,:],self.vPlusB[m_test].T)
                mse_test += (pred - r_test) ** 2
                ae_test += abs(pred - r_test)
            rmse_test = np.sqrt(mse_test / test.shape[0])
            mae_test = ae_test / test.shape[0]
            rmse_test_summary.append(rmse_test)
            print('Error testing',epoch,':', mse_test,'        RMSE testing', rmse_test,'        MAE testing', mae_test)  
        f,(ax1) = plt.subplots(1,1,sharex=True,figsize =(10,4) )
        ax1.plot(rmse_train_summary)#blue 
        ax1.plot(rmse_test_summary)#green 
        ax1.set_title('Root mean square error')
        plt.xlabel('Epochs(x1)') 
        plt.show()   
    def predict(self, u, i):
        return self.o[u] + self.p[i] + self.muy + np.dot(self.H[u,:],self.vPlusB[i].T)