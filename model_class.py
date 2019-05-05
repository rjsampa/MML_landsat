#!/usr/bin/env python
# coding: utf-8

# In[1]:


import rasterio as rio
import earthpy as et
#import earthpy.spatial as es
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import time
# # mapa base

banda6 = rio.open('../Dados/landsat/b6.tif')
# # Abrindo as bandas 
def abrindo():	
	b2 = rio.open('../Dados/landsat/g_b2.tif').read(1)
	b3 = rio.open('../Dados/landsat/g_b3.tif').read(1)
	b6 = rio.open('../Dados/landsat/g_b6.tif').read(1)
	ndvi = rio.open('../Dados/landsat/g_ndvi.tif').read(1)
	Stand = StandardScaler()
	b2 = Stand.fit(b2).transform(b2)
	b3 = Stand.fit(b3).transform(b3)
	b6 = Stand.fit(b6).transform(b6)
	df = pd.DataFrame({'b2':b2.ravel(),'b3':b3.ravel(),'b6':b6.ravel(),'ndvi':ndvi.ravel()})

	return df 

# # Aplicando modelos
def model_load():
	tree_d = joblib.load('../modelos/three_d.pk1')
	randon_f = joblib.load('../modelos/randon_f.pk1')
	svc  = joblib.load('../modelos/SVC.pk1')
	knn = joblib.load('../modelos/knn.pk1')
	return models = [tree_d, randon_f, svc, knn]


#modelos estimativa
def model_predict():
	tempo_treino = []
	for i in range(len(models)):

		t1=time.clock()
		globals()['m%s' %models[i] = models[i].predict(df.values)
		t2=time.clock()
		tf = t2-t1
		tempo_treino.append(tf)

	resultados = [m0,m1,m2,m3]

	return resultados, tempo_treino

# transformando os dados categoricos
def model_posprocessing(resultados):
	le = joblib.load('../modelos/inverse.pk1')
	r_final=[]
	tempo_transf=[]
	for i in range(len(resultados)):
		m = le.inverse_transform(resultados[i])
		t1=time.clock()
		clf =np.array([])

		for i in m:
		    if i=='AGRICULTURA':
		        c = 1    
		    if i =='AUMIDAS':
		        c=2    
		    if i=='FLORESTA':
		        c=3
		    if i=='MANGUE':
		        c=4
		    if i=='PASTAGEM':
		        c=5
		    if i=='URBANO':
		        c=6
		    if i=='SOLO':
		        c=7   
		    if i=='AGUA':
		        c=8   
		    if i=='ROCHA':
		        c=9   
		    clf=np.append(clf,c)

		r_final.append(clf)  

		t2=time.clock()
		tf = t2-t1
		tempo_transf.append(tf)
		return r_final, tempo_transf


#salvando raster
def model_save(r_final):
	nomes = ['tree_d', 'randon_f', 'svc', 'knn' ]
	for i in range(len(r_final)):
		clfm =r_final[i].reshape(1482, 1614)
		modelo1 = rio.open('../Dados/landsat/'+nomes[i],'w', driver='GTiff',
		                  height=clfm.shape[0],width=clfm.shape[1],
		                  count=1, dtype=clfm.dtype,
		                  crs='+proj=latlong',transform=banda6.transform)
		modelo1.write(clfm,1)
		modelo1.close()
	np.savetxt('treino.txt',tempo_treino)
	np.savetxt('processamento.txt',tempo_transf)