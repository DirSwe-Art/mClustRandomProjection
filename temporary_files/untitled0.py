# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 11:35:10 2024

@author: DirarTempAccount
"""


from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd

import copy, math, time, datetime, os, sys



'''
def aggregate(G):
	# dictionary for sample pairwise equality comparisons in each clustering
	dict_ = {}
	for s_id, S in enumerate(G):
		dict_[s_id]= np.equal.outer(S,S);print(s_id,'EqualOuter')
	
	xS = pd.DataFrame({}, index=range(len(G)), columns=range(len(G[0])))
	xC = pd.DataFrame({}, index=range(len(G[0])), columns=range(len(G[0])))
    
	print('len G0',len(G[0]))
	for x_id in range(len(G[0])):
		for s_id, S in enumerate(G):
			xS.iloc[s_id] = dict_[s_id][x_id]
		sums  = xS.sum(axis=0)
		xC.iloc[x_id] = sums
		xS = pd.DataFrame({}, index=range(len(G)), columns=range(len(G[0])))
		print('x', x_id)
	print('\n*** duration',datetime.timedelta(seconds=(time.time()-starting_time)),' ***')
		
	return GaussianMixture(n_components=len(set(G[0]))).fit_predict(xC).tolist()
'''



'''
def aggregate(G):
	# dictionary for sample pairwise equality comparisons in each clustering
	dict_ = {}
	for s_id, S in enumerate(G):
		exec('dict_'+str(s_id)+'={}')
		globals()['dict_'+str(s_id)][s_id] = np.equal.outer(S,S);print(s_id,'EqualOuter')
	
	xS = pd.DataFrame({}, index=range(len(G)), columns=range(len(G[0])))
	xC = pd.DataFrame({}, index=range(len(G[0])), columns=range(len(G[0])))
    
	print('len G0',len(G[0]))
	for x_id in range(len(G[0])):
		for s_id, S in enumerate(G):
			xS.iloc[s_id] = globals()['dict_'+str(s_id)][s_id]
			#xS.append(dict_[s_id][x_id])
		sums  = xS.sum(axis=0)
		xC.iloc[x_id] = sums
		xS = pd.DataFrame({}, index=range(len(G)), columns=range(len(G[0])))
		print('x', x_id)
	print('\n*** duration',datetime.timedelta(seconds=(time.time()-starting_time)),' ***')
		
	return GaussianMixture(n_components=len(set(G[0]))).fit_predict(xC).tolist()

'''

'''
# dictionary for sample pairwise equality comparisons in each clustering
def aggregate(G):
	for s_id, S in enumerate(G):
		exec('global dict_'+str(s_id))
		globals()['dict_G_S'+str(s_id)]={}
		globals()['dict_G_S'+str(s_id)][s_id] = np.equal.outer(S,S);print(s_id,'EqualOuter')
	
	xS = pd.DataFrame({}, index=range(len(G)), columns=range(len(G[0])), dtype=bool)
	xC = pd.DataFrame({}, index=range(len(G[0])), columns=range(len(G[0])), dtype=int)
	    
	print('len G0',len(G[0]))
	for x_id in range(len(G[0])):
		for s_id, S in enumerate(G):
			xS.iloc[s_id] = globals()['dict_G_S'+str(s_id)][s_id][x_id]
		sums  = xS.sum(axis=0)
		xC.iloc[x_id] = sums
		xS = pd.DataFrame({}, index=range(len(G)), columns=range(len(G[0])))
		print('x', x_id)
	
	print('\n*** duration',datetime.timedelta(seconds=(time.time()-starting_time)),' ***')

	return GaussianMixture(n_components=len(set(G[0]))).fit_predict(xC).tolist()
'''

'''

# better performance, map large arrays to external files, better for memory
def aggregate(G):
	# dictionary for sample pairwise equality comparisons in each clustering
	dict_ = {}
	for s_id, S in enumerate(G):
		dict_[s_id]= np.equal.outer(S,S);print(s_id,'EqualOuter')
	
	file1 = 'xS_.dat' 
	file2 = 'xC_.dat'
    
	print('len G0',len(G[0]))
	for x_id in range(len(G[0])):
		for s_id, S in enumerate(G):
			
			xS = np.memmap( file1, dtype='bool', mode='w+', shape= (len(G),len(G[0])) ) # Writing data to disk #
			xS[s_id, :] = dict_[s_id][x_id]
			xS.flush()  # Freeing memory
		
		xS = np.memmap(file1, dtype='bool', mode='r', shape= (len(G),len(G[0])) )
		sums  = np.sum(np.array(xS), axis=0)
		
		xC    = np.memmap(file2, dtype='int', mode='w+', shape= (len(G[0]),len(G[0])) ) # Writing data to disk #
		xC[x_id, :] = sums
		xC.flush()  # Freeing memory
		print('x', x_id)
		
	print('\n*** duration',datetime.timedelta(seconds=(time.time()-starting_time)),' ***')
	
	xC = np.memmap(file2, dtype='int', mode='r', shape= (len(G[0]),len(G[0])) ) # Writing data to disk #
		
	return GaussianMixture(n_components=len(set(G[0]))).fit_predict(xC).tolist()
'''


starting_time = time.time()
G = np.array([
	[1,1,1,0,1,0,0,1,1,0,1],
	[0,0,0,1,0,1,1,0,0,1,0],
	[0,0,0,0,0,0,0,0,0,0,0]
	])

def aggregate(G):
	# dictionary for sample pairwise equality comparisons in each clustering
	dict_ = {}
	for s_id, S in enumerate(G):
		dict_[s_id]= np.equal.outer(S,S);print(s_id,'EqualOuter')
	
	xS = pd.DataFrame( {}, columns=range(len(G))    )
	xC = pd.DataFrame( {}, columns=range(len(G[0])) )
	
	print('xS dict:\n',xS)
	print('xC dict:\n',xC)
	
	    
	print('len G0',len(G[0]))
	for x_id in range(len(G[0])):
		for s_id, S in enumerate(G):
			xS[s_id] = dict_[s_id][x_id]
		sums         = xS.sum(axis=1)
		xC[x_id] = sums
		xS = pd.DataFrame( {}, columns=range(len(G))  )
		print('x', x_id)
	print('\n*** duration',datetime.timedelta(seconds=(time.time()-starting_time)),' ***')
	
	print(G)
	print(GaussianMixture(n_components=len(set(G[0]))).fit_predict(xC).tolist())
	
aggregate(G)


'''
# dictionary for sample pairwise equality comparisons in each clustering
dict_ = {}
for s_id, S in enumerate(G):
	dict_[s_id]= np.equal.outer(S,S);print(s_id,'EqualOuter')

    
print('len G0',len(G[0]))
for x_id in range(len(G[0])):
	for s_id, S in enumerate(G):
		
		xS = np.memmap( 'xS_.dat' , dtype='bool', mode='w+', shape= (len(G),len(G[0])) ) # Writing data to disk #
		xS[s_id, :] = dict_[s_id][x_id]
		del xS  # Freeing memory
	
	xS = np.memmap('xS_.dat' , dtype='bool', mode='r', shape= (len(G),len(G[0])) )
	sums  = np.sum(np.array(xS), axis=0)
	
	xC    = np.memmap('xC_.dat', dtype='int', mode='w+', shape= (len(G[0]),len(G[0])) ) # Writing data to disk #
	xC[x_id, :] = sums
	del xC  # Freeing memory
	print('x', x_id)
	
print('\n*** duration',datetime.timedelta(seconds=(time.time()-starting_time)),' ***')

xC = np.memmap('xC_.dat', dtype='int', mode='r', shape= (len(G[0]),len(G[0])) ) # Writing data to disk #
	
print( GaussianMixture(n_components=len(set(G[0]))).fit_predict(xC).tolist())

'''

'''
def representativesSorted(DATA, R):
	sortedR = [ [] for _ in range(len(R)) ]
	for i, S in enumerate(R):
		sortedR[i] = large_labels_first(DATA, S)
	return sortedR

'''














