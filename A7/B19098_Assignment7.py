import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data=pd.read_csv('mnist-tsne-train.csv') #Train Data
test_data=pd.read_csv('mnist-tsne-test.csv')   #Test Data

from sklearn import metrics
from scipy.optimize import linear_sum_assignment

def purity_score(y_true, y_pred):
 # compute contingency matrix (also called confusion matrix)
 contingency_matrix=metrics.cluster.contingency_matrix(y_true, y_pred)
 # Find optimal one-to-one mapping between cluster labels and true labels
 row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
 # Return cluster accuracy
 return contingency_matrix[row_ind,col_ind].sum()/np.sum(contingency_matrix)

def Plot_PurityScore(prediction,data,x,N,cluster_centers=0): #Function for Scatter Plot and Purity Score
    colour=['grey','tan','crimson','darkorange','gold','lawngreen','yellowgreen','cyan','steelblue','orchid',
            'lightgrey','wheat','olive','seagreen','teal','slategray','lavender','slateblue','fuchsia','pink','crimson']
    plt.figure(figsize=(10,10))
    
    for i in range(N): #For loop through Clusters
        
        c=data[prediction==i] #Cluster i
        if len(c)==0:
            break
        plt.scatter(c[data.columns[0]],c[data.columns[1]],label='Cluster'+str(i),color=colour[i])   #Scatter Plot
        if x!='DBSCAN':
            plt.scatter(cluster_centers[i][0],cluster_centers[i][1],marker='*',s=250,color='black') #Cluster Centers
    
    c=data[prediction==-1] #Noise
    if len(c!=0):
        plt.scatter(c[data.columns[0]],c[data.columns[1]],label='Noise',color=colour[i+1])
    
    plt.xlabel('Dimenson 1');plt.ylabel('Dimenson 2')
    plt.legend();plt.title(x);plt.show()
    
    print('Purity Score = ',end='')
    print(purity_score(data[data.columns[-1]],prediction)) #Calling purity_score function
#QUESTION 1

#K-Means
print('\n'+'#'*28,'Question 1:','#'*28)
from sklearn.cluster import KMeans
K = 10
kmeans = KMeans(n_clusters=K,random_state=42)
kmeans.fit(train_data[train_data.columns[:-1]]) #Modeling

print('\nTraining Data:')
kmeans_prediction = kmeans.predict(train_data[train_data.columns[:-1]]) #Prediction on train data
Plot_PurityScore(kmeans_prediction,train_data,'K-Means',K,kmeans.cluster_centers_)

print('\nTest Data:')
kmeans_prediction = kmeans.predict(test_data[test_data.columns[:-1]])   #Prediction on test data
Plot_PurityScore(kmeans_prediction,test_data,'K-Means',K,kmeans.cluster_centers_)

#QUESTION 2
#GMM
print('\n'+'#'*28,'Question 2:','#'*28)
from sklearn.mixture import GaussianMixture
K = 10
gmm = GaussianMixture(n_components = K,random_state=42)
gmm.fit(train_data[train_data.columns[:-1]]) #Modeling

print('\nTraining Data:')
GMM_prediction = gmm.predict(train_data[train_data.columns[:-1]]) #Prediction on train data
Plot_PurityScore(GMM_prediction,train_data,'GMM',K,gmm.means_)

print('\nTest Data:')
GMM_prediction = gmm.predict(test_data[test_data.columns[:-1]])   #Prediction on test data
Plot_PurityScore(GMM_prediction,test_data,'GMM',K,gmm.means_)

#QUESTION 3
#DBSCAN
print('\n'+'#'*28,'Question 3:','#'*28)
from sklearn.cluster import DBSCAN
dbscan_model=DBSCAN(eps=5, min_samples=10).fit(train_data[train_data.columns[:-1]]) #Modeling

print('\nTraining Data:')
DBSCAN_predictions = dbscan_model.labels_ #Prediction on train data
Plot_PurityScore(DBSCAN_predictions,train_data,'DBSCAN',len(set(DBSCAN_predictions)))


from scipy import spatial as spatial
def dbscan_predict(dbscan_model, X_new, metric=spatial.distance.euclidean): #Prediction on test data
# Result is noise by default
    y_new = np.ones(shape=len(X_new), dtype=int)*-1 
# Iterate all input samples for a label
    for j, x_new in enumerate(X_new):
# Find a core sample closer than EPS
        for i, x_core in enumerate(dbscan_model.components_):
            if metric(x_new, x_core) < dbscan_model.eps:
# Assign label of x_core to x_new
                y_new[j] = dbscan_model.labels_[dbscan_model.core_sample_indices_[i]]
                break                
    return y_new
DBSCAN_predictions = dbscan_predict(dbscan_model, test_data[test_data.columns[:-1]].values, metric =
 spatial.distance.euclidean)

print('\nTest Data:')
Plot_PurityScore(DBSCAN_predictions,test_data,'DBSCAN',len(set(DBSCAN_predictions)))

x=input('\nPRESS THE ENTER KEY FOR BONUS QUESTIONS')
#BONUS QUESTIONS
print('\n','#'*25+' BONUS QUESTIONS: '+'#'*25)

#A
#K-Means
K=[2, 5, 8, 12, 18, 20] #K
Distortion=[] #Stores Distortion
from scipy.spatial.distance import cdist
for k in K:
    kmeans = KMeans(n_clusters=k,random_state=42)
    kmeans.fit(train_data[train_data.columns[:-1]]) #Modeling
    print('\n\t\t\tK =',k)
    print('\nTraining Data:')
    kmeans_prediction = kmeans.predict(train_data[train_data.columns[:-1]]) #Prediction on train data
    Plot_PurityScore(kmeans_prediction,train_data,'K-Means',k,kmeans.cluster_centers_)

    print('\nTest Data:')
    kmeans_prediction = kmeans.predict(test_data[test_data.columns[:-1]])   #Prediction on test data
    Plot_PurityScore(kmeans_prediction,test_data,'K-Means',k,kmeans.cluster_centers_)
    
    #Distortion
    Distortion.append(sum(np.min(cdist(train_data[train_data.columns[:-1]].values, kmeans.cluster_centers_, 'euclidean'), axis=1)) / train_data.shape[0])
plt.figure(figsize=(10,10))
plt.plot(K,Distortion) #Plot
plt.scatter(K,Distortion,marker='*',color='r')
plt.xlabel('K');plt.ylabel('Distortion')
plt.xticks(K)
plt.title('Elbow Method')
plt.show()

x=input('\nPRESS THE ENTER KEY TO CONTINUE')
#GMM
K=[2, 5, 8, 12, 18, 20]
Loglikelihood=[] #Stores log likelihood
for k in K:
    gmm = GaussianMixture(n_components = k,random_state=42)
    gmm.fit(train_data[train_data.columns[:-1]]) #Modeling
    print('\n\t\t\tK =',k)
    print('\nTraining Data:')
    GMM_prediction = gmm.predict(train_data[train_data.columns[:-1]]) #Prediction on train data
    Plot_PurityScore(GMM_prediction,train_data,'GMM',k,gmm.means_)
    
    print('\nTest Data:')
    GMM_prediction = gmm.predict(test_data[test_data.columns[:-1]])   #Prediction on test data
    Plot_PurityScore(GMM_prediction,test_data,'GMM',k,gmm.means_)
    
    Loglikelihood.append(gmm.lower_bound_) #Log-likelihood
plt.figure(figsize=(10,10))
plt.plot(K,Loglikelihood) #Plot
plt.scatter(K,Loglikelihood,marker='*',color='r')
plt.xlabel('K');plt.ylabel('Loglikelihood')
plt.title('Elbow Method')
plt.xticks(K)
plt.show()  

#B
x=input('\nPRESS THE ENTER KEY TO CONTINUE')
#DBSCAN
Eps=[1,5,10]
Min_Samples=[1,10,30,50]
for e in Eps:
    for m in Min_Samples:
        if e==1 and m==1:
            continue
        print('\n\t\teps =',e,'\tmin_samples =',m)
        dbscan_model=DBSCAN(eps=e, min_samples=m).fit(train_data[train_data.columns[:-1]]) #Modeling
        print('\nTraining Data:')
        DBSCAN_predictions = dbscan_model.labels_ #Prediction on Train Data
        Plot_PurityScore(DBSCAN_predictions,train_data,'DBSCAN',len(set(DBSCAN_predictions)))
        
        DBSCAN_predictions = dbscan_predict(dbscan_model, test_data[test_data.columns[:-1]].values, metric = spatial.distance.euclidean) #Prediction on Train Data
        print('\nTest Data:')
        Plot_PurityScore(DBSCAN_predictions,test_data,'DBSCAN',len(set(DBSCAN_predictions)))
