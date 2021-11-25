#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

from sklearn.model_selection import KFold  

#import the boston housing dataset 
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

#store breast cancer data into X_data matrix 
X_data, t = load_breast_cancer(return_X_y=True)

#split data into training set and test set
from sklearn.model_selection import train_test_split

#split the data into validation and training data 
X_train, X_valid, t_train, t_valid = train_test_split(X_data, t, test_size = 0.2, random_state = 2700)

#normalize training and testing data 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_valid = sc.transform(X_valid)

#suppress scientific notation
np.set_printoptions(suppress=True)

#import math library to perform square root 
import math


# In[2]:


#obtain weights for decision boundary with gradient descent
def compute_W(X_train, t_train, alpha, IT):

    N = X_train.shape[0] #number of training points 
    M = X_train.shape[1] #number of features 
    X0 = np.ones(N) #create dummy feature of ones 
    X_train = np.insert(X_train, 0, X0, axis=1)#insert dummy feature into training
    
    #initialize variables 
    z = np.zeros(N) #distances from decision boundary 
    W = np.ones(M + 1) #weights
    
    for n in range(IT):
        z = np.dot(X_train, W) #update current z
        y = 1/(1+np.exp(-z)) #probability it belongs to class 0 
        diff = y - t_train #compute difference between y and target
        gr = np.dot(X_train.T, diff)/N #compute gradient 
    
        #update current weights using batch gradient descent 
        W = W - alpha*gr

    return W


# In[3]:


#compute predictions based off of input data, computed weights from training data
#and threshold for distance from boundary to predict that cancer is malignant

def compute_predictions(X, W_train, thres):
    N = X.shape[0] #number of training points
    X0 = np.ones(N) #dummy vector 
    X = np.insert(X, 0, X0, axis=1) #insert dummy feature into X matrix 
    
    z = np.dot(W_train.T, X.T) #compute vector of distances from decision boundary 
    
    for i in range(N):
    
        if z[i] >= thres: #if example is above or equal to threshold then say cancerous 
            
            z[i] = 1
            
        else: #if example is below threshold then say non-cancerous
            z[i] = 0

    return z #return set of predictions 


# In[4]:


#compute thresholds for all z in training dataset 
def compute_thresholds(X_train, W_train):
    N = X_train.shape[0] #number of training points
    X0 = np.ones(N) #dummy vector 
    
    X_train = np.insert(X_train, 0, X0, axis=1) #insert dummy feature into X matrix 
    z  = np.dot(W_train.T, X_train.T) #compute vector of distances from decision boundary 
    z = np.sort(z) #sort thresholds from smallest to biggest 
    return  z #return sorted z thresholds from training data  


# In[5]:


#returns the misclassification rate between predictions and true targets 
#returns difference matrix where entries that contain 1's 
def MC_rate(y, t):
    
    N = y.shape[0] #number of datapoints
    
    #compute absolute difference between prediction and true target 
    #every element with entry of 1 is misclassification
    diff = np.absolute(y - t) 

    #sum this vector to compute total misclassifications
    errors = np.sum(diff) 
    
    MC = (errors/N)*100 #compute misclassification rate 
    
    return round(MC, 3) #return misclassification rate (%)


# In[6]:


def plot_thres_vs_MC(X_valid, t_valid, W, thresholds):
    
    N = thresholds.shape[0] #number of z thresholds 
    MC_valid_list = np.zeros(N) #create array that stores each validation misclassification rates 

    for i in range(N): #iterate over each threshold and compute MC_rate 
        y_curr = compute_predictions(X_valid, W, thresholds[i]) #current set of predicitions from current threshold
        MC_valid_list[i] = MC_rate(y_curr, t_valid) #add current misclassification to list
        
    plt.title("Z THRESHOLDS vs. VALIDATION MISCLASSIFICATION RATE")
    plt.xlabel("Threshold (Theta)")
    plt.ylabel("Misclassification Rate (%)")
    plt.scatter(thresholds, MC_valid_list)
    plt.show()


# In[7]:


#compute number of true positives 
def TP(t):
    return np.sum(t) #sum up positive entries in target matrix 


# In[8]:


#compute number of false positives 
def FP(y, t):
    FP = 0
    N = y.shape[0]
    for i in range(N): #iterate over each entry in both y and t
        if (t[i] == 0 and y[i] == 1): #if true target is benign and prediction is cancerous it's a FP
            FP += 1
    return FP


# In[9]:


#compute number of false negatives 
def FN(y, t):
    FN = 0
    N = y.shape[0]
    for i in range(N): #iterate over each entry in both y and t
        if (t[i] == 1 and y[i] == 0): #if true target is cancerous and prediction is benign is be it's a FN
            FN += 1
    return FN


# In[10]:


#computes the recall between predictions and true targets 
def recall(y, t):
    return TP(t)/(TP(t) + FP(y, t))


# In[11]:


#computes the precision between predictions and true targets 
def precision(y, t):
    return TP(t)/(TP(t) + FN(y, t))


# In[12]:


#compute F1
def compute_F1(y, t):
    return (2*precision(y,t)*recall(y,t))/(precision(y,t)+recall(y,t))


# In[13]:


#compute P, R datapoints for each threshold computed from training dataset 
def PR_curve(X, t, W, thresholds):
    
    N = thresholds.shape[0] #number of threshold data points
    P = np.zeros(N) #precision and recall entries 
    R = np.zeros(N)
    y_curr = np.zeros(N)
    
    for i in range(N): #compute P, R for each threshold 
        y_curr = compute_predictions(X, W, thresholds[i])
        P[i] = precision(y_curr, t) #compute
        R[i] = recall(y_curr, t)
    
    return P, R #return precision and recall sets 


# In[14]:


#plot own implementation of logistic regression PR curve
def plot_PR(P, R):
    plt.title("PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("precision")
    plt.scatter(R, P)
    plt.show()


# In[15]:


#compute F1 curve threshold values and F1 scores for each threshold value 
def F1_curve(X, t, W, thresholds):
    
    N = thresholds.shape[0]
    F1 = np.zeros(N) 
    y_curr = np.zeros(N)
    
    for i in range(N):
        y_curr = compute_predictions(X, W, thresholds[i])
        F1[i] = compute_F1(y_curr, t)
    
    return F1


# In[16]:


#plot F1 score for own implementation of logistic regression  
def plot_F1(thresholds, F1):
    plt.title("Threshold vs. F1")
    plt.xlabel("Threshold (Theta)")
    plt.ylabel("F1")
    plt.scatter(thresholds, F1)
    plt.show()


# In[17]:


def compute_min_threshold(X_train, X_valid, t_train, W_train, thresholds):
    thresholds = compute_thresholds(X_train, W_train) #compute all possible thresholds from training   
    N = thresholds.shape[0] #number of threshold values 
    MC_LIST = np.zeros((1, N)) #create empty list that will house each misclassification
    
    #iterate through each threshold and compute MC_rate
    for i in range(N): 
        
        y_curr = compute_predictions(X_valid, W_train, thresholds[i]) #get current prediction value
        MC_LIST[:, i] = MC_rate(y_curr, t_train)
    
    min_index = np.argmin(MC_LIST) #find index of lowest misclassification rate 
    return thresholds[min_index] #threshold that gives the best classificiation rate  


# In[18]:


#compute the highest F1 Score 
def compute_max_F1(F1):
    return round(np.amax(F1)*100, 3) #return highest F1 score


# In[19]:


#compute threshold that gives the best F1 score 
def compute_threshold_max_F1(thresholds, F1):
    max_index = np.argmax(F1) #determine 
    return thresholds[max_index] #determine threshold value that results in best F1 score


# In[20]:


#use built-in Logisitic Regression from Sci-Kit Learn
from sklearn.linear_model import LogisticRegression


# In[21]:


#compute vector of parameters for SKL Logistic regression model
def SKL_weights(X_train, t_train):
    
    N = t_train.shape[0] #number of data points 
    X0 = np.ones(N) #dummy vector 
    X_train = np.insert(X_train, 0, X0, axis=1) #insert dummy feature into X matrix 
    
    clf = LogisticRegression().fit(X_train, t_train) #fit data based on training data
    W_SKL = clf.coef_ #get vector of parameters
    return W_SKL.T[:, 0] #return vector of parameters


# In[22]:


#utilize Sci Kit Learn library to compute predictions 
def SKL_predictor(X_train, X_valid, t_train, t_valid, thres):
    
    N = t_train.shape[0] #number of data points 
    X0 = np.ones(N) #dummy vector 
    X_train = np.insert(X_train, 0, X0, axis=1) #insert dummy feature into X matrix 
    
    clf = LogisticRegression().fit(X_train, t_train) #fit data based on training data
    
    y = clf.predict_proba(X_valid) #predict outputs based on validation data 
    
    for i in range(N):
        if y[i, 1] >= thres:
            y[i, 1] = 1 #if probability is above threshold then say cancerous 
        
        else:
            y[i, 1] = 0 #if probability is below or equal to threshold then non-cancerous
    
    return y[:, 1]#return set of predictions 


# In[23]:


#BEGINNING OF KNN PORTION OF ASSIGNMENT 

#calculate the euclidean distances of points
#X_train and t_train will be used to compute the nearest neighbours for X_valid inputs
def distance(X_valid, X_train):
    
    N = X_valid.shape[1] #number of features
    distances = np.zeros((X_valid.shape[0], X_train.shape[0])) #create distance matrix
    
    for i in range(X_valid.shape[0]): #iterate over each validation example
        
        X_V_curr = X_valid[i:i+1, :] #extract current row vector
        
        for j in range(X_train.shape[0]): #iterate over each training example
            
            d = 0 #set current distance to 0
            X_T_curr = X_train[j:j+1, :] #extract current row vector

            for k in range(N): #iterate over each feature
                d += (X_V_curr[0, k] - X_T_curr[0, k])**2 #compute distance of current vector
            
            d = math.sqrt(d) #compute current valid-training point
            distances[i, j] = d #store current valid-training point in matrix 
 
    #each row -> validation examples
    #each column -> training examples
    #Eg. distances[1,3] corresponds to the distance between valid pt 1 and training pt 3 
    return distances #return the matrix of distances


# In[24]:


def KNN_indices(curr_row, K): 
    return np.argsort(curr_row)[:K] #return indices of KNN for current validation pt


# In[25]:


#compute classes of KNN for current validation point 
def compute_t_KNN(curr_row, t_train, K):
    
    #compute indices of KNN
    indices = KNN_indices(curr_row, K)
    t_KNN = np.zeros((K, 1)) #store classes of KNN
    
    for i in range(K): #iterate over indices of closest NN
        NN_index = indices[i] #index of NN
        t_KNN[i] =  t_train[NN_index]#add closest target values
        
    return t_KNN #compute KNN classes for current validation point 


# In[26]:


def compute_avg_dist(curr_row, t_train, t_KNN, K): #compute average distance of + and - class KNN
     
    num_pos = np.sum(t_KNN) #number of positive classes out of KNN
    num_neg = K - num_pos #number of negative classes KNN
    
    total_pos = 0 #total of positive class distances 
    total_neg = 0 #total of negative class distances 
    
    KNN_ind = KNN_indices(curr_row, K)

    for index in KNN_ind: #iterate through closest neighbour indices
        if (t_train[index] == 1): #if neighbour is positive class 
            total_pos += curr_row[index] #add distance value at index 
        else: #if neighbour is negative class (0)
            total_neg += curr_row[index] #add distance value at index 

    avg_pos = total_pos/num_pos
    avg_neg = total_neg/num_neg
    
    return avg_pos, avg_neg


# In[27]:


def NN_scenarios(curr_row, t_train, t_KNN):
    
    K = t_KNN.shape[0] #number of nearest neighbours we consider 
    
    #compute avg distance for +/- KNN
    avg_pos, avg_neg = compute_avg_dist(curr_row, t_train, t_KNN, K)

    num_pos = np.sum(t_KNN) #number of positive classes out of K closest neighbours
    num_neg = K - num_pos #number of negative classes K closest neighbours

    if num_pos != num_neg: #if number of num positive classes != negative         
        return 1 if num_pos > num_neg else 0 

    else: #if num of pos. and neg classes are equal
        if (avg_pos != avg_neg): #if average distance are not the same
            return 0 if avg_pos > avg_neg else 1 #use avg distance of points  
        
        else: #if avg pos distance and neg distance are the same
            t_KNN_next = compute_t_KNN(curr_row, t_train, K+1)
            return NN_scenarios(curr_row, t_train, t_KNN_next) #increase to K+1 NN  


# In[28]:


#own implementation of KNN 
def KNN(X_valid, X_train, t_train, K):
    distances = distance(X_valid, X_train) #compute distances of X_valid and train points
    num_valid = distances.shape[0] #number of valid. pts is # of rows of distance matrix
    y_NN = np.zeros((num_valid, 1)) #create predictor vector
    
    for i in range(num_valid): #iterate over each validation point 
        
        curr_row = distances[i, :] #look at each training point for current valid pt i
        t_KNN = compute_t_KNN(curr_row, t_train, K) #compute classes of KNN for current valid pt i 
        y_NN[i, 0] = NN_scenarios(curr_row, t_train, t_KNN) #determine class using KNN
        
    return y_NN[:, 0] #return predictions 


# In[29]:


def K_folds(X, t, K):
    
    #create K folds in the data 
    kf = KFold(n_splits=K)
    KFold(n_splits=2, random_state=None, shuffle=False)
    
    #create array to hold each training fold
    K_X_train = []
    K_t_train = []
    
    #create array to hold each validation fold 
    K_X_valid = []
    K_t_valid = []

    #iterate over split input data
    for train_index, valid_index in kf.split(X):

       # print("TRAIN:", train_index, "VALID:", valid_index)
        
        #add current folds to training and validation lists
        K_X_train.append(X[train_index])
        K_X_valid.append(X[valid_index])
        
        K_t_train.append(t[train_index])
        K_t_valid.append(t[valid_index])

    #return arrays that contain each testing and validation matrices for each K fold
    return K_X_train, K_t_train, K_X_valid, K_t_valid


# In[30]:


#input sorted list of nearest neighbour parameters (K)

#compute the best K for KNN based on MC Rate on training set  
def best_KNN_K(X_train, t_train, num_folds, K_list):
    
    K_X_train, K_t_train, K_X_valid, K_t_valid = K_folds(X_train, t_train, num_folds)
    
    N = len(K_list)
    error = np.zeros((N, 1)) #list that will store cross fold valid. error for each K in knn
    
    for i in range(N):
        
        num_NN = K_list[i] #update number of NN to be used
        fold_error = 0 #initialize fold error to 0
        
        for fold in range(num_folds): #iterate over the number of total folds
            
            #get current X valid,train and t valid and train for each fold  
            X_valid_curr = np.array(K_X_valid[fold]) 
            X_train_curr = np.array(K_X_train[fold])
            t_train_curr = np.array(K_t_train[fold])
            t_valid_curr = np.array(K_t_valid[fold])

            #compute y based on current fold
            y_curr = KNN(X_valid_curr, X_train_curr, t_train_curr, num_NN)
            
            #compute misclassification error based on current fold 
            MC_curr = MC_rate(y_curr, t_valid_curr) 
            fold_error += MC_curr #running total of error on current fold

        #add average cross valid error to error list 
        K_fold_error = fold_error/num_folds 
        error[i, 0] = K_fold_error
    
    #determine index of error matrix that returns the smallest misclassification rate 
    min_K_index = np.argmin(error) 
    
    #return best selected error using MC cross validation and error for each K
    return K_list[min_K_index], error


# In[31]:


from sklearn.neighbors import KNeighborsClassifier


# In[32]:


#compute predictions using KNN SKL implementation
def SKL_KNN(X_train, X_valid, t_train, K):
    SKL_KNN_ = KNeighborsClassifier(n_neighbors = K)
    SKL_KNN_.fit(X_train, t_train) #create KNN Model based of training data
    y_SKL_KNN = SKL_KNN_.predict(X_valid) #predict outputs of "testing"/validation data
    
    return y_SKL_KNN


# In[33]:


#input sorted list of nearest neighbour parameters (K)

#compute best K for SKL KNN based on misclassification rate 
def SKL_best_KNN_K(X_train, t_train, num_folds, K_list):
    K_X_train, K_t_train, K_X_valid, K_t_valid = K_folds(X_train, t_train, num_folds)
     
    N = len(K_list) #number of K's 
    error = np.zeros((N, 1)) #list to store total cross valid error values for each K
    
    for i in range(N):
        
        num_NN = K_list[i] #update number of NN to be used
        fold_error = 0 #initialize fold error to 0
        
        for fold in range(num_folds): #iterate over each fold 
            
            #get current X valid,train and t valid and train for each fold  
            X_valid_curr = np.array(K_X_valid[fold])
            X_train_curr = np.array(K_X_train[fold])
            t_train_curr = np.array(K_t_train[fold])
            t_valid_curr = np.array(K_t_valid[fold])
            
            #compute current fold prediction for current K
            y_curr = SKL_KNN(X_train_curr, X_valid_curr, t_train_curr, K_list[i])
            
            #compute misclassification error based on current fold 
            MC_curr = MC_rate(y_curr, t_valid_curr) 
            fold_error += MC_curr #running total of misclassification errors 

        #add average cross valid error to error list 
        K_fold_error = fold_error/num_folds
        error[i, 0] = K_fold_error
    
    #determine index of error matrix that returns the smallest misclassification rate 
    min_K_index = np.argmin(error)
    
    #return best selected error using MC cross validation and error for each K
    return K_list[min_K_index], error


# In[34]:


alpha, IT = 0.5, 100 #parameters for gradient descent 

print("***************** LOGISTIC REGRESSION CLASSIFIER BUILT FROM SCRATCH ***********************")
print("\n")

W_train = compute_W(X_train, t_train, alpha, IT) #compute weight for decision boundary 
thresholds = compute_thresholds(X_valid, W_train) #compute thresholds for training 

#compute best threshold value based on MC Rate 
thres_min = compute_min_threshold(X_train, X_valid, t_valid, W_train, thresholds) 

#compute predictions with both logistic regression and SKL model  
Y_pred = compute_predictions(X_valid, W_train, thres_min)

#plot threshold vs misclassification rate 
plot_thres_vs_MC(X_valid, t_valid, W_train, thresholds) 
prec, re = PR_curve(X_valid, t_valid, W_train, thresholds) #plot PR Curve
plot_PR(prec, re)

#compute F1 datapoints and plot F1 curve vs threshold
F1 = F1_curve(X_valid, t_valid, W_train, thresholds)
plot_F1(thresholds, F1)

best_F1_score = compute_max_F1(F1)
best_thres = compute_threshold_max_F1(thresholds, F1) 

print("BEST THRESHOLD based on MC RATE", thres_min)
print("BEST THRESHOLD based on F1:", best_thres)
print("\n")

print("Best 'Accuracy'/Lowest MC Rate:",100 - MC_rate(Y_pred, t_valid), "or", MC_rate(Y_pred, t_valid), "% misclassification")
print("Best F1 Score:", best_F1_score)


# In[35]:


print("******************* SKL LOGISTIC REGRESSION CLASSIFIER ********************************")
print("\n")
#compute SKL model weights 
W_SKL = SKL_weights(X_train, t_train)

#compute SKL thresholds 
SKL_thres = compute_thresholds(X_valid, W_SKL) 

#compute best threshold value to get minimum MC_rate 
min_thres_SKL = compute_min_threshold(X_train, X_valid, t_valid, W_SKL, SKL_thres)

#compute predictions with best threshold 
Y_SKL = compute_predictions(X_valid, W_SKL, min_thres_SKL)

plot_thres_vs_MC(X_valid, t_valid, W_SKL, SKL_thres) #plot threshold vs misclassification rate 

prec_SKL, re_SKL = PR_curve(X_valid, t_valid, W_SKL, SKL_thres) #plot PR Curve
plot_PR(prec_SKL, re_SKL)

F1_SKL = F1_curve(X_valid, t_valid, W_SKL, SKL_thres) #plot F1 Curve
plot_F1(SKL_thres, F1_SKL)


best_F1_SKL = compute_max_F1(F1_SKL)
best_thres_SKL = compute_threshold_max_F1(SKL_thres, F1_SKL) 

print("BEST THRESHOLD based on Best MC RATE", min_thres_SKL)
print("BEST THRESHOLD based on Best F1 SCORE:", best_thres_SKL)
print("\n")

print("Best 'Accuracy'/Lowest MC Rate:",100 - MC_rate(Y_SKL, t_valid), "or", MC_rate(Y_SKL, t_valid), "% misclassification")
print("Best F1 Score:", best_F1_SKL)


# In[36]:


print("******************* COMPUTE WEIGHTS OF BOTH LOGISTIC REGRESSION MODELS: ********************************")

print("Weights (Own Implementation):")
print(W_train)
print("\n")

print("SKL Weights")
print(W_SKL)


# In[37]:


print("***************** KNN CLASSIFIER BUILT FROM SCRATCH ************************")

K_list = [1,2,3,4,5] #list of K's to compute F1 and MC error 
num_folds = 10 #number of folds to compute misclassification/F1 score

#compute the best K using MC Rate, and the array that contains MC Rate for each K
best_K_MC, MC_K = best_KNN_K(X_train, t_train, num_folds, K_list)

#compute predictions using best_K selected using MC
y_KNN = KNN(X_valid, X_train, t_train, best_K_MC)

#print misclassification rates
print("Training Cross Validation Misclassification Rates for each K:")
print(MC_K)

#print K vs. cross validation MC Rate
plt.title("K vs Training Cross Validation MC Rate")
plt.xlabel("K")
plt.ylabel("MC Rate (%)")
plt.scatter(K_list, MC_K)
plt.show()

#compute validation misclassification rate between predictions and true target
KNN_MC_rate = MC_rate(y_KNN, t_valid)
KNN_accuracy = 100 - KNN_MC_rate 

#compute F1 scorevalidation 
KNN_F1_score = compute_F1(y_KNN, t_valid)

#print best K when using Misclassification 
print("Best K determined using Cross Validation:", best_K_MC)

#print computed misclassification rate when using the best K
print("Validation 'Accuracy'/MC Rate:", KNN_accuracy, "% or", KNN_MC_rate, "%")

print("Validation F1 Score:", KNN_F1_score*100, "%")


# In[38]:


print("***************** SKL KNN CLASSIFIER ************************")

K_list = [1,2,3,4,5] #list of K's to compute F1 and MC error 
num_folds = 10 #number of folds to compute misclassification/F1 score

#compute the best K using MC Rate, and the array that contains MC Rate for each K
best_K_MC_SKL, MC_K_SKL = SKL_best_KNN_K(X_train, t_train, num_folds, K_list)

#compute predictions using best_K selected using MC
y_SKL_KNN = SKL_KNN(X_train, X_valid, t_train, best_K_MC_SKL)

#print misclassification rates
print("Training Cross Validation Misclassification Rates for each K:")
print(MC_K_SKL)

#print K vs. cross validation MC Rate
plt.title("K vs Training Cross Validation MC Rate")
plt.xlabel("K")
plt.ylabel("MC Rate (%)")
plt.scatter(K_list, MC_K_SKL)
plt.show()

#compute validation misclassification rate between predictions and true target
SKL_KNN_MC_rate = MC_rate(y_SKL_KNN, t_valid)
SKL_KNN_accuracy = 100 - SKL_KNN_MC_rate 

#compute F1 scorevalidation 
SKL_KNN_F1_score = compute_F1(y_SKL_KNN, t_valid)

#print best K when using Misclassification 
print("Best K determined using Cross Validation:", best_K_MC_SKL)

#print computed misclassification rate when using the best K
print("Validation 'Accuracy'/MC Rate:", SKL_KNN_accuracy, "% or", SKL_KNN_MC_rate, "%")

print("Validation F1 Score:", SKL_KNN_F1_score*100, "%")

