import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# Taking the training data
df = pd.read_excel('./Training_Data.xlsx')
#print(df)

D=2 #Number of features (Added one more feature to consider the threshold value)
C=1 #Number of classes. 


tot_data=117
train_data=100

X_train=np.zeros((train_data,D)) # Time Taken (sec.) (Input)
Y_train=np.zeros((train_data,C)) # Glucose Level (Output)


#Getting the input data into dataframe
for index,row in df.iterrows():
    if index<train_data:
        #print(f"{index},{row['Time (sec.)']},{row['Glucose Level']}")
        X_train[index,0]=1
        X_train[index,1]=row['Time (sec.)']
        Y_train[index,0]=row['Glucose Level']
    else:
        break


Y_train_min = np.min(Y_train)
Y_train_max = np.max(Y_train)
# print(Y_train)
# print(Y_train_min)
# print(Y_train_max)

# Normalising the dataset
for i,y in enumerate(Y_train):
    Y_train[i] = (y-Y_train_min) / (Y_train_max - Y_train_min)

# Training the ANN model
W=np.random.rand(C,D)
U=np.zeros((train_data,C))
V=np.zeros((train_data,C)) 

def phi(x):
    return 1/(1+math.e**(-x))

tot_iterations=0
eta,E_tot,eps=0.01,10,0.01
while(E_tot>eps):
    tot_iterations+=1
    E_tot=0
    for n in range(train_data): #Number of training objects 
        E_curr=0
        for j in range(C):
            U[n,j]=0
            for i in range(D):
                U[n,j]+=X_train[n,i]*W[j,i];
            V[n,j]=phi(U[n,j])
            E_curr+=0.5*(Y_train[n,j]-V[n,j])**2
        # print(f"Data point : {n}")
        # print(f"Actual o/p : {Y_train[n,0]}")
        # print(f"Predicted o/p : {V[n,0]}\n")
        for j in range(C):
            for i in range(D):
                W[j,i]+=eta*(Y_train[n,j]-V[n,j])*(phi(U[n,j])*(1-phi(U[n,j])))*X_train[n,i] 
        E_tot+=E_curr           
    print("In-Sample Error : ",E_tot," Iterations : ",tot_iterations)

    if (tot_iterations>5000):
        break

print(f"Total Iterations:{tot_iterations}")
for i in range(C):
    print(f"{W[i,]}")


# Making predictions on testing data
# For testing 

test_data= tot_data-train_data
X_test=np.zeros((test_data,D))  
Y_test=np.zeros((test_data,C))

test_data= tot_data-train_data
X_test=np.zeros((test_data,D)) #list of all customers 
Y_test=np.zeros((test_data,C)) #list of outcomes of all customers
U_test=np.zeros((test_data,C)) 
V_test=np.zeros((test_data,C)) 

for index,row in df.iterrows():
    if train_data<=index<tot_data:
        i=index - train_data # Since i acts as row counter, it must start from 0

        X_test[i,0]=1
        X_test[i,1]=row['Time (sec.)']
        Y_test[i,0]=row['Glucose Level']

Err_test=0
for n in range(test_data): #Number of training objects 
    E_curr=0
    for j in range(C):
        for i in range(D):
            U_test[n,j]+=X_test[n,i]*W[j,i]
        # Denormalising the output of the regressor model
        V_test[n,j]= Y_train_min + (Y_train_max - Y_train_min) * phi(U_test[n,j])
        
        E_curr+=0.5*(Y_test[n,j]-V_test[n,j])**2
    Err_test+=E_curr
    print(f"\nActual o/p : {Y_test[n,0]}")
    print(f"Predicted o/p : {V_test[n,0]}")
print(f"Test Error : {E_curr/test_data}")