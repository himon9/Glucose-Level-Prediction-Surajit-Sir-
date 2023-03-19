import pandas as pd
import numpy as np
import math

# Taking the training data
df = pd.read_excel('./Training_Data.xlsx')
#print(df)

D=2 # Number of features (Added one more feature to consider the threshold value)
C=1 # Number of classes. 
H=2 # Number of nodes in the hidden layer

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



def normalise(x):
    return (x - Y_train_min) / (Y_train_max - Y_train_min)

def denormalise(x):
    return Y_train_min + (Y_train_max - Y_train_min) * x


# Normalising the dataset
for i,y in enumerate(Y_train):
    Y_train[i] = normalise(y)

def sigmoid(x):
    return 1/(1+math.e**(-x))

def derivative_sigmoid(x):
    return x * (1 - x)


# Training the ANN model
# Creating Multi-Layered Artificial Network, for non-linearly separable points

W1=np.random.rand(H,D)
W2=np.random.rand(C,H)

U1=np.zeros((train_data,H))
V1=np.zeros((train_data,H))
U2=np.zeros((train_data,C))
V2=np.zeros((train_data,C)) 

tot_iterations = 0
eta, E_tot, eps = 0.2, 10, 0.2



while(E_tot > eps):
    tot_iterations +=1
    E_tot = 0
    for n in range(train_data): #Number of training objects 

        # Forward feed
        E_curr=0
        for j in range(H):
            for i in range(D):
                U1[n,j] += X_train[n,i] * W1[j,i];
            V1[n,j] = sigmoid(U1[n,j])
        
        for j in range(C):
            for i in range(H):
                U2[n,j] += V1[n,i] * W2[j,i]
            V2[n,j] = sigmoid(U2[n,j])

            E_curr += 0.5 * (Y_train[n,j] - V2[n,j]) ** 2
        E_tot += E_curr

        print(f"\nPoint : {n}, Y_train : { denormalise(Y_train[n,0])},  V2_train : {denormalise(V2[n,0])},   Iter : {tot_iterations}")
        print(f"Itr : {tot_iterations}, Pt : {n}, Y_train : { (Y_train[n,0])},  V2 : {(V2[n,0])}, Err : {E_curr}")
        # print("Weight 1 :", W1)
        # print("Weight 2 :", W2)
        

        # Back-Propagting the error
        for j in range(C):

            for i in range(H):
                del2 = (Y_train[n,j] - V2[n,j]) * derivative_sigmoid(V2[n,j])
                W2[j,i] += eta *  del2 * V1[n,i]

                for k in range(D):
                    del1 = del2 * W2[j,i] * derivative_sigmoid(V1[n,j])
                    W1[i,k] += eta * del1 * X_train[n,k]

    print("In-sample error : ", E_tot / train_data)
    print("\n")
    if tot_iterations > 500:
        break


# For testing 
test_data = tot_data - train_data

X_test=np.zeros((test_data,D)) #list of all customers 
Y_test=np.zeros((test_data,C)) #list of outcomes of all customers

U1_test=np.zeros((test_data,H))
V1_test=np.zeros((test_data,H))
U2_test=np.zeros((test_data,C))
V2_test=np.zeros((test_data,C)) 

for index,row in df.iterrows():
    if train_data <= index < tot_data:
        i = index - train_data # Since i acts as row counter, it must start from 0

        X_test[i,0] = 1
        X_test[i,1] = row['Time (sec.)']
        Y_test[i,0] = row['Glucose Level']


# Normalising the dataset
for i,y in enumerate(Y_test):
    Y_test[i] = normalise(y)

# print("X_test :")
# print(X_test)
# print("Y_test :")
# print(Y_test)



E_tot = 0
for n in range(test_data): #Number of training objects 
    E_curr = 0
    for j in range(H):
        for i in range(D):
            U1_test[n,j] += X_test[n,i] * W1[j,i]
        V1_test[n,j] = sigmoid(U1_test[n,j])
    
    for j in range(C):
        for i in range(H):
            U2_test[n,j] += V1_test[n,i] * W2[j,i]
        V2_test[n,j] =  sigmoid(U2_test[n,j])

        E_curr += 0.5 * (Y_test[n,j] - V2_test[n,j]) ** 2
    E_tot += E_curr

    # print(f"\nActual o/p : {Y_test[n,0]}")
    # print(f"Predicted o/p : {V2_test[n,0]}")
    # print(f"Curr Test Error : {E_curr}")
    print(f"\nPt : {n}, Y_train : { (Y_test[n,0])},  V2_test : {(V2_test[n,0])}, Err : {E_curr}")
    print(f"Pt : {n}, Y_train : { denormalise(Y_test[n,0])},  V2_test : {denormalise(V2_test[n,0])}, Err : {0.5 * (denormalise(Y_test[n,0]) - denormalise(V2_test[n,0])) ** 2}")

print(f"Out-Sample Test Error : {E_tot / test_data}")