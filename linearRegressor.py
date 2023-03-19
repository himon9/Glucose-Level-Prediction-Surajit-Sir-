import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_excel('./Training_Data.xlsx')
#print(df)

D=2 #Number of features (Added one more feature to consider the threshold value)
C=1 #Number of classes. 

tot_data=117
train_data=50

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


# Checking linear separability using Scatter Plot

for x,y in zip(X_train,Y_train):
    if y > 190:
        plt.scatter(x[1], y, color='red')
    elif 160 < y <=190:
        plt.scatter(x[1], y, color='blue')
    else:
        plt.scatter(x[1], y, color='green')

plt.title('Scatter Plot for Checking Linear Separability')
plt.xlabel('Time (sec.)')
plt.ylabel('Glucose Level (mg/dL)')
plt.savefig('./GlucoseVsTime.png')
plt.close()
W=np.dot(np.dot(np.linalg.inv(np.dot( np.transpose(X_train),X_train)),np.transpose(X_train)),Y_train)

print(W)
# Y_predict=np.dot(np.transpose(W),X_train[1])
# print("Y_predict : ",Y_predict)
# print("Y_actual : ",Y_train[1])

# For testing 
test_data= tot_data-train_data
X_test=np.zeros((test_data,D))  
Y_test=np.zeros((test_data,C))


for index,row in df.iterrows():
    if train_data<=index<tot_data:
        i=index - train_data # Since i acts as row counter, it must start from 0

        X_test[i,0]=1
        X_test[i,1]=row['Time (sec.)']
        Y_test[i,0]=row['Glucose Level']

# print("X_test :")
# print(X_test)
# print("Y_test :")
# print(Y_test)

Err_test=0
Y_predict=[]
for n in range(test_data): #Number of training objects 
    Y_predict.append(np.dot(np.transpose(W),X_test[n]))
    Err_test+= (Y_predict[n] - Y_test[n])**2
    print("\nY_predict : ",Y_predict[n])
    print("Y_actual : ",Y_test[n])

mean_sqError = Err_test/test_data
print("Mean Squared Error : ",mean_sqError)

plt.plot(Y_test, color='blue', label='Actual Glucose Value')
plt.plot(Y_predict, color='green', label='Predicted Glucose Value')
plt.title('Linear Regressor Eq : W= inv(trans(X)*X) * trans(X) * Y')
plt.xlabel('Nubmer of values')
plt.ylabel('Glucose Level (mg/dL)')
plt.legend()
plt.savefig('./PredictedVsActual.png')

