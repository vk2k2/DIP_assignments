from keras.datasets import mnist
import numpy as np


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train_rowvector = np.reshape(x_train, (-1, 28*28))
x_train_colvector = np.copy(x_train_rowvector).T
x_test_rowvector = np.reshape(x_test, (-1, 28*28))
x_test_colvector = np.copy(x_test_rowvector).T




x_list= [0] * 10
for i in range(10):
   x_list[i] = (x_train_colvector[:, y_train == i])
   
means_list = [0] * 10
for i in range(10):
    means_list[i] = np.mean(x_list[i], axis=1)
    

u_list = [0] * 10
s_list = [0] * 10
v_list = [0] * 10

for i in range(10):

    u_list[i], s_list[i], v_list[i] = np.linalg.svd(x_list[i], full_matrices=False)
    

acc_list = [0] * 5
pred_list = [0] * 5

for k in range(5):

    uk_ukt_list = [0] * 10

    for i in range(10):
        uk = np.zeros((784, 784))
        uk[:,0:k+1] = u_list[i][:, 0:k+1]
        uk_ukt_list[i] = uk @ uk.T

    y_pred = np.zeros(len(y_test))

    for i in range(len(y_pred)):
  
        x = x_test_colvector[:, i]

        residuals = np.zeros(10)

        for j in range(10):

            residuals[j] = np.linalg.norm((np.identity(28*28) - uk_ukt_list[j]) @ x, ord=2)

        y_pred[i] = np.argmin(residuals)

    pred_list[k] = y_pred

    correct = np.where(y_pred == y_test, 1, 0)
    accuracy = np.sum(correct) / len(correct)
    print("Accuracy with", k + 1, "singular images: ", accuracy)
    acc_list[k] = accuracy
    