import numpy as np
import matplotlib.pyplot as plt

def sigmoid(X,W):
    z = np.dot(X,W)
    exp_z = np.exp(z)
    sum_exp_z = np.sum(exp_z,axis=1)
    sum_exp_z= sum_exp_z.reshape((X.shape[0],1))
    sigmoid = exp_z / sum_exp_z
     
    return sigmoid

def cross_entropy_loss(X,Y,W,alpha):
    n = X.shape[0]
    Y_cap = sigmoid(X,W)
    loss = (-1/n)*np.sum(np.sum(np.multiply(Y,np.log(Y_cap)),axis=1),axis=0)
    regularization = (alpha/2)*np.sum(np.multiply(W,W))
    cost = loss + regularization
    return cost

def grad(X,Y,Y_cap,W,alpha):
    n = X.shape[0]
    grad = (1/n)*np.dot(X.T,Y_cap-Y) + alpha*W
    return grad

def reportCosts (w, trainingFaces, trainingLabels, testingFaces, testingLabels, alpha = 0.):
    print ("Training cost: {}".format(J(w, trainingFaces, trainingLabels, alpha)))
    print ("Testing cost:  {}".format(J(w, testingFaces, testingLabels, alpha)))

def random_data():

    return 

def gradient_Descent(mnist_train_images,mnist_train_labels,alpha):
    
    w = np.zeros((mnist_train_images.shape[1],mnist_train_labels.shape[1]))
    epi = 0.0001
    tolerance  = 0.0000001
    w_old = w
    
    while(True):
        v = grad(mnist_train_images,mnist_train_labels,sigmoid(mnist_train_images,w_old),w_old,alpha)
        w_new = w_old - epi*v
        c = cross_entropy_loss(mnist_train_images,mnist_train_labels,w_new,alpha)
        d = cross_entropy_loss(mnist_train_images,mnist_train_labels,w_old,alpha)
        print(c)
        print(d)
        if (np.absolute(c-d) < tolerance):
            break
        else:
            w_old = w_new
        
    return w_new




if __name__ == "__main__":
    # Load data
    if ('mnist_train_images' not in globals()):  
        mnist_train_images = np.load("mnist_train_images.npy")
        mnist_train_labels = np.load("mnist_train_labels.npy")
        mnist_validation_images = np.load("mnist_validation_images.npy")
        mnist_validation_labels = np.load("mnist_validation_labels.npy")
        mnist_test_images = np.load("mnist_test_images.npy")
        mnist_test_labels = np.load("mnist_test_labels.npy")

    #print(mnist_train_images.shape)
    #print(mnist_train_labels.shape) 
    #print(mnist_validation_images.shape)
    #print(mnist_validation_labels.shape)
    #print(mnist_test_images.shape)
    #print(mnist_test_labels.shape) 
    #print(mnist_train_images[1])
    #print(mnist_train_labels[1]) 
    
    # sigmoid test
    # w = np.zeros((784,10))   
    # print(sigmoid(mnist_train_images,w))
    
    # cross_entropy_loss test 
    #cross_entropy_loss(mnist_train_images,mnist_train_labels,w,10) 
    

    # gradient descent test
    alpha = 0
    gradient_Descent(mnist_train_images,mnist_train_labels,alpha) 




