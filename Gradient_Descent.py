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


def ReportAccuracy_2(w,testingFaces,testingLabels):
    yhat = sigmoid(testingFaces,w)
    y_hat_fin = np.argmax(yhat,axis=1)
    testingLabels_fin = np.argmax(testingLabels,axis=1)
    acc = np.sum(y_hat_fin== testingLabels_fin)/(testingLabels.shape[0])
    print("Accuracy : {}".format(acc))
    return acc
        
def reportCosts (w, trainingFaces, trainingLabels, alpha = 0.):
    print ("cost:",cross_entropy_loss(trainingFaces, trainingLabels,w, alpha))

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

    # Implementing Gradient Descent 
    alpha = 0
    weights = gradient_Descent(mnist_train_images,mnist_train_labels,alpha) 
    
    print("\n")
    # Final Losses
    print("Final Unregularized cross_entropy_loss on : Training set ,validation set ,Testing set")
    print("=====================================================================================")
    # Accuracy on training set  
    print("Training set ",end=""),reportCosts(weights,mnist_train_images,mnist_train_labels)
    #Accuracy on validation set
    print("Validation set ",end=""),reportCosts(weights,mnist_validation_images,mnist_validation_labels)
    #Accuracy on testing set
    print("Testing set ",end=""),reportCosts(weights,mnist_test_images,mnist_test_label
    
    print("\n")
    # Final Accuracies
    print("Final Accuracies on : Training set ,validation set ,Testing set")
    print("================================================================")
    # Accuracy on training set  
    print("Training set ",end=""),ReportAccuracy_2(weights,mnist_train_images,mnist_train_labels)
    #Accuracy on validation set
    print("Validationset",end=""),ReportAccuracy_2(weights,mnist_validation_images,mnist_validation_labels)
    #Accuracy on testing set
    print("Testing set ",end=""),ReportAccuracy_2(weights,mnist_test_images,mnist_test_labels)
    
    



