import numpy as np

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
    loss = (-1/n)*np.sum(Y*np.log(Y_cap))
    regularization = (alpha/2)*np.sum(W*W)
    cost = loss + regularization
    return cost

def grad(X,Y,Y_cap,W,alpha):
    n = X.shape[0]
    grad = (1/n)*np.matmul(X.T,(Y_cap-Y)) + alpha*W
    return grad

def reportCosts (w, trainingFaces, trainingLabels, alpha = 0.):
    print ("cost:",cross_entropy_loss(trainingFaces, trainingLabels,w, alpha))
    return

def ReportCosts(w,trainingFaces,trainingLabels,testingFaces,testingLabels,alpha):

    print("loss:",cross_entropy_loss(trainingFaces,trainingLabels,w,alpha),end="")
    print(" ","acc:",reportAccuracy(w,trainingFaces,trainingLabels),end="")
    print(" ","validation loss:",cross_entropy_loss(testingFaces,testingLabels,w,alpha),end="")
    print(" ","val_acc:",reportAccuracy(w,testingFaces,testingLabels),end="")

    return 

def random_data(X,Y):
    rand_index = np.random.permutation(X.shape[0])
    return X[rand_index],Y[rand_index] 

def reportAccuracy(w,testingFaces,testingLabels):
    yhat = sigmoid(testingFaces,w)
    y_hat_fin = np.argmax(yhat,axis=1)
    testingLabels_fin = np.argmax(testingLabels,axis=1)
    acc = np.sum(y_hat_fin== testingLabels_fin)/(testingLabels.shape[0])
    # print("Accuracy : {}".format(acc))
    return acc

def ReportAccuracy_2(w,testingFaces,testingLabels):
    yhat = sigmoid(testingFaces,w)
    y_hat_fin = np.argmax(yhat,axis=1)
    testingLabels_fin = np.argmax(testingLabels,axis=1)
    acc = np.sum(y_hat_fin== testingLabels_fin)/(testingLabels.shape[0])
    print("Accuracy : {}".format(acc))
    return acc
    

def SGD(mnist_train_images,mnist_train_labels,mnist_validation_images,mnist_validation_labels,alpha):
    n,m = mnist_train_images.shape
    c = mnist_train_labels.shape[1]
    
    epi = 5e-1
    tolerance  = 1e-6

    w_old = np.random.normal(0,0.1,(m,c))

    #hyper parameters
    mini_batch = (2**8)
    num_of_epochs = 50
    total_num_of_rounds = int(np.ceil(n/mini_batch))
    iter_decay = 2
    decay = 0.1
        
    for epoch in range(num_of_epochs):
        

        #randomized the data 
        random_mnist_train_images,random_mnist_train_labels= random_data(mnist_train_images,mnist_train_labels)
        print("-")

        
        for round in range(total_num_of_rounds):
            

            j,k = round * mini_batch,(round+1)*mini_batch
            mini_batch_train_images,mini_batch_train_labels = random_mnist_train_images[j:k],random_mnist_train_labels[j:k]
            
            v = grad(mini_batch_train_images,mini_batch_train_labels,sigmoid(mini_batch_train_images,w_old),w_old,alpha)
            w_new = w_old - epi*v
            c = cross_entropy_loss(mini_batch_train_images,mini_batch_train_labels,w_new,alpha)
            d = cross_entropy_loss(mini_batch_train_images,mini_batch_train_labels,w_old,alpha)
            
            # print("prev cost:",d,"cost:",c)
            if (np.absolute(c-d) > tolerance):
                w_old = w_new
            else:
                break

        print('Epoch:',epoch+1,'/',num_of_epochs,"",end="")
        ReportCosts(w_new,random_mnist_train_images,random_mnist_train_labels,mnist_validation_images,mnist_validation_labels,alpha)      
        
        # Learing rate decay 
        if epoch+1 % iter_decay == 0:
            if epi > 0.0001:
                epi = epi*(1-decay)
                
                

    return [w_new,num_of_epochs,epi,mini_batch,decay,iter_decay]




if __name__ == "__main__":
    # Load data
    if ('mnist_train_images' not in globals()):  
        mnist_train_images = np.load("mnist_train_images.npy")
        mnist_train_labels = np.load("mnist_train_labels.npy")
        mnist_validation_images = np.load("mnist_validation_images.npy")
        mnist_validation_labels = np.load("mnist_validation_labels.npy")
        mnist_test_images = np.load("mnist_test_images.npy")
        mnist_test_labels = np.load("mnist_test_labels.npy")

    
    #training the weights 
    alpha = 0.0003
    weights,epochs,learning_rate,batch_size,decay_rate,decay_iter = SGD(mnist_train_images,mnist_train_labels,mnist_validation_images,mnist_validation_labels,alpha) 
    
    print("\n")
    #hyperparameters
    print("The Following hyperparameters were used:")
    print("=========================================")
    print("The number of epochs :",epochs)
    print("The learning rate :",learning_rate)
    print("The batch size :",batch_size)
    print("The regularization :",alpha)
    print("The decay rate : ", decay_rate)
    print("Iterations per decay :",decay_iter)

    print("\n")
    # Final Losses
    print("Final Unregularized cross_entropy_loss on : Training set ,validation set ,Testing set")
    print("=====================================================================================")
    # Accuracy on training set  
    print("Training set ",end=""),reportCosts(weights,mnist_train_images,mnist_train_labels)
    #Accuracy on validation set
    print("Validation set ",end=""),reportCosts(weights,mnist_validation_images,mnist_validation_labels)
    #Accuracy on testing set
    print("Testing set ",end=""),reportCosts(weights,mnist_test_images,mnist_test_labels)
     
    print("\n")
    # Final Accuracies
    print("Final Accuracies on : Training set ,validation set ,Testing set")
    print("================================================================")
    # Accuracy on training set  
    print("Training set ",end=""),ReportAccuracy_2(weights,mnist_train_images,mnist_train_labels)
    #Accuracy on validation set
    print("Validation set ",end=""),ReportAccuracy_2(weights,mnist_validation_images,mnist_validation_labels)
    #Accuracy on testing set
    print("Testing set ",end=""),ReportAccuracy_2(weights,mnist_test_images,mnist_test_labels)
       



