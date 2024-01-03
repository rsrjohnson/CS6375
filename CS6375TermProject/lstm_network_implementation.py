import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#sklearn packages
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

rdseed=8466 #seed to replicate results

#LSTM_Network class
class LSTM_Network:
  def __init__(self, X, Y, activation_func='softmax', output_size=2, layers_sizes=[], solver='vanilla' ,learning_rate=0.1, epochs=30,verbose=False):
   
    self.X = X  #storing sequences
    self.Y = Y  #storing response variable

    self.k, self.m, self.n = self.X.shape  # number of sequences, time stamps, attributes  
    self.output_size=output_size # number of classes

    #Selecting the activation function default softmax
    if (activation_func == 'sigmoid'):
      self.f_activation = self.sigmoid      
    else:
      self.f_activation = self.softmax
    
    #Storing solver for gradient descent, learning rate and epochs
    self.solver=solver
    self.learning_rate=learning_rate
    self.epochs=epochs      

    self.sizes = layers_sizes
    self.sizes.append(output_size)
    self.sizes.insert(0, X.shape[2])
    self.layers=[]

    #Base layer and subsequent stacked layers
    for i in range(len(self.sizes)-1):
        self.layers.append(LSTM_layer(self.sizes[i],self.sizes[i+1]))
    
    #Output layer Weights and biases
    np.random.seed(rdseed+3)
    self.W_output=np.random.normal(0, 1,size=(output_size,output_size))

    np.random.seed(rdseed+4)
    self.b_output=np.random.normal(0, 1,size=(output_size))

    #Momentum and speed for Adagrad and AMSGrad solvers to be applied on the output layer
    self.vt_W_output=np.zeros_like(self.W_output)
    self.mt_W_output=np.zeros_like(self.W_output)

    self.vt_b_output=np.zeros_like(self.b_output)
    self.mt_b_output=np.zeros_like(self.b_output)

    #Boolean to display loss and accuracy during training
    self.verbose=verbose

  #Activation functions
  #Sigmoid activation function
  def sigmoid(self,x):
    cond = [x > 0, x <= 0] #indices of positives and negative observations
    #applying equivalent transformations of sigmoid to avoid overflow
    sigmoids = [lambda x: 1 / (1 + np.exp(-x)), lambda x: np.exp(x) / (1 + np.exp(x))]

    return np.piecewise(x, cond, sigmoids)

  #Softmax activation function
  def softmax(self, x):      
    c=-max(x)
    xe = np.exp(x+c) #adding a constant to stabilize the softmax function avoiding overflow
    return xe / np.sum(xe) 

  #Adagrad solver for gradient descent
  def adagrad_solver(self,dw,db):
    eps = 1e-7

    #updating vt(s)
    self.vt_W_output+=dw*dw    
    self.vt_b_output+=db*db
    
    #final values to be multiply by the learning rate
    dw=dw/np.sqrt(self.vt_W_output+eps)    
    db=db/np.sqrt(self.vt_b_output+eps)

    return dw,db

  #Amsgrad solver for gradient descent
  def amsgrad_solver(self,dw,db):
    #setting up initial parameters
    eps = 1e-7
    # values used in the Neural Network experiment conducted by the authors(https://openreview.net/pdf?id=ryQu7f-RZ): b1 = 0.9 and b2 chosen from {0.99, 0.999}
    beta1 = 0.9
    beta2 = 0.999 

    #updating the momentums
    self.mt_W_output=beta1*self.mt_W_output+(1 - beta1)*dw  
    self.mt_b_output=beta1*self.mt_b_output+(1 - beta1)*db

    #finding the new v(s)
    vtempW=beta2*self.vt_W_output+(1 - beta2)*dw*dw    
    vtempb=beta2*self.vt_b_output+(1 - beta2)*db*db
    
    #ensuring that the current v values are always larger than the values from the previous iteration, np.fmax finds element wise maximum
    self.vt_W_output=np.fmax(vtempW,self.vt_W_output)   
    self.vt_b_output=np.fmax(vtempb,self.vt_b_output)

    #final values to be multiply by the learning rate
    dw=self.mt_W_output/(np.sqrt(self.vt_W_output)+eps)
    db=self.mt_b_output/(np.sqrt(self.vt_b_output)+eps)

    return dw,db


  #Forward propagation of whole network for one sequence Xj
  def network_forward(self,Xj):
    
    #lists to store the gates, cell and hidden states by layer
    gates_layers=[]
    states_layers=[]
    ht_layers=[]
    
    bot_layer=self.layers[0]
    #Bottom layer forward pass to obtain inputs of upper layer
    gates, states, ht_list=bot_layer.forward(Xj)

    #Storing gates and states values for future backward process
    gates_layers.append(gates)
    states_layers.append(states)
    ht_layers.append(ht_list)

    #Forward pass of subsequent upper layers
    for i in range(1,len(self.layers)):
        gates, states, ht_list = self.layers[i].forward(ht_list[1:])
        gates_layers.append(gates)
        states_layers.append(states)
        ht_layers.append(ht_list)

    #Output Layer probability outputs
    Yt=[]
    for t in range(1,len(ht_list)):
        vt = self.W_output.dot(ht_list[t]) + self.b_output
        Yt.append(self.f_activation(vt))

    return gates_layers, states_layers, ht_layers, Yt

  #Backpropagation process for whole network over all sequences, It receives validation data
  def network_backprop(self,X_val,Y_val):

    #If verbose is True we store the true labels of training and testing to display learning process
    if self.verbose:
      y_true_trn=self.Y.copy()
      y_true_trn=y_true_trn.reshape(-1,y_true_trn.shape[2])
      y_true_trn=np.argmax(y_true_trn,axis=1)

      y_true_tst=Y_val.copy()
      y_true_tst=y_true_tst.reshape(-1,y_true_tst.shape[2])
      y_true_tst=np.argmax(y_true_tst,axis=1)

    #lists to track historical errors
    hist_error = [] 
    hist_error_val=[]

    for e in range(self.epochs):
          
      #Looping over all sequences
      for Xj,Yj  in zip(self.X,self.Y):
        #Forward pass on the current sequence
        gates_layers, states_layers, ht_layers, Yt=self.network_forward(Xj)

        #Gradient of output layer
        dW_output = np.zeros_like(self.W_output)
        db_output = np.zeros_like(self.b_output)

        #Backward pass through output layer
        ht_list=ht_layers[-1]
        dx=[]
        for t in range(len(Yj)-1,-1,-1):

            d_vt = Yt[t] - Yj[t]

            dW_output += np.outer(d_vt, ht_list[t+1])
            db_output += d_vt

            dx.insert(0,self.W_output.T.dot(d_vt))

        #Updating output layer weights and biases depending on the solver, default solver vanilla
        if self.solver=='adagrad':
              dW_output,db_output=self.adagrad_solver(dW_output,db_output)
        elif self.solver=='amsgrad':
              dW_output,db_output=self.adagrad_solver(dW_output,db_output)

        self.W_output-=self.learning_rate*dW_output
        self.b_output-=self.learning_rate*db_output

        #Backpropagating through every layer below
        for layer in range(len(self.layers)-1,0,-1):

            dW, dU, db, dx=self.layers[layer].backward(ht_layers[layer-1], dx, gates_layers[layer], states_layers[layer], ht_layers[layer])
            #Updating layer weights
            self.layers[layer].weights_update(dW, dU, db,self.learning_rate,self.solver)

        dW, dU, db, dx = self.layers[0].backward(Xj, dx, gates_layers[0], states_layers[0], ht_layers[0])
        #Updating bottom layer weights
        self.layers[0].weights_update(dW, dU, db, self.learning_rate,self.solver)

      #Tracking training and testing errors      
      hist_error.append(self.total_error(self.X,self.Y))
      hist_error_val.append(self.total_error(X_val,Y_val))

      #If verbose is True we calculate the current accuracy of training and testing
      if self.verbose:
        #Finding accuracy of training set
        yt_probs=self.fit(self.X)
        y_pred=self.classify(yt_probs) 
        trn_accuracy=accuracy_score(y_true_trn,y_pred)

        #Finding accuracy of testing set
        yt_probs=self.fit(X_val)
        y_pred=self.classify(yt_probs) 
        tst_accuracy=accuracy_score(y_true_tst,y_pred)

        print('Epoch ' + str(e+1) +  ', Training Loss: ' +str(round(hist_error[-1],6))+ ', Test Loss: ' + str(round(hist_error_val[-1],6)) +
         ', Training Accuracy ' + str(round(trn_accuracy,6))+', Testing Accuracy ' + str(round(tst_accuracy,6)))
            

    return hist_error,hist_error_val
  
  #Function that returns the probabilities for each class
  def fit(self,X):  

    y_probs = []  
    
    #Forward process through layers
    for Xj in X:
      bot_cell = self.layers[0]
      gates, states, ht_list = bot_cell.forward(Xj)
      for i in range(1, len(self.layers)):
          gates, states, ht_list = self.layers[i].forward(ht_list[1:])

      # Output Layer
      for t in range(1,len(ht_list)):
          vt = self.W_output.dot(ht_list[t]) + self.b_output
          y_probs.append(self.f_activation(vt))

    return np.array(y_probs)  
  
  #Average log loss
  def total_error(self, X,Y):
        y_true=Y.copy()
        y_true=y_true.reshape(-1,y_true.shape[2])
        y_true=np.argmax(y_true,axis=1)

        yt_probs=self.fit(X)           

        return  log_loss(y_true,yt_probs)

  #Classifying given the probabilities of a class
  def classify(self,y_probs):
   
    labels=np.argmax(y_probs,axis=1)
            
    return labels

#LSTM_layer class
class LSTM_layer:
  def __init__(self,input_size,output_size):   
          
    # Generating random weights and biases
    np.random.seed(rdseed)
    self.W = np.random.normal(0, 1, size=(4, output_size, input_size)) #input weights

    np.random.seed(rdseed + 1)
    self.U = np.random.normal(0, 1, size=(4, output_size, output_size)) #hidden state weights

    np.random.seed(rdseed + 2)
    self.b = np.random.normal(0, 1, size=(4, output_size)) #biases

    self.output_size=output_size

    #Momentum and speed for Adagrad and AMSGrad solvers
    self.vt_W=np.zeros_like(self.W)
    self.mt_W=np.zeros_like(self.W)

    self.vt_U=np.zeros_like(self.U)
    self.mt_U=np.zeros_like(self.U)

    self.vt_b=np.zeros_like(self.b)
    self.mt_b=np.zeros_like(self.b)
  
  #Adagrad solver for gradient descent
  def adagrad_solver(self,dw,du,db):
    eps = 1e-7

    self.vt_W+=dw*dw
    self.vt_U+=du*du
    self.vt_b+=db*db
    
    #final values to be multiply by the learning rate
    dw=dw/np.sqrt(self.vt_W+eps)
    du=du/np.sqrt(self.vt_U+eps)
    db=db/np.sqrt(self.vt_b+eps)

    return dw,du,db


  def amsgrad_solver(self,dw,du,db):
    #setting up initial parameters
    eps = 1e-7
    # values used in the Neural Network experiment conducted by the authors(https://openreview.net/pdf?id=ryQu7f-RZ): b1 = 0.9 and b2 chosen from {0.99, 0.999}
    beta1 = 0.9
    beta2 = 0.999 

    #updating the momentums
    self.mt_W=beta1*self.mt_W+(1 - beta1)*dw
    self.mt_U=beta1*self.mt_U+(1 - beta1)*du
    self.mt_b=beta1*self.mt_b+(1 - beta1)*db

    #finding the new v(s)
    vtempW=beta2*self.vt_W+(1 - beta2)*dw*dw
    vtempU=beta2*self.vt_U+(1 - beta2)*du*du
    vtempb=beta2*self.vt_b+(1 - beta2)*db*db
    
    #ensuring that the current v values are always larger than the values from the previous iteration, np.fmax finds element wise maximum
    self.vt_W=np.fmax(vtempW,self.vt_W)
    self.vt_U=np.fmax(vtempU,self.vt_U)
    self.vt_b=np.fmax(vtempb,self.vt_b)

    #final values to be multiply by the learning rate
    dw=self.mt_W/(np.sqrt(self.vt_W)+eps)
    du=self.mt_U/(np.sqrt(self.vt_U)+eps)
    db=self.mt_b/(np.sqrt(self.vt_b)+eps)

    return dw,du,db        

  #Sigmoid activation function
  def sigmoid(self, x):
    cond = [x > 0, x <= 0]  # indices of positives and negative observations
    # applying equivalent transformations of sigmoid to avoid overflow
    sigmoids = [lambda x: 1 / (1 + np.exp(-x)), lambda x: np.exp(x) / (1 + np.exp(x))]

    return np.piecewise(x, cond, sigmoids)

  #Derivative of sigmoid in terms of outputs
  def diff_sigmoid(self, x):
    return x * (1 - x)

  #Tanh written on terms of sigmoid
  def tanh(self, x):
    return 2 * self.sigmoid(2 * x) - 1

  #Derivative of tanh in terms of outputs
  def diff_tanh(self, x):
    return 1 - x * x

  def forward(self, Xj):
    ht = np.zeros(self.output_size)  # initial hidden state
    ct = np.zeros(self.output_size)  # initial cell state

    ht_list = [ht]  # List to store outputs from the previous hidden state
    states = [ct]  # List to store cell states for backward pass
    gates = []  # List to store gates values for backward pass

    for t in range(Xj.shape[0]):
        zi = self.W.dot(Xj[t].T) + self.U.dot(ht) + self.b
        # hidden gate, input gate, forget gate and output gate
        a, ifo = self.tanh(zi[0]), self.sigmoid(zi[1:])

        d = {}
        d['a'], d['i'], d['f'], d['o']=a, ifo[0], ifo[1], ifo[2]
        gates.append(d)  # storing gates' values for future backward process

        #cell state
        ct = a * ifo[0] + ifo[1] * ct
        states.append(ct)

        #cell output
        ht = self.tanh(ct) * ifo[2]
        ht_list.append(ht)

    return np.array(gates), np.array(states), np.array(ht_list)

  def backward(self, Xj, dx, gates, states, ht_list):
    
    d_ct = np.zeros(self.output_size)
    d_htnext = np.zeros(self.output_size)
    ft_next = np.zeros(self.output_size)

    dW = np.zeros_like(self.W)
    dU = np.zeros_like(self.U)
    db = np.zeros_like(self.b)
    d_xt=[]

    for t in range(len(gates) - 1, -1, -1):
      d = gates[t] #gates from forward process
      at, it, ft, ot = d['a'], d['i'], d['f'], d['o']

      d_ht = dx[t] #incoming gradient from upper layer
      d_ht = d_ht + d_htnext #incoming gradient from next time step

      #gradient of the state
      d_ct = d_ht * ot * (1 - self.tanh(states[t + 1]) ** 2) + d_ct * ft_next

      #gradient of the grades
      d_at = d_ct * it * self.diff_tanh(at)
      d_it = d_ct * at * self.diff_sigmoid(it)
      d_ft = d_ct * states[t] * self.diff_sigmoid(ft)
      d_ot = d_ht * self.tanh(states[t + 1]) * self.diff_sigmoid(ot)

      d_gates = np.array([d_at, d_it, d_ft, d_ot])

      #storing gradients for layer below
      d_xt.insert(0,sum([self.W[i].T.dot(d_gates[i]) for i in range(4)]))

      #gradient going to the previous time step cell
      d_htnext = sum([self.U[i].T.dot(d_gates[i]) for i in range(4)])
      ft_next = ft

      # Accumulation of weights' gradients
      dW += [np.outer(d_gates[i], Xj[t]) for i in range(4)]
      dU += [np.outer(d_gates[i], ht_list[t]) for i in range(4)]
      db += d_gates

    return dW, dU, db, d_xt

  #Backprogagation process
  def backpropagation(self,Xj,Yj):

    gates, states, Yi = self.forward(Xj)

    dW, dU, db, dx= self.backward(Xj,Yj,gates, states, Yi)      

    return dW, dU, db, dx

  def weights_update(self,dW, dU, db,alpha,solver):
        
      if solver=='adagrad':
        dW, dU, db = self.adagrad_solver(dW, dU, db)
      elif solver=='amsgrad':
        dW, dU, db =self.amsgrad_solver(dW, dU, db)

      self.W-=alpha*dW
      self.U-=alpha*dU
      self.b-=alpha*db


#Data preprocessing function
def preprocess(data,sequence_length=23,test_perc=0.3):
  
  if sequence_length%23!=0:
    sequence_length=23

  Y = data.iloc[:,-1] #DataFrame of outputs
  X = data.iloc[:,:-1] #DataFrame of attributes
  
  Y[Y>1]=0 #converting outputs to binary data
  
  Y=pd.get_dummies(Y) #one hot encoding classes
 
  X=np.array(X)  
  Y=np.array(Y)

  #Splitting into training and testing set
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_perc, random_state=rdseed)
  
  #Applying feature standaratization
  scaler = StandardScaler() #scaler to standardize the data
  scaler.fit(X_train)
  X_train = scaler.transform(X_train)
  X_test = scaler.transform(X_test)  # applying same transformation to test data

  #Reshaping back training and testing data set into number of sequences, length of sequence, number of attributes  
  X_train=X_train.reshape(-1,sequence_length,X_train.shape[1])
  X_test=X_test.reshape(-1,sequence_length,X_test.shape[1])

  Y_train=Y_train.reshape(-1,sequence_length,Y_train.shape[1])
  Y_test=Y_test.reshape(-1,sequence_length,Y_test.shape[1])

  return X_train, X_test, Y_train, Y_test


url='https://raw.githubusercontent.com/rsrjohnson/CS6375/master/data.csv'
raw_data = pd.read_csv(url,sep=",",encoding= 'unicode_escape')

raw_data=raw_data.iloc[:,1:]

def run_experiments():
  solvers=['vanilla','adagrad','amsgrad'] 
  alphas=[0.1,0.4,0.02]

  fig, axs = plt.subplots(1, 3, figsize=(10,10))
  fig.suptitle('Historical Errors Training and Testing Sets')
    
  X_train, X_test, Y_train, Y_test=preprocess(raw_data,sequence_length=23,test_perc=.3)
  
  #Experiments 1-3  
  for e in range(3):    
  
    print('Experiment '+str(e+1)+': Solver: ' +solvers[e]+ ', epochs=30, train/test 70/30, learning rate = '+ str(alphas[e]), 'sequence length = 23\n')
    
    LSTMe= LSTM_Network(X_train, Y_train, 'softmax', output_size=2,layers_sizes=[], solver=solvers[e], learning_rate=alphas[e], epochs=30)

    hist_error,hist_error_validation=LSTMe.network_backprop(X_val=X_test,Y_val=Y_test)

    x = range(1, LSTMe.epochs+1) 
    axs[e].plot(x, hist_error, x, hist_error_validation)      
    axs[e].set_title("Method: "+ LSTMe.solver+', learning rate = ' + str(alphas[e]))      
    axs[e].grid(True)
    axs[e].legend(('train', 'test'))

    print('Training Dataset')
    yt_probs=LSTMe.fit(X_train)
    y_true=Y_train.copy()

    y_true=y_true.reshape(-1,y_true.shape[2])
    y_true=np.argmax(y_true,axis=1)  

    print('Cross Entropy Loss',round(log_loss(y_true,yt_probs),6))

    y_pred=LSTMe.classify(yt_probs) 
    print('Accuracy',round(accuracy_score(y_true,y_pred),6))
    print('Precision, Recall, F Score',*['%.6f' %x for x in precision_recall_fscore_support(y_true,y_pred,average='binary')[:-1]],'\n') 

    print('Test Dataset')
    yt_probs=LSTMe.fit(X_test)
    y_true=Y_test.copy()

    y_true=y_true.reshape(-1,y_true.shape[2])
    y_true=np.argmax(y_true,axis=1)

    print('Cross Entropy Loss',round(log_loss(y_true,yt_probs),6))

    y_pred=LSTMe.classify(yt_probs)
    print('Accuracy',round(accuracy_score(y_true,y_pred),6))
    print('Precision, Recall, F Score',*['%.6f' %x for x in precision_recall_fscore_support(y_true,y_pred,average='binary')[:-1]],'\n') 

    
    print()
    print('########################################################################################\n')

  for ax in axs.flat:
        ax.set(xlabel='Epoch', ylabel='log_loss')
        
  plt.show()


  alphas=[0.03,0.3,0.007]
  fig, axs = plt.subplots(1, 3, figsize=(10,10))
  fig.suptitle('Historical Errors Training and Testing Sets, extra layers [8,4]')  

  #Experiments 4-6
  for e in range(3):    
  
    print('Experiment '+str(e+4)+': Solver: ' +solvers[e]+ ', epochs=30, train/test 70/30, learning rate = '+ str(alphas[e]), 'sequence length = 23', 'Extra Layer sizes [8,4]\n')
    
    LSTMe= LSTM_Network(X_train, Y_train, 'softmax', output_size=2,layers_sizes=[8,4], solver=solvers[e], learning_rate=alphas[e], epochs=30)

    hist_error,hist_error_validation=LSTMe.network_backprop(X_val=X_test,Y_val=Y_test)

    x = range(1, LSTMe.epochs+1) 
    axs[e].plot(x, hist_error, x, hist_error_validation)      
    axs[e].set_title("Method: "+ LSTMe.solver+', learning rate = ' + str(alphas[e]))      
    axs[e].grid(True)
    axs[e].legend(('train', 'test'))

    print('Training Dataset')
    yt_probs=LSTMe.fit(X_train)
    y_true=Y_train.copy()

    y_true=y_true.reshape(-1,y_true.shape[2])
    y_true=np.argmax(y_true,axis=1)  

    print('Cross Entropy Loss',round(log_loss(y_true,yt_probs),6))

    y_pred=LSTMe.classify(yt_probs) 
    print('Accuracy',round(accuracy_score(y_true,y_pred),6))
    print('Precision, Recall, F Score',*['%.6f' %x for x in precision_recall_fscore_support(y_true,y_pred,average='binary')[:-1]],'\n') 

    print('Test Dataset')
    yt_probs=LSTMe.fit(X_test)
    y_true=Y_test.copy()

    y_true=y_true.reshape(-1,y_true.shape[2])
    y_true=np.argmax(y_true,axis=1)

    print('Cross Entropy Loss',round(log_loss(y_true,yt_probs),6))

    y_pred=LSTMe.classify(yt_probs)
    print('Accuracy',round(accuracy_score(y_true,y_pred),6))  
    print('Precision, Recall, F Score',*['%.6f' %x for x in precision_recall_fscore_support(y_true,y_pred,average='binary')[:-1]],'\n') 
    
    print()
    print('########################################################################################\n')

  for ax in axs.flat:
        ax.set(xlabel='Epoch', ylabel='log_loss')
        
  plt.show()

#Function to get user inputs
def getParameters():
    
  print("Type 'Y' if you want to enter custom values for the parameters to execute the LSTM network, in any other case the network will be executed with the default parameters.")
  print("The default parameters are: activation_func = 'softmax',  layers_sizes = [], solver = 'vanilla', learning_rate = 0.1, epochs = 30, verbose = False")
  if input() != 'Y':
    return 'softmax', [], 'vanilla', 0.1, 30, False

  print("Enter the parameters to execute the LSTM network, in case of an invalid parameter the net will be executed with the default parameter for that value.\n")
  print("Suggested parameters vanilla learner: \n activation_func = 'softmax', layers_sizes = [], learning_rate = 0.1, epochs = 30\n")
  print("Suggested parameters adagrad learner: \n activation_func = 'softmax', layers_sizes = [], learning_rate = 0.4, epochs = 30\n")
  print("Suggested parameters amsgrad learner: \n activation_func = 'softmax', layers_sizes = [], learning_rate = 0.02, epochs = 30\n")
  
  print("Enter the solver to be executed (adagrad or amsgrad), default: vanilla")
  solver = input()
  
  print("Enter the value for the output layer activation function (sigmoid or softmax), default: softmax.\n")
  activation_func = input()  

  print("The LSTM will be created by default with at least one base hidden layer. If you would like additional hidden layers, enter a sequence of integers separated by space representing the hidden layers with their number of neurons.")
  print("For example: 8 4 creates an LSTM with 2 additional hidden layers, the first one with 8 neurons and the second one with 4")
  try:
    layers_sizes = list(map(int, input().split()))
    if len(layers_sizes)==0:
      raise ValueError()
  except ValueError:    
      layers_sizes=[]
 

  print("Enter the learning rate.")
  try:
    learning_rate = float(input())
  except ValueError:
    if solver=='adagrad':
      learning_rate = 0.4
    elif solver=='amsgrad':
      learning_rate = 0.02
    else:
      learning_rate = 0.1
          

  print("Enter the number of epochs.")
  try:
    epochs = int(input())
  except ValueError:
    epochs = 30

  verbose=False
  print("Press 'Y' if you would like to display the training and testing loss per epoch.")
  if input()=='Y':
        verbose=True

  return activation_func, layers_sizes, solver, learning_rate, epochs, verbose

#Uncomment if you would like to run the experiments
#run_experiments()

#Getting LSTM parameters
activation, hidden, solver, learning_rate, epoch, verbose=getParameters()


print('Enter the testing set percent as a decimal, ex: .3 means 30%.')

try:
  test_perc = float(input())
  if test_perc>.9:
    raise ValueError
except ValueError:
  test_perc=0.3

print('Enter the length of the sequence the time series will be divided into (default value 23). If the value is not divisible by 23 then 23 will be set.')
try:
  seq_len=int(input())
except ValueError:
  seq_len=23

X_train, X_test, Y_train, Y_test=preprocess(raw_data,sequence_length=seq_len,test_perc=test_perc)

LSTMe= LSTM_Network(X_train, Y_train, activation, output_size=2,layers_sizes=hidden, solver=solver, learning_rate=learning_rate, epochs=epoch,verbose=verbose)

print("Training the LSTM Network:\n")
LSTMe.network_backprop(X_test,Y_test)

print()

print('Training Dataset Results')
yt_probs=LSTMe.fit(X_train) #probabilities of every class
y_true=Y_train.copy()

y_true=y_true.reshape(-1,y_true.shape[2])
y_true=np.argmax(y_true,axis=1)  #true training labels

print('Cross Entropy Loss',round(log_loss(y_true,yt_probs),6))

y_pred=LSTMe.classify(yt_probs) #classifying the probabilities into classes
print('Accuracy',round(accuracy_score(y_true,y_pred),6))
print('Precision, Recall, F Score',*['%.6f' %x for x in precision_recall_fscore_support(y_true,y_pred,average='binary')[:-1]],'\n') 

print('Test Dataset Results')
yt_probs=LSTMe.fit(X_test) #probabilities of every class
y_true=Y_test.copy()

y_true=y_true.reshape(-1,y_true.shape[2])
y_true=np.argmax(y_true,axis=1) #true test labels

print('Cross Entropy Loss',round(log_loss(y_true,yt_probs),6))

y_pred=LSTMe.classify(yt_probs) #classifying the probabilities into classes
print('Accuracy',round(accuracy_score(y_true,y_pred),6))  
print('Precision, Recall, F Score',*['%.6f' %x for x in precision_recall_fscore_support(y_true,y_pred,average='binary')[:-1]],'\n') 

print('Press enter to exit')

input()