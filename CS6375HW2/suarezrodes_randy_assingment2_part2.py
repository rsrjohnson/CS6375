#Student Name: Randy Suarez Rodes
#Course: CS 6375.002
#Assignment 2
## IDE: Pycharm Community Version 2019.3, Interpreter Python 3.8, Operating System: 10

#Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

#sklearn packages
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

rdseed=8466 #seed to replicate results


class ANN:
  #Class constructor, storing data attributes, output, activation function, sizes of hidden layers, GD algorithm, learning rate and number of epochs
  def __init__(self, X, Y, activation_func, hidden_layers=[6, 4], alg='AMSGrad', learning_rate=0.29, epochs=400, tol=0.0001):
    self.X = X #storing data attributes
    self.Y = np.array(Y, ndmin=2) #storing response variable
    self.sample_size = X.shape[0]

    #Selecting the activation function
    if (activation_func == 'sigmoid'):
      self.f_activation = self.sigmoid
      self.diff_activation = self.diff_sigmoid
    elif activation_func == 'tanh':
      self.f_activation = self.tanh
      self.diff_activation = self.diff_tanh
    else:
      self.f_activation = self.relu
      self.diff_activation = self.diff_relu

    self.alg = alg  #setting up the algorithm for gradient descent either Vanilla or AMSGrad

    #Additional parameters
    self.layers = len(hidden_layers) + 1
    self.learning_rate = learning_rate
    self.epochs = epochs
    self.tol = tol

    #Generating random initial weights including the biases weights
    W = []
    sizes = hidden_layers
    sizes.insert(0, X.shape[1])

    for i in range(self.layers - 1):
      np.random.seed((i + 2) * rdseed) #generating different random weights for every layer
      W.append(np.random.normal(0, 1, size=(sizes[i + 1], sizes[i] + 1)))

    # Finally the weights going to the output layer
    np.random.seed(rdseed)
    W.append(np.random.normal(0, 1, size=(1, sizes[-1] + 1)))

    self.W = np.array(W)

  # Activation functions and their derivatives in terms of the outputs

  #Sigmoid activation function
  def sigmoid(self, x):
    cond = [x > 0, x <= 0] #indices of positives and negative observations
    #applying equivalent transformations of sigmoid to avoid overflow
    sigmoids = [lambda x: 1 / (1 + np.exp(-x)), lambda x: np.exp(x) / (1 + np.exp(x))]

    return np.piecewise(x, cond, sigmoids)

  #Derivative of sigmoid in terms of outputs
  def diff_sigmoid(self, x):
    return x * (1 - x)

  #Tanh written on terms of sigmoid
  def tanh(self, x):
    return 2*self.sigmoid(2*x)-1

  #Derivative of tanh in terms of outputs
  def diff_tanh(self, x):
    return 1 - x * x

  #Relu activation function
  def relu(self, x):
    x[x <= 0] = 0
    return x

  # Derivative of Relu in terms of outputs
  def diff_relu(self, x):
    x[x > 0] = 1
    x[x <= 0] = 0
    return x

  # Method to find the gradient on each layer
  def gradient(self, O, delta):
    return -delta.dot(O.T)

  # Proposed enhancement AMSGrad. Vectorized implementation using the properties of numpy arrays
  def AMSGrad(self, grad, mt, vt, layer):
    # setting up initial parameters
    eps = 1e-7
    # values used in the Neural Network experiment conducted by the authors(https://openreview.net/pdf?id=ryQu7f-RZ): b1 = 0.9 and b2 chosen from {0.99, 0.999}
    beta1 = 0.9
    beta2 = 0.999

    mt[layer] = beta1 * mt[layer] + (1 - beta1) * grad  # updating the momentums for each layer

    vt_new = beta2 * vt[layer] + (1 - beta2) * grad * grad  # finding the new v(s) on the current layer
    vt[layer] = np.fmax(vt_new, vt[layer])  # ensuring that the current v values are always larger than the values from the previous iteration, np.fmax finds element wise maximum

    update_multiplier = mt[layer] / (np.sqrt(vt[layer]) + eps) #final value to be multiply by the learning rate

    return update_multiplier

  #Forward process
  def forward(self, Xi):

    # Input layer
    Xi = np.c_[np.ones(Xi.shape[0]), Xi] # adding column of 1 to find bias in a vectorized fashion
    zi = self.W[0].dot(Xi.T) #finding the net going to hidden layer 1
    O = [Xi.T] #Storing outputs for future backward pass

    for i in range(1, self.layers):
      oi = self.f_activation(zi) #outputs of the activation function
      oi = np.vstack((np.ones(oi.shape[1]), oi))  # adding row of 1 to find bias in a vectorized fashion
      O.append(oi)  # storing the activation values for the later backward process
      zi = self.W[i].dot(oi) #finding the net going to the next layer

    return O, zi  #We are in presence of a regression problem so we use the identity activation on the output layer

  #Enhanced Backward process using AMSGrad
  def backward_AMSGrad(self, Y, O, yi, mt, vt):

    deltaW = [] #list to store the weight changes
    alpha = self.learning_rate
    delta = 1 * (Y - yi)  #finding delta of output layer, since the last layer uses identity function, we have a derivative of 1 times the differences
    grad = self.gradient(O[-1], delta) #calculating the current gradient

    deltaW.append(-alpha * self.AMSGrad(grad, mt, vt, -1)) #uptades for the current layer weights

    delta = self.diff_activation(O[-1]) * self.W[-1].T.dot(delta) #delta for the first hidden layer
    n = len(self.W)

    for i in range(n - 2, -1, -1):
      grad = self.gradient(O[i], delta[1:]) #calculating the current gradient
      deltaW.insert(0, -alpha * self.AMSGrad(grad, mt, vt, i))  #uptades for the current layer weights
      delta = self.diff_activation(O[i]) * self.W[i].T.dot(delta[1:]) #delta of next hidden layer

    return deltaW

  #Traditional backward pass.
  def backward(self, Y, O, yi):

    deltaW = [] #list to store the weight changes
    alpha = self.learning_rate

    delta = 1 * (Y - yi)  #finding delta of output layer, since the last layer uses identity function, we have a derivative of 1 times the differences
    grad = self.gradient(O[-1], delta) #calculating the current gradient

    deltaW.append(-alpha * grad) #uptades for the current layer weights

    delta = self.diff_activation(O[-1]) * self.W[-1].T.dot(delta)
    n = len(self.W)

    for i in range(n - 2, -1, -1):
      grad = self.gradient(O[i], delta[1:]) #calculating the current gradient
      deltaW.insert(0, -alpha * grad) #uptades for the current layer weights
      delta = self.diff_activation(O[i]) * self.W[i].T.dot(delta[1:]) #delta of next hidden layer

    return deltaW

  def backprop(self):

    #If we are using AMSGrad we initialize the momentums and velocities for all weights as 0
    if self.alg == 'AMSGrad':
      mt = []
      vt = []

      for layer_weights in self.W:
        mt.append(np.zeros_like(layer_weights))
        vt.append(np.zeros_like(layer_weights))

    least_error = float("inf") #dummy value to start our analysis
    early_stop = 0 #value to test an early stop
    best_weights = self.W.copy() #variable to track the best weights
    hist_error = [] #list to track historical errors

    for i in range(self.epochs):
      # Randomly splitting our data into training and validation sets, proportion 2 to 1
      Xj, X_val, Yj, Y_val = train_test_split(self.X, self.Y.T, test_size=1/3, random_state=(i + 1) * rdseed)

      O, yi = self.forward(Xj)  # forward pass
      # Backward process either AMSGrad or traditional
      if self.alg == 'AMSGrad':
        deltaW = self.backward_AMSGrad(Yj.T, O, yi, mt, vt)
      else:
        deltaW = self.backward(Yj.T, O, yi)

      self.W += deltaW  # updating the weights after the backward

      y_predict, curr_error, _ = self.predict(X_val, Y_val)  # calculating error on validation set
      hist_error.append(curr_error)  # tracking historical errors

      #checking whether there is improvement or not
      if curr_error>least_error-self.tol:
        early_stop+=1
      else:
        early_stop=0

      #If the error have not improved for the past 50 epochs, we stop the process and return the best weights until now
      if early_stop==50:
        self.W=best_weights
        return hist_error, least_error,  i + 1, self.W

      #Updating the least error and best weights
      if curr_error < least_error:
        least_error = curr_error
        best_weights = self.W.copy()


    self.W=best_weights
    return hist_error, least_error,self.epochs, self.W

  # Function to find predictions using Xi data attributes and Yi outputs. It also computes the coefficient of determination R2 and the MSE for this prediction
  def predict(self, Xi,Yi):
    Xi = np.c_[np.ones(Xi.shape[0]), Xi]
    zi = self.W[0].dot(Xi.T)

    for i in range(1, self.layers):
      oi = self.f_activation(zi)
      oi = np.vstack((np.ones(oi.shape[1]), oi))
      zi = self.W[i].dot(oi)

    mse=mean_squared_error(Yi,zi.T)/2
    r2=r2_score(Yi, zi.T)

    return zi, mse, r2

# Function to handle the data preprocess, it receives data (DataFrame) and % assign to the testing set, default value 0.2
def preprocess(data, test_perc=0.2):
  del data['DP Temp']
  #deleting low correlation attributes
  del data['Humidity']
  del data['Wind speed']
  del data['Visibility']
  del data['Rainfall']
  del data['Snowfall']
  del data['Holiday']

  #Applying sine-cosine transformation to encode the cyclical behaviour of the variable Hour.
  sin_time = np.sin(2. * np.pi * data['Hour'] / max(data['Hour']))
  cos_time = np.cos(2. * np.pi * data['Hour'] / max(data['Hour']))
  data['sin_hour'] = sin_time
  data['cos_hour'] = cos_time

  Y = data.iloc[:, 0] #DataFrame of outputs
  X = data.iloc[:, 1:] #DataFrame of attributes

  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_perc, random_state=rdseed)  # spliting training/testing sets, assigning the percentage: test_perc to the test set, random seed fixed to replicate results

  hour_train=X_train['Hour']
  hour_test=X_test['Hour']
  del X_train['Hour']
  del X_test['Hour']

  scaler = StandardScaler() #scaler to normalize the data
  scaler.fit(X_train.iloc[:, :2])  # we only normalize the continuous variables
  X_train.iloc[:, :2] = scaler.transform(X_train.iloc[:, :2])
  X_test.iloc[:, :2] = scaler.transform(X_test.iloc[:, :2])  # applying same transformation to test data

  return hour_train, hour_test, X_train, X_test, Y_train, Y_test

#Original Data
url='https://raw.githubusercontent.com/rsrjohnson/CS6375/master/SeoulBikeData.csv'
raw_data = pd.read_csv(url,sep=",",encoding= 'unicode_escape')

working_data=raw_data.copy()

if any(working_data.isnull().sum())>0: #checking for NaN values
    print('NaN values found')

working_data.drop_duplicates(inplace=True) #dropping duplicates

del working_data['Date'] #Date variable is difficult, we will not include it in our analysis. See report for more information

working_data = pd.get_dummies(working_data,drop_first=True)  #we have the precessence of several categorical variables, we encode this variables into numerical values using one hot encoding provided by Dataframe.get_dummies


#renaming columns
working_data.columns=['Bike Count','Hour','Temperature','Humidity','Wind speed','Visibility','DP Temp','Solar Rad','Rainfall','Snowfall','Spring','Summer','Winter','Holiday','Funct Day']

hour_train, hour_test, X_train, X_test, Y_train, Y_test = preprocess(working_data) #Final preprocessed data


#Function to carry out the tuning process
def tuning_process():
  logfile = PdfPages('logfile.pdf')  # file created to log our experiments

  activations=['sigmoid','tanh','relu']

  learning_rates = np.linspace(0.0000001, 0.01, 8)

  from collections import defaultdict
  alpha_list1 = defaultdict(list)
  alpha_list2 = defaultdict(list)

  # Comparison Historical MSE and Iterations for a fixed learning rate see logfile pages 1-6
  layout = [(0, 0), (0, 1), (1, 0), (1, 1)]

  for i in range(3):
    for k in range(len(learning_rates)//4):
      fig, axs = plt.subplots(2, 2, figsize=(10, 10))
      fig.suptitle('Historical MSE vs Iterations, Activation ' + activations[i])
      alpha_MSE_vanilla = []
      alpha_MSE_AMSGrad = []
      for j in range(4 * k, 4 * k + 4):
        NN_vanilla = ANN(X_train, Y_train, activations[i], hidden_layers=[4,2], alg='vanilla', learning_rate=learning_rates[j],
                         epochs=200, tol=0.0001)
        NN_improved = ANN(X_train, Y_train, activations[i], hidden_layers=[4,2], alg='AMSGrad', learning_rate=learning_rates[j],
                          epochs=200, tol=0.0001)
        x, y = layout[j % 4][0], layout[j % 4][1]
        axs[x, y].set_title('Learning Rate = ' + str(learning_rates[j].round(8)))
        hist_error, least_error, total_iter, wt = NN_vanilla.backprop()
        alpha_MSE_vanilla.append(least_error)
        axs[x, y].plot(np.linspace(0, total_iter, total_iter), hist_error, 'c--', label='vanilla', alpha=0.8, linewidth=4)

        hist_error, least_error, total_iter, wt = NN_improved.backprop()
        alpha_MSE_AMSGrad.append(least_error)
        axs[x, y].plot(np.linspace(0, total_iter, total_iter), hist_error, 'r', label='AMSGrad', alpha=0.5, linewidth=5)

        axs[x, y].grid(True)

        axs[x, y].legend()

      for ax in axs.flat:
        ax.set(xlabel='Iterations', ylabel='MSE')

      logfile.savefig(fig)
      alpha_list1[activations[i]]+=alpha_MSE_vanilla
      alpha_list2[activations[i]]+=alpha_MSE_AMSGrad

  # Comparison Learning rates and MSE see logfile pages 7, 8, 9
  for f_act in activations:
    fig = plt.figure(figsize=(10, 10))
    plt.plot(learning_rates, alpha_list1[f_act], 'c^', label='vanilla', alpha=0.8, markersize=7)
    plt.plot(learning_rates, alpha_list2[f_act], 'rs', label='AMSGrad', alpha=0.5, markersize=11)
    plt.title('MSE vs Learning Rate, ' + f_act)
    plt.xlabel('Learning Rate')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    logfile.savefig(fig)

  # Additional exploration for AMSGrad
  alpha_list2 = defaultdict(list)
  learning_rates = np.linspace(0.005,2, 8)  # expanding the learning rates range for AMSGrad

  # Comparison Historical MSE and Iterations for a several learning rates see logfile page 10-15
  for i in range(3):
    for k in range(len(learning_rates) // 4):
      fig, axs = plt.subplots(2, 2, figsize=(10, 10))
      fig.suptitle('Historical MSE vs Iterations, '+activations[i])
      alpha_MSE_AMSGrad = []
      for j in range(4 * k, 4 * k + 4):
        x, y = layout[j % 4][0], layout[j % 4][1]
        axs[x, y].set_title('Learning Rate = ' + str(learning_rates[j].round(8)))

        NN_improved = ANN(X_train, Y_train, activations[i], hidden_layers=[4,2], alg='AMSGrad', learning_rate=learning_rates[j], epochs=200, tol=0.0001)

        hist_error, least_error, total_iter, wt = NN_improved.backprop()
        alpha_MSE_AMSGrad.append(least_error)
        axs[x, y].plot(np.linspace(0, total_iter, total_iter), hist_error, 'r.', label='AMSGrad')
        axs[x, y].grid(True)
        axs[x, y].legend()

      for ax in axs.flat:
        ax.set(xlabel='Iterations', ylabel='MSE')
        ax.label_outer()

      logfile.savefig(fig)
      alpha_list2[activations[i]] += alpha_MSE_AMSGrad

  #logfile page 16-18
  for f_act in activations:
    fig = plt.figure(figsize=(10, 10))
    plt.plot(learning_rates, alpha_list2[f_act], 'rs', label='AMSGrad', alpha=0.5, markersize=11)
    plt.title('MSE vs Learning Rate, ' + f_act)
    plt.xlabel('Learning Rate')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    logfile.savefig(fig)




  columns_names = ["Activation","Learning Rate", "Iterations", "MSE", "R2"]
  learning_rates = np.linspace(0.0000001, 0.000005, 15)  # range of values to compare Vanilla and AMSGrad based on previous graphs.
  iterations = [200, 300, 400]

  # Vectors to keep track of Learning Rate, Iterations, MSE and R2
  vanilla_track = []
  amsgrad_track = []

  # For each learning rate we will perform the predifined number of iterations and keep track of MSE and R2 for Vanilla and AMSGrad
  for i in range(3):
    for alfa in learning_rates:
      for iter in iterations:
        NN_vanilla = ANN(X_train, Y_train, activations[i], hidden_layers=[4,2], alg='vanilla', learning_rate=alfa,
                         epochs=iter, tol=0.0001)

        hist_error, least_error, total_iter, wt = NN_vanilla.backprop()
        y_predict, mse, r2 = NN_vanilla.predict(X_train, Y_train)
        vanilla_track.append([activations[i],alfa, total_iter, mse, r2])


  learning_rates = np.linspace(0.2, 1.45, 15)  # Learning rates range for AMSGrad, based of previous analysis

  for i in range(3):
    for alfa in learning_rates:
      for iter in iterations:
        NN_improved = ANN(X_train, Y_train, activations[i], hidden_layers=[4,2], alg='AMSGrad', learning_rate=alfa,
                          epochs=iter, tol=0.0001)

        hist_error, least_error, total_iter, wt = NN_improved.backprop()
        y_predict, mse, r2 = NN_improved.predict(X_train, Y_train)
        amsgrad_track.append([activations[i], alfa, total_iter, mse, r2])

  Comparison_TableV = pd.DataFrame(vanilla_track, columns=columns_names)
  Comparison_TableV.drop_duplicates(inplace=True)
  Comparison_TableV = Comparison_TableV[Comparison_TableV["R2"] > 0]
  Comparison_TableV = Comparison_TableV.sort_values(by='MSE')

  # logfile 19
  fig = plt.figure(figsize=(20, 20))
  plt.title('Performance Comparison Sorted by MSE Vanilla')
  ax = plt.subplot(111)
  ax.axis('off')
  table = ax.table(cellText=Comparison_TableV.values, colColours=['grey'] * Comparison_TableV.shape[1], bbox=[0, 0, 1, 1],
                   colLabels=Comparison_TableV.columns, cellLoc='center')
  table.auto_set_font_size(False)
  table.set_fontsize(17)
  table.scale(2, 2)
  logfile.savefig(fig)

  Comparison_TableA = pd.DataFrame(amsgrad_track, columns=columns_names)
  Comparison_TableA.drop_duplicates(inplace=True)
  Comparison_TableA = Comparison_TableA[Comparison_TableA["R2"] > 0]
  Comparison_TableA = Comparison_TableA.sort_values(by='MSE')

  #logfile 20
  fig = plt.figure(figsize=(20, 20))
  plt.title('Performance Comparison Sorted by MSE AMSGrad')
  ax = plt.subplot(111)
  ax.axis('off')
  table = ax.table(cellText=Comparison_TableA.values, colColours=['grey'] * Comparison_TableA.shape[1], bbox=[0, 0, 1, 1],
                   colLabels=Comparison_TableA.columns, cellLoc='center')
  table.auto_set_font_size(False)
  table.set_fontsize(17)
  table.scale(2, 2)
  logfile.savefig(fig)



  # Selecting best parameters combination
  best_learning_Vanilla={}
  best_learning_AMSGrad={}
  for f in activations:
    temp =  Comparison_TableV[Comparison_TableV['Activation']==f]
    temp = temp[temp['MSE']==temp['MSE'].min()]

    best_learning_Vanilla[f]=temp

    temp = Comparison_TableA[Comparison_TableA['Activation'] == f]
    temp = temp[temp['MSE'] == temp['MSE'].min()]

    best_learning_AMSGrad[f]=temp



  hidden_layers = [[4], [6], [8], [2, 2], [4,2], [4, 4], [6,2] ,[6, 4]]
  vanilla_track = []
  amsgrad_track=[]
  columns_names = ["Algorithm","Activation", "Learning Rate", "Epochs", "Layers", "MSE", "R2"]

  for f in activations:
    for h in hidden_layers:
      if not best_learning_Vanilla[f].empty:
        alfa = best_learning_Vanilla[f].iloc[0, 1]
        total_iter = best_learning_Vanilla[f].iloc[0, 2]
        NN_vanilla = ANN(X_train, Y_train, f, hidden_layers=h.copy(), alg='vanilla', learning_rate=alfa,epochs=total_iter, tol=0.0001)

        hist_error, least_error, total_iter, wt = NN_vanilla.backprop()
        y_predict, mse, r2 = NN_vanilla.predict(X_train, Y_train)
        vanilla_track.append(['Vanilla',f, alfa, total_iter,h, mse, r2])

      if not best_learning_AMSGrad[f].empty:
        alfa = best_learning_AMSGrad[f].iloc[0, 1]
        total_iter = best_learning_AMSGrad[f].iloc[0, 2]
        NN_improved = ANN(X_train, Y_train, f, hidden_layers=h.copy(), alg='AMSGrad',learning_rate=alfa,epochs=total_iter, tol=0.0001)

        hist_error, least_error, total_iter, wt = NN_improved.backprop()
        y_predict, mse, r2 = NN_improved.predict(X_train, Y_train)
        amsgrad_track.append(['AMSGrad',f, alfa, total_iter,h, mse, r2])

  tableV = pd.DataFrame(vanilla_track, columns=columns_names)
  tableA = pd.DataFrame(amsgrad_track, columns=columns_names)

  Comparison_Table = pd.concat([tableV, tableA], axis=0)
  Comparison_Table = Comparison_Table[Comparison_Table["R2"]>0]
  Comparison_Table = Comparison_Table.sort_values(by='MSE')

  # We sorted the comparison Table by least to greatest MSE values. It is easy to notice that AMSGrad outperforms the Vanilla GD for multiple values and with a lesser amount of iterations

  # Table of training tuning process see logfile page 21
  fig = plt.figure(figsize=(20, 20))
  plt.title('Performance Comparison Training Set Sorted by MSE')
  ax = plt.subplot(111)
  ax.axis('off')
  table = ax.table(cellText=Comparison_Table.values, colColours=['grey'] * Comparison_Table.shape[1], bbox=[0, 0, 1, 1],
                   colWidths = [.1, .1, .3, .1, .1, .3, .3],colLabels=Comparison_Table.columns, cellLoc='center')
  table.auto_set_font_size(False)
  table.set_fontsize(17)
  table.scale(2, 2)
  logfile.savefig(fig)

  best_vanilla = tableV[tableV['MSE'] == tableV['MSE'].min()]
  best_amsgrad = tableA[tableA['MSE'] == tableA['MSE'].min()]


  # With our best values already selected we will proceed to work with the Testing set.
  NN_vanilla = ANN(X_train, Y_train, best_vanilla.iloc[0,1], hidden_layers=best_vanilla.iloc[0,4].copy(),
                   alg='vanilla', learning_rate=best_vanilla.iloc[0,2], epochs=best_vanilla.iloc[0,3], tol=0.0001)

  #hist_error, least_error, total_iter, wt = NN_vanilla.backprop()
  NN_vanilla.backprop()
  y_predictV, mse_vanilla, r2_vanilla = NN_vanilla.predict(X_test, Y_test)


  NN_improved = ANN(X_train, Y_train, best_amsgrad.iloc[0,1], hidden_layers=best_amsgrad.iloc[0,4].copy(),
                   alg='AMSGrad', learning_rate=best_amsgrad.iloc[0,2],
                   epochs=best_amsgrad.iloc[0,3], tol=0.0001)

  #hist_error, least_error, total_iter, wt = NN_improved.backprop()
  NN_improved.backprop()
  y_predictA, mse_AMSGrad, r2_AMSGrad = NN_improved.predict(X_test, Y_test)



  d = {'Algorithm': ['Vanilla', 'AMSGrad'],'Activation':[best_vanilla.iloc[0,1],best_amsgrad.iloc[0,1]],
       'Learning Rate':[best_vanilla.iloc[0,2],best_amsgrad.iloc[0,2]],'Layers':[best_vanilla.iloc[0,4],best_amsgrad.iloc[0,4]],
       'MSE': [mse_vanilla, mse_AMSGrad], 'R2': [r2_vanilla, r2_AMSGrad]}
  Testing_Table = pd.DataFrame(data=d)

  #Testing set results logfile 22
  fig = plt.figure(figsize=(10, 10))
  plt.title('Performance Comparison Testing set')
  ax = plt.subplot(111)
  ax.axis('off')
  table = ax.table(cellText=Testing_Table.values, colColours=['grey'] * Testing_Table.shape[1], bbox=[0, 0, 1, 1],
                   colWidths=[.12, .12, .3, .12, .3, .3], colLabels=Testing_Table.columns, cellLoc='center')
  table.auto_set_font_size(False)
  table.set_fontsize(9)
  table.scale(2, 2)
  logfile.savefig(fig)


  # To finalize we will create some graphics with the variables Temperature and Hour vs the Bike Count. logfile 23, 24


  t = (X_test['Temperature'], hour_test)
  msg = ('Temperature', 'Hour')

  y_predictV.shape=Y_test.shape
  y_predictA.shape=Y_test.shape

  for i in range(2):

    fig, axs = plt.subplots(2, figsize=(10, 10))
    plt.rc('legend', fontsize=9)

    axs[0].plot(t[i], Y_test, 'b*', t[i], y_predictV, 'g.')
    axs[0].legend(['Test' + ' Set', 'Vanilla'])
    axs[0].grid(True)

    axs[1].plot(t[i], Y_test, 'b*', t[i], y_predictA, 'r^')
    axs[1].legend(['Test' + ' Set', 'AMSGrad'])
    axs[1].grid(True)
    plt.suptitle('Test' + ' values vs Predicted values, Attribute: ' + msg[i])


    for ax in axs.flat:
      ax.set(xlabel=msg[i], ylabel='Bike Count')
      ax.label_outer()

    logfile.savefig(fig)



  logfile.close()

#Function to get the Neural Net Parameters via console
def getParameters():

  print("Type 'Y' if you want to enter custom values for the parameters to execute the neuronal net, in any other case the net will be executed with the default parameters.")
  print("The default parameters are: activation_func = 'relu', hidden_layers = [6, 4], alg = 'AMSGrad', learning_rate = 0.29, epochs = 400, tol = 0.0001")
  if input() != 'Y':
    return ('relu',[6, 4], 'AMSGrad', 0.29, 400, 0.0001)

  print("Enter the parameters to execute the neural net, in case of an invalid parameter the net will be executed with the default parameter for that value.\n")
  print("Suggested parameters AMSGrad: \n activation_func = 'relu', hidden_layers = [6, 4], learning_rate = 0.29, epochs = 400, tol = 0.0001\n")
  print("Suggested parameters vanilla: \n activation_func = 'sigmoid', hidden_layers = [8], learning_rate = 0.0000008, epochs = 400, tol = 0.0001\n")

  vanilla=False

  print("Enter the value for the algorithm to be executed (AMSGrad or vanilla)")
  algorithm_input = input()
  if algorithm_input == 'vanilla':
    alg = 'vanilla'
    vanilla=True
  else:
    alg = 'AMSGrad'

  print("Enter the value for the activation function (relu, sigmoid or tanh)")
  activation_func_input = input()
  if activation_func_input!='sigmoid' and activation_func_input!='tanh' and activation_func_input!='relu':
    if vanilla:
      activation_func_input='sigmoid'
    else:
      activation_func_input='relu'

  print("Enter the structure of the neuronal net with at least a hidden layer. A sequence of integers separeted by space representing the hidden layers with the number of neurons on each.")
  print("For example: 6 4 creates a neuronal net with 2 hidden layers, the first one with 6 neurons and the second one with 4")
  try:
    hidden_layers = list(map(int, input().split()))
    if len(hidden_layers)==0:
      raise ValueError()
  except ValueError:
    if vanilla:
      hidden_layers=[8]
    else:
      hidden_layers = [6, 4]

  print("Enter the learning rate.")
  try:
    learning_rate = float(input())
  except ValueError:

    if vanilla:
      learning_rate = 0.0000008
    else:
      learning_rate = 0.29

  print("Enter the number of epochs.")
  try:
    epochs = int(input())
  except ValueError:
    epochs = 400

  print("Enter the tolerance.")
  try:
    tol = float(input())
  except ValueError:
    tol = 0.0001

  return activation_func_input,hidden_layers, alg, learning_rate, epochs, tol

#Uncomment to carry out the tuning process and generate a logfile
#tuning_process()

print()
func,hidden_layers, alg, learning_rate, epochs, tol=getParameters()

NN=ANN(X_train,Y_train, func,hidden_layers, alg, learning_rate, epochs, tol)

NN.backprop()
print()
print("Training Set Metrics")
y_predict_trn, mse_trn, r2_trn = NN.predict(X_train, Y_train)
print('MSE:', mse_trn, 'R2:',r2_trn,'\n')

print("Testing Set Metrics")
y_predict, mse, r2 = NN.predict(X_test, Y_test)
print('MSE:', mse, 'R2:',r2,'\n')

print('Press enter to exit')
input()