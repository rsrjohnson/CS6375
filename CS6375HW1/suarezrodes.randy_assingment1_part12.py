#Student Name: Randy Suarez Rodes
#Course: CS 6375.002
#Assignment 1
#September 15, 2020
## IDE: Pycharm Community Version 2019.3, Interpreter Python 3.8, Operating System: 10



#Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

#sklearn packages
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


logfile = PdfPages('logfile.pdf') #file created to log our experiments

rdseed=8466 #seed to replicate results

# Class for the gradient descent methods
class Gradient_Decent:
    # class constructor, storing attributes, output and number of observations
    # NOTE: X is a DataFrame of the Attributes variables where the first column (all 1) is used for finding the intercept estimate and carry on prediction operations in a vectorized style.
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.sample_size = X.shape[0]

    # Method to find the gradient
    # It receives Attributes Xi, Outputs Yi and weights wt
    def gradient(self, Xi, Yi, wt):
        return ((Xi.dot(wt) - Yi).T).dot(Xi) / self.sample_size

    # Vanilla Gradient Descent. Batch implementation using the vectorized properties of pandas.DataFrame
    # It receives given initial weights:wt, learning rate:alpha default value 0.1, maximum number of iterations: it default value 200 and tolerance: tol default value 0.0001
    def vanilla(self, wt, alpha=0.1, it=200, tol=0.0001):

        least_error = mean_squared_error(self.Y, self.X.dot(wt)) / 2  # finding the MSE, dividing by two since sklearn.metrics.mean_squared_error does not
        hist_error = [least_error]  # array to keep track of the error on each iteration

        total_iter = it  # return variable to keep track of the total iterations of the algorithm, set to default in the case of not early stopping

        for k in range(it):

            grad = self.gradient(self.X, self.Y, wt)  # finding the gradient
            wt = wt - alpha * grad  # simultaneous update of the vector of weights

            curr_error = mean_squared_error(self.Y, self.X.dot(wt)) / 2
            hist_error.append(curr_error)
            if abs(least_error - curr_error) < tol: #if our current error does not improve by a value less than the fixed tolerance, we stop the algorithm and record the number of iterations
                total_iter = k + 1
                break
            if curr_error < least_error:
                least_error = curr_error #updating the least error so far

        #returning historical MSE by iteration, total number of iterations and the final weights
        return hist_error, total_iter, wt

    # Proposed enhancement AMSGrad. Batch implementation using the vectorized properties of pandas.DataFrame
    # It receives given initial weights:wt, learning rate:alpha, default value 0.1, maximum number of iterations: it, default value 200 and tolerance: tol, default value 0.0001
    def AMSGrad(self, wt, alpha=0.1, it=200, tol=0.0001):
        # setting up initial parameters
        n = len(wt) #number of weights


        eps = 1e-7
        #values used in the Neural Network experiment conducted by the authors(https://openreview.net/pdf?id=ryQu7f-RZ): b1 = 0.9 and b2 chosen from {0.99, 0.999}
        beta1 = 0.9
        beta2 = 0.999

        #inititializing vectors to zero
        mt = np.zeros(n)
        vt0 = np.zeros(n)

        total_iter = it # return variable to keep track of the total iterations of the algorithm, set to default in the case of not early stopping

        least_error = mean_squared_error(self.Y, self.X.dot(wt)) / 2  # finding the MSE, dividing by two since sklearn.metrics.mean_squared_error does not
        hist_error = [least_error]  # array to keep track of the error on each iteration


        for k in range(it):

            grad = self.gradient(self.X, self.Y, wt) #finding the gradient

            mt = beta1 * mt + (1 - beta1) * grad #updating the momentums

            vt = beta2 * vt0 + (1 - beta2) * grad * grad #finding the new v
            vt = np.fmax(vt, vt0) #ensuring that we current v values are always larger than the values from the previous iteration, np.fmax finds component wise maximum
            vt0 = vt #updating the previous v values to the current ones

            wt = wt - (alpha / (np.sqrt(vt) + eps)) * mt # simultaneous update of the vector of weights

            curr_error = mean_squared_error(self.Y, self.X.dot(wt)) / 2
            hist_error.append(curr_error)
            if abs(least_error - curr_error) < tol: #if our current error does not improve by a value less than the fixed tolerance, we stop the algorithm and record the number of iterations
                total_iter = k + 1
                break
            if curr_error < least_error:
                least_error = curr_error #updating the least error so far

        # returning historical MSE by iteration, total number of iterations and the final weights
        return hist_error, total_iter, wt

    #Function to find predictions using Xi data attributes and the weights wt. It also computes the coefficient of determination R2 and the MSE for this prediction
    def predict(self,Xi,Yi,wt):
        predictions = Xi.dot(wt)
        mse = mean_squared_error(Yi, predictions) / 2
        r2 = r2_score(Yi, predictions)

        return predictions,mse,r2

# Function to handle the data preprocess, it receives data (DataFrame) and % assign to the testing set, default value 0.2
def preprocess(data, test_perc=0.2):
    del data['DP Temp']
    # deleting low correlation attributes, based on correlation matrix
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

    X_train.insert(loc=0, column='intcpt', value=1)  # Creating a column of 1s, used to estimate the intercept.
    X_test.insert(loc=0, column='intcpt', value=1)

    return hour_train, hour_test, X_train, X_test, Y_train, Y_test







#### Part 1

#Original Data
url='https://raw.githubusercontent.com/rsrjohnson/CS6375/master/SeoulBikeData.csv'
raw_data = pd.read_csv(url,sep=",",encoding= 'unicode_escape')

working_data=raw_data.copy()

if any(working_data.isnull().sum())>0: #checking for NaN values
    print('NaN values found')

working_data.drop_duplicates(inplace=True) #dropping duplicates

del working_data['Date'] #Date variable is difficult, we will not include it in our analysis. See report for more information

working_data = pd.get_dummies(working_data,drop_first=True)  #we have the precessence of several categorical variables, we encode this variables into numerical values using one hot encoding provided by Dataframe.get_dummies


print(working_data.describe()) # descriptive statistics of our data.
#we can appreciate how different are the scales of values of our attributes, so we will proceed to normalize our data after splitting the training and testing data sets.

#renaming columns
working_data.columns=['Bike Count','Hour','Temperature','Humidity','Wind speed','Visibility','DP Temp','Solar Rad','Rainfall','Snowfall','Spring','Summer','Winter','Holiday','Funct Day']

#Exploring correlations #See logfile page 1
correlation_matrix = working_data.corr().round(2)
fig=plt.figure(figsize=(10,10))
plt.title('Correlations')
sns.heatmap(data=correlation_matrix, annot=True)
logfile.savefig(fig) #saving image to logfile
#Notice the attributes Temperature and Dew point temperature are highly correlated. Of these two, Temperature provides more information which respect to the response variable, so we will proceed to drop Dew point temperature.
#Several attributes present low correlation with the target variable (Humidity, Wind speed, Visibility, Rainfall, Snowfall and Holiday), we will exclude these variables from our analysis


hour_train, hour_test, X_train, X_test, Y_train, Y_test = preprocess(working_data) #Final preprocessed data, normalized and low correlation variables dropped

#Let us find the coefficients analitically and its error metrics for future comparisons
ols_coeff=np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(Y_train) #using the analytical formula w=([X'X]^-1)X'Y

y_predict_ols_trn=X_train.dot(ols_coeff) #predictions of the training set
mseOls_trn = mean_squared_error(Y_train, y_predict_ols_trn)/2 #MSE training set
r2Ols_trn = r2_score(Y_train, y_predict_ols_trn) #R2 training set

y_predict_ols_test=X_test.dot(ols_coeff) #predictions of the testing set
mseOls = mean_squared_error(Y_test, y_predict_ols_test)/2 #MSE testing set
r2Ols = r2_score(Y_test, y_predict_ols_test) #R2 training set


####Part 1 tuning process

np.random.seed(rdseed) #Seed to replicate results
thetas = np.random.normal(0, 1, X_train.shape[1])  # initial weights initialize with a normal distribution

learning_rates = np.linspace(0.000001, 1, 12)  # Exploring possible initial learning rates.

GD = Gradient_Decent(X_train, Y_train) #initializing our class object

# List to keep track of MSE by iteration.
alpha_MSE_vanilla = []
alpha_MSE_AMSGrad = []

# Comparison Historical MSE and Iterations for a fixed learning rate see logfile pages 2-4
layout=[(0,0),(0,1),(1,0),(1,1)]

for i in range(3):
    fig, axs = plt.subplots(2, 2,figsize=(10,10))
    fig.suptitle('Historical MSE vs Iterations')
    for j in range(4*i,4*i+4):
        x,y=layout[j%4][0],layout[j%4][1]
        axs[x,y].set_title('Learning rate = ' + str(learning_rates[j].round(6)))
        hist_error, total_iter, wt = GD.vanilla(thetas.copy(), alpha=learning_rates[j])
        alpha_MSE_vanilla.append(hist_error[-1])
        axs[x,y].plot(np.linspace(0, total_iter, total_iter + 1), hist_error, 'c--', label='vanilla', alpha=0.8, linewidth=4)

        hist_error, total_iter, wt = GD.AMSGrad(thetas.copy(), alpha=learning_rates[j])
        alpha_MSE_AMSGrad.append(hist_error[-1])
        axs[x,y].plot(np.linspace(0, total_iter, total_iter + 1), hist_error, 'r', label='AMSGrad', alpha=0.5, linewidth=5)

        axs[x,y].grid(True)

        axs[x,y].legend()

    for ax in axs.flat:
        ax.set(xlabel='Iterations', ylabel='MSE')



    logfile.savefig(fig)




# Comparison Learning rates and MSE see logfile page 5
fig=plt.figure(figsize=(10, 10))
plt.plot(learning_rates, alpha_MSE_vanilla, 'c^', label='vanilla', alpha=0.8,markersize=7)
plt.plot(learning_rates, alpha_MSE_AMSGrad, 'rs', label='AMSGrad', alpha=0.5,markersize=11)
plt.title('MSE vs Learning rate')
plt.xlabel('Learning rate')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
logfile.savefig(fig)


# Additional exploration for AMSGrad
alpha_MSE_AMSGrad = []

learning_rates = np.linspace(1, 100, 12)  # expanding the learning rates range for AMSGrad see logfile page 6-8

# Comparison Historical MSE and Iterations for a several learning rates
for i in range(3):
    fig, axs = plt.subplots(2, 2,figsize=(10,10))
    fig.suptitle('Historical MSE vs Iterations')
    for j in range(4*i,4*i+4):
        x,y=layout[j%4][0],layout[j%4][1]
        axs[x,y].set_title('Learning rate = ' + str(learning_rates[j].round(6)))

        hist_error, total_iter, wt = GD.AMSGrad(thetas.copy(), alpha=learning_rates[j])
        alpha_MSE_AMSGrad.append(hist_error[-1])
        axs[x,y].plot(np.linspace(0, total_iter, total_iter + 1), hist_error, 'r.', label='AMSGrad')
        axs[x,y].grid(True)
        axs[x,y].legend()

    for ax in axs.flat:
        ax.set(xlabel='Iterations', ylabel='MSE')
        ax.label_outer()


    logfile.savefig(fig)
##end of pages 6-8


# Comparison Learning rates and MSE see logfile page 9
# We can see AMSGrad shows improvements on learning rates greater than 40.
fig=plt.figure(figsize=(10, 10))
plt.plot(learning_rates, alpha_MSE_AMSGrad, 'rs', label='AMSGrad', alpha=0.5)
plt.title('MSE vs Learning rate')
plt.xlabel('Learning rate')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
logfile.savefig(fig)
#end of page 9




# Final Tuning Learning Rates and iterations
columns_names=["Learning Rate","Iterations","MSE","R2"]
learning_rates=np.linspace(0.05,0.95,6) #range of values to compare Vanilla and AMSGrad based on previous graphs.
iterations=[200,500,1000]

#Vectors to keep track of Learning Rate, Iterations, MSE and R2
vanilla_track=[]
amsgrad_track=[]

#For each learning rate we will perform the predifined number of iterations and keep track of MSE and R2 for Vanilla and AMSGrad
for alfa in learning_rates:
  for iter in iterations:
    histE, total_iter, wt=GD.vanilla(thetas.copy(),alpha=alfa,it=iter)
    y_predict,mse,r2=GD.predict(X_train,Y_train,wt)
    vanilla_track.append([alfa,total_iter,mse,r2])

    histE, total_iter, wt=GD.AMSGrad(thetas.copy(),alpha=alfa,it=iter)
    y_predict,mse,r2=GD.predict(X_train,Y_train,wt)
    amsgrad_track.append([alfa,total_iter,mse,r2])


learning_rates=np.linspace(40,120,21) #extra learning rates range for AMSGrad, based of previous analysis

for alfa in learning_rates:
  for iter in iterations:
    histE, total_iter, wt=GD.AMSGrad(thetas.copy(),alpha=alfa,it=iter)
    y_predict,mse,r2=GD.predict(X_train,Y_train,wt)
    amsgrad_track.append([alfa,total_iter,mse,r2])


Comparison_TableV=pd.DataFrame(vanilla_track,columns=columns_names)
Comparison_TableV.insert(loc=0,column='Algorithm',value="Vanilla")

Comparison_TableA=pd.DataFrame(amsgrad_track,columns=columns_names)
Comparison_TableA.insert(loc=0,column='Algorithm',value="AMSGrad")

Comparison_Table= pd.concat([Comparison_TableV, Comparison_TableA], axis=0)
Comparison_Table.drop_duplicates(inplace=True)
Comparison_Table = Comparison_Table.sort_values(by ='MSE' )
#We sorted the comparison Table by least to greatest MSE values. It is easy to notice that AMSGrad outperforms the Vanilla GD for multiple values and with a lesser amount of iterations

#Table of tuning process see logfile page 10
fig = plt.figure(figsize=(20,20))
plt.title('Performance Comparison Sorted by MSE')
ax=plt.subplot(111)
ax.axis('off')
table = ax.table(cellText=Comparison_Table.values, colColours=['grey']*Comparison_Table.shape[1], bbox=[0, 0, 1, 1], colLabels=Comparison_Table.columns,cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(17)
table.scale(2,2)
logfile.savefig(fig)


#Selecting best parameters combination
best_learning_Vanilla=Comparison_TableV[Comparison_TableV['MSE']==Comparison_TableV['MSE'].min()]
best_learning_Vanilla.drop_duplicates(inplace=True)
best_learning_AMSGrad=Comparison_TableA[Comparison_TableA['MSE']==Comparison_TableA['MSE'].min()]
best_learning_AMSGrad.drop_duplicates(inplace=True)



#With our best values already selected we will proceed to work with the Testing set.
_ ,  _ , final_weights_vanilla=GD.vanilla(thetas.copy(),alpha=best_learning_Vanilla.iloc[0,1],it=best_learning_Vanilla.iloc[0,2]) #fitting with the best parameters
test_predict_vanilla, mse_vanilla,r2_vanilla=GD.predict(X_test,Y_test,final_weights_vanilla) #predicting the testing set

_, _, final_weights_AMSGrad=GD.AMSGrad(thetas.copy(),alpha=best_learning_AMSGrad.iloc[0,1],it=best_learning_AMSGrad.iloc[0,2]) #fitting with the best parameters
test_predict_AMSGrad,mse_AMSGrad,r2_AMSGrad=GD.predict(X_test,Y_test,final_weights_AMSGrad) #predicting the testing set


d={'Algorithm':['Vanilla','AMSGrad'],'MSE':[mse_vanilla,mse_AMSGrad],'R2':[r2_vanilla,r2_AMSGrad]}
Testing_Table=pd.DataFrame(data=d)

#Predictions of training and testing sets for future comparisons
vanilla_vals=(GD.predict(X_train,Y_train,final_weights_vanilla)[0],test_predict_vanilla)
AMSGrad_vals=(GD.predict(X_train,Y_train,final_weights_AMSGrad)[0],test_predict_AMSGrad)








#### Part 2

from sklearn.linear_model import SGDRegressor
#On sklearn there is not an exact algorithm to match the AMSGrad algorithm, hence we tried to configure SGDRegressor as closest as possible to this algorithm
#For this, we eliminated the regularization penalty function, set the same tolerance treshold, no initial a random_state and n_iter_no_change set to 1 to match our stopping condition.


#We start by deleting the bias column since sklearn library does not require it
del X_train['intcpt']
del X_test['intcpt']

#Exploring learning rates
learning_rates = np.linspace(0.00001, 1, 30) #Initial learning rate range

# List to keep track of MSE by iteration.
alpha_SGDRegressor = []


for alfa in learning_rates:
    reg = SGDRegressor(penalty=None, tol=0.0001, shuffle=False, random_state=None, eta0=alfa,n_iter_no_change=1)
    reg.fit(X_train, Y_train)
    y_predict = reg.predict(X_train)
    mse = mean_squared_error(Y_train, y_predict) / 2
    alpha_SGDRegressor.append(mse)


#Graph on logfile page 11. We can see that the algorithm convergence gets worse as the learning rates increases
fig=plt.figure(figsize=(10, 10))
plt.plot(learning_rates, alpha_SGDRegressor, 'mo', label='SGDRegressor', alpha=0.8)
plt.title('MSE vs Learning rate')
plt.xlabel('Learning rate')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
logfile.savefig(fig)





#Tuning Learning Rates and iterations
columns_names_ML = ["Learning Rate", "Iterations", "MSE", "R2"]

learning_rates = np.linspace(0.0001, 0.1,16 ) #Tuning the learning rate, based on previous graph findings
iterations=[200,500,1000]

SGDRegressor_track = []

for alfa in learning_rates:
    for iter in iterations:
        reg = SGDRegressor(penalty=None, max_iter=iter, tol=0.0001, shuffle=False,  eta0=alfa,n_iter_no_change=1)
        reg.fit(X_train, Y_train)
        y_predict = reg.predict(X_train)
        mse = mean_squared_error(Y_train, y_predict) / 2
        r2 = r2_score(Y_train, y_predict)

        SGDRegressor_track.append([alfa, iter, mse, r2])



Comparison_Table_ML = pd.DataFrame(SGDRegressor_track, columns=columns_names_ML)
Comparison_Table_ML = Comparison_Table_ML.sort_values(by ='MSE' )

#Table of tuning process see logfile page 12
fig = plt.figure(figsize=(20,20))
plt.title('SGDRegressor Performance Comparison Sorted by MSE')
ax=plt.subplot(111)
ax.axis('off')
table = ax.table(cellText=Comparison_Table_ML.values, colColours=['grey']*Comparison_Table_ML.shape[1], bbox=[0, 0, 1, 1], colLabels=Comparison_Table_ML.columns,cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(2,2)
logfile.savefig(fig)

# Selecting best parameters combination
best_learning_SGD = Comparison_Table_ML[Comparison_Table_ML['MSE'] == Comparison_Table_ML['MSE'].min()]
best_learning_SGD.drop_duplicates(inplace=True)
best_learning_SGD['Algorithm']=['SGDRegressor']


#We added the analytical coefficients metrics for extra comparison
d={'Algorithm': ['Analytical Coeff'],'Learning Rate':'N/A','Iterations':'N/A', 'MSE': mseOls_trn, 'R2': r2Ols_trn}
analytical_Table=pd.DataFrame(data=d)
Training_Table=pd.concat([best_learning_Vanilla,best_learning_AMSGrad,best_learning_SGD,analytical_Table],axis=0)


#Table of comparison of Training Set results. see logfile page 13
fig = plt.figure(figsize=(10,10))
plt.title('MSE and R2 from Training Set using the best Previously Tuned parameters')
ax=plt.subplot(111)
ax.axis('off')
table = ax.table(cellText=Training_Table.values, colColours=['grey']*Training_Table.shape[1], bbox=[0, 0, 1, 1], colLabels=Training_Table.columns,cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(2,2)
logfile.savefig(fig)




# Testing set analysis
final_weights_SGD = SGDRegressor(penalty=None, max_iter=best_learning_SGD.iloc[0, 1], tol=0.0001, shuffle=False, learning_rate='invscaling', eta0=best_learning_SGD.iloc[0, 0],n_iter_no_change=1)
final_weights_SGD.fit(X_train, Y_train)
test_predict_SGD = final_weights_SGD.predict(X_test)

mse_SGD = mean_squared_error(Y_test, test_predict_SGD) / 2
r2_SGD = r2_score(Y_test, test_predict_SGD)


d = {'Algorithm': ['SGDRegressor'], 'MSE': [mse_SGD], 'R2': [r2_SGD]}
Testing_Table_ML = pd.DataFrame(data=d)

#Performance Comparison on testing set, see logfile page 14
Combined_Table= pd.concat([Testing_Table,Testing_Table_ML],axis=0)

fig = plt.figure(figsize=(10,10))
plt.title('Performance Comparison Testing set')
ax=plt.subplot(111)
ax.axis('off')
table = ax.table(cellText=Combined_Table.values, colColours=['grey']*Combined_Table.shape[1], bbox=[0, 0, 1, 1], colLabels=Combined_Table.columns,cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(2,2)
logfile.savefig(fig)



#Coefficients Comparison see page 15
Final_Comp=pd.DataFrame()

Final_Comp['Vanilla']=final_weights_vanilla
Final_Comp['AMSGrad']=final_weights_AMSGrad
Final_Comp['SGDRegressor']=np.concatenate((final_weights_SGD.intercept_, final_weights_SGD.coef_), axis=None)
Final_Comp['Analitical Coef']=ols_coeff


fig = plt.figure(figsize=(10,10))
ax=plt.subplot(111)
ax.axis('off')
table = ax.table(cellText=Final_Comp.values, colColours=['grey']*Final_Comp.shape[1], bbox=[0, 0, 1, 1], colLabels=Final_Comp.columns,rowLabels=Final_Comp.index,cellLoc='center',colWidths=[0.1]*Final_Comp.shape[1])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(2,2)
logfile.savefig(fig)


#To finalize we will create some graphics with the variables Temperature and Hour vs the Bike Count. logfile 16-19

t1=(X_train['Temperature'],X_test['Temperature'])
t2=(hour_train,hour_test)
t=(t1,t2)

ys=(Y_train,Y_test)

msg=('Temperature','Hour')
msg1=('Trn','Test')


SGD_vals=(final_weights_SGD.predict(X_train),test_predict_SGD)
ols_vals=(y_predict_ols_trn,y_predict_ols_test)


for i in range(2):
    for j in range(2):

        fig, axs = plt.subplots(2, 2,figsize=(10,10))
        plt.rc('legend',fontsize=9)

        axs[0, 0].plot(t[i][j],ys[j],'b*',t[i][j],vanilla_vals[j],'c.')
        axs[0, 0].legend([msg1[j]+' Set','Vanilla'])
        axs[0, 0].grid(True)

        axs[0, 1].plot(t[i][j],ys[j],'b*',t[i][j],AMSGrad_vals[j],'r^')
        axs[0, 1].legend([msg1[j]+' Set','AMSGrad'])
        axs[0, 1].grid(True)


        axs[1, 0].plot(t[i][j],ys[j],'b*',t[i][j],SGD_vals[j],'m+')
        axs[1, 0].legend([msg1[j]+' Set','SGDRegressor'])
        axs[1, 0].grid(True)


        axs[1, 1].plot(t[i][j],ys[j],'b*',t[i][j],ols_vals[j],'g2')
        axs[1, 1].legend([msg1[j]+' Set','Analytical'])
        axs[1, 1].grid(True)
        plt.suptitle(msg1[j]+' values vs Predicted '+msg1[j]+' values, Attribute: '+msg[i])

        for ax in axs.flat:
            ax.set(xlabel=msg[i], ylabel='Bike Count')
            ax.label_outer()

        logfile.savefig(fig)

logfile.close()

