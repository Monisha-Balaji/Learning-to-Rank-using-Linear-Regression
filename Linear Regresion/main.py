# import statements help import functionalities to be used by the program
from sklearn.cluster import KMeans      # Class sklear.cluster enables using KMeans for clustering the dataset
import numpy as np                      # For the use of number functions like array,arange etc.
import csv                              # Enables processing the CSV(comma-separated values) files
import math                             # Enables the use of mathematical operations like exponentials, powers,squareroot etc.
import matplotlib.pyplot                # Enables plotting graphs 
from matplotlib import pyplot as plt

maxAcc = 0.0
maxIter = 0
C_Lambda = 0.07                         # Coefficient of the Weight decay regularizer term; regularizer term helps avoid overfitting of data
TrainingPercent = 80                    # Given data set is partitioned; 80% of the dataset is assigned for training 
ValidationPercent = 10                  # Given data set is partitioned; 10% of the dataset is assigned for validation
TestPercent = 10                        # Given data set is partitioned; 10% of the dataset is assigned for testing
M = [3]                                # No of Basis functions; M=[3,5,7,10,12,15,100,500] is considered for graph construction;M is assigned only one value in this code as running Closed-form for multiple values of M and then sequentially running SGD will cause 'math range' error
PHI = []                                # this is the design matrix that comprises of all the basis functions
IsSynthetic = False

# Create the Target vector by assessing output information from the target file 
def GetTargetVector(filePath):
    t = []
    with open(filePath, 'rU') as f:
        reader = csv.reader(f)                         #Reads the Target csv file
        for row in reader:  
            t.append(int(row[0]))                      #Appends Values from each row of the file to the target array
    #print("Raw Training Generated..")
    return t

# Create the input feature matrix by assessing the values from the input_data csv file 
def GenerateRawData(filePath, IsSynthetic):    
    dataMatrix = [] 
    with open(filePath, 'rU') as fi:
        reader = csv.reader(fi)                        # Reads from the input csv
        for row in reader:                             # Iterates through each row in the file
            dataRow = []
            for column in row:                         # Iterates through each column in the file
                dataRow.append(float(column))          # Creates the data matrix by appending data row by row, reading through each of the columns
            dataMatrix.append(dataRow)   
    
    if IsSynthetic == False :
        # Delete the feature columns that have variance=0 i.e preprocessig data 
        dataMatrix = np.delete(dataMatrix, [5,6,7,8,9], axis=1)   
    dataMatrix = np.transpose(dataMatrix)     #Take the transpose of the Data matrix
    #print ("Data Matrix Generated..")
    return dataMatrix

# Create the Target vector for the Training dataset after partitioning the data  
def GenerateTrainingTarget(rawTraining,TrainingPercent = 80):
    # Calculate the number of data that corresponds to 80% of the original dataset
    TrainingLen = int(math.ceil(len(rawTraining)*(TrainingPercent*0.01)))
   
    t           = rawTraining[:TrainingLen]
    #print(str(TrainingPercent) + "% Training Target Generated..")
    return t

# Create the input feature data matrix for the Training dataset that corresponds to 80% of the whole data
def GenerateTrainingDataMatrix(rawData, TrainingPercent = 80):
    # Calculate the number of data rows that corresponds to 80% of the original dataset
    T_len = int(math.ceil(len(rawData[0])*0.01*TrainingPercent))
    # Create a new Data matrix with only 80% of the Data
    d2 = rawData[:,0:T_len]
    #print(str(TrainingPercent) + "% Training Data Generated..")
    return d2

# Create the input feature data matrix for the Validation dataset that corresponds to 10% of the whole data
def GenerateValData(rawData, ValPercent, TrainingCount): 
    # Calculate the number of data rows that corresponds to 10% of the original dataset without overlapping with the Training Data 
    valSize = int(math.ceil(len(rawData[0])*ValPercent*0.01))
    # Compute the ending index for the validation set
    V_End = TrainingCount + valSize
    # Create a new Data matrix with only 10% of the Data
    dataMatrix = rawData[:,TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Data Generated..")  
    return dataMatrix

# Create the Target vector for the Validation dataset after partitioning the data i.e. 10% of the original dataset
def GenerateValTargetVector(rawData, ValPercent, TrainingCount): 
     # Calculate the number of target rows that corresponds to 10% of the original dataset without overlapping with the Training Data
    valSize = int(math.ceil(len(rawData)*ValPercent*0.01))
    # Compute the ending index for the validation set
    V_End = TrainingCount + valSize
    # Create a new target vector with only 10% of the Data
    t =rawData[TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Target Data Generated..")
    return t

# Computes the Spread Of the Radial Basis Functions i.e the variance
def GenerateBigSigma(Data, MuMatrix,TrainingPercent,IsSynthetic):
    BigSigma    = np.zeros((len(Data),len(Data)))                 # Computing a matrix of 41x41 with entries as zero as the length of Data is 41; corresponds to the number of rows
    DataT       = np.transpose(Data)                              # Computing the transpose of the Data matrix; the dimensions are now (69623,41)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))    # Computing the Length of the training data set which is 55699   
    varVect     = []                                              # Initializing an array to store the variance
    for i in range(0,len(DataT[0])):                              # running the loop from 0 to 41
        vct = []
        for j in range(0,int(TrainingLen)):                       # running an inner loop from 0 to 55699
            vct.append(Data[i][j])                                # append the values in Date[i][j] to the vct array
        varVect.append(np.var(vct))                               # Compute the variance for the features
        
    for j in range(len(Data)):                                    # Iterating 41 times
        BigSigma[j][j] = varVect[j]                               # Appending the computed values of variance along the diagonal of the Covariance matrix
    if IsSynthetic == True:
        BigSigma = np.dot(3,BigSigma)                             # If the condition is true, BigSigma matrix is multiplied by a scalar value 3
    else:
        BigSigma = np.dot(200,BigSigma)                           # If the condition is false,then BigSigma matrix is multiplied by a scalar value 200
    ##print ("BigSigma Generated..")
    return BigSigma                                               # Return the BigSigma matrix

# Calculate the Value of the terms In the powers of the exponential term of the guassian RBF
def GetScalar(DataRow,MuRow, BigSigInv):  
    R = np.subtract(DataRow,MuRow)          # Subtract the values of inputs and the mean and store in R
    T = np.dot(BigSigInv,np.transpose(R))   # Multiply the transpose of R with the Covariance matrix(BigSigma) and store in T
    L = np.dot(R,T)                         # Dot product of R and T gives a scalar value
    return L                                # Return the scalar value

# Calculate the Gaussian radial basis function
def GetRadialBasisOut(DataRow,MuRow, BigSigInv):    
    phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv))   # Calculate the gaussian RBF by the formula i.e exponential of negavtive 0.5 multiplied with the calculated scalar value 
    return phi_x                                                # Return the value of the basis function 

# Generate the design matrix PHI that contains the basis functions for all input features
def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 80):
    DataT = np.transpose(Data)                                  # Tranpose of the Data matrix; dimensions are now (69623,41)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))  # Length of the training set which is 55699    
    PHI = np.zeros((int(TrainingLen),len(MuMatrix)))            # Initialize a Matrix (80% data)xM with entries as zeroes i.e (55699,10)
    BigSigInv = np.linalg.inv(BigSigma)                         # Inverse of the BigSigma matrix 
    for  C in range(0,len(MuMatrix)):                           # running a loop from 0 to 15
        for R in range(0,int(TrainingLen)):                     # running an inner loop from 0 to 55699
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv)  # Calculate the RBF value using the formula
    #print ("PHI Generated..")
    return PHI                                                  # Return the design matrix

# Compute the weights of the Closed form solution to minimize the error
def GetWeightsClosedForm(PHI, T, Lambda):
    Lambda_I = np.identity(len(PHI[0]))      # Create an indentity matrix with dimenions as (15,15)     
    # Computing the regularization term of the Closed form equation i.e Lambda multipled with the identity matrix
    for i in range(0,len(PHI[0])):
        Lambda_I[i][i] = Lambda                   # Gives Lambda along the diagonal
    # Compute the Closed form solution equation 
    PHI_T       = np.transpose(PHI)               # Transpose of the PHI matrix i.e. dimensions are (15, 55699)
    PHI_SQR     = np.dot(PHI_T,PHI)               # Dot product of the Phi matrix with its Transpose. Dimensions are (15,15)
    PHI_SQR_LI  = np.add(Lambda_I,PHI_SQR)        # Add the product with the Regularization term. Dimensions are (15,15)
    PHI_SQR_INV = np.linalg.inv(PHI_SQR_LI)       # Compute Inverse of the sum 
    INTER       = np.dot(PHI_SQR_INV, PHI_T)      # Resultant matrix is multipled with the transpose of PHI. Dimensions are (15, 55699)
    W           = np.dot(INTER, T)                # Finally multipled with the target values of the training set giving a (15,1) shape
    ##print ("Training Weights Generated..")
    return W                                      # Return the weight matrix of dimensions (15,1)

# Calulate the Target values of the Testing dataset using the calculated weights
def GetValTest(VAL_PHI,W):
    Y = np.dot(W,np.transpose(VAL_PHI))     # Compute the target values from the product of the adjusted weights and PHI
    ##print ("Test Out Generated..")
    return Y                                # return the Target vector for test dataset

# Calculate the root mean square value for the Validation data output with respect to the actual validation output
def GetErms(VAL_TEST_OUT,ValDataAct):
    sum = 0.0
    t=0
    accuracy = 0.0
    counter = 0
    val = 0.0
    # Compute Sum of the square of differences between the Predicted and Actual data
    for i in range (0,len(VAL_TEST_OUT)):                           # Running a loop from 0 to 15 times
        sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]),2)   # Calculate the Differene in prediction
        # Increment counter if the predicted value is equal to the actual value
        if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]):    # np.around() rounds the number to the given number of decimals
            counter+=1
    accuracy = (float((counter*100))/float(len(VAL_TEST_OUT)))      # Compute accuarcy of the validation set
    ##print ("Accuracy Generated..")
    ##print ("Validation E_RMS : " + str(math.sqrt(sum/len(VAL_TEST_OUT))))
    return (str(accuracy) + ',' +  str(math.sqrt(sum/len(VAL_TEST_OUT))))  # Compute the Accuracy and RMS by finding the squareroot of the mean i.e sum/N; Returns both the values separated by ','
    
## Fetch and Prepare Dataset    
RawTarget = GetTargetVector('Querylevelnorm_t.csv')                 # Fetch raw target values
RawData   = GenerateRawData('Querylevelnorm_X.csv',IsSynthetic)     # Fetch raw data values

## Prepare Training Data
TrainingTarget = np.array(GenerateTrainingTarget(RawTarget,TrainingPercent))   # Create Training target
TrainingData   = GenerateTrainingDataMatrix(RawData,TrainingPercent)           # Create Training Data
print(TrainingTarget.shape)                                                    # Print the dimensions of Training target
print(TrainingData.shape)                                                      # Print the dimensions of Training data

## Prepare Validation Data
ValDataAct = np.array(GenerateValTargetVector(RawTarget,ValidationPercent, (len(TrainingTarget)))) # Create Validation target
ValData    = GenerateValData(RawData,ValidationPercent, (len(TrainingTarget)))                     # Create Validation Data
print(ValDataAct.shape)                                                                            # Print the dimensions of Validation target
print(ValData.shape)                                                                               # Print the dimensions of Validation data

## Prepare Test Data
TestDataAct = np.array(GenerateValTargetVector(RawTarget,TestPercent, (len(TrainingTarget)+len(ValDataAct)))) # Create Testing target
TestData = GenerateValData(RawData,TestPercent, (len(TrainingTarget)+len(ValDataAct)))                        # Create Testing Data
print(ValDataAct.shape)                                                                                       # Print the dimensions of Testing target
print(ValData.shape)                                                                                          # Print the dimensions of Testing data

## Closed Form Solution [Finding Weights using Moore- Penrose pseudo- Inverse Matrix]
ErmsArr = []
AccuracyArr = []
# Cluster the Training dataset into M clusters using KMeans
for i in M:
    kmeans = KMeans(n_clusters=i, random_state=0).fit(np.transpose(TrainingData))
    # Form the Mu matrix with the centers of the M clusters 
    Mu = kmeans.cluster_centers_  
    
    BigSigma     = GenerateBigSigma(RawData, Mu, TrainingPercent,IsSynthetic)    # Compute Spread of Basis functions i.e. covariance matrix (Function call to GenerateBigSigma())
    TRAINING_PHI = GetPhiMatrix(RawData, Mu, BigSigma, TrainingPercent)          # Compute the design matrix for training set (Function call to GetPhiMatrix())
    W            = GetWeightsClosedForm(TRAINING_PHI,TrainingTarget,(C_Lambda))  # Compute weights (Function call to GetWeightsClosedForm())
    TEST_PHI     = GetPhiMatrix(TestData, Mu, BigSigma, 100)                     # Compute the desgin matrix for testing set (Function call to GetPhiMatrix())
    VAL_PHI      = GetPhiMatrix(ValData, Mu, BigSigma, 100)                      # Compute the desgin matrix for validation set (Function call to GetPhiMatrix())
    
    ## Finding Erms on training, validation and test set 
    TR_TEST_OUT  = GetValTest(TRAINING_PHI,W)                         # Compute the target values for training set
    VAL_TEST_OUT = GetValTest(VAL_PHI,W)                              # Compute the target values for validation set
    TEST_OUT     = GetValTest(TEST_PHI,W)                             # Compute the target values for testing set
    
    TrainingAccuracy   = str(GetErms(TR_TEST_OUT,TrainingTarget))     # Calculate accuracy for computed training target values
    ValidationAccuracy = str(GetErms(VAL_TEST_OUT,ValDataAct))        # Calculate accuracy for computed validation target values
    TestAccuracy       = str(GetErms(TEST_OUT,TestDataAct))           # Calculate accuracy for computed testing target values
    print ("E_rms Training = " + str(float(TrainingAccuracy.split(',')[1])))         # Print Erms for training set; Converts the Second component of the TrainingAccuracy term i.e. the value after ',' which is the E-Rms value to a float 
    print ("E_rms Validation = " + str(float(ValidationAccuracy.split(',')[1])))       # Print Erms for validation set; Converts the Second component of the ValidationAccuracy term i.e. the value after ',' which is the E-Rms value to a float
    print ("E_rms Testing    = " + str(float(TestAccuracy.split(',')[1])))             # Print Erms for testing set; Converts the Second component of the TestAccuracy term i.e. the value after ',' which is the E-Rms value to a float
    ErmsArr.append(float(TestAccuracy.split(',')[1]))                 #E-RMS values calculated for each of the different values of M is appended to the list for graph construction
    AccuracyArr.append(float(TestAccuracy.split(',')[0]))             #Accuracy values calculated for each of the different values of M is appended to the list for graph construction
 
print ('UBITname      = XXXXXXXX')
print ('Person Number = YYYYYYYY')
print ('----------------------------------------------------')
print ("------------------LeToR Data------------------------")
print ('----------------------------------------------------')
print ("-------Closed Form with Radial Basis Function-------")
print ('----------------------------------------------------') 
print(Mu.shape)                            # Print dimensions of Mu matrix
print(BigSigma.shape)                      # Print dimensions of covariance matrix
print(TRAINING_PHI.shape)                  # Print dimensions of training desgin matrix
print(W.shape)                             # Print dimensions of Weight vector
print(VAL_PHI.shape)                       # Print dimensions of validation design matrix
print(TEST_PHI.shape)                      # Print dimensions of testing design matrix


# GRAPH FOR NO OF BASIS FUNCTIONS VS E-RMS
plt.plot([3,5,7,10,12,15,50,100,500],[0.6317812714668385,0.6311308539478039,0.6288051919072434,0.6281031608753205,0.6293846550733718,0.6277580658083383,0.6205927045587823,0.6192613317793846,0.6174755514821012],'ro')
plt.ylabel('E-RMS Value for Testing')
plt.xlabel("No of Basis Functions")
plt.title("Root Mean Square Error(E-RMS) Plot")
plt.show()

# GRAPH FOR NO OF BASIS FUNCTIONS VS TESTING ACCURACY
plt.plot([3,5,7,10,12,15,50,100,500], [70.21979600632093,70.11923574199109,70.16233299813246,69.91811521333142,69.65953167648327,69.31475362735239,69.27165637121104,68.94124407412728,68.79758655365609],'ro')
plt.ylabel('Testing Accuracy of the model')
plt.xlabel("No of Basis Functions")
plt.title("Accuracy Plot")
plt.show()

# GRAPH FOR REGULARIZATION TERM VS TESTING ACCURACY
plt.plot([0.005,0.05,0.09,0.25,0.5],[70.21979600632093,70.21979600632093,70.21979600632093,70.21979600632093,70.21979600632093])
plt.ylabel('Testing Accuracy of the model')
plt.xlabel("Regularization term(Closed-Form)")
plt.title("Regularization term VS Accuracy Plot")
plt.show()

## Gradient Descent solution for Linear Regression
print ('----------------------------------------------------')
print ('--------------Please Wait for 2 mins!----------------')
print ('----------------------------------------------------')
W_Now        = np.dot(220, W)                                           # weights are initialized by multiplying the weights calculated from closed form with a scalar value 220
La           = 2                                                        # Regularization term(lambda)
learningRate = [0.001,0.002,0.003,0.005,0.009,0.01,0.015,0.02,0.03]     # Learning rate determines how big the sizes of the learning steps will be; It is multiplied with the derivative of the cost function in the weight updation process
L_Erms_Val   = []                                                       # An empty list is initialized to store the Erms for validation data
L_Erms_TR    = []                                                       # An empty list is initialized to store the Erms for Training data
L_Erms_Test  = []                                                       # An empty list is initialized to store the Erms for testing data
W_Mat        = []                                                       # An empty list is initialized to store the updated weights
Erms = []
Acc_Test=[]
acc = []
for j in learningRate:                                                  # running the process for each of the different learning rate values
    for i in range(0,400):                                                  # Running a loop from 0 to 400
        # compute the weights using the SGD Weight equation
        #print ('---------Iteration: ' + str(i) + '--------------')
        Delta_E_D     = -np.dot((TrainingTarget[i] - np.dot(np.transpose(W_Now),TRAINING_PHI[i])),TRAINING_PHI[i]) # Delta_E_D values is the negative of the dot product of a computed value from a set of terms and the design matrix of the training set 
        La_Delta_E_W  = np.dot(La,W_Now)                                     # Dot product of lambda with the initial weight matrix
        Delta_E       = np.add(Delta_E_D,La_Delta_E_W)                       # Add the matrices Delta_E_D and La_Delta_E_W
        Delta_W       = -np.dot(j,Delta_E)                                   # Negative of the dot product between Learning rate and Delta_E; this is the Derivative term multiplied by the learning rate
        W_T_Next      = W_Now + Delta_W                                      # Weights are updated by adding the altered initial weights and Delta_W
        W_Now         = W_T_Next                                             # The updated weights are copied to the W_Now matrix
        
        #-----------------TrainingData Accuracy---------------------#
        TR_TEST_OUT   = GetValTest(TRAINING_PHI,W_T_Next)                    # Get training target (function call to GetValTest())
        Erms_TR       = GetErms(TR_TEST_OUT,TrainingTarget)                  # Get training E-RMS (Function call to GetErms())
        L_Erms_TR.append(float(Erms_TR.split(',')[1]))                       # Append the calculated Erms value to the list; Erms value is the second component of Erms_TR coming after the ','
        
        #-----------------ValidationData Accuracy---------------------#
        VAL_TEST_OUT  = GetValTest(VAL_PHI,W_T_Next)                         # Get validation target (function call to GetValTest())
        Erms_Val      = GetErms(VAL_TEST_OUT,ValDataAct)                     # Get Validation E-RMS (Function call to GetErms())
        L_Erms_Val.append(float(Erms_Val.split(',')[1]))                     # Append the calculated Erms value to the list; Erms value is the second component of Erms_Val coming after the ','
        
        #-----------------TestingData Accuracy---------------------#
        TEST_OUT      = GetValTest(TEST_PHI,W_T_Next)                        # Get Testing target (function call to GetValTest())
        Erms_Test = GetErms(TEST_OUT,TestDataAct)                            # Get testing E-RMS (Function call to GetErms())
        L_Erms_Test.append(float(Erms_Test.split(',')[1]))                   # Append the calculated Erms value to the list; Erms value is the second component of Erms_Test coming after the ',' 
        Acc_Test.append(float(Erms_Test.split(',')[0]))
    print ('----------Gradient Descent Solution--------------------')
    print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))           # Print Erms for training set; In SGD for each iteration E-RMS for the considered point is calculated. So the minimum of all E-Rms is considered.
    print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))          # Print Erms for validation set; Minimum of the validation E-Rms is considered
    print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),5)))         # Print Erms for testing set; Minimum of the testing E-rms is considered.
    Erms.append(min(L_Erms_Test))                                              # the minimum value of all the calculated e-rms values is appended to the list for graph construction
    acc.append(np.around(max(Acc_Test),5))                                     # the maximum value of all the calculated accuracy values is appended to the list for graph construction
    
# GRAPH FOR LEARNING RATE VS E-RMS
plt.plot(learningRate, Erms,'ro')
plt.ylabel('E-RMS Value for Testing')
plt.xlabel("Learning Rate")
plt.title("Learning Rate VS E-RMS Plot")
plt.show()

# GRAPH FOR LEARNING RATE VS TESTING ACCURACY
plt.plot(learningRate,acc,'ro')
plt.ylabel('Testing Accuracy of the model')
plt.xlabel("Learning Rate")
plt.title("Learning Rate VS Accuracy Plot")
plt.show()

# GRAPH FOR LAMBDA VS E-RMS; the values are calculated by running the program for the different values of lambda
plt.plot([0.05,1,2,5],[18.761496815300184,0.6236074980309467,0.6307095247656204,0.6242509239418752])
plt.ylabel('E-RMS Value for Testing')
plt.xlabel("Regularization term(SGD)")
plt.title("Regularization term VS E-RMS Plot")
plt.show()

#GRAPH FOR LAMBDA VS TESTING ACCURACY; the values are calculated by running the program for the different values of lambda
plt.plot([0.05,1,2,5],[2.672029880764258,70.4065507829335,70.27725901450941,70.30599051860365])
plt.ylabel('Testing ACcuracy for Testing')
plt.xlabel("Regularization term(SGD)")
plt.title("Regularization term VS Accuracy Plot")
plt.show()
