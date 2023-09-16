# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required library and read the dataframe.
2. Write a function computeCost to generate the cost function.
3. Perform iterations og gradient steps with learning rate.
4. Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: S.Shanmathi
RegisterNumber:  212222100049
*/
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv("/content/ex1.txt", header = None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
  m=len(y)
  h=X.dot(theta)
  square_err=(h-y)**2
  return 1/(2*m)*np.sum(square_err)

data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(X,y,theta)

def gradientDescent(X,y,theta,alpha,num_iters):
  m=len(y)
  J_history=[]
  for i in range(num_iters):
    predictions=X.dot(theta)
    error=np.dot(X.transpose(),(predictions-y))
    descent=alpha* 1/m * error
    theta-=descent
    J_history.append(computeCost(X,y,theta))
  return theta,J_history

theta,J_history=gradientDescent(X,y,theta,0.01,1500)
print("h(X)="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit($10,000)")
plt.title("Profit Prediction")

def predict(x,theta):
  predictions=np.dot(theta.transpose(),x)
  return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000 ,we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000,we predict a profit of $"+str(round(predict2,0)))
```
## Output:
#### Profit Prediction Graph
![Screenshot 2023-09-16 201723](https://github.com/ShanmathiShanmugam/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121243595/bdfed25f-f9de-4610-b416-91d57108ca01)
![Screenshot 2023-09-16 201958](https://github.com/ShanmathiShanmugam/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121243595/1e13149e-9075-4760-ae5b-222d826f8e2f)

#### Compute Cost Value

![Screenshot 2023-09-16 201911](https://github.com/ShanmathiShanmugam/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121243595/c5b67543-f5f4-4b8d-81cb-3c8d9b004ed1)

#### h(x) Value

![Screenshot 2023-09-16 201924](https://github.com/ShanmathiShanmugam/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121243595/4df03a5c-f0cd-46e5-9972-a3828b17db36)

#### Cost function using Gradient Descent Graph

![Screenshot 2023-09-16 201941](https://github.com/ShanmathiShanmugam/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121243595/7413a196-1aa7-42b2-b44f-5f8660011a33)

#### Profit for the Population 35,000

![Screenshot 2023-09-16 202019](https://github.com/ShanmathiShanmugam/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121243595/46ee0673-8dff-40fb-9509-eba77f5bd6cd)

#### Profit for the Population 70,000

![Screenshot 2023-09-16 202028](https://github.com/ShanmathiShanmugam/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121243595/ba7a740c-90e9-4460-ac73-eee452c92f0b)
```


```
## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
