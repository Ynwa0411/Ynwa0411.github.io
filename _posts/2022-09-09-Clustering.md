1. KNN clustering


```python
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
iris = load_iris()
iris_data = pd.DataFrame({iris.feature_names[0]:iris.data[:,0],
                          iris.feature_names[1]:iris.data[:,1],
                          iris.feature_names[2]:iris.data[:,2],
                          iris.feature_names[3]:iris.data[:,3],
                          'target':iris.target})
X = iris.data
y = iris.target
print(X.shape,y.shape)
```

    (150, 4) (150,)
    


```python
setosa = iris_data[iris_data['target']==0]
versicolor = iris_data[iris_data['target']==1]
virginica = iris_data[iris_data['target']==2]

plt.scatter(list(setosa.iloc[:,0]),list(setosa.iloc[:,1]),label = 'setosa')
plt.scatter(list(versicolor.iloc[:,0]),list(versicolor.iloc[:,1]),label = 'versicolor')
plt.scatter(list(virginica.iloc[:,0]),list(virginica.iloc[:,1]),label = 'virginica')
plt.title('Sepal length and width')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.legend()
plt.show()
```


    
![png](output_2_0.png)
    



```python
plt.scatter(list(setosa.iloc[:,2]),list(setosa.iloc[:,3]),label = 'setosa')
plt.scatter(list(versicolor.iloc[:,2]),list(versicolor.iloc[:,3]),label = 'versicolor')
plt.scatter(list(virginica.iloc[:,2]),list(virginica.iloc[:,3]),label = 'virginica')
plt.title('Petal length and width')
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.legend()
plt.show()
```


    
![png](output_3_0.png)
    



```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
```

    (120, 4) (30, 4) (120,) (30,)
    


```python
from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier(n_neighbors = 5)
kn.fit(X_train,y_train)
kn.score(X_test, y_test)
```




    1.0




```python
setosa_test = X_test[y_test == 0]
versicolor_test = X_test[y_test == 1]
virginica_test = X_test[y_test == 2]

setosa_train = X_train[y_train == 0]
versicolor_train = X_train[y_train == 1]
virginica_train = X_train[y_train == 2]
```


```python
plt.scatter(setosa_test[:,0],setosa_test[:,1], label = '{}_test'.format(iris.target_names[0]), marker = 'X', s=200, alpha = 0.5, color = 'red')
plt.scatter(versicolor_test[:,0],versicolor_test[:,1], label = '{}_test'.format(iris.target_names[1]), marker = 'X', s=200, alpha = 0.5, color = 'blue')
plt.scatter(virginica_test[:,0],virginica_test[:,1], label = '{}_test'.format(iris.target_names[2]), marker = 'X', s=200, alpha = 0.5, color = 'green')

plt.scatter(setosa_train[:,0],setosa_train[:,1], label = iris.target_names[0], color = 'red')
plt.scatter(versicolor_train[:,0],versicolor_train[:,1], label = iris.target_names[1], color = 'blue')
plt.scatter(virginica_train[:,0],virginica_train[:,1], label = iris.target_names[2], color = 'green')

plt.title('Sepal length and width (k = 5)')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.legend()
plt.show()
```


    
![png](output_7_0.png)
    



```python
plt.scatter(setosa_test[:,2],setosa_test[:,3], label = '{}_test'.format(iris.target_names[0]), marker = 'X', s=200, alpha = 0.5, color = 'red')
plt.scatter(versicolor_test[:,2],versicolor_test[:,3], label = '{}_test'.format(iris.target_names[1]), marker = 'X', s=200, alpha = 0.5, color = 'blue')
plt.scatter(virginica_test[:,2],virginica_test[:,3], label = '{}_test'.format(iris.target_names[2]), marker = 'X', s=200, alpha = 0.5, color = 'green')

plt.scatter(setosa_train[:,2],setosa_train[:,3], label = iris.target_names[0], color = 'red')
plt.scatter(versicolor_train[:,2],versicolor_train[:,3], label = iris.target_names[1], color = 'blue')
plt.scatter(virginica_train[:,2],virginica_train[:,3], label = iris.target_names[2], color = 'green')

plt.title('Petal length and width (k = 5)')
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.legend()
plt.show()
```


    
![png](output_8_0.png)
    


2. KMeans Clustering using while


```python
# while 문 사용 #
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

km = pd.read_csv(r'KMeansData.csv')
kmdata = km.iloc[:,:]
X = sc.fit_transform(kmdata)
x = X[:,0].reshape(-1,1)
y = X[:,1].reshape(-1,1)
x.shape
```




    (100, 1)




```python
x_mean = np.mean(x)
x_std = np.std(x)
y_mean = np.mean(y)
y_std = np.std(y)

z0 = np.array([[x_mean+x_std, y_mean-y_std]]).reshape(1,2)
z1 = np.array([[x_mean-x_std, y_mean+y_std]]).reshape(1,2)
z = np.r_[z0,z1]
```


```python
def distance(a,b):
  return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
```


```python
k = np.zeros(len(x))
for i in range(len(x)):
  kd0 = distance([x[i],y[i]], [z0[0,0], z0[0,1]])
  kd1 = distance([x[i],y[i]], [z1[0,0], z1[0,1]])
  if kd0 < kd1:
    k[i] = 0.0
  else:
    k[i] = 1.0
```


```python
x0 = x[k==0]
x1 = x[k==1]
y0 = y[k==0]
y1 = y[k==1]

z0 = np.array([[np.mean(x0), np.mean(y0)]]).reshape(1,2)
z1 = np.array([[np.mean(x1), np.mean(y1)]]).reshape(1,2)
z = np.r_[z0,z1]
```


```python
j = 0
while True:
  k_old = np.copy(k)
  z_old = np.copy(z)
  for i in range(len(x)):
    kd0 = distance([x[i],y[i]], [z0[0,0], z0[0,1]])
    kd1 = distance([x[i],y[i]], [z1[0,0], z1[0,1]])
    if kd0 < kd1:
      k[i] = 0.0
    else:
      k[i] = 1.0

  if np.array_equal(k_old, k) == True:
    break

  x0 = x[np.where(k==0)]
  x1 = x[np.where(k==1)]
  y0 = y[np.where(k==0)]
  y1 = y[np.where(k==1)]

  z0 = np.array([[np.mean(x0), np.mean(y0)]]).reshape(1,2)
  z1 = np.array([[np.mean(x1), np.mean(y1)]]).reshape(1,2)
  z = np.r_[z0,z1]
  j += 1
```


```python
j
```




    6




```python
plt.scatter(x0,y0,color = 'red', label = 0)
plt.scatter(x1,y1,color = 'blue', label = 1)
plt.scatter(z[:,0],z[:,1],marker = 'X',s=200, label = 'centroid')
plt.legend()
plt.title("KMeans clustering")
plt.show()
print("Centroid of 0 is {}, {}".format(z[0,0], z[0,1]))
print("Centroid of 1 is {}, {}".format(z[1,0], z[1,1]))
```


    
![png](output_17_0.png)
    


    Centroid of 0 is 0.8963445287159334, -0.15741875802609
    Centroid of 1 is -0.9329300196839314, 0.16384401345572663
    

3. KMeans Clustering using sklearn


```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 2)
y_predict = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_
plt.scatter(x[y_predict == 0], y[y_predict == 0],color = 'red', label = 0)
plt.scatter(x[y_predict == 1], y[y_predict == 1],color = 'blue', label = 1)
plt.scatter(centers[:,0],centers[:,1],marker = 'X',s = 200, label = 'centroid')
plt.legend()
plt.title("KMeans clustering using sklearn")
plt.show()
print("Centroid of 0 is {}, {}".format(centers[0,0], centers[0,1]))
print("Centroid of 1 is {}, {}".format(centers[1,0], centers[1,1]))
```


    
![png](output_19_0.png)
    


    Centroid of 0 is -0.9329300196839315, 0.16384401345572663
    Centroid of 1 is 0.8963445287159334, -0.15741875802609004
    

4. GMM clustering


```python
N = 50
np.random.seed(1)
d1 = np.random.multivariate_normal(mean = [2,1], cov = [[1.2,1],[1,1.2]], size = N)
d2 = np.random.multivariate_normal(mean = [3,6], cov = [[1.2,1],[1,1.2]], size = N)

plt.scatter(d1[:,0], d1[:,1])
plt.scatter(d2[:,0], d2[:,1])
plt.show()
```


    
![png](output_21_0.png)
    



```python
X = np.r_[d1, d2]
x = X[:,0]
y = X[:,1]
numK = 2 # clustering 개수
```


```python
mx = np.mean(x)
sx = np.std(x)
my = np.mean(y)
sy = np.std(y)

u0 = np.array([mx+sx,my+sy])
u1 = np.array([mx-sx,my-sy])
sigma0 = np.array([[sx*sx/4,0],[0,sy*sy/4]])
sigma1 = np.array([[sx*sx/4,0],[0,sy*sy/4]])

# Responsibility
R = np.ones([len(X),numK]) * 1/numK
pi = np.ones(numK) * 1/numK
```


```python
import scipy as sp
from scipy.stats import multivariate_normal
N0 = sp.stats.multivariate_normal.pdf(X, mean=u0, cov=sigma0)
N1 = sp.stats.multivariate_normal.pdf(X, mean=u1, cov=sigma1)

plt.figure(figsize = (10,5))
plt.subplot(1,2,1)
plt.bar(np.linspace(0,len(X),len(X)),N0)
plt.subplot(1,2,2)
plt.bar(np.linspace(0,len(X),len(X)),N1)
plt.show()
```


    
![png](output_24_0.png)
    



```python
R = np.c_[(pi[0]*N0 / (pi[0]*N0+pi[1]*N1)),(pi[1]*N1 / (pi[0]*N0+pi[1]*N1))]
pi = np.ones(len(X)).reshape(1,-1).dot(R)/ len(X)
```


```python
k = np.ones(len(X))
for i in range(len(X)):
  if R[i,0] < R[i,1]:
    k[i] = 0
  else:
    k[i] = 1
k
```




    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 0., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 0.,
           1., 0., 1., 0., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1.])




```python
j = 0
while True:
  u0 = (R.T.dot(X))[0,:]/np.sum(R[:,0])
  u1 = (R.T.dot(X))[1,:]/np.sum(R[:,1])
  # u0 = X.T.dot(R)[:,0] / sum(R[:,0])
  # u1 = X.T.dot(R)[:,1] / sum(R[:,1])

  sigma0 = X.T.dot(np.multiply(R[:,0].reshape(-1,1),X))/np.sum(R[:,0]) - u0.reshape(-1,1).dot(u0.reshape(-1,1).T)
  sigma1 = X.T.dot(np.multiply(R[:,1].reshape(-1,1),X))/np.sum(R[:,1]) - u1.reshape(-1,1).dot(u1.reshape(-1,1).T)

  N0 = multivariate_normal.pdf(X, mean=u0, cov=sigma0)
  N1 = multivariate_normal.pdf(X, mean=u1, cov=sigma1)

  R = np.c_[(pi[0][0]*N0 / (pi[0][0]*N0+pi[0][1]*N1)),(pi[0][1]*N1 / (pi[0][0]*N0+pi[0][1]*N1))]
  pi = np.ones(len(X)).reshape(1,-1).dot(R)/ len(X)

  k_old = np.copy(k)
  for i in range(len(X)):
    if R[i,0] < R[i,1]:
      k[i] = 0
    else:
      k[i] = 1

  if np.array_equal(k_old, k) == True:
    break
  j += 1
```


```python
j
```




    4




```python
plt.figure(figsize = (15,5))
plt.subplot(1,2,1)
plt.title("GMM clustering")
plt.scatter(X[k==0][:,0],X[k==0][:,1],color = 'red',label = 0)
plt.scatter(X[k==1][:,0],X[k==1][:,1],color = 'blue',label = 1)
plt.legend()

plt.subplot(1,2,2)
plt.title("Sample")
plt.scatter(d1[:,0], d1[:,1], label = 0)
plt.scatter(d2[:,0], d2[:,1], label = 1)
plt.legend()
plt.show()

print("centroid of 0 is {}, {}".format(np.mean(X[k==0][:,0]),np.mean(X[k==0][:,1])))
print("centroid of 1 is {}, {}".format(np.mean(X[k==1][:,0]),np.mean(X[k==1][:,1])))

print("centroid of sample 0 is {}, {}".format(np.mean(d1[:,0]), np.mean(d1[:,1])))
print("centroid of sample 1 is {}, {}".format(np.mean(d2[:,0]), np.mean(d2[:,1])))
```


    
![png](output_29_0.png)
    


    centroid of 0 is 1.860807263241245, 0.8503497440085822
    centroid of 1 is 2.672153892430456, 5.665816156465964
    centroid of sample 0 is 1.860807263241245, 0.8503497440085822
    centroid of sample 1 is 2.672153892430456, 5.665816156465964
    
