Linear Regression using Tensorflow


```python
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler
tf.__version__
```




    '2.10.0'




```python
dfLoad = pd.read_csv("https://raw.githubusercontent.com/hanwoolJeong/lectureUniv/main/testData_H_vs_W.txt", sep ="\s+")
HeightRaw = np.array(dfLoad["Height"]).reshape(-1,1)
WeightRaw = np.array(dfLoad["Weight"]).reshape(-1,1)
sc1 = StandardScaler()
df_std = sc1.fit_transform(dfLoad)
Height_std = np.array(df_std[:,0]).reshape(-1,1)
Weight_std = np.array(df_std[:,1]).reshape(-1,1)

X_np_std = np.c_[np.ones(len(dfLoad)), Height_std]
y_np_std = Weight_std
```


```python
X = tf.constant(X_np_std, dtype = tf.float32)
y = tf.constant(y_np_std, dtype = tf.float32)
theta = tf.Variable(tf.random.uniform([2,1],-1,1, dtype = tf.float32))

learning_rate = 0.1
n_epoch = 100
for i in range(n_epoch):
  with tf.GradientTape() as g:
    y_pred = tf.matmul(X, theta)
    error = y - y_pred
    mse = tf.reduce_mean(tf.square(error))
    gradients = g.gradient(mse,[theta])
    theta.assign(theta - learning_rate * gradients[0].numpy())
```


```python
w0_std2 = theta.numpy()[0]
w1_std2 = theta.numpy()[1]

# w0_std, w1_std 를 unfit_transform #
w1 = w1_std2 * np.sqrt(sc1.var_[1] / sc1.var_[0])
w0 = w0_std2 * np.sqrt(sc1.var_[1]) - sc1.mean_[0] * w1_std2 * np.sqrt(sc1.var_[1] / sc1.var_[0]) + sc1.mean_[1]

# f1, ax1 = plt.subplots(figsize = (6,6))
ax1 = plt.subplot(1,2,1)
X_std_plt = np.linspace(min(Height_std), max(Height_std), 100)
ax1.plot(X_std_plt, w1_std2*X_std_plt+w0_std2,'g:')
ax1.scatter(Height_std, Weight_std, s = 20)
ax1.set(xlabel = 'Height (Scaled)', ylabel = 'Weight(Scaled)')
ax1.set_title('Height and Weight (Scaled)')

# f2, ax2 = plt.subplots(figsize = (6,6))
ax2 = plt.subplot(1,2,2)
X_plt = np.linspace(min(HeightRaw), max(HeightRaw), 100)
ax2.plot(X_plt, w1*X_plt+w0, 'g:')
ax2.scatter(HeightRaw, WeightRaw, s = 20)
ax2.set(xlabel = 'Height', ylabel = 'Weight')
ax2.set_title('Height and Weight (Unscaled)')
plt.show()
```


    
![png](output_4_0.png)
    



    
![png](output_4_1.png)
    


Deep MLP using Tensorflow


```python
import matplotlib.cm as cm
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

plt.imshow(X_train[0], cmap = cm.gray)
plt.colorbar()
plt.show()
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
    11490434/11490434 [==============================] - 1s 0us/step
    


    
![png](output_6_1.png)
    



```python
# Normalize X, One-hot encode y #
X_train_std, X_test_std = X_train / 255.0, X_test / 255.0
X_train_std = X_train_std.astype('float32')
X_test_std = X_test_std.astype('float32')
y_train_onehot = tf.keras.utils.to_categorical(y_train, 10)
y_test_onehot = tf.keras.utils.to_categorical(y_test, 10)
```


```python
def neuron_layer(X, W, b, activation=None):
  z = tf.matmul(X, W)+b
  if activation is None:
    return z
  else:
    return activation(z)

# layer : 3 , cross entropy loss function
def MLP(X_flatten, W, b):
  hidden1 = neuron_layer(X_flatten,W[0], b[0], activation=tf.nn.sigmoid)
  hidden2 = neuron_layer(hidden1, W[1], b[1], activation=tf.nn.sigmoid)
  logits = neuron_layer(hidden2, W[2], b[2], activation=None)
  y_pred = tf.nn.softmax(logits)
  return y_pred

# layer : 2 , MSE
def MLP_MSE(X_flatten, W, b):
  hidden = neuron_layer(X_flatten,W[0], b[0], activation=tf.nn.sigmoid)
  logits = neuron_layer(hidden, W[1], b[1], activation=None)
  y_pred = tf.nn.softmax(logits)
  return y_pred
```


```python
# 크로스엔트로피 loss function, layer : 3 #

n_inputs = np.array([28**2, 256, 128])
n_nodes = np.array([256, 128, 10])
n_layer = 3
W, b = {},{}
for layer in range(n_layer):
  stddev = 2 / np.sqrt(n_inputs[layer]+n_nodes[layer])
  W_init = tf.random.truncated_normal((n_inputs[layer],n_nodes[layer]),stddev = stddev)
  W[layer] = tf.Variable(W_init)
  b[layer] = tf.Variable(tf.zeros([n_nodes[layer]]))
```


```python
n_epoch = 40
batchSize = 200
nTrain = len(X_train)
nBatch = int(nTrain/batchSize)

# SGD Optimizer #
opt = tf.keras.optimizers.SGD(learning_rate = 0.01)

# W, b 업데이트
for epoch in range(n_epoch):
  idxShuffle = np.random.permutation(X_train.shape[0]) # 섞어주기
  for idxSet in range(nBatch):
    X_batch = X_train_std[idxShuffle[idxSet*batchSize:(idxSet+1)*batchSize], :] # X_train_std 에서 
    X_batch_tensor = tf.convert_to_tensor(X_batch.reshape(-1, 28*28))
    y_batch = y_train_onehot[idxShuffle[idxSet*batchSize:(idxSet+1)*batchSize], :]
    with tf.GradientTape() as tape:
      y_pred = MLP(X_batch_tensor, W, b)
      loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_batch, y_pred)) # loss function : CrossEntropy
    gradients = tape.gradient(loss, [W[2],W[1],W[0],b[2],b[1],b[0]]) # gradient 구하기
    opt.apply_gradients(zip(gradients, [W[2],W[1],W[0],b[2],b[1],b[0]])) # 반영해서 update
  print('epoch : {}'.format(epoch))

  if epoch % 5 ==0:
     correct = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_batch, 1))
     accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)).numpy()
     print("epoch : {}, accuarcy : {}".format(epoch, accuracy))
```

    epoch : 0
    0.285
    epoch : 1
    epoch : 2
    epoch : 3
    epoch : 4
    epoch : 5
    0.71
    epoch : 6
    epoch : 7
    epoch : 8
    epoch : 9
    epoch : 10
    0.83
    epoch : 11
    epoch : 12
    epoch : 13
    epoch : 14
    epoch : 15
    0.825
    epoch : 16
    epoch : 17
    epoch : 18
    epoch : 19
    epoch : 20
    0.825
    epoch : 21
    epoch : 22
    epoch : 23
    epoch : 24
    epoch : 25
    0.89
    epoch : 26
    epoch : 27
    epoch : 28
    epoch : 29
    epoch : 30
    0.85
    epoch : 31
    epoch : 32
    epoch : 33
    epoch : 34
    epoch : 35
    0.885
    epoch : 36
    epoch : 37
    epoch : 38
    epoch : 39
    


```python
success = 0
failure = 0
prediction = MLP(X_test_std.reshape(-1,28*28), W, b)
for i in range(len(y_test_onehot)):
  y_pred = np.argmax(prediction[i])
  if y_pred == np.argmax(y_test_onehot[i]):
    success += 1
  else:
    failure += 1
print("Prediction score is {}".format(success / (success + failure)))
```

    0.8973
    


```python
# MSE loss function, layer : 2 #

n_inputs = np.array([28*28, 128])
n_nodes = np.array([128, 10])
n_layer = 2
W, b = {}, {}
for layer in range(n_layer):
  stddev = 2 / np.sqrt(n_inputs[layer] + n_nodes[layer])
  W_init = tf.random.truncated_normal((n_inputs[layer], n_nodes[layer]), stddev = stddev)
  W[layer] = tf.Variable(W_init)
  b[layer] = tf.Variable(tf.zeros([n_nodes[layer]]))
```




    (784, 128)




```python
# MSE loss function, layer : 2 #
# layer가 3개로 늘어나면 Vanishing gradient problem 발생 (기울기 0에 근사) #

n_epoch = 40
batchSize = 200
nBatch = int(nTrain/batchSize)
opt = tf.keras.optimizers.SGD(learning_rate=0.1)
for epoch in range(n_epoch):
  idxShuffle = np.random.permutation(X_train.shape[0])
  for idxSet in range(nBatch):
    X_batch = X_train_std[idxShuffle[idxSet*batchSize:(idxSet+1)*batchSize], :]
    X_batch_tensor = tf.convert_to_tensor(X_batch.reshape(-1, 28*28))
    y_batch = y_train_onehot[idxShuffle[idxSet*batchSize:(idxSet+1)*batchSize], :]
    with tf.GradientTape() as tape:
      y_pred = MLP_MSE(X_batch_tensor, W, b)
      loss = tf.reduce_mean(tf.keras.losses.MSE(y_batch, y_pred))
    gradients = tape.gradient(loss, [W[1], W[0], b[1], b[0]])
    opt.apply_gradients(zip(gradients, [W[1], W[0], b[1], b[0]]))
  
  if epoch % 5 ==0:
    correct = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_batch, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)).numpy()
    print("epoch : {}, accuarcy : {}".format(epoch, accuracy))
```

    epoch : 0, accuarcy : 0.9300000071525574
    epoch : 5, accuarcy : 0.8799999952316284
    epoch : 10, accuarcy : 0.9049999713897705
    epoch : 15, accuarcy : 0.8650000095367432
    epoch : 20, accuarcy : 0.9350000023841858
    epoch : 25, accuarcy : 0.8999999761581421
    epoch : 30, accuarcy : 0.9100000262260437
    epoch : 35, accuarcy : 0.9049999713897705
    


```python
success = 0
failure = 0
prediction = MLP_MSE(X_test_std.reshape(-1,28*28), W, b)
for i in range(len(y_test_onehot)):
  y_pred = np.argmax(prediction[i])
  if y_pred == np.argmax(y_test_onehot[i]):
    success += 1
  else:
    failure += 1
print("Prediction score is {}".format(success / (success + failure)))
```

    0.9077
    

Deep MLP using Tensorflow in simpler way


```python
# SGD Optimizer, MSE loss function

layers = [tf.keras.layers.Flatten(input_shape = (28,28)),
          tf.keras.layers.Dense(256, activation = tf.nn.relu),
          tf.keras.layers.Dense(128, activation = tf.nn.relu),
          tf.keras.layers.Dense(10, activation = tf.nn.softmax)]

myMLP = tf.keras.Sequential(layers)
myMLP.compile(optimizer = 'SGD', loss = 'MSE',
              metrics = ['accuracy'])

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train_std, X_test_std = X_train / 255.0, X_test / 255.0
y_train_onehot = tf.keras.utils.to_categorical(y_train, 10)
y_test_onehot = tf.keras.utils.to_categorical(y_test, 10)

history = myMLP.fit(X_train_std, y_train_onehot, epochs = 40, batch_size = 200)
```

    Epoch 1/40
    300/300 [==============================] - 3s 6ms/step - loss: 0.0908 - accuracy: 0.1007
    Epoch 2/40
    300/300 [==============================] - 2s 7ms/step - loss: 0.0897 - accuracy: 0.1489
    Epoch 3/40
    300/300 [==============================] - 3s 9ms/step - loss: 0.0886 - accuracy: 0.2048
    Epoch 4/40
    300/300 [==============================] - 3s 9ms/step - loss: 0.0875 - accuracy: 0.2558
    Epoch 5/40
    300/300 [==============================] - 3s 10ms/step - loss: 0.0863 - accuracy: 0.3118
    Epoch 6/40
    300/300 [==============================] - 3s 9ms/step - loss: 0.0850 - accuracy: 0.3719
    Epoch 7/40
    300/300 [==============================] - 3s 9ms/step - loss: 0.0834 - accuracy: 0.4242
    Epoch 8/40
    300/300 [==============================] - 3s 9ms/step - loss: 0.0817 - accuracy: 0.4575
    Epoch 9/40
    300/300 [==============================] - 3s 9ms/step - loss: 0.0797 - accuracy: 0.4797
    Epoch 10/40
    300/300 [==============================] - 3s 10ms/step - loss: 0.0775 - accuracy: 0.4992
    Epoch 11/40
    300/300 [==============================] - 3s 9ms/step - loss: 0.0752 - accuracy: 0.5203
    Epoch 12/40
    300/300 [==============================] - 3s 9ms/step - loss: 0.0729 - accuracy: 0.5447
    Epoch 13/40
    300/300 [==============================] - 3s 9ms/step - loss: 0.0704 - accuracy: 0.5680
    Epoch 14/40
    300/300 [==============================] - 2s 6ms/step - loss: 0.0679 - accuracy: 0.5877
    Epoch 15/40
    300/300 [==============================] - 2s 6ms/step - loss: 0.0652 - accuracy: 0.6042
    Epoch 16/40
    300/300 [==============================] - 2s 5ms/step - loss: 0.0626 - accuracy: 0.6166
    Epoch 17/40
    300/300 [==============================] - 2s 6ms/step - loss: 0.0600 - accuracy: 0.6274
    Epoch 18/40
    300/300 [==============================] - 2s 6ms/step - loss: 0.0576 - accuracy: 0.6381
    Epoch 19/40
    300/300 [==============================] - 2s 5ms/step - loss: 0.0552 - accuracy: 0.6492
    Epoch 20/40
    300/300 [==============================] - 2s 6ms/step - loss: 0.0529 - accuracy: 0.6627
    Epoch 21/40
    300/300 [==============================] - 2s 5ms/step - loss: 0.0507 - accuracy: 0.6799
    Epoch 22/40
    300/300 [==============================] - 2s 6ms/step - loss: 0.0487 - accuracy: 0.6992
    Epoch 23/40
    300/300 [==============================] - 2s 5ms/step - loss: 0.0467 - accuracy: 0.7190
    Epoch 24/40
    300/300 [==============================] - 2s 7ms/step - loss: 0.0447 - accuracy: 0.7403
    Epoch 25/40
    300/300 [==============================] - 2s 8ms/step - loss: 0.0429 - accuracy: 0.7573
    Epoch 26/40
    300/300 [==============================] - 3s 10ms/step - loss: 0.0412 - accuracy: 0.7747
    Epoch 27/40
    300/300 [==============================] - 3s 10ms/step - loss: 0.0396 - accuracy: 0.7908
    Epoch 28/40
    300/300 [==============================] - 2s 7ms/step - loss: 0.0380 - accuracy: 0.8036
    Epoch 29/40
    300/300 [==============================] - 2s 7ms/step - loss: 0.0366 - accuracy: 0.8158
    Epoch 30/40
    300/300 [==============================] - 2s 6ms/step - loss: 0.0352 - accuracy: 0.8246
    Epoch 31/40
    300/300 [==============================] - 2s 8ms/step - loss: 0.0339 - accuracy: 0.8317
    Epoch 32/40
    300/300 [==============================] - 2s 6ms/step - loss: 0.0328 - accuracy: 0.8367
    Epoch 33/40
    300/300 [==============================] - 2s 7ms/step - loss: 0.0317 - accuracy: 0.8413
    Epoch 34/40
    300/300 [==============================] - 2s 6ms/step - loss: 0.0307 - accuracy: 0.8451
    Epoch 35/40
    300/300 [==============================] - 2s 7ms/step - loss: 0.0298 - accuracy: 0.8483
    Epoch 36/40
    300/300 [==============================] - 2s 6ms/step - loss: 0.0289 - accuracy: 0.8514
    Epoch 37/40
    300/300 [==============================] - 3s 9ms/step - loss: 0.0282 - accuracy: 0.8540
    Epoch 38/40
    300/300 [==============================] - 2s 8ms/step - loss: 0.0275 - accuracy: 0.8569
    Epoch 39/40
    300/300 [==============================] - 3s 10ms/step - loss: 0.0268 - accuracy: 0.8590
    Epoch 40/40
    300/300 [==============================] - 4s 12ms/step - loss: 0.0262 - accuracy: 0.8608
    


```python
test_loss, test_acc = myMLP.evaluate(X_test_std, y_test_onehot)
test_loss, test_acc
```

    313/313 [==============================] - 2s 6ms/step - loss: 0.0249 - accuracy: 0.8691
    




    (0.02494288980960846, 0.8690999746322632)




```python
# Optimizer Adam, Cross Entropy loss function

layers = [tf.keras.layers.Flatten(input_shape = (28,28)),
          tf.keras.layers.Dense(256, activation = tf.nn.relu),
          tf.keras.layers.Dense(128, activation = tf.nn.relu),
          tf.keras.layers.Dense(10, activation = tf.nn.softmax)]

myMLP_adam = tf.keras.Sequential(layers)
myMLP_adam.compile(optimizer = 'Adam', loss = 'CategoricalCrossentropy',
              metrics = ['accuracy'])

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train_std, X_test_std = X_train / 255.0, X_test / 255.0
y_train_onehot = tf.keras.utils.to_categorical(y_train, 10)
y_test_onehot = tf.keras.utils.to_categorical(y_test, 10)

history_adam = myMLP_adam.fit(X_train_std, y_train_onehot, epochs = 40, batch_size = 200)
```

    Epoch 1/40
    300/300 [==============================] - 4s 8ms/step - loss: 0.3042 - accuracy: 0.9135
    Epoch 2/40
    300/300 [==============================] - 2s 7ms/step - loss: 0.1164 - accuracy: 0.9660
    Epoch 3/40
    300/300 [==============================] - 2s 8ms/step - loss: 0.0764 - accuracy: 0.9770
    Epoch 4/40
    300/300 [==============================] - 2s 6ms/step - loss: 0.0549 - accuracy: 0.9837
    Epoch 5/40
    300/300 [==============================] - 2s 7ms/step - loss: 0.0415 - accuracy: 0.9873
    Epoch 6/40
    300/300 [==============================] - 2s 7ms/step - loss: 0.0309 - accuracy: 0.9904
    Epoch 7/40
    300/300 [==============================] - 2s 8ms/step - loss: 0.0228 - accuracy: 0.9933
    Epoch 8/40
    300/300 [==============================] - 2s 7ms/step - loss: 0.0186 - accuracy: 0.9944
    Epoch 9/40
    300/300 [==============================] - 2s 6ms/step - loss: 0.0159 - accuracy: 0.9948
    Epoch 10/40
    300/300 [==============================] - 2s 7ms/step - loss: 0.0124 - accuracy: 0.9964
    Epoch 11/40
    300/300 [==============================] - 2s 6ms/step - loss: 0.0082 - accuracy: 0.9978
    Epoch 12/40
    300/300 [==============================] - 2s 6ms/step - loss: 0.0095 - accuracy: 0.9971
    Epoch 13/40
    300/300 [==============================] - 2s 7ms/step - loss: 0.0104 - accuracy: 0.9966
    Epoch 14/40
    300/300 [==============================] - 2s 6ms/step - loss: 0.0114 - accuracy: 0.9962
    Epoch 15/40
    300/300 [==============================] - 2s 7ms/step - loss: 0.0078 - accuracy: 0.9974
    Epoch 16/40
    300/300 [==============================] - 3s 10ms/step - loss: 0.0042 - accuracy: 0.9989
    Epoch 17/40
    300/300 [==============================] - 3s 9ms/step - loss: 0.0081 - accuracy: 0.9976
    Epoch 18/40
    300/300 [==============================] - 3s 9ms/step - loss: 0.0074 - accuracy: 0.9977
    Epoch 19/40
    300/300 [==============================] - 3s 10ms/step - loss: 0.0089 - accuracy: 0.9970
    Epoch 20/40
    300/300 [==============================] - 2s 6ms/step - loss: 0.0050 - accuracy: 0.9983
    Epoch 21/40
    300/300 [==============================] - 2s 6ms/step - loss: 0.0053 - accuracy: 0.9983
    Epoch 22/40
    300/300 [==============================] - 2s 6ms/step - loss: 0.0017 - accuracy: 0.9996
    Epoch 23/40
    300/300 [==============================] - 2s 7ms/step - loss: 3.1403e-04 - accuracy: 1.0000
    Epoch 24/40
    300/300 [==============================] - 2s 7ms/step - loss: 1.2508e-04 - accuracy: 1.0000
    Epoch 25/40
    300/300 [==============================] - 2s 6ms/step - loss: 8.7854e-05 - accuracy: 1.0000
    Epoch 26/40
    300/300 [==============================] - 2s 6ms/step - loss: 6.4802e-05 - accuracy: 1.0000
    Epoch 27/40
    300/300 [==============================] - 2s 6ms/step - loss: 5.5532e-05 - accuracy: 1.0000
    Epoch 28/40
    300/300 [==============================] - 2s 7ms/step - loss: 4.8006e-05 - accuracy: 1.0000
    Epoch 29/40
    300/300 [==============================] - 2s 6ms/step - loss: 4.2016e-05 - accuracy: 1.0000
    Epoch 30/40
    300/300 [==============================] - 2s 6ms/step - loss: 3.7059e-05 - accuracy: 1.0000
    Epoch 31/40
    300/300 [==============================] - 2s 5ms/step - loss: 3.2533e-05 - accuracy: 1.0000
    Epoch 32/40
    300/300 [==============================] - 2s 8ms/step - loss: 2.8932e-05 - accuracy: 1.0000
    Epoch 33/40
    300/300 [==============================] - 2s 6ms/step - loss: 2.5596e-05 - accuracy: 1.0000
    Epoch 34/40
    300/300 [==============================] - 2s 6ms/step - loss: 2.2819e-05 - accuracy: 1.0000
    Epoch 35/40
    300/300 [==============================] - 2s 6ms/step - loss: 1.9806e-05 - accuracy: 1.0000
    Epoch 36/40
    300/300 [==============================] - 2s 7ms/step - loss: 1.7765e-05 - accuracy: 1.0000
    Epoch 37/40
    300/300 [==============================] - 2s 6ms/step - loss: 1.5810e-05 - accuracy: 1.0000
    Epoch 38/40
    300/300 [==============================] - 2s 7ms/step - loss: 1.3998e-05 - accuracy: 1.0000
    Epoch 39/40
    300/300 [==============================] - 2s 6ms/step - loss: 1.2755e-05 - accuracy: 1.0000
    Epoch 40/40
    300/300 [==============================] - 2s 7ms/step - loss: 1.0799e-05 - accuracy: 1.0000
    


```python
test_loss, test_acc = myMLP_adam.evaluate(X_test_std, y_test_onehot)
test_loss, test_acc
```

    313/313 [==============================] - 1s 3ms/step - loss: 0.0936 - accuracy: 0.9835
    




    (0.09355340898036957, 0.9835000038146973)




```python
plt.plot(history.history['accuracy'], label = 'SGD / MSE')
plt.plot(history_adam.history['accuracy'], label = 'Adam / Cross Entropy')
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```


    
![png](output_20_0.png)
    


Deep MLP using GDM, Sigmoid


```python
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train.shape, y_train.shape
```




    ((60000, 28, 28), (60000,))




```python
class Perceptron:
  def __init__(self, input_dim = 28**2, hidden_dim = 100, output_dim = 10, lr):
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim
    self.w1 = np.random.normal(0, hidden_dim**(-0.5), (input_dim,hidden_dim))
    self.w2 = np.random.normal(0, hidden_dim**(-0.5), (hidden_dim,output_dim))
    self.h = np.zeros([hidden_dim, 1])
    self.theta = 0
    self.lr = lr    

  def sigmoid(self, x):
    return 1 / (1+np.exp(-x))
  
  def feedforward_1(self,x):
    return self.sigmoid(x.dot(self.w1)-self.theta)

  def feedforward_2(self, x):
    self.h = self.sigmoid(x.dot(self.w1) - self.theta)
    return self.sigmoid(self.h.dot(self.w2) - self.theta)

  def backprop_w2(self, g, y):
    q = -2*(g-y)*y*(1-y)
    return self.h.T.dot(q)

  def backprop_w1(self, g, y, x):
    q1 = -2*(g-y)*y*(1-y)
    q2 = q1.dot(self.w2.T)
    return x.T.dot(q2*self.h*(1-self.h))

  def training(self, input, target):
    g = target
    x = input
    y = self.feedforward_2(input)
    self.w2 = self.w2 - self.lr*self.backprop_w2(g,y)
    self.w1 = self.w1 - self.lr*self.backprop_w1(g,y,x)
```


```python
y_train_onehot = tf.keras.utils.to_categorical(y_train, 10)
y_test_onehot = tf.keras.utils.to_categorical(y_test, 10)

X_train_reshape = X_train.reshape(-1,28*28)
X_test_reshape = X_test.reshape(-1,28*28)

X_train_std = X_train_reshape / 255.0
X_test_std = X_test_reshape / 255.0
X_train_std.shape, y_train_onehot.shape
```




    ((60000, 784), (60000, 10))




```python
# Test score by lr #
n_epoch = 40
batchSize = 200
nTrain = len(X_train)
nBatch = int(nTrain/batchSize)

score_list = []
for lr_test in [10**(-i) for i in range(10)]:
  pct_test = Perceptron(lr = lr_test)
  for epoch in range(n_epoch):
    idxShuffle = np.random.permutation(X_train.shape[0])
    for idxSet in range(nBatch):
      X_batch = X_train_std[idxShuffle[idxSet*batchSize:(idxSet+1)*batchSize], :]
      y_batch = y_train_onehot[idxShuffle[idxSet*batchSize:(idxSet+1)*batchSize], :]
      pct_test.training(X_batch, y_batch)
    if epoch % 5 == 0:
        print("epoch : {}".format(epoch))
  success = 0
  failure = 0
  for i in range(len(X_test_std)):
    target = np.argmax(y_test_onehot[i])
    prediction = np.argmax(pct_test.feedforward_2(X_test_std[i]))
    if target == prediction:
      success += 1
    else:
      failure += 1
  score_list.append([lr_test, (success)/ (failure + success)])
  print("lr : {} and prediction score is {}".format(lr_test, (success)/ (failure + success)))
```

    C:\Users\윤철환\AppData\Local\Temp\ipykernel_14472\3133858287.py:13: RuntimeWarning: overflow encountered in exp
      return 1 / (1+np.exp(-x))
    

    epoch : 0
    epoch : 5
    epoch : 10
    epoch : 15
    epoch : 20
    epoch : 25
    epoch : 30
    epoch : 35
    lr : 1 and prediction score is 0.098
    epoch : 0
    epoch : 5
    epoch : 10
    epoch : 15
    epoch : 20
    epoch : 25
    epoch : 30
    epoch : 35
    lr : 0.1 and prediction score is 0.098
    epoch : 0
    epoch : 5
    epoch : 10
    epoch : 15
    epoch : 20
    epoch : 25
    epoch : 30
    epoch : 35
    lr : 0.01 and prediction score is 0.0958
    epoch : 0
    epoch : 5
    epoch : 10
    epoch : 15
    epoch : 20
    epoch : 25
    epoch : 30
    epoch : 35
    lr : 0.001 and prediction score is 0.9427
    epoch : 0
    epoch : 5
    epoch : 10
    epoch : 15
    epoch : 20
    epoch : 25
    epoch : 30
    epoch : 35
    lr : 0.0001 and prediction score is 0.8878
    epoch : 0
    epoch : 5
    epoch : 10
    epoch : 15
    epoch : 20
    epoch : 25
    epoch : 30
    epoch : 35
    lr : 1e-05 and prediction score is 0.5501
    epoch : 0
    epoch : 5
    epoch : 10
    epoch : 15
    epoch : 20
    epoch : 25
    epoch : 30
    epoch : 35
    lr : 1e-06 and prediction score is 0.2177
    epoch : 0
    epoch : 5
    epoch : 10
    epoch : 15
    epoch : 20
    epoch : 25
    epoch : 30
    epoch : 35
    lr : 1e-07 and prediction score is 0.102
    epoch : 0
    epoch : 5
    epoch : 10
    epoch : 15
    epoch : 20
    epoch : 25
    epoch : 30
    epoch : 35
    lr : 1e-08 and prediction score is 0.1013
    epoch : 0
    epoch : 5
    epoch : 10
    epoch : 15
    epoch : 20
    epoch : 25
    epoch : 30
    epoch : 35
    lr : 1e-09 and prediction score is 0.0645
    


```python
plt.plot(np.arange(0,len(score_list),1),np.array(score_list)[:,1])
plt.xticks(np.arange(0,len(score_list),1),[(-i) for i in range(10)])
plt.xlabel('log(learning rate)')
plt.ylabel('Score')
plt.title('Test score by learning rate')
plt.scatter(np.argmax(np.array(score_list)[:,1]), max(np.array(score_list)[:,1]), marker = 'X', color = 'orange', s = 200, label = '10**(-3) (Max)')
plt.legend()
plt.show()
```


```python
# Using lr decay #
n_epoch = 40
batchSize = 200
nTrain = len(X_train)
nBatch = int(nTrain/batchSize)

score_list_decay = []

for decay_size in [2, 4, 5, 10, 20, 30, 40]:
  lr_init = 0.001
  pct_d = Perceptron(lr = lr_init)
  for epoch in range(n_epoch):
    idxShuffle = np.random.permutation(X_train.shape[0])
    for idxSet in range(nBatch):
      X_batch = X_train_std[idxShuffle[idxSet*batchSize:(idxSet+1)*batchSize], :]
      y_batch = y_train_onehot[idxShuffle[idxSet*batchSize:(idxSet+1)*batchSize], :]
      pct_d.training(X_batch, y_batch)
    print("epoch : {}, lr : {}".format(epoch, pct_d.lr))
    if epoch != 0 and epoch % decay_size == 0:
      pct_d.lr = 0.8 * pct_d.lr
  success = 0
  failure = 0
  for i in range(len(X_test_std)):
    target = np.argmax(y_test_onehot[i])
    prediction = np.argmax(pct_d.feedforward_2(X_test_std[i]))
    if target == prediction:
      success += 1
    else:
      failure += 1
  score_list_decay.append([decay_size, (success)/ (failure + success)])
  print("Decay size : {} and prediction score is {}".format(decay_size, (success)/ (failure + success)))
```

    epoch : 0, lr : 0.001
    epoch : 1, lr : 0.001
    epoch : 2, lr : 0.001
    epoch : 3, lr : 0.0008
    epoch : 4, lr : 0.0008
    epoch : 5, lr : 0.00064
    epoch : 6, lr : 0.00064
    epoch : 7, lr : 0.0005120000000000001
    epoch : 8, lr : 0.0005120000000000001
    epoch : 9, lr : 0.0004096000000000001
    epoch : 10, lr : 0.0004096000000000001
    epoch : 11, lr : 0.0003276800000000001
    epoch : 12, lr : 0.0003276800000000001
    epoch : 13, lr : 0.0002621440000000001
    epoch : 14, lr : 0.0002621440000000001
    epoch : 15, lr : 0.00020971520000000012
    epoch : 16, lr : 0.00020971520000000012
    epoch : 17, lr : 0.0001677721600000001
    epoch : 18, lr : 0.0001677721600000001
    epoch : 19, lr : 0.00013421772800000008
    epoch : 20, lr : 0.00013421772800000008
    epoch : 21, lr : 0.00010737418240000007
    epoch : 22, lr : 0.00010737418240000007
    epoch : 23, lr : 8.589934592000007e-05
    epoch : 24, lr : 8.589934592000007e-05
    epoch : 25, lr : 6.871947673600006e-05
    epoch : 26, lr : 6.871947673600006e-05
    epoch : 27, lr : 5.497558138880005e-05
    epoch : 28, lr : 5.497558138880005e-05
    epoch : 29, lr : 4.3980465111040044e-05
    epoch : 30, lr : 4.3980465111040044e-05
    epoch : 31, lr : 3.5184372088832036e-05
    epoch : 32, lr : 3.5184372088832036e-05
    epoch : 33, lr : 2.814749767106563e-05
    epoch : 34, lr : 2.814749767106563e-05
    epoch : 35, lr : 2.2517998136852506e-05
    epoch : 36, lr : 2.2517998136852506e-05
    epoch : 37, lr : 1.8014398509482006e-05
    epoch : 38, lr : 1.8014398509482006e-05
    epoch : 39, lr : 1.4411518807585605e-05
    Decay size : 2 and prediction score is 0.9147
    epoch : 0, lr : 0.001
    epoch : 1, lr : 0.001
    epoch : 2, lr : 0.001
    epoch : 3, lr : 0.001
    epoch : 4, lr : 0.001
    epoch : 5, lr : 0.0008
    epoch : 6, lr : 0.0008
    epoch : 7, lr : 0.0008
    epoch : 8, lr : 0.0008
    epoch : 9, lr : 0.00064
    epoch : 10, lr : 0.00064
    epoch : 11, lr : 0.00064
    epoch : 12, lr : 0.00064
    epoch : 13, lr : 0.0005120000000000001
    epoch : 14, lr : 0.0005120000000000001
    epoch : 15, lr : 0.0005120000000000001
    epoch : 16, lr : 0.0005120000000000001
    epoch : 17, lr : 0.0004096000000000001
    epoch : 18, lr : 0.0004096000000000001
    epoch : 19, lr : 0.0004096000000000001
    epoch : 20, lr : 0.0004096000000000001
    epoch : 21, lr : 0.0003276800000000001
    epoch : 22, lr : 0.0003276800000000001
    epoch : 23, lr : 0.0003276800000000001
    epoch : 24, lr : 0.0003276800000000001
    epoch : 25, lr : 0.0002621440000000001
    epoch : 26, lr : 0.0002621440000000001
    epoch : 27, lr : 0.0002621440000000001
    epoch : 28, lr : 0.0002621440000000001
    epoch : 29, lr : 0.00020971520000000012
    epoch : 30, lr : 0.00020971520000000012
    epoch : 31, lr : 0.00020971520000000012
    epoch : 32, lr : 0.00020971520000000012
    epoch : 33, lr : 0.0001677721600000001
    epoch : 34, lr : 0.0001677721600000001
    epoch : 35, lr : 0.0001677721600000001
    epoch : 36, lr : 0.0001677721600000001
    epoch : 37, lr : 0.00013421772800000008
    epoch : 38, lr : 0.00013421772800000008
    epoch : 39, lr : 0.00013421772800000008
    Decay size : 4 and prediction score is 0.9298
    epoch : 0, lr : 0.001
    epoch : 1, lr : 0.001
    epoch : 2, lr : 0.001
    epoch : 3, lr : 0.001
    epoch : 4, lr : 0.001
    epoch : 5, lr : 0.001
    epoch : 6, lr : 0.0008
    epoch : 7, lr : 0.0008
    epoch : 8, lr : 0.0008
    epoch : 9, lr : 0.0008
    epoch : 10, lr : 0.0008
    epoch : 11, lr : 0.00064
    epoch : 12, lr : 0.00064
    epoch : 13, lr : 0.00064
    epoch : 14, lr : 0.00064
    epoch : 15, lr : 0.00064
    epoch : 16, lr : 0.0005120000000000001
    epoch : 17, lr : 0.0005120000000000001
    epoch : 18, lr : 0.0005120000000000001
    epoch : 19, lr : 0.0005120000000000001
    epoch : 20, lr : 0.0005120000000000001
    epoch : 21, lr : 0.0004096000000000001
    epoch : 22, lr : 0.0004096000000000001
    epoch : 23, lr : 0.0004096000000000001
    epoch : 24, lr : 0.0004096000000000001
    epoch : 25, lr : 0.0004096000000000001
    epoch : 26, lr : 0.0003276800000000001
    epoch : 27, lr : 0.0003276800000000001
    epoch : 28, lr : 0.0003276800000000001
    epoch : 29, lr : 0.0003276800000000001
    epoch : 30, lr : 0.0003276800000000001
    epoch : 31, lr : 0.0002621440000000001
    epoch : 32, lr : 0.0002621440000000001
    epoch : 33, lr : 0.0002621440000000001
    epoch : 34, lr : 0.0002621440000000001
    epoch : 35, lr : 0.0002621440000000001
    epoch : 36, lr : 0.00020971520000000012
    epoch : 37, lr : 0.00020971520000000012
    epoch : 38, lr : 0.00020971520000000012
    epoch : 39, lr : 0.00020971520000000012
    Decay size : 5 and prediction score is 0.9281
    epoch : 0, lr : 0.001
    epoch : 1, lr : 0.001
    epoch : 2, lr : 0.001
    epoch : 3, lr : 0.001
    epoch : 4, lr : 0.001
    epoch : 5, lr : 0.001
    epoch : 6, lr : 0.001
    epoch : 7, lr : 0.001
    epoch : 8, lr : 0.001
    epoch : 9, lr : 0.001
    epoch : 10, lr : 0.001
    epoch : 11, lr : 0.0008
    epoch : 12, lr : 0.0008
    epoch : 13, lr : 0.0008
    epoch : 14, lr : 0.0008
    epoch : 15, lr : 0.0008
    epoch : 16, lr : 0.0008
    epoch : 17, lr : 0.0008
    epoch : 18, lr : 0.0008
    epoch : 19, lr : 0.0008
    epoch : 20, lr : 0.0008
    epoch : 21, lr : 0.00064
    epoch : 22, lr : 0.00064
    epoch : 23, lr : 0.00064
    epoch : 24, lr : 0.00064
    epoch : 25, lr : 0.00064
    epoch : 26, lr : 0.00064
    epoch : 27, lr : 0.00064
    epoch : 28, lr : 0.00064
    epoch : 29, lr : 0.00064
    epoch : 30, lr : 0.00064
    epoch : 31, lr : 0.0005120000000000001
    epoch : 32, lr : 0.0005120000000000001
    epoch : 33, lr : 0.0005120000000000001
    epoch : 34, lr : 0.0005120000000000001
    epoch : 35, lr : 0.0005120000000000001
    epoch : 36, lr : 0.0005120000000000001
    epoch : 37, lr : 0.0005120000000000001
    epoch : 38, lr : 0.0005120000000000001
    epoch : 39, lr : 0.0005120000000000001
    Decay size : 10 and prediction score is 0.937
    epoch : 0, lr : 0.001
    epoch : 1, lr : 0.001
    epoch : 2, lr : 0.001
    epoch : 3, lr : 0.001
    epoch : 4, lr : 0.001
    epoch : 5, lr : 0.001
    epoch : 6, lr : 0.001
    epoch : 7, lr : 0.001
    epoch : 8, lr : 0.001
    epoch : 9, lr : 0.001
    epoch : 10, lr : 0.001
    epoch : 11, lr : 0.001
    epoch : 12, lr : 0.001
    epoch : 13, lr : 0.001
    epoch : 14, lr : 0.001
    epoch : 15, lr : 0.001
    epoch : 16, lr : 0.001
    epoch : 17, lr : 0.001
    epoch : 18, lr : 0.001
    epoch : 19, lr : 0.001
    epoch : 20, lr : 0.001
    epoch : 21, lr : 0.0008
    epoch : 22, lr : 0.0008
    epoch : 23, lr : 0.0008
    epoch : 24, lr : 0.0008
    epoch : 25, lr : 0.0008
    epoch : 26, lr : 0.0008
    epoch : 27, lr : 0.0008
    epoch : 28, lr : 0.0008
    epoch : 29, lr : 0.0008
    epoch : 30, lr : 0.0008
    epoch : 31, lr : 0.0008
    epoch : 32, lr : 0.0008
    epoch : 33, lr : 0.0008
    epoch : 34, lr : 0.0008
    epoch : 35, lr : 0.0008
    epoch : 36, lr : 0.0008
    epoch : 37, lr : 0.0008
    epoch : 38, lr : 0.0008
    epoch : 39, lr : 0.0008
    Decay size : 20 and prediction score is 0.9393
    epoch : 0, lr : 0.001
    epoch : 1, lr : 0.001
    epoch : 2, lr : 0.001
    epoch : 3, lr : 0.001
    epoch : 4, lr : 0.001
    epoch : 5, lr : 0.001
    epoch : 6, lr : 0.001
    epoch : 7, lr : 0.001
    epoch : 8, lr : 0.001
    epoch : 9, lr : 0.001
    epoch : 10, lr : 0.001
    epoch : 11, lr : 0.001
    epoch : 12, lr : 0.001
    epoch : 13, lr : 0.001
    epoch : 14, lr : 0.001
    epoch : 15, lr : 0.001
    epoch : 16, lr : 0.001
    epoch : 17, lr : 0.001
    epoch : 18, lr : 0.001
    epoch : 19, lr : 0.001
    epoch : 20, lr : 0.001
    epoch : 21, lr : 0.001
    epoch : 22, lr : 0.001
    epoch : 23, lr : 0.001
    epoch : 24, lr : 0.001
    epoch : 25, lr : 0.001
    epoch : 26, lr : 0.001
    epoch : 27, lr : 0.001
    epoch : 28, lr : 0.001
    epoch : 29, lr : 0.001
    epoch : 30, lr : 0.001
    epoch : 31, lr : 0.0008
    epoch : 32, lr : 0.0008
    epoch : 33, lr : 0.0008
    epoch : 34, lr : 0.0008
    epoch : 35, lr : 0.0008
    epoch : 36, lr : 0.0008
    epoch : 37, lr : 0.0008
    epoch : 38, lr : 0.0008
    epoch : 39, lr : 0.0008
    Decay size : 30 and prediction score is 0.9414
    epoch : 0, lr : 0.001
    epoch : 1, lr : 0.001
    epoch : 2, lr : 0.001
    epoch : 3, lr : 0.001
    epoch : 4, lr : 0.001
    epoch : 5, lr : 0.001
    epoch : 6, lr : 0.001
    epoch : 7, lr : 0.001
    epoch : 8, lr : 0.001
    epoch : 9, lr : 0.001
    epoch : 10, lr : 0.001
    epoch : 11, lr : 0.001
    epoch : 12, lr : 0.001
    epoch : 13, lr : 0.001
    epoch : 14, lr : 0.001
    epoch : 15, lr : 0.001
    epoch : 16, lr : 0.001
    epoch : 17, lr : 0.001
    epoch : 18, lr : 0.001
    epoch : 19, lr : 0.001
    epoch : 20, lr : 0.001
    epoch : 21, lr : 0.001
    epoch : 22, lr : 0.001
    epoch : 23, lr : 0.001
    epoch : 24, lr : 0.001
    epoch : 25, lr : 0.001
    epoch : 26, lr : 0.001
    epoch : 27, lr : 0.001
    epoch : 28, lr : 0.001
    epoch : 29, lr : 0.001
    epoch : 30, lr : 0.001
    epoch : 31, lr : 0.001
    epoch : 32, lr : 0.001
    epoch : 33, lr : 0.001
    epoch : 34, lr : 0.001
    epoch : 35, lr : 0.001
    epoch : 36, lr : 0.001
    epoch : 37, lr : 0.001
    epoch : 38, lr : 0.001
    epoch : 39, lr : 0.001
    Decay size : 40 and prediction score is 0.9429
    


```python
plt.plot(np.arange(7),np.array(score_list_decay)[:,1])
plt.xlabel('log(learning rate)')
plt.ylabel('Score')
plt.xticks(np.arange(7),[2, 4, 5, 10, 20, 30, 40])
plt.title('Test score by Decay size')
plt.scatter(np.argmax(np.array(score_list_decay)[:,1]), max(np.array(score_list_decay)[:,1]), marker = 'X', color = 'orange', s = 200, label = 'Max')
plt.legend()
plt.show()
```


    
![png](output_28_0.png)
    



```python
# Using optimal lr and decay size #
n_epoch = 40
batchSize = 200
nTrain = len(X_train)
nBatch = int(nTrain/batchSize)
decay_size = 40
lr_init = 0.001
pct_opt = Perceptron(lr = lr_init)

for epoch in range(n_epoch):
  idxShuffle = np.random.permutation(X_train.shape[0])
  for idxSet in range(nBatch):
    X_batch = X_train_std[idxShuffle[idxSet*batchSize:(idxSet+1)*batchSize], :]
    y_batch = y_train_onehot[idxShuffle[idxSet*batchSize:(idxSet+1)*batchSize], :]
    pct_opt.training(X_batch, y_batch)
  print("epoch : {}, lr : {}".format(epoch, pct_opt.lr))
  if epoch != 0 and epoch % decay_size == 0:
    pct_opt.lr = 0.8 * pct_opt.lr
```

    epoch : 0, lr : 0.001
    epoch : 1, lr : 0.0008
    epoch : 2, lr : 0.0008
    epoch : 3, lr : 0.0008
    epoch : 4, lr : 0.0008
    epoch : 5, lr : 0.0008
    epoch : 6, lr : 0.0008
    epoch : 7, lr : 0.0008
    epoch : 8, lr : 0.0008
    epoch : 9, lr : 0.0008
    epoch : 10, lr : 0.0008
    epoch : 11, lr : 0.0008
    epoch : 12, lr : 0.0008
    epoch : 13, lr : 0.0008
    epoch : 14, lr : 0.0008
    epoch : 15, lr : 0.0008
    epoch : 16, lr : 0.0008
    epoch : 17, lr : 0.0008
    epoch : 18, lr : 0.0008
    epoch : 19, lr : 0.0008
    epoch : 20, lr : 0.0008
    epoch : 21, lr : 0.0008
    epoch : 22, lr : 0.0008
    epoch : 23, lr : 0.0008
    epoch : 24, lr : 0.0008
    epoch : 25, lr : 0.0008
    epoch : 26, lr : 0.0008
    epoch : 27, lr : 0.0008
    epoch : 28, lr : 0.0008
    epoch : 29, lr : 0.0008
    epoch : 30, lr : 0.0008
    epoch : 31, lr : 0.0008
    epoch : 32, lr : 0.0008
    epoch : 33, lr : 0.0008
    epoch : 34, lr : 0.0008
    epoch : 35, lr : 0.0008
    epoch : 36, lr : 0.0008
    epoch : 37, lr : 0.0008
    epoch : 38, lr : 0.0008
    epoch : 39, lr : 0.0008
    


```python
pred_failure_list = []
failure_list = []
for i in range(len(X_test_std)):
  target = np.argmax(y_test_onehot[i])
  prediction = np.argmax(pct_opt.feedforward_2(X_test_std[i]))
  if target != prediction:
    pred_failure_list.append([prediction,target])
    failure_list.append(X_test_std[i])
print("Prediction score is {}".format((success)/ (failure + success)))
```

    Prediction score is 0.9429
    


```python
# 10개 랜덤 데이터 #
idxShuffle_test = np.random.permutation(len(X_test_std))
for i in range(10):
  idx = idxShuffle_test[i]
  # print("Prediction is {}, label is {}".format(np.argmax(pct_opt.feedforward_2(X_test_std[idx])), np.argmax(y_test_onehot[idx])))
  f, ax = plt.subplots(figsize = (6,6))
  ax.set_title("Pred : {} Label : {}".format(np.argmax(pct_opt.feedforward_2(X_test_std[idx])), np.argmax(y_test_onehot[idx])))
  ax.imshow(X_test_std[idx].reshape(28,28), cmap = cm.gray)
```


    
![png](output_31_0.png)
    



    
![png](output_31_1.png)
    



    
![png](output_31_2.png)
    



    
![png](output_31_3.png)
    



    
![png](output_31_4.png)
    



    
![png](output_31_5.png)
    



    
![png](output_31_6.png)
    



    
![png](output_31_7.png)
    



    
![png](output_31_8.png)
    



    
![png](output_31_9.png)
    



```python
# 10개 실패 데이터 #
for i in np.random.permutation(len(pred_failure_list))[:10]:
  f, ax = plt.subplots(figsize = (6,6))
  ax.imshow(failure_list[i].reshape(28,28), cmap = cm.gray)
  ax.set_title("Pred : {} Label : {}".format(pred_failure_list[i][0],pred_failure_list[i][1]))
```


    
![png](output_32_0.png)
    



    
![png](output_32_1.png)
    



    
![png](output_32_2.png)
    



    
![png](output_32_3.png)
    



    
![png](output_32_4.png)
    



    
![png](output_32_5.png)
    



    
![png](output_32_6.png)
    



    
![png](output_32_7.png)
    



    
![png](output_32_8.png)
    



    
![png](output_32_9.png)
    


AutoEncoder


```python
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train_reshape = X_train.reshape(-1,28*28)
X_test_reshape = X_test.reshape(-1,28*28)
X_train_std = X_train_reshape / 255.0
X_test_std = X_test_reshape / 255.0
```


```python
# 노이즈 #
plt.imshow(np.random.normal(0.0,0.1,(28,28)), cmap = cm.gray)
plt.title("Noise")
plt.colorbar()
plt.show()
```


    
![png](output_35_0.png)
    



```python
X_train_noise = X_train_std + np.random.normal(0.0,0.1,(60000,28**2))
X_train_noise[X_train_noise>1.0] = 1.0
```


```python
n_epoch = 10
batchSize = 200
nTrain = len(X_train)
nBatch = int(nTrain/batchSize)

pct_noise = Perceptron(output_dim = 28**2, lr = 0.001)

for epoch in range(n_epoch):
  idxShuffle = np.random.permutation(X_train_noise.shape[0])
  for idxSet in range(nBatch):
    X_batch = X_train_noise[idxShuffle[idxSet*batchSize:(idxSet+1)*batchSize], :]
    y_batch = X_train_std[idxShuffle[idxSet*batchSize:(idxSet+1)*batchSize], :]
    pct_noise.training(X_batch, y_batch)
  print("epoch : {}".format(epoch))
```

    epoch : 0
    epoch : 1
    epoch : 2
    epoch : 3
    epoch : 4
    epoch : 5
    epoch : 6
    epoch : 7
    epoch : 8
    epoch : 9
    


```python
X_test_noise = X_test_std + np.random.normal(0.0,0.1,(len(X_test_std),28**2))
X_test_noise[X_test_noise>1.0] = 1.0
```


```python
idxShuffle = np.random.permutation(len(X_test_noise))
plt.figure(figsize = (10,30))
plt.suptitle("Data vs Data + Noise")
plt.subplots_adjust(top = 0.96,hspace = 0.4)
for i in range(10):
  idx = idxShuffle[i]
  hi = pct_noise.feedforward_2(X_test_noise[idx].reshape(1,-1)).reshape(28,28)
  plt.subplot(10,2,2*i+1)
  plt.title("{} (test)".format(np.argmax(y_test_onehot[idx])))
  plt.imshow(hi, cmap = cm.gray)
  plt.subplot(10,2,2*i+2)
  plt.title("{} (noise)".format(np.argmax(y_test_onehot[idx])))
  plt.imshow(X_test_noise[idx].reshape(28,28), cmap = cm.gray)
```


    
![png](output_39_0.png)
    

