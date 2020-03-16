---
layout: post
title: "DeepFool 구현하기"
comments : true
category : Machine Learning
---

DeepFool 알고리즘을 텐서플로우 1.X를 이용해서 구현해보았다.
<br/>
코드는 <https://github.com/LTS4/universal>를 참조했다. 
<br/>
(저기에 있는 코드를 거의 그대로 따라했다.)

## 필요한 모듈 import
```python
from keras.datasets import cifar10
import tensorflow as tf
from tensorflow.keras.layers import Flatten
import numpy as np
from keras.utils import to_categorical 
from sklearn.utils import shuffle
import os
import matplotlib.pyplot as plt
```
<br/>

## Lenet 구현
#### 데이터 전처리
```python
def pre_data():
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()

  x_train = x_train / 255.
  x_test = x_test / 255.

  y_train = to_categorical(y_train, 10,dtype='float32')
  y_test = to_categorical(y_test, 10,dtype='float32')  

  return x_train, y_train, x_test, y_test
```
<br/>

#### Classifier 생성
```python
def init_weight(shape):
  w = tf.truncated_normal(shape = shape, stddev = 0.1)
  return tf.Variable(w)

def init_bias(shape):
  b = tf.zeros(shape)
  return tf.Variable(b)
  
def CNN_classifier(x):

  conv1_w = init_weight((5,5,3,64))
  conv1_b = init_bias(64)
  conv1 = tf.nn.conv2d(x,conv1_w, strides = [1,1,1,1], padding = 'SAME') + conv1_b
  conv1 = tf.nn.relu(conv1)

  conv1 = tf.nn.max_pool2d(conv1,ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

  conv2_w = init_weight((5,5,64,64))
  conv2_b = init_bias(64)
  conv2 = tf.nn.conv2d(conv1,conv2_w, strides = [1,1,1,1], padding = "SAME") + conv2_b
  conv2 = tf.nn.relu(conv2)

  conv2 = tf.nn.max_pool2d(conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = "SAME")

  conv3_w = init_weight((5,5,64,64))
  conv3_b = init_bias(64)
  conv3 = tf.nn.conv2d(conv2,conv3_w, strides = [1,1,1,1], padding = "SAME") + conv3_b
  conv3 = tf.nn.relu(conv3)

  conv3 = tf.nn.max_pool2d(conv3, ksize = [1,2,2,1], strides = [1,2,2,1], padding = "SAME")

  fc0 = Flatten()(conv3)

  fc1_w = init_weight((1024,120))
  fc1_b = init_bias(120)
  fc1 = tf.matmul(fc0,fc1_w) + fc1_b
  fc1 = tf.nn.relu(fc1)

  fc2_w = init_weight((120,84))
  fc2_b = init_bias(84)
  fc2 = tf.matmul(fc1,fc2_w) + fc2_b
  fc2 = tf.nn.relu(fc2)

  fc3_w = init_weight((84,10))
  fc3_b = init_bias(10)
  fc3 = tf.add(tf.matmul(fc2,fc3_w),fc3_b, name = 'last_layer')
  output = tf.nn.softmax(fc3, name = 'logits')
  
  return output
  
x_train, y_train, x_test, y_test = pre_data()
x_train, y_train = shuffle(x_train, y_train)

EPOCHS = 15
BATCH_SIZE = 128

x = tf.placeholder(tf.float32, (None,32,32,3), name = 'x')
y = tf.placeholder(tf.float32, (None,10), name = 'y')

logits = CNN_classifier(x)
loss_operation = tf.reduce_mean(-tf.reduce_sum(y * tf.log(logits), 1))

train_operation = tf.train.AdamOptimizer(1e-3).minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, axis = 1),tf.argmax(y, axis =1), name = 'correct_prediction')
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = 'accuracy_operation')
saver = tf.train.Saver()
```
<br/>

#### Classifier 훈련
```python
persisted_sess = tf.Session()
print (tf.global_variables())
persisted_sess.run(tf.global_variables_initializer())

num_examples = len(x_train)

print('Training...')

for i in range(EPOCHS):
  x_train, y_train = shuffle(x_train, y_train)

  for offset in range(0,num_examples,BATCH_SIZE):
    batch_x = x_train[offset:offset+BATCH_SIZE]
    batch_y = y_train[offset:offset+BATCH_SIZE]

    persisted_sess.run(train_operation, feed_dict = {x : batch_x, y: batch_y})
    
  train_accuracy = persisted_sess.run(accuracy_operation, feed_dict = { x : batch_x, y: batch_y})
  print ("EPOCH {}...".format(i+1))
  print ("Training Accuracy = {:.3f}".format(train_accuracy))
  
saver.save(persisted_sess,'CNN_classifier')
print ("Model Saved")

persisted_sess.close()
```
<br/>

#### Classifier 평가
```python
def evaluate(x_data, y_data):
  num_examples = len(x_data)
  total_accuracy = 0

  
  for offset in range(0,num_examples,BATCH_SIZE):
    batch_x, batch_y = x_test[offset:offset + BATCH_SIZE] , y_test[offset : offset + BATCH_SIZE]
    
    accuracy = sess.run(accuracy_operation, feed_dict = { x : batch_x, y : batch_y})
    total_accuracy += accuracy * len(batch_x)

  return total_accuracy / num_examples

with tf.Session() as sess:
  saver = tf.train.import_meta_graph('CNN_classifier.meta')
  saver.restore(sess, tf.train.latest_checkpoint('.'))
 
  test_accuracy = evaluate(x_test,y_test)
  print ("Test Accuracy : {}".format(test_accuracy))
```
<br/>

#### DeepFool 구현
```python
tf.reset_default_graph()

sess = tf.Session()
saver = tf.train.import_meta_graph('CNN_classifier.meta')
saver.restore(sess, tf.train.latest_checkpoint('.'))

num_classes = 10

def jacobian(y_flat,input_image,inds):
  grads = []

  for i in range(num_classes):
    grad = tf.gradients(y_flat[inds[i]],input_image)
    grads.append(grad)    

  return grads
  
last_layer = tf.get_default_graph().get_tensor_by_name("last_layer:0")
last_layer_flat = tf.reshape(last_layer, (-1,))

inds = tf.placeholder(tf.int32, (num_classes,))
x = tf.get_default_graph().get_tensor_by_name("x:0")

dydx = jacobian(last_layer_flat,x,inds)

def f(image):
  
  f_i = sess.run(last_layer, feed_dict = { x : image})

  return f_i

def grad_fs(image, idx):
  
  grads = sess.run(dydx, feed_dict = {x:image, inds : idx})
  grads = np.array(grads)
  grads = grads.squeeze(axis = 1)

  return grads

def DeepFool(image, f, grad_fs, num_classes = 10, overshoot = 0.02, max_iter = 50):
  
  f_image = f(image).flatten()

  I = f_image.argsort()[::-1]
  I = I[:num_classes]

  label = I[0]

  pert_image = image

  f_i = f(pert_image).flatten()
  k_i = int(np.argmax(f_i))

  w = np.zeros(image.shape)
  r_tot = np.zeros(image.shape)

  loop_i = 0
  
  while k_i == label and loop_i < max_iter:

    pert = np.inf
    gradients = grad_fs(pert_image, I)

    for k in range(1,num_classes):
      w_k = gradients[k,:,:,:,:] - gradients[0,:,:,:]
      f_k = f_i[I[k]] - f_i[I[0]]

      pert_k = abs(f_k) / np.sqrt(np.sum(w_k * w_k))
      #pert_k = abs(f_k) / np.linalg.norm(w_k.flatten(),2)


      if pert_k < pert:
        pert = pert_k
        w = w_k

    r_i = pert * w / np.sqrt(np.sum(w_k*w_k))
    # r_i = pert * w / np.linalg.norm(w_k.flatten(),2)

    r_tot = r_tot + r_i

    pert_image = image + (1+overshoot) * r_tot
    
    f_i = f(pert_image).flatten()
    k_i = int(np.argmax(f_i))

    loop_i = loop_i + 1

  r_tot = (1+overshoot) * r_tot

  return r_tot, loop_i, k_i, pert_image
```
<br/>

#### DeepFool 적용
```python
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

image = x_train[10]
c_ans = y_train[10]

image = np.array(image)
image = image /255.

image = image.reshape(1,32,32,3)

r_tot, loop_i, k_i, pert_image = DeepFool(image,f,grad_fs)

fig = plt.figure()

orig_pred = np.argmax(f(image))
deepfool_pred = np.argmax(f(pert_image))

ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(image.reshape(32,32,3))
ax1.set_title('original image ' + 'prediction : ' + str(orig_pred))

ax2 = fig.add_subplot(1, 2, 2)
ax2.imshow(pert_image.reshape(32,32,3))
ax2.set_title('perturbed image '+ 'prediction : ' + str(deepfool_pred))

plt.show()
```

