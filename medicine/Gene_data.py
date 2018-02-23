
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import gc
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import KFold, StratifiedKFold, cross_val_score
from datetime import datetime
#from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler,LabelEncoder, OneHotEncoder, minmax_scale, scale, LabelBinarizer
from sklearn import tree
from sklearn import linear_model
from sklearn import svm
from sklearn import neighbors
from sklearn import ensemble
from sklearn.feature_selection import SelectFromModel, VarianceThreshold,RFE, f_regression
#from minepy import MINE
from mlxtend.regressor import StackingRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import gc
color = sns.color_palette()
from mlxtend.regressor import StackingRegressor, StackingCVRegressor

import tensorflow as tf
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn import preprocessing


# # Data Pre

# In[2]:


gene=pd.read_csv('/home/dope/data_set/medicine/Gene_data.csv')
train=pd.read_csv('/home/dope/data_set/medicine/f_train_20180204.csv',encoding='gbk')
label_train=train['label']
y_train=[]
for index , value in enumerate(label_train):
    y_train.append([1,0]) if value == 1 else y_train.append([0,1])
gene_scaled=preprocessing.scale(gene)
gene_test=gene_scaled[1000:]
gene_train=gene_scaled[0:1000]


# gene_scaled[0].reshape(702,9,1)

# # CNN

# In[3]:


INPUT_WIDTH=702
INPUT_HEIGHT=9
CONV_WIDTH=3
TRAINNING_STEP=1000
IN_CHANNEL_SIZE=1
OUT_CHANNEL_SIZE=32
OUT_CHANNEL_1_SIZE=64
OUT_CHANNEL_2_SIZE=128
OUTPUT_SIZE=2

LEARNING_RATE_BASE = 0.025
LEARNING_RATE_DECAY = 0.99


# In[4]:


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def get_train_data(data,times):
    return_data=[]
    for i in range(10):
        return_data.append(np.array(train.ix[times*10+i]).reshape([INPUT_WIDTH,INPUT_HEIGHT,1]))
    return return_data
def get_train_y(label,times):
    return_y=[]
    for i in range(10):
        return_y.append(y_train[times*10+i])
    return return_y

def get_test_data(data):
    return_data=[]
    for i in range(200):
        return_data.append(np.array(train.ix[800+i]).reshape([INPUT_WIDTH,INPUT_HEIGHT,1]))
    return return_data

def get_test_y(data):
    return_y=[]
    for i in range(200):
        return_y.append(y_train[800+i])
    return return_y

def get_train_data_trainfuc(data,times):
    return_data=[]
    for i in range(10):
        return_data.append(np.array(data[times*10+i]).reshape([INPUT_WIDTH,INPUT_HEIGHT,1]))
    return return_data
def get_train_y_trainfuc(label,times):
    return_y=[]
    for i in range(10):
        return_y.append(label[times*10+i])
    return return_y

def get_test_data_trainfuc(data):
    return_data=[]
    for i in range(200):
        return_data.append(np.array(data[i]).reshape([INPUT_WIDTH,INPUT_HEIGHT,1]))
    return return_data

def get_test_y_trainfuc(data):
    return_y=[]
    for i in range(200):
        return_y.append(data[i])
    return return_y

def cal_acc():
    pass


# In[5]:


with tf.variable_scope("cnn",reuse=tf.AUTO_REUSE):
    train_X = tf.placeholder(tf.float32,[None,INPUT_WIDTH,INPUT_HEIGHT,1])
    train_y = tf.placeholder(tf.float32,[None,OUTPUT_SIZE])
    keep_prob = tf.placeholder(tf.float32)
    global_step = tf.Variable(0, trainable=False)
    # conv1 layer
    W_conv1 = weight_variable([CONV_WIDTH,CONV_WIDTH,IN_CHANNEL_SIZE,OUT_CHANNEL_SIZE])
    b_conv1 = bias_variable([OUT_CHANNEL_SIZE])
    h_conv1 = tf.nn.tanh(conv2d(train_X,W_conv1)+b_conv1) #output size is 702x9xOUT_CHANNEL_SIZE
    h_pool1 = max_pool_2x2(h_conv1) #output size is 351x5xOUT_CHANNEL_SIZE
    
    #conv2 layer
    W_conv2 = weight_variable([CONV_WIDTH,CONV_WIDTH,OUT_CHANNEL_SIZE,OUT_CHANNEL_1_SIZE])
    b_conv2 = bias_variable([OUT_CHANNEL_1_SIZE])
    h_conv2 = tf.nn.tanh(conv2d(h_pool1,W_conv2)+b_conv2)#output size is 351x5xOUT_CHANNEL_1_SIZE
    h_pool2 = max_pool_2x2(h_conv2) #output size is  x xOUT_CHANNEL_1_SIZE
    
    #conv3 layer
    #W_conv3 = weight_variable([CONV_WIDTH,CONV_WIDTH,OUT_CHANNEL_1_SIZE,OUT_CHANNEL_2_SIZE])
    #b_conv3 = bias_variable([OUT_CHANNEL_2_SIZE])
    #h_conv3 = tf.nn.tanh(conv2d(h_pool2,W_conv3)+b_conv3) #output size is 4x4xOUT_CHANNEL_2_SIZE
    #h_pool3 = max_pool_2x2(h_conv3) #output size is 2*2*OUT_CHANNEL_2_SIZE
    
    #func1 layer
    W_fc1 = weight_variable([351*5*OUT_CHANNEL_1_SIZE,6])
    b_fc1 = bias_variable([6])
    h_pool1_flat = tf.reshape(h_conv2,[-1,351*5*OUT_CHANNEL_1_SIZE])
    h_fc1 = tf.nn.tanh(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    #func2 layer
    W_fc2 = weight_variable([6,2])
    b_fc2 = bias_variable([2])
    prediction = tf.nn.softmax(tf.matmul(h_fc1,W_fc2) + b_fc2)
    
    #ops
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(train_y * tf.log(prediction),reduction_indices=[1]))
    #loss_op = tf.reduce_mean(tf.square(train_y - prediction))
    
    #other param
    learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            200, LEARNING_RATE_DECAY,
            staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(cross_entropy,global_step=global_step)
    
    #accuracy
    y_true = tf.argmin(train_y,1)
    y_pred= tf.argmin(prediction,1)
    correct_prediction = tf.equal(tf.argmin(prediction,1), tf.argmin(train_y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()


# In[6]:


def train_func(train_data,train_label,test_data,test_label):
    with tf.Session() as sess:
        sess.run(init)
        for j in range(86):
            for i in range(int(len(train_data)/10)):
                _data=get_train_data_trainfuc(train_data,i)
                _label=get_train_y_trainfuc(train_label,i)
                sess.run(train_op,feed_dict={train_X:_data , train_y:_label , keep_prob:0.3})
            #print(j,sess.run(cross_entropy,feed_dict={train_X:[_data[0]] , train_y:[_label[0]] , keep_prob:0.3}))
        #sess.run(train_op,feed_dict={train_X: [np.array(train.ix[0]).reshape([8,8,1]) , np.array(train.ix[1]).reshape([8,8,1])] , train_y:[y_train[0],y_train[1]] , keep_prob:0.3})
        test_data_fuc = get_test_data_trainfuc(test_data)
        test_y_fuc = get_test_y_trainfuc(test_label)
        print(sess.run(prediction,feed_dict={train_X:_data , train_y:_label , keep_prob:0.3}))
        print(sess.run(accuracy,feed_dict={train_X:test_data_fuc , train_y:test_y_fuc , keep_prob:0.3}))
            
        true_y , pred_y = sess.run([y_true , y_pred],feed_dict={train_X:test_data_fuc , train_y:test_y_fuc })
        p = precision_score(true_y,pred_y,average='binary')
        r = recall_score(true_y,pred_y,average='binary')
        f1score = f1_score(true_y,pred_y,average='binary')
        print(p,r,f1score)
        #print(W_conv1.eval())


# In[ ]:


test=list(gene_train[0:0])+list(gene_train[200:])
len(test[0])


# In[ ]:


for index in range(5):
    test_data=list(gene_train[index*200:index*200+200])#pd.concat([train.ix[index*200:index*200+200]],ignore_index =True)
    test_label=y_train[index*200:index*200+200]
    train_data=list(gene_train[0:index*200])+list(gene_train[index*200+200:])#pd.concat([train[0:index*200],train[index*200+200:]],ignore_index=True )
    train_label=y_train[0:index*200]+y_train[index*200+200:]
    train_func(train_data,train_label,test_data,test_label)

