
# coding: utf-8

# In[1]:


import tensorflow as tf
import pandas as pd
import numpy as np


# In[2]:


shop_info = pd.read_csv('D:\\train_ccf_first_round_shop_info.csv')
user_shop_info = pd.read_csv('D:\\train_ccf_first_round_user_shop_behavior.csv')
testA_B = pd.read_csv('D:\\AB_test_evaluation_public.csv') 


# In[3]:


user_shop_info['user_id']=user_shop_info['user_id'].str[2:]
user_shop_info['shop_id']=user_shop_info['shop_id'].str[2:]
user_shop_info


# In[4]:


user_shop_info["date"] = pd.to_datetime(user_shop_info["time_stamp"])
user_shop_info["date_year"] = user_shop_info["date"].dt.year
user_shop_info["date_month"] = user_shop_info["date"].dt.month
user_shop_info["date_day"] = user_shop_info["date"].dt.day
user_shop_info["date_hour"] = user_shop_info["date"].dt.hour
user_shop_info["date_min"] = user_shop_info["date"].dt.minute
new_user_info=user_shop_info.drop(['time_stamp','date','wifi_infos'],axis=1)
shop_set=set(new_user_info['shop_id'])
y_=new_user_info['shop_id']
new_user_info=new_user_info.drop(['shop_id'],axis=1)
y__=pd.DataFrame([])  
for indexs in y_.index:
    y__[y_.loc[indexs]+'']=y_.loc[indexs]
    s=y_.loc[indexs]+''
    new= pd.DataFrame({s:y_.loc[indexs]},index=[0])
    y__=y__.append(new,ignore_index=True)
y__=np.array(y__)  
new_user_info=np.array(new_user_info)
y_tensor=tf.constant(y__, dtype = tf.float32, shape=list(y_.shape))
new_user_tensor=tf.constant(new_user_info,dtype = tf.float32, shape=list(new_user_info.shape))


# In[ ]:


y__


# In[ ]:


INPUT_NODE = 8
OUTPUT_NODE = len(shop_set)
LAYER1_NODE = 92

def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None: tf.add_to_collection('losses', regularizer(weights))
    return weights


def inference(input_tensor, regularizer):
    with tf.variable_scope('layer1',reuse=tf.AUTO_REUSE):

        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope('layer2',reuse=tf.AUTO_REUSE):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases

    return layer2


# In[ ]:


BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH="MNIST_model/"
MODEL_NAME="mnist_model"


def train(new_user_tensor,y_tensor):

    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)


    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        len(user_shop_info) / BATCH_SIZE, LEARNING_RATE_DECAY,
        staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')


    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        xs=new_user_tensor
        ys=y_tensor
        xs,ys=sess.run([xs,ys])
        
        for i in range(TRAINING_STEPS):
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))


# In[ ]:


train(new_user_tensor,y_tensor)

