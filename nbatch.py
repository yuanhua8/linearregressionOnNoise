import tensorflow as tf
import numpy as np
import math
import glob
import random
 
w = tf.Variable(tf.random_normal([127, 1], stddev=0.01))
l2w = tf.Variable(tf.random_normal([127, 1], stddev=0.01))
l1w = tf.Variable(tf.random_normal([127, 1], stddev=0.01))

x = tf.placeholder(tf.float32, shape=[None, 127])
yl = tf.placeholder(tf.float32, shape=[None, 1])
#l2_regularation = tf.reduce_sum(tf.square(w) * l2w) + tf.reduce_sum((tf.abs(w) * l1w))
l2_regularation = tf.reduce_sum(tf.square(w)) 
with tf.device('/gpu:0'):
    y = tf.matmul(x, w)
loss = tf.add(l2_regularation,  0.0001 * tf.nn.l2_loss(y - yl) )
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

s = tf.Session()
s.run(tf.global_variables_initializer())
 
train_list = glob.glob("data.*")
random.shuffle(train_list)
for index in range(len(train_list)):
    print("load=====>" + train_list[index])
    d = np.genfromtxt(train_list[index], dtype='float32')
    print("loaded=====>" + train_list[index])
    last_loss = 0.0
    num = len(d)
    dx = d[:, :127]
    dy = d[:, 129:130]
    while(1):
        # s.run(train_step, feed_dict={x: np.reshape(xv, [127,num]), yl: np.reshape(yv, [1, num])})
		# this_loss =s.run(loss, feed_dict={x: np.reshape(xv, [127,num]), yl: np.reshape(yv, [1, num])})
        s.run(train_step, feed_dict={x: dx, yl: dy})
        this_loss = s.run(loss, feed_dict={x: dx, yl: dy})
        print("loss" + str(this_loss))
        if (abs(this_loss - last_loss) < 0.0000001):
            print("last loss=" + str(this_loss))
            break
        last_loss = this_loss
           
 
    print("evaluate==>")
    print(s.run(w))
    #eval_list = ["data.20130123", "data.20130116", "data.20130117", "data.20130118", "data.20130121", "data.20130122"]
    eval_list=train_list
    for index in range(min(len(eval_list), 16)):
        xx = 0.0
        xy = 0.0
        yy = 0.0
        d = np.genfromtxt(eval_list[index], dtype='float32')
        for i in range(len(d)):
            d1 = d[i]
            y_target = d1[130]
            y_actual = s.run(y, feed_dict={x: np.reshape(d1[:127], [1,127])})
            wt = min(d1[127], 2)
            xx += y_target**2 * wt
            xy += y_actual * y_target * wt
            yy += y_actual**2 * wt
        corr = xy/ math.sqrt(xx * yy)
        print(eval_list[index]+"===>" + str(corr))
  
