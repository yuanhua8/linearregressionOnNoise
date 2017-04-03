import tensorflow as tf
import numpy as np
import math
 
w = tf.Variable(tf.random_normal([1, 127], stddev=0.35))
b = tf.Variable(tf.random_normal([1, 1], stddev=0.35))
x = tf.placeholder(tf.float32, shape=[127, None])
yl = tf.placeholder(tf.float32, shape=[1, None])
l2_loss = 0.001 * (tf.nn.l2_loss(w) + tf.nn.l2_loss(b))
with tf.device('/gpu:0'):
    y = tf.matmul(w, x) + b
loss = tf.add(l2_loss, tf.nn.l2_loss(y - yl))
train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

s = tf.Session(config=tf.ConfigProto(log_device_placement=True))
s.run(tf.global_variables_initializer())
 
#train_list = ["data.20130116", "data.20130117", "data.20130118", "data.20130121", "data.20130122"]
train_list =[
"data.20130104",	"data.20130107",	"data.20130108",	"data.20130110",	"data.20130111",	
"data.20130114",	"data.20130115",	"data.20130116",	"data.20130117",	"data.20130118",
"data.20130121",	"data.20130122",	"data.20130123",	"data.20130124",	"data.20130125",
"data.20130128",	"data.20130129",	"data.20130130"
]
batch=10000
for index in range(len(train_list)):
    print("=====>")
    print(train_list[index])
    d = np.genfromtxt(train_list[index], dtype='float32')
    for k in range(18):
        for i in range(int(len(d)/batch)):
            num = min(batch, len(d)- batch*i)
            if (0 == num):
               continue

            xv = []
            yv = []
            for g in range(num) :
               d1 = d[i*batch+g]
               xv = np.concatenate((xv, d1[:127]), axis= 0)
               yv = np.concatenate((yv, [d1[130]]), axis= 0)
            s.run(train_step, feed_dict={x: np.reshape(xv, [127,num]), yl: np.reshape(yv, [1, num])})
            if (0 == (i % 10)) :
                print(s.run(w))
                print(s.run(b)) 
 
print("evaluate==>")
print(s.run(w))
print(s.run(b)) 
#eval_list = ["data.20130123", "data.20130116", "data.20130117", "data.20130118", "data.20130121", "data.20130122"]
eval_list=train_list
for index in range(len(eval_list)):
    xx = 0.0
    xy = 0.0
    yy = 0.0
    d = np.genfromtxt(eval_list[index], dtype='float32')
    for i in range(len(d)):
        d1 = d[i]
        y_target = d1[130]
        y_actual = s.run(y, feed_dict={x: np.reshape(d1[:127], [127,1])})
        wt = min(d1[127], 2)
        xx += y_target**2 * wt
        xy += y_actual * y_target * wt
        yy += y_actual**2 * wt
    corr = xy/ math.sqrt(xx * yy)
    print(eval_list[index]+"===>")
    print(corr)
  
