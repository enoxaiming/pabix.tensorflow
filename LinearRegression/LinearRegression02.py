# -*- coding: utf-8 -*-

import tensorflow as tf

#PlaceHolder를 통한 X,Y 변수 선언
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

#[1]은 값이 하나인 1차원 어레이를 이야기함
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

#hyphothesis node
hyphothesis = X * W + b

#cost / loss function
#reduce_mean = avearge
cost = tf.reduce_mean(tf.square(hyphothesis - Y))

#Cost minimize - Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

#Session 생성
sess = tf.Session()

#W,b라는 TensorFlow Variable를 사용하면 아래 코드를 선언해주어야함
sess.run(tf.global_variables_initializer())

#Feed_dict를 이용하여 변수 지정 가능
for step in range(2001):
    cost_val, W_val, b_val, _ = \
        sess.run([cost, W, b, train],
                 feed_dict={X: [1, 2, 3, 4, 5],
                            Y: [2.1, 3.1, 4.1, 5.1, 6.1]})
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)
