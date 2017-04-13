# -*- coding: utf-8 -*-

import tensorflow as tf

#trainable variable
x_train = [1,2,3]
y_train = [1,2,3]

#[1]은 값이 하나인 1차원 어레이를 이야기함
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

#hyphothesis node
hyphothesis = x_train * W + b

#cost / loss function
#reduce_mean = avearge
cost = tf.reduce_mean(tf.square(hyphothesis - y_train))

#Cost minimize - Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)


sess = tf.Session()

#W,b라는 TensorFlow Variable를 사용하면 아래 코드를 선언해주어야함
sess.run(tf.global_variables_initializer())

for step in range(2001):
    # 학습이 일어남
    sess.run(train)
    if step % 20 == 0 :
        #출력
        print(step, sess.run(cost), sess.run(W), sess.run(b))