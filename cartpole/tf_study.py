import tensorflow as tf
import numpy as np

# tf.global_variables_initializer()
# tf.placeholder(dtype, shape=None, name=None)
# tf.gradients(y,[xs])
# tf.train.Saver.save(sess, seve_path, global_step=None,...)
# saver.restore(sess, 'checkpoints/name_of_the_checkpoint')
x=2
y=3
op1 = tf.add(x,y)
op2 =tf.multiply(x,y)
op3 = tf.pow(op2,op1)
with tf.Session() as sess:
    op3 = sess.run(op3)
print(op3)
x1=tf.constant([[1]])
with tf.Session() as sess:
    x1=sess.run(x1)
print(x1)
w1 = tf.Variable(0,name='w1')
s1 = tf.placeholder(tf.float32)
s2 = tf.placeholder(tf.float32)
out = tf.multiply(s1,s2)
with tf.Session() as sess:
    output = sess.run(out, feed_dict={s1:[7.0], s2:[8.0]})
    print(output)
saver = tf.train.Saver()