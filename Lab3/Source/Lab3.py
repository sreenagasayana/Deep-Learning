# import TensorFlow and MNIST dataset
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/Users/jayantis/Desktop/MNIST_data", one_hot=True)

# Create placeholder for 2-D tensor, x
x = tf.placeholder(tf.float32, [None, 784])

# Initialize weight and bias variables with tensors of all 0s
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Define and implement softmax (multinomial linear) regression model
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Create new placeholder for correct values
y_ = tf.placeholder(tf.float32, [None, 10])

# Implement cross-entropy function to compare correct and predicted answers and find the model's loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# Ask TensorFlow to minimize this loss
train_step = tf.train.AdamOptimizer(5.0).minimize(cross_entropy)

# Launch interactive session in which we'll run the model
sess = tf.InteractiveSession()
writer= tf.summary.FileWriter("/Users/jayantis/Desktop/graphs",sess.graph)


# Initialize variables so they can be used within session
tf.global_variables_initializer().run()

# Run the training step 1000 times
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # Finally assign values to our placeholders, x and y_
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Compute accuracy of model predictions
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))