import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def current_directory():
    return os.path.dirname(os.path.realpath(__file__))

def weight_init(shape, name):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01), name=name)

def bias_init(name):
    return tf.Variable(tf.constant(0.0001, shape=[layers[name]]))

def model(X, w_h1, w_h2, w_h3, w_o, b_h1, b_h2, b_h3, dropout):
    with tf.name_scope("hidden_layer_1"):
        h1 = tf.nn.relu(tf.add(tf.matmul(X, w_h1), b_h1))
    with tf.name_scope("hidden_layer_2"):
        h1 = tf.nn.dropout(h1, dropout)
        h2 = tf.nn.relu(tf.add(tf.matmul(h1, w_h2), b_h2))
    with tf.name_scope("hidden_layer_3"):
        h2 = tf.nn.dropout(h2, dropout)
        h3 = tf.nn.relu(tf.add(tf.matmul(h2, w_h3), b_h3))
    with tf.name_scope("output_layer"):
        h3 = tf.nn.dropout(h3, dropout)
        return tf.matmul(h3, w_o)

# download the MNIST data in <current directory>/MNIST_data/
mnist_dataset   = input_data.read_data_sets(current_directory() + "/MNIST_data/", one_hot=True) #one_hot=True: one-hot-encoding

training_X      = mnist_dataset.train.images
training_Y      = mnist_dataset.train.labels
testing_X       = mnist_dataset.test.images
testing_Y       = mnist_dataset.test.labels

# define the size of the layers - how many neurons in each layer
layers = {
    "input"     : 784,
    "hidden1"   : 256,
    "hidden2"   : 128,
    "hidden3"   : 64,
    "output"    : 10 
}

X = tf.placeholder(tf.float32, [None, layers["input"]], name = "X")
Y = tf.placeholder(tf.float32, [None, layers["output"]], name = "Y")

w_h1 = weight_init([layers["input"], layers["hidden1"]], "w_h1")
w_h2 = weight_init([layers["hidden1"], layers["hidden2"]], "w_h2")
w_h3 = weight_init([layers["hidden2"], layers["hidden3"]], "w_h3")
w_o  = weight_init([layers["hidden3"], layers["output"]], "w_o")

tf.histogram_summary("w_h1_sum", w_h1)
tf.histogram_summary("w_h2_sum", w_h2)
tf.histogram_summary("w_h3_sum", w_h3)
tf.histogram_summary("w_o_sum", w_o)

b_h1 = bias_init("hidden1")
b_h2 = bias_init("hidden2")
b_h3 = bias_init("hidden3")

tf.histogram_summary("b_h1_sum", b_h1)
tf.histogram_summary("b_h2_sum", b_h2)
tf.histogram_summary("b_h3_sum", b_h3)

dropout = tf.placeholder(tf.float32, name = "dropout")

py_x = model(X, w_h1, w_h2, w_h3, w_o, b_h1, b_h2, b_h3, dropout)

with tf.name_scope("cost"):
    cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y)) 
    learning_rate   = 1e-3
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost_function)
    tf.scalar_summary("cost_function", cost_function)

with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(Y,1), tf.argmax(py_x, 1))
    acc_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.scalar_summary("accuracy", acc_op)

with tf.Session() as sess:
    writer = tf.train.SummaryWriter(current_directory() + "/logs/", sess.graph)
    merged = tf.merge_all_summaries()

    tf.initialize_all_variables().run() # initialize a session for running the graph

    # train the model
    for i in range(30):
        for start, end in zip(range(0, len(training_X), 128), range(128, len(training_X)+1, 128)):
            sess.run(train_op, feed_dict={X: training_X[start:end], Y: training_Y[start:end], dropout: 0.5})
        
        summary, train_accuracy = sess.run([merged, acc_op], feed_dict={X: testing_X, Y: testing_Y, dropout: 1.0})
        writer.add_summary(summary, i)
        print(str(i), "\t accuracy = ", str(train_accuracy))

     # test the model
    test_accuracy = sess.run(acc_op, feed_dict={X: testing_X, Y: testing_Y, dropout: 1.0})
    print("test data accuracy:", "{:.0%}".format(test_accuracy))