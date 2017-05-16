import matplotlib
matplotlib.use('Agg') # Allows us to save figure on the server
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.lines as mlines
from sklearn.model_selection import train_test_split

'''
Implements a simple neural network to classify Doc2Vec feature vectors as document categories.
Adapted from Google's tensorflow tutorial
Architecture: Input layer -> hidden layer -> softmax layer (+ cross entropy loss for training)
'''

# Import training data
# Used pandas to import data instead of numpy since it is much faster

# Import vectors data through pandas as a numpy ndarray
vectors = pd.read_csv('inf_vecs.csv', delimiter=',', header=None).as_matrix()
print("loaded vectors")

# print vectors.shape
# print vectors[0]

# Import labels data through pandas as a numpy ndarray
labels = pd.read_csv('labels.csv', delimiter='\n', header=None).as_matrix()
labels = labels.flatten() # Flatten the ndarray to a one-dimensional vector (needed in the next step)
print("loaded labels")

# print labels.shape
# print labels

# Split data into training and testing set, with testing set being 20% of the data
print "Spliting data into training and testing sets"
trainX, testX, yTrainVals, yTestVals = train_test_split(vectors, labels, test_size=0.2)

# print testX.shape
# print yTestVals.shape


sizeInputVector = trainX.shape[1]  # Size of the input layer of the neural net and the input vectors
sizeOutputVector = 8  # Size of the output layer of the neural net and the y-label vectors

# Builds an array of one-hot y-label vectors for training data where the index of each 1 corresponds to the correct label
trainY = np.zeros((len(trainX), sizeOutputVector))  # Filling with zeros
trainY[np.arange(len(trainX)), yTrainVals - 1] = 1  # Inserting a 1 at the location corresponding to the correct label
# Subtracting 1 from yTrainVals because labels begin at 1 instead of 0
# print trainY.shape

# Builds an array of y-label vectors for testing data where the index of each 1 corresponds to the label
testY = np.zeros((len(testX), sizeOutputVector))  # Filling with zeros
testY[np.arange(len(testX)), yTestVals - 1] = 1  # Inserting a 1 at the location corresponding to the correct label
# Subtracting 1 from yTrainVals because labels begin at 1 instead of 0
# print testY.shape



# Create the Tensorflow model

# x is a Tensorflow placeholder array. The first dimension is None, meaning it can expand as needed. The second
# dimension is the size of the input vector to the neural network.
x = tf.placeholder(tf.float32, [None, sizeInputVector])

# W and b are Variables (modifiable tensors) that will hold the weights and biases of the neural network. The input
# layer and hidden layer are fully connected. So there will be sizeInputVector * sizeOutputVector weights.
W = tf.Variable(tf.zeros([sizeInputVector, sizeOutputVector]))  # Representing weights as a variable 2D array
b = tf.Variable(tf.zeros([sizeOutputVector]))  # Representing biases as a variable 1D array

# The output of the network is the dot product of the input, x, with the weights, W. This is added element-wise to
# the biases, b.
y = tf.matmul(x, W) + b

# This will hold the true labels of the inputs (truth-value output).
y_ = tf.placeholder(tf.float32, [None, sizeOutputVector])

# Using the cross entropy cost function averaged over all the samples in each batch. We apply the softmax function
# on the unnormalized outputs to normalize them then compute the cost.
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# Each training step is supposed to minimize the cross_entropy loss using the gradient descent algorithm with a
# learning rate of 0.5.
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


nEpochs = [5, 10, 15, 25, 50]  # list of epochs to train the neural network on
batchSize = 30  # Number of samples in each batch
numBatches = int(len(trainX) / batchSize)  # Number of batches in each epoch

# Run for each epoch count
for epoch in nEpochs:

    trainAccuracies = [None] * epoch  # Vector to store the training accuracies
    testAccuracies = [None] * epoch  # Vector to store the testing accuracies

    # This launches the model in a Session
    sess = tf.InteractiveSession()

    tf.global_variables_initializer().run()

    # Checks whether or not the networks prediction, tf.argmax(y, 1), matches the correct prediction, tf.argmax(y_, 1)
    # tf.argmax gives the index of the highest entry in a tensor along some axis
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

    # To determine what fraction of the neural net's predictions are correct, we cast the correct_prediction vector to
    # floating point numbers and then take the mean.
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print "\n\nTraining Neural Net with", epoch, "epochs.\n\n"

    # Train the neural network
    for i in range(epoch):

        # Generate a random permutation of trainX and trainY while maintaining their relative order
        p = np.random.permutation(len(trainX))  # Generate a permutation of the numbers from 0 to len(trainX)
        xData = trainX[p]  # Shuffle trainX using the indices p
        yData = trainY[p]  # Shuffle trainY using the indices p

        # Iterate over the number of batches
        for m in range(0, numBatches):
            currBatchX = xData[m * batchSize: m * batchSize + batchSize]  # Extract batch X data
            currBatchY = yData[m * batchSize: m * batchSize + batchSize]  # Extract batch Y data

            # Train the neural net for a single step using the current batch's X and Y data
            sess.run(train_step, feed_dict={x: currBatchX, y_: currBatchY})

        # Calculate training accuracy after each epoch
        train_accuracy = accuracy.eval(feed_dict={x: trainX, y_: trainY})  # Evaluate training accuracy
        print("Epoch %d, training accuracy %g" % (i, train_accuracy))
        trainAccuracies[i] = train_accuracy

        # Calculate testing accuracy after each epoch
        test_accuracy = accuracy.eval(feed_dict={x: testX, y_: testY})  # Evaluate testing accuracy
        print("Epoch %d, testing accuracy %g" % (i, test_accuracy))
        testAccuracies[i] = test_accuracy

    sess.close()


    plt.plot(trainAccuracies, 'bo')  # Plot training accuracies as blue circles
    plt.plot(trainAccuracies, 'b--')  # Plot training accuracies as blue dashed lines
    plt.plot(testAccuracies, 'ro')  # Plot testing accuracies as red circles
    plt.plot(testAccuracies, 'r--')  # Plot training accuracies as red dashed lines
    plt.ylabel('Accuracy')  # Accuracy label
    plt.xlabel('Epoch')  # Epoch label


    plt.yticks(np.arange(0, 1.1, 0.05))

    # Lines for the legends
    train_line = mlines.Line2D([], [], linestyle='--', color='blue', marker='o', markersize=5, label='Training Data')
    test_line = mlines.Line2D([], [], linestyle='--', color='red', marker='o', markersize=5, label='Testing Data')

    # Draw legend at the upper left corner with legend lines
    plt.legend(handles=[train_line, test_line], loc='lower right')
    plt.title('Plot of Testing and Training Accuracies | Simple Neural Net')
    ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    midXLoc = (xmax - xmin) / 2
    midYLoc = ymax - (ymax - ymin) / 2
    topYLoc = ymax - (ymax - ymin) * 0.025
    plt.text(midXLoc, midYLoc, 'Num Epochs: ' + str(epoch) + '\nBatch Size: ' + str(batchSize), ha='center', va='top')
    # plt.show()  # Show plot
    plt.savefig(str(epoch) + 'epochs_simpleNetAccuracies', dpi=300)
    plt.close()