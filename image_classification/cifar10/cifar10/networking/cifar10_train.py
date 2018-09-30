
import cifar10.dataloader.cifar10_loader as cl
import cifar10.networking.cifar10_vgg16_model as cm
import cifar10.constants as constants

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import os

NUM_EPOCHS = 400
NUM_TRAINING_RECORDS = 50000
NUM_EVALUATION_RECORDS = 10000
LEARNING_RATE = 0.0001
NUM_BATCHES = NUM_TRAINING_RECORDS // cm.BATCH_SIZE


def plot(values, xname, yname, title):
    plt.plot(np.squeeze(values))
    plt.ylabel(yname)
    plt.xlabel(xname)
    plt.title(title)
    plt.show()


def initialize_variables_or_restore(session, saver):

    if os.path.isfile(os.path.join(constants.MODEL_DIR_PATH, constants.MODEL_PREFIX + ".index")):
        print("Initializing parameters with pre-trained model [%s]: " % (constants.MODEL_PREFIX + ".index"))
        model_file = os.path.join(constants.MODEL_DIR_PATH, constants.MODEL_PREFIX)
        saver.restore(session, model_file)
    else:
        # if there are not variables to load, then initialize them
        session.run(tf.global_variables_initializer())

    return saver


def evaluate(session, accuracy, input_placeholder, labels_holder, images, labels):
    accuracy_value = 0
    for i in range(NUM_BATCHES):
        minibatch = images[cm.BATCH_SIZE * i:cm.BATCH_SIZE + cm.BATCH_SIZE * i]
        minibatch_labels = labels[cm.BATCH_SIZE * i:cm.BATCH_SIZE + cm.BATCH_SIZE * i]

        accuracy_value += session.run([accuracy],
                                      feed_dict={input_placeholder: minibatch, labels_holder: minibatch_labels})[0]
    return accuracy_value/NUM_BATCHES


def evaluate_with_evaluation_data_set(session, accuracy, input_placeholder, labels_holder):
    labels, images = cl.load_data_set(evaluate=True)
    return evaluate(session, accuracy, input_placeholder, labels_holder, images, labels)


def train(images, labels, x_placeholder, y_placeholder, cost_fnc, train_fnc, accuracy_fnc):
    print("======================= training model")
    costs_values = []
    accuracy_values = []

    saver = tf.train.Saver()

    with tf.Session() as session:
        initialize_variables_or_restore(session=session, saver=saver)

        for epoch in range(NUM_EPOCHS):
            epoch_cost = 0.
            for i in range(NUM_BATCHES):
                minibatch = images[cm.BATCH_SIZE * i:cm.BATCH_SIZE + cm.BATCH_SIZE * i]
                minibatch_labels = labels[cm.BATCH_SIZE * i:cm.BATCH_SIZE + cm.BATCH_SIZE * i]

                _, minibatch_cost = session.run([train_fnc, cost_fnc],
                                                feed_dict={x_placeholder: minibatch,
                                                           y_placeholder: minibatch_labels})

                epoch_cost += minibatch_cost

            if epoch == 0 or epoch % 20 == 0:
                saver.save(session, constants.MODEL_DIR_PATH, global_step=epoch)
                accuracy_train = evaluate(session, accuracy_fnc, x_placeholder, y_placeholder,
                                                images, labels)
                accuracy_values.append(accuracy_train)
                print("Accuracy (training):  %f and cost %f for epoch %f"
                      % (accuracy_train, (epoch_cost / NUM_BATCHES), epoch))

            costs_values.append(epoch_cost / NUM_BATCHES)
        saver.save(session, constants.MODEL_DIR_PATH, global_step=epoch)

        accuracy_eval = evaluate_with_evaluation_data_set(session, accuracy_fnc, x_placeholder, y_placeholder)
        print("Accuracy (evaluation): %f" % accuracy_eval)

        return costs_values, accuracy_values


def cifar10_train_main():

    print("========================= train")
    tf.reset_default_graph()
    tf.set_random_seed(1)

    labels, images = cl.load_data_set()
    print("Read [%s] images and [%s] labels." % (len(images), len(labels)))

    x_placeholder, y_placeholder = cm.create_placeholder()
    y_onehot_matrix = tf.one_hot(y_placeholder, cm.NUM_CLASSES)
    parameters = cm.initialize_parameters()

    logits = cm.forward_propagation(input=x_placeholder, parameters=parameters)
    cost_fnc = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_onehot_matrix))
    train_fnc = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(cost_fnc)

    correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_placeholder)
    accuracy_fnc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    costs_values, accuracy_values = train(images, labels, x_placeholder, y_placeholder, cost_fnc, train_fnc, accuracy_fnc)

    plot(costs_values, 'values', 'cost', '==== cost ===')
    plot(accuracy_values, 'values', 'accuracy', '==== accuracy ===')


if __name__ == '__main__':
    cifar10_train_main()
