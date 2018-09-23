
import cifar10.dataloader.cifar10_loader as cl
import cifar10.networking.cifar10_vgg16_model as cm
import cifar10.constants as constants

import tensorflow as tf

import numpy as np
import operator
import matplotlib.pyplot as plt
import os

NUM_EPOCHS = 400
NUM_TRAINING_RECORDS = 50000
NUM_EVALUATION_RECORDS = 10000
LEARNING_RATE = 0.0001
NUM_BATCHES = NUM_TRAINING_RECORDS // cm.BATCH_SIZE


def plot(values):
    plt.plot(np.squeeze(values))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("=== Costs ===")
    plt.show()


def initialize_variables_or_restore(session):
    saver = tf.train.Saver()

    if os.path.isfile(os.path.join(constants.MODEL_DIR_PATH, constants.MODEL_PREFIX + ".index")):
        model_file = os.path.join(constants.MODEL_DIR_PATH, constants.MODEL_PREFIX)
        saver.restore(session, model_file)
    else:
        # if there are not variables to load, then initialize them
        session.run(tf.global_variables_initializer())

    return saver


def evaluate():
    pass


def train(images, labels, input_placeholder, labels_placeholder, model, cost_function, train_function):
    print("======================= training model")
    costs = []

    with tf.Session() as session:
        initialize_variables_or_restore(session=session)

        for epoch in range(NUM_EPOCHS):
            epoch_cost = 0.
            for i in range(NUM_BATCHES):
                minibatch = images[cm.BATCH_SIZE * i:cm.BATCH_SIZE + cm.BATCH_SIZE * i]
                minibatch_labels = labels[cm.BATCH_SIZE * i:cm.BATCH_SIZE + cm.BATCH_SIZE * i]

                _, minibatch_cost = session.run([train_function, cost_function],
                                                feed_dict={input: minibatch, labels_placeholder: minibatch_labels})

                epoch_cost += minibatch_cost

            print("Convolution result: ", minibatch_cost, "for epoch ", epoch)
            if epoch == 0 or epoch % 20 == 0:
                costs.append(epoch_cost / NUM_EPOCHS)
                saver.save(session, '/tmp/model/', global_step=epoch)

                # TODO this needs to be changed
                predicted_classes = []
                for i in range(num_batches):
                    predictions = model.eval(
                        {input_placeholder: images[cm.BATCH_SIZE * i:cm.BATCH_SIZE + cm.BATCH_SIZE * i],
                         labels_placeholder: labels[cm.BATCH_SIZE * i:cm.BATCH_SIZE + cm.BATCH_SIZE * i]})

                    for prediction in predictions:
                        predicted_class = prediction.tolist().index(max(prediction))
                        predicted_classes.append(predicted_class)

                accuracy = 100 - sum([0 if x == 0 else 1 for x in
                                      list(map(operator.sub, predicted_classes, labels))]) * 100 / len(
                    predicted_classes)
                print("Accuracy: %f%%" % accuracy)
                # ======================
            print("========================= train")

        # TODO this needs to be changed ######################
        costs.append(epoch_cost / NUM_EPOCHS)
        saver.save(session, '/tmp/model/', global_step=epoch)
        predicted_classes = []
        for i in range(num_batches):
            predictions = model.eval({input_placeholder: images[cm.BATCH_SIZE * i:cm.BATCH_SIZE + cm.BATCH_SIZE * i],
                                      labels_placeholder: labels[cm.BATCH_SIZE * i:cm.BATCH_SIZE + cm.BATCH_SIZE * i]})

            for prediction in predictions:
                predicted_class = prediction.tolist().index(max(prediction))
                predicted_classes.append(predicted_class)

        accuracy = 100 - sum([0 if x == 0 else 1 for x in
                              list(map(operator.sub, predicted_classes, labels))]) * 100 / len(predicted_classes)
        print("Accuracy: %f%%" % accuracy)
        # #####################################################
        # ===================================
        labels, images = cl.load_data_set(max_records=10000, evaluate=True)
        num_batches = int(NUM_EVALUATION_RECORDS / cm.BATCH_SIZE)

        predicted_classes = []
        for i in range(num_batches):
            predictions = model.eval(
                {input_placeholder: images[cm.BATCH_SIZE * i:cm.BATCH_SIZE + cm.BATCH_SIZE * i],
                 labels_placeholder: labels[cm.BATCH_SIZE * i:cm.BATCH_SIZE + cm.BATCH_SIZE * i]})

            for prediction in predictions:
                predicted_class = prediction.tolist().index(max(prediction))
                predicted_classes.append(predicted_class)

        accuracy = 100 - sum([0 if x == 0 else 1 for x in
                              list(map(operator.sub, predicted_classes, labels))]) * 100 / len(predicted_classes)
        print("Accuracy (evaluation): %f%%" % accuracy)

        plot(costs)


def cifar10_train_main():
    print("========================= train")
    tf.reset_default_graph()
    tf.set_random_seed(1)

    labels, images = cl.load_data_set(max_records=NUM_TRAINING_RECORDS)
    print("Read [%s] images and [%s] labels." % (len(images), len(labels)))

    input_placeholder, labels_placeholder = cm.create_placeholder()
    labels_onehot = tf.one_hot(labels_placeholder, cm.NUM_CLASSES)
    parameters = cm.initialize_parameters()

    logits = cm.forward_propagation(input=input_placeholder, parameters=parameters)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_onehot))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

    train(images, labels, input_placeholder, labels_placeholder, logits, cost, optimizer)


if __name__ == '__main__':
    cifar10_train_main()
