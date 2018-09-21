
import main.cifar10_loader as cl
import main.cifar10_vgg16_model as cm

import tensorflow as tf

import numpy as np
import operator
import matplotlib.pyplot as plt

NUM_EPOCHS = 400
NUM_TRAINING_RECORDS = 50000
NUM_EVALUATION_RECORDS = 10000


def plot(values):
    plt.plot(np.squeeze(values))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("=== Costs ===")
    plt.show()


def cifar10_train_main():
    print("========================= train")

    #### tf.reset_default_graph()
    #### tf.set_random_seed(1)

    labels, images = cl.load_data_set(max_records=NUM_TRAINING_RECORDS)
    print("Read [%s] images and [%s] labels." % (len(images), len(labels)))
    var_names = ['conv1_test3', 'conv2_test3', 'conv3_test3', 'conv4_test3', 'fc1_test3', 'fc2_test3',
                 'softmax_linear_test3']
    input_placeholder, labels_placeholder = cm.create_placeholder()

    model = cm.forward_propagation(input=input_placeholder, variable_names=var_names)
    labels_onehot = tf.one_hot(labels_placeholder, cm.NUM_CLASSES)
    cost = cm.compute_cost(model, labels_onehot)
    optimizer = cm.backward_propagation(cost)

    saver = tf.train.Saver()
    print("======================= training model")
    costs = []

    with tf.Session() as session:
        #### session.run(tf.global_variables_initializer())
        saver.restore(session, '/tmp/model/-980')
        num_batches = int(NUM_TRAINING_RECORDS/cm.BATCH_SIZE)
        for epoch in range(NUM_EPOCHS):
            epoch_cost = 0.
            for i in range(num_batches):
                _, minibatch_cost = session.run([optimizer, cost],
                                            feed_dict={input_placeholder: images[cm.BATCH_SIZE*i:cm.BATCH_SIZE+cm.BATCH_SIZE*i],
                                                       labels_placeholder: labels[cm.BATCH_SIZE*i:cm.BATCH_SIZE+cm.BATCH_SIZE*i]})

                epoch_cost += minibatch_cost

            print("Convolution result: ", minibatch_cost, "for epoch ", epoch)
            if epoch == 0 or epoch % 20 == 0:
                costs.append(epoch_cost / NUM_EPOCHS)
                saver.save(session, '/tmp/model/', global_step=epoch)

                predicted_classes = []
                for i in range(num_batches):
                    predictions = model.eval({input_placeholder: images[cm.BATCH_SIZE*i:cm.BATCH_SIZE+cm.BATCH_SIZE*i],
                                          labels_placeholder: labels[cm.BATCH_SIZE*i:cm.BATCH_SIZE+cm.BATCH_SIZE*i]})

                    for prediction in predictions:
                        predicted_class = prediction.tolist().index(max(prediction))
                        predicted_classes.append(predicted_class)

                accuracy = 100 - sum([0 if x == 0 else 1 for x in
                                      list(map(operator.sub, predicted_classes, labels))])*100/len(predicted_classes)
                print("Accuracy: %f%%" % accuracy)
            print("========================= train")

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


if __name__ == '__main__':
    cifar10_train_main()
