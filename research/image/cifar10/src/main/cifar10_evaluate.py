
import main.cifar10_vgg16_model as cm
import main.cifar10_loader as cl

import os

import tensorflow as tf

import operator

NUM_TRAINING_RECORDS = 10000

CURRENT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

MODEL_FILE_PREFIX = os.path.join(CURRENT_DIR_PATH, '../../resources/model/-980')

print("current dir %s" % CURRENT_DIR_PATH)

def cifar10_evaluate_main():
    var_names = ['conv1_test3', 'conv2_test3', 'conv3_test3', 'conv4_test3', 'fc1_test3', 'fc2_test3',
                 'softmax_linear_test3']
    input_placeholder, labels_placeholder = cm.create_placeholder()

    model = cm.forward_propagation(input=input_placeholder, variable_names=var_names)

    saver = tf.train.Saver()

    labels, images = cl.load_data_set(max_records=50000, evaluate=True)

    with tf.Session() as sess:
        # Restore variables from disk.
        # saver.restore(sess, "/tmp/model/-980")
        saver.restore(sess, MODEL_FILE_PREFIX)
        num_batches = int(NUM_TRAINING_RECORDS / cm.BATCH_SIZE)


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
        print("Accuracy: %f%%" % accuracy)


if __name__ == '__main__':
    cifar10_evaluate_main()
