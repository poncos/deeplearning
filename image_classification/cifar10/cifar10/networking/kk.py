import tensorflow as tf


x_placeholder = tf.placeholder(tf.float32, [None, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH], name=INPUT_PLACEHOLDER)

correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), labels_placeholder)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

