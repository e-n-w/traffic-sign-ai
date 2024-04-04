import tensorflow as tf

result = tf.config.experimental.list_physical_devices('GPU')
print(result)