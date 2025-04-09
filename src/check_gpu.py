import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("\nGPU Available:", tf.config.list_physical_devices('GPU'))
print("\nGPU Device Name:", tf.test.gpu_device_name())
print("\nGPU Memory Growth:", tf.config.experimental.get_memory_growth(tf.config.list_physical_devices('GPU')[0]) if tf.config.list_physical_devices('GPU') else "No GPU available") 