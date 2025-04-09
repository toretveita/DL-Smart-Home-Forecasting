import tensorflow as tf
import numpy as np
import time

def print_gpu_info():
    """Print detailed GPU information"""
    print("\n=== GPU Information ===")
    print("TensorFlow version:", tf.__version__)
    print("\nGPU Devices:", tf.config.list_physical_devices('GPU'))
    print("GPU Device Name:", tf.test.gpu_device_name())
    
    # Get GPU memory info
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.get_memory_info(gpu)
                print(f"\nGPU Memory Info for {gpu}:")
                print(tf.config.experimental.get_memory_info(gpu))
        except RuntimeError as e:
            print("Error getting GPU memory info:", e)
    else:
        print("No GPU devices found!")

def test_gpu_performance():
    """Test GPU performance with matrix multiplication"""
    print("\n=== GPU Performance Test ===")
    
    # Create large matrices
    size = 5000
    print(f"\nCreating {size}x{size} matrices...")
    
    # Create random matrices
    matrix_a = np.random.rand(size, size).astype(np.float32)
    matrix_b = np.random.rand(size, size).astype(np.float32)
    
    # Convert to TensorFlow tensors
    tf_matrix_a = tf.convert_to_tensor(matrix_a)
    tf_matrix_b = tf.convert_to_tensor(matrix_b)
    
    # Test CPU performance
    print("\nTesting CPU performance...")
    start_time = time.time()
    with tf.device('/CPU:0'):
        result_cpu = tf.matmul(tf_matrix_a, tf_matrix_b)
    cpu_time = time.time() - start_time
    print(f"CPU Matrix Multiplication Time: {cpu_time:.2f} seconds")
    
    # Test GPU performance
    print("\nTesting GPU performance...")
    start_time = time.time()
    with tf.device('/GPU:0'):
        result_gpu = tf.matmul(tf_matrix_a, tf_matrix_b)
    gpu_time = time.time() - start_time
    print(f"GPU Matrix Multiplication Time: {gpu_time:.2f} seconds")
    
    # Calculate speedup
    speedup = cpu_time / gpu_time
    print(f"\nGPU Speedup: {speedup:.2f}x faster than CPU")

def test_gpu_training():
    """Test GPU performance with a simple neural network"""
    print("\n=== GPU Training Test ===")
    
    # Create a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(100,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    
    # Generate random data
    x_train = np.random.rand(1000, 100)
    y_train = np.random.rand(1000, 10)
    
    # Test training performance
    print("\nTraining on CPU...")
    with tf.device('/CPU:0'):
        start_time = time.time()
        model.fit(x_train, y_train, epochs=5, verbose=0)
        cpu_time = time.time() - start_time
    print(f"CPU Training Time: {cpu_time:.2f} seconds")
    
    print("\nTraining on GPU...")
    with tf.device('/GPU:0'):
        start_time = time.time()
        model.fit(x_train, y_train, epochs=5, verbose=0)
        gpu_time = time.time() - start_time
    print(f"GPU Training Time: {gpu_time:.2f} seconds")
    
    # Calculate speedup
    speedup = cpu_time / gpu_time
    print(f"\nGPU Training Speedup: {speedup:.2f}x faster than CPU")

def main():
    print("Starting GPU verification...")
    
    # Print GPU information
    print_gpu_info()
    
    # Test GPU performance
    test_gpu_performance()
    
    # Test GPU training
    test_gpu_training()
    
    print("\nGPU verification complete!")

if __name__ == "__main__":
    main() 