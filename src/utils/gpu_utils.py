import tensorflow as tf

def check_gpu_availability():
    """
    Check if a GPU is available for use.

    Returns:
        bool: True if a GPU is available, False otherwise.
    """
    physical_devices = tf.config.list_physical_devices('GPU')
    return len(physical_devices) > 0

def configure_gpu():
    """
    Configure TensorFlow to use the GPU if available.
    """
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)