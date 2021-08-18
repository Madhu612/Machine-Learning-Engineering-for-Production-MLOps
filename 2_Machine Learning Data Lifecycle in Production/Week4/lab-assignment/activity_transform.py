
import tensorflow as tf
import tensorflow_transform as tft

import activity_constants
import importlib

importlib.reload(activity_constants)

# Unpack the contents of the constants module
_INT_FEATURES = activity_constants.INT_FEATURES
_FLOAT_FEATURES = activity_constants.FLOAT_FEATURES
_LABEL_KEY = activity_constants.LABEL_KEY
_transformed_name = activity_constants.transformed_name


# Define the transformations
def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.
    Args:
        inputs: map from feature keys to raw not-yet-transformed features.
    Returns:
        Map from string feature key to transformed feature operations.
    """
    outputs = {}

    outputs[_transformed_name(_LABEL_KEY)] = tft.compute_and_apply_vocabulary(inputs[_LABEL_KEY],vocab_filename=_LABEL_KEY)

    for key in _FLOAT_FEATURES:
        outputs[_transformed_name(key)] = tft.scale_by_min_max(inputs[key])

    return outputs
