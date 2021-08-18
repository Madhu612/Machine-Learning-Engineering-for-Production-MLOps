
# Numerical features that are marked as continuous
INT_FEATURES = ['user_id', 'timestamp']

# Feature that can be grouped into buckets
FLOAT_FEATURES = ['x-acc', 'y-acc', 'z-acc']

# Feature that the model will predict
LABEL_KEY = 'activity'

# Utility function for renaming the feature
def transformed_name(key):
    return key + '_xf'
