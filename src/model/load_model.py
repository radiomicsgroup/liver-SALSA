import os

def load_model(model_dir):
    """
    Placeholder for compatibility – nnU-Net handles model loading internally
    """

    if os.path.exists(model_dir):
        return model_dir
    else:
        print('Model weights are not where they are supposed to be :(')
        return none