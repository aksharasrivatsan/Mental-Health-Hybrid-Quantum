import tensorflow as tf
import tensorflow_datasets as tfds

def load_crema_d_splits():
    """
    Loads CREMA-D with pre-defined speaker-independent splits.
    Returns: train, validation, and test datasets.
    """
    print("Loading CREMA-D dataset...")
    
    train_ds, val_ds, test_ds = tfds.load(
        'crema_d', 
        split=['train', 'validation', 'test'], 
        as_supervised=True
    )
    
    print(f"Loaded {len(train_ds)} Training samples")
    print(f"Loaded {len(val_ds)} Validation samples")
    print(f"Loaded {len(test_ds)} Test samples")
    
    return train_ds, val_ds, test_ds

if __name__ == "__main__":
    train, val, test = load_crema_d_splits()
