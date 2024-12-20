import h5py


def load_data():
    with h5py.File('bird_data.hdf5', 'r') as hf:
        images = hf['data']['images'][...]
        labels = hf['data']['labels'][...]
        classes = hf['data']['classes'][...]

    return (images, labels, classes)
