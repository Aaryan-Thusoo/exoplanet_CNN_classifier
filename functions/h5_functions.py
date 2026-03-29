import h5py
import numpy as np

def h5_keys(data_path):
    with h5py.File(data_path, "r") as h5:
        print("Top-level keys:")
        print(list(h5.keys()))

def num_h5_keys(data_path):
    with h5py.File(data_path, "r") as h5:
        return len(list(h5.keys()))

def specific_h5_key(data_path, kic):

    with h5py.File(data_path, "r") as h5:
        if kic in h5:
            data = h5[kic][:]
            print("Shape:", data.shape)
            print("Dtype:", data.dtype)
            print("First 10 values:", data[:10])
        else:
            print("KIC not found in file.")

def print_h5_structure(data_path):
    def print_h5(name, obj):
        print(name, "|", type(obj))

    with h5py.File(data_path, "r") as h5:
        print("\nFull file structure:")
        h5.visititems(print_h5)


def pull_h5_data(data_path, kic):
    with h5py.File(data_path, "r") as h5:
        if kic in h5:
            data = h5[kic][:]
        else:
            raise Exception("KIC not found in file.")
    return data

def load_kic_noise_dict(h5_path):
    """
    Load an HDF5 noise library into a dictionary.

    Parameters
    ----------
    h5_path : str
        Path to the .h5 file.

    Returns
    -------
    dict[str, np.ndarray]
        Keys are KIC IDs (strings), values are noise arrays.
    """
    noise_dict = {}

    with h5py.File(h5_path, "r") as h5:
        for kic in h5.keys():
            noise_dict[str(kic)] = h5[kic][:]  # [:] converts to NumPy array

    return noise_dict

def create_h5_file(path, kic_dict):
    with h5py.File(path, "w") as h5:
        for key, value in kic_dict.items():
            h5.create_dataset(key, data=value)