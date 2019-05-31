import numpy as np
import nibabel as nib

def load_nii_crop(vol_name): # load nii

    X = nib.load(vol_name).get_data()
    X = np.reshape(X, (1,) + X.shape + (1,))
    return X


def load_nii_by_name(vol_name, seg_name): # load nii

    X = nib.load(vol_name).get_data()
    X = np.reshape(X, X.shape + (1,))
    return_vals = [X]

    X_seg = nib.load(seg_name).get_data()
    return_vals.append(X_seg)

    return tuple(return_vals)
