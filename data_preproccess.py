import pickle
import numpy as np
from mmsdk import mmdatasdk


AUDIO = "b'COAVAREP'"
VISUAL = "b'FACET 4.2'"
TEXT = "b'glove_vectors'"
TARGET = "b'All Labels'"


if __name__ == '__main__':
    dataset = mmdatasdk.mmdataset('data')
    tensors = dataset.get_tensors(seq_len=50, non_sequences=["b'All Labels'"], direction=False, folds=[mmdatasdk.cmu_mosei.standard_folds.standard_train_fold, mmdatasdk.cmu_mosei.standard_folds.standard_valid_fold, mmdatasdk.cmu_mosei.standard_folds.standard_test_fold])
    
    tensors_ = []
    for data in tensors:
        del_indices = []
        for i, sample in enumerate(data[AUDIO]):
            if np.isinf(sample).any() or np.isnan(sample).any():
                del_indices.append(i)
        for elem in [AUDIO, VISUAL, TEXT, TARGET]:
            data[elem] = np.delete(data[elem], del_indices, axis=0)
        tensors_.append(data)
    with open('data/mosei', 'wb') as file:
        pickle.dump(tensors_, file)
    