import numpy as np
np.random.seed(42)

from scipy.io import savemat, loadmat

import utils, data_paths

default_data_path = data_paths.train_data_path
val_path = data_paths.val_data_path


class DataLoader(object):
    def __init__(self, data_path=default_data_path):
        self.data_path = data_path
        data = loadmat(self.data_path)
        self.data = {
            'train': data.get('train', None),
            'val': data.get('val', None),
            'test': data.get('test', None),
        }

    def get_data(self, name):
        return self.data[name]

    def get_num_examples(self, name):
        return self.get_data(name).shape[0]

    def get_num_batches(self, name, batch_size):
        num_examples = self.get_num_examples(name)
        return (num_examples) // batch_size

    def shuffle_data(self, name):
        perm = np.random.permutation(self.data[name].shape[0])
        self.data[name] = self.data[name][perm]

    def get_batch_data(self, name, batch_size, batch_index=0, shuffle=False, norm=False, norm_axis=2):
        num_batches = self.get_num_batches(name, batch_size)
        if batch_index >= num_batches:
            raise Exception('batch_index %d exceeds the total batch count %d' % (batch_index, num_batches))

        if shuffle:
            self.shuffle_data(name)

        data = self.get_data(name)

        start_id = batch_index * batch_size
        end_id = min(start_id + batch_size, self.get_num_examples(name))

        batch_data = data[start_id: end_id, :17]

        if norm:
            batch_data = utils.unit_norm(batch_data, axis=norm_axis)
        return batch_data

    @property
    def train_data(self):
        return self.get_data('train')

    @property
    def test_data(self):
        return self.get_data('test')

    @property
    def num_train_examples(self):
        return self.get_num_examples('train')

    @property
    def num_test_examples(self):
        return self.get_num_examples('test')

    def num_train_batches(self, batch_size):
        return self.get_num_batches('train', batch_size)

    def num_test_batches(self, batch_size):
        return self.get_num_batches('test', batch_size)

    def get_train_data_batch(self, batch_size, batch_index=0, shuffle=False, norm=False):
        return self.get_batch_data('train', batch_size, batch_index, shuffle, norm)

    def get_test_data_batch(self, batch_size, batch_index=0, shuffle=False, norm=False):
        return self.get_batch_data('test', batch_size, batch_index, shuffle, norm)

    def get_uniform_batch(self, batch_size, ndim, low=-1, high=1):
        return np.random.uniform(low, high, (batch_size, ndim))

    def get_clipped_normal_batch(self, batch_size, ndim, mu=0.0, sigma=0.27, low=-1, high=1):
        return np.clip(np.random.normal(mu, sigma, (batch_size, ndim)), low, high)

    def get_random_batch(self, batch_size, ndim, label='normal'):
        if label == 'normal':
            return self.get_clipped_normal_batch(batch_size, ndim)
        elif label == 'uniform':
            return self.get_uniform_batch(batch_size, ndim)
        else:
            raise Exception('DataLoader.get_random_batch: Invalid Label: %s' % label)


############################################################################
##### RUN THIS ONLY ONCE TO MAKE SURE TEST AND TRAIN DONT GET MIXED UP #####
def combine_all_data(data_path):
    from dataset import cmu_dataset, mpi_dataset, ytube_dataset
    cmu_data = cmu_dataset.get_sk_fit_mat_data(key='pred_17_local')
    mpi_data = mpi_dataset.get_sk_fit_mat_data(key='pred_17_local')
    ytb_data = ytube_dataset.get_sk_fit_mat_data(key='pred_17_local')

    bundles = [cmu_data, mpi_data, ytb_data]

    data = np.array([joints for bundle in bundles for joints_seq in bundle for joints in joints_seq])

    n_data = data.shape[0]

    random_permutation = np.random.permutation(n_data)

    shuffled_data = data[random_permutation]

    train_ratio = 0.8
    n_train = int(n_data * train_ratio)

    train_data = shuffled_data[:n_train]
    test_data = shuffled_data[n_train:]

    savemat(data_path, {
        'train': train_data,
        'test': test_data
    })


def get_data_loader(label='default'):
    if label == 'default':
        return DataLoader()
    elif label == 'val':
        return DataLoader(val_path)
    else:
        raise Exception('Invalid Label: %s for DataLoader' % label)
