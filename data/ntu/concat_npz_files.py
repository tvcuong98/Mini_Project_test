import os
import os.path as osp
import numpy as np
import glob
import argparse
def concat_npz_files(save_path, evaluation, dataset_type, min_chunk_index, max_chunk_index):
    """
    Concatenate npz files saved in chunks without loading all data into memory.

    Args:
        save_path (str): The path where the npz files are saved.
        evaluation (str): The evaluation type ('CS' or 'CV').
        dataset_type (str): The type of the dataset ('train' or 'test').
        min_chunk_index (int): The minimum chunk index to process (inclusive).
        max_chunk_index (int): The maximum chunk index to process (inclusive).
    """
    file_pattern = osp.join(save_path, 'NTU60_%s_%s_chunk_*.npz' % (evaluation, dataset_type))
    files = sorted(glob.glob(file_pattern))

    x_list = []
    y_list = []

    for chunk_index in range(min_chunk_index, max_chunk_index + 1):
        file = osp.join(save_path, 'NTU60_%s_%s_chunk_%d.npz' % (evaluation, dataset_type, chunk_index))
        if file in files:
            data = np.load(file)
            x_list.append(data['x_%s' % dataset_type])
            y_list.append(data['y_%s' % dataset_type])

    x = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    return x, y

def delete_chunk_files(save_path, evaluation, dataset_type):
    """
    Delete chunk npz files.

    Args:
        save_path (str): The path where the npz files are saved.
        evaluation (str): The evaluation type ('CS' or 'CV').
        dataset_type (str): The type of the dataset ('train' or 'test').
    """
    file_pattern = osp.join(save_path, 'NTU60_%s_%s_chunk_*.npz' % (evaluation, dataset_type))
    files = glob.glob(file_pattern)
    for f in files:
        os.remove(f)
def concat_train_test_files(save_path, evaluation):
    """
    Concatenate train and test npz files.

    Args:
        save_path (str): The path where the npz files are saved.
        evaluation (str): The evaluation type ('CS' or 'CV').
    """
    train_file = osp.join(save_path, 'NTU60_%s_train.npz' % evaluation)
    test_file = osp.join(save_path, 'NTU60_%s_test.npz' % evaluation)

    train_data = np.load(train_file)
    test_data = np.load(test_file)

    x_train = train_data['x_train']
    y_train = train_data['y_train']

    x_test = test_data['x_test']
    y_test = test_data['y_test']

    # x = np.concatenate((x_train, x_test), axis=0)
    # y = np.concatenate((y_train, y_test), axis=0)

    return x_train, y_train,x_test,y_test


def main():
    save_path_cs = './NTU60_CS'  # The path where the npz files are saved
    save_path_cv = './NTU60_CV'
    concat_save_path = './'
    evaluations = ['CS']
    for evaluation in evaluations:
        if (evaluation=='CS'):
            save_path = save_path_cs
        elif (evaluation == 'CV'):
            save_path = save_path_cv
        # train_x, train_y = concat_npz_files(save_path, evaluation, 'train',0,10)  # please uncoomment
        # test_x, test_y = concat_npz_files(save_path, evaluation, 'test',0,10)  # please uncoomment

        # Save the concatenated data
        # np.savez(osp.join(concat_save_path, 'NTU60_%s_train.npz' % evaluation), x_train=train_x, y_train=train_y)  # please uncoomment
        # np.savez(osp.join(concat_save_path, 'NTU60_%s_test.npz' % evaluation), x_test=test_x, y_test=test_y)  # please uncoomment

        # # Delete the non-concat
        # delete_chunk_files(save_path, evaluation, 'train')
        # delete_chunk_files(save_path, evaluation, 'test')

        # Concatenate train and test files
        _x_train, _y_train,_x_test,_y_test = concat_train_test_files(concat_save_path, evaluation)
        np.savez(osp.join(concat_save_path, 'NTU60_%s.npz' % evaluation), x_train=_x_train, y_train=_y_train, x_test=_x_test,y_test=_y_test)


if __name__ == '__main__':
    main()
