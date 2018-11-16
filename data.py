# -*- coding: UTF-8 -*-
# author = kidozh

import pandas as pd
import numpy as np
import csv

class FixedIntervalDataset(object):
    # extract all data
    cache_dir_path = '/home/kidozh/PycharmProjects/nonalignment_signal_tool_wear/.cache/'
    total_signal_num = 315
    # only 1,4 and 6 is correct
    sample_label = [1, 4, 6]
    sample_loc = 1

    @property
    def sample_data_storage(self):
        return 'all_sample_dat_num_%s' % (self.sample_num)

    @property
    def res_data_storage(self):
        return 'all_res_dat_num_%s' % (self.sample_num)

    @property
    def res_data_path(self):
        return '/home/kidozh/PycharmProjects/nonalignment_signal_tool_wear/c%s_wear.csv' % (self.sample_loc)

    def __init__(self, force_update=False, sample_num=5000):
        self.sample_num = sample_num
        self.force_update = force_update
        pass

    def get_sample_csv_path(self, num):
        return '/home/kidozh/PycharmProjects/nonalignment_signal_tool_wear/c%s/c_%s_%03d.csv' % (self.sample_loc, self.sample_loc, num)

    def get_signal_data_by_pandas(self, num):
        try:
            return pd.read_csv(self.get_sample_csv_path(num), header=None)
        except Exception as e:
            print(e)
            print("Error ! csv is not correct %s, try python parser instead."%(num))
            return pd.read_csv(self.get_sample_csv_path(num), header=None,engine='python')

    @property
    def get_res_data_in_numpy(self):
        # remove cache because it's not needed
        res_csv_data = self.get_res_data_by_pandas
        res_array = np.array([np.array(i).reshape(3) for i in res_csv_data.values])
        # np.save(storage_path, res_array)
        return res_array

    @property
    def get_res_data_by_pandas(self):
        return pd.read_csv(self.res_data_path, index_col='cut')

    def gen_x_batch_by_num(self, num):
        pd_data = self.get_signal_data_by_pandas(num)
        print('Retreive data from %s' % (self.get_sample_csv_path(num)))
        # print(np.array(pd_data.values).shape)
        # reduce sample freq for accelerating speed
        # secondary sampling

        interval = 100

        print('# Num %s, total %s, interval %s, that should be %s' % (num, len(pd_data.values), interval,len(pd_data.values)// interval))


        return np.array(pd_data.values[::interval])

    def get_all_sample_data(self):
        storage_path = self.cache_dir_path + self.sample_data_storage
        res_dat = []
        for i in range(1, self.total_signal_num + 1):
            res_dat.append(self.gen_x_batch_by_num(i))

        # res_dat = np.array(res_dat)
        # np.save(storage_path, res_dat)
        return res_dat

    def get_all_loc_y_sample_data(self):
        storage_path = self.cache_dir_path + self.res_data_storage
        if not self.force_update:
            try:

                print('#' * 20)
                res_dat = np.load(storage_path + '.npy')
                return res_dat
            except Exception as e:
                print('Ohhh, sample cache is not found or destroyed. Reason lists as following.')
                print('-' * 20)
                print(e)
                print('-' * 20)

        self.sample_loc = 1
        y_dat = self.get_res_data_in_numpy
        for i in [4, 6]:
            print('Fetch %s' % (i))
            self.sample_loc = i
            y_dat = np.append(self.get_res_data_in_numpy, y_dat, axis=0)

        print(y_dat.shape)

        print('-' * 40)
        print('Your computer may become very slow to run, please keep nothing util computer start to respond.')
        print('-' * 40)

        np.save(storage_path, y_dat)
        return y_dat

    def get_all_loc_x_sample_data(self):
        storage_path = self.cache_dir_path + self.sample_data_storage
        if not self.force_update:
            try:
                print('# Attention !')
                print('#' * 20)
                res_dat = np.load(storage_path + '.npy')
                return res_dat
            except Exception as e:
                print('Ohhh, sample cache is not found or destroyed. Reason lists as following.')
                print('-' * 20)
                print(e)
                print('-' * 20)

        self.sample_loc = 1
        x_dat = self.get_all_sample_data()
        for i in [4, 6]:
            self.sample_loc = i
            print('Fetch %s' % (i))
            x_dat = np.append(self.get_all_sample_data(), x_dat, axis=0)

        print('-' * 40)
        print('Your computer may become very slow to run, please keep nothing util computer start to respond.')
        print('-' * 40)

        print(x_dat.shape)

        np.save(storage_path, x_dat)
        return x_dat

    def get_duration_data(self):
        rul_path = 'rul_data'
        storage_path = self.cache_dir_path + rul_path
        if not self.force_update:
            try:
                print('# Attention !')
                print('#' * 20)
                res_dat = np.load(storage_path + '.npy')
                return res_dat
            except Exception as e:
                print('Ohhh, sample cache is not found or destroyed. Reason lists as following.')
                print('-' * 20)
                print(e)
                print('-' * 20)
        save_array = np.zeros([self.total_signal_num * 3])
        cnt = -1

        for i in [1, 4, 6]:
            cnt += 1
            self.sample_loc = i
            # read data from dataset
            for j in range(1, self.total_signal_num + 1):
                pd_data = self.get_signal_data_by_pandas(j)

                period_time = 1 / (50e3)

                duration = len(pd_data.values)

                print(i, j, duration)

                save_array[cnt * self.total_signal_num + j - 1] = period_time * duration

        np.save(storage_path, save_array)
        return save_array


if __name__ == "__main__":
    # wavelet_dat = wavelet_dataset()
    # rul_dat = wavelet_dat.get_rul_dat()
    # print(rul_dat)

    fixData = FixedIntervalDataset()
    fixData.get_all_loc_x_sample_data()
    fixData.get_all_loc_y_sample_data()

    # rul = rul_data_source()
    # train_x, train_y = rul.get_data()
    # print(train_x.shape, train_y.shape)

    # a = allDataSet(force_update=True)
    # y = a.get_all_loc_y_sample_data()
    # x = a.get_all_loc_x_sample_data()
