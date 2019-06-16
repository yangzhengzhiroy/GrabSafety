# -*- coding: UTF-8 -*-
"""
This module prepares the input names and gender label.
"""
import os
import pickle
import logging
import numpy as np
import pandas as pd
from copy import deepcopy
from tensorflow.keras.preprocessing.sequence import pad_sequences
from grabsafety import PARENT_DIR


logger = logging.getLogger(__name__)


class KalmanFilter(object):
    """ Simple Multi-variate Kalman Filter. """
    def __init__(self, dof=6):

        self.dof = dof
        self.A = np.eye(dof)
        self.H = np.eye(dof)
        self.B = 0
        self.Q = np.zeros(shape=(dof, dof))
        self.R = np.eye(dof) / 50
        self.P = np.eye(dof)
        self.x = np.zeros((dof, 1))

    def predict(self, u=0):
        self.x = np.dot(self.A, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        intermediate_mat = np.dot(self.P, self.H.T)
        S = self.R + np.dot(self.H, intermediate_mat)
        K = np.dot(intermediate_mat, np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)


class TripRecord(object):
    """ Encode the name list into encoded char-to-int 2-D numpy array. """
    _scaler_file_name = 'safety_scaler.pkl'
    _scaler_path = os.path.join(PARENT_DIR, 'grabsafety/models', _scaler_file_name)

    def __init__(self, pad_size=3600, padding='post'):
        self._pad_size = pad_size
        self._padding = padding

    @staticmethod
    def measurement_change(df, new_col, value_col, time_col):
        """ Get the unit time sensor reading change. """
        df[new_col] = (df[value_col] - df[value_col].shift(1)) / df[time_col]
        return df

    @staticmethod
    def col_rearrange(df):
        """ Re-arrange the columns to map to correct coordinates. """
        df_copy = deepcopy(df)
        axis = df['axis'].values[0]
        axis_sign = df['axis_sign'].values[0]
        if axis_sign == 1:
            if axis == 'x':
                df['acceleration_x'] = -df_copy['acceleration_y']
                df['acceleration_y'] = df_copy['acceleration_x']
                df['gyro_x'] = -df_copy['gyro_y']
                df['gyro_y'] = df_copy['gyro_x']
            elif axis == 'z':
                df['acceleration_y'] = df_copy['acceleration_z']
                df['acceleration_z'] = -df_copy['acceleration_y']
                df['gyro_y'] = df_copy['gyro_z']
                df['gyro_z'] = -df_copy['gyro_y']
        else:
            if axis == 'x':
                df['acceleration_x'] = df_copy['acceleration_y']
                df['acceleration_y'] = -df_copy['acceleration_x']
                df['gyro_x'] = df_copy['gyro_y']
                df['gyro_y'] = -df_copy['gyro_x']
            elif axis == 'y':
                df['acceleration_x'] = -df_copy['acceleration_x']
                df['acceleration_y'] = -df_copy['acceleration_y']
                df['gyro_x'] = -df_copy['gyro_x']
                df['gyro_y'] = -df_copy['gyro_y']
            else:
                df['acceleration_x'] = -df_copy['acceleration_x']
                df['acceleration_y'] = -df_copy['acceleration_z']
                df['acceleration_z'] = -df_copy['acceleration_y']
                df['gyro_x'] = -df_copy['gyro_x']
                df['gyro_y'] = -df_copy['gyro_z']
                df['gyro_z'] = -df_copy['gyro_y']
        return df

    @staticmethod
    def high_pass_filter(series, f_cutoff, b):
        """ high-pass filter to remove gravity. """
        series_len = len(series)
        if series_len >= 9:
            b_tmp = 4 / (series_len - 1)
            b = b if b >= b_tmp else b_tmp
            bw = int(np.ceil(4 / b))
            bw = bw if bw % 2 else bw + 1
            n = np.arange(bw)
            h = np.sinc(2 * f_cutoff * (n - (bw - 1) / 2))
            w = np.blackman(bw)
            h *= w
            h = h / np.sum(h)
            h = -h
            h[(bw - 1) // 2] += 1
            s = np.convolve(series, h, mode='same')
            s = s[:series_len]
            return pd.Series(s, index=series.index)
        else:
            return series

    @staticmethod
    def low_pass_filter(series, f_cutoff, b):
        """ low-pass filter to remove noise. """
        series_len = len(series)
        if series_len >= 9:
            b_tmp = 4 / (series_len - 1)
            b = b if b >= b_tmp else b_tmp
            bw = int(np.ceil(4 / b))
            bw = bw if bw % 2 else bw + 1
            n = np.arange(bw)
            h = np.sinc(2 * f_cutoff * (n - (bw - 1) / 2))
            w = np.blackman(bw)
            h *= w
            h = h / np.sum(h)
            s = np.convolve(series, h, mode='same')
            s = s[:series_len]
            return pd.Series(s, index=series.index)
        else:
            return series

    @staticmethod
    def kalman_filter(df, dof=6):
        """ data cleaning through kalman filter. """
        kalman = KalmanFilter(dof)
        arr = df.values
        output = []
        for item in arr:
            item = item.reshape((kalman.dof, 1))
            output.append(kalman.predict().flatten())
            kalman.update(item)
        return pd.DataFrame(output, index=df.index, columns=df.columns)

    def trip_clean(self, df):
        """ Clean the input name string. """
        try:
            if not isinstance(df, pd.DataFrame):
                logger.error('Input trip data is not in pd.DataFrame format.')
                pass
            df = deepcopy(df)

            # Sort the order.
            df = df.sort_values(by=['bookingID', 'second'])
            # Remove invalid entries.
            df = df[df['Speed'] != -1]
            # Adjust the time to start from 0.
            df['second'] = df.groupby(by='bookingID')['second'].apply(lambda x: x - x.min())
            # Remove unreasonable records.
            df = df[df['second'] < 36000]
            # Set y as the vertical axis, identify which is corresponding the correct y axis.
            df_median = df.groupby(by='bookingID')[['acceleration_x', 'acceleration_y', 'acceleration_z']]. \
                median().reset_index()
            df_median = df_median.\
                assign(acceleration_x_abs=np.abs(np.abs(df_median['acceleration_x']) - 9.81),
                       acceleration_y_abs=np.abs(np.abs(df_median['acceleration_y']) - 9.81),
                       acceleration_z_abs=np.abs(np.abs(df_median['acceleration_z']) - 9.81))
            df_median['acceleration_abs_min'] = \
                df_median[['acceleration_x_abs', 'acceleration_y_abs', 'acceleration_z_abs']].min(axis=1)

            df_median['axis'] = np.where(df_median['acceleration_abs_min'] > 2, 'y',
                                         np.where(df_median['acceleration_y_abs'] == df_median['acceleration_abs_min'], 'y',
                                         np.where(df_median['acceleration_z_abs'] == df_median['acceleration_abs_min'],
                                                  'z', 'x')))
            df_median['axis_sign'] = \
                np.where((df_median['axis'] == 'y') & (df_median['acceleration_y'] > 1) |
                         (df_median['axis'] == 'z') & (df_median['acceleration_z'] > 1) |
                         (df_median['axis'] == 'x') & (df_median['acceleration_x'] > 1), 1, -1)
            df = df.merge(df_median[['bookingID', 'axis', 'axis_sign']], how='left', on='bookingID')
            df = df.groupby(by='bookingID').apply(lambda x: self.col_rearrange(x))
            # Get the time interval.
            df['next_second'] = df.groupby(by='bookingID')['second'].shift(-1)
            df['duration'] = df['next_second'] - df['second']
            df['duration'].fillna(0, inplace=True)
            # Get the previous second.
            df['prev_second'] = df.groupby(by='bookingID')['second'].shift(1)
            df.loc[pd.isnull(df['prev_second']), 'prev_second'] = df.loc[pd.isnull(df['prev_second']), 'second'] - 1
            df['last_duration'] = df['second'] - df['prev_second']
            # Get the unit bearing change.
            df = df.groupby(by='bookingID').apply(lambda x: self.measurement_change(x, 'bearing_change', 'Bearing', 'last_duration'))
            df['bearing_change'].fillna(0, inplace=True)
            df['bearing_change'] = np.abs(df['bearing_change'])
            df['bearing_change'] = np.where(df['bearing_change'] > 180, 360 - df['bearing_change'], df['bearing_change'])
            # Get the unit speed change.
            df = df.groupby(by='bookingID').apply(lambda x: self.measurement_change(x, 'speed_change', 'Speed', 'last_duration'))
            df['speed_change'].fillna(0, inplace=True)
            # High/low pass filter on acceleration.
            df['acceleration_x'] = df.groupby(by='bookingID')['acceleration_x'].\
                apply(lambda x: self.high_pass_filter(x, 1 / len(x), 0.1))
            df['acceleration_y'] = df.groupby(by='bookingID')['acceleration_y'].\
                apply(lambda x: self.high_pass_filter(x, 1 / len(x), 0.1))
            df['acceleration_z'] = df.groupby(by='bookingID')['acceleration_z'].\
                apply(lambda x: self.high_pass_filter(x, 1 / len(x), 0.1))
            df['acceleration_x'] = df.groupby(by='bookingID')['acceleration_x'].\
                apply(lambda x: self.low_pass_filter(x, 0.3, 0.1))
            df['acceleration_y'] = df.groupby(by='bookingID')['acceleration_y'].\
                apply(lambda x: self.low_pass_filter(x, 0.3, 0.1))
            df['acceleration_z'] = df.groupby(by='bookingID')['acceleration_z'].\
                apply(lambda x: self.low_pass_filter(x, 0.3, 0.1))
            # Kalman filter on angular velocity.
            df[['gyro_x', 'gyro_y', 'gyro_z']] = \
                df.groupby(by='bookingID')[['gyro_x', 'gyro_y', 'gyro_z']].apply(lambda x: self.kalman_filter(x, 3))
            # Total acceleration.
            df['acceleration'] = np.sqrt(np.square(df['acceleration_x']) + np.square(df['acceleration_y']) +
                                         np.square(df['acceleration_z']))
            # Additional features related to speed.
            df['speed_acceleration'] = df['Speed'] * df['acceleration']
            df['speed_bearing'] = df['Speed'] * df['bearing_change']
            df['speed_gyro_y'] = df['Speed'] * df['gyro_y']
            # Limit the record number.
            df = df.groupby(by='bookingID').head(self._pad_size)

            # Scaling of features.
            with open(self._scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            scaling_col = ['acceleration_x', 'acceleration_y', 'acceleration_z', 'gyro_x', 'gyro_y', 'gyro_z', 'Speed',
                           'bearing_change', 'speed_change', 'acceleration', 'speed_acceleration',
                           'speed_bearing', 'speed_gyro_y']
            df[scaling_col] = scaler.transform(df[scaling_col])
            feature_ls = scaling_col + ['duration']
            df = df.groupby(by='bookingID').apply(lambda x: x[feature_ls].values).rename('activity').reset_index()
            # Padding.
            input_arr = pad_sequences(df['activity'].values, maxlen=self._pad_size, dtype='float', padding=self._padding)
            return input_arr
        except (TypeError, AttributeError) as e:
            logger.exception(f'trip_clean error: {e}')
