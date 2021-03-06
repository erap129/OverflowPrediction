import sys

sys.path.append('..')
import os
from copy import deepcopy
from braindecode.datasets.bbci import BBCIDataset
from braindecode.datasets.bcic_iv_2a import BCICompetition4Set2A
from braindecode.datautil.signal_target import SignalAndTarget
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from EEGNAS.data.TUH.TUH_loader import DiagnosisSet, create_preproc_functions, TrainValidSplitter
from EEGNAS.data.netflow.netflow_data_utils import preprocess_netflow_data, turn_netflow_into_classification, \
    get_time_freq, \
    turn_dataset_to_timefreq, get_whole_netflow_data, get_handover_locations
from EEGNAS.utilities.config_utils import set_default_config, set_params_by_dataset
from EEGNAS.utilities.data_utils import split_sequence, noise_input, split_parallel_sequences, export_data_to_file, \
    EEG_to_TF_mne, EEG_to_TF_matlab, sktime_to_numpy, set_global_vars_by_sktime, set_global_vars_by_dataset
from EEGNAS.utilities.misc import concat_train_val_sets, unify_dataset, MOABB_DATASETS
from EEGNAS.data.MTS_benchmarks.MTS_utils import Data_utility
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from collections import OrderedDict
from braindecode.mne_ext.signalproc import mne_apply, resample_cnt
from braindecode.datautil.splitters import split_into_two_sets
from braindecode.datautil.signalproc import bandpass_cnt, exponential_running_standardize, highpass_cnt
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne
from sklearn.model_selection import train_test_split, KFold
from moabb.datasets import Cho2017, BNCI2014004
from moabb.paradigms import (LeftRightImagery, MotorImagery,
                             FilterBankMotorImagery)
import torchvision
import torchvision.transforms as transforms
import scipy.io
from EEGNAS.data.Bloomberg.bloomberg_preproc import get_bloomberg
from EEGNAS.data.HumanActivity.human_activity_preproc import get_human_activity
from EEGNAS.data.Opportunity.sliding_window import sliding_window
from sklearn import preprocessing
import pickle
from EEGNAS import global_vars
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
log = logging.getLogger(__name__)
log.setLevel('DEBUG')


def makeDummySignalTargets(X_train, y_train, X_val, y_val, X_test, y_test):
    train_set = DummySignalTarget(X_train, y_train)
    valid_set = DummySignalTarget(X_val, y_val)
    test_set = DummySignalTarget(X_test, y_test)
    return train_set, valid_set, test_set


def get_ch_names():
    train_filename = 'A01T.gdf'
    test_filename = 'A01E.gdf'
    train_filepath = os.path.join('../data/BCI_IV/', train_filename)
    test_filepath = os.path.join('../data/BCI_IV/', test_filename)
    train_label_filepath = train_filepath.replace('.gdf', '.mat')

    train_loader = BCICompetition4Set2A(train_filepath, labels_filename=train_label_filepath)
    train_cnt = train_loader.load()
    return train_cnt.ch_names


def get_bci_iv_2a_train_val_test(data_folder,subject_id, low_cut_hz):
    ival = [-500, 4000]  # this is the window around the event from which we will take data to feed to the classifier
    high_cut_hz = 38  # cut off parts of signal higher than 38 hz
    factor_new = 1e-3  # ??? has to do with exponential running standardize
    init_block_size = 1000  # ???

    train_filename = 'A{:02d}T.gdf'.format(subject_id)
    test_filename = 'A{:02d}E.gdf'.format(subject_id)
    train_filepath = os.path.join(data_folder, train_filename)
    test_filepath = os.path.join(data_folder, test_filename)
    train_label_filepath = train_filepath.replace('.gdf', '.mat')
    test_label_filepath = test_filepath.replace('.gdf', '.mat')

    train_loader = BCICompetition4Set2A(train_filepath, labels_filename=train_label_filepath)
    test_loader = BCICompetition4Set2A(test_filepath, labels_filename=test_label_filepath)
    train_cnt = train_loader.load()
    test_cnt = test_loader.load()

    train_cnt = train_cnt.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'])
    if len(train_cnt.ch_names) > 22:
        train_cnt = train_cnt.drop_channels(['STI 014'])
    assert len(train_cnt.ch_names) == 22

    # convert measurements to millivolt
    train_cnt = mne_apply(lambda a: a * 1e6, train_cnt)
    train_cnt = mne_apply(  # signal processing procedure that I don't understand
        lambda a: bandpass_cnt(a, low_cut_hz, high_cut_hz, train_cnt.info['sfreq'],
                               filt_order=3,
                               axis=1), train_cnt)
    train_cnt = mne_apply(  # signal processing procedure that I don't understand
        lambda a: exponential_running_standardize(a.T, factor_new=factor_new,
                                                  init_block_size=init_block_size,
                                                  eps=1e-4).T,
        train_cnt)

    test_cnt = test_cnt.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'])
    if len(test_cnt.ch_names) > 22:
        test_cnt = test_cnt.drop_channels(['STI 014'])
    assert len(test_cnt.ch_names) == 22

    # convert measurements to millivolt
    test_cnt = mne_apply(lambda a: a * 1e6, test_cnt)
    test_cnt = mne_apply(
        lambda a: bandpass_cnt(a, low_cut_hz, high_cut_hz, test_cnt.info['sfreq'],
                               filt_order=3,
                               axis=1), test_cnt)
    test_cnt = mne_apply(
        lambda a: exponential_running_standardize(a.T, factor_new=factor_new,
                                                  init_block_size=init_block_size,
                                                  eps=1e-4).T,
        test_cnt)

    marker_def = OrderedDict([('Left Hand', [1]), ('Right Hand', [2],),
                              ('Foot', [3]), ('Tongue', [4])])

    train_set = create_signal_target_from_raw_mne(train_cnt, marker_def, ival)
    test_set = create_signal_target_from_raw_mne(test_cnt, marker_def, ival)
    train_set, valid_set = split_into_two_sets(
        train_set, first_set_fraction=1 - global_vars.get('valid_set_fraction'))

    return train_set, valid_set, test_set


def load_bbci_data(filename, low_cut_hz, debug=False):
    load_sensor_names = None
    if debug:
        load_sensor_names = ['C3', 'C4', 'C2']
    # we loaded all sensors to always get same cleaning results independent of sensor selection
    # There is an inbuilt heuristic that tries to use only EEG channels and that definitely
    # works for datasets in our paper
    loader = BBCIDataset(filename, load_sensor_names=load_sensor_names)

    log.info("Loading data...")
    cnt = loader.load()

    # Cleaning: First find all trials that have absolute microvolt values
    # larger than +- 800 inside them and remember them for removal later
    log.info("Cutting trials...")

    marker_def = OrderedDict([('Right Hand', [1]), ('Left Hand', [2],),
                              ('Rest', [3]), ('Feet', [4])])
    clean_ival = [0, 4000]

    set_for_cleaning = create_signal_target_from_raw_mne(cnt, marker_def,
                                                  clean_ival)

    clean_trial_mask = np.max(np.abs(set_for_cleaning.X), axis=(1, 2)) < 800

    log.info("Clean trials: {:3d}  of {:3d} ({:5.1f}%)".format(
        np.sum(clean_trial_mask),
        len(set_for_cleaning.X),
        np.mean(clean_trial_mask) * 100))

    # now pick only sensors with C in their name
    # as they cover motor cortex
    C_sensors = ['FC5', 'FC1', 'FC2', 'FC6', 'C3', 'C4', 'CP5',
                 'CP1', 'CP2', 'CP6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2',
                 'C6',
                 'CP3', 'CPz', 'CP4', 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h',
                 'FCC5h',
                 'FCC3h', 'FCC4h', 'FCC6h', 'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h',
                 'CPP5h',
                 'CPP3h', 'CPP4h', 'CPP6h', 'FFC1h', 'FFC2h', 'FCC1h', 'FCC2h',
                 'CCP1h',
                 'CCP2h', 'CPP1h', 'CPP2h']
    if debug:
        C_sensors = load_sensor_names
    cnt = cnt.pick_channels(C_sensors)

    # Further preprocessings as descibed in paper
    log.info("Resampling...")
    cnt = resample_cnt(cnt, 250.0)
    log.info("Highpassing...")
    cnt = mne_apply(
        lambda a: highpass_cnt(
            a, low_cut_hz, cnt.info['sfreq'], filt_order=3, axis=1),
        cnt)
    log.info("Standardizing...")
    cnt = mne_apply(
        lambda a: exponential_running_standardize(a.T, factor_new=1e-3,
                                                  init_block_size=1000,
                                                  eps=1e-4).T,
        cnt)

    # Trial interval, start at -500 already, since improved decoding for networks
    ival = [-500, 4000]

    dataset = create_signal_target_from_raw_mne(cnt, marker_def, ival)
    dataset.X = dataset.X[clean_trial_mask]
    dataset.y = dataset.y[clean_trial_mask]
    return dataset


def load_train_valid_test_bbci(
        train_filename, test_filename, low_cut_hz, debug=False):
    log.info("Loading train...")
    full_train_set = load_bbci_data(
        train_filename, low_cut_hz=low_cut_hz, debug=debug)

    log.info("Loading test...")
    test_set = load_bbci_data(
        test_filename, low_cut_hz=low_cut_hz, debug=debug)
    valid_set_fraction = 0.8
    train_set, valid_set = split_into_two_sets(full_train_set,
                                               valid_set_fraction)

    log.info("Train set with {:4d} trials".format(len(train_set.X)))
    if valid_set is not None:
        log.info("Valid set with {:4d} trials".format(len(valid_set.X)))
    log.info("Test set with  {:4d} trials".format(len(test_set.X)))

    return train_set, valid_set, test_set


def get_hg_train_val_test(data_folder, subject_id, low_cut_hz):
    return load_train_valid_test_bbci(f"{data_folder}train/{subject_id}.mat",
                                      f"{data_folder}test/{subject_id}.mat",
                                        low_cut_hz)


class DummySignalTarget:
    def __init__(self, X, y, y_type=np.longlong):
        self.X = np.array(X, dtype=np.float32)
        if global_vars.get('autoencoder'):
            y_type = np.float32
        if global_vars.get('problem') == 'regression':
            self.y = np.array(y)
        else:
            self.y = np.array(y, dtype=y_type)


def get_ner_train_val_test(data_folder):
    X = np.load(f"{os.path.dirname(os.path.abspath(__file__))}/{data_folder}NER15/preproc/epochs.npy")
    y, User = np.load(f"{os.path.dirname(os.path.abspath(__file__))}/{data_folder}NER15/preproc/infos.npy")
    X_test = np.load(f"{os.path.dirname(os.path.abspath(__file__))}/{data_folder}NER15/preproc/test_epochs.npy")
    y_test = pd.read_csv(f"{os.path.dirname(os.path.abspath(__file__))}/{data_folder}NER15/preproc/true_labels.csv").values.reshape(-1)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=global_vars.get('valid_set_fraction'))
    train_set, valid_set, test_set = makeDummySignalTargets(X_train, y_train, X_val, y_val, X_test, y_test)
    return train_set, valid_set, test_set


def get_cho_train_val_test(subject_id):
    paradigm = LeftRightImagery()
    dataset = Cho2017()
    subjects = [subject_id]
    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=subjects)
    y = np.where(y=='left_hand', 0, y)
    y = np.where(y=='right_hand', 1, y)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.3) # test = 30%, same as paper
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                      test_size=global_vars.get('valid_set_fraction'))
    train_set, valid_set, test_set = makeDummySignalTargets(X_train, y_train, X_val, y_val, X_test, y_test)
    return train_set, valid_set, test_set


def get_moabb_train_val_test(subject_id):
    paradigm = MotorImagery()
    dataset_names = [type(dataset).__name__ for dataset in paradigm.datasets]
    dataset = paradigm.datasets[dataset_names.index(global_vars.get('dataset'))]
    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject_id])
    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    kf = KFold(n_splits=global_vars.get('n_folds'), shuffle=False)
    train_idxs, test_idxs = list(kf.split(list(range(len(X)))))[global_vars.get('test_fold_idx')]
    X_train_val = X[train_idxs]
    X_test = X[test_idxs]
    y_train_val = y[train_idxs]
    y_test = y[test_idxs]
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                      test_size=global_vars.get('valid_set_fraction'))

    train_set, valid_set, test_set = makeDummySignalTargets(X_train, y_train, X_val, y_val, X_test, y_test)
    return train_set, valid_set, test_set

def get_bci_iv_2b_train_val_test(subject_id):
    paradigm = LeftRightImagery()
    dataset = BNCI2014004()
    subjects = [subject_id]
    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=subjects)
    y = np.where(y=='left_hand', 0, y)
    y = np.where(y=='right_hand', 1, y)
    train_indexes = np.logical_or(metadata.values[:, 1] == 'session_0',
                                       metadata.values[:, 1] == 'session_1',
                                       metadata.values[:, 1] == 'session_2')
    test_indexes = np.logical_or(metadata.values[:, 1] == 'session_3',
                                       metadata.values[:, 1] == 'session_4')
    X_train_val = X[train_indexes]
    y_train_val = y[train_indexes]
    X_test = X[test_indexes]
    y_test = y[test_indexes]
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                      test_size=global_vars.get('valid_set_fraction'))
    train_set, valid_set, test_set = makeDummySignalTargets(X_train, y_train, X_val, y_val, X_test, y_test)
    return train_set, valid_set, test_set


def get_bloomberg_train_val_test(data_folder):
    X_train_val, y_train_val, X_test, y_test = get_bloomberg(f'{os.path.dirname(os.path.abspath(__file__))}/{data_folder}/Bloomberg')
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                      test_size=global_vars.get('valid_set_fraction'))
    train_set, valid_set, test_set = makeDummySignalTargets(X_train, y_train, X_val, y_val, X_test, y_test)
    return train_set, valid_set, test_set


def get_nyse_train_val_test(data_folder):
    df = pd.read_csv(f"{os.path.dirname(os.path.abspath(__file__))}/{data_folder}/NYSE/prices-split-adjusted.csv", index_col=0)
    df["adj close"] = df.close  # Moving close to the last column
    df.drop(['close'], 1, inplace=True)  # Moving close to the last column
    df2 = pd.read_csv(f"{os.path.dirname(os.path.abspath(__file__))}/{data_folder}/NYSE/fundamentals.csv")
    symbols = list(set(df.symbol))

    df = df[df.symbol == 'GOOG']
    df.drop(['symbol'], 1, inplace=True)
    min_max_scaler = preprocessing.MinMaxScaler()
    df['open'] = min_max_scaler.fit_transform(df.open.values.reshape(-1, 1))
    df['high'] = min_max_scaler.fit_transform(df.high.values.reshape(-1, 1))
    df['low'] = min_max_scaler.fit_transform(df.low.values.reshape(-1, 1))
    df['volume'] = min_max_scaler.fit_transform(df.volume.values.reshape(-1, 1))
    df['adj close'] = min_max_scaler.fit_transform(df['adj close'].values.reshape(-1, 1))
    amount_of_features = len(df.columns)  # 5
    data = df.as_matrix()
    sequence_length = 200 + 1  # index starting from 0
    result = []

    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    result = np.array(result)
    row = round(0.7 * result.shape[0])  # 90% split
    train = result[:int(row), :]  # 90% date, all features

    X_train_val = train[:, :-1]
    y_train_val = np.array([i > j for i,j in zip(train[:, -1][:, -1], train[:, -2][:, -1])]).astype(int) # did the price go up?

    X_test = result[int(row):, :-1]
    y_test = np.array([i > j for i,j in zip(result[int(row):, -1][:, -1], result[int(row):, -2][:, -1])]).astype(int)

    X_train_val = np.reshape(X_train_val, (X_train_val.shape[0], amount_of_features, X_train_val.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], amount_of_features, X_test.shape[1]))
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                      test_size=0.2)
    train_set, valid_set, test_set = makeDummySignalTargets(X_train, y_train, X_val, y_val, X_test, y_test)
    return train_set, valid_set, test_set


def get_human_activity_train_val_test(data_folder, subject_id):
    X_train_val, y_train_val, X_test, y_test = get_human_activity(f'{os.path.dirname(os.path.abspath(__file__))}/{data_folder}/HumanActivity', subject_id)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                      test_size=global_vars.get('valid_set_fraction'))
    train_set, valid_set, test_set = makeDummySignalTargets(X_train, y_train, X_val, y_val, X_test, y_test)
    return train_set, valid_set, test_set


def get_mental_imagery(data_folder, sub_dataset, subject_id):
    dataset_files = [f for f in os.listdir(f'{os.path.dirname(os.path.abspath(__file__))}/{data_folder}MentalImagery/{sub_dataset}') if
                       os.path.isfile(f'{os.path.dirname(os.path.abspath(__file__))}/{data_folder}MentalImagery/{sub_dataset}/{f}')]
    selected_file = ''
    for subj_file in dataset_files:
        if f'sub_{subject_id}' in subj_file:
            selected_file = f'{os.path.dirname(os.path.abspath(__file__))}/{data_folder}MentalImagery/{sub_dataset}/{subj_file}'
    data = scipy.io.loadmat(selected_file)['eeg_data_wrt_task_rep_no_eog_256Hz_last_beep']
    X = []
    y = []
    cls_idx = 0
    for cls_examples in data:
        for example in cls_examples:
            X.append(example[:64])
            y.append(cls_idx)
        cls_idx += 1
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=
                                                      global_vars.get('valid_set_fraction'))
    train_set, valid_set, test_set = makeDummySignalTargets(X_train, y_train, X_val, y_val, X_test, y_test)
    return train_set, valid_set, test_set


def opp_sliding_window(data_x, data_y, ws, ss):
    data_x = sliding_window(data_x,(ws,data_x.shape[1]),(ss,1))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y,ws,ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)


def get_opportunity_train_val_test(data_folder):
    with open(f'{os.path.dirname(os.path.abspath(__file__))}/{data_folder}/Opportunity/oppChallenge_gestures.data', 'rb') as pickle_file:
        data = pickle.load(pickle_file)
        X_train, y_train = data[0]
        X_test, y_test = data[1]
        X_test, y_test = opp_sliding_window(X_test, y_test, 128, 12)
        X_train_val, y_train_val = opp_sliding_window(X_train, y_train, 128, 12)
        X_test = np.swapaxes(X_test, 1, 2)
        X_train_val = np.swapaxes(X_train_val, 1, 2)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                          test_size=global_vars.get('valid_set_fraction'))
        train_set, valid_set, test_set = makeDummySignalTargets(X_train, y_train, X_val, y_val, X_test, y_test)
        return train_set, valid_set, test_set


def get_sonarsub_train_val_test(data_folder):
    X = np.load(f"{os.path.dirname(os.path.abspath(__file__))}/{data_folder}SonarSub/data_file_aug.npy")
    y = np.load(f"{os.path.dirname(os.path.abspath(__file__))}/{data_folder}SonarSub/labels_file_aug.npy")
    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=global_vars.get('valid_set_fraction'))
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=global_vars.get('valid_set_fraction'))
    train_set, valid_set, test_set = makeDummySignalTargets(X_train, y_train, X_val, y_val, X_test, y_test)
    return train_set, valid_set, test_set


def get_netflow_train_val_test(data_folder, shuffle=True, n_sequences=32):
    X = np.load(f"{os.path.dirname(os.path.abspath(__file__))}/{data_folder}netflow/X.npy")[:, :n_sequences]
    y = np.load(f"{os.path.dirname(os.path.abspath(__file__))}/{data_folder}netflow/y.npy")[:, :n_sequences]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=global_vars.get('valid_set_fraction'),
                                                        shuffle=shuffle)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=global_vars.get('valid_set_fraction'),
                                                      shuffle=shuffle)
    train_set, valid_set, test_set = makeDummySignalTargets(X_train, y_train, X_val, y_val, X_test, y_test)
    return train_set, valid_set, test_set


def get_netflow_asflow_train_val_test(data_folder, shuffle=False):
    if global_vars.get('netflow_subfolder'):
        subfolder_str = f"/{global_vars.get('netflow_subfolder')}"
    else:
        subfolder_str = ''
    file_paths = [f"{os.path.dirname(os.path.abspath(__file__))}/{data_folder}{global_vars.get('dataset')}{subfolder_str}/{ats}_" \
                  f"{global_vars.get('date_range')}.csv" for ats in global_vars.get('autonomous_systems')]
    X, y, _, _ = preprocess_netflow_data(file_paths, global_vars.get('input_height'), global_vars.get('steps_ahead'),
                                         global_vars.get('jumps'), global_vars.get('prediction_buffer'),
                                         global_vars.get('netflow_handover_locations'))
    global_vars.set('eeg_chans', X.shape[1])
    if global_vars.get('problem') != 'classification':
        global_vars.set('n_classes', global_vars.get('steps_ahead'))
        if global_vars.get('per_handover_prediction'):
            global_vars.set('n_classes', global_vars.get('n_classes') * X.shape[1])
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=global_vars.get('valid_set_fraction'), shuffle=shuffle)
    if global_vars.get('aggregate_netflow_sum'):
        y_train = np.sum(y_train, axis=1)
        y_test = np.sum(y_test, axis=1)
        global_vars.set('n_classes', 1)
    if global_vars.get('problem') == 'classification' and not global_vars.get('highest_handover_overflow'):
        global_vars.set('n_classes', 2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=global_vars.get('valid_set_fraction'),
                                                      shuffle=shuffle)
    train_set, valid_set, test_set = makeDummySignalTargets(X_train, y_train, X_val, y_val, X_test, y_test)
    return train_set, valid_set, test_set


def get_netflow_asflow_AE(data_folder, shuffle=True):
    dataset_netflow_path_pr = f'{os.path.dirname(os.path.abspath(__file__))}/{data_folder}netflow/asflowAE/as_flow_vol_final.csv'
    pd_df = pd.read_csv(dataset_netflow_path_pr, index_col=0).fillna(0)
    time_steps = pd_df.columns
    devs_id = pd_df.index
    my_data = pd_df.values.transpose()
    scaler = MinMaxScaler()
    my_data_scaled = scaler.fit_transform(my_data)
    X = split_sequence(my_data_scaled, global_vars.get('steps'))[0]
    X = X.swapaxes(1, 2)
    y = deepcopy(X)
    X = noise_input(X, devs_id)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=global_vars.get('valid_set_fraction'),
                                                        shuffle=shuffle)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=global_vars.get('valid_set_fraction'),
                                                      shuffle=shuffle)
    train_set, valid_set, test_set = makeDummySignalTargets(X_train, y_train, X_val, y_val, X_test, y_test)
    return train_set, valid_set, test_set


def get_cifar10(data_folder):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform)
    X_train = trainset.train_data
    y_train = trainset.train_labels
    X_test = testset.test_data
    y_test = testset.test_labels
    X_train = X_train.reshape(-1, 3, 32, 32)
    X_test = X_test.reshape(-1, 3, 32, 32)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=global_vars.get('valid_set_fraction'))
    train_set, valid_set, test_set = makeDummySignalTargets(X_train, y_train, X_val, y_val, X_test, y_test)
    return train_set, valid_set, test_set


def get_tuh_train_val_test(data_folder):
    preproc_functions = create_preproc_functions(
        sec_to_cut_at_start=global_vars.get('sec_to_cut_at_start'),
        sec_to_cut_at_end=global_vars.get('sec_to_cut_at_end'),
        duration_recording_mins=global_vars.get('duration_recording_mins'),
        max_abs_val=global_vars.get('max_abs_val'),
        clip_before_resample=global_vars.get('clip_before_resample'),
        sampling_freq=global_vars.get('sampling_freq'),
        divisor=global_vars.get('divisor'))

    test_preproc_functions = create_preproc_functions(
        sec_to_cut_at_start=global_vars.get('sec_to_cut_at_start'),
        sec_to_cut_at_end=global_vars.get('sec_to_cut_at_end'),
        duration_recording_mins=global_vars.get('test_recording_mins'),
        max_abs_val=global_vars.get('max_abs_val'),
        clip_before_resample=global_vars.get('clip_before_resample'),
        sampling_freq=global_vars.get('sampling_freq'),
        divisor=global_vars.get('divisor'))

    training_set = DiagnosisSet(n_recordings=global_vars.get('n_recordings'),
                                max_recording_mins=global_vars.get('max_recording_mins'),
                                preproc_functions=preproc_functions,
                                train_or_eval='train',
                                sensor_types=global_vars.get('sensor_types'))

    test_set = DiagnosisSet(n_recordings=global_vars.get('n_recordings'),
                            max_recording_mins=None,
                            preproc_functions=test_preproc_functions,
                            train_or_eval='eval',
                            sensor_types=global_vars.get('sensor_types'))
    X, y = training_set.load()
    global_vars.set('input_height', X[0].shape[1])
    splitter = TrainValidSplitter(10, i_valid_fold=0, shuffle=global_vars.get('shuffle'))
    train_set, valid_set = splitter.split(X, y)
    test_X, test_y = test_set.load()
    test_set = SignalAndTarget(test_X, test_y)
    train_set.X = np.array(train_set.X)
    valid_set.X = np.array(valid_set.X)
    test_set.X = np.array(test_set.X)
    return train_set, valid_set, test_set


def get_multivariate_ts(data_folder):
    dataset_name = global_vars.get('dataset')
    set_global_vars_by_sktime(f'{os.path.dirname(os.path.abspath(__file__))}/{data_folder}Multivariate_ts/'
                                       f'{dataset_name}/{dataset_name}_TRAIN.ts',
                              f'{os.path.dirname(os.path.abspath(__file__))}/{data_folder}Multivariate_ts/'
                                      f'{dataset_name}/{dataset_name}_TEST.ts')
    X_train, y_train = sktime_to_numpy(f'{os.path.dirname(os.path.abspath(__file__))}/{data_folder}Multivariate_ts/'
                                       f'{dataset_name}/{dataset_name}_TRAIN.ts')
    X_test, y_test = sktime_to_numpy(f'{os.path.dirname(os.path.abspath(__file__))}/{data_folder}Multivariate_ts/'
                                       f'{dataset_name}/{dataset_name}_TEST.ts')
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=global_vars.get('valid_set_fraction'))
    train_set, valid_set, test_set = makeDummySignalTargets(X_train, y_train, X_val, y_val, X_test, y_test)
    return train_set, valid_set, test_set


def get_MTS_benchmark_train_val_test(data_folder):
    Data = Data_utility(f'{os.path.dirname(os.path.abspath(__file__))}/{data_folder}MTS_benchmarks/'
                        f'{global_vars.get("dataset")}.txt', 0.6, 0.2, device='cpu', window=24*7, horizon=12)
    train_set, valid_set, test_set = makeDummySignalTargets(np.swapaxes(Data.train[0], 1, 2), Data.train[1],
                                                            np.swapaxes(Data.valid[0], 1, 2), Data.valid[1],
                                                            np.swapaxes(Data.test[0], 1, 2), Data.test[1])
    global_vars.set('eeg_chans', train_set.X.shape[1])
    global_vars.set('input_height', train_set.X.shape[2])
    global_vars.set('n_classes', train_set.y.shape[1])
    return train_set, valid_set, test_set


def get_data_from_npy(data_folder):
    X_train = np.load(f'{os.path.dirname(os.path.abspath(__file__))}/{data_folder}{global_vars.get("dataset")}/X_train.npy')
    X_test = np.load(f'{os.path.dirname(os.path.abspath(__file__))}/{data_folder}{global_vars.get("dataset")}/X_test.npy')
    y_train = np.load(f'{os.path.dirname(os.path.abspath(__file__))}/{data_folder}{global_vars.get("dataset")}/y_train.npy')
    y_test = np.load(f'{os.path.dirname(os.path.abspath(__file__))}/{data_folder}{global_vars.get("dataset")}/y_test.npy')
    global_vars.set('eeg_chans', X_train.shape[1])
    global_vars.set('input_height', X_train.shape[2])
    if X_train.ndim > 3:
        global_vars.set('input_width', X_train.shape[3])
    if global_vars.get('problem') == 'regression':
        global_vars.set('n_classes', y_train.shape[1])
    else:
        global_vars.set('n_classes', len(np.unique(y_train)))
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=global_vars.get('valid_set_fraction'))
    train_set, valid_set, test_set = makeDummySignalTargets(X_train, y_train, X_val, y_val, X_test, y_test)
    return train_set, valid_set, test_set


def get_pure_cross_subject(data_folder, exclude=[]):
    train_x = []
    val_x = []
    test_x = []
    train_y = []
    val_y = []
    test_y = []
    for subject_id in global_vars.get('subjects_to_check'):
        if subject_id not in exclude:
            train_set, valid_set, test_set = get_train_val_test(data_folder, subject_id)
            train_x.append(train_set.X)
            val_x.append(valid_set.X)
            test_x.append(test_set.X)
            train_y.append(train_set.y)
            val_y.append(valid_set.y)
            test_y.append(test_set.y)
    train_set, valid_set, test_set = makeDummySignalTargets(np.concatenate(train_x), np.concatenate(train_y),
                                                            np.concatenate(val_x), np.concatenate(val_y),
                                                            np.concatenate(test_x), np.concatenate(test_y))
    return train_set, valid_set, test_set


def get_leave_one_out(data_folder, test_subject_id):
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for subject_id in global_vars.get('subjects_to_check'):
        if subject_id != test_subject_id:
            dataset = get_dataset(subject_id)
            dataset = unify_dataset(dataset)
            X_train.extend(dataset.X)
            y_train.extend(dataset.y)
    test_dataset = get_dataset(test_subject_id)
    test_dataset = unify_dataset(test_dataset)
    X_test.extend(test_dataset.X)
    y_test.extend(test_dataset.y)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=global_vars.get('valid_set_fraction'))
    train_set, valid_set, test_set = makeDummySignalTargets(X_train, y_train, X_val, y_val, X_test, y_test)
    return train_set, valid_set, test_set


def get_train_val_test(data_folder, subject_id):
    if global_vars.get('dataset') == 'BCI_IV_2a':
        return get_bci_iv_2a_train_val_test(f"{os.path.dirname(os.path.abspath(__file__))}/{data_folder}BCI_IV/",
                                            subject_id, global_vars.get('low_cut_hz'))
    elif global_vars.get('dataset') == 'HG':
        return get_hg_train_val_test(f"{os.path.dirname(os.path.abspath(__file__))}/{data_folder}HG/", subject_id,
                                     global_vars.get('low_cut_hz'))
    elif global_vars.get('dataset') == 'NER15':
        return get_ner_train_val_test(data_folder)
    elif global_vars.get('dataset') == 'Cho':
        return get_cho_train_val_test(subject_id)
    elif global_vars.get('dataset') == 'BCI_IV_2b':
        return get_bci_iv_2b_train_val_test(subject_id)
    elif global_vars.get('dataset') == 'Bloomberg':
        return get_bloomberg_train_val_test(data_folder)
    elif global_vars.get('dataset') == 'NYSE':
        return get_nyse_train_val_test(data_folder)
    elif global_vars.get('dataset') == 'HumanActivity':
        return get_human_activity_train_val_test(data_folder, subject_id)
    elif global_vars.get('dataset') == 'Opportunity':
        return get_opportunity_train_val_test(data_folder)
    elif global_vars.get('dataset') == 'SonarSub':
        return get_sonarsub_train_val_test(data_folder)
    elif global_vars.get('dataset') == 'MentalImageryLongWords':
        return get_mental_imagery(data_folder, "LongWords", subject_id)
    elif global_vars.get('dataset') == 'MentalImageryVowels':
        return get_mental_imagery(data_folder, "Vowels", subject_id)
    elif global_vars.get('dataset') == 'TUH':
        return get_tuh_train_val_test(data_folder)
    elif global_vars.get('dataset') == 'netflow':
        return get_netflow_train_val_test(data_folder, n_sequences=global_vars.get('n_classes'))
    elif global_vars.get('dataset') in ['netflow_asflow', 'overflow_prediction']:
        return get_netflow_asflow_train_val_test(data_folder)
    elif global_vars.get('dataset') == 'netflow_asflowAE':
        return get_netflow_asflow_AE(data_folder)
    elif global_vars.get('dataset') == 'cifar10':
        return get_cifar10(data_folder)
    elif global_vars.get('dataset') in ["ArticularyWordRecognition", "AtrialFibrillation", "BasicMotions", "CharacterTrajectories", "Cricket", "DuckDuckGeese", "EigenWorms", "Epilepsy", "ERing", "EthanolConcentration", "FaceDetection", "FingerMovements", "HandMovementDirection", "Handwriting", "Heartbeat", "InsectWingbeat", "JapaneseVowels", "Libras", "LSST", "MotorImagery", "NATOPS", "PEMS-SF", "PenDigits", "PhonemeSpectra", "RacketSports", "SelfRegulationSCP1", "SelfRegulationSCP2", "SpokenArabicDigits", "StandWalkJump", "UWaveGestureLibrary"]:
        return get_multivariate_ts(data_folder)
    elif global_vars.get('dataset') in MOABB_DATASETS:
        return get_moabb_train_val_test(subject_id)
    elif global_vars.get('dataset') in ['solar', 'exchange_rate', 'traffic']:
        return get_MTS_benchmark_train_val_test(data_folder)
    else:
        return get_data_from_npy(data_folder)


def get_dataset(subject_id):
    dataset = {}
    if subject_id == 'all':
        dataset['train'], dataset['valid'], dataset['test'] = \
            get_pure_cross_subject(global_vars.get('data_folder'))
    else:
        dataset['train'], dataset['valid'], dataset['test'] =\
            get_train_val_test(global_vars.get('data_folder'), subject_id)
    if global_vars.get('time_frequency'):
        EEG_to_TF_mne(dataset)
        set_global_vars_by_dataset(dataset['train'])
    return dataset


if __name__ == '__main__':
    set_default_config('configurations/config.ini')
    for dataset in ["netflow_asflow"]:
        global_vars.set('dataset', dataset)

        if dataset == 'netflow_asflow':
            global_vars.set('autonomous_systems', [20940, 16509, 6185, 15133, 32934, 15169, 22822, 2906, 32590, 202818, 3356])
            global_vars.set('date_range', "1.7.2017-25.12.2019")
            global_vars.set('as_to_test', 20940)
            global_vars.set('start_hour', 15)
            global_vars.set('input_height', 240)
            global_vars.set('prediction_buffer', 2)
            global_vars.set('steps_ahead', 5)
            global_vars.set('jumps', 24)
            global_vars.set('max_handovers', 11)
            global_vars.set('normalize_netflow_data', True)
            global_vars.set('per_handover_prediction', True)
            global_vars.set('max_handovers', 43)

            # global_vars.set('interpolate_netflow', True)
            global_vars.set('subjects_to_check', [1])
            # global_vars.set('netflow_importance_path', '/home/user/Documents/eladr/netflowinsights/CDN_overflow_prediction/feature_importances/regular')


        # set_params_by_dataset('configurations/dataset_params.ini')
        # file_paths = [f"{os.path.dirname(os.path.abspath(__file__))}/data/netflow/{ats}_" \
        #               f"{global_vars.get('date_range')}.csv" for ats in global_vars.get('autonomous_systems')]
        # for AS_name, file_path in zip(global_vars.get('autonomous_systems'), file_paths):
        #     all_data = get_whole_netflow_data(file_path)
        #     all_data.to_csv(f'data/export_data/{global_vars.get("dataset")}/raw/{AS_name}.csv')

        file_paths = [f"data/netflow/{ats}_" \
                          f"{global_vars.get('date_range')}.csv" for ats in global_vars.get('autonomous_systems')]
        global_vars.set('netflow_handover_locations', get_handover_locations(file_paths))
        dataset = get_dataset('all')
        concat_train_val_sets(dataset)
        export_data_to_file(dataset, 'numpy', f'data/export_data/{global_vars.get("dataset")}_samelocs_2', transpose_time=False, unify=False)



