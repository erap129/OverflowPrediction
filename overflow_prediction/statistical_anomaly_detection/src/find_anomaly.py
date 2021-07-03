import json
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.pyplot import figure
from configparser import ConfigParser

from numpy import array


def sliding_chunker(data, window_len, slide_len):
    """
    Split a list into a series of sub-lists, each sub-list window_len long,
    sliding along by slide_len each time. If the list doesn't have enough
    elements for the final sub-list to be window_len long, the remaining data
    will be dropped.
    e.g. sliding_chunker(range(6), window_len=3, slide_len=2)
    gives [ [0, 1, 2], [2, 3, 4] ]
    """
    chunks = []
    for pos in range(0, len(data), slide_len):
        chunk = np.copy(data[pos:pos + window_len])
        if len(chunk) != window_len:
            continue
        chunks.append(chunk)

    return chunks


def reconstruct(data, window, clusterer):
    """
    Reconstruct the given data using the cluster centers from the given
    clusterer.
    """
    window_len = len(window)
    slide_len = int(window_len / 2)
    segments = sliding_chunker(data, window_len, slide_len)
    reconstructed_data = np.zeros(len(data))
    for segment_n, segment in enumerate(segments):
        # window the segment so that we can find it in our clusters which were
        # formed from windowed data
        segment = (segment * window)
        nearest_match_idx = clusterer.predict([segment])[0]
        nearest_match = np.copy(clusterer.cluster_centers_[nearest_match_idx])

        pos = segment_n * slide_len
        reconstructed_data[pos:pos + window_len] += nearest_match

    return reconstructed_data


def find_anomaly(as_id, data, error_data, output_path, folder_name, segment_len, slide_len, n_plot_samples, n_clusters):

    # sorting by ts
    df = pd.DataFrame(data, index=['ts', 'vol']).T
    df = df.sort_values(by='ts')
    data_samp = df['vol']

    segments = sliding_chunker(data_samp, segment_len, slide_len)

    print("Produced %d waveform segments" % len(segments))
    window, windowed_segments = windowing(segment_len, segments)

    clusterer = clustering(data_samp, n_clusters, segment_len, segments, windowed_segments)

    # ------------------------Reconstruction from Clusters-----------------
    data_samp_anomalous = np.copy(data_samp)
    reconstruction = reconstruct(data_samp_anomalous, window, clusterer)
    error = reconstruction - data_samp_anomalous

    error_data[str(as_id)] = {'avg error': np.average(np.absolute(error)), 'max error': error.max()}
    error_98th_percentile = np.percentile(error, 98)
    print("Maximum reconstruction error was %.1f" % error.max())
    print("98th percentile of reconstruction error was %.1f" % error_98th_percentile)
    plot_results(as_id, data_samp_anomalous, error, folder_name, n_plot_samples, output_path, reconstruction)


def plot_results(as_id, data_samp_anomalous, error, folder_name, n_plot_samples, output_path, reconstruction):
    figure(num=None, figsize=(20, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(data_samp_anomalous[0:n_plot_samples], label="Original Data")
    plt.plot(reconstruction[0:n_plot_samples], label="Reconstructed Data")
    plt.plot(error[0:n_plot_samples], label="Reconstruction Error")
    plt.legend()
    plt.title("AS id: " + str(as_id) + ", From 01.01.2019 to 16.06.2019")
    plt.savefig(output_path + folder_name + '/AS_' + str(as_id))
    plt.show()


def windowing(segment_len, segments):
    window_rads = np.linspace(0, np.pi, segment_len)
    window = np.sin(window_rads) ** 2
    windowed_segments = []
    for segment in segments:
        windowed_segment = np.copy(segment) * window
        windowed_segments.append(windowed_segment)
    return window, windowed_segments


def clustering(data_samp, n_clusters, segment_len, segments, windowed_segments):
    # len(segments) should be >= n_clusters
    try:
        clusterer = KMeans(n_clusters)
        clusterer.fit(windowed_segments)
    except:
        raise Exception('len(segments)={} should be >= n_clusters={}, len(data_samp)={} should be >= segment_len={}'
                        .format(len(segments), n_clusters, len(data_samp), segment_len))
    return clusterer


def main(folder_name, data_file, data_path, output_path):
    if not os.path.exists(output_path + folder_name):
        os.mkdir(output_path + folder_name)
    df = pd.read_csv(data_path + data_file)
    vols = {}
    for index, row in df.iterrows():
        vols[row['id']] = [row['ts'], row['vol']]
    error_data = {}
    for key, value in vols.items():
        find_anomaly(key, [json.loads(value[0]), json.loads(value[1])], error_data, output_path, folder_name,
                     int(conf.get('Find_Anomaly', 'segment_len')), int(conf.get('Find_Anomaly', 'slide_len')),
                     int(conf.get('Find_Anomaly', 'n_plot_samples')), int(conf.get('Find_Anomaly', 'n_clusters')))
    handle_errors(error_data, folder_name, output_path)


def handle_errors(error_data, folder_name, output_path):
    errors = pd.DataFrame(columns=['AS id', 'avg error', 'max error'])
    idx = 0
    for key, value in error_data.items():
        errors.loc[idx] = [key, value['avg error'], value['max error']]
        idx = idx + 1
    errors.to_csv(output_path + folder_name + '/errors.csv')


if __name__ == '__main__':
    conf = ConfigParser()
    conf.read('config.ini')
    main(conf.get('Paths', 'folder_name'), conf.get('Paths', 'data_file'), conf.get('Paths', 'data_path'),
         conf.get('Paths', 'output_path'))
