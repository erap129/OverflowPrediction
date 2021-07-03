import os
import pandas as pd
from datetime import datetime, timedelta
import json
from sacred import Experiment
ex = Experiment('Preprocess')


def get_whole_netflow_data(file):
    orig_df = pd.read_csv(file)
    vols = {}
    dfs = []
    for index, row in orig_df.iterrows():
        vols[row['id']] = [row['ts'], row['vol']]
    idx = 0
    for key, value in vols.items():
        datetimes = [datetime.utcfromtimestamp(int(tm)) for tm in json.loads(value[0])]
        df = pd.DataFrame(list(zip(datetimes, json.loads(value[1]))), columns=['ts', orig_df.iloc[idx]['id']])
        df = df.sort_values(by='ts')
        df.index = pd.to_datetime(df['ts'])
        df = df.drop(columns=['ts'])
        df_T = df.T
        df_T.index.name = 'id'
        df_T.index = df_T.index.map(str)

        try:
            # df_T.index = df_T.index.str.replace('\\','_')
            df_T.index._data[0] = df_T.index._data[0].replace('\\','_')
        except:
            print ('exception!!!!')
        df_T.index = subdir_name + '_' + df_T.index.astype(str)
        dfs.append(df_T)
        # try:
        #     df_T.to_csv(os.path.join(out_path, f'{df_T.index._data[0]}_flow.csv'))
        # except:
        #     print ('exception!!!!')
        idx += 1
    df = pd.concat(dfs)
    return df


def get_whole_snmp_data(file, subdir, out_path):
    orig_df = pd.read_csv(file, index_col=0)
    vols = {}
    dfs = []
    interfaces = orig_df['group'].unique()
    interfaces.sort()
    subdir_name = subdir.split('\\')[-1]

    for interface in interfaces:
        df_per_itf = orig_df[orig_df['group'] == interface]
        datetimes = [datetime.utcfromtimestamp(int(tm)) for tm in df_per_itf['timestamp'].to_numpy()]
        full_hour_times = [x for x in datetimes if x.minute == 0]
        df_per_itf['full_dates'] = datetimes
        df_per_itf_filtered = df_per_itf[df_per_itf['full_dates'].isin(full_hour_times)][
            ['bytes', 'full_dates']].set_index('full_dates')
        df_per_itf_filtered = df_per_itf_filtered.T
        df_per_itf_filtered.index = [subdir_name + '_' + str(interface)]
        df_per_itf_filtered.index.name = 'device'
        df_per_itf_filtered.sort_index(axis=1, inplace=True)
        #
        # device = subdir_name + '_' + str(interface)
        # dict_per_itf = dict(zip(full_hour_times, df_per_itf['bytes']))
        # o_per_itf_df = pd.DataFrame(dict_per_itf,  index=[device])
        # o_per_itf_df.index.name = 'device'
        # o_per_itf_df.sort_index(axis=1, inplace=True)
        df_per_itf_filtered.to_csv(os.path.join(out_path, f'{subdir_name}_{interface}_snmp.csv'))


def union_all_routers_intfs():
    routers_path = 'D:\\BGU\\Thesis\\NetflowInsights\\CNN Denoising Autoencoders Code\\data\\ingress_routers'
    out_path = './data/snmp/routers_cleaned'
    df_list = []
    for subdir, dirs, files in os.walk(routers_path):
        subdir_name = subdir.split('\\')[-1]
        print (subdir_name)
        for file in files:
            if 'timeseries' in file:
                f = os.path.join(subdir, file)
                all_intrf = get_whole_netflow_data(f)
                # new_f.index = subdir_name + '_' + new_f.index.astype(str)
                df_list.append(all_intrf)
    big_df = pd.concat(df_list)
    big_df.index.name = 'id'
    big_df = big_df.fillna(0)
    big_df.to_csv(os.path.join(out_path, f'all_routers_intrfces.csv'))


def union_all_routers_intfs_snmp():
    routers_path = './data/snmp/per_routers_interface'
    out_path = 'D:\\BGU\\Thesis\\NetflowInsights\\CNN Denoising Autoencoders Code\\data\\snmp\\'
    df_list = []
    for subdir, dirs, files in os.walk(routers_path):
        subdir_name = subdir.split('\\')[-1]
        print (subdir_name)
        for file in files:
            print(file)
            df = pd.read_csv(os.path.join(subdir, file), index_col='device')
            df_list.append(df)
    big_df = pd.concat(df_list)
    big_df.index.name = 'id'
    big_df = big_df.fillna(0)
    big_df.to_csv(os.path.join(out_path, f'all_routers_intrfces_snmp.csv'))


def process_all_snmp_intfs():
    routers_path = 'D:\\BGU\\Thesis\\NetflowInsights\\CNN Denoising Autoencoders Code\\data\\ingress_routers' # C:\\Users\\Administrator\\Desktop\\netflow\\data\\ingress_routers
    out_path = 'D:\\BGU\\Thesis\\NetflowInsights\\CNN Denoising Autoencoders Code\\data\\snmp\\per_routers_interface' # C:\\Users\\Administrator\\Desktop\\netflow\\data\\snmp\\per_routers_interface
    df_list = []
    for subdir, dirs, files in os.walk(routers_path):
        subdir_name = subdir.split('\\')[-1]
        print(subdir_name)
        for file in files:
            if 'external_link_utilization' in file:
                f = os.path.join(subdir, file)
                print(f)
                per_itfs = get_whole_snmp_data(f, subdir_name, out_path)
                # new_f.index = subdir_name + '_' + new_f.index.astype(str)
                # per_itfs.to_csv(os.path.join(out_path, f'{subdir_name}_all_routers_intrfces.csv'))


def daterange(start_date, end_date):
    delta = timedelta(hours=1)
    while start_date < end_date:
        yield start_date
        start_date += delta


def update_dates():
    path_to_full_data = 'D:\\BGU\\Thesis\\NetflowInsights\\CNN Denoising Autoencoders Code\\data\\routers_cleaned\\all_routers_interfaces_given.csv'
    full_df = pd.read_csv(path_to_full_data)
    dates_list = full_df.columns[1:]

    start_date = datetime.strptime(dates_list[0], '%Y-%m-%d %H:%M:%S')
    end_date = datetime.strptime(dates_list[-1], '%Y-%m-%d %H:%M:%S')
    needed_list_dates = list(daterange(start_date, end_date))
    # for single_date in daterange(start_date, end_date):
    #     print(single_date.strftime("%Y-%m-%d %H:%M"))

    # res = [datetime.strptime(dates_list[i + 1], '%Y-%m-%d %H:%M:%S')
    #        - datetime.strptime(dates_list[i], '%Y-%m-%d %H:%M:%S') == timedelta(hours=1)
    #        for i in range(len(dates_list)-1)]
    needed_list_dates.insert(0, 'id')
    new_df = pd.DataFrame(columns=needed_list_dates)
    for new_col in needed_list_dates:
        if new_col == 'id':
            new_df[new_col] = full_df[new_col]
        elif new_col.strftime('%Y-%m-%d %H:%M:%S') in full_df.columns:
            new_df[new_col] = full_df[new_col.strftime('%Y-%m-%d %H:%M:%S')]
        else:
            new_df[new_col] = 0
    # <class 'list'>: [4034, 4990, 10263]
    # dates_list[4034:4037]
    # Index(['2018-01-10 23:00:00', '2018-01-11 04:00:00', '2018-01-11 05:00:00'], dtype='object')


def remove_rows_with_zeros():
    path_to_all_dates_full_data = 'D:\\BGU\\Thesis\\NetflowInsights\\CNN Denoising Autoencoders Code\\data\\all_routers_interfaces_onefile\\all_dates_routers_interfaces.csv'
    full_dates = pd.read_csv(path_to_all_dates_full_data)
    full_dates.index = full_dates['id']
    full_dates.drop('id', axis=1, inplace=True)
    th = 0.5 * full_dates.shape[1]
    counts = full_dates[(full_dates > 0).sum(axis=1) >= th]
    print


def create_dfs_from_one():
    routers_path = 'D:\\BGU\\Thesis\\NetflowInsights\\CNN Denoising Autoencoders Code\\data\\all_routers_interfaces_onefile\\' \
                   'all_dates_routers_interfaces_gt0.75zeros.csv'
    out_path = 'D:\\BGU\\Thesis\\NetflowInsights\\CNN Denoising Autoencoders Code\\data\\per_router_interface_all_dates_gt0.75zeros'
    full_dates = pd.read_csv(routers_path)
    full_dates.index = full_dates['id']
    full_dates.drop('id', axis=1, inplace=True)
    for index,row in full_dates.iterrows():
        device = index
        device_df = pd.DataFrame(full_dates.loc[index]).T
        device_df.to_csv(f'D:\\BGU\\Thesis\\NetflowInsights\\CNN Denoising Autoencoders Code\\data\\'
                         f'per_router_interface_all_dates_gt0.75zeros\\{device}_gt0.75zeros.csv')


def is_done(r_file):
    for done_f in os.listdir(results_path):
        if r_file in done_f:
            return True


def remove_rows_with_zeros_and_create_df():
    path_to_all_dates_full_data = './data/snmp/all_routers_intrfces_snmp.csv'
    full_dates = pd.read_csv(path_to_all_dates_full_data)
    full_dates.index = full_dates['id']
    full_dates.drop('id', axis=1, inplace=True)
    for index, row in full_dates.iterrows():
        device = index
        # if is_done(device):
        #     continue
        device_df = pd.DataFrame(full_dates.loc[index]).T
        device_df_csf = device_df[device_df.cumsum(axis=1).gt(0)]
        device_df_csb = device_df_csf[device_df_csf[device_df_csf.columns[::-1]].cumsum(axis=1).gt(0)[::-1]].dropna(axis=1)
        th = 0.9 * device_df_csb.shape[1]
        counts = device_df_csb[(device_df_csb > 0).sum(axis=1) >= th]
        if not counts.empty:
            device_df_csb.to_csv(f'D:\\BGU\\Thesis\\NetflowInsights\\CNN Denoising Autoencoders Code\\data\\snmp\\'
                         f'NEWper_router_interface_no_leading_zeros_gt0.9zeros\\{device}_gt0.9_no_stend_zeros_snmp.csv')

@ex.automain
def main():
    process_all_snmp_intfs()
    union_all_routers_intfs_snmp()
    remove_rows_with_zeros_and_create_df()