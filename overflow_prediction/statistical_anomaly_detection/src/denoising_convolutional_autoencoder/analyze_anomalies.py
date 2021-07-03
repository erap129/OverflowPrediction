import pandas as pd
from sacred import Experiment
ex = Experiment('Analysis')



def analysis_1():
    input_anomalies_file ='D:\\BGU\\Thesis\\NetflowInsights\\CNN Denoising Autoencoders Code\\results\\snmp_results\\AD\\AD_snmp_rw_gen_router_gt0.9_no_stend_zeros.csv'

    an_df = pd.read_csv(input_anomalies_file)
    # max_diff = an_df.groupby(['device'], sort=False)['abs_diff'].max()
    an_df = an_df[an_df['gen_pd_anomaly'] == True]
    an_df = an_df[an_df['ts_date'] >= '2019-01-01']
    idx = an_df.groupby(['device'])['abs_diff'].transform(max) == an_df['abs_diff']
    an_df_max = an_df[idx]
    an_df_max.to_csv('./results/netflow_results/highest_anomalies_2019.csv')


def anomalies_only():
    input_anomalies_file = './results/snmp_results/AD/AD_snmp_rw_gen_router_gt0.9_no_stend_zeros.csv'
    an_df = pd.read_csv(input_anomalies_file)
    an_df = an_df[an_df['gen_pd_anomaly'] == True]
    an_df.to_csv('./results/snmp_results/general_anomalies_full_snmp.csv')


def netflow_snmp_corr():
    netflow_path = './results/netflow_results/v1_routers/AD/AD_rw_gen_router_gt0.9_no_stend_zeros_v1.csv'
    snmp_path = './results/snmp_results/AD/AD_snmp_rw_gen_router_gt0.9_no_stend_zeros.csv'
    net_df = pd.read_csv(netflow_path)
    snmp_df = pd.read_csv(snmp_path)
    joined_df = pd.merge(net_df, snmp_df, how='inner', on=['device', 'ts_date'])
    joined_df_filtered = joined_df[['device', 'ts_date', 'abs_diff_x', 'abs_diff_y' ,'gen_pd_anomaly_x', 'gen_pd_anomaly_y']]
    joined_df_filtered = joined_df_filtered[
        joined_df_filtered['gen_pd_anomaly_x'] == joined_df_filtered['gen_pd_anomaly_y']]
    joined_df_filtered = joined_df_filtered[joined_df_filtered['gen_pd_anomaly_x'] == True]
    joined_df_filtered.to_csv('./results/netf_snmp_mutual_anomalies_full.csv')

@ex.automain
def main():
    anomalies_only()
    # analysis_1()
    # netflow_snmp_corr()

