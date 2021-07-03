import os
from configparser import ConfigParser
import json
import pandas as pd
import requests
import time
import datetime
from itertools import product
from numpy import array

token = '8afaa9ab14d2303636b92ffef97f767495a94ab7e25aee6042008001a6d066f0'

def replace_invalid_chars(name):
    inv_chars = '\/:*?"<>|'
    for c in name:
        if c in inv_chars:
            name = name.replace(c, '-')
    return name


def get_link_interfaces(id1, id2, start_date, end_date):
    headers = {"Accept": "application/json; version=1", "Content-Type": "application/json; version=1",
               "Authorization": "Token " + token}
    params = {"endDate": end_date, "inRouter": id1, "outRouter": id2, "startDate": start_date}

    response = requests.get('https://dt.analytics.benocs.com/api/linkutil/interfaces/', headers=headers, params=params)
    interface_names = eval(response.content)
    interfaces = []
    for pair in interface_names:
        start_interface = pair['linkStartInterfaceName']
        end_interface = pair['linkEndInterfaceName']
        interfaces.append((start_interface, end_interface))
    return interfaces


def get_all_links_names():
    df = pd.read_csv(data_path + 'all_links.csv')
    links = []
    for ids in df['linkID']:
        id1, id2 = ids.split('<->')
        links.append((id1, id2))
    return links


def get_links_data(start_date, end_date, output_file, routers=None):
    if not os.path.exists(data_path + output_file):
        os.mkdir(data_path + output_file)
    links = get_all_links_names()
    if routers is not None:
        links = list(filter(lambda x: any(name == x[0] or name == x[1] for name in routers), links))
    for pair in links:
        id1, id2 = pair
        r_id = ''
        if routers is not None:
            r_id = id1 if id1 in routers else id2
        folder_path = data_path + output_file + '/' + r_id + '/{}_{}'.format(id1, id2)
        get_link_data(id1, id2, start_date, end_date, folder_path)


def get_link_data(id1, id2,  start_date, end_date, folder_path):
    interfaces = get_link_interfaces(id1, id2, start_date, end_date)
    for interface_pair in interfaces:
        start_interface, end_interface = interface_pair
        headers = {"Accept": "application/json; version=2", "Content-Type": "application/json",
                   "Authorization": "Token " + token}
        params = {"dataType": "single_interface_pair", "endDate": end_date, "endInterface": end_interface,
                  "inRouter": id1, "outRouter": id2, "startDate": start_date, "startInterface": start_interface}

        response = requests.get('https://dt.analytics.benocs.com/api/linkutil/internal', headers=headers,
                                params=params)

        c = eval(response.content)
        if len(c["data"]) != 0:
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            df = pd.DataFrame.from_dict(c["data"])
            df.to_csv(folder_path + '/{}_{}.csv'.format(replace_invalid_chars(start_interface),
                                                        replace_invalid_chars(end_interface)))


def get_timeseries_with_data_str(data_str):
    import requests
    headers = {"Accept": "application/json; version=3", "Content-Type": "application/json",
               "Authorization": f"Token {token}"}
    response = requests.post('https://as3320.analytics.benocs.com/api/timeseries/', headers=headers, data=data_str)
    c = eval(response.content)
    return pd.DataFrame.from_dict(c["data"])


def get_timeseries(start_date, end_date, pos='handoverAS', src_as=None, dst_as=None, handover=None, next_hop=None):
    src_as_str = '\"{}\"'.format(src_as) if src_as is not None else ''
    dst_as_str = '\"{}\"'.format(dst_as) if dst_as is not None else ''
    handover_str = '\"{}\"'.format(handover) if handover is not None else ''
    next_hop_str = '\"{}\"'.format(next_hop) if next_hop is not None else ''
    pos_str = '\"{}\"'.format(pos)
    start_date = time.mktime(datetime.datetime.strptime(start_date, "%Y-%m-%d").timetuple())
    end_date = time.mktime(datetime.datetime.strptime(end_date, "%Y-%m-%d").timetuple())

    import requests
    headers = {"Accept": "application/json; version=3", "Content-Type": "application/json",
               "Authorization": f"Token {token}"}
    params = {}
    data = '{{"filters":{{"srcAS":[{0}],"handoverAS":[{4}],"router":[],"inInterface":[],"BGPnextHop":[],' \
           '"nextHopAS":[{5}],"dstAS":[{1}]}},"start":{2},"end":{3},"pos":{6},' \
           '"whitelist":{{"srcAS":[{0}],"handoverAS":[{4}],"router":[],"inInterface":[],"BGPnextHop":[],' \
           '"nextHopAS":[{5}],"dstAS":[{1}]}}}}'.format(src_as_str, dst_as_str, int(start_date), int(end_date),
                                                        handover_str, next_hop_str, pos_str)
    response = requests.post('https://dt.analytics.benocs.com/api/timeseries/', headers=headers, data=data)
    c = eval(response.content)
    return pd.DataFrame.from_dict(c["data"])


def get_sankey_data(filename, start_date, end_date, src_as=None, dst_as=None):
    start_date = time.mktime(datetime.datetime.strptime(start_date, "%Y-%m-%d").timetuple())
    end_date = time.mktime(datetime.datetime.strptime(end_date, "%Y-%m-%d").timetuple())
    src_as_str = '\"{}\"'.format(src_as) if src_as is not None else ''
    dst_as_str = '\"{}\"'.format(dst_as) if dst_as is not None else ''
    headers = {"Accept": "application/json; version=3", "Content-Type": "application/json",
               "Authorization": "Token " + token}
    data = '{{"filters":{{"srcAS":[{0}],"handoverAS":[],"router":[],"inInterface":[],"BGPnextHop":[],"nextHopAS":[],' \
           '"dstAS":[{1}]}},"start":{2},"end":{3},"aggregations":{{}},' \
           '"output_dimensions":["srcAS","handoverAS","nextHopAS","dstAS"],' \
           '"whitelist":{{"srcAS":[{0}],"handoverAS":[],"router":[],"inInterface":[],"BGPnextHop":[],' \
           '"nextHopAS":[],"dstAS":[{1}]}}}}'.format(src_as_str, dst_as_str, int(start_date), int(end_date))

    response = requests.post('https://dt.analytics.benocs.com/api/sankey/', headers=headers, data=data)
    c = eval(response.content)
    df1 = pd.DataFrame.from_dict(c["data"]["nodes"])
    df2 = pd.DataFrame.from_dict(c["data"]["links"])
    df1.to_csv(data_path + filename + '_nodes.csv')
    df2.to_csv(data_path + filename + '_links.csv')


def get_timeseries_for_kitsune(filename, nodes_file, links_file):
    df = pd.read_csv(data_path + nodes_file)
    src_ASes = []
    dst_ASes = []
    handovers = []
    next_hops = []
    for index, row in df.iterrows():
        dim = row['dimension']
        if dim == 'srcAS':
            src_ASes.append((row['as_id'], row['index']))
        if dim == 'dstAS':
            dst_ASes.append((row['as_id'], row['index']))
        if dim == 'handoverAS':
            handovers.append((row['as_id'], row['index']))
        if dim == 'nextHopAS':
            next_hops.append((row['as_id'], row['index']))

    links = pd.read_csv(data_path + links_file)
    links = list(zip(links['source'], links['target']))
    ids_prod = list(product(src_ASes, handovers, next_hops, dst_ASes))
    ids_prod = list(
        filter(lambda x: (x[0][1], x[1][1]) in links and (x[1][1], x[2][1]) in links
                         and (x[2][1], x[3][1]) in links, ids_prod))
    res = {'src_as': [], 'handover': [], 'next_hop': [], 'dst_as': [], 'ts': [], 'vol': []}
    c = 0
    for src_as, handover, next_hop, dst_as in ids_prod:
        if c < 5500:
            c = c + 1
            continue
        timeseries_df = get_timeseries(str(conf.get('Data', 'start_date')), str(conf.get('Data', 'end_date')),
                                       src_as[0], dst_as[0],
                                       handover[0], next_hop[0])
        if not timeseries_df.empty:
            for ts, vol in list(zip(json.loads(timeseries_df['ts'][0]), json.loads(timeseries_df['vol'][0]))):
                res['src_as'].append(src_as[0])
                res['handover'].append(handover[0])
                res['next_hop'].append(next_hop[0])
                res['dst_as'].append(dst_as[0])
                res['ts'].append(ts)
                res['vol'].append(vol)
        print(c)
        if c % 50 == 0:
            df = pd.DataFrame.from_dict(res)
            df.to_csv(data_path + filename)
        c = c + 1
    df = pd.DataFrame.from_dict(res)
    df.to_csv(data_path + filename, index=False)


def sort_as_flow_data(source_file, dest_file):
    df = pd.read_csv(data_path + source_file)
    id = []
    ts = []
    vol = []
    for _, row in df.iterrows():
        id.append(row['id'])
        sorted_vols = pd.DataFrame([json.loads(row['ts']), json.loads(row['vol'])], index=['ts', 'vol']).T.sort_values(
            by='ts')
        ts.append([int(x) for x in list(sorted_vols['ts'])])
        vol.append(list(sorted_vols['vol']))
    ans = {'id': id, 'ts': ts, 'vol': vol}
    pd.DataFrame.from_dict(ans).to_csv(data_path + dest_file)


def get_asflow_interface_statistics(router_name, start_date, end_date):
    headers = {"Accept": "application/json", "Authorization": "Token " + token}
    params = {"endDate": end_date, "isDirectedIn": "true", "router": router_name, "startDate": start_date}
    response = requests.get('https://dt.analytics.benocs.com/api/linkutil/interfacestatistics', headers=headers,
                            params=params)
    return eval(response.content)['data']


def get_asflow_intefraces_timeseries(router_name, start_date, end_date):
    headers = {"Accept": "application/json; version=2", "Content-Type": "application/json",
               "Authorization": "Token " + token}
    params = {"BGPnextHop": "", "dstAS": "", "end": end_date, "handoverAS": "", "heavy_hitters_limit": "99",
              "nextHopAS": "", "pos": "inInterface", "router": router_name, "srcAS": "", "start": start_date}

    response = requests.get('https://dt.analytics.benocs.com/api/timeseries/6d/', headers=headers, params=params)
    return eval(response.content)['data']


def get_asflow_ingress_data(start_date, end_date, ingress_file, out_filename):
    start_date = int(time.mktime(datetime.datetime.strptime(start_date, "%Y-%m-%d").timetuple()))
    end_date = int(time.mktime(datetime.datetime.strptime(end_date, "%Y-%m-%d").timetuple()))
    if not os.path.exists(data_path + out_filename):
        os.mkdir(data_path + out_filename)
    df = pd.read_csv(data_path + ingress_file)
    routers_names = set(df['group'])
    # get_links_data(start_date, end_date, out_filename, routers_names)
    for name in routers_names:
        path = data_path + out_filename + '/{}/'.format(name)
        if not os.path.exists(path):
            os.mkdir(path)
        stat = get_asflow_interface_statistics(name, start_date, end_date)
        timeseries = get_asflow_intefraces_timeseries(name, start_date, end_date)
        external_link_utilization = get_link_router_data(start_date, end_date, name)
        pd.DataFrame.from_dict(external_link_utilization).to_csv(path + 'external_link_utilization.csv')
        pd.DataFrame.from_dict(stat).to_csv(path + 'statistics.csv')
        pd.DataFrame.from_dict(timeseries).to_csv(path + 'timeseries.csv')
        print('finished with router ' + name)


def get_link_router_data(start_date, end_date, router_name):
    headers = {"Accept": "application/json; version=2", "Content-Type": "application/json",
               "Authorization": "Token " + token}
    params = {"endDate": end_date, "groupColumn": "interfaceIndex", "isDirectedIn": "true", "routerList": router_name,
              "startDate": start_date}

    response = requests.get('https://dt.analytics.benocs.com/api/linkutil/external', headers=headers, params=params)
    return json.loads(response.content.decode("utf-8"))['data']


def get_handovers_sum(input_path, output_path, source_as_id):
    df = pd.read_csv(input_path)
    vols = {}
    for index, row in df.iterrows():
        vols[row['id']] = [row['ts'], row['vol']]
    vols_arrays = []
    for key, value in vols.items():
        df = pd.DataFrame([json.loads(value[0]), json.loads(value[1])], index=['ts', 'vol']).T
        df = df.sort_values(by='ts')
        vols_arrays.append(array(df['vol']))
    sums = [sum(x) for x in zip(*vols_arrays)]
    res = {'id': [source_as_id], 'ts': [json.dumps(list(df['ts']))], 'vol': [json.dumps(list(sums))]}
    pd.DataFrame(res).to_csv(output_path)


if __name__ == '__main__':
    conf = ConfigParser()
    conf.read('config.ini')
    data_path = conf.get('Paths', 'data_path')

    # get_asflow_ingress_data(conf.get('Data', 'start_date'), conf.get('Data', 'end_date'), 'ingress_routers.csv', 'ingress_routers')
    get_handovers_sum(conf.get('Paths', 'data_path') + conf.get('Paths', 'data_file'), os.path.join(conf.get('Paths', 'data_path'), 'akamai_dt_2_years_sum.csv'), conf.get('LSTM', 'source_as'))


