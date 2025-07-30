from stroke_simulation import *
from postprocess_simulation_results import *
import multiprocessing as mp
import argparse
import pathlib
from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help = 'directory containing parquet files', type = pathlib.Path)
parser.add_argument('-n', '--n_cores', help = 'number of cores for mp.Pool', type = int, default = 10)
parser.add_argument('-w', '--width', help = 'confidence interval width as a proportion', type = float, default = 0.95)
parser.add_argument('-o', '--output', help = 'output directory for simulation and analysis', type = pathlib.Path, default = '/work/users/p/w/pwlin/re_output')
args = parser.parse_args()

num_cores = args.n_cores
if not args.output.exists():
    args.output.mkdir(parents = True)

parquet_list = sorted(args.input.glob('*.parquet'))

def group_all_data(filepath, psc_only = False):
    map_number = int(filepath.stem.split('_')[1])
    df = pd.read_parquet(filepath)
    df = df.loc[~df['scenario'].isin([8, 15])]
    df.loc[:, 'diagnostic'] = df['sensitivity'].replace(
        to_replace = {0.9: 'high', 0.75: 'mid', 0.6: 'low'}
    )
    df.loc[df['threshold'] == 0, 'diagnostic'] = 'base'
    df['destination_type'] = df['destination'].copy()
    df.loc[df['destination'].str.contains('PSC'), 'destination_type'] = 'PSC'
    if psc_only:
        df = df.loc[df['closest_destination'] != 'CSC', :]

    ############
    triage_indicators = df.loc[df['closest_destination'] != 'CSC', ['seed', 'diagnostic', 'threshold', 'hasLVO', 'destination_type']]
    triage_indicators['overtriage'] = (triage_indicators['destination_type'] == 'CSC') & (~triage_indicators['hasLVO'])
    triage_indicators['undertriage'] = (triage_indicators['destination_type'] != 'CSC') & (triage_indicators['hasLVO'])

    ischemic_indicators = df.loc[df['ischemic'], ['seed', 'diagnostic', 'threshold', 'IVTtime', 'ischemic']]
    lvo_indicators = df.loc[df['hasLVO'], ['seed', 'diagnostic', 'threshold', 'EVTtime', 'hasLVO']]

    ivt_times = df.loc[(df['ischemic']) & (df['IVTtime'] < 270), ['seed', 'diagnostic', 'threshold', 'IVTtime']]
    evt_times = df.loc[(df['hasLVO']) & (df['EVTtime'] < 24 * 60), ['seed', 'diagnostic', 'threshold', 'EVTtime']]

    ischemic_mRS = df.loc[(df['ischemic']), ['seed', 'diagnostic', 'threshold', 'PrOut']].rename({'PrOut': 'mRS_ischemic'}, axis = 1)
    lvo_mRS = df.loc[(df['hasLVO']), ['seed', 'diagnostic', 'threshold', 'PrOut']].rename({'PrOut': 'mRS_lvo'}, axis = 1)
    #############
    overtriage_undertriage = triage_indicators.drop(['hasLVO', 'destination_type'], axis = 1).groupby(['seed', 'diagnostic', 'threshold']).mean()

    ischemic_count = ischemic_indicators.drop('IVTtime', axis = 1).groupby(['seed', 'diagnostic', 'threshold']).count().rename({'ischemic': 'ischemic_count'}, axis = 1)
    ischemic_ivt_count = ischemic_indicators.loc[ischemic_indicators['IVTtime'] < 270, :].drop('IVTtime', axis = 1).groupby(['seed', 'diagnostic', 'threshold']).count().rename({'ischemic': 'ischemic_ivt_count'}, axis = 1)

    lvo_count = lvo_indicators.drop('EVTtime', axis = 1).groupby(['seed', 'diagnostic', 'threshold']).count().rename({'hasLVO': 'lvo_count'}, axis = 1)
    lvo_evt_count = lvo_indicators.loc[lvo_indicators['EVTtime'] < 24 * 60, :].drop('EVTtime', axis = 1).groupby(['seed', 'diagnostic', 'threshold']).count().rename({'hasLVO': 'lvo_evt_count'}, axis = 1)

    ivt_cohort_avg = ivt_times.groupby(['seed', 'diagnostic', 'threshold']).mean()
    evt_cohort_avg = evt_times.groupby(['seed', 'diagnostic', 'threshold']).mean()
    mRS_ischemic_cohort_avg = ischemic_mRS.groupby(['seed', 'diagnostic','threshold']).mean()
    mRS_lvo_cohort_avg = lvo_mRS.groupby(['seed', 'diagnostic', 'threshold']).mean()

    grouped_avgs = overtriage_undertriage.join(
        (ischemic_count, ischemic_ivt_count,
            lvo_count, lvo_evt_count, ivt_cohort_avg, evt_cohort_avg,
            mRS_ischemic_cohort_avg, mRS_lvo_cohort_avg),
        validate = '1:1'
    )
    grouped_avgs['ischemic_ivt_prop'] = grouped_avgs['ischemic_ivt_count'] / grouped_avgs['ischemic_count']
    grouped_avgs['lvo_evt_prop'] = grouped_avgs['lvo_evt_count'] / grouped_avgs['lvo_count']
    #################

    grouped_avgs_diff = grouped_avgs.copy()
    for i in np.unique(grouped_avgs.index.get_level_values('seed').values):
        grouped_avgs_diff.loc[pd.IndexSlice[i, :, :], :] = grouped_avgs.loc[pd.IndexSlice[i, :, :], :] - grouped_avgs.loc[pd.IndexSlice[i, 'base', 0], :]
    grouped_avgs_diff.drop(['ischemic_count', 'lvo_count'], axis = 1, inplace = True)
    grouped_avgs_diff.rename(lambda x: x+'_diff', axis = 1, inplace = True)
    
    joined_avgs = grouped_avgs.join(grouped_avgs_diff, validate = '1:1')
    joined_avgs['map'] = map_number

    k = np.unique(grouped_avgs.index.get_level_values('seed').values).shape[0]
    
    grouped_prop_avg = joined_avgs[['ischemic_ivt_prop', 'lvo_evt_prop', 'ischemic_ivt_prop_diff', 'lvo_evt_prop_diff']].reset_index(['diagnostic', 'threshold']).groupby(['diagnostic','threshold'])
    grouped_prop_avg_mean = grouped_prop_avg.mean() 
    lower_prop_ci = grouped_prop_avg_mean - stats.norm.ppf(1 - (1 - args.width)/ 2) * np.sqrt(grouped_prop_avg_mean * (1 - grouped_prop_avg_mean) / k)
    upper_prop_ci = grouped_prop_avg_mean + stats.norm.ppf(1 - (1 - args.width)/ 2) * np.sqrt(grouped_prop_avg_mean * (1 - grouped_prop_avg_mean) / k)

    joined_avgs.drop(['ischemic_ivt_prop', 'lvo_evt_prop', 'ischemic_ivt_prop_diff', 'lvo_evt_prop_diff'], axis = 1, inplace = True)

    full_grouped_avgs = joined_avgs.drop('map', axis = 1).reset_index(['diagnostic', 'threshold']).groupby(['diagnostic', 'threshold'])
    means = full_grouped_avgs.mean()
    vars = full_grouped_avgs.var()
    lower = means - stats.t.ppf(1 - (1 - args.width) / 2, df = k - 1) * np.sqrt(vars / k)
    upper = means + stats.t.ppf(1 - (1 - args.width) / 2, df = k - 1) * np.sqrt(vars / k)

    means = means.join(grouped_prop_avg_mean, validate = '1:1')
    lower = lower.join(lower_prop_ci, validate = '1:1')
    upper = upper.join(upper_prop_ci, validate = '1:1')

    means.rename(lambda x: x+'_means', inplace = True, axis = 1)
    lower.rename(lambda x: x+'_lower', inplace = True, axis = 1)
    upper.rename(lambda x: x+'_upper', inplace = True, axis = 1)
    intervals = means.join((lower, upper), validate = '1:1')
    intervals = intervals.sort_index(axis = 1)
    intervals.drop(list(intervals.filter(regex = 'count')), axis = 1, inplace = True)
    intervals['map'] = map_number
    return joined_avgs.reset_index(['diagnostic', 'threshold']), intervals.reset_index()

if __name__ == '__main__':
    with mp.Pool(num_cores) as pool:
        results = pool.starmap(group_all_data, list(zip(parquet_list, [False] * len(parquet_list))))
        psc_results = pool.starmap(group_all_data, list(zip(parquet_list, [True] * len(parquet_list))))
    grouped_avgs, intervals = zip(*results)
    psc_grouped_avgs, psc_intervals= zip(*results)
    pd.concat(grouped_avgs).to_csv(args.output / 'all_cohort_avgs.csv')
    pd.concat(intervals).to_csv(args.output / 'map_scenario_intervals.csv', index = False)
    pd.concat(psc_grouped_avgs).to_csv(args.output / 'psc_cohort_avgs.csv')
    pd.concat(psc_intervals).to_csv(args.output / 'psc_map_scenario_intervals.csv', index = False)