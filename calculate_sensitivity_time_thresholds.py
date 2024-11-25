import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib
from postprocess_simulation_results import *
import multiprocessing as mp
import argparse
import warnings
# map_nums = [711, 126, 244, 984, 671, 414, 701, 31, 93, 703]
# map_nums = [414, 703]

warnings.filterwarnings("ignore")

map_nums = [i for i in range(1000)]
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help = 'path to post-analysis csv files', type = pathlib.Path, required = True)

args = parser.parse_args()

col_names = [
            'ischemic_patients_diff',
            'lvo_patients_diff',
            'evt_lvo_mean_diff'
        ]

def calc_single_map_time_thresholds(map_number):
    csv_paths = args.path.glob(f'*{map_number}.csv')
    retval = {
        0.141: {},
        0.241: {},
        0.341: {}
    }
    for path in csv_paths:
        lvo_prevalance = path.stem.split('_')[0]
        match lvo_prevalance:
            case 'low':
                prevalance = 0.141
            case 'mid':
                prevalance = 0.241
            case 'high':
                prevalance = 0.341
        df = pd.read_csv(path)
        df = remove_base_case_and_non_diffs(df, remove_base = True, remove_nondiffs = True)
        df_means = df.groupby(['sensitivity', 'threshold']).mean()

        retval[prevalance] = {
            'mRS_ischemic': (df_means['ischemic_patients_diff'].idxmax(),df_means['ischemic_patients_diff'].max()),
            'mRS_lvo': (df_means['lvo_patients_diff'].idxmax(), df_means['lvo_patients_diff'].max()), 
            'time_evt_lvo': (df_means['evt_lvo_mean_diff'].idxmin(),df_means['evt_lvo_mean_diff'].min())
        }
    retval_df = pd.DataFrame.from_dict(retval)
    for lvo in (0.141, 0.241, 0.341):
        try:
            # print(pd.DataFrame(retval_df[lvo].tolist(), index = retval_df.index))
            retval_df[[(lvo, 'scenario'), (lvo, 'value')]] = pd.DataFrame(retval_df[lvo].tolist(), index = retval_df.index)
            # retval_df[[(lvo, 'sensitivity'), (lvo, 'threshold')]] = pd.DataFrame(retval_df[(lvo, 'scenario')].tolist(), index = retval_df.index)
            retval_df.drop(lvo, axis = 1, inplace = True)

            retval_df[[(lvo, 'sensitivity'), (lvo, 'threshold')]] = pd.DataFrame(retval_df[(lvo, 'scenario')].tolist(), index = retval_df.index)
            retval_df.drop((lvo, 'scenario'), axis = 1, inplace = True)
            # retval_df.drop((lvo, 'scenario'), axis = 1, inplace = True)
        except:
            retval_df[[(lvo, 'scenario'), (lvo, 'value')]] = pd.DataFrame([[None, None], [None, None], [None, None]], index = retval_df.index)
            retval_df.drop(lvo, axis = 1, inplace = True)

            retval_df[[(lvo, 'sensitivity'), (lvo, 'threshold')]]= pd.DataFrame([[None, None], [None, None], [None, None]], index = retval_df.index)
            retval_df.drop((lvo, 'scenario'), axis = 1, inplace = True)
            # retval_df[[[(lvo, 'sensitivity'), (lvo, 'threshold')]]] = None, None
            # retval_df.drop((lvo, 'scenario'), axis = 1, inplace = True)
    retval_df.columns = pd.MultiIndex.from_tuples(retval_df.columns)
    retval_df.index = pd.MultiIndex.from_tuples(zip([map_number for i in range(len(col_names))], retval_df.index))
    return retval_df
    
        
if __name__ == '__main__':
    with mp.Pool(20) as pool:
        results = pool.map(calc_single_map_time_thresholds, map_nums)
    all_thresholds = pd.concat(results, axis = 0)
    all_thresholds.to_csv(args.path.parent / 'optimal_thresholds.csv')
    all_thresholds.to_excel(args.path.parent / 'optimal_thresholds.xlsx')

    df = all_thresholds
    sensitivities = df.xs('sensitivity', axis = 1, level = 1)
    thresholds = df.xs('threshold', axis = 1, level = 1)

    for col in sensitivities.columns:
        sensitivities[col] = sensitivities[col].astype(pd.api.types.CategoricalDtype(categories = ['low', 'mid', 'high'], ordered = True)).copy()

    for metric in df.index.unique(level = 1):
        fig, axes = plt.subplots(nrows = 1, ncols = 2)
        sensitivities.xs(metric, axis = 0, level = 1).apply(pd.value_counts).fillna(0).transpose().loc[:,['low','mid','high']].plot.bar(rot = 0, ax = axes[0])

        ax_threshold = thresholds.xs(metric, axis = 0, level = 1).apply(pd.value_counts).fillna(0).sort_index().transpose().plot.bar(rot = 0, ax = axes[1])

        fig.suptitle(metric)
        axes[0].set_title('Optimal sensitivity')
        axes[1].set_title('Optimal thresholds')

        fig.savefig(args.path.parent / f'optimal_{metric}.png')
        fig.close()

        

