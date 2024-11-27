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
parser.add_argument('-p', '--path', help =
 'path to post-analysis csv files', type = pathlib.Path, required = True)

parser.add_argument('-b', '--box', help = 'Set to True to make threshold plot a box plot', type = bool, action = 'store_true')

args = parser.parse_args()

col_names = [
            'ischemic_patients_diff',
            'lvo_patients_diff',
            'evt_lvo_mean_diff'
        ]

panel = '''
ABC
ABD
ABE
'''

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
    values = df.xs('value', axis = 1, level = 1)

    for col in sensitivities.columns:
        sensitivities[col] = sensitivities[col].astype(pd.api.types.CategoricalDtype(categories = ['low', 'mid', 'high'], ordered = True)).copy()

    for metric in df.index.unique(level = 1):
        fig, axes = plt.subplots(nrows = 1, ncols = 2)
        
        fig, axes = plt.subplot_mosaic(panel, sharex = False, figsize = (20, 10))

        sensitivities.xs(metric, axis = 0, level = 1).apply(pd.value_counts).fillna(0).transpose().loc[:,['low','mid','high']].plot.bar(rot = 0, ax = axes['A'])

        if args.box:
            thresholds.xs(metric, axis = 0, level = 1).plot.box(rot = 0, ax = axes['B'])
        else:
            thresholds.xs(metric, axis = 0, level = 1).apply(pd.value_counts).fillna(0).sort_index().transpose().plot.bar(rot = 0, ax = axes['B'])

        metric_values = values.xs(metric, axis = 0, level = 1)
        
        metric_values[0.141].hist(ax = axes['C'], color = 'C0')
        metric_values[0.241].hist(ax = axes['D'], color = 'C1')
        metric_values[0.341].hist(ax = axes['E'], color = 'C2')

        axes['A'].set_title('Optimal sensitivities')
        axes['B'].set_title('Optimal thresholds')
        axes['C'].set_title('Treatment values')

        axes['C'].sharex(axes['D'])
        axes['E'].sharex(axes['D'])
        
        axes['A'].set_xlabel('LVO Prevalance')
        axes['B'].set_xlabel('LVO Prevalance')
        axes['C'].set_ylabel('0.141')
        axes['D'].set_ylabel('0.241')
        axes['E'].set_ylabel('0.341')

        fig.suptitle(metric, fontsize = 'xx-large')

        fig.savefig(args.path.parent / f'psc_optimal_{metric}.png')
        plt.close()

        

