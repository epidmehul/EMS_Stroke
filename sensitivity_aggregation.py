import numpy as np
import pandas as pd
import seaborn as sns
import pathlib
from postprocess_simulation_results import *

map_nums = [711, 126, 244, 984, 671, 414, 701, 31, 93]

def plot_all_sensitivity(df, y, sensitivity_levels = None, lvo_levels = None, title_str = '', save = False, output_path = None):    
    if sensitivity_levels is not None or lvo_levels is not None:
        def check_for_substrings(series, substring_list):
            """Checks if a pandas series of strings has any substrings in a given list."""
            mask = series.str.contains('|'.join(substring_list))
            return mask

        if sensitivity_levels is not None:    
            df = df.loc[check_for_substrings(df['sensitivity'], sensitivity_levels).values, :]
        if lvo_levels is not None:
            lvo_levels = [str(i) for i in lvo_levels]
            df = df.loc[check_for_substrings(df['sensitivity'], lvo_levels).values, :]
        data = df
    else:
        data = df
    # return data
    ax = sns.lineplot(data, x = 'threshold', y = y, hue = 'sensitivity', errorbar = None, marker = 'o')
    ax.set_title(title_str)
    if save:
        
        ax.get_figure().savefig()
    return ax

def single_map_aggregation(map_number, input_path):
    '''
    Loads in sensitivity analyses and creates visualizations
    '''
    csv_paths = input_path.glob(f'*{map_number}.csv')
    lvo_df_dict = {}
    for path in csv_paths:
        lvo_prevalance = path.stem.split('_')[0]
        lvo_df_dict[lvo_prevalance] = pd.read_csv(path)
    print(lvo_df_dict.keys())
    low_lvo = remove_base_case_and_non_diffs(lvo_df_dict['low'], remove_base = True, remove_nondiffs = False)
    low_lvo['sensitivity'] = low_lvo['sensitivity'].apply(lambda x: x+'_0.141')

    mid_lvo = remove_base_case_and_non_diffs(lvo_df_dict['mid'], remove_base = True, remove_nondiffs = False)
    mid_lvo['sensitivity'] = mid_lvo['sensitivity'].apply(lambda x: x+'_0.241')

    high_lvo = remove_base_case_and_non_diffs(lvo_df_dict['high'], remove_base = True, remove_nondiffs = False)
    high_lvo['sensitivity'] = high_lvo['sensitivity'].apply(lambda x: x+'_0.341')

    full_data = pd.concat((low_lvo, mid_lvo, high_lvo), axis = 0)
    return full_data

if __name__ == '__main__':
    # input_dir = '/proj/patellab/peter/output_sens/all_numbers'
    input_dir = '/work/users/p/w/pwlin/output_sens/all_numbers'
    # output_dir = '/proj/patellab/peter/output_sens/results'
    output_dir = '/work/users/p/w/pwlin/output_sens/results'
    for map in map_nums:
        output_path = pathlib.Path(output_dir) / f'map_{str(map).zfill(3)}/aggregated'
        if not output_path.is_dir():
            output_path.mkdir(parents = True)
        full_data = single_map_aggregation(map, pathlib.Path(input_dir))

        col_names = [
            'lvo_patients_diff',
            'ischemic_patients_diff',
            'overtriage_diff',
            'undertriage_diff',
            'ivt_ischemic_mean',
            'evt_lvo_mean'
        ]
        title_strs = [
            'mRS LVO',
            'mRS ischemic',
            'overtriage',
            'undertriage',
            'IVT time ischemic',
            'EVT time LVO'
        ]

        for i, col in enumerate(col_names):
            title = title_strs[i]
            ax = plot_all_sensitivity(full_data, col, sensitivity_levels=None, title_str = title)
            ax.get_figure().savefig(output_path / f'{title.replace(' ','_')}')
            plt.close()

