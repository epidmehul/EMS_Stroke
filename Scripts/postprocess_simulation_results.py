from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors
import pathlib
from scipy.spatial import distance
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy import stats
from stroke_simulation import *
import openpyxl

def triage_outcomes(df):
    '''
        Given a subsetted pdDataFrame, calculates the confusion matrix for LVO status vs destination type for all patients in df
    '''
    # Arguments: ground truth, predicted label
    # Resulting metrics: Check index 1 for results for class 1 (which here is going to CSC)
    retval = {}
    try:
        df = df.loc[df['closest_destination'] != 'CSC', :]
        cm = confusion_matrix(df['hasLVO'], df['destination_type'] == 'CSC')
        # print('---------------')
        # print(cm)
        # print('----------------')
        FP = cm.sum(axis=0) - np.diag(cm)  
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)

        # Sensitivity, hit rate, recall, or true positive rate
        # TPR = TP/(TP+FN)
        # Specificity or true negative rate
        # TNR = TN/(TN+FP) 
        # Precision or positive predictive value
        PPV = TP/(TP+FP)
        # Negative predictive value
        NPV = TN/(TN+FN)
        # Fall out or false positive rate
        FPR = FP/(FP+TN)
        # False negative rate
        FNR = FN/(TP+FN)
        # False discovery rate
        # FDR = FP/(TP+FP)

        # Overall accuracy
        ACC = (TP+TN)/(TP+FP+FN+TN)
        # print(ACC)

        retval['correct_triage'] = ACC[1]
        retval['undertriage'] = FNR[1]
        retval['overtriage'] = FPR[1]
        retval['PPV'] = PPV[1]
        retval['NPV'] = NPV[1]
    except Exception as e:
        print(e)
        print(cm)
        retval['correct_triage'] = None
        retval['undertriage'] = None
        retval['overtriage'] = None
        retval['PPV'] = None
        retval['NPV'] = None
    return retval

def time_results(s):
    '''
    Calculate descriptive statistics for a pandas Series s

    s is assumed to be a Series containing the relevant time lengths already
    '''
    s_descriptive = s.describe()
    s_descriptive['iqr'] = s_descriptive['75%'] - s_descriptive['25%']
    s_descriptive['median'] = s_descriptive['50%']
    return s_descriptive[['mean','std','median','iqr','min','max']].to_dict()

def all_time_results(df):
    '''
    Calculate descriptive statistics for the time metrics for all patients, ischemic patients, and LVO patients in df
    '''
    retval = {}

    # All patients
    prehospital = time_results(df['lkw2door'])
    ems_transport = time_results(df['time2Hospital'])
    ivt = time_results(df['IVTtime'])
    evt = time_results(df['EVTtime'])
    for key in ['mean', 'std', 'median', 'iqr', 'min', 'max']:
        retval['prehospital_all_' + key] = prehospital[key]
        retval['ems_transport_all_' + key] = ems_transport[key]
        retval['ivt_all_' + key] = ivt[key]
        retval['evt_all_' + key] = evt[key]

    # Ischemic patients
    prehospital = time_results(df.loc[df['ischemic'], 'lkw2door'])
    ems_transport = time_results(df.loc[df['ischemic'],'time2Hospital'])
    ivt = time_results(df.loc[(df['ischemic']) & (df['IVTtime'] <= 270) & (df['IVTtreatment']),'IVTtime'])
    evt = time_results(df.loc[df['ischemic'],'EVTtime'])
    for key in ['mean', 'std', 'median', 'iqr', 'min', 'max']:
        retval['prehospital_ischemic_' + key] = prehospital[key]
        retval['ems_transport_ischemic_' + key] = ems_transport[key]
        retval['ivt_ischemic_' + key] = ivt[key]
        retval['evt_ischemic_' + key] = evt[key]

    # LVO patients
    prehospital = time_results(df.loc[df['hasLVO'], 'lkw2door'])
    ems_transport = time_results(df.loc[df['hasLVO'],'time2Hospital'])
    ivt = time_results(df.loc[df['hasLVO'] & df['IVTtreatment'] & (df['IVTtime'] <= 270),'IVTtime'])
    evt = time_results(df.loc[(df['hasLVO']) & (df['EVTtime'] <= 24 * 60) & (df['EVTtreatment']),'EVTtime'])
    for key in ['mean', 'std', 'median', 'iqr', 'min', 'max']:
        retval['prehospital_lvo_' + key] = prehospital[key]
        retval['ems_transport_lvo_' + key] = ems_transport[key]
        retval['ivt_lvo_' + key] = ivt[key]
        retval['evt_lvo_' + key] = evt[key]
    return retval

def mRS_probs(df):
    '''
    Calculates average of mRS probabilities for all patients, ischemic patients, and LVO patients in df
    '''
    retval = {}
    retval['all_patients'] = df['PrOut'].mean()
    retval['ischemic_patients'] = df.loc[df['ischemic']]['PrOut'].mean()
    retval['lvo_patients'] = df.loc[df['hasLVO']]['PrOut'].mean()
    return retval
    
def map_df_to_dict(df_results, map_number = 0, seed = 0):
    '''
    Extracts the relevant rows for a particular map and seed combination

    Removes the base case duplication
    Set map or seed to None if no filtering down for either variable is desired

    Scenario information:
    Base cases: 1, 8, 15
    High sens, low spec: 2-7
    Mid: 9-14
    Low sens, high spec: 16-21

    Thresholds within each range go 10, 20, 30, 40, 50, 60
    '''
    df_map = df_results
    if map_number is not None:
        df_map = df_map.loc[df_map['map_number'] == map_number, :]
    if seed is not None:
        df_map = df_map.loc[df_map['seed'] == seed, :]
    retval = {}
    retval['base'] = df_map.loc[df_map['scenario'] == 1]
    for i in range(2, 8):
        retval['high_sens_'+str((i-1)*10)] = df_map.loc[df_map['scenario'] == i]
        retval['mid_sens_'+str((i-1)*10)] = df_map.loc[df_map['scenario'] == i+7]
        retval['low_sens_'+str((i-1)*10)] = df_map.loc[df_map['scenario'] == i+14]
    return retval
        
def aggregate_outcomes(dict_dfs):
    '''
    Input: Dict of scenario: pd.DataFrame

    Supposed to be used after map_df_to_dict()

    For each outcome group of interest, aggregates relevant measures for base case and all sensitivity/specificty and threshold combinations into one dataframe
    '''
    df_base = dict_dfs['base']
    class_outcomes = {}
    time_outcomes = {}
    mRS_outcomes = {}

    class_outcomes['base'] = triage_outcomes(df_base)
    time_outcomes['base'] = all_time_results(df_base)
    mRS_outcomes['base'] = mRS_probs(df_base)
    for i in ('high', 'mid', 'low'):
        for thresh in range(10, 70, 10):
            class_outcomes[i + '_sens_' + str(thresh)] = triage_outcomes(dict_dfs[i + '_sens_' +str(thresh)])
            time_outcomes[i + '_sens_' + str(thresh)] = all_time_results(dict_dfs[i + '_sens_' +str(thresh)])
            mRS_outcomes[i + '_sens_' + str(thresh)] = mRS_probs(dict_dfs[i + '_sens_' +str(thresh)])
    classification_df = pd.DataFrame.from_dict(class_outcomes).transpose()
    time_df = pd.DataFrame.from_dict(time_outcomes).transpose()
    mRS_df = pd.DataFrame.from_dict(mRS_outcomes).transpose()

    return get_thresholds_sensitivities(classification_df), get_thresholds_sensitivities(time_df), get_thresholds_sensitivities(mRS_df)
        
def get_thresholds_sensitivities(df):
    '''
    Helper function to add the threshold and sensitivity from df.index as separate columns in df

    Assigns threshold = 0 and sensitivity = 'none' to the base case
    '''
    thresholds_as_arr = df.index.str.split('_').str[-1].values
    thresholds_as_arr[thresholds_as_arr == 'base'] = '0'
    # threshold_idx = thresholds_as_list.index('base')
    # thresholds_as_list[threshold_idx] = '0'

    sensitivities_as_arr = df.index.str.split('_').str[0].values

    df.insert(0, 'threshold', thresholds_as_arr)
    df.insert(1, 'sensitivity', sensitivities_as_arr)
    df['threshold'] = df['threshold'].astype(int)
    df['sensitivity'] = df['sensitivity'].astype(pd.api.types.CategoricalDtype(categories = ['base','high', 'mid', 'low'], ordered = True))
    return df

def add_differences_columns(df):
    '''
    Adds additional columns of differences in outcome metrics from base case
    '''    
    metric_names = df.columns.drop(['threshold','sensitivity'], errors = 'ignore')
    for name in metric_names:
        try:
            df[name + '_diff'] = df[name] - df.loc['base', name]
        except:
            df[name + '_diff'] = df[name]
    return df

def single_map_results(df, map_number = 0, include_seed = False) -> list[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    Analyzes map data, separating out by every seed possible

    Adds seed column for interval calculation later if include_seed == True and concatenates results back together
    '''
    seeds = df['seed'].unique()
    # print(seeds)
    class_df_list = []
    time_df_list = []
    mRS_df_list = []
    df_map = df.loc[df['map_number'] == map_number, :]
    # print(df_map['map_number'].value_counts())

    # Run calculations for each random seed separately
    for seed in seeds:
        df_dicts = map_df_to_dict(df_map, None, seed)
        class_df, time_df, mRS_df = aggregate_outcomes(df_dicts)
        for result_df in [class_df, time_df, mRS_df]:
            result_df = add_differences_columns(result_df)
            if include_seed:
                result_df.insert(0, 'seed', seed)
        class_df_list.append(class_df)
        time_df_list.append(time_df)
        mRS_df_list.append(mRS_df)

    # Recombine all calculations back into one larger pd.DataFrame
    # May or may not have a seed column
    return pd.concat(class_df_list), pd.concat(time_df_list), pd.concat(mRS_df_list)

def remove_base_case_and_non_diffs(df, remove_base = True, remove_nondiffs = True):
    '''
    Removes the base case and original output columns from df for the purposes of heatmap visualizations and interval calculations
    '''
    if remove_nondiffs:
        diff_columns = df.columns.map(lambda x: ("diff" in x) or x=="sensitivity" or x=="threshold")
        df.loc[:, diff_columns] = df
      
    if remove_base:
        df = df.loc[(df['threshold'] > 0), :]
        df['sensitivity'] = df['sensitivity'].astype(pd.api.types.CategoricalDtype(categories = ['high', 'mid', 'low'], ordered = True))

        # Remove the 'none' group from the ordered categorical variable so that it doesn't appear in any df.groupby() results or in heatmaps
        # df['sensitivity'] = df['sensitivity'].astype(pd.api.types.CategoricalDtype(categories = ['high', 'mid', 'low'], ordered = True))
    return df

def calculate_intervals(df, width = 0.9):
    '''
    Calculates the quantiles needed for a (100 * width)% interval for the differences between scenarios and the base case. Does not use standard error calculations 

    Also calculates the mean value and returns as retval_0
    '''
    try:
        alpha = (1 - width)/2
        # df = remove_base_case_and_non_diffs(df)
        retval = df.groupby(['sensitivity', 'threshold'], observed = True).quantile([alpha, 1 - alpha])
        retval_0 = df.groupby(['sensitivity', 'threshold'], observed = True).mean()
        retval.index.set_names(['sensitivity', 'threshold', 'quantile'], inplace = True)
        retval_1 = retval.reset_index().pivot(columns = "quantile", index = ['sensitivity', 'threshold'])
    except:
        return None, None
    return retval_0, retval_1    

def calculate_intervals_theoretical(df, width = 0.9):
    try:
        alpha = (1-width) / 2
        grouped_df = df.groupby(['sensitivity','threshold'], observed = True)
        k = np.unique(grouped_df.count().values)[0]
        means = grouped_df.mean()
        var = grouped_df.var()
        ci_lower = means - stats.t.ppf(1 - alpha, df = k - 1) * np.sqrt(var / k)
        ci_upper = means + stats.t.ppf(1 - alpha, df = k - 1) * np.sqrt(var / k)
        return means, pd.DataFrame({alpha: ci_lower, 1-alpha: ci_upper})
    except:
        return None, None
        

def generate_heatmap(df, title_str = "", col_names = None, differenced = False, save = False, additional_file_name = '', output_path = None):
    '''
    Generate the heatmap visualization of the differences when averaged across random seeds
    '''
    df = remove_base_case_and_non_diffs(df, remove_nondiffs = False)
    df = df.groupby(['sensitivity', 'threshold'], observed = True).mean().reset_index()

    if col_names is None:
        if differenced:
            diff_columns = df.columns.map(lambda x: "diff" in x)
        else:
            diff_columns = df.columns.map(lambda x: "diff" not in x and x!="sensitivity" and x!="threshold")
        # print(diff_columns)
        diff_columns_names = df.columns[diff_columns]
        # display(diff_columns_names)
    else:
        if differenced:
            diff_columns_names = [i+'_diff' for i in col_names]
        else:
            diff_columns_names = col_names
    # print(diff_columns_names)
    ax_list = []
    for col_name in diff_columns_names:
        df[col_name] = df[col_name].astype(float)
        ax = sns.heatmap(df.pivot(columns = 'threshold', index = 'sensitivity', values = col_name), annot = True, fmt = '.4f')
        ax.set_title(f"{title_str}: {col_name}")
        if save:
            if output_path is None:
                raise FileNotFoundError
            output_fig_path = output_path / f'{additional_file_name}{'_' if additional_file_name != '' else ''}{title_str.replace(' ','_')}_{col_name}.png'
            # output_fig_path = pathlib.Path(f"{output_dir}/map_{str(map_number).zfill(3)}/{additional_file_name}_{title_str.replace(' ','_')}_{col_name}.png")
            ax.get_figure().savefig(output_fig_path)
        ax_list.append(ax)
        plt.close()
    return ax_list

def generate_line_graphs(df, title_str = "", col_names = None, differenced = False, save = False, additional_file_name = '', output_path = None, errorbar = False, alpha = 0.9):
    '''
    Generates line graphs for col_names in df, with a line for each sensitivity level with transport threshold along x-axis
    '''
    df = remove_base_case_and_non_diffs(df, remove_nondiffs = False)
    # df = df.groupby(['sensitivity', 'threshold'], observed = True).mean().reset_index()
    if col_names is None:
        if differenced:
            diff_columns = df.columns.map(lambda x: "diff" in x)
        else:
            diff_columns = df.columns.map(lambda x: "diff" not in x and x!="sensitivity" and x!="threshold")
        # print(diff_columns)
        diff_columns_names = df.columns[diff_columns]
        # display(diff_columns_names)
    else:
        if differenced:
            diff_columns_names = [i+'_diff' for i in col_names]
        else:
            diff_columns_names = col_names
    ax_list = []
    for col_name in diff_columns_names:
        try:
            df[col_name] = df[col_name].astype(float)
            if not errorbar:
                ax = sns.lineplot(df, x = 'threshold', y=col_name, hue = 'sensitivity', marker = 'o', errorbar = None)
            else:
                ax = sns.lineplot(df, x = 'threshold', y=col_name, hue = 'sensitivity', marker = 'o', errorbar = ('pi', 100 * alpha))
            ax.set_title(f"{title_str}: {col_name}")
            if save:
                if output_path is None:
                    raise FileNotFoundError
                output_fig_path = output_path / f'{additional_file_name}{'_' if additional_file_name != '' else ''}{title_str.replace(' ','_')}_{col_name}_line.png'
                # output_fig_path = pathlib.Path(f"{output_dir}/map_{str(map_number).zfill(3)}/{additional_file_name}_{title_str.replace(' ','_')}_{col_name}.png")
                ax.get_figure().savefig(output_fig_path)
            ax_list.append(ax)
            plt.close()
        except:
            continue
    return ax_list

def get_map_plot(df, map_number = 0, save = True, additional_file_name = '', output_path = None, threshold = None, save_map_csv = True):
    '''
    Writes a text file containing the various coordinates and any other statistics about a given map number

    Additionally visualizes the Voronoi plot of the coordinates for visualization
    '''
    map_df = df.loc[df['map_number'] == map_number, :]
    xPSC, yPSC = map_df.iloc[0][['xPSC','yPSC']].values
    xPSC2, yPSC2 = map_df.iloc[0][['xPSC2','yPSC2']].values
    geoscale = map_df.iloc[0]['geoscale']
    drivespeed = map_df.iloc[0]['drivespeed']
    med_coords = np.array([[0.5 * geoscale, 0.5 * geoscale],
                        [xPSC, yPSC],
                        [xPSC2, yPSC2]])
    
    coord_labels = ['CSC', 'PSC', 'PSC2']
    voronoi_colors = ['blue', 'green', 'red']
    voronoi_markers = ['^','o','o']

    grid_points, grid_bools = get_map_points_threshold(med_coords, geoscale, drivespeed, threshold)

    # simulated_coords = rng.uniform(low = 0, high = geoscale, size = (num_points, 2))
    equipoise = np.sum(np.argmin(distance.cdist(grid_points, med_coords), axis = 1) != 0) / grid_points.shape[0]

    triangle_area = 0.5 * np.abs(np.linalg.det(np.vstack((med_coords.T, np.ones((1, med_coords.shape[0]))))))
    triangle_area_normalized = triangle_area / geoscale**2

    max_dist = distance.pdist(med_coords).max()
    max_dist_normalized = max_dist / geoscale

    sc_center = med_coords.mean(axis = 0)
    sc_sse = np.sum((med_coords - np.broadcast_to(np.expand_dims(sc_center, axis = 0), med_coords.shape))**2)

    sc_sse_normalized = sc_sse / geoscale**2

    # print(equipoise)
    map_csv_file = output_path.parent.parent / 'maps.csv'
    current_map_info = pd.DataFrame(
        {'map': map_number,
         'equipoise': equipoise,
         'area': triangle_area,
         'area_normalized': triangle_area_normalized,
         'sc_max_dist': max_dist,
         'sc_max_dist_normalized': max_dist_normalized,
         'sc_sse': sc_sse,
         'sc_sse_normalized': sc_sse_normalized,
         'geoscale': geoscale,
         'xPSC': med_coords[1, 0],
         'yPSC': med_coords[1, 1],
         'xPSC2': med_coords[2, 0],
         'yPSC2': med_coords[2, 1]
        }, index = [map_number])

    if save_map_csv:
        if map_csv_file.exists():
            current_map_info.to_csv(map_csv_file, header = False, index = False, mode = 'a')
        else:
            current_map_info.to_csv(map_csv_file, header = True, index = False, mode = 'w')

    distant_coords = np.array([[-8 * geoscale, -8 * geoscale],
                           [8 * geoscale, 8 * geoscale],
                           [-8 * geoscale, 9 * geoscale],
                           [8 * geoscale, -7 * geoscale]])

    full_coords = np.vstack((med_coords, distant_coords))
    vor = Voronoi(full_coords)
    voronoi_plot_2d(vor, show_vertices = False, show_points = False)
    for i, hosp in enumerate(coord_labels):
        poly = [vor.vertices[j] for j in vor.regions[vor.point_region[i]]]
        plt.fill(*zip(*poly), color = voronoi_colors[i], alpha = 0.25)
        plt.scatter(med_coords[i,0], med_coords[i,1], c = voronoi_colors[i], label = hosp, marker = voronoi_markers[i])
    plt.xlim(0, geoscale)
    plt.ylim(0, geoscale)
    plt.legend()
    plt.gca().set_aspect('equal')
    plt.title(f'Map {map_number}')
    if threshold is not None:
        # grid_points, grid_bools = get_map_points_threshold(med_coords, geoscale, drivespeed, threshold)

        new_cmap = colors.ListedColormap(['purple'])
        # for simplex in grid_hull.simplices:
        #     plt.plot(grid_points[simplex, 0], grid_points[simplex, 1], c='purple')

        # plt.scatter(grid_points[:,0], grid_points[:,1], c='purple', alpha=0.2, marker = '.', s = 1/80, lw = 1/80)
        
        # plt.pcolormesh(x, y, grid_bools, alpha=np.where(grid_bools, 0.4, 0), cmap = new_cmap)

        plt.imshow(grid_bools, cmap = new_cmap, aspect = 'equal', origin = 'lower',
                   alpha = np.where(grid_bools, 0.4, 0), extent = [0, geoscale, 0, geoscale])
        # plt.scatter(grid_points[:,0], grid_points[:,1], c = 'purple')
    if save:
        output_fig_path = output_path / f'{additional_file_name}{'_' if additional_file_name != '' else ''}map_{map_number}.png'
        plt.savefig(output_fig_path)
        plt.close()

        # output_txt_path = output_path / f'{additional_file_name}{'_' if additional_file_name != '' else ''}map_{map_number}.txt'
        # with open(output_txt_path,'w') as f:
        #     f.write(f'Map {map_number}: \n\nCoordinates: \n')
        #     for i, hosp in enumerate(coord_labels):
        #         f.write(f'{hosp}: ({med_coords[i,:]}\n')
        #     f.write('\n')
        #     f.write(f'Equipoise: {equipoise * 100}%\n')
        #     f.write('\n')
        #     f.write(f'Drivespeed: {drivespeed}')
    return current_map_info

def get_map_points_threshold(med_coords, geoscale, drivespeed, threshold):
    x,y = np.meshgrid(geoscale * np.arange(0, 1, 0.001), geoscale * np.arange(0, 1, 0.001))
    grid_points = np.stack((x.flatten(), y.flatten()), axis = -1)
    grid_dists = distance.cdist(grid_points, med_coords)
    grid_closest_med = grid_dists.argmin(axis = 1)
    grid_closest_dists = grid_dists.min(axis = 1)
    if threshold is None:
        threshold = 0
    grid_within_threshold_bools = (grid_closest_med != 0) & (grid_dists[:,0] - grid_closest_dists < threshold/60 * drivespeed)
    # grid_within_threshold_points = grid_points[grid_within_threshold_bools, :]
    # hull = ConvexHull(grid_within_threshold_points)
    return grid_points, grid_within_threshold_bools.reshape(x.shape)

def single_map_analysis_output(sim_results, map_number = 0, heatmap_diff = True, save = True, output_dir_str = None, additional_file_name = '', threshold = None, line_errorbars = False, generated_map = True, theoretical_ci = False):
    '''
    Takes direct outputted pd.DataFrame (after destination type is added)

    Filters the pd.DataFrame down to the correct map and calls the interval
    and heatmap functions

    Can save tables to a file in output_dir_str, either as an Excel file 
    '''
    # print(sim_results['map_number'].value_counts())
    # print('map_number',map_number)
    class_df, time_df, mRS_df = single_map_results(sim_results, map_number = map_number)

    if not theoretical_ci:
        class_mean_df, class_intervals_df = calculate_intervals(class_df)
        time_mean_df, time_intervals_df = calculate_intervals(time_df)
        mRS_mean_df, mRS_intervals_df = calculate_intervals(mRS_df)
    else:
        class_mean_df, class_intervals_df = calculate_intervals_theoretical(class_df)
        time_mean_df, time_intervals_df = calculate_intervals_theoretical(time_df)
        mRS_mean_df, mRS_intervals_df = calculate_intervals_theoretical(mRS_df)

    if save:
        if output_dir_str is None:
            output_dir_str = 'output'
        output_dir = pathlib.Path(f"{output_dir_str}/map_{str(map_number).zfill(3)}")
        if not output_dir.is_dir():
            output_dir.mkdir(parents = True)

        # map_csv_path = output_dir.parent.parent / 'maps.csv'
        # if map_csv_path.exists():
        #     map_csv_path.unlink()

        output_file = output_dir / f'{additional_file_name}{'_' if additional_file_name != '' else ''}map_{map_number}.xlsx'
        
        try:
            with pd.ExcelWriter(output_file) as writer:
                class_mean_df.to_excel(writer, sheet_name = 'Triage metrics')
                class_intervals_df.to_excel(writer, sheet_name = 'Triage metric intervals')
                time_mean_df.to_excel(writer, sheet_name = 'Time metrics')
                time_intervals_df.to_excel(writer, sheet_name = 'Time metric intervals')
                mRS_mean_df.to_excel(writer, sheet_name = 'mRS metrics')
                mRS_intervals_df.to_excel(writer, sheet_name = 'mRS metric intervals')
        except:
            print(f'{output_file} failed to write excel')

        generate_heatmap(class_df, output_path = output_dir, title_str = f"Map {map_number} Triage", col_names = ['undertriage','overtriage'], additional_file_name = additional_file_name, differenced = heatmap_diff, save = save)
        # generate_heatmap(class_df, output_path = output_dir, title_str =f"Map {map_number} Triage", col_names = ['undertriage','overtriage'], additional_file_name = additional_file_name, differenced = (not heatmap_diff), save = save)
        generate_heatmap(mRS_df, output_path = output_dir, title_str = f"Map {map_number} mRS", differenced = heatmap_diff, save = save, additional_file_name = additional_file_name, col_names = ['ischemic_patients', 'lvo_patients'])
        # generate_heatmap(mRS_df, output_path = output_dir, title_str = f"Map {map_number} mRS", differenced = (not heatmap_diff), save = save, additional_file_name = additional_file_name)

        generate_heatmap(time_df, output_path = output_dir, title_str = f"Map{map_number} Time", col_names = ['ivt_ischemic_mean', 'evt_lvo_mean'], additional_file_name = additional_file_name, differenced = heatmap_diff, save = save)

        generate_line_graphs(class_df, output_path = output_dir, title_str = f"Map {map_number} Triage", col_names = ['undertriage','overtriage'], additional_file_name = additional_file_name, differenced = heatmap_diff, save = save, errorbar = line_errorbars)

        generate_line_graphs(mRS_df, output_path = output_dir, title_str = f"Map {map_number} mRS", differenced = heatmap_diff, save = save, additional_file_name = additional_file_name, col_names = ['ischemic_patients', 'lvo_patients'], errorbar = False)

        generate_line_graphs(time_df, output_path = output_dir, title_str = f"Map{map_number} Time", col_names = ['ivt_ischemic_mean', 'evt_lvo_mean'], additional_file_name = additional_file_name, differenced = heatmap_diff, save = save, errorbar = line_errorbars)

        if generated_map:
            get_map_plot(sim_results, map_number = map_number, output_path = output_dir, threshold = threshold, additional_file_name=additional_file_name, save = save)

    classification_metrics = ['overtriage','undertriage','overtriage_diff','undertriage_diff']
    time_metrics = ['ivt_ischemic_mean', 'evt_lvo_mean','ivt_ischemic_mean_diff','evt_lvo_mean_diff']
    mRS_metrics = ['ischemic_patients', 'lvo_patients','ischemic_patients_diff','lvo_patients_diff']

    retval = class_df.loc[:, classification_metrics].copy()
    retval = pd.concat(
        [retval, time_df.loc[:, time_metrics]], axis = 1
    )
    retval = pd.concat(
        [retval, mRS_df.loc[:, mRS_metrics]], axis = 1
    )
    retval['map'] = np.full(retval.shape[0], map_number)


    # retval = class_mean_df.loc[:, classification_metrics].copy()
    # retval = pd.concat(
    #     [retval, time_mean_df.loc[:, time_metrics]], axis = 1
    # )
    # retval = pd.concat(
    #     [retval, mRS_mean_df.loc[:, mRS_metrics]], axis = 1
    # )    
    return get_thresholds_sensitivities(retval)

def single_map_analysis_output_psc(sim_results, **kwargs):
    sim_results_psc = sim_results.loc[sim_results['closest_destination'] != 'CSC', :]
    return single_map_analysis_output(sim_results_psc, **kwargs)

def get_map_output_path(map_number, output_dir = 'output'):
    return pathlib.Path(f'{output_dir}/map_{str(map_number).zfill(3)}')

def read_output(filestr, save_format = 'csv', config = None):
    match save_format:
        case 'csv':
            sim_results = pd.read_csv(filestr)
        case 'parquet':
            sim_results = pd.read_parquet(filestr)
    try: # Requires numpy 2
        if config is None:
            sim_results['destination_type'] = np.where(np.strings.find(sim_results['destination'].values.astype(np.dtypes.StringDType), 'CSC') >= 0, 'CSC', 'PSC')
        else:
            sim_results['destination_type'] = np.repeat([''], repeats = sim_results.shape[0])

            sim_results['destination_type'][sim_results['destination'].str.contains(config['csc_prefix'])] = 'CSC'
            sim_results['destination_type'][sim_results['destination'].str.contains(config['psc_prefix'])] = 'PSC'
            sim_results['destination_type'][sim_results['destination'].str.contains(config['nsc_prefix'])] = 'NSC'
    except:
        def destination_type_func(row):
            if 'PSC' in row['destination']:
                return 'PSC'
            else:
                return 'CSC'

        sim_results["destination_type"] = sim_results.apply(destination_type_func, axis = 1)
    return sim_results

def process_chunks(chunk):
    '''
    destination_type calculation for a chunk from read_csv
    '''
    try: # Requires numpy 2
        chunk['destination_type'] = np.where(np.strings.find(chunk['destination'].values.astype(np.dtypes.StringDType), 'CSC') >= 0, 'CSC', 'PSC')
    except:
        def destination_type_func(row):
            if 'PSC' in row['destination']:
                return 'PSC'
            else:
                return 'CSC'

        chunk["destination_type"] = chunk.apply(destination_type_func, axis = 1)
    return chunk

def read_csv_with_header(file_path, chunksize=1000):
    """Reads a CSV file in chunks, yielding the header separately."""

    # Read the header row
    header = pd.read_csv(file_path, nrows=1).columns

    # Read the rest of the file in chunks
    for chunk in pd.read_csv(file_path, chunksize=chunksize, skiprows=1, header=None):
        # Assign the header to the chunk
        chunk.columns = header
        yield chunk

def generate_maps_csv(map_num, maps_csv_path, save = True):
    '''
    Function to create maps.csv without having to rerun full analysis code
    '''
    
    _, coords, geoscale = generate_map(map_num)
    actual_coords = geoscale * coords
    drivespeed = get_drivespeed(geoscale)
    temp_df = pd.DataFrame.from_dict(
        {
            'map_number': [map_num],
            'xPSC': [actual_coords[1,0]],
            'yPSC': [actual_coords[1,1]],
            'xPSC2': [actual_coords[2,0]],
            'yPSC2': [actual_coords[2,1]],
            'geoscale': [geoscale],
            'drivespeed': [drivespeed]
        }
    )
    return get_map_plot(temp_df, map_number = map_num, output_path = maps_csv_path, save = save, save_map_csv = False)

##############################################

def preprocess_data(df, config = None):
    '''
    Preprocesses raw simulation output for cohort grouping
    '''
    # map_number = int(filepath.stem.split('_')[1])
    # df = pd.read_parquet(filepath)
    df = df.loc[~df['scenario'].isin([8, 15])]
    df.loc[:, 'diagnostic'] = df['sensitivity'].replace(
        to_replace = {0.9: 'high', 0.75: 'mid', 0.6: 'low'}
    )
    df.loc[df['threshold'] == 0, 'diagnostic'] = 'base'
    df['destination_type'] = df['destination'].copy()
    try:
        df.loc[df['destination'].str.contains(config['csc_prefix']), 'destination_type'] = 'CSC'
        df.loc[df['destination'].str.contains(config['psc_prefix']), 'destination_type'] = 'PSC'
        df.loc[df['destination'].str.contains(config['nsc_prefix']), 'destination_type'] = 'NSC'
    except:
        df.loc[df['destination'].str.contains('PSC'), 'destination_type'] = 'PSC'
        df.loc[df['destination'].str.contains('CSC'), 'destination_type'] = 'CSC'
    return df

def calc_triage(df):
    '''
    Calculates over- and undertriage for each cohort and scenario combination
    '''
    triage_indicators = df.loc[df['closest_destination'] != 'CSC', ['seed', 'diagnostic', 'threshold', 'hasLVO', 'destination_type']]
    triage_indicators['overtriage_ind'] = (triage_indicators['destination_type'] == 'CSC') & (~triage_indicators['hasLVO'])
    triage_indicators['undertriage_ind'] = (triage_indicators['destination_type'] != 'CSC') & (triage_indicators['hasLVO'])

    overtriage_undertriage = triage_indicators.drop( 'destination_type', axis = 1).groupby(['seed', 'diagnostic', 'threshold']).sum()
    lvo_counts = triage_indicators[['seed', 'diagnostic', 'threshold', 'hasLVO']].groupby(['seed', 'diagnostic', 'threshold']).sum()
    no_lvo_counts = (triage_indicators[['seed', 'diagnostic', 'threshold', 'hasLVO']].groupby(['seed', 'diagnostic', 'threshold']).count() - lvo_counts).rename({'hasLVO': 'no_lvo_count'}, axis = 1)
    lvo_counts.rename({'hasLVO': 'lvo_count'}, axis = 1, inplace = True)
    no_lvo_counts.rename({'hasLVO': 'no_lvo_count'}, axis = 1, inplace = True)
    overtriage_undertriage = overtriage_undertriage.join((lvo_counts, no_lvo_counts), validate = '1:1')
    overtriage_undertriage['undertriage'] = overtriage_undertriage['undertriage_ind'] / (overtriage_undertriage['lvo_count'])
    overtriage_undertriage['overtriage'] = overtriage_undertriage['overtriage_ind'] / overtriage_undertriage['no_lvo_count']
    return overtriage_undertriage[['overtriage','undertriage']]

def calc_time(df):
    '''
    Calculates IVT and EVT time by cohort and scenario combination
    '''
    ivt_times = df.loc[df['IVTtreatment'], ['seed', 'diagnostic', 'threshold', 'IVTtime']]
    evt_times = df.loc[df['EVTtreatment'], ['seed', 'diagnostic', 'threshold', 'EVTtime']]
    ivt_cohort_avg = ivt_times.groupby(['seed', 'diagnostic', 'threshold']).mean()
    evt_cohort_avg = evt_times.groupby(['seed', 'diagnostic', 'threshold']).mean()
    return ivt_cohort_avg.join(evt_cohort_avg, validate = '1:1')

def calc_mRS(df):
    '''
    Calculates probability of good mRS by cohort and scenario combination
    '''
    ischemic_mRS = df.loc[(df['ischemic']), ['seed', 'diagnostic', 'threshold', 'PrOut']].rename({'PrOut': 'mRS_ischemic'}, axis = 1)
    lvo_mRS = df.loc[(df['hasLVO']), ['seed', 'diagnostic', 'threshold', 'PrOut']].rename({'PrOut': 'mRS_lvo'}, axis = 1)
    mRS_ischemic_cohort_avg = ischemic_mRS.groupby(['seed', 'diagnostic','threshold']).mean()
    mRS_lvo_cohort_avg = lvo_mRS.groupby(['seed', 'diagnostic', 'threshold']).mean()
    return mRS_ischemic_cohort_avg.join(mRS_lvo_cohort_avg, validate = '1:1')

def calc_counts_props(df):
    '''
    Calculates counts and proportions of ischemic and LVO patients who received IVT and EVT
    '''
    ischemic_indicators = df.loc[df['ischemic'], ['seed', 'diagnostic', 'threshold', 'IVTtreatment', 'ischemic']]
    lvo_indicators = df.loc[df['hasLVO'], ['seed', 'diagnostic', 'threshold', 'EVTtreatment', 'hasLVO']]

    ischemic_count = ischemic_indicators.drop('IVTtreatment', axis = 1).groupby(['seed', 'diagnostic', 'threshold']).count().rename({'ischemic': 'ischemic_count'}, axis = 1)
    ischemic_ivt_count = ischemic_indicators.loc[ischemic_indicators['IVTtreatment'], :].drop('IVTtreatment', axis = 1).groupby(['seed', 'diagnostic', 'threshold']).count().rename({'ischemic': 'ischemic_ivt_count'}, axis = 1)

    lvo_count = lvo_indicators.drop('EVTtreatment', axis = 1).groupby(['seed', 'diagnostic', 'threshold']).count().rename({'hasLVO': 'lvo_count'}, axis = 1)
    lvo_evt_count = lvo_indicators.loc[lvo_indicators['EVTtreatment'], :].drop('EVTtreatment', axis = 1).groupby(['seed', 'diagnostic', 'threshold']).count().rename({'hasLVO': 'lvo_evt_count'}, axis = 1)

    retval = ischemic_count.join(
        (ischemic_ivt_count, lvo_count, lvo_evt_count),
        validate = '1:1'
    )
    retval['ischemic_ivt_prop'] = retval['ischemic_ivt_count'] / retval['ischemic_count']
    retval['lvo_evt_prop'] = retval['lvo_evt_count'] / retval['lvo_count']
    return retval

def process_data(filepath = None, plots = True, errorbars = False, additional_file_name = None, config = None, psc_only = False, output_dir = None, intervals = True, interval_width = 0.95, save_format = 'parquet', df = None, map_number = None):
    '''
    Analyzes the simulation output

    Optionally generates confidence intervals
    '''
    if not output_dir.exists():
        output_dir.mkdir(parents = True)
    if filepath is None and df is None:
        raise Exception("Need to provide either the filepath to the parquet file storing simulation output or the dataframe itself")
    if df is None:
        map_number = int(filepath.stem.split('_')[1])
        df = pd.read_parquet(filepath)
    else:
        if map_number is None:
            raise Exception("Provide a map_number argument so that the output files can be saved accordingly")
    df = preprocess_data(df, config = config)
    if psc_only:
        if config is None:
            df = df.loc[~(df['closest_destination'].str.contains('CSC')), :]
        else:
            df = df.loc[~(df['closest_destination'].str.contains(config['csc_prefix'])), :]
    triage_avgs = calc_triage(df)
    time_avgs = calc_time(df)
    mRS_avgs = calc_mRS(df)
    props = calc_counts_props(df)
    grouped_avgs = triage_avgs.join(
        (time_avgs, mRS_avgs, props), validate = '1:1'
    )
    grouped_avgs_diff = grouped_avgs.copy()
    for i in np.unique(grouped_avgs.index.get_level_values('seed').values):
        grouped_avgs_diff.loc[pd.IndexSlice[i, :, :], :] = grouped_avgs.loc[pd.IndexSlice[i, :, :], :] - grouped_avgs.loc[pd.IndexSlice[i, 'base', 0], :]
    grouped_avgs_diff.drop(['ischemic_count', 'lvo_count'], axis = 1, inplace = True)
    grouped_avgs_diff.rename(lambda x: x+'_diff', axis = 1, inplace = True)

    joined_avgs = grouped_avgs.join(grouped_avgs_diff, validate = '1:1')
    joined_avgs['map'] = map_number
    match save_format:
        case 'csv':
            if filepath is not None:
                joined_avgs.to_csv(output_dir / f'map_{str(map_number).zfill(3)}' / f'{'psc_' if psc_only else ''}{additional_file_name}{'_' if additional_file_name != '' else ''}{filepath.stem}.csv')
            else:
                joined_avgs.to_csv(output_dir / f'map_{str(map_number).zfill(3)}' / f'{'psc_' if psc_only else ''}{additional_file_name}{'_' if additional_file_name != '' else ''}map_{map_number}.csv')
        case 'parquet':
            if filepath is not None:
                joined_avgs.to_parquet(output_dir / f'map_{str(map_number).zfill(3)}' / f'{'psc_' if psc_only else ''}{additional_file_name}{'_' if additional_file_name != '' else ''}{filepath.stem}.parquet')
            else:
                joined_avgs.to_parquet(output_dir / f'map_{str(map_number).zfill(3)}' / f'{'psc_' if psc_only else ''}{additional_file_name}{'_' if additional_file_name != '' else ''}map_{map_number}.parquet')
    intervals_df = None
    if intervals:
        k = np.unique(grouped_avgs.index.get_level_values('seed').values).shape[0]
        alpha = (1 - interval_width) / 2
        full_grouped_avgs = joined_avgs.drop('map', axis = 1).reset_index(['diagnostic', 'threshold']).groupby(['diagnostic', 'threshold'])
        means = full_grouped_avgs.mean()
        variance = full_grouped_avgs.var()
        lower = means - stats.t.ppf(1 - alpha, df = k - 1) * np.sqrt(variance / k)
        upper = means + stats.t.ppf(1 - alpha, df = k - 1) * np.sqrt(variance / k)

        means.rename(lambda x: x+'_means', inplace = True, axis = 1)
        lower.rename(lambda x: x+'_lower', inplace = True, axis = 1)
        upper.rename(lambda x: x+'_upper', inplace = True, axis = 1)

        intervals_df = means.join((lower, upper), validate = '1:1')
        intervals_df = intervals_df.sort_index(axis = 1)

        if filepath is not None:
            intervals_df.to_csv(output_dir / f'map_{str(map_number).zfill(3)}' / f'{'psc_intervals_' if psc_only else 'intervals_'}{(additional_file_name+'_') if additional_file_name is not None else ''}{filepath.stem}.csv')
        else:
            intervals_df.to_csv(output_dir / f'map_{str(map_number).zfill(3)}' / f'{'psc_intervals_' if psc_only else 'intervals_'}{(additional_file_name+'_') if additional_file_name is not None else ''}map_{map_number}.csv')
        # match save_format:
        #     case 'csv':
        #         if filepath is not None:
        #             intervals_df.to_csv(output_dir / f'{'psc_intervals_' if psc_only else 'intervals_'}{additional_file_name}{'_' if additional_file_name != '' else ''}{filepath.stem}.csv')
        #         else:
        #             intervals_df.to_csv(output_dir / f'{'psc_intervals_' if psc_only else 'intervals_'}{additional_file_name}{'_' if additional_file_name != '' else ''}map_{map_number}.csv')
        #     case 'parquet':
        #         if filepath is not None:
        #             intervals_df.to_parquet(output_dir / f'{'psc_intervals_' if psc_only else 'intervals_'}{additional_file_name}{'_' if additional_file_name != '' else ''}{filepath.stem}.parquet')
        #         else:
        #             intervals_df.to_parquet(output_dir / f'{'psc_intervals_' if psc_only else 'intervals_'}{additional_file_name}{'_' if additional_file_name != '' else ''}map_{map_number}.parquet')
    if plots:
        sns.set_theme()
        if errorbars:
            errorbar = ('ci', interval_width)
        else:
            errorbar = None
        # diffed_avgs = joined_avgs.drop(joined_avgs.filter(regex = '_diff', axis = 1), axis = 1)
        diffed_avgs = joined_avgs.reset_index()
        diffed_avgs = diffed_avgs.loc[diffed_avgs['threshold'] > 0, :]
        
        # triage_fig, triage_axes = plt.subplots(1, 2)
        # sns.lineplot(diffed_avgs, x = 'threshold', y = 'overtriage_diff', hue = 'diagnostic', marker = 'o', errorbar = errorbar, ax = triage_axes[0])
        # sns.lineplot(diffed_avgs, x = 'threshold', y = 'undertriage_diff', hue = 'diagnostic', marker = 'o', errorbar = errorbar, ax = triage_axes[1])
        # triage_axes[0].set_title('overtriage')
        # triage_axes[1].set_title('undertriage')
        triage_vals = pd.melt(diffed_avgs[['diagnostic','threshold','overtriage_diff','undertriage_diff']], id_vars = ['diagnostic', 'threshold'], var_name = 'triage', value_name = 'val')
        triage_plots = sns.relplot(triage_vals, x = 'threshold', y = 'val', col = 'triage', hue = 'diagnostic', kind = 'line', facet_kws = {'sharey': False})

        # time_fig, time_axes = plt.subplots(1, 2)
        # sns.lineplot(diffed_avgs, x = 'threshold', y = 'IVTtime_diff', hue = 'diagnostic', marker = 'o', errorbar = errorbar, ax = time_axes[0])
        # sns.lineplot(diffed_avgs, x = 'threshold', y = 'EVTtime_diff', hue = 'diagnostic', marker = 'o', errorbar = errorbar, ax = time_axes[1])
        # time_axes[0].set_title('IVT time')
        # time_axes[1].set_title('EVT time')
        time_vals = pd.melt(diffed_avgs[['diagnostic','threshold','IVTtime_diff','EVTtime_diff']], id_vars = ['diagnostic', 'threshold'], var_name = 'time', value_name = 'val')
        time_plots = sns.relplot(time_vals, x = 'threshold', y = 'val', col = 'time', hue = 'diagnostic', kind = 'line', facet_kws = {'sharey': False})

        # mRS_fig, mRS_axes = plt.subplots(1, 2)
        # sns.lineplot(diffed_avgs, x = 'threshold', y = 'mRS_ischemic_diff', hue = 'diagnostic', marker = 'o', errorbar = errorbar, ax = mRS_axes[0])
        # sns.lineplot(diffed_avgs, x = 'threshold', y = 'mRS_lvo_diff', hue = 'diagnostic', marker = 'o', errorbar = errorbar, ax = mRS_axes[1])
        # mRS_axes[0].set_title('ischemic')
        # mRS_axes[1].set_title('LVO')
        mRS_vals = pd.melt(diffed_avgs[['diagnostic','threshold','mRS_ischemic_diff','mRS_lvo_diff']], id_vars = ['diagnostic', 'threshold'], var_name = 'mRS', value_name = 'val')
        mRS_plots = sns.relplot(mRS_vals, x = 'threshold', y = 'val', col = 'mRS', hue = 'diagnostic', kind = 'line', facet_kws = {'sharey': False})

        # triage_fig.savefig(output_dir / f'{'psc_' if psc_only else ''}{additional_file_name if additional_file_name is not None else ''}{'_' if additional_file_name != '' else ''}map_{map_number}_triage_plot.png')

        # time_fig.savefig(output_dir / f'{'psc_' if psc_only else ''}{additional_file_name if additional_file_name is not None else ''}{'_' if additional_file_name != '' else ''}map_{map_number}_time_plot.png')

        # mRS_fig.savefig(output_dir / f'{'psc_' if psc_only else ''}{additional_file_name if additional_file_name is not None else ''}{'_' if additional_file_name != '' else ''}map_{map_number}_mRS_plot.png')
        triage_plots.savefig(output_dir / f'map_{str(map_number).zfill(3)}'/ f'{'psc_' if psc_only else ''}{additional_file_name if additional_file_name is not None else ''}{'_' if additional_file_name != '' else ''}map_{map_number}_triage_plot.png')

        time_plots.savefig(output_dir / f'map_{str(map_number).zfill(3)}'/ f'{'psc_' if psc_only else ''}{additional_file_name if additional_file_name is not None else ''}{'_' if additional_file_name != '' else ''}map_{map_number}_time_plot.png')

        mRS_plots.savefig(output_dir / / f'map_{str(map_number).zfill(3)}'f'{'psc_' if psc_only else ''}{additional_file_name if additional_file_name is not None else ''}{'_' if additional_file_name != '' else ''}map_{map_number}_mRS_plot.png')

    return joined_avgs, intervals_df
