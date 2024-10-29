from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors
import pathlib
from scipy.spatial import distance
from scipy.spatial import Voronoi, voronoi_plot_2d

def triage_outcomes(df):
    '''
        Given a subsetted pdDataFrame, calculates the confusion matrix for LVO status vs destination type for all patients in df
    '''
    # Arguments: ground truth, predicted label
    # Resulting metrics: Check index 1 for results for class 1 (which here is going to CSC)

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

    retval = {}
    retval['correct_triage'] = ACC[1]
    retval['undertriage'] = FNR[1]
    retval['overtriage'] = FPR[1]
    retval['PPV'] = PPV[1]
    retval['NPV'] = NPV[1]
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
    ivt = time_results(df.loc[(df['ischemic']) & (df['IVTtime'] <= 270),'IVTtime'])
    evt = time_results(df.loc[df['ischemic'],'EVTtime'])
    for key in ['mean', 'std', 'median', 'iqr', 'min', 'max']:
        retval['prehospital_ischemic_' + key] = prehospital[key]
        retval['ems_transport_ischemic_' + key] = ems_transport[key]
        retval['ivt_ischemic_' + key] = ivt[key]
        retval['evt_ischemic_' + key] = evt[key]

    # LVO patients
    prehospital = time_results(df.loc[df['hasLVO'], 'lkw2door'])
    ems_transport = time_results(df.loc[df['hasLVO'],'time2Hospital'])
    ivt = time_results(df.loc[df['hasLVO'],'IVTtime'])
    evt = time_results(df.loc[(df['hasLVO']) & (df['EVTtime'] <= 24 * 60),'EVTtime'])
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
    thresholds_as_list = df.index.str.split('_').str[-1].tolist()
    threshold_idx = thresholds_as_list.index('base')
    thresholds_as_list[threshold_idx] = '0'

    sensitivities_as_list = df.index.str.split('_').str[0].tolist()
    sensitivities_idx = sensitivities_as_list.index('base')
    sensitivities_as_list[sensitivities_idx] = 'base'

    df.insert(0, 'threshold', thresholds_as_list)
    df.insert(1, 'sensitivity', sensitivities_as_list)
    df['threshold'] = df['threshold'].astype(int)
    df['sensitivity'] = df['sensitivity'].astype(pd.api.types.CategoricalDtype(categories = ['base','high', 'mid', 'low'], ordered = True))
    return df

def add_differences_columns(df):
    '''
    Adds additional columns of differences in outcome metrics from base case
    '''    
    metric_names = df.columns.drop(['threshold','sensitivity'], errors = 'ignore')
    for name in metric_names:
        df[name + '_diff'] = df[name] - df.loc['base', name]
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
    alpha = (1 - width)/2
    # df = remove_base_case_and_non_diffs(df)
    retval = df.groupby(['sensitivity', 'threshold'], observed = True).quantile([alpha, 1 - alpha])
    retval_0 = df.groupby(['sensitivity', 'threshold'], observed = True).mean()
    retval.index.set_names(['sensitivity', 'threshold', 'quantile'], inplace = True)
    retval_1 = retval.reset_index().pivot(columns = "quantile", index = ['sensitivity', 'threshold'])
    return retval_0, retval_1    

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
    return ax_list

def get_map_plot(df, map_number = 0, save = True, additional_file_name = '', output_path = None, threshold = None):
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
    # print(equipoise)
    map_csv_file = output_path.parent.parent / 'maps.csv'
    current_map_info = pd.DataFrame(
        {'ID': map_number,
         'equipoise': equipoise,
         'geoscale': geoscale,
         'xPSC': med_coords[1, 0],
         'yPSC': med_coords[1, 1],
         'xPSC2': med_coords[2, 0],
         'yPSC2': med_coords[2, 1]
        }, index = [map_number])

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

def single_map_analysis_output(sim_results, map_number = 0, heatmap_diff = True, save = True, output_dir_str = None, additional_file_name = '', threshold = None, line_errorbars = False):
    '''
    Takes direct outputted pd.DataFrame (after destination type is added)

    Filters the pd.DataFrame down to the correct map and calls the interval
    and heatmap functions

    Can save tables to a file in output_dir_str, either as an Excel file 
    '''
    # print(sim_results['map_number'].value_counts())
    # print('map_number',map_number)
    class_df, time_df, mRS_df = single_map_results(sim_results, map_number = map_number)

    class_mean_df, class_intervals_df = calculate_intervals(class_df)
    time_mean_df, time_intervals_df = calculate_intervals(time_df)
    mRS_mean_df, mRS_intervals_df = calculate_intervals(mRS_df)

    if save:
        if output_dir_str is None:
            output_dir_str = 'output'
        output_dir = pathlib.Path(f"{output_dir_str}/map_{str(map_number).zfill(3)}")
        if not output_dir.is_dir():
            output_dir.mkdir(parents = True)

        map_csv_path = output_dir.parent.parent / 'maps.csv'
        if map_csv_path.exists():
            map_csv_path.unlink()

        output_file = output_dir / f'{additional_file_name}{'_' if additional_file_name != '' else ''}map_{map_number}.xlsx'
        
        with pd.ExcelWriter(output_file) as writer:
            class_mean_df.to_excel(writer, sheet_name = 'Triage metrics')
            class_intervals_df.to_excel(writer, sheet_name = 'Triage metric intervals')
            time_mean_df.to_excel(writer, sheet_name = 'Time metrics')
            time_intervals_df.to_excel(writer, sheet_name = 'Time metric intervals')
            mRS_mean_df.to_excel(writer, sheet_name = 'mRS metrics')
            mRS_intervals_df.to_excel(writer, sheet_name = 'mRS metric intervals')

        generate_heatmap(class_df, output_path = output_dir, title_str = f"Map {map_number} Triage", col_names = ['undertriage','overtriage'], additional_file_name = additional_file_name, differenced = heatmap_diff, save = save)
        # generate_heatmap(class_df, output_path = output_dir, title_str =f"Map {map_number} Triage", col_names = ['undertriage','overtriage'], additional_file_name = additional_file_name, differenced = (not heatmap_diff), save = save)
        generate_heatmap(mRS_df, output_path = output_dir, title_str = f"Map {map_number} mRS", differenced = heatmap_diff, save = save, additional_file_name = additional_file_name, col_names = ['ischemic_patients', 'lvo_patients'])
        # generate_heatmap(mRS_df, output_path = output_dir, title_str = f"Map {map_number} mRS", differenced = (not heatmap_diff), save = save, additional_file_name = additional_file_name)

        generate_heatmap(time_df, output_path = output_dir, title_str = f"Map{map_number} Time", col_names = ['ivt_ischemic_mean', 'evt_lvo_mean'], additional_file_name = additional_file_name, differenced = heatmap_diff, save = save)

        generate_line_graphs(class_df, output_path = output_dir, title_str = f"Map {map_number} Triage", col_names = ['undertriage','overtriage'], additional_file_name = additional_file_name, differenced = heatmap_diff, save = save, errorbar = line_errorbars)

        generate_line_graphs(mRS_df, output_path = output_dir, title_str = f"Map {map_number} mRS", differenced = heatmap_diff, save = save, additional_file_name = additional_file_name, col_names = ['ischemic_patients', 'lvo_patients'], errorbar = False)

        generate_line_graphs(time_df, output_path = output_dir, title_str = f"Map{map_number} Time", col_names = ['ivt_ischemic_mean', 'evt_lvo_mean'], additional_file_name = additional_file_name, differenced = heatmap_diff, save = save, errorbar = line_errorbars)

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

def get_map_output_path(map_number, output_dir = 'output'):
    return pathlib.Path(f'{output_dir}/map_{str(map_number).zfill(3)}')

def read_output(filestr, save_format = 'csv'):
    match save_format:
        case 'csv':
            sim_results = pd.read_csv(filestr)
        case 'parquet':
            sim_results = pd.read_parquet(filestr)
    try: # Requires numpy 2
        sim_results['destination_type'] = np.where(np.strings.find(sim_results['destination'].values.astype(np.dtypes.StringDType), 'CSC') >= 0, 'CSC', 'PSC')
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
