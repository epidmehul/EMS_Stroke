from stroke_simulation import *
from postprocess_simulation_results import *
import multiprocessing as mp
import argparse
import pathlib
from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help = 'config file with simulation parameters', type = pathlib.Path, default = None)
parser.add_argument('-d', '--data', help = 'data file containing patient information on hexes and LKW times', type = pathlib.Path, default = None)
parser.add_argument('-t', '--times', help = 'data file containing travel times from hexes and hospitals to hospitals', type = pathlib.Path, default = None)
parser.add_argument('-n', '--n_cores', help = 'number of cores for mp.Pool', type = int, default = 10)
parser.add_argument('-f', '--cohort_file', help = 'file containing the cohort configurations and map seed number', type = pathlib.Path)
parser.add_argument('-w', '--width', help = 'confidence interval width as a proportion', type = float, default = 0.9)
parser.add_argument('-o', '--output', help = 'output directory for simulation and analysis', type = pathlib.Path, default = '/work/users/p/w/pwlin/output2/parquet_files')
args = parser.parse_args()

num_cores = args.n_cores
cohort_runs = pd.read_csv(args.cohort_file)
cohort_list = list(cohort_runs.itertuples(index = False, name = None))

# map_seeds = [i for i in range(1000)]
# patient_seeds = [i for i in range(args.seeds)]


try:
    config_dict = read_config(args.config, args.data, args.times)
except:
    config_dict = None

def run_map_combo(cohort_option):
    map_seed, num_cohorts, num_patients = cohort_option
    run_map_simulations([map_seed], num_patients = num_patients, num_patient_seeds = num_cohorts, save_format = 'parquet', output_dir = output_dir, config = config_dict)

def analyze_parquet(cohort_option):
    map_num = cohort_option[0]
    file_name = f'map_{str(map_num).zfill(3)}.parquet'
    df = read_output(pathlib.Path(output_dir) / file_name, save_format = 'parquet')
    return single_map_analysis_output(df, map_number = map_num, heatmap_diff = True, save = True, output_dir_str = '/work/users/p/w/pwlin/output2/results', line_errorbars = True, generated_map = False) 

def run_analyze(cohort_option):
    print(cohort_option)
    map_seed, num_cohorts, num_patients = cohort_option
    run_map_simulations([map_seed], num_patients = num_patients, num_patient_seeds = num_cohorts, save_format = 'parquet', output_dir = output_dir, config = config_dict)
    file_name = f'map_{str(map_seed).zfill(3)}.parquet'
    df = read_output(pathlib.Path(output_dir) / file_name, save_format = 'parquet')
    return single_map_analysis_output(df, map_number = map_seed, heatmap_diff = True, save = True, output_dir_str = '/work/users/p/w/pwlin/output2/results', line_errorbars = True, generated_map = False, theoretical_ci = True) 

def get_time_ci(df, map_number, output_dir_str = None,):
    seeds = df['seed'].unique()
    time_df_list = []
    for seed in seeds:
        time_outcomes = {}
        df_dicts = map_df_to_dict(df, None, seed)

        small_df = df_dicts['base']
        ivt = small_df.loc[(small_df['ischemic'] & small_df['IVTtime'] <= 270 & small_df['IVTtreatment']), 'IVTtime']
        evt = small_df.loc[(small_df['hasLVO']) & (small_df['EVTtime'] <= 24 * 60) & (small_df['EVTtreatment']),'EVTtime']
        time_outcomes['base'] = {
            'ivt_ischemic_mean': ivt.mean(),
            'evt_lvo_mean': evt.mean()
        }

        for i in ('high', 'mid', 'low'):
            for thresh in range(10, 70, 10):
                small_df = df_dicts[i + '_sens_' +str(thresh)]
                ivt = small_df.loc[(small_df['ischemic'] & small_df['IVTtime'] <= 270 & small_df['IVTtreatment']), 'IVTtime']
                evt = small_df.loc[(small_df['hasLVO']) & (small_df['EVTtime'] <= 24 * 60) & (small_df['EVTtreatment']),'EVTtime']
                time_outcomes[i + '_sens_' + str(thresh)] = {
                    'ivt_ischemic_mean': ivt.mean(),
                    'evt_lvo_mean': evt.mean()
                }
        time_df = pd.DataFrame.from_dict(time_outcomes).transpose()
        time_df_list.append(add_differences_columns(get_thresholds_sensitivities(time_df)))
    full_time_df = pd.concat(time_df_list)
    means = full_time_df.groupby(['sensitivity', 'threshold'], observed = True).mean()
    vars = full_time_df.groupby(['sensitivity','threshold'], observed = True).var()
    half_widths = (stats.t.ppf(1 - (1-args.width)/2, df = seeds.shape[0] - 1) * np.sqrt(vars / (seeds.shape[0])))
    half_widths_avg = half_widths.mean()
    
    # upper_ci = intervals.loc[:,pd.IndexSlice[:, 1-(1-args.width)/2]]
    # lower_ci = intervals.loc[:,pd.IndexSlice[:, (1-args.width)/2]]

    # upper_ci.columns = upper_ci.columns.droplevel(1)
    # lower_ci.columns = lower_ci.columns.droplevel(1)
    # ci_widths = (upper_ci - lower_ci).mean(axis = 0)
    try:
        lower_ci = means - half_widths
        upper_ci = means + half_widths

        lower_ci.columns = pd.MultiIndex.from_product(lower_ci.columns.levels + [[(1- args.width)/2]])
        upper_ci.columns = pd.MultiIndex.from_product(lower_ci.columns.levels + [[1 - (1- args.width)/2]])
        intervals = pd.concat((lower_ci, upper_ci), axis = 1)

        output_dir = pathlib.Path(output_dir_str)
        if not output_dir.exists():
            output_dir.mkdir(parents = True)
        # output_file = output_dir / f'map_{map_number}.xlsx'
        # with pd.ExcelWriter(output_file) as writer:
        #     intervals.to_excel(writer, sheet_name = 'Time metric intervals')
        intervals.to_csv(output_dir / 'time_ci.csv', index = False)
    except:
        print('failed calculating or saving actual interval')
    return half_widths_avg

def run_analyze_time_ci_widths(cohort_option):
    map_seed, num_cohorts, num_patients = cohort_option
    run_map_simulations([map_seed], num_patients = num_patients, num_patient_seeds = num_cohorts, save_format = 'parquet', output_dir = output_dir, config = config_dict)
    file_name = f'map_{str(map_seed).zfill(3)}.parquet'
    df = read_output(pathlib.Path(output_dir) / file_name, save_format = 'parquet', config = config_dict)
    ci_width =  get_time_ci(df, map_number = map_seed, output_dir_str = args.output.parent) 

    ci_width['map'] = map_seed
    ci_width['num_cohorts'] = num_cohorts
    ci_width['num_patients'] = num_patients
    return pd.DataFrame(ci_width.reindex(index = ['map', 'num_cohorts', 'num_patients', 'ivt_ischemic_mean', 'ivt_ischemic_mean_diff', 'evt_lvo_mean', 'evt_lvo_mean_diff'])).transpose()

if __name__ == '__main__':
    output_dir = args.output
    output_dir_path = pathlib.Path(output_dir)
    if not output_dir_path.exists():
        output_dir_path.mkdir(parents = True)
    with mp.Pool(num_cores) as pool:
        results = pool.map(run_analyze_time_ci_widths, cohort_list)
    pd.concat(results).to_csv(output_dir_path.parent / 'avg_ci_half_widths.csv', index = False)
    # ivt_widths = []
    # ivt_diff_widths = []
    # evt_widths = []
    # evt_diff_widths = []
    # for i in range(len(cohort_list)):
    #     map_number = cohort_list[i][0]
    #     temp_data = pd.read_excel(output_dir_path.parent / 'results' / f'map_{str(map_number).zfill(3)}' / f'map_{map_number}.xlsx', 
    #                               sheet_name = 'Time metric intervals',
    #                               header = [0, 1],
    #                               index_col = [0, 1])
    #     ivt_ci = temp_data['ivt_ischemic_mean'].values
    #     ivt_diff_ci = temp_data['ivt_ischemic_mean_diff'].values
    #     evt_ci = temp_data['evt_lvo_mean'].values
    #     evt_diff_ci = temp_data['evt_lvo_mean_diff'].values

    #     ivt_widths.append((ivt_ci[:, 1] - ivt_ci[:, 0]).mean())
    #     ivt_diff_widths.append((ivt_diff_ci[:, 1] - ivt_diff_ci[:, 0]).mean())
    #     evt_widths.append((evt_ci[:, 1] - evt_ci[:, 0]).mean())
    #     evt_diff_widths.append((evt_diff_ci[:, 1] - evt_diff_ci[:, 0]).mean())
    #     # ivt_widths.append(temp_data.loc[:,['ivt_ischemic_mean', '0.95']] - temp_data.loc[:,['ivt_ischemic_mean', '0.05']])
    #     # evt_widths.append(temp_data.loc[:,['evt_lvo_mean', '0.95']] - temp_data.loc[:,['evt_lvo_mean', '0.05']])
    # cohort_runs['ivt_widths'] = ivt_widths
    # cohort_runs['ivt_diff_widths'] = ivt_diff_widths
    # cohort_runs['evt_widths'] = evt_widths
    # cohort_runs['evt_diff_widths'] = evt_diff_widths
    # cohort_runs.to_csv(output_dir_path.parent / 'avg_ci_full_widths.csv', index = False)
        