import pandas as pd
import pathlib
import argparse
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', help = 'file containing the ci widths', type = pathlib.Path, default = None)
args = parser.parse_args()

ci_widths = pd.read_csv(args.file)
ci_widths = ci_widths.iloc[:-5, :].drop('map', axis = 1)

long_pivot_ci_widths = pd.melt(ci_widths,
                               id_vars = ['num_cohorts', 'num_patients'],
                               value_vars = ['ivt_ischemic_mean', 'evt_lvo_mean'],
                               var_name = 'time_metric',
                               value_name = 'avg_half_width')

widths_plot = sns.relplot(data = long_pivot_ci_widths,
            x = 'num_cohorts',
            y = 'avg_half_width',
            hue = 'num_patients',
            col = 'time_metric',
            kind = 'line')

widths_plot.savefig(args.file.parent / 'avg_ci_widths.png')