# EMS Stroke Triage and Transport Model (ESTTM)

For quick run, run the following (after optionally setting up a virtual environment):
```
pip install -r requirements.txt
python scripts/quick_run_analyze.py
```

## Simulation

The simulation functions are in `scripts/sim_code/stroke_simulation.py`.

The main functions to note are `read_config(...)`, `simulation(...)`, and `run_map_simulations(...)`.
* read_config: Reads in data files to update simulation parameters
* simulation: Runs simulation for a single patient cohort on a single map for all scenarios
* run_map_simulations: Runs simulation multiple times to aggregate all patient cohorts on a single map

When running the simulation, there are three possible cases to choose from:
1. All hospitals are valid to transport patients to
2. The simulation uses empirical data to randomly determine which hospital patients are initially transported to
3. Only CSCs and PSCs are used in the simulation (any noncertified hospitals are dropped)

### Configuration Files

There are 4 types of configuration files accepted to update the simulation parameters or input data:
1. YAML file containing simulation parameter overrides
2. CSV file containing patient information on LKW and hex information
3. CSV file containing transport times from each hex to each hospital and from each hospital to all hospitals
4. CSV file containing probabilities or counts for determining which hospital patients are initially transported to

Note that the simulation does not need all 4 files to be provided. 

#### YAML file

Example file: `config_files/test4.yaml`

The following parameters can be entered in YAML format:
* Stroke prevalance
    - patients_none_all: Proportion of all potential patients with no stroke 
    - patients_tia_all: Proportion of all potential patients with TIA
    - patients_hemorrhaging_all: Proportion of all potential patients with hemorrhaging
    - patients_ischemic_all: Proportion of all potential patients with ischemic stroke
    - patients_lvo_ischemic: Proportion of ischemic strokes that are LVO
* Last known well time bins
    - Each bin requires a probability of a stroke patient being within that bin
    - Time distributions come from `numpy.random.Generator` objects and need to be entered as `rng.distribution` since `rng` is the Generator variable used internally
    - Within kwargs, the arguments for the distribution in question need to be specified according to the Numpy documentation.
    - This will be overwritten if a patient data file with valid LKW times is provided
* Hospital time distributions
    - The distributions that can be modified are door2IVT, NSC2IVT, door2EVT, IVT2out, and NSCIVT2out, door2EVT2
        - door2IVT: Door-to-needle time for IVT at a CSC or PSC
        - NSC2IVT: Door-to-needle time for IVT at a noncertified hospital
        - door2EVT: Door-to-treatment time for EVT at a CSC for non-transferred patients
        - IVT2out: IVT-to-departure time at a PSC
        - NSCIVT2out: IVT-to-departure time at a noncertified hospital
        - door2EVT2: Door-to-treatment time at a CSC for a transfer patient
* Treatment parameters
    - Time thresholds and probabilities of treatment
        - simulations_ivt_threshold: Time threshold for IVT
        - simulations_evt_threshold: Time threshold for EVT (only for LVO patients)
        - simulations_ivt_probability: Probability of receiving IVT at a PSC or CSC given IVT time threshold is already met
        - simulations_nsc_ivt_probability: Probability of receiving IVT at a noncertified hospital given IVT time threshold is already met
        - simulations_evt_probability: Probability of receiving EVT given EVT time threshold is already met
        - simulations_early_repurfusion_probability: Probability of early repurfusion after IVT

For example, see `config_files/test4.yaml`

#### Patient information
This CSV file provides information on LKW bins and spawn hexes
* Each row corresponds to a single patient and contains the LKW time and spawn hex ID
- Column name for LKW: last_well
- Column name for hex: hex
- Ultimately, this creates the keys `lkw_bins` and `hexes` in the final configurartion dict that is passed to the simulation function and so these can be manually overwritten after the config dict is created but before the simulation is run 

#### Transport times

This CSV file should contain a matrix of travel times such that:
* Each row corresponds to the travel time from a hex or a hospital to all hospitals
* Rows corresponding to the hospitals are at the bottom, with all hospitals represented
* Hospitals are coded with the prefixes specfied in the provided config file (or the default)

Note that these times are used deterministically and there is no use of random noise to modify travel times.

See `input_data/county_test_all_times.csv` for an example.

#### Transport probabilities

CSV file containing empirical counts or probabilities on where patients are initially sent given the hex they spawn in.
* Each row corresponds to a hex, with columns corresponding to the different hospitals
* Each cell should be the probability or count of patients from that hex being initially transported to that hospital.
* This is only used in base case 2.

See `input_data/county_test_hex_hosp_probs.csv` for an example

### Defaults

Running the simulation without any config files will lead it to fall back to default settings:
- Map is a square grid
- 1 CSC in exact middle of the square grid with 2 randomly generated PSC locations
- Size of the square grid is also randomly generated
- Patients spawn uniformly across the square map

### Simulation Output

`simulation()` returns a  `pandas.DataFrame` with each row corresponding to a simulated patient.

`run_map_simulation()` concatenates these outputs into a single DataFrame that is then saved as a parquet file. 

The outputted dataframe include the following columns:
- Patient ID (within a cohort)
- Cohort number
- Scenario number
    - 1 through 7: high sensitivity and low specificity at thresholds 0, 10, 20, ..., 60 minutes
    - 8 through 14: medium sensitivity and specificity at thresholds 0, 10, ..., 60 minutes
    - 15 through 21: low sensitivity and high specificity at thresholds 0, 10, ..., 60 minutes
    - Note scenarios 1, 8, and 15 are all the base case and postprocessing code removes 8 and 15
- Map number (mainly used to separate runs and create filepaths for saving files)
- Initial hospital destination (considered the default)
- Closest hospital destination
- Actual hospital destination
- Spawn hex (or (x, y) coordinates if using the default square grid)
- Stroke diagnosis indicators
- LKW to door time
- IVT and EVT treatment indicators
- IVT and EVT treatment times (note these are generated for all patients)
- Probability of good outcome (defined as mRS 0-2)
- Coordinate locations of randomly generated hospitals if using randomly generated grid
- Base case number that was used in the run
- Size and EMS driving speed for randomly generated maps

Note if configuration and data files are provided, some of these columns will still have values generated, but as a placeholder

## Postprocessing

The code for postprocessing is in `scripts/sim_code/postprocess_simulation_results.py`.

There are several functions used for processing the simulation output:

Preprocessing removes the duplicate base cases (scenarios 8 and 15) and creates some new variables:
1. `diagnostic` is a conversion of `sensitivity` to `high`, `med`, `low`, or `base`.
2. `destination_type` is the type of hospital that `destination` is (CSC, PSC, or NSC)
3. `initial_type` is the type of hospital that `initial_destination` is
4. `closest_type` is the type of hospital that `closest_destination` is.

Patient cohorts are kept track of by grouping  `seed`, `diagnostic`, and `threshold`. After grouping, the following values are calculated for each cohort:
1. Overtriage and Undertriage for patients not already closest or initially transported to a CSC
    - Overtriage is the proportion of non-LVO patients transported to a CSC
    - Undertriage is the proportion of LVO patients transported to a non-CSC
2. Average time to treatment (from LKW) for patients
    - Time to IVT is calculated for all ischemic patients who received treatment
    - Time to EVT is calculated for all LVO patients who received treatment
3. Average probability of a good outcome (defined as mRS 0-2)
    - Calculated over ischemic stroke patients and separately over LVO patients

Within each scenario, these cohort averages are then averaged together and used to generate confidence intervals. In other words, these calculations result in 1 value for each scenario. Confidence intervals can also be calculated using the averages for each cohort and saved as separate files.

Lastly, plotting is done using seaborn and matplotlib. These plots are saved directly and not explictly displayed by the function. Additionally, stacked bar charts are generated to visualize times until treatment for ischemic and LVO patients to show the change in patients receiving treatment as well as how the times until treatment change.