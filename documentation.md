# EMS Stroke Triage and Transport Model (ESTTM)

For quick run, run the following:
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

### Configuration Files

There are 4 types of configuration files accepted to update the simulation parameters or input data:
1. YAML file containing simulation parameter overrides
2. CSV file containing patient information on LKW and hex information
3. CSV file containing transport times from each hex to each hospital and from each hospital to all hospitals
4. CSV file containing probabilities or counts for determining which hospital patients are initially transported to

#### YAML file

Example file: `config_files/sampson_test.yaml`

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

For example, see `config_files/sampson_test.yaml`

#### Transport times

This CSV file should contain a matrix of travel times such that:
* Each row corresponds to the travel time from a hex or a hospital to all hospitals
* The hospitals form the last few rows
* Hospitals are coded using the prefixes in the config file (or the default)

See `input_data/sampson_nsc_times.csv` for an example.

#### Transport probabilities

CSV file containing empirical counts or probabilities on where patients are initially sent given the hex they spawn in.
* Each row corresponds to a hex 

See `input_data/sampson_hex_hosp_probs.csv` for an example

Contents from the above files get read into a dict that can be modified separately.

## Postprocessing

The code for postprocessing is in `scripts/sim_code/postprocess_simulation_results.py`. 