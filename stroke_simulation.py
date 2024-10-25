import numpy as np
import pandas as pd
from scipy import spatial
# from sklearn.metrics import confusion_matrix
import pathlib

def get_drivespeed(geoscale: float):
    '''
    Calculates the EMS driving speed for a square of size geoscale

    Params:
        - geoscale: float

    Returns:
        - speed: float
    '''
    if geoscale <= 70:
        return 25 + (geoscale - 30)/2
    return 45.0

def generate_patient_cohort(num_patients, seed):
    '''
    Params:
        - num_patients: Number of patients to be generated
        - seed: Random seed to initialize the generator 

    Returns:
        - patient_df: pd.DataFrame containing all patient data. Note
                        coordinates are normalized (i.e not multiplied by any geoscale factor)
    '''
    rng = np.random.default_rng(seed)

    patient_coords_normalized = rng.random((num_patients, 2))

    # where does the 0.4 come from?
    stroke = (rng.random(num_patients) < 0.4) 

    # Around 85-87% of strokes due to ischemic event
    hemorrhaging = np.full(num_patients, False)
    ischemic = np.full(num_patients, False)

    hemorrhaging_ischemic_rng = (rng.random(np.sum(stroke)) < 0.13) # indicator for hemorrhaging
    hemorrhaging[stroke] = hemorrhaging_ischemic_rng
    ischemic[stroke] = ~hemorrhaging_ischemic_rng

    # Of ischemic stroke patients, 10-46% depending on definition of LVO (Saini)
    # Up to 40% (Dabus)
    lvo_status = np.full(num_patients, False)
    lvo_status[ischemic] = (rng.random(np.sum(ischemic)) < 0.387) # currently set to 38.7% 

    # probs = np.array([0.44, 0.22, 0.29, 0.05])
    probs = np.array([0.206, 0.062, 0.09, 0.559, 0.083])
    lastWell_bins = rng.choice(a = [i for i in range(1, len(probs) + 1)], p = probs / np.sum(probs, dtype = float), size = num_patients)

    # Note uniform distribution from numpy.random takes different arguments
    # than the uniform distribution from scipy.stats
    # last_well = ( (lastWell_bins == 1) * rng.uniform(0.1, 3, num_patients) + 
    #             (lastWell_bins == 2) * rng.uniform(3, 6, num_patients) + 
    #             (lastWell_bins == 3) * rng.uniform(6, 24, num_patients) + 
    #             (lastWell_bins == 4) * rng.uniform(24, 48, num_patients) )

    
    last_well = ( (lastWell_bins == 1) * rng.uniform(0.1, 2, num_patients) + 
                (lastWell_bins == 2) * rng.uniform(2, 3.5, num_patients) + 
                (lastWell_bins == 3) * rng.uniform(3.5, 8, num_patients) + 
                (lastWell_bins == 4) * rng.uniform(8, 24, num_patients)  +
                (lastWell_bins == 5) * rng.uniform(24, 48, num_patients) )


    patient_df = pd.DataFrame({
        'ID': np.arange(1, num_patients + 1),
        'x_coord': patient_coords_normalized[:,0],
        'y_coord': patient_coords_normalized[:,1],
        'stroke': stroke,
        'hemorrhaging': hemorrhaging,
        'ischemic': ischemic,
        'lvo_status': lvo_status,
        'last_well': last_well,
        'seed': seed
    })

    return patient_df

def generate_map(seed, num_psc = 2):
    '''
    Params:
        - seed: Random seed to initialize the generator
        - num_psc: Number of PSC locations to use

    Returns:
        - med_coords: (num_psc + 1) x 2 array containing hospital coordinates
            Row 0 is hard-coded as (0.5, 0.5) due to being CSC location
        - geoscale: 
    '''
    rng = np.random.default_rng(seed)
    geoscale = rng.uniform(30, 100)
    csc = np.array([0.5, 0.5])
    while True:
        psc_coords = rng.random((num_psc, 2))
        med_coords = np.vstack((csc, psc_coords))
        coord_dists = spatial.distance.pdist(med_coords)

        if np.all(geoscale * coord_dists > 1):
            break
    med_labels = [f'PSC{i}' for i in range(1, num_psc + 1)]
    med_labels.insert(0, 'CSC')
    med_labels = np.array(med_labels)
    return med_labels, med_coords, geoscale

def simulation(num_patients, patient_seed, map_seed, sens_spec_vals = np.array([[0.9, 0.6], [0.75, 0.75], [0.6, 0.9]]), thresholds = np.arange(0, 70, 10)):
    '''
    Runs a simulation for a patient-map combination across all desired LVO diagnosis test parameters and transport thresholds

    Uses (num_patients, num_scenarios, num_thresholds) arrays to store information

    Params:
        - num_patients: Size of the patient cohort to use
        - patient_seed: Seed used for patient cohort
        - map_seed: Seed used for map generation
        - sens_spec_vals: m x 2 array containing LVO diagnosis 
            sensitivity and specificity values
        - thresholds: 1-D array containing the time thresholds to use

    Returns:
        - metrics: DataFrame to be written to the overall map csv
    '''

    ################# Patient and map initialization ##################
    patient_df = generate_patient_cohort(num_patients, seed = patient_seed)
    med_labels, med_coords, geoscale = generate_map(map_seed, num_psc = 2)

    drivespeed = get_drivespeed(geoscale)

    patient_coords = patient_df[['x_coord', 'y_coord']].values
    patient_med_dists = geoscale * spatial.distance.cdist(patient_coords, med_coords)
    patient_med_times = patient_med_dists / drivespeed * 60

    closest_med_ind = np.argmin(patient_med_times, axis = 1)
    closest_med = med_labels[closest_med_ind]
    closest_med_times = np.min(patient_med_times, axis = 1)

    last_well = patient_df['last_well'].values

    rng = np.random.default_rng(patient_seed)
    num_scenarios = sens_spec_vals.shape[0]
    num_thresholds = thresholds.shape[0]

    ################### LVO diagnosis ##############################
    # Calculates the LVO diagnosis for all patients across all sensitivity/specificity parameter values
    lvo_diagnosis_rng = rng.random(num_patients)
    lvo_status = patient_df['lvo_status'].values
    
    expanded_diagnosis_rng = np.expand_dims(lvo_diagnosis_rng, axis = 1)
    expanded_lvo_status = np.expand_dims(lvo_status, axis = 1)
    expanded_sensitivity = np.broadcast_to(sens_spec_vals[:,0], (num_patients, num_scenarios))
    expanded_specificity = np.broadcast_to(sens_spec_vals[:,1], (num_patients, num_scenarios))

    expanded_lvo_diagnosis = (expanded_lvo_status & (expanded_diagnosis_rng < expanded_sensitivity)) | (~expanded_lvo_status & (expanded_diagnosis_rng > expanded_specificity))

    ##################### Destination logic #############################

    correct_destination = closest_med.copy()
    correct_destination_ind = closest_med_ind.copy()
    correct_destination[lvo_status & (last_well <= 24)] = 'CSC'
    correct_destination_ind[lvo_status & (last_well <= 24)] = 0

    destination_arr = np.broadcast_to(np.expand_dims(closest_med, axis = (1, 2)), (num_patients, num_scenarios, num_thresholds)).copy()

    # Eligible to be redirected to CSC under each scenario type
    eligible_patients = (expanded_lvo_diagnosis) & (np.expand_dims(last_well, axis = 1) <= 24)

    eligibility_arr = np.broadcast_to(np.expand_dims(eligible_patients, axis = 2), (num_patients, num_scenarios, num_thresholds))
    thresholds_arr = np.broadcast_to(thresholds, (num_patients, num_scenarios, num_thresholds))
    additional_transport_arr = np.broadcast_to(np.expand_dims(patient_med_times[:,0] - closest_med_times, axis = (1, 2)), (num_patients, num_scenarios, num_thresholds))

    redirected_patients = eligibility_arr & (additional_transport_arr <= thresholds_arr)
    destination_arr[redirected_patients] = 'CSC'

    ##################### Time variables ############################
    # Time to scene
    time_to_scene = 1.62 + rng.normal(15.1, 7, size = num_patients)
    time_to_scene_arr = np.broadcast_to(np.expand_dims(time_to_scene, axis = (1, 2)), (num_patients, num_scenarios, num_thresholds)).copy()

    # Time at scene
    time_at_scene = 40 * rng.beta(2.91, 6.056, size = num_patients)
    time_at_scene_arr = np.broadcast_to(np.expand_dims(time_at_scene, axis = (1, 2)), (num_patients, num_scenarios, num_thresholds)).copy()

    # Time from scene to hospital
    time_to_hospital_arr = np.broadcast_to(np.expand_dims(closest_med_times, axis = (1, 2)), (num_patients, num_scenarios, num_thresholds)).copy()

    patient_csc_times_arr = np.broadcast_to(np.expand_dims(patient_med_times[:,0], axis = (1, 2)), (num_patients, num_scenarios, num_thresholds))

    time_to_hospital_arr[redirected_patients] = patient_csc_times_arr[redirected_patients]

    # Time aggregation
    time_in_system_arr = time_to_scene_arr + time_at_scene_arr + time_to_hospital_arr

    lkw_to_door_arr = 60 * np.broadcast_to(np.expand_dims(last_well, axis = (1, 2)), (num_patients, num_scenarios, num_thresholds)) + time_in_system_arr

    ####################### Outcomes ######################
    door2IVT = 45
    door2EVT = 90
    IVT2out = 45
    door2EVT2 = 45
    transdist1 = np.linalg.norm(med_coords[0,:] - med_coords[1,:]) * geoscale
    transtime1 = (transdist1/drivespeed)*60
    transdist2 = np.linalg.norm(med_coords[0,:] - med_coords[2,:]) * geoscale
    transtime2 = (transdist2/drivespeed)*60

    ivt_time_threshold = 4.5 * 60
    evt_time_threshold = 24 * 60

    lvo_status_arr = np.broadcast_to(np.expand_dims(lvo_status, axis = (1, 2)), (num_patients, num_scenarios, num_thresholds))

    ischemic_arr = np.broadcast_to(np.expand_dims(patient_df['ischemic'].values, axis = (1, 2)), (num_patients, num_scenarios, num_thresholds))

    hemorrhaging_arr = np.broadcast_to(np.expand_dims(patient_df['hemorrhaging'].values, axis = (1, 2)), (num_patients, num_scenarios, num_thresholds))

    IVTtime = lvo_status_arr * (lkw_to_door_arr + door2IVT) + ((~lvo_status_arr) & ischemic_arr) * (lkw_to_door_arr + door2IVT)

    EVTtime = lvo_status_arr * ((destination_arr == 'CSC') * (lkw_to_door_arr + door2EVT) +
                               (destination_arr == 'PSC1') * (IVTtime + IVT2out + transtime1 + door2EVT2) +
                               (destination_arr == 'PSC2') * (IVTtime + IVT2out + transtime2 + door2EVT2))
    PrOut = (
            lvo_status_arr * (((IVTtime < ivt_time_threshold) & (EVTtime >= evt_time_threshold)) * (0.2359 + 0.0000002 * IVTtime**2 - 0.0004  * IVTtime)
                          + (((IVTtime >= ivt_time_threshold) & (EVTtime < evt_time_threshold)) * (0.3394 + 0.00000004 * EVTtime**2 - 0.0002*EVTtime)) +
                          ((IVTtime < ivt_time_threshold) & (EVTtime < evt_time_threshold)) * (0.5753 + 0.0000002 * IVTtime**2 + 0.00000004 * EVTtime**2 - 0.0004 * IVTtime - 0.0002*EVTtime - (0.2359 + 0.0000002 * IVTtime**2 - 0.0004 * IVTtime) * (0.3394 + 0.00000004 * EVTtime**2 - 0.0002 *EVTtime))
                        + ((IVTtime >= 270) & (EVTtime >= 360)) * 0.129)
            + ((~lvo_status_arr) & ischemic_arr) * ((IVTtime < ivt_time_threshold) * (0.6343 - 0.00000005 * IVTtime**2 - 0.0005 * IVTtime) + (IVTtime >= ivt_time_threshold) * 0.4622)
            + (~lvo_status_arr & ~ischemic_arr & hemorrhaging_arr) * 0.24 
            + (~lvo_status_arr & ~ischemic_arr & ~hemorrhaging_arr) * 0.9
    )

    ############### Data reorganization #################
    results_df = pd.DataFrame({
        'patient_ID': np.repeat(np.arange(1, num_patients + 1), num_scenarios * num_thresholds),
        'seed': np.repeat(patient_seed, num_patients * num_scenarios * num_thresholds),
        'scenario': np.repeat(np.arange(1, num_scenarios * num_thresholds + 1).reshape(1, num_scenarios, num_thresholds), num_patients, axis = 0).flatten(),
        'sensitivity': np.broadcast_to(np.expand_dims(sens_spec_vals[:,0], axis = (0, 2)), (num_patients, num_scenarios, num_thresholds)).flatten(),
        'specificity': np.broadcast_to(np.expand_dims(sens_spec_vals[:,1], axis = (0, 2)), (num_patients, num_scenarios, num_thresholds)).flatten(),
        'threshold': thresholds_arr.flatten(),
        'destination': destination_arr.flatten(),
        'closest_destination': np.broadcast_to(np.expand_dims(closest_med, axis = (1,2)), shape = (num_patients, num_scenarios, num_thresholds)).flatten(),
        'x_coord': np.repeat(patient_df['x_coord'].values, num_scenarios * num_thresholds),
        'y_coord': np.repeat(patient_df['y_coord'].values, num_scenarios * num_thresholds),
        'map_number': np.full(num_patients * num_scenarios * num_thresholds, map_seed),
        'hasLVO': lvo_status_arr.flatten(),
        'lvo_diagnosis': np.broadcast_to(np.expand_dims(expanded_lvo_diagnosis, axis = 2), (num_patients, num_scenarios, num_thresholds)).flatten(),
        'ischemic': ischemic_arr.flatten(),
        'hemorrhaging': hemorrhaging_arr.flatten(),
        'lkw2door': lkw_to_door_arr.flatten(),
        'time2Hospital': time_to_hospital_arr.flatten(),
        'IVTtime': IVTtime.flatten(),
        'EVTtime': EVTtime.flatten(),
        'PrOut': PrOut.flatten(),
        'xPSC': geoscale * np.repeat(med_coords[1,0], num_patients * num_scenarios * num_thresholds),
        'yPSC': geoscale * np.repeat(med_coords[1,1], num_patients * num_scenarios * num_thresholds),
        'xPSC2': geoscale * np.repeat(med_coords[2,0], num_patients * num_scenarios * num_thresholds),
        'yPSC2': geoscale * np.repeat(med_coords[2,1], num_patients * num_scenarios * num_thresholds),
        'geoscale': np.repeat(geoscale, num_patients * num_scenarios * num_thresholds),
        'drivespeed': np.repeat(drivespeed, num_patients * num_scenarios * num_thresholds)
    })
    return results_df

def run_map_simulations(map_seeds, num_patients = 1000, num_patient_seeds = 50, save_format = 'csv', output_dir = None):
    min_map = min(map_seeds)
    max_map = max(map_seeds)

    if output_dir is None:
        output_dir = 'output'
    output_dir_path = pathlib.Path(output_dir)
    if not output_dir_path.is_dir():
        output_dir_path.mkdir(parents = True)
    match save_format:
        case 'csv':
            output_file = output_dir_path / f'maps_{min_map}_{max_map}.csv'
            if output_file.exists():
                output_file.unlink()  
    
    for i in map_seeds:
        map_output_list = []
        for j in range(num_patient_seeds):
            temp = simulation(num_patients, patient_seed = j, map_seed = i)
            map_output_list.append(temp)
            # map_df = pd.concat((map_df, temp))
        map_output_df = pd.concat(map_output_list)
        match save_format:
            case 'csv':
                if output_file.exists():
                    map_output_df.to_csv(output_file, 
                            mode = 'a',
                            index = False,
                            header = False)
                else:
                    map_output_df.to_csv(output_file,
                                mode = 'w',
                                index = False,
                                header = True)
            case 'parquet':
                output_file = output_dir_path / f'map_{str(i).zfill(3)}.parquet'
                map_output_df.to_parquet(output_file, index = False)
    return None

#############################################################################
#############################################################################

def create_map_csv(filepath, map_seeds, num_points = 1000000):
    map_dict = {}
    rng = np.random.default_rng(2024)
    simulated_coords = rng.uniform(low = 0, high = geoscale, size = (num_points, 2))
    for seed in map_seeds:
        labels, coords, geoscale = generate_map(seed)
        drivespeed = get_drivespeed(geoscale)
        med_coords = geoscale * coords

        equipoise = np.sum(np.argmin(spatial.distance.cdist(simulated_coords, med_coords), axis = 1) != 0) / num_points

        map_dict[seed] = {'ID': seed, 'geoscale': geoscale, 'equipoise': equipoise}
    map_df = pd.DataFrame(map_dict).transpose()
    map_df['ID'] = map_df['ID'].astype(int)
    map_df.to_csv(filepath, index = False)


def read_csv_results(filepath):
    pass