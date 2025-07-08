import numpy as np
import pandas as pd
from scipy import spatial
from scipy.special import logit, expit
# from sklearn.metrics import confusion_matrix
import pathlib
import yaml

def data_to_config(patient_filestr, times_filestr):
    '''
    Reads in a data file containing patient level LKW and hex locations

    Outputs config dictionary to be merged with a config dict from a YAML file
    '''
    # Patient level data -- LKW and hex frequencies
    try:
        data = pd.read_csv(patient_filestr)
        hex_config = dict(data.groupby(data['hex']).size())
        data['lkw_bins'] = pd.cut(data['last_well'], 
            bins = [0, 1/60, 1/6, 2, 3.5, 8, 24, 48, 72])
        patient_config = {}
        lkw_info = data.groupby('lkw_bins').size()
        for bin in lkw_info.index:
            patient_config[bin] = {
                'prob': lkw_info.loc[bin],
                'dist': 'rng.uniform',
                'kwargs': {'low': bin.left, 'high': bin.right}
            }
        patient_retval = {'hexes': hex_config,
            'patient_lkw_bins': patient_config}
    except:
        patient_retval = None

    # Times
    try:
        times_data = pd.read_csv(times_filestr)
        num_hosps = len(times_data.columns) - 1
        hex_hosp_times = times_data[:-num_hosps].set_index(times_data.columns[0])
        hosp_hosp_times = times_data[-num_hosps:].set_index(times_data.columns[0])
    except:
        hex_hosp_times = None
        hosp_hosp_times = None

    return patient_retval, hex_hosp_times, hosp_hosp_times

def read_config(yaml_filestr = None, patient_data_filestr = None, times_filestr = None):
    '''
    Reads in a config dictionary from a YAML file

    Merges two dictionaries together, with values kept from YAML file if applicable
    '''
    with open(yaml_filestr) as f:
        config_override = yaml.safe_load(f)
    params = {
        'patients_none_all': 0.6765,
        'patients_tia_all': 0.0795,
        'patients_hemorrhaging_all': 0.042,
        'patients_ischemic_all': 0.202,
        'patients_lvo_ischemic': 0.241,
        'patients_num_lkw_bins': 5,
        'simulations_ivt_threshold': 270,
        'simulations_evt_threshold': 1440,
        'patient_lkw_bins': None,
        'hexes': None,
        'simulations_ivt_threshold': 270,
        'simulations_evt_threshold': 1440,
        'simulations_ivt_probability': 0.55,
        'simulations_evt_probability': 0.85,
        'simulations_early_repurfusion': 0.11,
        'csc_prefix': 'X',
        'psc_prefix': 'Y',
        'nsc_prefix': 'Z'
    }
    data_config, transport_times, transfer_times = data_to_config(patient_data_filestr, times_filestr)
    retval = params | config_override | data_config
    retval['transport_times'] = transport_times
    retval['transfer_times'] = transfer_times
    return retval

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

def generate_patient_cohort(num_patients, seed, config = None):
    '''
    Params:
        - num_patients: Number of patients to be generated
        - seed: Random seed to initialize the generator 

    Returns:
        - patient_df: pd.DataFrame containing all patient data. Note
                        coordinates are normalized (i.e not multiplied by any geoscale factor)
    '''
    rng = np.random.default_rng(seed)

    ## Patient spawn locations
    if config is None:
        patient_hexes = None
        patient_coords_normalized = rng.random((num_patients, 2))
    else:
        hex_names = list(config['hexes'].keys())
        hex_probs = np.array(list(config['hexes'].values()))
        patient_hexes = rng.choice(hex_names, p = hex_probs/hex_probs.sum(), replace = True, size = num_patients)
        patient_coords_normalized = np.full((num_patients, 2), None)

    ## Actual stroke status generation
    stroke_types = np.array(['none', 'tia', 'hemorrhaging', 'ischemic'])
    try:
        probs = np.array([config['patients_none_all'],
                      config['patients_tia_all'],
                      config['patients_hemorrhaging_all'],
                      config['patients_ischemic_all']])
    except:
        probs = np.array([0.6765, 0.0795, 0.042, 0.202])
    patient_stroke_diagnoses = rng.choice(stroke_types, size = num_patients, p = probs/np.sum(probs))

    stroke = (patient_stroke_diagnoses == 'none')
    tia = (patient_stroke_diagnoses == 'tia')
    hemorrhaging = (patient_stroke_diagnoses == 'hemorrhaging')
    ischemic = (patient_stroke_diagnoses == 'ischemic')

    try:
        lvo_ischemic_proportion = config['patients_lvo_ischemic']
    except:
        lvo_ischemic_proportion = 0.241
    lvo_status = np.full(num_patients, False)
    lvo_status[ischemic] = (rng.random(np.sum(ischemic)) < lvo_ischemic_proportion)

    # sensitivity analysis, check 14.1 and 34.1 on 9 manually specified maps



    # # where does the 0.4 come from?
    # stroke = (rng.random(num_patients) < 1 - 0.6765) 

    # # Around 85-87% of strokes due to ischemic event
    # hemorrhaging = np.full(num_patients, False)
    # ischemic = np.full(num_patients, False)

    # hemorrhaging_ischemic_rng = (rng.random(np.sum(stroke)) < 0.13) # indicator for hemorrhaging
    # hemorrhaging[stroke] = hemorrhaging_ischemic_rng
    # ischemic[stroke] = ~hemorrhaging_ischemic_rng

    # Of ischemic stroke patients, 10-46% depending on definition of LVO (Saini)
    # Up to 40% (Dabus)
    # lvo_status = np.full(num_patients, False)
    # lvo_status[ischemic] = (rng.random(np.sum(ischemic)) < 0.387) # currently set to 38.7% 

    # probs = np.array([0.44, 0.22, 0.29, 0.05])
    # lastWell_bins = rng.choice(a = [i for i in range(1, len(probs) + 1)], p = probs / np.sum(probs, dtype = float), size = num_patients)

    # Note uniform distribution from numpy.random takes different arguments
    # than the uniform distribution from scipy.stats
    # last_well = ( (lastWell_bins == 1) * rng.uniform(0.1, 3, num_patients) + 
    #             (lastWell_bins == 2) * rng.uniform(3, 6, num_patients) + 
    #             (lastWell_bins == 3) * rng.uniform(6, 24, num_patients) + 
    #             (lastWell_bins == 4) * rng.uniform(24, 48, num_patients) )
    
    # probs = np.array([0.206, 0.062, 0.09, 0.559, 0.083])
    # last_well_distributions = [
    #     {'type': rng.uniform, 'kwargs': {'low': 1/6, 'high': 2}},
    #     {'type': rng.uniform, 'kwargs': {'low': 2, 'high': 3.5}},
    #     {'type': rng.uniform, 'kwargs': {'low': 3.5, 'high': 8}},
    #     {'type': rng.uniform, 'kwargs': {'low': 8, 'high': 24}},
    #     {'type': rng.uniform, 'kwargs': {'low': 24, 'high': 48}}
    # ]

    ## LKW time generation
    try:
        lkw_bins = config['patients_lkw_bins']
        last_well_distributions = []
        probs = []
        for bin in lkw_bins:
            probs.append(lkw_bins[bin]['prob'])
            last_well_distributions.append({'type': eval(lkw_bins[bin]['dist']),
                                            'kwargs': lkw_bins[bin]['kwargs']
                                            })
    except:
        probs = np.array([0.206, 0.062, 0.09, 0.559, 0.083])
        last_well_distributions = [
        {'type': rng.uniform, 'kwargs': {'low': 1/6, 'high': 2}},
        {'type': rng.uniform, 'kwargs': {'low': 2, 'high': 3.5}},
        {'type': rng.uniform, 'kwargs': {'low': 3.5, 'high': 8}},
        {'type': rng.uniform, 'kwargs': {'low': 8, 'high': 24}},
        {'type': rng.uniform, 'kwargs': {'low': 24, 'high': 48}}
        ]


    last_well_generated_times = np.zeros((num_patients, len(last_well_distributions)))
    for i, distr in enumerate(last_well_distributions):
        last_well_generated_times[:, i] = distr['type'](size = num_patients, **distr['kwargs'])
    lastWell_bins = rng.choice(a = np.arange(len(last_well_distributions)), p = probs / np.sum(probs, dtype = float), size = num_patients)

    last_well = last_well_generated_times[np.arange(num_patients), lastWell_bins]

    # last_well = rng.lognormal(mean = -0.725238324950042, sigma = np.sqrt(3.70290891530286), size = num_patients)
    
    # last_well = ( (lastWell_bins == 1) * rng.uniform(0.1, 2, num_patients) + 
    #             (lastWell_bins == 2) * rng.uniform(2, 3.5, num_patients) + 
    #             (lastWell_bins == 3) * rng.uniform(3.5, 8, num_patients) + 
    #             (lastWell_bins == 4) * rng.uniform(8, 24, num_patients)  +
    #             (lastWell_bins == 5) * rng.uniform(24, 48, num_patients) )


    patient_df = pd.DataFrame({
        'ID': np.arange(1, num_patients + 1),
        'x_coord': patient_coords_normalized[:,0],
        'y_coord': patient_coords_normalized[:,1],
        'hex': patient_hexes,
        'stroke': stroke,
        'tia': tia,
        'hemorrhaging': hemorrhaging,
        'ischemic': ischemic,
        'lvo_status': lvo_status,
        'last_well': last_well,
        'seed': seed
    })

    return patient_df

def generate_map(seed, num_psc = 2, config = None):
    '''
    Params:
        - seed: Random seed to initialize the generator
        - num_psc: Number of PSC locations to use

    Returns:
        - med_coords: (num_psc + 1) x 2 array containing hospital coordinates
            Row 0 is hard-coded as (0.5, 0.5) due to being CSC location
        - geoscale: 
    '''
    if config is None:
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
    else:
        try:
            hospital_dict = config['hosp_coords']
            med_labels = []
            med_coords = np.zeros((0, 2))
            for key in hospital_dict:
                med_labels.append(hospital_dict[key]['type'])
                med_coords = np.vstack((med_coords, np.array(hospital_dict[key]['coords'])))
            return med_labels, med_coords, np.max(med_coords)
        except:
            return generate_map(seed, num_psc = 2, config = None)
        

def simulation(num_patients, patient_seed, map_seed, sens_spec_vals = np.array([[0.9, 0.6], [0.75, 0.75], [0.6, 0.9]]), thresholds = np.arange(0, 70, 10), config = None):
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
    patient_df = generate_patient_cohort(num_patients, seed = patient_seed, config = config)
    med_labels, med_coords, geoscale = generate_map(map_seed, num_psc = 2, config = config)

    drivespeed = get_drivespeed(geoscale)

    if config is None:
        patient_coords = patient_df[['x_coord', 'y_coord']].values
        patient_med_dists = geoscale * spatial.distance.cdist(patient_coords, med_coords)
        patient_med_times = patient_med_dists / drivespeed * 60
        closest_med_ind = np.argmin(patient_med_times, axis = 1)
        closest_med = med_labels[closest_med_ind]
        closest_med_times = np.min(patient_med_times, axis = 1)
    else:
        patient_med_times = patient_df[['ID','hex']].set_index('hex').join(config['transport_times']).set_index('ID')
        closest_med_times = patient_med_times.min(axis = 1).values
        closest_med = patient_med_times.idxmin(axis = 1).values
        # is_closest_csc = patient_med_times.idxmin(axis = 1).str.contains(config['csc_prefix'])
        csc_transport_times = patient_med_times.filter(regex = config['csc_prefix'], axis = 1)
        if len(csc_transport_times) == 1:
            csc_transport_times = np.expand_dims(csc_transport_times, axis = 1)
        # patient_med_times = 

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
    if config is None:
        correct_destination_ind = closest_med_ind.copy()
        correct_destination[lvo_status & (last_well <= 24)] = 'CSC'
        correct_destination_ind[lvo_status & (last_well <= 24)] = 0
    else:
        csc_correct_ind = lvo_status & (last_well <= 24)
        correct_destination[csc_correct_ind] = csc_transport_times.iloc[csc_correct_ind].idxmin(axis = 1)

    destination_arr = np.broadcast_to(np.expand_dims(closest_med, axis = (1, 2)), (num_patients, num_scenarios, num_thresholds)).copy()

    # Eligible to be redirected to CSC under each scenario type
    eligible_patients = (expanded_lvo_diagnosis) & (np.expand_dims(last_well, axis = 1) <= 24)

    eligibility_arr = np.broadcast_to(np.expand_dims(eligible_patients, axis = 2), (num_patients, num_scenarios, num_thresholds))
    thresholds_arr = np.broadcast_to(thresholds, (num_patients, num_scenarios, num_thresholds))
    
    if config is None:
        additional_transport_arr = np.broadcast_to(np.expand_dims(patient_med_times[:,0] - closest_med_times, axis = (1, 2)), (num_patients, num_scenarios, num_thresholds))

        redirected_patients = eligibility_arr & (additional_transport_arr <= thresholds_arr)
        destination_arr[redirected_patients] = 'CSC'
    else:
        additional_transport_arr = np.broadcast_to(np.expand_dims(csc_transport_times.min(axis = 1).values - closest_med_times, axis = (1, 2)), (num_patients, num_scenarios, num_thresholds))

        redirected_patients = eligibility_arr & (additional_transport_arr <= thresholds_arr)
        destination_arr[redirected_patients] = csc_transport_times.iloc[redirected_patients].idxmin(axis = 1).values

    ##################### Time variables ############################
    # Time to scene
    time_to_scene = 1.62 + rng.normal(15.1, 7, size = num_patients)
    time_to_scene_arr = np.broadcast_to(np.expand_dims(time_to_scene, axis = (1, 2)), (num_patients, num_scenarios, num_thresholds)).copy()

    # Time at scene
    time_at_scene = 40 * rng.beta(2.91, 6.056, size = num_patients)
    # time_at_scene = 40 * rng.gamma(shape = 4.9146291403, scale = 1/0.2563312894, size = num_patients)

    time_at_scene_arr = np.broadcast_to(np.expand_dims(time_at_scene, axis = (1, 2)), (num_patients, num_scenarios, num_thresholds)).copy()

    # Time from scene to hospital
    time_to_hospital_arr = np.broadcast_to(np.expand_dims(closest_med_times, axis = (1, 2)), (num_patients, num_scenarios, num_thresholds)).copy()

    if config is None:
        patient_csc_times_arr = np.broadcast_to(np.expand_dims(patient_med_times[:,0], axis = (1, 2)), (num_patients, num_scenarios, num_thresholds))

        time_to_hospital_arr[redirected_patients] = patient_csc_times_arr[redirected_patients]
    else:

        patient_csc_times_arr = np.broadcast_to(np.expand_dims(patient_med_times.filter(regex = config['csc_prefix'], axis = 1).min(axis = 1).values, axis = (1, 2)), (num_patients, num_scenarios, num_thresholds))

        time_to_hospital_arr[redirected_patients] = patient_csc_times_arr[redirected_patients]

    # Time aggregation
    time_in_system_arr = time_to_scene_arr + time_at_scene_arr + time_to_hospital_arr

    lkw_to_door_arr = 60 * np.broadcast_to(np.expand_dims(last_well, axis = (1, 2)), (num_patients, num_scenarios, num_thresholds)) + time_in_system_arr

    ####################### Outcomes ######################
    try:
        door2IVT = config['door2IVT']
        door2EVT = config['door2EVT']
        IVT2out = config['IVT2out']
        door2EVT2 = config['door2EVT2']
    except:
        door2IVT = 45
        door2EVT = 90
        IVT2out = 45
        door2EVT2 = 45
    transdist1 = np.linalg.norm(med_coords[0,:] - med_coords[1,:]) * geoscale
    transtime1 = (transdist1/drivespeed)*60
    transdist2 = np.linalg.norm(med_coords[0,:] - med_coords[2,:]) * geoscale
    transtime2 = (transdist2/drivespeed)*60

    try:
        ivt_time_threshold = config['simulations_ivt_threshold']
        evt_time_threshold = config['simulations_evt_threshold']
        ivt_probability = config['simulations_ivt_probability']
        evt_probability = config['simulations_evt_probability']
        early_repurfusion_probability = config['simulations_early_repurfusion_probability']
    except:
        ivt_time_threshold = 4.5 * 60
        evt_time_threshold = 24 * 60
        ivt_probability = 0.55
        evt_probability = 0.85
        early_repurfusion_probability = 0.11

    lvo_status_arr = np.broadcast_to(np.expand_dims(lvo_status, axis = (1, 2)), (num_patients, num_scenarios, num_thresholds))

    ischemic_arr = np.broadcast_to(np.expand_dims(patient_df['ischemic'].values, axis = (1, 2)), (num_patients, num_scenarios, num_thresholds))

    hemorrhaging_arr = np.broadcast_to(np.expand_dims(patient_df['hemorrhaging'].values, axis = (1, 2)), (num_patients, num_scenarios, num_thresholds))

    tia_arr = np.broadcast_to(np.expand_dims(patient_df['tia'].values, axis = (1, 2)), (num_patients, num_scenarios, num_thresholds))

    if config is None:
        IVTtime = lvo_status_arr * (lkw_to_door_arr + door2IVT) + ((~lvo_status_arr) & ischemic_arr) * (lkw_to_door_arr + door2IVT)

        EVTtime = lvo_status_arr * ((destination_arr == 'CSC') * (lkw_to_door_arr + door2EVT) +
                                (destination_arr == 'PSC1') * (IVTtime + IVT2out + transtime1 + door2EVT2) +
                                (destination_arr == 'PSC2') * (IVTtime + IVT2out + transtime2 + door2EVT2))
    else:
        IVTtime = ((pd.Series(destination_arr.flatten()).str.contains('[' + config['csc_prefix'] + '|' + config['psc_prefix'] + ']').values)).reshape((num_patients, num_scenarios, num_thresholds)) * (lkw_to_door_arr + door2IVT)

        csc_transfer_times = config['transfer_times'].filter(regex = config['csc_prefix'], axis = 1).min(axis = 1)
        transtime = csc_transfer_times[destination_arr.flatten()].values.reshape((num_patients, num_scenarios, num_thresholds))

        EVTtime = lvo_status_arr * (
                            ((pd.Series(destination_arr.flatten()).str.contains(config['csc_prefix']).values).reshape((num_patients, num_scenarios, num_thresholds)) * (lkw_to_door_arr + door2EVT))
                            + ((pd.Series(destination_arr.flatten()).str.contains(config['psc_prefix']).values)).reshape((num_patients, num_scenarios, num_thresholds)) * 
                            (lkw_to_door_arr + door2IVT + IVT2out + transtime + door2EVT2)
        )
        
    ### Randomizing whether or not a patient receives IVT
    try:
        IVTtreatment = ischemic_arr & (IVTtime < ivt_time_threshold) & (rng.random(IVTtime.shape) < ivt_probability) & pd.Series(destination_arr.flatten()).str.contains(regex = '[' + config['csc_prefix'] + '|' + config['psc_prefix'] + ']', axis = 1).reshape((num_patients, num_scenarios, num_thresholds))
        IVTrepurfusion = lvo_status_arr & IVTtreatment & (rng.random(IVTtreatment.shape) < early_repurfusion_probability)

        EVTtreatment = lvo_status_arr & (EVTtime < evt_time_threshold) & (rng.random(EVTtime.shape) < evt_probability)
    except:
        IVTtreatment = np.ones_like(lvo_status_arr)
        IVTrepurfusion = np.zeros_like(lvo_status_arr)
        EVTtreatment = np.ones_like(lvo_status_arr)


    ### Updating risk equations for mRS 0-2
    lvo_base_prob_mRS_02 = 0.05 + 0.08 + 0.14
    no_lvo_base_prob_mRS_02 = 0.13 + 0.19 + 0.12 
    lvo_base_logit_mRS_02 = logit(lvo_base_prob_mRS_02)
    no_lvo_base_logit_mRS_02 = logit(no_lvo_base_prob_mRS_02)

    LogitOut = (lvo_status_arr * (IVTrepurfusion * (1.35 - 0.0026 * IVTtime + lvo_base_logit_mRS_02)
                      + (~IVTrepurfusion & EVTtreatment) * (1.35 - 0.0026 * EVTtime + lvo_base_logit_mRS_02)
                      + (~IVTrepurfusion & ~EVTtreatment) * lvo_base_logit_mRS_02
                      )
    + (~lvo_status_arr & ischemic_arr) * (IVTtreatment * (0.56 - 0.0019 * IVTtime + no_lvo_base_logit_mRS_02)
                           + (~IVTtreatment) * no_lvo_base_logit_mRS_02)
    + (~lvo_status_arr & ~ischemic_arr & hemorrhaging_arr) * logit(0.375)
    + (~lvo_status_arr & ~ischemic_arr & ~hemorrhaging_arr) * logit(0.9)
    )

    PrOut = expit(LogitOut)
    ###


    # PrOut = (
    #         lvo_status_arr * (((IVTtime < ivt_time_threshold) & (EVTtime >= evt_time_threshold)) * (0.2359 + 0.0000002 * IVTtime**2 - 0.0004  * IVTtime)
    #                       + (((IVTtime >= ivt_time_threshold) & (EVTtime < evt_time_threshold)) * (0.3394 + 0.00000004 * EVTtime**2 - 0.0002*EVTtime)) +
    #                       ((IVTtime < ivt_time_threshold) & (EVTtime < evt_time_threshold)) * (0.5753 + 0.0000002 * IVTtime**2 + 0.00000004 * EVTtime**2 - 0.0004 * IVTtime - 0.0002*EVTtime - (0.2359 + 0.0000002 * IVTtime**2 - 0.0004 * IVTtime) * (0.3394 + 0.00000004 * EVTtime**2 - 0.0002 *EVTtime))
    #                     + ((IVTtime >= ivt_time_threshold) & (EVTtime >= evt_time_threshold)) * 0.129)
    #         + ((~lvo_status_arr) & ischemic_arr) * ((IVTtime < ivt_time_threshold) * (0.6343 - 0.00000005 * IVTtime**2 - 0.0005 * IVTtime) + (IVTtime >= ivt_time_threshold) * 0.4622)
    #         + (~lvo_status_arr & ~ischemic_arr & hemorrhaging_arr) * 0.24 
    #         + (~lvo_status_arr & ~ischemic_arr & ~hemorrhaging_arr) * 0.9
    # )

    ## Risk equation updates
    ## stroke mimics - keep at 90% for now but will probably change
    ## hemorrhaging - change to 0.375 for now

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
        'hex': np.repeat(patient_df['hex'].values, num_scenarios * num_thresholds),
        'map_number': np.full(num_patients * num_scenarios * num_thresholds, map_seed),
        'hasLVO': lvo_status_arr.flatten(),
        'lvo_diagnosis': np.broadcast_to(np.expand_dims(expanded_lvo_diagnosis, axis = 2), (num_patients, num_scenarios, num_thresholds)).flatten(),
        'ischemic': ischemic_arr.flatten(),
        'hemorrhaging': hemorrhaging_arr.flatten(),
        'tia': tia_arr.flatten(),
        'lkw2door': lkw_to_door_arr.flatten(),
        'time2Hospital': time_to_hospital_arr.flatten(),
        'IVTtime': IVTtime.flatten(),
        'EVTtime': EVTtime.flatten(),
        'IVTtreatment': IVTtreatment.flatten(),
        'EVTtreatment': EVTtreatment.flatten(),
        'EarlyRepurfusion': IVTrepurfusion.flatten(),
        'PrOut': PrOut.flatten(),
        'xPSC': geoscale * np.repeat(med_coords[1,0], num_patients * num_scenarios * num_thresholds),
        'yPSC': geoscale * np.repeat(med_coords[1,1], num_patients * num_scenarios * num_thresholds),
        'xPSC2': geoscale * np.repeat(med_coords[2,0], num_patients * num_scenarios * num_thresholds),
        'yPSC2': geoscale * np.repeat(med_coords[2,1], num_patients * num_scenarios * num_thresholds),
        'geoscale': np.repeat(geoscale, num_patients * num_scenarios * num_thresholds),
        'drivespeed': np.repeat(drivespeed, num_patients * num_scenarios * num_thresholds)
    })
    return results_df

def run_map_simulations(map_seeds, num_patients = 1000, num_patient_seeds = 50, save_format = 'csv', output_dir = None, config = None, additional_file_name = ''):
    
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
            temp = simulation(num_patients, patient_seed = j, map_seed = i, config = config)
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
                output_file = output_dir_path / f'{additional_file_name}{'_' if additional_file_name != '' else ''}map_{str(i).zfill(3)}.parquet'
                map_output_df.to_parquet(output_file, index = False)
    return None

def run_patient_simulations(map_seeds = [0], num_patients = 1000, patient_seeds = [0], save_format = 'csv', output_dir = None, config = None, additional_file_name = ''):
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
    
    for i in patient_seeds:
        patient_output_list = []
        for j in range(map_seeds):
            temp = simulation(num_patients, patient_seed = i, map_seed = j, config = config)
            patient_output_list.append(temp)
            # map_df = pd.concat((map_df, temp))
        map_output_df = pd.concat(patient_output_list)
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
                output_file = output_dir_path / f'{additional_file_name}{'_' if additional_file_name != '' else ''}map_{str(i).zfill(3)}.parquet'
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
