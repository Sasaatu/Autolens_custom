import os
import math
import numpy as np
import logging
from datetime import datetime
from deeplens.basics import Glass_Table
from auto_lens_design import default_inputs, config, design_lens

if __name__ == '__main__':
    # define specifications
    fov = 60.0          # target FOV in degree
    tol_hfov = 0.2      # tolerance to target hfov
    waves = [520]
    num_lens = 4
    res_grid = 3
    num_combo = 3       # number of glass combinations for spot size test
    iter = 500
    iter_test = 50
    iter_last = 200
    iter_test_last = 50
    is_sphere = True
    is_conic = False
    is_asphere = False

    # define configuration grid
    epd_range = np.linspace(1.0, 5.0, res_grid)
    imgh_range = np.linspace(2.0, 10.0, res_grid)
    dist_range = np.linspace(5.0, 10.0, res_grid)
    
    # set unchanged config
    args = default_inputs()
    hfov_tgt = math.radians(fov) / 2
    args['element'] = num_lens
    args['WAVES'] = waves
    args['iter'] = iter
    args['iter_test'] = iter_test   
    args['iter_last'] = iter_last
    args['iter_test_last'] = iter_test_last
    args['is_sphere'] = is_sphere
    args['is_conic'] = is_conic
    args['is_asphere'] = is_asphere
    args['save_steps'] = False

    ################################################################
    # Trainig data folder under AutoLens/results/

    current_time = datetime.now().strftime("%m%d-%H%M%S")
    sequence = 'GA'*num_lens
    num_gen = res_grid ** 3
    dir_results = f'./results/{current_time}_{sequence}_{num_gen}_Training_Designs'
    os.makedirs(dir_results, exist_ok=True)
    args['results_root'] = dir_results

    ################################################################
    # Logging configuration

    # log file locations
    glass_filename = dir_results + '/glass_rms.log'
    error_filename = dir_results + '/process_error.log'

    # log for glass combination test
    logging.basicConfig(
        level=logging.INFO,  # Set the level to INFO so that we log our data
        format='%(asctime)s - %(message)s',  # Format to include timestamp in the log
    )
    glass_logger = logging.getLogger('glass_logger')
    glass_handler = logging.FileHandler(glass_filename)
    glass_logger.addHandler(glass_handler)
    error_logger = logging.getLogger('error_logger')
    error_handler = logging.FileHandler(error_filename)
    error_logger.addHandler(error_handler)

    ################################################################
    # Step1: Define lens materials
    
    # omit file saving
    args['save_global'] = False

    if num_lens <= 3:
        if num_lens == 1:
            args['GLASSES'] = ['n-bk7']
        elif num_lens == 2:
            args['GLASSES'] = ['n-bk7', 'sf2']
        elif num_lens == 3:
            args['GLASSES'] = ['sk16', 'f2', 'sk16']

        glass_logger.info(f"Glasses: {args['GLASSES']} is selected.")
    else:
        # generate material combinations
        # duplication between combos: NO, materials: YES
        combinations = set()
        while len(combinations) < num_combo:
            combo = tuple(
                np.random.choice(Glass_Table, num_lens, replace=True).tolist()
            )
            combinations.add(combo)
        # convert set to list
        combinations = list(combinations)

        rms_array = np.zeros(num_combo)
        # test all glass combinations
        for i in range(num_combo):
            args['GLASSES'] = list(combinations[i])
            
            # design lens at diagonal points on the grid
            rms_diag = np.zeros(res_grid)
            for j in range(res_grid):
                # set config
                epd = float(epd_range[j])
                imgh = float(imgh_range[j])
                dist = float(dist_range[j])

                hfov_ref = math.atan((imgh/2)/dist)
                fnum = imgh/(2*epd*math.tan(hfov_ref))
                fnum_start = fnum*1.5
                diag_start = imgh/2.0
                rff = dist / imgh
                args['HFOV'] = hfov_ref
                args['FNUM'] = fnum
                args['DIAG'] = imgh
                args['FNUM_START'] = fnum_start
                args['DIAG_START'] = diag_start
                args['rff'] = rff
                args = config(args)
                
                try:
                    lens = design_lens(args)
                    # evaluate spot size
                    rms_diag[j] = lens.evaluate_spotsize()
                    # distruct instance
                    del lens
                except Exception as e:
                    # append error log and continue
                    error_logger.error(f"Glass error: {args['GLASSES']} for EPD {epd:.2f}, IMGH {imgh:.2f}, DIST {dist:.2f} with error {e}")    
                    rms_diag[j] = float('nan')
                    continue
            
            rms_glass = float(np.nanmean(rms_diag))
            rms_array[i] = rms_glass
            glass_logger.info(f"Glasses: {args['GLASSES']}, RMS Spot Size: {rms_glass:.4f} mm")

        # select best material combination wheere spot size is minimum
        idx = np.argmin(rms_array)
        args['GLASSES'] = list(combinations[idx])
        glass_logger.info(f"Glasses #{idx+1}: {args['GLASSES']} is selected.")
    glass_logger.info(f"Glass combination process is finished.")
     
    ################################################################
    # Step2: Generate designs

    # save last designs
    args['save_global'] = True
    
    # csv file name
    idx_GA = dir_results.find('GA')
    csv_name = dir_results + '/' + dir_results[idx_GA:] + '.csv'
    
    # iterate over design grid points
    for epd in epd_range:
        for imgh in imgh_range:
            for dist in dist_range:
                try:
                    # set config
                    epd = float(epd)
                    imgh = float(imgh)
                    dist = float(dist)

                    hfov_ref = math.atan((imgh/2)/dist)
                    if hfov_tgt>=hfov_ref*(1-tol_hfov) and hfov_tgt<=hfov_ref*(1+tol_hfov):
                        hfov_eff = hfov_ref
                    else:
                        hfov_eff = hfov_tgt
                    fnum = imgh/(2*epd*math.tan(hfov_eff))
                    fnum_start = fnum*1.5
                    diag_start = imgh/2.0
                    rff = dist / imgh
                    args['HFOV'] = hfov_eff
                    args['FNUM'] = fnum
                    args['DIAG'] = imgh
                    args['FNUM_START'] = fnum_start
                    args['DIAG_START'] = diag_start
                    args['rff'] = rff
                    # arrange design configuration
                    args = config(args)
                    
                    # create lens
                    lens = design_lens(args)

                    # distruct instance
                    del lens
                except Exception as e:
                    # append error log and continue
                    error_logger.error(f"Design error: EPD {epd:.2f}, IMGH {imgh:.2f}, DIST {dist:.2f} with error {e}")
                    continue
