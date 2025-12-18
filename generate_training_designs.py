import os
import math
import numpy as np
import logging
from datetime import datetime
from deeplens.basics import Glass_Table
from auto_lens_design import default_inputs, config, design_lens

if __name__ == '__main__':
    # define specifications
    fov = 80.0              # target FOV in degree
    rate_fnum_start = 5.0   # rate of start/target f#
    rate_diag_start = 0.05  # rate of start/target image height
    waves = [520]           # wavelength list
    num_lens = 3            # number of lens
    res_grid = 3            # number of grid per axis of configuration grid
    num_combo = 3           # number of glass combinations
    num_step = 10           # number of curriculum step
    iter = 500              # iteration per step
    iter_test = 50          # frequency of lens shape & interval correction
    iter_last = 500         # extra iteration for last step with denser rays
    iter_test_last = 50     # frequency of lens correction during extra iteration
    is_sphere = True        # flag of curvature radius optimization
    is_conic = False        # flag of conic constant optimization
    is_asphere = False      # flag of aspherical coefficients optimization

    # define configuration grid
    epd_range = np.linspace(1.0, 5.0, res_grid)
    imgh_range = np.linspace(2.0, 10.0, res_grid)
    dist_range = np.linspace(5.0, 10.0, res_grid)
    
    # set unchanged config
    hfov_tgt = math.radians(fov) / 2
    args = default_inputs()
    args['WAVES'] = waves
    args['element'] = num_lens
    args['curriculum_steps'] = num_step
    args['iter'] = iter
    args['iter_test'] = iter_test   
    args['iter_last'] = iter_last
    args['iter_test_last'] = iter_test_last
    args['is_sphere'] = is_sphere
    args['is_conic'] = is_conic
    args['is_asphere'] = is_asphere

    ################################################################
    # Trainig data folder under AutoLens/results/

    current_time = datetime.now().strftime("%m%d-%H%M%S")
    sequence = 'GA'*num_lens
    num_gen = res_grid ** 3
    dir_results = f'./results/{current_time}_{sequence}_{num_gen}_Training_Designs'
    os.makedirs(dir_results, exist_ok=True)
    args['results_root'] = dir_results
    
    # csv file name
    idx_GA = dir_results.find('GA')
    csv_name = dir_results + '/' + dir_results[idx_GA:] + '.csv'
    args['designs_csv'] = csv_name

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
    args['save_steps'] = False
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
                fnum_start = fnum * rate_fnum_start
                diag_start = imgh * rate_diag_start
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

    # save intermediate & last designs
    args['save_steps'] = True
    args['save_global'] = True
    # set target hfov
    args['HFOV'] = hfov_tgt
    
    # iterate over design grid points
    for epd in epd_range:
        for imgh in imgh_range:
            for dist in dist_range:
                try:
                    # set config
                    epd = float(epd)
                    imgh = float(imgh)
                    dist = float(dist)

                    fnum = imgh/(2*epd*math.tan(hfov_tgt))
                    fnum_start = fnum * rate_fnum_start
                    diag_start = imgh * rate_diag_start
                    rff = dist / imgh
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
