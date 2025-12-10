import os
import math
import numpy as np
import logging
from datetime import datetime
from deeplens.basics import Glass_Table
from auto_lens_design import default_inputs, config, design_lens

if __name__ == '__main__':
    # define specifications
    fov = 70.0
    waves = [480, 520]
    num_lens = 3
    res_grid = 3
    num_combo = 100
    iter = 500
    iter_test = 50
    iter_last = 200
    iter_test_last = 50
    is_sphere = True
    is_conic = False
    is_asphere = False
    save_final_design = True

    # define configuration grid
    epd_range = np.linspace(0.5, 5.0, res_grid)
    imgh_range = np.linspace(1.0, 50.0, res_grid)
    dist_range = np.linspace(4.0, 100.0, res_grid)
    
    # set unchanged config
    args = default_inputs()
    hfov_rad = math.radians(fov) / 2
    args['HFOV'] = hfov_rad
    args['element'] = num_lens
    args['WAVES'] = waves
    args['ITER'] = iter
    args['ITER_TEST'] = iter_test   
    args['ITER_LAST'] = iter_last
    args['ITER_TEST_LAST'] = iter_test_last
    args['is_sphere'] = is_sphere
    args['is_conic'] = is_conic
    args['is_asphere'] = is_asphere
    
    ################################################################
    # Step1: Define lens materials
    
    if num_lens <= 3:
        if num_lens == 1:
            args['GLASSES'] = ['n-bk7']
        elif num_lens == 2:
            args['GLASSES'] = ['n-bk7', 'sf2']
        elif num_lens == 3:
            args['GLASSES'] = ['sk16', 'f2', 'sk16'] 
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
                epd = epd_range[j]
                imgh = imgh_range[j]
                dist = dist_range[j]
                fnum = imgh/(2*epd*math.tan(hfov_rad))
                fnum_start = fnum*1.5
                diag_start = imgh/2.0
                rff = dist / imgh
                args['FNUM'] = fnum
                args['DIAG'] = imgh
                args['FNUM_START'] = fnum_start
                args['DIAG_START'] = diag_start
                args['rff'] = rff
                args = config(args)
                
                lens = design_lens(args, False, False)
                # evaluate spot size
                rms_diag[j] = lens.evaluate_spotsize()
                # distruct instance
                del lens
                
            rms_array[i] = np.mean(rms_diag)
        # select best material combination wheere spot size is minimum
        idx = np.argmin(rms_array)
        args['GLASSES'] = combinations[idx]
     
    
    ################################################################
    # Step2: Generate designs
    
    # create results directory
    current_time = datetime.now().strftime("%m%d-%H%M%S")
    sequence = 'GA'*num_lens
    num_gen = res_grid ** 3
    dir_results = f'./results/{current_time}_{sequence}_{num_gen}training_designs'
    os.makedirs(dir_results, exist_ok=True)
    args['results_root'] = dir_results
    # csv file name
    csv_name = dir_results + f'/{current_time}_{sequence}_{num_gen}training_designs.csv'
    # create error log file
    error_log_name = dir_results + '/generation_error.txt'
    with open(error_log_name, "w") as f:
        f.write("")
    
    # iterate over design grid points
    for epd in epd_range:
        for imgh in imgh_range:
            for dist in dist_range:
                try:
                    # set config
                    fnum = imgh/(2*epd*math.tan(hfov_rad))
                    fnum_start = fnum*1.5
                    diag_start = imgh/2.0
                    rff = dist / imgh
                    args['FNUM'] = fnum
                    args['DIAG'] = imgh
                    args['FNUM_START'] = fnum_start
                    args['DIAG_START'] = diag_start
                    args['rff'] = rff
                    # arrange design configuration
                    args = config(args)
                    
                    # create lens
                    lens = design_lens(args, False, save_final_design)

                    # save design in zmx, json and png file
                    logging.info(f'Actual: FOV {lens.hfov}, IMGH {lens.r_last}, F/{lens.fnum}.')
                    if save_final_design:
                        result_dir = args['result_dir']
                        lens.write_lensfile(f'{result_dir}/final_lens.txt', write_zmx=True)
                        lens.analysis(save_name=f'{result_dir}/final_lens', draw_layout=True)
                        lens.append_csv(csv_name)
                    
                    # distruct instance
                    del lens
                except Exception as e:
                    # append error text and continue
                    with open(error_log_name, "a") as f:
                        f.write(f"Design failed for EPD: {epd}, IMGH: {imgh}, DIST: {dist} with error {e}\n")
                    continue
