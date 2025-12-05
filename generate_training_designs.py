import math
import numpy as np
import logging
from auto_lens_design import default_inputs, config, design_lens

if __name__ == '__main__':
    fov = 70.0
    num_lens = 2
    num_combi = 20
    res_grid = 3

    # define configuration grid
    epd_range = np.linspace(0.5, 5.0, res_grid)
    imgh_range = np.linspace(1.0, 50.0, res_grid)
    dist_range = np.linspace(4.0, 100.0, res_grid)
    
    # set unchanged config
    args = default_inputs()
    hfov_rad = math.radians(fov) / 2
    args['HFOV'] = hfov_rad
    args['element'] = num_lens
    args['WAVES'] = [480, 520, 640]
    args['ITER'] = 100
    args['ITER_TEST'] = 10
    args['ITER_LAST'] = 100
    args['ITER_TEST_LAST'] = 10
    
    ################################################################
    # Define lens materials
    rms_array = np.zeros(num_combi)
    # test all glass combinations
    for i in range(num_combi):
        args['GlASSES'] = mediums_list[i]
        
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
            
            lens = design_lens(args)
            # evaluate spot size
            rms_diag[j] = lens.evaluate_spot_size()
            # distruct instance
            del lens
            
        rms_array[i] = np.mean(rms_diag)
    # select best material combination wheere spot size is minimum
    idx = np.argmin(rms_array)
    args['GLASSES'] = mediums_list[idx]
    
    ################################################################
    # Generate designs
    # create error log file
    error_log_name = "results/generation_error.txt"
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
                    lens = design_lens(args)

                    # save design in zmx, json and png file
                    logging.info(f'Actual: FOV {lens.hfov}, IMGH {lens.r_last}, F/{lens.fnum}.')
                    result_dir = args['result_dir']
                    lens.write_lensfile(f'{result_dir}/final_lens.txt', write_zmx=True)
                    lens.write_lens_json(f'{result_dir}/final_lens.json')
                    lens.analysis(save_name=f'{result_dir}/final_lens', draw_layout=True)
                    
                    # distruct instance
                    del lens
                except Exception as e:
                    # append error text and continue
                    with open(error_log_name, "a") as f:
                        f.write(f"Design failed for EPD: {epd}, IMGH: {imgh}, DIST: {dist} with error {e}\n")
                    continue
