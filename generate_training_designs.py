import math
import numpy as np
import logging
from auto_lens_design import default_inputs, define_lens, config, curriculum_learning

if __name__ == '__main__':
    
    args = default_inputs()
    
    # set unchanged config
    fov = 37.0
    num_lens = 2
    hfov_rad = math.radians(fov) / 2
    args['HFOV'] = hfov_rad
    args['element'] = num_lens
    args['GLASSES'] = [[1.5346, 56.10]] * num_lens
    args['WAVES'] = [480, 520, 640]
    args['ITER'] = 100
    args['ITER_TEST'] = 10
    args['ITER_LAST'] = 100
    args['ITER_TEST_LAST'] = 10
    
    # define configuration grid
    res = 3
    epd_range = np.linspace(0.5, 5.0, res)
    img_h_range = np.linspace(1.0, 50.0, res)
    dist_range = np.linspace(4.0, 100.0, res)
    
    # create error log file
    error_log_name = "results/generation_error.txt"
    with open(error_log_name, "w") as f:
        f.write("")
    
    # iterate over design grid points
    for epd in epd_range:
        for img_h in img_h_range:
            for dist in dist_range:
                try:
                    # set config
                    fnum = img_h/(2*epd*math.tan(hfov_rad))
                    fnum_start = fnum*1.5
                    diag_start = img_h/2.0
                    rff = dist / img_h
                    args['FNUM'] = fnum
                    args['DIAG'] = img_h
                    args['FNUM_START'] = fnum_start
                    args['DIAG_START'] = diag_start
                    args['rFF'] = rff
                    
                    # arrange design configuration
                    args = config(args)
                    
                    # create lens
                    lens = define_lens(args)
                    # refine lens
                    curriculum_learning(lens, args)
                    
                    # analyze final result
                    lens.prune(outer=0.2)
                    lens.post_computation()

                    logging.info(f'Actual: FOV {lens.hfov}, IMGH {lens.r_last}, F/{lens.fnum}.')
                    result_dir = args['result_dir']
                    lens.write_lensfile(f'{result_dir}/final_lens.txt', write_zmx=True)
                    lens.write_lens_json(f'{result_dir}/final_lens.json')
                    lens.analysis(save_name=f'{result_dir}/final_lens', draw_layout=True)
                    
                    # distruct instance
                    del lens
                except Exception as e:
                    with open(error_log_name, "a") as f:
                        f.write(f"Design failed for EPD: {epd}, IMGH: {img_h}, DIST: {dist} with error {e}\n")
                    continue
