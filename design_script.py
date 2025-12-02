""" Automated lens design with curriculum learning, using RMS errors as loss function.
"""
import numpy as np
import logging
from auto_lens_design import default_inputs, define_lens, config, curriculum_learning

if __name__ == '__main__':
    args = default_inputs
    
    # custom parmeters
    epd = 0.5
    fov = 37.0
    image_height = 8.0
    total_distance = 5.0
    num_lens = 2
    Glasses = [[1.5346, 56.10], [1.5346, 56.10]]
    Waves = [675]
    
    # update inputs
    hfov_rad = np.deg2rad(fov) / 2
    fnum = image_height/(2*epd*np.tan(hfov_rad))
    rff = total_distance/image_height
    args['HFOV'] = hfov_rad
    args['FNUM'] = fnum
    args['DIAG'] = image_height
    args['element'] = num_lens
    args['rff'] = rff
    args['GLASS'] = Glasses
    args['WAVES'] = Waves
    
    # arrange design configuration
    args = config(args)
    
    # create lens
    lens = define_lens(args)
    # refine lens
    lens = curriculum_learning(lens, args)

    # analyze final result
    lens.prune(outer=0.2)
    lens.post_computation()

    logging.info(f'Actual: FOV {lens.hfov}, IMGH {lens.r_last}, F/{lens.fnum}.')
    lens.write_lensfile(f'{args['result_dir']}/final_lens.txt', write_zmx=True)
    lens.write_lens_json(f'{args['result_dir']}/final_lens.json')
    lens.analysis(save_name=f'{args['result_dir']}/final_lens', draw_layout=True)