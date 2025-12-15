""" Automated lens design with curriculum learning, using RMS errors as loss function.
"""
import math
from auto_lens_design import default_inputs, config, design_lens

if __name__ == '__main__':
    args = default_inputs()
    
    # custom parmeters
    fov = 70.0
    epd = 1.5
    epd_start = 1.0
    image_height = 8.0
    image_height_start = 4.0
    total_distance = 5.0
    num_lens = 3
    Glasses = [[1.5346, 56.10]]*num_lens
    Waves = [675]
    iter = 500
    iter_test = 50
    iter_last = 200
    iter_test_last = 50
    
    # update inputs
    hfov_rad = math.radians(fov) / 2
    fnum = image_height/(2*epd*math.tan(hfov_rad))
    fnum_start = fnum*(epd/epd_start)
    diag_start = image_height_start
    rff = total_distance/image_height
    args['HFOV'] = hfov_rad
    args['FNUM'] = fnum
    args['DIAG'] = image_height
    args['FNUM_START'] = fnum_start
    args['DIAG_START'] = diag_start
    args['rff'] = rff
    args['element'] = num_lens
    args['GLASSES'] = Glasses
    args['WAVES'] = Waves
    args['iter'] = iter
    args['iter_test'] = iter_test
    args['iter_last'] = iter_last
    args['iter_test_last'] = iter_test_last
    
    # arrange design configuration
    args = config(args)
    
    # design lens system
    lens = design_lens(args)