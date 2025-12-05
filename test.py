""" Automated lens design with curriculum learning, using RMS errors as loss function.
"""
import numpy as np
from matplotlib import pyplot as plt
from deeplens.basics import DEPTH
from auto_lens_design import default_inputs, config, design_lens

if __name__ == '__main__':
    M = 9
    spp = 64
    
    args = default_inputs()
    # custom parmeters
    Waves = [520]
    num_step = 1
    iter = 1
    iter_test = 1
    iter_last = 1
    iter_test_last = 1
    
    # update inputs
    args['WAVES'] = Waves
    args['curriculum_steps'] = num_step
    args['iter'] = iter
    args['iter_test'] = iter_test
    args['iter_last'] = iter_last
    args['iter_test_last'] = iter_test_last
    # arrange design configuration
    args = config(args)
    
    # design lens system
    lens = design_lens(args)

    # evaluate spot size
    mag = 1 / lens.calc_scale_pinhole(DEPTH)
    obj_r = lens.sensor_size[0]/2/mag
    wave = lens.wave[0]
    ray = lens.sample_point_source(M=M,  spp=spp,  depth=DEPTH,  R=obj_r,  pupil=True, wavelength=wave)
    ray, _, _ = lens.trace(ray)
    xy = ray.project_to(lens.d_sensor)
    
    # plot spot diagram
    plt.figure()
    rms_array = np.zeros(M*M)
    # iterate over field points
    for m in range(M):
        for n in range(M):
            xy_s = xy[:, m, n].detach().cpu().numpy().astype(float)
            xy_cnt = np.mean(xy_s, axis=0)
            dist = np.linalg.norm(xy_s - xy_cnt, axis=1)
            rms_array[(M-1)*m+n] = np.sqrt(np.mean(dist**2))
            
            plt.scatter(xy_s[:, 0], xy_s[:, 1], s=1)
    rms_value = float(np.mean(rms_array))
    print(f'RMS spot size: {rms_value*1e3:.2f} um')
    plt.axis('equal')
    plt.title('Spot diagram')
    plt.xlabel('X [mm]')
    plt.ylabel('Y [mm]')
    plt.grid()
    plt.show()
    
    
    
    bjbjb
    
