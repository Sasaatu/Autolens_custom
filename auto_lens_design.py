""" Automated lens design with curriculum learning, using RMS errors as loss function.
"""
import string
import argparse
import yaml
import os
import logging
from datetime import datetime
import math
import numpy as np
import torch
import deeplens
from deeplens.utils import *
from deeplens.optics import create_lens
from deeplens.basics import DEPTH


def default_inputs():
    """ Set default inputs for lens design
    """
    args = dict()
    # experiment options
    args['results_root'] = './results'
    args['DEBUG'] = True
    args['brute_force'] = True
    args['seed'] = 1

    # Design spec
    args['HFOV'] = 0.39     # half diagonal FoV in radian, foclen = imgh / 2 / np.tan(hfov)
    args['FNUM'] = 4.0      # F-number, fnum = foclen/aper_D
    args['DIAG'] = 3.0      # sensor diagonal length in mm, imgh = diag, hfov = arctan(diag / 2 / foclen)
    args['element'] = 2     # number of lens

    # curriculum steps
    args['curriculum_steps'] = 5
    args['FNUM_START'] = 6
    args['DIAG_START'] = 2
    args['iter'] = 200
    args['iter_test'] = 20
    args['iter_last'] = 400
    args['iter_test_last'] = 20

    # Learning rate
    args['lrs'] = [5e-4, 1e-4, 1e-1, 1e-4]
    args['ai_lr_decay'] = 0.1

    # System lengths
    args['flange'] = 1.2    # distance from last surface to sensor
    args['rff'] = 1.33      # d_total = imgh * rff # total distance  
    args['d_aper'] = 0.2    # aperture thickness, d_total = d_opt + d_apt + flange

    # Surface geometry types
    args['is_sphere'] = True
    args['is_conic'] = True
    args['is_asphere'] = True
    args['ai_degree'] = 6   # degree of even asphere
    
    # Refractive index parameters
    args['WAVES'] = [520]
    args['GLASS'] = ['n-bk7'] * 2
    
    # Ray tracing parameters for curriculum learning
    args['num_ray'] = 256   # number of rays per field point
    args['num_source'] = 9  # number of field points per axis
    
    return args


def config(args):
    """ Config file for training.
    """
    # Result dir
    results_root = args['results_root']
    num_lens = args['element']
    hfov_rad  = args['HFOV']
    fnum = args['FNUM']
    img_h = args['DIAG']
    epd  = img_h / (2 * fnum * math.tan(hfov_rad))
    tot_dist = img_h * args['rff']
    epd_str = f"{epd:.2f}"
    img_h_str = f"{img_h:.2f}"
    tot_dist_str = f"{tot_dist:.2f}"
    current_time = datetime.now().strftime("%m%d-%H%M%S")
    exp_name = current_time + '-' + str(num_lens) + 'P' + '_Epd' + epd_str + '_ImgH' + img_h_str + '_TotLen' + tot_dist_str
    result_dir = f'{results_root}/{exp_name}'
    os.makedirs(result_dir, exist_ok=True)
    args['result_dir'] = result_dir
    
    # random seed
    set_seed(args['seed'])
    
    # Log
    set_logger(result_dir)
    logging.info(f'EXP: {result_dir}')

    # Device
    num_gpus = torch.cuda.device_count()
    args['num_gpus'] = num_gpus
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    args['device'] = device
    logging.info(f'Using {num_gpus} GPUs')

    return args


def design_lens(args, save_inter_design, save_final_design):
    """ Create lens instance
    """
    HFOV = args['HFOV']
    FNUM = args['FNUM']
    DIAG = args['DIAG']
    result_dir = args['result_dir']
    device = args['device']

    # =====> 1. Load or create lens
    if args['brute_force']:
        create_lens(rff=float(args['rff']), flange=float(args['flange']), d_aper=args['d_aper'], hfov=HFOV, imgh=DIAG, fnum=FNUM, surfnum=args['element'], glass=args['GLASS'], dir=result_dir)
        lens_name = f'./{result_dir}/starting_point_hfov{HFOV}_imgh{DIAG}_fnum{FNUM}.txt'
        lens = deeplens.Lensgroup(filename=lens_name, device=device)
        lens.wave = args['WAVES']
        lens.is_sphere = args['is_sphere']
        lens.is_conic = args['is_conic']
        lens.is_asphere = args['is_asphere']
        for i in lens.find_diff_surf():
            if lens.is_sphere:
                lens.surfaces[i].init_c()
            if lens.is_conic:
                lens.surfaces[i].init_k()
            if lens.is_asphere:
                lens.surfaces[i].init_ai(args['ai_degree'])
            lens.surfaces[i].init_d()
        # delete starting design file
        if not save_final_design:
            os.remove(lens_name)
    else:
        lens = deeplens.Lensgroup(filename=args['filename'])
        lens.correct_shape()
        
    # set target performance
    lens.set_target_fov_fnum(hfov=HFOV, fnum=FNUM, imgh=DIAG)
    # refine lens
    curriculum_learning(lens, args, save_inter_design)

    # analyze final result
    lens.prune(outer=0.2)
    lens.post_computation()
    
    return lens


def change_lens(lens, diag, fnum):
    """ Change lens for each curriculum step.
    """
    # sensor
    lens.r_last = diag / 2
    lens.hfov = np.arctan(lens.r_last / lens.foclen)

    # aperture
    lens.fnum = fnum
    aper_r = lens.foclen / lens.fnum / 2
    lens.surfaces[lens.aper_idx].r = aper_r
    
    return lens


def curriculum_learning(lens, args, save_design):
    """ Curriculum learning for lens design.
    """
    lrs = [float(lr) for lr in args['lrs']]

    curriculum_steps = args['curriculum_steps']
    fnum_target = args['FNUM']
    fnum_end = fnum_target * 0.95
    fnum_start = args['FNUM_START']
    diag_target = args['DIAG']
    diag_end = diag_target * 1.05
    diag_start = args['DIAG_START']
    result_dir = args['result_dir']
    iter = args['iter']
    iter_test = args['iter_test']
    iter_last = args['iter_last']
    iter_test_last = args['iter_test_last']
    num_source = args['num_source']
    num_ray = args['num_ray']
    
    for step in range(curriculum_steps+1):
        
        # ==> Design target for this step
        args['step'] = step
        diag1 = diag_start + (diag_end - diag_start) * np.sin(step / curriculum_steps * np.pi/2)
        fnum1 = fnum_start + (fnum_end - fnum_start) * np.sin(step / curriculum_steps * np.pi/2)
        lens = change_lens(lens, diag1, fnum1)

        logging.info(f'==> Curriculum learning step {step}, target: FOV {round(lens.hfov * 2 * 57.3, 2)}, DIAG {round(2 * lens.r_last, 2)}mm, F/{lens.fnum}.')
        
        # ==> Lens design using RMS errors
        lens.refine(lrs=lrs, decay=args['ai_lr_decay'], iterations=iter, test_per_iter=iter_test, num_source=num_source, num_ray=num_ray, importance_sampling=False, result_dir=result_dir, save_design=save_design)

    # ==> Refine lens at the last step
    lens.refine(iterations=iter_last, test_per_iter=iter_test_last, num_source=num_source, num_ray=num_ray, centroid=True, importance_sampling=True, result_dir=result_dir, save_design=save_design)
    logging.info('==> Training finish.')

    # ==> Final lens
    lens = change_lens(lens, diag_target, fnum_target)


def evaluate_spotsize(lens, M=5, spp=128):
    """ Evaluate spot size of designed lens.
    """
    # evaluate spot size
    mag = 1 / lens.calc_scale_pinhole(DEPTH)
    obj_r = lens.sensor_size[0]/2/mag
    wave = lens.wave[0]
    ray = lens.sample_point_source(M=M,  spp=spp,  depth=DEPTH,  R=obj_r,  pupil=True, wavelength=wave)
    ray, _, _ = lens.trace(ray)
    xy = ray.project_to(lens.d_sensor)
    
    # plot spot diagram
    rms_array = np.zeros(M*M)
    # iterate over field points
    for m in range(M):
        for n in range(M):
            xy_s = xy[:, m, n].detach().cpu().numpy().astype(float)
            xy_cnt = np.mean(xy_s, axis=0)
            dists = np.linalg.norm(xy_s - xy_cnt, axis=1)
            rms_array[M*m+n] = np.sqrt(np.mean(dists**2))
    rms_value = float(np.mean(rms_array))
    
    return rms_value