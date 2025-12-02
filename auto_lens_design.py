""" Automated lens design with curriculum learning, using RMS errors as loss function.
"""
import string
import argparse
import yaml
import os
import logging
from datetime import datetime
import torch
import deeplens
from deeplens.utils import *
from deeplens.optics import create_lens


def change_lens(lens, diag, fnum):
    """ Change lens for each curriculum step.
    """
    # sensor
    lens.r_last = float(diag / 2)
    lens.hfov = float(np.arctan(lens.r_last / lens.foclen))

    # aperture
    lens.fnum = float(fnum)
    aper_r = lens.foclen / lens.fnum / 2
    lens.surfaces[lens.aper_idx].r = aper_r
    
    return lens


def curriculum_learning(lens, args):
    """ Curriculum learning for lens design.
    """
    lrs = [float(lr) for lr in args['lrs']]

    curriculum_steps = args['curriculum_steps']
    fnum_target = args['FNUM'] * 0.95
    fnum_start = args['FNUM_START']
    diag_target = args['DIAG'] * 1.05
    diag_start = args['DIAG_START']
    result_dir = args['result_dir']
    iter = args['iter']
    iter_test = args['iter_test']
    iter_last = args['iter_last']
    iter_test_last = args['iter_test_last']
    
    for step in range(args['curriculum_steps']+1):
        
        # ==> Design target for this step
        args['step'] = step
        diag1 = diag_start + (diag_target - diag_start) * np.sin(step / curriculum_steps * np.pi/2)
        fnum1 = fnum_start + (fnum_target - fnum_start) * np.sin(step / curriculum_steps * np.pi/2)
        lens = change_lens(lens, diag1, fnum1)

        lens.analysis(save_name=f'{result_dir}/step{step}_starting_point', zmx_format=True)
        # lens.write_lensfile(f'{result_dir}/step{step}_starting_point.txt', write_zmx=True)
        lens.write_lens_json(f'{result_dir}/step{step}_starting_point.json')
        logging.info(f'==> Curriculum learning step {step}, target: FOV {round(lens.hfov * 2 * 57.3, 2)}, DIAG {round(2 * lens.r_last, 2)}mm, F/{lens.fnum}.')
        
        # ==> Lens design using RMS errors
        lens.refine(lrs=lrs, decay=args['ai_lr_decay'], iterations=iter, test_per_iter=iter_test, importance_sampling=False, result_dir=result_dir)

    # ==> Refine lens at the last step
    lens.refine(iterations=iter_last, test_per_iter=iter_test_last, centroid=True, importance_sampling=True, result_dir=result_dir)
    logging.info('==> Training finish.')

    # ==> Final lens
    lens = change_lens(lens, args['DIAG'], args['FNUM'])
    

def define_lens(args):
    """ Create lens instance
    """
    # =====> 1. Load or create lens
    if args['brute_force']:
        create_lens(rff=float(args['rff']), flange=float(args['flange']), d_aper=args['d_aper'], hfov=args['HFOV'], imgh=args['DIAG'], fnum=args['FNUM'], surfnum=args['element'], glass=args['GLASS'], dir=result_dir)
        lens_name = f'./{args['result_dir']}/starting_point_hfov{args["HFOV"]}_imgh{args["DIAG"]}_fnum{args["FNUM"]}.txt'
        lens = deeplens.Lensgroup(filename=lens_name, device=args['device'])
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
    else:
        lens = deeplens.Lensgroup(filename=args['filename'])
        lens.correct_shape()
    
    lens.set_target_fov_fnum(hfov=args['HFOV'], fnum=args['FNUM'], imgh=args['DIAG'])
    logging.info(f'==> Design target: FOV {round(args["HFOV"]*2*57.3, 2)}, DIAG {args["DIAG"]}mm, F/{args["FNUM"]}, FOCLEN {round(args["DIAG"]/2/np.tan(args["HFOV"]), 2)}mm.')
    lens.analysis(save_name=f'{args['result_dir']}/lens_starting_point')
    
    return lens


def default_inputs():
    """ Set default inputs for lens design
    """
    args = dict()
    # experiment options
    args['DEBUG'] = True
    args['brute_force'] = True
    args['seed'] = 1

    # Design spec
    args['HFOV'] = 0.39
    args['FNUM'] = 4.0
    args['DIAG'] = 3.0
    args['element'] = 2

    # curriculum steps
    args['curriculum_steps'] = 5
    args['FNUM_START'] = 6
    args['DIAG_START'] = 2
    args['iter'] = 100  # 1000
    args['iter_test'] = 10  # 50
    args['iter_last'] = 100 # 5000
    args['iter_test_last'] = 10 # 100

    # Learning rate
    args['lrs'] = [5e-4, 1e-4, 1e-1, 1e-4]
    args['ai_lr_decay'] = 0.1

    # Random initialization
    args['ai_degree'] = 6
    args['flange'] = 1.2
    args['rff'] = 1.33
    args['d_aper'] = 0.2

    # New parameters
    args['is_sphere'] = True
    args['is_conic'] = True
    args['is_asphere'] = True
    args['WAVES'] = [675]
    args['GLASS'] = [[1.5346, 56.10], [1.5346, 56.10]]
    
    return args

def config(args):
    """ Config file for training.
    """
    # Result dir
    num_lens = args['element']
    fov_rad  = args['HFOV']*2
    fnum = args['FNUM']
    img_h = args['DIAG']
    epd  = img_h / (2 * fnum * np.tan(fov_rad/2))
    tot_dist = img_h * args['rff']
    current_time = datetime.now().strftime("%m%d-%H%M%S")
    exp_name = current_time + '-' + str(num_lens) + 'P' + '_Epd' + str(epd) + '_ImgH' + str(img_h) + '_TotLen' + str(tot_dist)
    result_dir = f'./results/{exp_name}'
    os.makedirs(result_dir, exist_ok=True)
    args['result_dir'] = result_dir
    
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


# def main(filename):
#     args = config(filename)
#     result_dir = args['result_dir']
#     device = args['device']

#     # =====> 1. Load or create lens
#     if args['brute_force']:
#         create_lens(rff=float(args['rff']), flange=float(args['flange']), d_aper=args['d_aper'], hfov=args['HFOV'], imgh=args['DIAG'], fnum=args['FNUM'], surfnum=args['element'], glass=args['GLASS'], dir=result_dir)
#         lens_name = f'./{result_dir}/starting_point_hfov{args["HFOV"]}_imgh{args["DIAG"]}_fnum{args["FNUM"]}.txt'
#         lens = deeplens.Lensgroup(filename=lens_name, device=device)
#         lens.wave = args['WAVES']
#         lens.is_sphere = args['is_sphere']
#         lens.is_conic = args['is_conic']
#         lens.is_asphere = args['is_asphere']
#         for i in lens.find_diff_surf():
#             if lens.is_sphere:
#                 lens.surfaces[i].init_c()
#             if lens.is_conic:
#                 lens.surfaces[i].init_k()
#             if lens.is_asphere:
#                 lens.surfaces[i].init_ai(args['ai_degree'])
#             lens.surfaces[i].init_d()
#     else:
#         lens = deeplens.Lensgroup(filename=args['filename'])
#         lens.correct_shape()
    
#     lens.set_target_fov_fnum(hfov=args['HFOV'], fnum=args['FNUM'], imgh=args['DIAG'])
#     logging.info(f'==> Design target: FOV {round(args["HFOV"]*2*57.3, 2)}, DIAG {args["DIAG"]}mm, F/{args["FNUM"]}, FOCLEN {round(args["DIAG"]/2/np.tan(args["HFOV"]), 2)}mm.')
#     lens.analysis(save_name=f'{result_dir}/lens_starting_point')

#     # =====> 2. Curriculum learning with RMS errors
#     curriculum_learning(lens, args)

#     # =====> 3. Analyze final result
#     lens.prune(outer=0.2)
#     lens.post_computation()

#     logging.info(f'Actual: FOV {lens.hfov}, IMGH {lens.r_last}, F/{lens.fnum}.')
#     lens.write_lensfile(f'{result_dir}/final_lens.txt', write_zmx=True)
#     lens.write_lens_json(f'{result_dir}/final_lens.json')
#     lens.analysis(save_name=f'{result_dir}/final_lens', draw_layout=True)


# if __name__=='__main__':
#     filename = './configs/20250828_2P_Asphere_FOV45_Diag3_W675.yml'
#     main(filename)