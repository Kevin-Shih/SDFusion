import os
import inspect
from datetime import datetime

from termcolor import colored, cprint
from tqdm import tqdm

import torch.backends.cudnn as cudnn
# cudnn.benchmark = True

from options.test_options import TestOptions, TestMyDataOptions
from models.base_model import create_model

import torch
from utils.visualizer import Visualizer

from utils.demo_util import SDFusionImage2ShapeOpt
from pytorch3d import io
from models.base_model import create_model
from utils.util_3d import sdf_to_mesh, save_mesh_as_gif
from utils.demo_util import preprocess_image
from PIL import Image
import numpy as np

def test_img2shape(opt, model, test_mask, save_result=True):
    cprint('[*] Start testing. name: %s' % opt.name, 'blue')
    pbar = tqdm(test_mask, ncols=100)

    for i, test_mask in enumerate(pbar):
        mask_np = np.array(Image.open(test_mask).convert('1'))
        if not np.any(mask_np):
            continue
        img_name = os.path.basename(test_mask).split('.')[0]
        img_path = os.path.join('/home/nycu-reconstruction-2/dataDisk/Dataset/coco/val2017', img_name.split('_')[0]+'.jpg')
        # img_path = os.path.join('data/coco/val2017', img_name.split('_')[0]+'.jpg')
        sdf_gen = model.img2shape(image=img_path, mask=test_mask, ddim_steps=opt.ddim_steps, 
                                  ddim_eta=opt.ddim_eta, uc_scale=opt.uc_scale)
        if save_result:
            if not os.path.exists(opt.out_dir):
                os.makedirs(opt.out_dir)
            save_name = f'{opt.out_dir}/img2shape_{img_name}'
            img = Image.open(img_path)
            img.save(f'{opt.out_dir}/{img_name}.png')
            mask = Image.open(test_mask)
            mask.save(f'{opt.out_dir}/{img_name}_mask.png')

            masked_img, _ = preprocess_image(img_path, test_mask)
            masked_img.save(save_name + '.png')
            mesh_gen = sdf_to_mesh(sdf_gen)
            # io.IO().save_mesh(mesh_gen, save_name + '.ply')
            final_verts, final_faces = mesh_gen.get_mesh_verts_faces(0)
            io.save_obj(save_name + '.obj', final_verts, final_faces)
            # vis as gif
            save_mesh_as_gif(model.renderer, mesh_gen, out_name=save_name + '.gif')
        pbar.update(1)
    pbar.close()
        

if __name__ == "__main__":
    # this will parse args, setup log_dirs, multi-gpus
    # opt = TestOptions().parse_and_setup()
    opt = TestMyDataOptions(seed= 2024).init_model_args()

    # test input
    # test_mask = [os.path.join("../pred_masks", filename) for filename in os.listdir("../pred_masks")[:10]]
    test_mask = ["../pred_masks/000000003934_57_0.png", "../pred_masks/000000004495_57_0.png", "../pred_masks/000000578500_57_1.png"]
    dataset_size = len(test_mask)
    cprint('[*] # testing images = %d' % dataset_size, 'yellow')

    # main loop
    model = create_model(opt)
    cprint(f'[*] "{opt.model}" initialized.', 'cyan')

    with torch.no_grad():
        test_img2shape(opt, model, test_mask, save_result=opt.save_result)
        # test_img2shape(opt, model, test_mask, save_result=True)
