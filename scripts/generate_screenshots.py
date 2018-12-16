import os

import argparse
import numpy as np
from PIL import Image
import SimpleITK as sitk 



def get_parser():
    parser = argparse.ArgumentParser(description="Screenshot generator")
    parser.add_argument('--dir_vol', required=True)
    parser.add_argument('--dir_seg', required=True)
    parser.add_argument('--dir_ref', required=True)
    parser.add_argument('--dir_out', required=True)
    parser.add_argument('--alpha', type=float, default=0.3, required=False)
    parser.add_argument('--level', type=int, default=100, required=False)
    parser.add_argument('--window', type=int, default=300, required=False)
    return parser


def create_labels(v_ref, v_seg, v_vol):
    '''
    Creates a volume with labels as:
    0 - background
    1 - undersegmentation
    2 - correct segmentation
    3 - oversegmentation
    '''
    v_ref_np = sitk.GetArrayFromImage(v_ref).astype(np.bool)
    v_seg_np = sitk.GetArrayFromImage(v_seg).astype(np.bool)
    v_vol_np = sitk.GetArrayFromImage(v_vol).astype(np.bool)
    assert v_ref_np.shape==v_seg_np.shape==v_vol_np.shape
    
    v_out_np = np.zeros(v_ref_np.shape, np.uint8)
    v_out_np[np.logical_and(v_ref_np==1, v_seg_np==0)] = 1
    v_out_np[np.logical_and(v_ref_np==1, v_seg_np==1)] = 2
    v_out_np[np.logical_and(v_ref_np==0, v_seg_np==1)] = 3
    
    return v_out_np


def save_slices_with_lesions(volume, labels, path, base, alpha=0.3):
    if not os.path.exists(path):
        os.makedirs(path)
    indices = np.unique(np.where(labels)[0])
    for idx in indices:
        v_slice = volume[idx,:,:]
        v_slice_rgb = np.zeros(v_slice.shape+(4,), dtype=np.uint8)
        v_slice_rgb[:,:,:3] = v_slice[:,:,None]
        v_slice_rgb[:,:,3] = 255
        v_im = Image.fromarray(np.hstack([v_slice_rgb,
                                          v_slice_rgb]), mode='RGBA')
        
        l_slice = labels[idx,:,:]
        l_slice_rgb = np.zeros(l_slice.shape+(4,), dtype=np.uint8)
        l_slice_rgb[:,:,0][l_slice==1] = 255    # Red, undersegmentation
        l_slice_rgb[:,:,1][l_slice==2] = 255    # Green, correct
        l_slice_rgb[:,:,2][l_slice==3] = 255    # Blue, oversegmentation
        l_slice_rgb[:,:,3][l_slice!=0] = alpha*255
        l_zero_rgb = np.zeros(l_slice_rgb.shape, dtype=np.uint8)
        l_im = Image.fromarray(np.hstack([l_zero_rgb,
                                          l_slice_rgb]), mode='RGBA')
        
        o_im = Image.alpha_composite(v_im, l_im)
        o_im.save(os.path.join(path, "{}__{}.png".format(base, idx)))


if __name__=='__main__':
    parser = get_parser()
    args = parser.parse_args()
    
    for path in [args.dir_vol, args.dir_seg, args.dir_ref]:
        if not os.path.exists(path):
            raise ValueError("{} does not exist".format(path))

    for fn in sorted(os.listdir(args.dir_ref)):
        if 'CAN_91_' not in fn:
            continue
        base = fn.rsplit('.')[0]
        print(base)
        
        # Read images.
        v_ref = sitk.ReadImage(os.path.join(args.dir_ref, fn))
        v_seg = sitk.ReadImage(os.path.join(args.dir_seg, fn))
        v_vol = sitk.ReadImage(os.path.join(args.dir_vol, base+"_CT.nii.gz"))
        
        # Window and level settings on volume.
        rescaler = sitk.IntensityWindowingImageFilter()
        rescaler.SetWindowMinimum(args.level-args.window//2)
        rescaler.SetWindowMaximum(args.level+args.window//2)
        rescaler.SetOutputMinimum(0)
        rescaler.SetOutputMaximum(255)
        v_vol_np = sitk.GetArrayFromImage(rescaler.Execute(v_vol))
        v_vol_np = v_vol_np.astype(np.uint8)
        
        # Create axial slice screenshots with informative labels.
        v_out_np = create_labels(v_ref, v_seg, v_vol)
        save_slices_with_lesions(volume=v_vol_np,
                                 labels=v_out_np,
                                 path=args.dir_out,
                                 base=base,
                                 alpha=args.alpha)
