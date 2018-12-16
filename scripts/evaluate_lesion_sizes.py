import os

import argparse
import numpy as np
import SimpleITK as sitk
from scipy.ndimage.measurements import label as label_connected_components

from minimum_bounding_box import minimum_bounding_box
from binary_morphology import binary_dilation


def get_parser():
    parser = argparse.ArgumentParser(description="Evaluate lesion sizes")
    parser.add_argument('--dir_vol', help="path to directory with reference "
                                          "segmentation volumes",
                        required=True, type=str)
    parser.add_argument('--min_diameter', help="minimum lesion diameter (mm) "
                                               "to count lesion",
                        required=False, default=0, type=float)
    return parser


def compute_size(mask, voxel_spacing):
    # Volume
    volume = np.count_nonzero(mask)*np.prod(voxel_spacing)
    
    # Maximum diameter
    x, y, z = np.where(mask)
    points_all = list(zip(x, y, z))
    diam_max = 0
    for z_unique in set(z):
        points = [p[:2] for p in points_all if p[2]==z_unique]
        if len(points)==0:
            diam = 0
        elif len(points)==1:
            diam = max(voxel_spacing[0], voxel_spacing[1])
        elif len(points)==2:
            diam = np.subtract(points[0], points[1]).astype(np.float32)
            diam *= np.array(voxel_spacing[:2])
            diam = np.sqrt(np.sum(diam**2))
        else:
            # Check if co-linear
            x_set, y_set = list(zip(*points))
            x_set = set(x_set)
            y_set = set(y_set)
            if len(x_set)==1:
                diam = len(y_set)*voxel_spacing[1]
            elif len(y_set)==1:
                diam = len(x_set)*voxel_spacing[0]
            else:
                # Not co-linear, find bounding box
                bbox = minimum_bounding_box(points)
                uvec_len = bbox.unit_vector*np.array(voxel_spacing[:2])
                uvec_len = np.sqrt(np.sum(uvec_len**2))
                diam = bbox.length_parallel*uvec_len
        if diam > diam_max:
            diam_max = diam
    length = diam_max
    
    return volume, length


if __name__=='__main__':
    parser = get_parser()
    args = parser.parse_args()
    
    if not os.path.exists(args.dir_vol):
        raise ValueError("{} not found".format(args.dir_vol))
    
    count = []
    volumes = []
    lengths = []
    for fn in sorted(os.listdir(args.dir_vol)):
        vol = sitk.ReadImage(os.path.join(args.dir_vol, fn))
        vol_arr = sitk.GetArrayFromImage(vol)
        voxel_spacing = vol.GetSpacing()
        # Dilate the reference lesion mask in the axial plane by 1 before
        # finding connected components in order to get rid of spurious extra
        # "lesions" that are just artefacts or noise about a reference lesion.
        vol_arr_dilated = binary_dilation(vol_arr==2,
                                          spacing=[1,1,0],
                                          radius=1,
                                          flat_struct=True)
        mask_integer, n = label_connected_components(vol_arr_dilated)
        mask_integer[vol_arr!=2] = 0   # Trim the dilation away.
        count.append(n)
        case_volumes = []
        case_lengths = []
        for i in range(1, n+1):
            volume, length = compute_size(mask_integer==i, voxel_spacing)
            if length < args.min_diameter:
                print("WARNING: skipping lesion with diameter {}"
                      "".format(length))
                continue
            case_volumes.append(volume)
            case_lengths.append(length)
        def stringify(val_list):
            return ", ".join("{:.2f}".format(val) for val in val_list)
        print("{}".format(fn))
        print("VOLUMES : {}".format(stringify(case_volumes)))
        print("LENGTHS : {}".format(stringify(case_lengths)))
        print("")
        volumes.extend(case_volumes)
        lengths.extend(case_lengths)
        
    print("-----")
    print("Total number of lesions: {}".format(np.sum(count)))
    print("Per patient, mean (stdev) lesions : {} ({})"
          "".format(np.mean(count), np.std(count)))
    print("Volume mean (stdev) [range] : {} ({}) [{}, {}]"
          "".format(np.mean(volumes),
                    np.std(volumes),
                    np.min(volumes),
                    np.max(volumes)))
    print("Diameter mean (stdev) [range] : {} ({}) [{}, {}]"
          "".format(np.mean(lengths),
                    np.std(lengths),
                    np.min(lengths),
                    np.max(lengths)))
