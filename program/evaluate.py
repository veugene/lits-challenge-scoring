#!/usr/bin/env python
from __future__ import print_function
import sys
import os
import nibabel as nb
import numpy as np
from scipy.ndimage.measurements import label as label_connected_components
import glob
import gc

from helpers.calc_metric import (dice,
                                 detect_lesions,
                                 compute_segmentation_scores,
                                 compute_tumor_burden)



# Check input directories.
submit_dir = os.path.join(sys.argv[1], 'res')
truth_dir = os.path.join(sys.argv[1], 'ref')
if not os.path.isdir(submit_dir):
    print("{} doesn't exist".format(submit_dir))
    sys.exit()
if not os.path.isdir(truth_dir):
    sys.exit()

# Create output directory.
output_dir = sys.argv[2]
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize results dictionaries
lesion_detection_stats = {'TP': 0, 'FP': 0, 'FN': 0}
lesion_segmentation_scores = {}
liver_segmentation_scores = {}
dice_per_case = {'lesion': [], 'liver': []}
dice_global_x = {'lesion': {'I': 0, 'S': 0},
                 'liver':  {'I': 0, 'S': 0}} # 2*I/S
tumor_burden_list = []

# Iterate over all volumes in the reference list.
reference_volume_list = sorted(glob.glob(truth_dir+'/*.nii'))
for reference_volume_fn in reference_volume_list:
    print("Starting with volume {}".format(reference_volume_fn))
    submission_volume_path = os.path.join(submit_dir,
                                          os.path.basename(reference_volume_fn))
    if os.path.exists(submission_volume_path):
        print("Found corresponding submission file {} for reference file {}"
              "".format(reference_volume_fn, submission_volume_path))

        # Load reference and submission volumes with Nibabel.
        reference_volume = nb.load(reference_volume_fn)
        submission_volume = nb.load(submission_volume_path)

        # Get the current voxel spacing.
        voxel_spacing = reference_volume.header.get_zooms()[:3]

        # Get Numpy data and compress to int8.
        reference_volume = (reference_volume.get_data()).astype(np.int8)
        submission_volume = (submission_volume.get_data()).astype(np.int8)
        
        # Ensure that the shapes of the masks match.
        if submission_volume.shape!=reference_volume.shape:
            raise AttributeError("Shapes do not match! Prediction mask {}, "
                                 "ground truth mask {}"
                                 "".format(submission_volume.shape,
                                           reference_volume.shape))
        
        # Create lesion and liver masks with labeled connected components.
        # (Assuming there is always exactly one liver - one connected comp.)
        pred_mask_lesion, num_predicted = label_connected_components( \
                                          submission_volume==2, output=np.int16)
        true_mask_lesion, num_reference = label_connected_components( \
                                          reference_volume==2, output=np.int16)
        pred_mask_liver = submission_volume>=1
        true_mask_liver = reference_volume>=1
        
        # Begin computing metrics.
        print("Start calculating metrics for submission file {}"
              "".format(submission_volume_path))
        
        # Identify detected lesions.
        detected_mask_lesion, num_detected = detect_lesions( \
                                              prediction_mask=pred_mask_lesion,
                                              reference_mask=true_mask_lesion,
                                              min_overlap=0.5)
        
        # Count true/false positive and false negative detections.
        lesion_detection_stats['TP']+=num_detected
        lesion_detection_stats['FP']+=num_predicted-num_detected
        lesion_detection_stats['FN']+=num_reference-num_detected
        
        # Compute segmentation scores for DETECTED lesions.
        import time
        print("DEBUG {}: Computing lesion scores.".format(time.time()))
        lesion_scores = compute_segmentation_scores( \
                                          prediction_mask=detected_mask_lesion,
                                          reference_mask=true_mask_lesion,
                                          voxel_spacing=voxel_spacing)
        for metric in lesion_scores:
            if metric not in lesion_segmentation_scores:
                lesion_segmentation_scores[metric] = []
            lesion_segmentation_scores[metric].extend(lesion_scores[metric])
        
        # Compute liver segmentation scores. 
        print("DEBUG {}: Computing liver scores.".format(time.time()))
        liver_scores = compute_segmentation_scores( \
                                          prediction_mask=pred_mask_liver,
                                          reference_mask=true_mask_liver,
                                          voxel_spacing=voxel_spacing)
        for metric in liver_scores:
            if metric not in liver_segmentation_scores:
                liver_segmentation_scores[metric] = []
            liver_segmentation_scores[metric].extend(liver_scores[metric])
            
        # Compute per-case (per patient volume) dice.
        print("DEBUG {}: Computing lesion dice.".format(time.time()))
        dice_per_case['lesion'].append(dice(pred_mask_lesion,
                                            true_mask_lesion))
        dice_per_case['liver'].append(dice(pred_mask_liver,
                                           true_mask_liver))
        
        # Accumulate stats for global (dataset-wide) dice score.
        dice_global_x['lesion']['I'] += np.count_nonzero( \
            np.logical_and(pred_mask_lesion, true_mask_lesion))
        dice_global_x['lesion']['S'] += np.count_nonzero(pred_mask_lesion) + \
                                        np.count_nonzero(true_mask_lesion)
        dice_global_x['liver']['I'] += np.count_nonzero( \
            np.logical_and(pred_mask_liver, true_mask_liver))
        dice_global_x['liver']['S'] += np.count_nonzero(pred_mask_liver) + \
                                       np.count_nonzero(true_mask_liver)
            
        # Compute tumor burden.
        print("DEBUG {}: Computing tumor burden.".format(time.time()))
        tumor_burden = compute_tumor_burden(prediction_mask=submission_volume,
                                            reference_mask=reference_volume)
        tumor_burden_list.append(tumor_burden)
        
        gc.collect()
        
        
# Compute lesion detection metrics.
TP = lesion_detection_stats['TP']
FP = lesion_detection_stats['FP']
FN = lesion_detection_stats['FN']
lesion_detection_metrics = {'precision': float(TP)/(TP+FP),
                            'recall': float(TP)/(TP+FN)}

# Compute lesion segmentation metrics.
lesion_segmentation_metrics = {}
for m in lesion_segmentation_scores:
    lesion_segmentation_metrics[m] = np.mean(lesion_segmentation_scores[m])
lesion_segmentation_metrics['dice_per_case'] = np.mean(dice_per_case['lesion'])
dice_global = 2.*dice_global_x['lesion']['I']/dice_global_x['lesion']['S']
lesion_segmentation_metrics['dice_global'] = dice_global
    
# Compute liver segmentation metrics.
liver_segmentation_metrics = {}
for m in liver_segmentation_scores:
    liver_segmentation_metrics[m] = np.mean(liver_segmentation_scores[m])
liver_segmentation_metrics['dice_per_case'] = np.mean(dice_per_case['liver'])
dice_global = 2.*dice_global_x['liver']['I']/dice_global_x['liver']['S']
liver_segmentation_metrics['dice_global'] = dice_global

# Compute tumor burden.
tumor_burden_rmse = np.sqrt(np.mean(np.square(tumor_burden_list)))
tumor_burden_max = np.max(tumor_burden_list)


# Print results to stdout.
print("Computed LESION DETECTION metrics:")
for metric, value in lesion_detection_metrics.items():
    print("{}: {:.2f}".format(metric, float(value)))
print("Computed LESION SEGMENTATION metrics (for detected lesions):")
for metric, value in lesion_segmentation_metrics.items():
    print("{}: {:.2f}".format(metric, float(value)))
print("Computed LIVER SEGMENTATION metrics:")
for metric, value in liver_segmentation_metrics.items():
    print("{}: {:.2f}".format(metric, float(value)))
print("Computed TUMOR BURDEN: \n"
    "rmse: {:.2f}\nmax: {:.2f}".format(tumor_burden_rmse, tumor_burden_max))

# Write metrics to file.
output_filename = os.path.join(output_dir, 'scores.txt')
output_file = open(output_filename, 'wb')
for metric, value in lesion_detection_metrics.items():
    output_file.write("lesion_{}: {:.2f}\n".format(metric, float(value)))
for metric, value in lesion_segmentation_metrics.items():
    output_file.write("lesion_{}: {:.2f}\n".format(metric, float(value)))
for metric, value in liver_segmentation_metrics.items():
    output_file.write("liver_{}: {:.2f}\n".format(metric, float(value)))

#Tumorburden
output_file.write("RMSE_Tumorburden: {:.2f}\n".format(tumor_burden_rmse))
output_file.write("MAXERROR_Tumorburden: {:.2f}\n".format(tumor_burden_max))

output_file.close()
