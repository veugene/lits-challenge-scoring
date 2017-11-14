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
                                 compute_tumor_burden,
                                 LARGE)
from helpers.binary_morphology import binary_dilation
from helpers.utils import time_elapsed


# Check input directories.
truth_dir = os.path.join(sys.argv[1])
submit_dir = os.path.join(sys.argv[2])
if not os.path.isdir(submit_dir):
    print("submit_dir {} doesn't exist".format(submit_dir))
    sys.exit()
if not os.path.isdir(truth_dir):
    print("truth_dir {} doesn't exist".format(submit_dir))
    sys.exit()

# Create output directory.
output_dir = sys.argv[3]
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
# Segmentation metrics and their default values for when there are no detected
# objects on which to evaluate them.
#
# Surface distance (and volume difference) metrics between two masks are
# meaningless when any one of the masks is empty. Assign maximum (infinite)
# penalty. The average score for these metrics, over all objects, will thus
# also not be finite as it also loses meaning.
segmentation_metrics = {'dice': [0],
                        'jaccard': [0],
                        'voe': [1],
                        'rvd': [LARGE],
                        'assd': [LARGE],
                        'rmsd': [LARGE],
                        'msd': [LARGE]}

# Initialize results dictionaries
lesion_detection_stats = {0:   {'TP': [], 'FP': [], 'FN': []},
                          0.5: {'TP': [], 'FP': [], 'FN': []}}
split_merge_errors = {0:   {'merge': [], 'split': []},
                      0.5: {'merge': [], 'split': []}}
detection_status = {0: [], 0.5: []}
lesion_segmentation_scores = {}
liver_segmentation_scores = {}
dice_per_case = {'lesion': [], 'liver': []}
dice_global_x = {'lesion': {'I': 0, 'S': 0},
                 'liver':  {'I': 0, 'S': 0}} # 2*I/S
tumor_burden_list = []
volume_id_list = []
lesion_sizes = {'reference': [], 'prediction': []}

               
"""
Iterate over all volumes in the reference list, computing metrics.
"""
reference_volume_list = sorted(glob.glob(truth_dir+'/*.nii.gz'))
for reference_volume_fn in reference_volume_list:
    print("Starting with volume {}".format(reference_volume_fn))
    submission_volume_path = os.path.join(submit_dir,
                                          os.path.basename(reference_volume_fn))
    if not os.path.exists(submission_volume_path):
        raise ValueError("Submission volume not found - terminating!\n"
                         "Missing volume: {}".format(submission_volume_path))
    print("Found corresponding submission file {} for reference file {}"
          "".format(submission_volume_path, reference_volume_fn))
    volume_id = os.path.basename(reference_volume_fn)[:-len(".nii.gz")]
    volume_id_list.append(volume_id)
    t = time_elapsed()

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
    print("Done loading files ({:.2f} seconds)".format(t()))
    
    # Create lesion and liver masks with labeled connected components.
    # (Assuming there is always exactly one liver - one connected comp.)
    #
    # Dilate the reference lesion mask in the axial plane by 1 before
    # finding connected components in order to get rid of spurious extra
    # "lesions" that are just artefacts or noise about a reference lesion.
    pred_mask_lesion, n_predicted = label_connected_components( \
                                         submission_volume==2, output=np.int16)
    dilated_reference_volume = binary_dilation(reference_volume==2,
                                               spacing=[1,1,0],
                                               radius=1,
                                               flat_struct=True)
    true_mask_lesion, n_reference = label_connected_components( \
                                     dilated_reference_volume, output=np.int16)
    true_mask_lesion[reference_volume!=2] = 0   # Trim the dilation away.
    pred_mask_liver = submission_volume>=1
    true_mask_liver = reference_volume>=1
    liver_exists = np.any(submission_volume==1) and np.any(reference_volume==1)
    print("Done finding connected components ({:.2f} seconds)".format(t()))
    
    # Compute lesion volume.
    volume_ref = np.product(voxel_spacing)*np.count_nonzero(true_mask_lesion)
    volume_pre = np.product(voxel_spacing)*np.count_nonzero(pred_mask_lesion)
    lesion_sizes['reference'].append(volume_ref)
    lesion_sizes['prediction'].append(volume_pre)
    
    # Identify detected lesions.
    # Retain detected_mask_lesion for overlap > 0.5
    for overlap in [0, 0.5]:
        detection_out = detect_lesions(prediction_mask=pred_mask_lesion,
                                       reference_mask=true_mask_lesion,
                                       min_overlap=overlap)
        detected_mask_lesion, mod_ref_mask, \
        n_detected, n_merge_errors, n_split_errors, \
        g_id_detected = detection_out
        
        # Count true/false positive and false negative detections.
        lesion_detection_stats[overlap]['TP'].append(n_detected)
        lesion_detection_stats[overlap]['FP'].append(n_predicted-n_detected)
        lesion_detection_stats[overlap]['FN'].append(n_reference-n_detected)
        
        # Count merge and split errors.
        split_merge_errors[overlap]['merge'].append(n_merge_errors)
        split_merge_errors[overlap]['split'].append(n_split_errors)
        
        # Note which reference lesions were detected and which were not.
        detection_status[overlap].append(g_id_detected)
        
    print("Done identifying detected lesions ({:.2f} seconds)".format(t()))
    
    # Compute segmentation scores for DETECTED lesions.
    if n_detected>0:
        lesion_scores = compute_segmentation_scores( \
                                          prediction_mask=detected_mask_lesion,
                                          reference_mask=mod_ref_mask,
                                          voxel_spacing=voxel_spacing)
        for metric in segmentation_metrics:
            if metric not in lesion_segmentation_scores:
                lesion_segmentation_scores[metric] = []
            lesion_segmentation_scores[metric].append(lesion_scores[metric])
        print("Done computing lesion scores ({:.2f} seconds)".format(t()))
    else:
        print("No lesions detected, skipping lesion score evaluation")
    
    # Compute liver segmentation scores. 
    if liver_exists:
        liver_scores = compute_segmentation_scores( \
                                          prediction_mask=pred_mask_liver,
                                          reference_mask=true_mask_liver,
                                          voxel_spacing=voxel_spacing)
        for metric in segmentation_metrics:
            if metric not in liver_segmentation_scores:
                liver_segmentation_scores[metric] = []
            liver_segmentation_scores[metric].append(liver_scores[metric])
        print("Done computing liver scores ({:.2f} seconds)".format(t()))
    else:
        # No liver label. Record default score values (zeros, inf).
        # NOTE: This will make some metrics evaluate to inf over the entire
        # dataset.
        for metric in segmentation_metrics:
            if metric not in liver_segmentation_scores:
                liver_segmentation_scores[metric] = []
            liver_segmentation_scores[metric].append(\
                                                  segmentation_metrics[metric])
        print("No liver label provided, skipping liver score evaluation")
        
    # Compute per-case (per patient volume) dice.
    if not np.any(pred_mask_lesion) and not np.any(true_mask_lesion):
        dice_per_case['lesion'].append(1.)
    else:
        dice_per_case['lesion'].append(dice(pred_mask_lesion,
                                            true_mask_lesion))
    if liver_exists:
        dice_per_case['liver'].append(dice(pred_mask_liver,
                                           true_mask_liver))
    else:
        dice_per_case['liver'].append(0)
    
    # Accumulate stats for global (dataset-wide) dice score.
    dice_global_x['lesion']['I'] += np.count_nonzero( \
        np.logical_and(pred_mask_lesion, true_mask_lesion))
    dice_global_x['lesion']['S'] += np.count_nonzero(pred_mask_lesion) + \
                                    np.count_nonzero(true_mask_lesion)
    if liver_exists:
        dice_global_x['liver']['I'] += np.count_nonzero( \
            np.logical_and(pred_mask_liver, true_mask_liver))
        dice_global_x['liver']['S'] += np.count_nonzero(pred_mask_liver) + \
                                       np.count_nonzero(true_mask_liver)
    else:
        # NOTE: This value should never be zero.
        dice_global_x['liver']['S'] += np.count_nonzero(true_mask_liver)
        
        
    print("Done computing additional dice scores ({:.2f} seconds)"
          "".format(t()))
        
    # Compute tumor burden.
    tumor_burden = compute_tumor_burden(prediction_mask=submission_volume,
                                        reference_mask=reference_volume)
    tumor_burden_list.append(tumor_burden)
    print("Done computing tumor burden diff ({:.2f} seconds)".format(t()))
    
    print("Done processing volume (total time: {:.2f} seconds)"
          "".format(t.total_elapsed()))
    gc.collect()
    
    
"""
Compute and record global metrics given recorded metrics info.
"""
# Compute lesion detection metrics.
lesion_detection_metrics = {}
for overlap in [0, 0.5]:
    TP = np.sum(lesion_detection_stats[overlap]['TP'])
    FP = np.sum(lesion_detection_stats[overlap]['FP'])
    FN = np.sum(lesion_detection_stats[overlap]['FN'])
    precision = float(TP)/(TP+FP) if TP+FP else 0
    recall = float(TP)/(TP+FN) if TP+FN else 0
    n_merge_err = np.sum(split_merge_errors[overlap]['merge'])
    n_split_err = np.sum(split_merge_errors[overlap]['split'])
    lesion_detection_metrics['precision_'+str(overlap)] = precision
    lesion_detection_metrics['recall_'+str(overlap)] = recall
    lesion_detection_metrics['TP_'+str(overlap)] = TP
    lesion_detection_metrics['FP_'+str(overlap)] = FP
    lesion_detection_metrics['FN_'+str(overlap)] = FN
    lesion_detection_metrics['num_merge_errors_'+str(overlap)] = n_merge_err
    lesion_detection_metrics['num_split_errors_'+str(overlap)] = n_split_err

# Compute lesion segmentation metrics.
lesion_segmentation_metrics = {}
for m in lesion_segmentation_scores:
    scores = sum(lesion_segmentation_scores[m], [])     # flatten lists
    lesion_segmentation_metrics[m] = np.mean(scores)
if len(lesion_segmentation_scores)==0:
    # Nothing detected - set default values.
    lesion_segmentation_metrics.update(segmentation_metrics)
lesion_segmentation_metrics['dice_per_case'] = np.mean(dice_per_case['lesion'])
dice_global = 2.*dice_global_x['lesion']['I']/dice_global_x['lesion']['S']
lesion_segmentation_metrics['dice_global'] = dice_global
    
# Compute liver segmentation metrics.
liver_segmentation_metrics = {}
for m in liver_segmentation_scores:
    scores = sum(liver_segmentation_scores[m], [])     # flatten lists
    liver_segmentation_metrics[m] = np.mean(scores)
if len(liver_segmentation_scores)==0:
    # Nothing detected - set default values.
    liver_segmentation_metrics.update(segmentation_metrics)
liver_segmentation_metrics['dice_per_case'] = np.mean(dice_per_case['liver'])
dice_global = 2.*dice_global_x['liver']['I']/dice_global_x['liver']['S']
liver_segmentation_metrics['dice_global'] = dice_global

# Compute tumor burden.
tumor_burden_rmse = np.sqrt(np.mean(np.square(tumor_burden_list)))
tumor_burden_max = np.max(tumor_burden_list)


# Print results to stdout.
print("Computed LESION DETECTION metrics:")
for metric, value in lesion_detection_metrics.items():
    print("{}: {:.3f}".format(metric, float(value)))
print("Computed LESION SEGMENTATION metrics (for detected lesions):")
for metric, value in lesion_segmentation_metrics.items():
    print("{}: {:.3f}".format(metric, float(value)))
print("Computed LIVER SEGMENTATION metrics:")
for metric, value in liver_segmentation_metrics.items():
    print("{}: {:.3f}".format(metric, float(value)))
print("Computed TUMOR BURDEN: \n"
    "rmse: {:.3f}\nmax: {:.3f}".format(tumor_burden_rmse, tumor_burden_max))

# Write metrics to file.
output_filename = os.path.join(output_dir, 'scores_global.txt')
try:
    output_file = open(output_filename, 'w')
except:
    raise IOError("Failed to open file {}".format(output_filename))
for metric, value in lesion_detection_metrics.items():
    output_file.write("lesion_{}: {:.3f}\n".format(metric, float(value)))
for metric, value in lesion_segmentation_metrics.items():
    output_file.write("lesion_{}: {:.3f}\n".format(metric, float(value)))
for metric, value in liver_segmentation_metrics.items():
    output_file.write("liver_{}: {:.3f}\n".format(metric, float(value)))

# Tumor burden
output_file.write("RMSE_Tumorburden: {:.3f}\n".format(tumor_burden_rmse))
output_file.write("MAXERROR_Tumorburden: {:.3f}\n".format(tumor_burden_max))

output_file.close()


"""
For each metric, record per case (and per lesion, if applicable) score.
"""
def record_metric(score_list, metric_name):
    output_filename = os.path.join(output_dir,
                                   'scores_{}.txt'.format(metric_name))
    try:
        output_file = open(output_filename, 'w')
    except:
        raise IOError("Failed to open file {}".format(output_filename))
    for i in range(len(score_list)):
        if hasattr(score_list[i], '__len__'):
            out_line = ",".join([str(s) for s in score_list[i]])
        else:
            out_line = str(score_list[i])
        out_line = "{},{}\n".format(str(volume_id_list[i]), out_line)
        output_file.write(out_line)
    output_file.close()
    
for overlap in [0, 0.5]:
    for metric in lesion_detection_stats[overlap]:
        record_metric(score_list=lesion_detection_stats[overlap][metric],
                      metric_name='{}_{}'.format(metric, overlap))
    record_metric(score_list=split_merge_errors[overlap]['merge'],
                  metric_name='num_merge_errors_'+str(overlap))
    record_metric(score_list=split_merge_errors[overlap]['split'],
                  metric_name='num_split_errors_'+str(overlap))
    detected = []
    for d in detection_status[overlap]:
        _detected = []
        for key in sorted(d.keys()):
            assert(key in range(1, len(d)+1))
            _detected.append(d[key])
        detected.append(_detected)
    record_metric(score_list=detected,
                  metric_name='ref_detection_status_'+str(overlap))
        
for metric in lesion_segmentation_scores:
    record_metric(score_list=lesion_segmentation_scores[metric],
                  metric_name='lesion_{}'.format(metric))
for metric in lesion_segmentation_scores:
    record_metric(score_list=liver_segmentation_scores[metric],
                  metric_name='liver_{}'.format(metric))
record_metric(score_list=dice_per_case['lesion'],
              metric_name='lesion_dice_per_case')
record_metric(score_list=dice_per_case['liver'],
              metric_name='liver_dice_per_case')
record_metric(score_list=tumor_burden_list, metric_name='tumor_burden')
record_metric(score_list=lesion_sizes['reference'],
              metric_name='lesion_volume_reference')
record_metric(score_list=lesion_sizes['prediction'],
              metric_name='lesion_volume_prediction')
