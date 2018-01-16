"""
Reads scores and recompiles information so that it is matched to lesion sizes.
Resulting csv files can be imported in a spreadsheet with one column per 
metric.
"""


import os
import sys
from collections import OrderedDict


# Files of interest contain these strings in their names.
detection_status_str = "ref_detection_status"
ref_size_strings = ["lesion_volume_reference",
                    "lesion_length_reference"]
pre_size_strings = ["lesion_volume_prediction",
                    "lesion_length_prediction"]
other_per_lesion_strings = ["lesion_assd",
                            "lesion_rmsd",
                            "lesion_msd",
                            "lesion_dice.txt",
                            "lesion_voe"]

# Get location of scores.
scores_dir = sys.argv[1]


# How to read csv files.
def read_csv(fn):
    contents = OrderedDict()
    try:
        f_ = open(fn, 'rt')
        for line in f_:
            items = line.strip().split(',')
            contents[items[0]] = items[1:]
        f_.close()
    except:
        raise IOError("Failed to read file {}".format(fn))
    return contents
    
    
# Find the first file with this string in the filename.
def find_first_file(rootdir, fn_str):
    path = None
    for fn in os.listdir(rootdir):
        if fn_str in fn:
            if path is not None:
                raise Exception("More than one file with {} in the filename."
                                "".format(fn_str))
            path = os.path.join(rootdir, fn)
    if path is None:
        raise Exception("No file found with {} in the filename found."
                        "".format(fn_str))
    return path
    
    
for dn in os.listdir(scores_dir):
    rootdir = os.path.join(scores_dir, dn)
    print("Processing {}".format(rootdir))
        
    # Set up data structures to store sizes and other metrics.
    # Dict of dicts. First dict has metrics as keys. Second has cases as keys.
    # Contains list of metric values for the case.
    ref_size = OrderedDict()
    pre_size = OrderedDict()
    metrics = OrderedDict()
    
    # Read detection status file.
    detection_status = read_csv(find_first_file(rootdir, detection_status_str))

    # Read reference sizes.
    for m_str in ref_size_strings:
        ref_size[m_str] = read_csv(find_first_file(rootdir, m_str))
            
    # Read prediction sizes.
    for m_str in pre_size_strings:
        pre_size[m_str] = read_csv(find_first_file(rootdir, m_str))
    
    # Read other metrics.
    for m_str in other_per_lesion_strings:
        metrics[m_str] = read_csv(find_first_file(rootdir, m_str))
                    
    
    # Target filenames start with this string.
    scores_str = scores_dir.split('/')[-1]
    
    # Save csv tracking false positives by size.
    path = os.path.join(scores_dir, dn, scores_str+"_FP_by_size.txt")
    try:
        f_ = open(path, 'wt')
        line = ",".join(['case_name']+pre_size_strings)+"\n"
        f_.write(line)
        for case_name in detection_status:
            # For every reference lesion, there is a predicted lesion 
            # size in the same order as the reference lesion sizes. All
            # sizes reported thereafter are for false positive 
            # predictions.
            #
            # Aggregate the sizes for these false positives.
            n_ref = len(ref_size[ref_size_strings[0]][case_name])
            n_pre = len(pre_size[pre_size_strings[0]][case_name])
            for idx in range(n_ref, n_pre):
                p_list = [pre_size[key][case_name][idx] for key in pre_size]
                line = "{},{}\n".format(case_name, ",".join(p_list))
                f_.write(line)
    except:
        raise IOError("Failed to write file {}".format(path))
        
    
    # Save csv for sizes + detection status.
    path = os.path.join(scores_dir, dn, scores_str+"_detection_with_sizes.txt")
    try:
        f_ = open(path, 'wt')
        line = ",".join(['case_name'] +\
                        list(ref_size.keys()) +\
                        list(pre_size.keys()) +\
                        ['detection_status']) + "\n"
        f_.write(line)
        for case_name in detection_status:
            num_lesions = len(detection_status[case_name])
            for idx in range(num_lesions):
                r_list = [ref_size[key][case_name][idx] for key in ref_size]
                p_list = [pre_size[key][case_name][idx] for key in pre_size]
                line = "{},{},{},{}\n".format(case_name,
                                              ",".join(r_list),
                                              ",".join(p_list),
                                              detection_status[case_name][idx])
                f_.write(line)
        f_.close()
    except:
        raise IOError("Failed to write file {}".format(path))
    
    # Save csv for sizes + segmentation metrics.
    path = os.path.join(scores_dir, dn,
                        scores_str+"_segmentation_with_sizes.txt")
    try:
        f_ = open(path, 'wt')
        line = ",".join(['case_name'] +\
                        list(ref_size.keys()) +\
                        list(pre_size.keys()) +\
                        list(metrics.keys())) + "\n"
        f_.write(line)
        for case_name in detection_status:
            num_lesions = len(detection_status[case_name])
            for idx in range(num_lesions):
                r_list = [ref_size[key][case_name][idx] for key in ref_size]
                if detection_status[case_name][idx]=="1":
                    p_list = [pre_size[key][case_name][idx] for key in pre_size]
                    m_list = [metrics[key][case_name][idx] for key in metrics]
                else:
                    p_list = ["" for key in pre_size]
                    m_list = ["" for key in metrics]
                line = "{},{},{},{}\n".format( \
                                              case_name,
                                              ",".join(r_list),
                                              ",".join(p_list),
                                              ",".join(m_list))
                f_.write(line)
        f_.close()
    except:
        raise IOError("Failed to write file {}".format(path))
