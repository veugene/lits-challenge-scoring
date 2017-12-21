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
size_strings = ["lesion_volume_reference",
                "lesion_length_reference"]
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
    

# Given a dict with case names as keys and lists of per-lesion metrics, return
# a list with all per-lesion metrics concatenated.
#
# If detection_status is passed, put blank for every missing lesion.
def dict_to_list(d, detection_status=None):
    ret_list = []
    for key in d:
        if detection_status is None:
            ret_list.extend(d[key])
        else:
            for i, status in enumerate(detection_status[key]):
                if float(status)==0:
                    ret_list.append("")
                else:
                    ret_list.append(d[key][i])
        #print(key, len(ret_list))
    return ret_list
    
    
for dn in os.listdir(scores_dir):
    print("Processing {}".format(os.path.join(scores_dir, dn)))

    # Read detection status file.
    detection_status = None
    for fn in os.listdir(os.path.join(scores_dir, dn)):
        if detection_status_str in fn:
            if detection_status is not None:
                raise Exception("More than one file with "
                                "\'detection_status\' in the filename.")
            detection_status = read_csv(os.path.join(scores_dir, dn, fn))
    if detection_status is None:
        raise Exception("No file with \'detection status\' in the filename "
                        "found.")
        
    # Set up data structures to store sizes and other metrics.
    # Dict of lists; first list is case name. Each inner list represents 
    # column. One column per metric.
    size_grid = OrderedDict((('case_name', []),))
    metrics_grid = OrderedDict((('case_name', []),))

    # Set up first column. Case name for each lesion.
    case_names = []
    for key in detection_status.keys():
        name_list = [key]*len(detection_status[key])
        case_names.extend(name_list)
    size_grid['case_name'] = case_names
    metrics_grid['case_name'] = case_names

    # Read sizes.
    for m_str in size_strings:
        if m_str not in size_grid:
            size_grid[m_str] = []
        for fn in os.listdir(os.path.join(scores_dir, dn)):
            if m_str not in fn:
                continue
            contents_dict = read_csv(os.path.join(scores_dir, dn, fn))
            size_grid[m_str].extend( dict_to_list(contents_dict) )
    

    # Read other metrics.
    for m_str in other_per_lesion_strings:
        if m_str not in metrics_grid:
            metrics_grid[m_str] = []
        for fn in os.listdir(os.path.join(scores_dir, dn)):
            if m_str not in fn:
                continue
            #print("DEBUG", fn)
            contents_dict = read_csv(os.path.join(scores_dir, dn, fn))
            contents_list = dict_to_list(contents_dict, detection_status)
            metrics_grid[m_str].extend(contents_list)
    
    
    scores_str = scores_dir.split('/')[-1]
    
    # Save csv for sizes + detection status.
    for key in size_grid:
        assert(len(size_grid[key])==len(size_grid['case_name']))
    path = os.path.join(scores_dir, dn, scores_str+"_detection_with_sizes.txt")
    try:
        f_ = open(path, 'wt')
        line = ",".join(list(size_grid.keys())+['detection_status'])+"\n"
        f_.write(line)
        detection_status_list = []
        for key in detection_status.keys():
            detection_status_list.extend(detection_status[key])
        for idx, case_name in enumerate(size_grid['case_name']):
            s_list = [size_grid[key][idx] for key in size_grid.keys()]
            line = ",".join(s_list)+",{}\n".format(detection_status_list[idx])
            f_.write(line)
        f_.close()
    except:
        raise IOError("Failed to write file {}".format(path))
    
    # Save csv for sizes + segmentation metrics.
    for key in metrics_grid:
        assert(len(metrics_grid[key])==len(metrics_grid['case_name']))
    path = os.path.join(scores_dir, dn,
                        scores_str+"_segmentation_with_sizes.txt")
    try:
        f_ = open(path, 'wt')
        key_list = list(size_grid.keys())+list(metrics_grid.keys())[1:]
        line = ",".join(key_list)+"\n"
        f_.write(line)
        for idx in range(len(metrics_grid['case_name'])):
            s_list = [size_grid[key][idx] for key in size_grid.keys()]
            m_list = [metrics_grid[key][idx] for key in metrics_grid.keys()
                      if key!='case_name']
            line = ",".join(s_list+m_list)+"\n"
            f_.write(line)
        f_.close()
    except:
        raise IOError("Failed to write file {}".format(path))
