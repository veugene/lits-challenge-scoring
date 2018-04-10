#sensitivity
#positive predictive value
#Dice
#MSSD
#ASSD
#pooled kappa
#quantity disagreement
#allocation disagreement
#ICC

from multiprocessing import Pool
from functools import partial
import argparse
import os
from collections import OrderedDict

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Arithmetic Task")
    parser.add_argument("scores_dir", type=str,
                        help="path to the scores directory")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="for 1-alpha confidence interval")
    parser.add_argument('--bootstrap_iterations', type=int, default=10000,
                        help="number of iterations to run bootstraping")
    parser.add_argument('--n_proc', type=int, default=None,
                        help="number of processes to spread bootstrapping "
                             "over")
    args = parser.parse_args()
    return args


##############################################################################
# 
#  DEFINE: Compute confidence intervals for a metric, using bootstrapping.
# 
##############################################################################
def bootstrap(data, metric_function, alpha,
              n_iterations=10000, n_proc=None, rng=None):
    if n_proc is None:
        n_proc = os.cpu_count()
        if n_proc is None:
            n_proc = 1
    if rng is None:
        rng = np.random.RandomState()
    data = np.array(data)
    n_indices = len(data)
    
    # Mean over orignal sample.
    metric_mean = np.mean(metric_function(data))
    
    # Multiprocessing: resample with replacement and recompute metric.
    pool = Pool(n_proc)
    def get_sample():
        indices = rng.choice(n_indices, size=n_indices, replace=True)
        data_sample = data[indices]
        return data_sample
    r = pool.map_async(metric_function,
                       (get_sample() for i in range(n_iterations)))
    sample_metric = r.get()
    pool.close()
    pool.join()
    delta = [np.mean(sample_metric[i])-metric_mean \
                                                  for i in range(n_iterations)]
    
    # Confindence interval
    delta = sorted(delta)
    idx0 = int(len(delta)*alpha//2)
    idx1 = int(len(delta)-1-idx0)
    confidence_interval = (metric_mean-delta[idx1], metric_mean-delta[idx0])
    
    return confidence_interval


##############################################################################
# 
#  DEFINE: Detection metrics.
# 
##############################################################################

''' Computes kappa, quantity disagreement, and allocation disagreement. '''
def _kappa(data, idxA, idxB):
    a1 = data[:, idxA]==1
    a0 = data[:, idxA]==0
    b1 = data[:, idxB]==1
    b0 = data[:, idxB]==0
    
    n11 = sum(a1*b1)
    n10 = sum(a1*b0) 
    n01 = sum(a0*b1)
    n00 = sum(a0*b0)
    
    a_quant = abs(n01-n10)/float(n11+n00)
    b_quant = abs(n10-n01)/float(n11+n00)
    a_alloc = 2*min(n10, n01)/float(n11+n10+n01+n00)
    b_alloc = 2*min(n10, n01)/float(n11+n10+n01+n00)
    
    quantity_disagreement = (a_quant+b_quant)/2.
    allocation_disagreemnt = (a_alloc+b_alloc)/2.
    
    po = (n11+n00)/float(n11+n10+n01+n00)
    pe = ((n11+n10)*(n11+n01)+(n10+n01)*(n10+n00))/float(n11+n10+n01+n00)**2
    
    kappa = (po-pe)/(1-pe)
    
    values = {'po': po,
              'pe': pe,
              'kappa': kappa,
              'quantity_disagreement': quantity_disagreement,
              'allocation_disagreemnt': allocation_disagreemnt}
    
    return values


def pooled_kappa(data, index_pairs):
    po_list, pe_list = [], []
    for idxA, idxB in index_pairs:
        v = _kappa(data, idxA=idxA, idxB=idxB)
        po_list.append(v['po'])
        pe_list.append(v['pe'])
    po = np.mean(po_list)
    pe = np.mean(pe_list)
    kappa = (po-pe)/(1-pe)
    return kappa


def quantity_disagreement(data, index_pairs):
    quant_list = []
    for idxA, idxB in index_pairs:
        v = _kappa(data, idxA=idxA, idxB=idxB)
        quant_list.append(v['quantity_disagreement'])
    quantity_disagreement = np.mean(quant_list)
    return quantity_disagreement


def allocation_disagreement(data, index_pairs):
    alloc_list = []
    for idxA, idxB in index_pairs:
        v = _kappa(data, idxA=idxA, idxB=idxB)
        alloc_list.append(v['allocation_disagreemnt'])
    allocation_disagreement = np.mean(alloc_list)
    return allocation_disagreement


##############################################################################
# 
#  DEFINE: Segmentation metrics.
# 
##############################################################################

# TODO


##############################################################################
# 
#  DEFINE: Load data.
# 
##############################################################################

subdirectories = ['automatic', 'correction_A1', 'correction_A2',
                  'correction_W1', 'correction_W2', 'manual_A1',
                  'manual_A2', 'manual_W1', 'manual_W2']


# File names and the order of their contents.
# Generated by `merge_scores.py`.
detection_file_str = "detection_with_sizes.txt"
segmentation_file_str = "segmentation_with_sizes.txt"
segmentation_entries = ['case_name', 'volume_reference',
                        'length_reference', 'volume_prediction',
                        'length_prediction', 'assd', 'rmsd', 'msd',
                        'dice', 'voe']
detection_entries = ['case_name', 'volume_reference', 'length_reference',
                     'volume_prediction', 'length_prediction',
                     'detection_status']


def load_data(scores_dir):
    # Find the first file with this string in the filename.
    def find_first_file(rootdir, fn_str):
        path = None
        for fn in os.listdir(rootdir):
            if fn_str in fn:
                if path is not None:
                    raise Exception("More than one file with {} in the "
                                    "filename.".format(fn_str))
                path = os.path.join(rootdir, fn)
        if path is None:
            raise Exception("No file found with {} in the filename found."
                            "".format(fn_str))
        return path
    
    # How to read csv files.
    def read_csv(fn):
        contents = []
        try:
            f_ = open(fn, 'rt')
            for line in f_:
                items = line.strip().split(',')
                contents.append(items)
            f_.close()
            contents = contents[1:]     # Drop first line.
        except:
            raise IOError("Failed to read file {}".format(fn))
        return contents
    
    # Load file from each subdirectory.
    csv_det_dict = {}
    csv_seg_dict = {}
    for dn in subdirectories:
        rootdir = os.path.join(scores_dir, dn)
        csv_det = read_csv(find_first_file(rootdir, detection_file_str))
        csv_seg = read_csv(find_first_file(rootdir, segmentation_file_str))
        csv_det_dict[dn] = csv_det
        csv_seg_dict[dn] = csv_seg
        
    return csv_det_dict, csv_seg_dict


def scrub_detection_status(csv_det_dict):
    data = []
    idx = detection_entries.index('detection_status')
    for dn in subdirectories:
        column = []
        for line in csv_det_dict[dn]:
            column.append(int(line[idx]))
        data.append(column)
    data = np.array(data).T
    return data


def scrub_predicted_volumes(csv_det_dict):
    data = []
    idx = detection_entries.index('volume_prediction')
    for dn in subdirectories:
        column = []
        for line in csv_det_dict[dn]:
            column.append(int(line[idx]))
        data.append(column)
    data = np.array(data).T
    return data


##############################################################################
# 
#  Setup.
# 
##############################################################################

args = parse_args()
csv_det_dict, csv_seg_dict = load_data(args.scores_dir)


##############################################################################
# 
#  Evaluate detection metrics.
# 
##############################################################################

data_det = scrub_detection_status(csv_det_dict)
index_combinations = OrderedDict((
    ('inter-rater, manual',
         [(subdirectories.index('manual_A1'),
           subdirectories.index('manual_W1')),
          (subdirectories.index('manual_A1'),
           subdirectories.index('manual_W2')),
          (subdirectories.index('manual_A2'),
           subdirectories.index('manual_W1')),
          (subdirectories.index('manual_A2'),
           subdirectories.index('manual_W2'))]
    ),
    ('inter-rater, corrected',
        [(subdirectories.index('correction_A1'),
          subdirectories.index('correction_W1')),
         (subdirectories.index('correction_A1'),
          subdirectories.index('correction_W2')),
         (subdirectories.index('correction_A2'),
          subdirectories.index('correction_W1')),
         (subdirectories.index('correction_A2'),
          subdirectories.index('correction_W2'))]
    ),
    ('intra-rater, manual',
        [(subdirectories.index('manual_A1'),
          subdirectories.index('manual_A2')),
         (subdirectories.index('manual_W1'),
          subdirectories.index('manual_W2'))]
    ),
    ('intra-rater, corrected',
        [(subdirectories.index('correction_A1'),
          subdirectories.index('correction_A2')),
         (subdirectories.index('correction_W1'),
          subdirectories.index('correction_W2'))]
    ),
    ('inter-method, manual vs automatic',
         [(subdirectories.index('automatic'),
           subdirectories.index('manual_A1')),
          (subdirectories.index('automatic'),
           subdirectories.index('manual_A2')),
          (subdirectories.index('automatic'),
           subdirectories.index('manual_W1')),
          (subdirectories.index('automatic'),
           subdirectories.index('manual_W2'))]
    ),
    ('inter-method, corrected vs automatic',
         [(subdirectories.index('automatic'),
           subdirectories.index('correction_A1')),
          (subdirectories.index('automatic'),
           subdirectories.index('correction_A2')),
          (subdirectories.index('automatic'),
           subdirectories.index('correction_W1')),
          (subdirectories.index('automatic'),
           subdirectories.index('correction_W2'))]
    ),
    ('inter-method, manual vs corrected',
         [(subdirectories.index('manual_A1'),
           subdirectories.index('correction_A1')),
          (subdirectories.index('manual_A1'),
           subdirectories.index('correction_A2')),
          (subdirectories.index('manual_A1'),
           subdirectories.index('correction_W1')),
          (subdirectories.index('manual_A1'),
           subdirectories.index('correction_W2')),
          (subdirectories.index('manual_A2'),
           subdirectories.index('correction_A1')),
          (subdirectories.index('manual_A2'),
           subdirectories.index('correction_A2')),
          (subdirectories.index('manual_A2'),
           subdirectories.index('correction_W1')),
          (subdirectories.index('manual_A2'),
           subdirectories.index('correction_W2')),
          (subdirectories.index('manual_W1'),
           subdirectories.index('correction_A1')),
          (subdirectories.index('manual_W1'),
           subdirectories.index('correction_A2')),
          (subdirectories.index('manual_W1'),
           subdirectories.index('correction_W1')),
          (subdirectories.index('manual_W1'),
           subdirectories.index('correction_W2')),
          (subdirectories.index('manual_W2'),
           subdirectories.index('correction_A1')),
          (subdirectories.index('manual_W2'),
           subdirectories.index('correction_A2')),
          (subdirectories.index('manual_W2'),
           subdirectories.index('correction_W1')),
          (subdirectories.index('manual_W2'),
           subdirectories.index('correction_W2'))]
    )
    ))


for key, index_pairs in index_combinations.items():
    
    print("DETECTION: {}".format(key))
    
    # pooled kappa
    m = pooled_kappa(data_det, index_pairs=index_pairs)
    m_ci = bootstrap(data_det,
                    metric_function=partial(pooled_kappa,
                                            index_pairs=index_pairs),
                    alpha=args.alpha,
                    n_iterations=args.bootstrap_iterations,
                    n_proc=args.n_proc)
    print("Pooled kappa = {} ({}, {})".format(m, *m_ci))

    # quantity disagreement
    m = quantity_disagreement(data_det, index_pairs=index_pairs)
    m_ci = bootstrap(data_det,
                    metric_function=partial(quantity_disagreement,
                                            index_pairs=index_pairs),
                    alpha=args.alpha,
                    n_iterations=args.bootstrap_iterations,
                    n_proc=args.n_proc)
    print("Quantity disagreement = {} ({}, {})".format(m, *m_ci))

    # allocation disagreement
    m = allocation_disagreement(data_det, index_pairs=index_pairs)
    m_ci = bootstrap(data_det,
                    metric_function=partial(allocation_disagreement,
                                            index_pairs=index_pairs),
                    alpha=args.alpha,
                    n_iterations=args.bootstrap_iterations,
                    n_proc=args.n_proc)
    print("Allocation disagreement = {} ({}, {})".format(m, *m_ci))

    print("\n")
