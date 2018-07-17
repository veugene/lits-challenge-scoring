from multiprocessing import Pool
from functools import partial
import argparse
import os
from collections import OrderedDict

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Bootstrap statistics")
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
#  `data_sets` is a list of data sets, across which to average statistics.
# 
##############################################################################
def bootstrap(data_sets, metric_function, alpha,
              n_iterations=10000, n_proc=None, rng=None):
    if n_proc is None:
        n_proc = os.cpu_count()
        if n_proc is None:
            n_proc = 1
    if rng is None:
        rng = np.random.RandomState()
    
    # Sample metric.
    metric_mean_list, sample_metric_list = [], []
    for data in data_sets:
        metric_mean, sample_metric = _sample(data=data,
                                             metric_function=metric_function,
                                             n_iterations=n_iterations,
                                             n_proc=n_proc,
                                             rng=rng)
        metric_mean_list.append(metric_mean)
        sample_metric_list.append(sample_metric)
    metric_mean = np.mean(metric_mean_list)
    sample_metric = np.mean(sample_metric_list, axis=0)
    
    # Confindence interval
    delta = [np.mean(sample_metric[i])-metric_mean
             for i in range(n_iterations)]
    delta = sorted(delta)
    idx0 = int(len(delta)*alpha//2)
    idx1 = int(len(delta)-1-idx0)
    confidence_interval = (metric_mean-delta[idx1], metric_mean-delta[idx0])
    
    return confidence_interval


def _sample(data, metric_function, n_iterations, n_proc, rng):
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
    
    return metric_mean, sample_metric


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
    
    p11 = n11/float(n00+n01+n10+n11)
    p10 = n10/float(n00+n01+n10+n11)
    p01 = n01/float(n00+n01+n10+n11)
    p00 = n00/float(n00+n01+n10+n11)
    
    a_quant = abs(p01-p10)
    b_quant = abs(p10-p01)
    a_alloc = 2*min(p01, p10)
    b_alloc = 2*min(p10, p01)
    
    quantity_disagreement = (a_quant+b_quant)/2.
    allocation_disagreement = (a_alloc+b_alloc)/2.
    
    po = (n11+n00)/float(n11+n10+n01+n00)
    pe = ((n11+n10)*(n11+n01)+(n10+n01)*(n10+n00))/float(n11+n10+n01+n00)**2
    
    kappa = (po-pe)/(1-pe)
    
    values = {'po': po,
              'pe': pe,
              'kappa': kappa,
              'quantity_disagreement': quantity_disagreement,
              'allocation_disagreement': allocation_disagreement}
    
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
        alloc_list.append(v['allocation_disagreement'])
    allocation_disagreement = np.mean(alloc_list)
    return allocation_disagreement


def icc(data, indices_A, indices_B):
    # https://pdfs.semanticscholar.org/378e/
    # 3e71bc3f2c6cc5af891c91d2f8d65086f363.pdf
    
    assert len(indices_A)==len(indices_B)
    n_replicates = len(indices_A)
    n_observers = 2
    n_samples = len(data)
    
    # means
    mean   = np.mean(data[:,indices_A+indices_B])
    mean_j = np.array([np.mean(data[:,indices_A]), np.mean(data[:,indices_B])])
    mean_i = np.mean(data[:,indices_A+indices_B], axis=1, keepdims=True)
    indices = [indices_A, indices_B]
    
    # mean square alpha
    msa = 0
    for i in range(n_samples):
        msa += (np.mean(data[i,indices_A+indices_B])-mean)**2
    msa *= n_observers*n_replicates/float(n_samples-1)
    
    # mean square error
    mse = 0
    for i in range(n_samples):
        for j in range(n_observers):
            for k in range(n_replicates):
                mse += (data[i,indices[j][k]]-mean_i[i]-mean_j[j]+mean)**2
    mse /= float((n_observers*n_replicates-1)*n_samples-n_observers+1)
    
    # mean square beta
    msb = 0
    for j in range(n_observers):
        msb += (mean_j[j]-mean)**2
    msb *= n_samples*n_replicates/(n_observers-1)
    
    # icc2
    icc2 = (msa-mse) / (msa+(n_observers*n_replicates-1)*mse+\
                        n_observers*(msb-mse)/float(n_samples))

    return icc2[0]


def recall(data):  # also sensitivity
    fp = np.sum(data==None)
    fn = np.sum(data==0)
    tp = np.sum(data==1)
    return tp/float(tp+fn)


def precision(data):  # also positive predictive value
    fp = np.sum(data==None)
    fn = np.sum(data==0)
    tp = np.sum(data==1)
    return tp/float(tp+fp)


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
fp_file_str = "FP_by_size.txt"
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
    csv_fp_dict  = {}
    csv_det_dict = {}
    csv_seg_dict = {}
    for dn in subdirectories:
        rootdir = os.path.join(scores_dir, dn)
        csv_fp  = read_csv(find_first_file(rootdir, fp_file_str))
        csv_det = read_csv(find_first_file(rootdir, detection_file_str))
        csv_seg = read_csv(find_first_file(rootdir, segmentation_file_str))
        csv_fp_dict[dn]  = csv_fp
        csv_det_dict[dn] = csv_det
        csv_seg_dict[dn] = csv_seg
        
    return csv_fp_dict, csv_det_dict, csv_seg_dict


def _bad_diameter(diameter, min_diameter, max_diameter):
    if min_diameter is not None:
        if float(diameter) <  min_diameter:
            return True
    if max_diameter is not None:
        if float(diameter) >= max_diameter:
            return True
    return False


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


def scrub_detection_status_with_fp(csv_det_dict, csv_fp_dict, subdir,
                                   min_diameter=None, max_diameter=None):
    data = []
    idx = detection_entries.index('detection_status')
    d_idx = segmentation_entries.index('length_reference')
    for line in csv_det_dict[subdir]:
        if _bad_diameter(line[d_idx], min_diameter, max_diameter):
            continue
        data.append(int(line[idx]))
    for line in csv_fp_dict[dn]:
        if _bad_diameter(line[d_idx], min_diameter, max_diameter):
            continue
        data.append(None)     # false positive
    data = np.array(data)
    return data


def scrub_predicted_volumes(csv_det_dict):
    data = []
    idx = detection_entries.index('volume_prediction')
    for dn in subdirectories:
        column = []
        for line in csv_det_dict[dn]:
            column.append(float(line[idx]))
        data.append(column)
    data = np.array(data).T
    return data


def scrub_lesion_metric(csv_seg_dict, subdir, metric_name,
                        min_diameter=None, max_diameter=None):
    data = []
    idx = segmentation_entries.index(metric_name)
    d_idx = segmentation_entries.index('length_reference')
    for line in csv_seg_dict[subdir]:
        if _bad_diameter(line[d_idx], min_diameter, max_diameter):
            continue
        if line[idx]=="":
            continue
        data.append(float(line[idx]))
    data = np.array(data)
    return data


##############################################################################
# 
#  Setup.
# 
##############################################################################

args = parse_args()
csv_fp_dict, csv_det_dict, csv_seg_dict = load_data(args.scores_dir)
bootstrap_kwargs =  {'alpha': args.alpha,
                     'n_iterations': args.bootstrap_iterations,
                     'n_proc': args.n_proc}


##############################################################################
# 
#  Evaluate detection metrics (kappa, etc).
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
    m_ci = bootstrap([data_det],
                     metric_function=partial(pooled_kappa,
                                             index_pairs=index_pairs),
                     **bootstrap_kwargs)
    print("Pooled kappa = {:.3f} ({:.3f}, {:.3f})".format(m, *m_ci))

    # quantity disagreement
    m = quantity_disagreement(data_det, index_pairs=index_pairs)
    m_ci = bootstrap([data_det],
                     metric_function=partial(quantity_disagreement,
                                             index_pairs=index_pairs),
                     **bootstrap_kwargs)
    print("Quantity disagreement = {:.3f} ({:.3f}, {:.3f})".format(m, *m_ci))

    # allocation disagreement
    m = allocation_disagreement(data_det, index_pairs=index_pairs)
    m_ci = bootstrap([data_det],
                     metric_function=partial(allocation_disagreement,
                                             index_pairs=index_pairs),
                     **bootstrap_kwargs)
    print("Allocation disagreement = {:.3f} ({:.3f}, {:.3f})".format(m, *m_ci))

    print("\n")
    
    
##############################################################################
# 
#  Evaluate detection metrics (icc).
# 
##############################################################################

data_det = scrub_detection_status(csv_det_dict)
data_vol = scrub_predicted_volumes(csv_det_dict)
index_combinations = OrderedDict((
    ('inter-rater, manual',
         [(subdirectories.index('manual_A1'),
           subdirectories.index('manual_A2')),
          (subdirectories.index('manual_W1'),
           subdirectories.index('manual_W2'))]
    ),
    ('inter-rater, corrected',
        [(subdirectories.index('correction_A1'),
          subdirectories.index('correction_A2')),
         (subdirectories.index('correction_W1'),
          subdirectories.index('correction_W2'))]
    ),
    ('intra-rater, manual',
        [(subdirectories.index('manual_A1'),
          subdirectories.index('manual_W1')),
         (subdirectories.index('manual_A2'),
          subdirectories.index('manual_W2'))]
    ),
    ('intra-rater, corrected',
        [(subdirectories.index('correction_A1'),
          subdirectories.index('correction_W1')),
         (subdirectories.index('correction_A2'),
          subdirectories.index('correction_W2'))]
    )
    ))


for key, indices in index_combinations.items():
    indices_A, indices_B = indices
    
    print("DETECTION: {}".format(key))
    
    # ICC (detection)
    m = icc(data_det, indices_A=indices_A, indices_B=indices_B)
    m_ci = bootstrap([data_det],
                     metric_function=partial(icc,
                                             indices_A=indices_A,
                                             indices_B=indices_B),
                     **bootstrap_kwargs)
    print("ICC (detection) = {:.3f} ({:.3f}, {:.3f})".format(m, *m_ci))
    
    # ICC (volumes)
    m = icc(data_vol, indices_A=indices_A, indices_B=indices_B)
    m_ci = bootstrap([data_vol],
                     metric_function=partial(icc,
                                             indices_A=indices_A,
                                             indices_B=indices_B),
                     **bootstrap_kwargs)
    print("ICC (volume) = {:.3f} ({:.3f}, {:.3f})".format(m, *m_ci))
    
    print("\n")
    
    
##############################################################################
# 
#  Evaluate detection metrics (sensitivity, positive predictive value).
# 
##############################################################################

index_combinations = OrderedDict((
    ('manual 1',    (subdirectories.index('manual_A1'),
                     subdirectories.index('manual_A2'))),
    ('manual 2',    (subdirectories.index('manual_W1'),
                     subdirectories.index('manual_W2'))),
    ('corrected 1', (subdirectories.index('correction_A1'),
                     subdirectories.index('correction_A2'))),
    ('corrected 2', (subdirectories.index('correction_W1'),
                     subdirectories.index('correction_W2'))),
    ('automated',   (subdirectories.index('automatic'),))
    ))
    
    
diameter_ranges = [(None, None), (None, 10), (10, 20), (20, None)]
for key, indices in index_combinations.items():
    print("DETECTION: {}".format(key))
    
    for limits in diameter_ranges:
        data_det = []
        for dn_idx in indices:
            dn = subdirectories[dn_idx]
            data = scrub_detection_status_with_fp(csv_det_dict,
                                                  csv_fp_dict,
                                                  subdir=dn,
                                                  min_diameter=limits[0],
                                                  max_diameter=limits[1])
            if len(data):
                data_det.append(data)
        
        if len(data_det)==0:
            print("Warning: no data in size range [{}, {}) for {}."
                  "".format(*limits, key))
            continue
            
        m = np.mean([recall(data) for data in data_det])
        m_ci = bootstrap(data_det,
                         metric_function=recall,
                         **bootstrap_kwargs)
        print("Recall    [{}, {}] = {:.3f} ({:.3f}, {:.3f})"
              "".format(*limits, m, *m_ci))
        
        m = np.mean([precision(data) for data in data_det])
        m_ci = bootstrap(data_det,
                         metric_function=precision,
                         **bootstrap_kwargs)
        print("Precision [{}, {}] = {:.3f} ({:.3f}, {:.3f})"
              "".format(*limits, m, *m_ci))
    
    print("\n")
    
    
###############################################################################
## 
##  Evaluate segmentation metrics.
## 
###############################################################################

index_combinations = OrderedDict((
    ('manual 1',    (subdirectories.index('manual_A1'),
                     subdirectories.index('manual_A2'))),
    ('manual 2',    (subdirectories.index('manual_W1'),
                     subdirectories.index('manual_W2'))),
    ('corrected 1', (subdirectories.index('correction_A1'),
                     subdirectories.index('correction_A2'))),
    ('corrected 2', (subdirectories.index('correction_W1'),
                     subdirectories.index('correction_W2'))),
    ('automated',   (subdirectories.index('automatic'),))
    ))
    
    
diameter_ranges = [(None, None), (None, 10), (10, 20), (20, None)]
for metric_name in ['dice', 'assd', 'msd']:
    
    print("SEGMENTATION: {}".format(metric_name))
    
    for key, indices in index_combinations.items():
        for limits in diameter_ranges:
            data_seg = []
            for dn_idx in indices:
                dn = subdirectories[dn_idx]
                data = scrub_lesion_metric(csv_seg_dict,
                                           subdir=dn,
                                           metric_name=metric_name,
                                           min_diameter=limits[0],
                                           max_diameter=limits[1])
                if len(data):
                    data_seg.append(data)
            
            if len(data_seg)==0:
                print("Warning: no data in size range [{}, {})"
                      "".format(*limits))
                continue
                
            m = np.mean([np.mean(data) for data in data_seg])
            m_ci = bootstrap(data_seg,
                             metric_function=np.mean,
                             **bootstrap_kwargs)
            print("{} [{}, {}] = {:.3f} ({:.3f}, {:.3f})"
                  "".format(key, *limits, m, *m_ci))
        
    print("\n")
