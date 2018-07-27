import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from functools import partial
import argparse
import os
from collections import OrderedDict

import numpy as np
import scipy
import scipy.stats


def parse_args():
    parser = argparse.ArgumentParser(description="Plot bland altman")
    parser.add_argument("--scores_dir", type=str,
                        help="path to the scores directory")
    parser.add_argument("--x", type=float, default=10,
                        help="horizontal size of each figure")
    parser.add_argument("--y", type=float, default=5,
                        help="vertical size of each figure")
    parser.add_argument("--fontsize", type=int, default=14,
                        help="font size for labels and title")
    parser.add_argument("--save_to", type=str, default='.',
                        help="directory to save figures to")
    args = parser.parse_args()
    return args


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
segmentation_file_str = "segmentation_with_sizes.txt"
segmentation_entries = ['case_name', 'volume_reference',
                        'length_reference', 'volume_prediction',
                        'length_prediction', 'assd', 'rmsd', 'msd',
                        'dice', 'voe']


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
    csv_seg_dict = {}
    for dn in subdirectories:
        rootdir = os.path.join(scores_dir, dn)
        csv_seg = read_csv(find_first_file(rootdir, segmentation_file_str))
        csv_seg_dict[dn] = csv_seg
        
    return csv_seg_dict


def scrub_predicted_volumes(csv_seg_dict):
    data = []
    idx = segmentation_entries.index('volume_prediction')
    for dn in subdirectories:
        column = []
        for line in csv_seg_dict[dn]:
            if line[idx]=="":
                column.append(np.nan)
            else:
                column.append(float(line[idx]))
        data.append(column)
    data = np.array(data).T
    return data


##############################################################################
# 
#  DEFINE: Bland Altman computation and plotting.
# 
##############################################################################
def points_bland_altman(data, indices_A, indices_B):
    masked_data = np.ma.masked_invalid(data)
    count_A = np.count_nonzero(masked_data[:, indices_A],
                               axis=1).astype(np.float32)
    count_B = np.count_nonzero(masked_data[:, indices_B],
                               axis=1).astype(np.float32)
    
    # Drop all data lacking at least one replicate both in A and in B.
    at_least_one_of_each = ~(count_A*count_B).mask
    masked_data = masked_data[at_least_one_of_each]/1000.
    count_A = count_A[at_least_one_of_each]
    count_B = count_B[at_least_one_of_each]
    
    # Compute Bland Altman plot points.
    mean_A = np.mean(masked_data[:, indices_A], axis=1)
    mean_B = np.mean(masked_data[:, indices_B], axis=1)
    mean   = np.mean(masked_data[:, indices_A+indices_B], axis=1)
    diff   = (mean_A-mean_B)/mean
    
    # 95% limits of agreement.
    hmean_A = scipy.stats.hmean(count_A)
    hmean_B = scipy.stats.hmean(count_B)
    n_A = np.sum(count_A)
    n_B = np.sum(count_B)
    n   = len(count_B)
    s_d = np.std(diff, ddof=1 if len(diff)>1 else 0)
    _s_A2 = (np.std(masked_data[:, indices_A], axis=1, ddof=1)**2)/mean
    _s_B2 = (np.std(masked_data[:, indices_B], axis=1, ddof=1)**2)/mean
    _s_A2[count_A==1] = 0
    _s_B2[count_B==1] = 0
    s_A2 = np.sum(_s_A2*(count_A-1)/mean)/max(float(n_A-n), 1)
    s_B2 = np.sum(_s_B2*(count_B-1)/mean)/max(float(n_B-n), 1)
    std_corrected2 = s_d**2 + (1-1/hmean_A)*s_A2 + (1-1/hmean_B)*s_B2
    margin_loa = np.sqrt(std_corrected2)
    m_d = np.mean(diff)
    limits_of_agreement = (m_d-1.96*margin_loa, m_d+1.96*margin_loa)
    
    # 95% confidence of limits of agreement.
    margin_conf = ( (s_d**4)/float(n-1)
                   +((1-1/hmean_A)**2)*(s_A2**2)/max(float(n_A-n), 1)
                   +((1-1/hmean_B)**2)*(s_B2**2)/max(float(n_B-n), 1))
    conf_variance = (s_d**2)/float(n)+(1.96**2)/(2.*std_corrected2)*margin_conf
    limits_conf = 1.96*np.sqrt(conf_variance)
    
    # 95% confidence interval of the bias.
    bias_conf = scipy.stats.t.ppf(0.975, n-1)*np.sqrt(std_corrected2/float(n))
    
    # Output.
    output = {'mean': mean,
              'diff': diff,
              'limits': limits_of_agreement,
              'limits_conf': limits_conf,
              'bias': m_d,
              'bias_conf': bias_conf}
    return output


def linewidth_from_data_units(linewidth, axis, reference='y'):
    """
    Convert a linewidth in data units to linewidth in points.
    
    https://stackoverflow.com/questions/19394505/
            matplotlib-expand-the-line-with-specified-width-in-data-unit

    Parameters
    ----------
    linewidth: float
        Linewidth in data units of the respective reference-axis
    axis: matplotlib axis
        The axis which is used to extract the relevant transformation
        data (data limits and size must not change afterwards)
    reference: string
        The axis that is taken as a reference for the data width.
        Possible values: 'x' and 'y'. Defaults to 'y'.

    Returns
    -------
    linewidth: float
        Linewidth in points
    """
    fig = axis.get_figure()
    if reference == 'x':
        length = fig.bbox_inches.width * axis.get_position().width
        value_range = np.diff(axis.get_xlim())
    elif reference == 'y':
        length = fig.bbox_inches.height * axis.get_position().height
        value_range = np.diff(axis.get_ylim())
    # Convert length to points
    length *= 72
    # Scale linewidth to value range
    return linewidth * (length / value_range)


def plot_bland_altman(mean, diff, limits, limits_conf, bias, bias_conf,
                      xlim, ylim, ylabel, title, figsize, fontsize, save_to):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    ax.set_facecolor('0.9')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axhline(bias, color='blue', alpha=0.2,
               linewidth=linewidth_from_data_units(bias_conf, ax))
    ax.axhline(limits[0], color='red', alpha=0.2,
               linewidth=linewidth_from_data_units(limits_conf, ax))
    ax.axhline(limits[1], color='red', alpha=0.2,
               linewidth=linewidth_from_data_units(limits_conf, ax))
    ax.axhline(bias, color='blue')
    ax.annotate("Mean\n{:.2f}".format(bias),
                xy=(0.99, 0.5+bias/float(ylim[1]-ylim[0])),
                horizontalalignment='right',
                verticalalignment='center',
                fontsize=fontsize,
                xycoords='axes fraction')
    ax.axhline(limits[0], color='red', linestyle='dashed')
    ax.annotate("-1.96 SD\n{:.2f}".format(limits[0]),
                xy=(0.99, 0.5+limits[0]/float(ylim[1]-ylim[0])),
                horizontalalignment='right',
                verticalalignment='center',
                fontsize=fontsize,
                xycoords='axes fraction')
    ax.axhline(limits[1], color='red', linestyle='dashed')
    ax.annotate("+1.96 SD\n{:.2f}".format(limits[1]),
                xy=(0.99, 0.5+limits[1]/float(ylim[1]-ylim[0])),
                horizontalalignment='right',
                verticalalignment='center',
                fontsize=fontsize,
                xycoords='axes fraction')
    plt.semilogx(mean, diff, color='black', marker='o', linestyle='None',
                 markerfacecolor='none', figure=fig)
    plt.ylabel(ylabel,
               fontsize=fontsize*1.4,
               multialignment='center')
    plt.xlabel("$\mathtt{Mean\ volume\ (mL)}$", fontsize=fontsize)
    for item in ax.get_xticklabels()+ax.get_yticklabels():
        item.set_fontsize(fontsize)
    plt.title(title, fontsize=fontsize)
    plt.show(fig)
    plt.savefig(save_to)


##############################################################################
# 
#  Setup.
# 
##############################################################################
args = parse_args()
csv_seg_dict = load_data(args.scores_dir)
data_vol = scrub_predicted_volumes(csv_seg_dict)
if not os.path.exists(args.save_to):
    os.makedirs(args.save_to)


##############################################################################
# 
#  Evaluate Bland Altman.
# 
##############################################################################
y_axis_labels = OrderedDict((
    ('Inter-reader, manual (both replicates)',
     r'$\frac{\mathtt{reader\ 1\ -\ reader\ 2}}'
      '{\mathtt{mean(reader\ 1\ -\ reader\ 2)}}$'),
    ('Inter-reader, corrected (both replicates)',
     r'$\frac{\mathtt{reader\ 1\ -\ reader\ 2}}'
      '{\mathtt{mean(reader\ 1\ -\ reader\ 2)}}$'),
    ('Intra-reader, manual (both readers)',
     r'$\frac{\mathtt{replicate\ 1\ -\ replicate\ 2}}'
      '{\mathtt{mean(replicate\ 1\ -\ replicate\ 2)}}$'),
    ('Intra-reader, corrected (both readers)',
     r'$\frac{\mathtt{replicate\ 1\ -\ replicate\ 2}}'
      '{\mathtt{mean(replicate\ 1\ -\ replicate\ 2)}}$'),
    ('Inter-method, manual vs automated',
     r'$\frac{\mathtt{all\ manual -\ all\ automated}}'
      '{\mathtt{mean(all\ manual -\ all\ automated)}}$'),
    ('Inter-method, corrected vs automated',
     r'$\frac{\mathtt{all\ corrected -\ all\ automated}}'
      '{\mathtt{mean(all\ corrected -\ all\ automated)}}$'),
    ('Inter-method, manual vs corrected',
     r'$\frac{\mathtt{all\ manual -\ all\ corrected}}'
      '{\mathtt{mean(all\ manual -\ all\ corrected)}}$'),
    ))
index_combinations = OrderedDict((
    ('Inter-reader, manual (both replicates)',
         [(subdirectories.index('manual_A1'),
           subdirectories.index('manual_A2')),
          (subdirectories.index('manual_W1'),
           subdirectories.index('manual_W2'))]
    ),
    ('Inter-reader, corrected (both replicates)',
        [(subdirectories.index('correction_A1'),
          subdirectories.index('correction_A2')),
         (subdirectories.index('correction_W1'),
          subdirectories.index('correction_W2'))]
    ),
    ('Intra-reader, manual (both readers)',
        [(subdirectories.index('manual_A1'),
          subdirectories.index('manual_W1')),
         (subdirectories.index('manual_A2'),
          subdirectories.index('manual_W2'))]
    ),
    ('Intra-reader, corrected (both readers)',
        [(subdirectories.index('correction_A1'),
          subdirectories.index('correction_W1')),
         (subdirectories.index('correction_A2'),
          subdirectories.index('correction_W2'))]
    ),
    ('Inter-method, manual vs automated',
         [(subdirectories.index('manual_A1'),
           subdirectories.index('manual_A2'),
           subdirectories.index('manual_W1'),
           subdirectories.index('manual_W2')),
          (subdirectories.index('automatic'),)]
    ),
    ('Inter-method, corrected vs automated',
         [(subdirectories.index('correction_A1'),
           subdirectories.index('correction_A2'),
           subdirectories.index('correction_W1'),
           subdirectories.index('correction_W2')),
          (subdirectories.index('automatic'),)]
    ),
    ('Inter-method, manual vs corrected',
         [(subdirectories.index('manual_A1'),
           subdirectories.index('manual_A2'),
           subdirectories.index('manual_W1'),
           subdirectories.index('manual_W2')),
          (subdirectories.index('correction_A1'),
           subdirectories.index('correction_A2'),
           subdirectories.index('correction_W1'),
           subdirectories.index('correction_W2'))]
    )
    ))


for i, (key, indices) in enumerate(index_combinations.items()):
    print("\n{}".format(key))
    indices_A, indices_B = indices
    out = points_bland_altman(data_vol, indices_A, indices_B)
    print("limits: {}".format(out['limits']))
    print("limits_conf: {}".format(out['limits_conf']))
    print("bias: {}".format(out['bias']))
    print("bias_conf: {}".format(out['bias_conf']))
    plot_bland_altman(**out,
                      xlim=[0.01, 1000],
                      ylim=[-2, 2],
                      ylabel=y_axis_labels[key],
                      figsize=(args.x, args.y),
                      title=key,
                      fontsize=args.fontsize,
                      save_to=os.path.join(args.save_to,
                                           "fig{}.png".format(i)))
