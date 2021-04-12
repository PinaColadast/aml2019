"""plot some plots"""

import sys
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import dask.dataframe as dd
from pandas.core.index import CategoricalIndex, RangeIndex, Index, MultiIndex

def plot_confusion_matrix(cm, classes, title_real =(), normalize=False, cmap=plt.cm.Blues, png_output=None, show=True, 
                          fmt = (), ft = ()):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title='Normalized confusion matrix'
    else:
        title='Confusion matrix'
     # Calculate chart area size
    leftmargin = 0.5 # inches
    rightmargin = 0.5 # inches
    categorysize = 0.45 # inches
    figwidth = leftmargin + rightmargin + (len(classes) * categorysize)

    f = plt.figure(figsize=(figwidth, figwidth))

    # Create an axes instance and ajust the subplot size
    ax = f.add_subplot(111)
    ax.set_aspect(1)
    f.subplots_adjust(left=leftmargin/figwidth, right=1-rightmargin/figwidth, top=0.94, bottom=0.1)

    res = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title_real, fontsize = ft)
    plt.colorbar(res)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=75, ha='right')
    ax.set_yticklabels(classes)
    
    fmt = fmt
#     fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if png_output is not None:
        os.makedirs(png_output, exist_ok=True)
        f.savefig(os.path.join(png_output,'confusion_matrix.png'), bbox_inches='tight', dpi = 600)

    if show:
        plt.show()
        plt.close(f)
    else:
        plt.close(f)