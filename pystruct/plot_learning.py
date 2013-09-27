#!/usr/bin/python
"""
This module provides a callable for easy evaluation of stored models.
"""
import argparse
from os.path import commonprefix
from numbers import Number
import matplotlib.pyplot as plt
import numpy as np

from pystruct.utils import SaveLogger

def halton(index, base):
       result = 0
       f = 1. / base
       i = index
       while(i > 0):
           result = result + f * (i % base)
           i = np.floor(i / base)
           f = f / base
       return result


def get_color(offset=0):
    i = 0
    while True:
        c1 = halton(i + offset, 2)
        c2 = halton(i + offset, 3)
        c3 = halton(i + offset, 5)
        i += 1
        yield [c1, c2, c3]

colors = ['b', 'r', 'g', 'c', 'm', 'DarkOrange', 'y', 'k']


def main():

    parser = argparse.ArgumentParser(description='Plot learning progress for one or several SSVMs.')
    parser.add_argument('pickles', metavar='N', type=str, nargs='+',
                        help='pickle files containing SSVMs')
    parser.add_argument('--time', dest='time', action='store_const',
                        const=True, default=False, help='Plot against '
                       'wall-clock time (default: plot against iterations.)')
    parser.add_argument('--dual', dest='dual', action='store_const',
                        const=True, default=False, help='Plot primal and dual '
                       'values (default: plot primal suboptimality.)')
    parser.add_argument('--loss', dest='loss', action='store_const',
                        const=True, default=False, help='Plot loss '
                       'values (default: False.)')
    parser.add_argument('--absolute-loss', dest='absolute_loss', action='store_const',
                        const=True, default=False, help='Plot full loss value '
                       ' (default: plot difference to best loss.)')
    parser.add_argument('--save', type=str, help='save plot to given file ', default=None)
    args = parser.parse_args()
    ssvms = []
    for file_name in args.pickles:
        print("loading %s ..." % file_name)
        ssvms.append(SaveLogger(file_name=file_name).load())
    if args.loss and np.any([hasattr(ssvm.logger, 'loss_') for ssvm in ssvms]):
        n_plots = 2
        figures = [plt.figure(figsize=(5, 3)), plt.figure(figsize=(5, 3))]
        axes = [f.gca() for f in figures]
    else:
        n_plots = 1
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 3))
    #fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 3))

    # find best dual value among all objectives
    if not args.dual:
        best_dual = -np.inf
        for ssvm in ssvms:
            if hasattr(ssvm, 'dual_objective_curve_'):
                best_dual = max(best_dual, np.max(ssvm.dual_objective_curve_))

    #if not args.absolute_loss and n_plots ==2:
        #best_loss = np.inf
        #for ssvm in ssvms:
            #best_loss = min(best_loss, np.min(ssvm.logger.loss_))
    #else:
    best_loss = 0

    if args.dual or not np.isfinite(best_dual):
        best_dual = None

    #for i, (ssvm, file_name, color) in enumerate(zip(ssvms, args.pickles, get_color(10))):
    for i, (ssvm, file_name, color) in enumerate(zip(ssvms, args.pickles, colors)):
        prefix = ""
        if len(ssvms) > 1:
            common_prefix_length = len(commonprefix(args.pickles))
            #common_prefix_length = 0
            prefix = file_name[common_prefix_length:-7] + " "
        plot_learning(ssvm, axes=axes, prefix=prefix, time=args.time,
                      color=color, suboptimality=best_dual,
                      loss_bound=best_loss, loss=args.loss)
    if args.save is not None:
        if n_plots == 1:
            plt.savefig(args.save + ".pdf", bbox_inches='tight')
        else:
            figures[0].savefig(args.save + ".pdf", bbox_inches='tight')
            figures[1].savefig(args.save + "_loss.pdf", bbox_inches='tight')
    plt.show()


def plot_learning(ssvm, time=True, axes=None, prefix="", color=None,
    show_caching=False, suboptimality=None, loss_bound=0, loss=False):
    """Plot optimization curves and cache hits.

    Create a plot summarizing the optimization / learning process of an SSVM.
    It plots the primal and cutting plane objective (if applicable) and also
    the target loss on the training set against training time.
    For one-slack SSVMs with constraint caching, cached constraints are also
    contrasted against inference runs.

    Parameters
    -----------
    ssvm : object
        Learner to evaluate. Should work with all learners.

    time : boolean, default=True
        Whether to use wall clock time instead of iterations as the x-axis.

    prefix : string, default=""
        Prefix for legend.

    color : matplotlib color.
        Color for the plots.

    show_caching : bool, default=False
        Whether to include iterations using cached inference in 1-slack ssvm.

    suboptimality : float or None, default=None
        If a float is given, only plot primal suboptimality with respect to
        this optimum.

    loss_bound : float, default=0
        Lower bound for the loss for plotting in log-domain

    Notes
    -----
    Warm-starting a model might mess up the alignment of the curves.
    So if you warm-started a model, please don't count on proper alignment
    of time, cache hits and objective.
    """
    print(ssvm)
    if hasattr(ssvm, 'base_ssvm'):
        ssvm = ssvm.base_ssvm
    logger = ssvm.logger
    primal_objective = np.array(logger.primal_objective_)
    if suboptimality is not None:
        primal_objective -= suboptimality
    if len(logger.loss_) and loss:
        n_plots = 2
    else:
        n_plots = 1
    if axes is None:
        fig, axes = plt.subplots(1, n_plots)
    #if not isinstance(axes, np.ndarray):
    if not isinstance(axes, list):
        axes = [axes]

    if time:
        inds = np.array(logger.timestamps_) / 60.
        axes[0].set_xlabel('Training time (min)')
    else:
        axes[0].set_xlabel('Passes through training data')
        inds = np.arange(len(logger.timestamps_)) * logger.log_every + 1 # +1 for log plots
        #inds = np.array(logger.iterations_) + 1   # +1 for log plots
    inds = inds[:len(primal_objective)]  # i have no idea why we need this
    if suboptimality is None and len(logger.dual_objective_):
        primal_prefix = "primal objective"
        axes[0].plot(inds, logger.dual_objective_, '--', label=prefix + "dual objective", color=color, linewidth=2)
    else:
        primal_prefix = ""
    axes[0].plot(inds, primal_objective, label=prefix + primal_prefix, color=color,
                 linewidth=2)
    #axes[0].legend(loc='best')
    axes[0].set_yscale('log')
    #axes[0].set_ylim(10 ** -1,  10 ** 3)
    #if time:
        #axes[0].set_xlim(10 ** -2,  10 ** 3)
    #else:
        #axes[0].set_xlim(0,  10 ** 3)

    axes[0].set_xscale('log')
    if n_plots == 2:
        if primal_prefix:
            axes[0].set_title("Objective")
        else:
            axes[0].set_title("Primal Suboptimality")
        if time:
            axes[1].set_xlabel('Training time (min)')
        else:
            axes[1].set_xlabel('Passes through training data')
        if not isinstance(logger.loss_[0], Number):
            # backward compatibility fix me!
            loss = [np.sum(l) for l in logger.loss_]
        else:
            loss = logger.loss_
        #if prefix == "SZLJSP ":
            #print("GNAH")
            #inds[1] += 1
            #loss[0] += 100

        loss = np.maximum(.1, np.array(loss) - loss_bound)
        axes[1].plot(inds, loss, color=color, linewidth=2)
        axes[1].set_title("Training Error")
        axes[1].set_ylim(10 ** -1,  10 ** 4)
        #if time:
            #axes[1].set_xlim(10 ** -2,  10 ** 3)
        #else:
            #axes[1].set_xlim(0,  10 ** 3)
        axes[1].set_yscale('log')
        axes[1].set_xscale('log')
    return axes


if __name__ == "__main__":
    main()
