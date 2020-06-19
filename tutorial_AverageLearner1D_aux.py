# -*- coding: utf-8 -*-

import random
from collections import defaultdict
import adaptive

import numpy as np
from time import sleep
from random import random
from adaptive.learner import AverageLearner
import math
from functools import partial
from scipy.interpolate import interp1d
from scipy.integrate import quad

from matplotlib.ticker import ScalarFormatter, NullFormatter, LogFormatter

# For animations
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
import copy
from IPython.display import HTML
import warnings

#____________________________________________________________________
#______________________RUN AND PLOT LEARNER__________________________
#____________________________________________________________________
def plot_learner(learner, equalaxes=False, ylim=None, alphafun=0.3, alphaline=1, alphabars=0.3, Nfun=200):
    '''Plots the learner data and the underlying function. If the learner is
       an AverageLearner1D, plots data with error bars.
       ---Inputs---
            learner: learner (learner1D or averagelearner1D)
            equalaxes: optional, if True, sets axes aspect ratio to 1 (bool)
            ylim: optional, vertical limits of the plot (tuple)
            alphafun: transparency of the function (0<float<1)
            alphaline: transparency of the learner line (0<float<1)
            alphabars: transparency of the error bars (0<float<1)
            Nfun: number of points in which the function is evaluated and plotted (int)'''
    xfun = np.linspace(learner.bounds[0],learner.bounds[1],Nfun)
    yfun = []
    for xi in xfun:
        yfun.append(learner.function(xi))

    x, y = zip(*sorted(learner.data.items()))
    try: # AverageLearner1D
        yfun0 = []
        for xi in xfun:
            yfun0.append(learner.function(xi,sigma=0))

        plt.plot(xfun,yfun0,color='k', linewidth=1)
        plt.autoscale(False)
        plt.plot(xfun,yfun,alpha=alphafun,color='tab:orange')

        plt.plot(x, y, color='tab:blue', linewidth=2, alpha=alphaline)
        _, err = zip(*sorted(learner._error_in_mean.items()))
        plt.errorbar(x, y, yerr=err, linewidth=0, marker='o', color='k',
                     markersize=2, elinewidth=1, capsize=3, capthick=1, alpha=alphabars)
        plt.title('N=%d'%learner.total_samples)
    except: # Learner1D
        plt.plot(xfun,yfun,linewidth=5,alpha=alphafun,color='tab:orange')

        plt.plot(x, y, linewidth=1, color='tab:blue', marker='o', markersize=2,
                 markeredgecolor='k', markerfacecolor='k')
        plt.title('N=%d'%len(learner.data))
    plt.xlim(learner.bounds)
    if equalaxes:
        plt.gca().set_aspect('equal', adjustable='box')
    if ylim:
        plt.ylim(ylim)

def run_N(learner, N):
    '''Runs the learner until it has N samples'''
    from tqdm.notebook import tqdm
    try: # AverageLearner1D
        N0 = learner.total_samples
        if N-N0>0:
            for _ in tqdm(np.arange(N-N0)):
                    xs, _ = learner.ask(1)
                    for x in xs:
                        y = learner.function(x)
                        learner.tell(x, y)
    except: # Learner1D
        N0 = len(learner.data)
        if N-N0>0:
            for _ in tqdm(np.arange(N-N0)):
                    xs, _ = learner.ask(1)
                    for x in xs:
                        y = learner.function(x)
                        learner.tell(x, y)

def simple_liveplot(learner, goal = lambda l: l.total_samples==500, N_frame = 100, alphafun=0.3, alphaline=1, alphabars=0.3, N_fun=200):
    '''Plots the learner data and the underlying function in real time.
       If the learner is an AverageLearner1D, plots data with error bars.
       ---Inputs---
            learner: learner (learner1D or averagelearner1D)
            goal: end condition for the calculation. This function must take
                  the learner as its sole argument, and return True when we should
                  stop requesting more points (callable)
            N_frame: number of samples per frame (int)
            alphafun: transparency of the function (0<float<1)
            alphaline: transparency of the learner line (0<float<1)
            alphabars: transparency of the error bars (0<float<1)
            Nfun: number of points in which the function is evaluated and plotted (int)'''
    import pylab as pl
    from IPython import display
    xfun = np.linspace(learner.bounds[0],learner.bounds[1],N_fun)
    try:
        yfun0 = learner.function(xfun, sigma=0)
    except:
        yfun0 = []
        for xi in xfun:
            yfun0.append(learner.function(xi,sigma=0))

    yfun = []
    for xi in xfun:
        yfun.append(learner.function(xi))
    try:
        while not goal(learner):
            for i in np.arange(N_frame):
                xs, _ = learner.ask(1)
                for x in xs:
                    y = learner.function(x)
                    learner.tell(x, y)
            x, y = zip(*sorted(learner.data.items()))
            plt.cla()
            try: # AverageLearner1D
                plt.xlim(learner.bounds[0],learner.bounds[1])
                plt.plot(xfun, yfun0, color='k', linewidth=1)
                plt.plot(x, y, linewidth=2, alpha=alphaline)
                plt.autoscale(False)

                _, err = zip(*sorted(learner._error_in_mean.items()))
                plt.errorbar(x, y, yerr=err, linewidth=0, marker='o', color='k',
                             markersize=2, elinewidth=1, capsize=3, capthick=1, alpha=alphabars)
                plt.title('N=%d, n=%d'%(learner.total_samples,len(learner.data)))
                plt.plot(xfun, yfun, alpha=alphafun, color='tab:orange')
            except: # Learner1D
                plt.xlim(learner.bounds[0],learner.bounds[1])
                plt.plot(xfun, yfun, alpha=alphafun ,color='tab:orange')
                plt.plot(x, y, linewidth=1, color='tab:blue', marker='o', markersize=2,
                         markeredgecolor='k', markerfacecolor='k')
                plt.title('N=%d'%len(learner.data))
            display.clear_output(wait=True)
            display.display(plt.gcf())
    except KeyboardInterrupt:
        plt.cla()
        display.clear_output(wait=True)
        display.display(pl.gcf())
        plot_learner_(learner, Nfun=N_fun)
    display.clear_output(wait=True)

#____________________________________________________________________
#____________________________FUNCTIONS_______________________________
#____________________________________________________________________
def plot_fun(function,xlim,N=200,title=None,ylim=None,**function_kwargs):
    '''Plots a symbolic function within a specific interval.
       ---Inputs---
            function: function to plot (callable)
            xlim: bounds of the interval in which the function can be
                      evaluated (tuple)
            N: number of points (int)
            title: optional, title for the plot (string)
            ylim: optional, vertical limits of the plot (tuple)'''
    import matplotlib.pyplot as plt
    x = np.linspace(xlim[0],xlim[1],N)
    y = []
    for xi in x:
        y.append(function(xi,**function_kwargs))
    plt.plot(x,y)
    plt.xlim(xlim[0],xlim[1])
    if ylim:
        plt.ylim(ylim[0],ylim[1])
    plt.xlabel('x')
    plt.ylabel('f(x)')
    if title:
        plt.title(title)
    return
