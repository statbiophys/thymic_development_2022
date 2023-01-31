#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    *** Pro Plot ***
    Copyright (C) February 2023 Francesco Camaglia, CNRS LPENS 

    Reference :
    Francesco Camaglia, Arie Ryvkin, Erez Greenstein, Shlomit Reich-Zeliger, Benny Chain, Thierry Mora, Aleksandra M. Walczak, Nir Friedman (2023).
    Quantifying changes in the T cell receptor repertoire during thymic development. [https://doi.org/10.7554/eLife.81622]
'''

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.colors import is_color_like, Normalize
from itertools import product

_greys_dic_ = {'signal': '#282828', 'pidgeon': '#606e8c', 'silver': '#c0c0c0' }

import seaborn as sns
from matplotlib import cm
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering
from scipy.spatial.distance import squareform


################
#  DENDROGRAM  #
################


def Dendrogram(df, method="ward", cmap=cm.magma, figsize=(8,8), fontsize=30) :
    '''
    It reorders a distance matrix `df` according to HAC and plots it with its respective argument.
    The HAC `method` can be specified.
    Adapatation from Giulio Isacchini : https://github.com/statbiophys/soNNia/blob/6d99a55cb8c6b71f0ef110f1eefccbd71f789d8d/sonnia/compare_repertoires.py
    '''
    
    condensed_dist_matrix = squareform( df )
    
    Z = linkage( condensed_dist_matrix, method=method ) 
    res_linkage = optimal_leaf_ordering( Z, condensed_dist_matrix )
    
    g = sns.clustermap(df, row_linkage=res_linkage, col_linkage=res_linkage, 
                       figsize=figsize, cmap=cmap )
    plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=fontsize)
    for a in g.ax_row_dendrogram.collections:
        a.set_linewidth(3)
    for a in g.ax_col_dendrogram.collections:
        a.set_linewidth(3)
    g.ax_row_dendrogram.set_visible(False)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=fontsize)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=fontsize)
    g.cax.yaxis.set_ticks_position("left")
    g.cax.yaxis.set_label_position('left')
    dendro_box = g.ax_row_dendrogram.get_position()
    dendro_box.x0 = (dendro_box.x0 + 2 * dendro_box.x1) / 3 -0.01
    dendro_box.x1 = dendro_box.x1-0.01
    g.cax.set_position(dendro_box)

#################
#  BAR PLOTTER  #
#################

def barPlotter( dataframe, colors,
               columns=None, figsize=None, errorbars=None, rotation=90,
               hatch=None, grid=False, legend=False ) :
    '''
    '''
    
    # >>>>>>>>>>>>>>>>>>>
    #  OPTIONS LOADING  #
    # >>>>>>>>>>>>>>>>>>>
    
    #  dataframe
    df = dataframe.copy()
    
    #  columns
    if columns is not None :
        if np.any( [c not in df.columns for c in columns] ) :
            raise KeyError( "Some requested `columns` are not in `dataframe`." )
    else : # default
        columns = df.columns.values
        
    #  colors 
    # NOTE: a nice way to make this variable optional?
    if not np.all( [is_color_like(c) for c in colors] ) :
        raise IOError( "Some provided `colors` are not recognized." )
    if len(colors) == 1 :
        colors = list(colors) * len(columns)
    elif len(colors) < len(columns) :
        colors = list(colors) * int( 1+np.ceil( len(columns)/len(colors)-1 ) )
        colors = colors[:len(columns)]
                
    #  figsize
    if figsize is not None :
        if type(figsize) != tuple or len(figsize) != 2 :
            raise IOError( "Wrong choice for `figsize` format." )
    else : # default   
        figsize = ( int(0.75*len(df.index)), 4 )
        
    #  errorbars
    if errorbars is not None :
        if np.any( [c not in df.columns for c in errorbars] ) :
            raise KeyError( "Some requested `errorbars` are not in `dataframe`." )
            
    # rotation
    rotation = int(rotation)
    
    # hatch
    if hatch is not None :
        if len(hatch) == 1 :
            hatch = list(hatch) * len(columns)
        elif len(hatch) < len(columns) :
            hatch = list(hatch) * int( 1+np.ceil( len(columns)/len(hatch)-1 ) )
            hatch = hatch[:len(columns)]
    else :
        hatch = [''] * len(columns)
            
    # grid
    grid = bool(grid)
    
    # legend
    legend = bool(legend)
        
        
    # >>>>>>>>>>>>
    #  PLOTTING  #
    # >>>>>>>>>>>>    
    
    fig, ax = plt.subplots( nrows=1, ncols=1, figsize=figsize,
                           subplot_kw={'adjustable':'box'} )

    x = np.arange(0.5, len(df.index)+0.5)
    breaks = np.linspace(0, 1, len(columns)+2) - 0.5
    width = breaks[1] - breaks[0]

    ylim_min, ylim_max = [], []
    for i, col in enumerate(columns) :
        ax.bar( x + breaks[i+1], df[col], hatch=hatch[i], edgecolor='white', lw=0.5,
                width=width, color=colors[i], label=col, zorder=1 )  
        if errorbars is not None :
            err_col = errorbars[i]
            ax.errorbar( x + breaks[i+1], df[col], yerr=df[err_col], 
                        ls="", color="black", fmt='', label="")
            ylim_min.append( np.min( df[col] - df[err_col] ) )
            ylim_max.append( np.max( df[col] + df[err_col] ) )
        else :
            ylim_min.append( np.min( df[col] ) )
            ylim_max.append( np.max( df[col] ) )
   
    # lim
    ax.set_xlim( [x[0]-0.75, x[-1]+0.75] )
    ax.set_ylim( [ 0.99 * np.min( ylim_min ), 1.01* np.max( ylim_max ) ] )
    
    # ticks
    ax.set_xticks( x )
    ax.set_xticklabels( df.index, rotation=rotation )

    # grid
    if grid is True :
        ax.yaxis.grid(which="both", color=_greys_dic_["silver"], ls='--', zorder=-10)
        ax.set_axisbelow(True)

    # legend
    
    if legend is True :
        plt.legend(loc="center left", ncol=1, 
                   shadow=False, edgecolor='inherit', framealpha=1, bbox_to_anchor=(1,.5))

    return ax
###



###################
#  CURVE PLOTTER  #
###################

def curvePlotter( dataframe, colors, custom_Axe=None,
               columns=None, figsize=None, errorbars=None, markerstyle="o",
               grid=False, legend=False, linestyle="-", fill_error=False, zorder=None, lw=1 ) :
    '''
    '''
    
    # >>>>>>>>>>>>>>>>>>>
    #  OPTIONS LOADING  #
    # >>>>>>>>>>>>>>>>>>>
    
    #  dataframe
    df = dataframe.copy()
    
    #  columns
    if columns is not None :
        if np.any( [c not in df.columns for c in columns] ) :
            raise KeyError( "Some requested `columns` are not in `dataframe`." )
    else : # default
        columns = df.columns.values
        
    #  colors 
    # NOTE: a nice way to make this variable optional?
    if not np.all( [is_color_like(c) for c in colors] ) :
        raise IOError( "Some provided `colors` are not recognized." )
    if len(colors) == 1 :
        colors = list(colors) * len(columns)
    elif len(colors) < len(columns) :
        colors = list(colors) * int( 1+np.ceil( len(columns)/len(colors)-1 ) )
        colors = colors[:len(columns)]
                
    #  figsize
    if figsize is not None :
        if type(figsize) != tuple or len(figsize) != 2 :
            raise IOError( "Wrong choice for `figsize` format." )
    else : # default   
        figsize = ( 6, 4 )

    #  markerstyle 
    if len(markerstyle) == 1 :
        markerstyle = list(markerstyle) * len(columns)
    elif len(markerstyle) < len(columns) :
        markerstyle = list(markerstyle) * int( 1+np.ceil( len(columns)/len(markerstyle)-1 ) )
        markerstyle = markerstyle[:len(columns)]

    #  linestyle 
    if len(linestyle) == 1 :
        linestyle = list(linestyle) * len(columns)
    elif len(linestyle) < len(columns) :
        linestyle = list(linestyle) * int( 1+np.ceil( len(columns)/len(linestyle)-1 ) )
        linestyle = linestyle[:len(columns)]
        
    #  errorbars
    if errorbars is not None :
        if np.any( [c not in df.columns for c in errorbars] ) :
            raise KeyError( "Some requested `errorbars` are not in `dataframe`." )

    # zorder
    if zorder is None :
        zorder = [1] * len(columns)      
    elif len(zorder) == 1 :
        zorder = list(zorder) * len(columns)
    elif len(zorder) < len(columns) :
        zorder = list(zorder) * int( 1+np.ceil( len(columns)/len(zorder)-1 ) )
        zorder = zorder[:len(columns)]

    # grid
    grid = bool(grid)
    
    # legend
    legend = bool(legend)

    # >>>>>>>>>>>>
    #  PLOTTING  #
    # >>>>>>>>>>>>    
    if custom_Axe is None :
        fig, ax = plt.subplots( nrows=1, ncols=1, figsize=figsize,
                            subplot_kw={'adjustable':'box'} )
    else :
        ax = custom_Axe

    x = df.index.values

    ylim_min, ylim_max = [], []
    for i, col in enumerate(columns) :
        y = df[col]
        ax.plot( x, y, linestyle=linestyle[i], lw=lw, marker=markerstyle[i],
                color=colors[i], label=col, zorder=zorder[i] )  

        if errorbars is not None :
            err_col = errorbars[i]
            yerr = df[err_col]
            if fill_error is True :
                ax.fill_between( x, y-yerr, y+yerr, color=colors[i], alpha=0.5, edgecolor=None, zorder=zorder[i], )
            
            ax.errorbar( x, y, yerr=yerr, label=col, zorder=zorder[i], 
                        ls=linestyle[i], lw=lw, color=colors[i], fmt='' )
         
            ylim_min.append( np.min( y - yerr ) )
            ylim_max.append( np.max( y + yerr ) )

        else :
            ylim_min.append( np.min( y ) )
            ylim_max.append( np.max( y ) )
    # lim
    ax.set_ylim( [ 0.98 * np.min( ylim_min ), 1.02* np.max( ylim_max ) ] )
    
    # grid
    if grid is True :
        ax.yaxis.grid(which="both", color=_greys_dic_["silver"], ls='--', zorder=-10)
        ax.set_axisbelow(True)

    # legend
    
    if legend is True :
        plt.legend(loc="center left", ncol=1, 
                   shadow=False, edgecolor='inherit', framealpha=1, bbox_to_anchor=(1,.5))

    return ax
###


###################
#  HEATMAP TABLE  #
###################

def heatmap_table( df, cmap=cm.magma, norm=Normalize(vmin=0, vmax=1), figsize=None, digits=3, diagonal=False ) :
    
    assert np.all( df.columns == df.index )
        
    fig, ax = plt.subplots( figsize=figsize )
    _ = ax.imshow( df.values, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest', origin='upper' )
    df = df.fillna("NaN")

    for idxNeg, idxPos in product( np.arange(len(df)), np.arange(len(df)) ) :    
        LabPos = df.index[idxPos]
        LabNeg = df.index[idxNeg]
        
        note = df.at[LabPos,LabNeg]
        if note != "NaN" :
            note_fmt = f"{note:.{digits}f}"
            ax.annotate( note_fmt, (idxNeg, idxPos), ha='center', va='center', color='black' )
            
    # diagonal
    if diagonal is True :
        for idx in np.arange(len(df)) :    
            Lab = df.index[idx]
            ax.annotate( Lab, (idx - .3, idx), ha='left', va='center', color='black' )
            
    ax.set_xticks(np.arange(-.5, len(df)+1, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(df)+1, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)

    plt.tick_params( axis='x', which='both', bottom=False, top=False, labelbottom=False )
    plt.tick_params( axis='y', which='both', left=False, right=False, labelleft=False )

    ax.set_yticks
    _ = [ ax.spines[w].set_visible(False) for w in ax.spines ]
    
    return ax