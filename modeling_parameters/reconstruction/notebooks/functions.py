from __future__ import annotations
import numpy as np
import os
import ast
from modeling_parameters.config import (
    PRIM_PARTICLE, 
    THETA,
    COORDINATES_PATH,
    RECONSTRUCTION_GEANT_OUTPUT_DIR,
    MATPLOTLIBRC_PATH,
    CENTRAL_STATIONS,
    PLOTS_GEANT_DIR,
)

from modeling_parameters.reconstruction.geant.functions import(
     compute_r,
)


import matplotlib as mpl
mpl.rc_file(MATPLOTLIBRC_PATH)
import matplotlib.pyplot as plt
import seaborn as sns


def plot_single_distribution(opt, true, var, E, limit, bin_width, min_stations, top_k, ymax, save):
    """
    Строит график для delta(var)
    opt: np.array оптимизированных значений
    true: np.array настоящих значений
    var: имя переменной
    """
    diff = opt - true
    mean = diff.mean()
    
    abs_dev = np.abs(diff - mean)
    maxabs = max(abs_dev)
    radius = np.percentile(abs_dev, 68)
    lower = mean - radius
    upper = mean + radius

    fig, ax = plt.subplots(figsize=(15, 12))
    bin_width = bin_width
    bins = np.arange(-maxabs - bin_width - 1, maxabs + bin_width + 1, bin_width)
    
    if limit is not None:
        x_min, x_max = -limit, limit
    else:
        x_min, x_max = -maxabs - bin_width - 1, maxabs + bin_width + 1

    sns.histplot(
        diff,
        bins=bins,
        kde=False,
        color='royalblue',
        stat='probability',
        element='step',
        fill=False,
        linewidth=2.5
    )

    counts, bin_edges, _ = plt.hist(
        diff,
        bins=bins,
        weights=np.ones_like(diff)/len(diff),
        alpha=0
    )

    for i in range(len(bin_edges)-1):
        bin_start, bin_end = bin_edges[i], bin_edges[i+1]
        if bin_end <= lower or bin_start >= upper:
            continue
        fill_start = max(bin_start, lower)
        fill_end = min(bin_end, upper)
        plt.fill_between(
            [fill_start, fill_end],
            [0, 0],
            [counts[i], counts[i]],
            color='royalblue',
            alpha=0.2,
            edgecolor='none'
        )

    plt.axvline(mean, color='royalblue', linestyle='-.', label=f'Среднее = {mean:.1f}')
    
    legend_elements = [
        plt.Line2D([0], [0], color='none', label=rf'$\mathrm{{{PRIM_PARTICLE}}},\ E_0 = 10^{{{E}}}\text{{эВ}},\ \theta = {THETA}\degree$'),
        plt.Line2D([0], [0], color='none', label=f'мин сработавших кластеров  НШ: {min_stations}'),
        plt.Line2D([0], [0], color='none', label=f'Число событий: {len(diff)}'),
        plt.Line2D([0], [0], color='royalblue', linestyle='-.', lw=2, label = rf'$\mu_{{\Delta {var}}}$={mean:.1f}'),
        plt.Line2D([0], [0], color='royalblue', alpha=0.3, lw=10, label='68%')
    ]

    plt.ylabel('Вероятность')
    plt.xlabel(rf'$\Delta {var}$')
    ymin = 0
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(np.linspace(ymin, ymax, 6))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(x_min, x_max)
    ax.legend(handles=legend_elements, loc='best', framealpha=0.9)
    
    if save:
        folder_path = os.path.join(PLOTS_GEANT_DIR, f"{PRIM_PARTICLE}{THETA}", f"E{E}")
        os.makedirs(folder_path, exist_ok=True)  
        filename = os.path.join(folder_path, f"delta_{var}_1.png")
        plt.savefig(filename, dpi=500, bbox_inches='tight')
        plt.close()
        print(f"График сохранён в {filename}")
    else:
        plt.show()

    
def plot_single_distribution_by_name(arrays, varname, E, limit=None, bin_width=5, min_stations=1, top_k=1, ymax=1, save=False):

    opt_all = arrays[f'{varname}_opt']
    true_all = arrays[f'{varname}']
    plot_single_distribution(opt_all, true_all, varname, E, limit, bin_width, min_stations, top_k, ymax, save)

    

def plot_two_distributions(opt_all, true_all, opt_top4, true_top4, var, E, limit, bin_width, min_stations, top_k, ymax, save):
    """
    Строит график для delta(var) двух выборок: всех событий и top4central событий.
    
    opt_all, true_all: np.array для всех событий
    opt_top4, true_top4: np.array для топ-центральных событий
    var: имя переменной
    """
    def get_symmetric_range(data):
        mean = data.mean()
        abs_dev = np.abs(data - mean)
        radius = np.percentile(abs_dev, 68)
        return mean - radius, mean + radius

    diff_all = opt_all - true_all
    diff_top4 = opt_top4 - true_top4

    lower_all, upper_all = get_symmetric_range(diff_all)
    lower_top4, upper_top4 = get_symmetric_range(diff_top4)

    mean = diff_all.mean()
    abs_dev = np.abs(diff_all - mean)
    maxabs = max(abs_dev)
    bin_width = bin_width
    bins = np.arange(-maxabs - bin_width - 1, maxabs + bin_width + 1, bin_width)
    
    if limit is not None:
        x_min, x_max = -limit, limit
    else:
        x_min, x_max = -maxabs - bin_width - 1, maxabs + bin_width + 1

    fig, ax = plt.subplots(figsize=(12, 10))

    sns.histplot(
        diff_all,
        bins=bins,
        kde=False,
        color='royalblue',
        stat='probability',
        element='step',
        fill=False,
        linewidth=2.5,
        linestyle='-',
        ax=ax,
        label='All events'
    )

    sns.histplot(
        diff_top4,
        bins=bins,
        kde=False,
        color='crimson',
        stat='probability',
        element='step',
        fill=False,
        linewidth=2.5,
        linestyle='-',
        ax=ax,
        label='Top4 Central'
    )

    counts_all, bin_edges_all, _ = plt.hist(
        diff_all,
        bins=bins,
        weights=np.ones_like(diff_all)/len(diff_all),
        alpha=0
    )

    counts_top4, bin_edges_top4, _ = plt.hist(
        diff_top4,
        bins=bins,
        weights=np.ones_like(diff_top4)/len(diff_top4),
        alpha=0
    )


    for i in range(len(bin_edges_all)-1):
        bin_start, bin_end = bin_edges_all[i], bin_edges_all[i+1]
        if bin_end <= lower_all or bin_start >= upper_all:
            continue
        fill_start = max(bin_start, lower_all)
        fill_end = min(bin_end, upper_all)
        plt.fill_between(
            [fill_start, fill_end],
            [0, 0],
            [counts_all[i], counts_all[i]],
            color='royalblue',
            alpha=0.2,
            edgecolor='none'
        )

    for i in range(len(bin_edges_top4)-1):
        bin_start, bin_end = bin_edges_top4[i], bin_edges_top4[i+1]
        if bin_end <= lower_top4 or bin_start >= upper_top4:
            continue
        fill_start = max(bin_start, lower_top4)
        fill_end = min(bin_end, upper_top4)
        plt.fill_between(
            [fill_start, fill_end],
            [0, 0],
            [counts_top4[i], counts_top4[i]],
            color='crimson',
            alpha=0.2,
            edgecolor='none'
        )

    plt.axvline(diff_all.mean(), color='royalblue', linestyle='-.', label=r'$\mu_{\text{all}}$')
    plt.axvline(diff_top4.mean(), color='crimson', linestyle='-.', label=r'$\mu_{\text{top4}}$')

    legend_elements = [
        plt.Line2D([0], [0], color='none', label=rf'$\mathrm{{{PRIM_PARTICLE}}},\ E_0 = 10^{{{E}}}\text{{эВ}},\ \theta = {THETA}\degree$'),
        # plt.Line2D([0], [0], color='none', label=f'мин сработавших кластеров  НШ: {min_stations}'),
        # plt.Line2D([0], [0], color='none', label=r'$C \subset S$ - центр. станции'),
        # plt.Line2D([0], [0], color='none', label = rf't{top_k}c: $\text{{argmax}}^{{{top_k}}}_{{\substack{{i \in S}}}} \rho_i \subset C$'),
        plt.Line2D([0], [0], color='none', label=f'число t{top_k}c: {len(diff_top4)}/{len(diff_all)}'),
        # plt.Line2D([0], [0], color='royalblue', lw=2, label='все события'),
        # plt.Line2D([0], [0], color='crimson', lw=2, label=f't{top_k}c события'),
        plt.Line2D([0], [0], color='crimson', linestyle='-.', lw=2, label=rf'$\mu_{{\Delta {var}}}=${diff_top4.mean():.1f}'),
        plt.Line2D([0], [0], color='crimson', alpha=0.2, lw=10, label='68%')
    ]

    plt.ylabel('Вероятность')
    plt.xlabel(rf'$\Delta {var}$')
    ymin = 0
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(np.linspace(ymin, ymax, 6))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(x_min, x_max)
    ax.legend(handles=legend_elements, loc='best', framealpha=0.9)
    
    if save:
        folder_path = os.path.join(PLOTS_GEANT_DIR, f"{PRIM_PARTICLE}{THETA}", f"E{E}")
        os.makedirs(folder_path, exist_ok=True)  
        filename = os.path.join(folder_path, f"delta_{var}_2.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"График сохранён в {filename}")
    else:
        plt.show()
    

def plot_two_distributions_by_name(arrays, varname, E, limit=None, bin_width=5, min_stations=1, top_k=1, ymax=1, save=False):

    opt_all = arrays[f'{varname}_opt']
    true_all = arrays[f'{varname}']
    opt_top4 = arrays[f't4c_{varname}_opt']
    true_top4 = arrays[f't4c_{varname}']

    plot_two_distributions(opt_all, true_all, opt_top4, true_top4, varname, E, limit, bin_width, min_stations, top_k, ymax, save)




def plot_single_distribution_Ne(opt, true, var, E, limit, bin_width, min_stations, top_k, ymax, save):
    """
    Строит график для delta(var)
    opt: np.array оптимизированных значений
    true: np.array настоящих значений
    var: имя переменной
    """
    diff = np.log10(opt / true)
    mean = diff.mean()
    
    abs_dev = np.abs(diff - mean)
    maxabs = max(abs_dev)
    radius = np.percentile(abs_dev, 68)
    lower = mean - radius
    upper = mean + radius

    fig, ax = plt.subplots(figsize=(15, 12))
    bin_width = bin_width
    bins = np.arange(-maxabs - bin_width - 1, maxabs + bin_width + 1, bin_width)
    
    if limit is not None:
        x_min, x_max = -limit, limit
    else:
        x_min, x_max = -maxabs - bin_width - 1, maxabs + bin_width + 1

    sns.histplot(
        diff,
        bins=bins,
        kde=False,
        color='royalblue',
        stat='probability',
        element='step',
        fill=False,
        linewidth=2.5
    )

    counts, bin_edges, _ = plt.hist(
        diff,
        bins=bins,
        weights=np.ones_like(diff)/len(diff),
        alpha=0
    )

    for i in range(len(bin_edges)-1):
        bin_start, bin_end = bin_edges[i], bin_edges[i+1]
        if bin_end <= lower or bin_start >= upper:
            continue
        fill_start = max(bin_start, lower)
        fill_end = min(bin_end, upper)
        plt.fill_between(
            [fill_start, fill_end],
            [0, 0],
            [counts[i], counts[i]],
            color='royalblue',
            alpha=0.2,
            edgecolor='none'
        )

    plt.axvline(mean, color='royalblue', linestyle='-.', label=f'Среднее = {mean:.1f}')
    
    legend_elements = [
        plt.Line2D([0], [0], color='none', label=rf'$\mathrm{{{PRIM_PARTICLE}}},\ E_0 = 10^{{{E}}}\text{{эВ}},\ \theta = {THETA}\degree$'),
        plt.Line2D([0], [0], color='none', label=f'мин сработавших кластеров НШ: {min_stations}'),
        plt.Line2D([0], [0], color='none', label=f'Число событий: {len(diff)}'),
        plt.Line2D([0], [0], color='royalblue', linestyle='-.', lw=2, label = rf'$\mu_{{\Delta  lg(N_e)}}$={mean:.1f}'),
        plt.Line2D([0], [0], color='royalblue', alpha=0.3, lw=10, label='68%')
    ]

    plt.ylabel('Вероятность')
    plt.xlabel(rf'$\Delta  lg(N_e)$')
    
    ymin = 0
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(np.linspace(ymin, ymax, 6))
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(x_min, x_max)
    ax.legend(handles=legend_elements, loc='best', framealpha=0.9)
    
    if save:
        folder_path = os.path.join(PLOTS_GEANT_DIR, f"{PRIM_PARTICLE}{THETA}", f"E{E}")
        os.makedirs(folder_path, exist_ok=True)  
        filename = os.path.join(folder_path, f"delta_lg_{var}_1.png")
        plt.savefig(filename, dpi=500, bbox_inches='tight')
        plt.close()
        print(f"График сохранён в {filename}")
    else:
        plt.show()


def plot_two_distributions_Ne(opt_all, true_all, opt_top4, true_top4, var, E, limit, bin_width, min_stations, top_k, ymax, save):
    """
    Строит график для delta(var) двух выборок: всех событий и top4central событий.
    
    opt_all, true_all: np.array для всех событий
    opt_top4, true_top4: np.array для топ-центральных событий
    var: имя переменной
    """
    def get_symmetric_range(data):
        mean = data.mean()
        abs_dev = np.abs(data - mean)
        radius = np.percentile(abs_dev, 68)
        return mean - radius, mean + radius

    diff_all = np.log10(opt_all / true_all)
    diff_top4 = np.log10(opt_top4 / true_top4)

    lower_all, upper_all = get_symmetric_range(diff_all)
    lower_top4, upper_top4 = get_symmetric_range(diff_top4)

    mean = diff_all.mean()
    abs_dev = np.abs(diff_all - mean)
    maxabs = max(abs_dev)
    bin_width = bin_width
    bins = np.arange(-maxabs - bin_width - 1, maxabs + bin_width + 1, bin_width)
    
    if limit is not None:
        x_min, x_max = -limit, limit
    else:
        x_min, x_max = -maxabs - bin_width - 1, maxabs + bin_width + 1

    fig, ax = plt.subplots(figsize=(15, 12))

    sns.histplot(
        diff_all,
        bins=bins,
        kde=False,
        color='royalblue',
        stat='probability',
        element='step',
        fill=False,
        linewidth=2.5,
        linestyle='-',
        ax=ax,
        label='All events'
    )

    sns.histplot(
        diff_top4,
        bins=bins,
        kde=False,
        color='crimson',
        stat='probability',
        element='step',
        fill=False,
        linewidth=2.5,
        linestyle='-',
        ax=ax,
        label='Top4 Central'
    )

    counts_all, bin_edges_all, _ = plt.hist(
        diff_all,
        bins=bins,
        weights=np.ones_like(diff_all)/len(diff_all),
        alpha=0
    )

    counts_top4, bin_edges_top4, _ = plt.hist(
        diff_top4,
        bins=bins,
        weights=np.ones_like(diff_top4)/len(diff_top4),
        alpha=0
    )


    for i in range(len(bin_edges_all)-1):
        bin_start, bin_end = bin_edges_all[i], bin_edges_all[i+1]
        if bin_end <= lower_all or bin_start >= upper_all:
            continue
        fill_start = max(bin_start, lower_all)
        fill_end = min(bin_end, upper_all)
        plt.fill_between(
            [fill_start, fill_end],
            [0, 0],
            [counts_all[i], counts_all[i]],
            color='royalblue',
            alpha=0.2,
            edgecolor='none'
        )

    for i in range(len(bin_edges_top4)-1):
        bin_start, bin_end = bin_edges_top4[i], bin_edges_top4[i+1]
        if bin_end <= lower_top4 or bin_start >= upper_top4:
            continue
        fill_start = max(bin_start, lower_top4)
        fill_end = min(bin_end, upper_top4)
        plt.fill_between(
            [fill_start, fill_end],
            [0, 0],
            [counts_top4[i], counts_top4[i]],
            color='crimson',
            alpha=0.2,
            edgecolor='none'
        )

    plt.axvline(diff_all.mean(), color='royalblue', linestyle='-.', label=r'$\mu_{\text{all}}$')
    plt.axvline(diff_top4.mean(), color='crimson', linestyle='-.', label=r'$\mu_{\text{top4}}$')

    legend_elements = [
        plt.Line2D([0], [0], color='none', label=rf'$\mathrm{{{PRIM_PARTICLE}}},\ E_0 = 10^{{{E}}}\text{{эВ}},\ \theta = {THETA}\degree$'),
        plt.Line2D([0], [0], color='none', label=f'мин сработавших кластеров  НШ: {min_stations}'),
        plt.Line2D([0], [0], color='none', label=r'$C \subset S$ - центр. станции'),
        plt.Line2D([0], [0], color='none', label = rf't{top_k}c: $\text{{argmax}}^{{{top_k}}}_{{\substack{{i \in S}}}} \rho_i \subset C$'),
        plt.Line2D([0], [0], color='none', label=f'число t{top_k}c: {len(diff_top4)}/{len(diff_all)}'),
        plt.Line2D([0], [0], color='royalblue', lw=2, label='все события'),
        plt.Line2D([0], [0], color='crimson', lw=2, label=f't{top_k}c события'),
        plt.Line2D([0], [0], color='crimson', linestyle='-.', lw=2,  label = rf'$\mu_{{\Delta  lg(N_e)}}$={mean:.1f}'),
        plt.Line2D([0], [0], color='crimson', alpha=0.2, lw=10, label='68%')
    ]

    plt.ylabel('Вероятность')
    plt.xlabel(rf'$\Delta  lg(N_e)$')
    
    ymin = 0
    plt.ylim(ymin, ymax)
    ax.set_yticks(np.linspace(ymin, ymax, 6))
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(x_min, x_max)
    ax.legend(handles=legend_elements, loc='best', framealpha=0.9)
    
    if save:
        folder_path = os.path.join(PLOTS_GEANT_DIR, f"{PRIM_PARTICLE}{THETA}", f"E{E}")
        os.makedirs(folder_path, exist_ok=True)  
        filename = os.path.join(folder_path, f"delta_lg_{var}_2.png")
        plt.savefig(filename, dpi=500, bbox_inches='tight')
        plt.close()
        print(f"График сохранён в {filename}")
    else:
        plt.show()
        


def ast_df(df):
    df['r'] = df['r'].apply(ast.literal_eval)
    df['rho'] = df['rho'].apply(ast.literal_eval)
    df['mask'] = df['mask'].apply(ast.literal_eval)
    df['worked_clusters'] = df['worked_clusters'].apply(ast.literal_eval)
    df['worked_stations'] = df['worked_stations'].apply(ast.literal_eval)
    

def process_df(df, X_det, Y_det, Z_det, min_clusters=2, top_k=3):
    def count_triggered_clusters(worked_stations):
        clusters = [list(range(i, i + 4)) for i in range(0, 36, 4)] 
        count = 0
        for cluster in clusters:
            triggered = len(set(cluster) & set(worked_stations))
            if triggered >= 2:
                count += 1
        return count


    df_filtered = df[df['worked_stations'].apply(count_triggered_clusters) >= min_clusters]

    theta = np.array(df_filtered['theta'])
    phi = np.array(df_filtered['phi'])

    X = np.array(df_filtered['X0'])
    X_opt = np.array(df_filtered['X0_opt'])

    Y = np.array(df_filtered['Y0'])
    Y_opt = np.array(df_filtered['Y0_opt'])

    Ne = np.array(df_filtered['Ne'])
    Ne_opt = np.array(df_filtered['Ne_opt'])

    s = np.array(df_filtered['s'])
    s_opt = np.array(df_filtered['s_opt'])

    rho = np.array(df_filtered['rho'])
    loss = np.array(df_filtered['loss'])

    r = []
    r_opt = []

    for i in range(len(X)):
        r.append(compute_r(X[i], Y[i], -18, theta[i], phi[i], X_det, Y_det, Z_det))
        r_opt.append(compute_r(X_opt[i], Y_opt[i], -18, theta[i], phi[i], X_det, Y_det, Z_det))

    r = np.array(r)
    r_opt = np.array(r_opt)

    def is_topk_central(row, top_k=4):
        worked_stations = row['worked_stations']
        rho = row['rho']
        stations_rho = list(zip(worked_stations, rho))
        stations_rho_sorted = sorted(stations_rho, key=lambda x: x[1], reverse=True)
        topk_stations = [station for station, _ in stations_rho_sorted[:top_k]]
        return all(station in CENTRAL_STATIONS for station in topk_stations)

    topkc_mask = df_filtered.apply(lambda row: is_topk_central(row, top_k=top_k), axis=1)
    topkc_df = df_filtered[topkc_mask]

    t4c_X = np.array(topkc_df['X0'])
    t4c_X_opt = np.array(topkc_df['X0_opt'])

    t4c_Y = np.array(topkc_df['Y0'])
    t4c_Y_opt = np.array(topkc_df['Y0_opt'])

    t4c_Ne = np.array(topkc_df['Ne'])
    t4c_Ne_opt = np.array(topkc_df['Ne_opt'])

    t4c_s = np.array(topkc_df['s'])
    t4c_s_opt = np.array(topkc_df['s_opt'])

    t4c_r = r[topkc_mask]
    t4c_r_opt = r_opt[topkc_mask]

    t4c_rho = np.array(topkc_df['rho'])
    t4c_loss = np.array(topkc_df['loss'])

    r = np.concatenate(r)
    r_opt = np.concatenate(r_opt)

    t4c_r = np.concatenate(t4c_r)
    t4c_r_opt = np.concatenate(t4c_r_opt)

    arrays = {
        'X': X,
        'X_opt': X_opt,
        'Y': Y,
        'Y_opt': Y_opt,
        'Ne': Ne,
        'Ne_opt': Ne_opt,
        's': s,
        's_opt': s_opt,
        'r': r,
        'r_opt': r_opt,

        'rho': rho,
        'loss': loss,

        't4c_X': t4c_X,
        't4c_X_opt': t4c_X_opt,
        't4c_Y': t4c_Y,
        't4c_Y_opt': t4c_Y_opt,
        't4c_Ne': t4c_Ne,
        't4c_Ne_opt': t4c_Ne_opt,
        't4c_s': t4c_s,
        't4c_s_opt': t4c_s_opt,
        't4c_r': t4c_r,
        't4c_r_opt': t4c_r_opt,

        't4c_rho': t4c_rho,
        't4c_loss': t4c_loss,
    }

    return arrays