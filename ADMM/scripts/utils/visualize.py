""" Helper function to create plots. """

import numpy as np
import matplotlib.pyplot as plt
from UC.scripts.utils.utils import *


def plot_optimization_results(p, result_opt, x0_params):
    """Plot optimization results of QAOA parameters.
    
    Args:
        p (int): Number of layers in the QAOA circuit.
        result_opt (ndarray): Optimized parameters from the QAOA algorithm.
        x0_params (ndarray): Initial parameters for the QAOA algorithm.
    """

    # Use the default style
    plt.style.use('default')

    p_vals = np.arange(1, p + 1)

    plt.figure(figsize=(10, 6))

    # Plotting optimized parameters
    plt.plot(p_vals, result_opt.x[1::2], marker='o', color='forestgreen', markersize=8,
             label=r'$\beta_i$ (Optimized)', linestyle='-')
    plt.plot(p_vals, x0_params[1::2], color='lightgreen', label=r'$\beta_{x0}$ (Initial)', linestyle='--')

    plt.plot(p_vals, result_opt.x[0::2], marker='o', color='firebrick', markersize=8,
             label=r'$\gamma_i$ (Optimized)', linestyle='-')
    plt.plot(p_vals, x0_params[0::2], color='lightcoral', label=r'$\gamma_{x0}$ (Initial)', linestyle='--')

    plt.title('Optimization of QAOA Parameters', fontsize=16, fontweight='bold')
    plt.xlabel('p (Number of Layers)', fontsize=14)
    plt.ylabel('Parameter Values', fontsize=14)
    plt.xticks(p_vals)  # Show all p values on x-axis
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()

    plt.show()



def plot_custom_histogram(counts, highlighted_outcome=None, figsize=(12, 6), 
                          bar_color='skyblue', highlight_color='crimson', 
                          title='Sample Histogram', xlabel='Bitstrings', 
                          ylabel='Counts', max_bitstrings=20, bitstring_rankings=None,
                          remove_xticks=False, display_text=True):
    """
    Plots a custom histogram with an option to highlight a specific bitstring. 
    If there are too many bitstrings, only the top `max_bitstrings` are displayed.
    Optionally, displays performance ranking at the bottom of each bar (inside).

    Parameters:
    counts (dict): Dictionary containing bitstrings as keys and counts as values.
    highlighted_outcome (str): The specific bitstring to highlight in a different color.
    figsize (tuple): Figure size of the plot.
    bar_color (str): Color of the bars (default is 'skyblue').
    highlight_color (str): Color for the highlighted bar (default is 'crimson').
    title (str): Title of the plot.
    xlabel (str): X-axis label.
    ylabel (str): Y-axis label.
    max_bitstrings (int): Maximum number of bitstrings to display on the x-axis.
    bitstring_rankings (dict, optional): Dictionary mapping bitstrings to their performance ranking.
    """
    
    # Sort the counts based on the values in descending order
    sorted_counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))

    # Limit to the top `max_bitstrings` bitstrings
    if len(sorted_counts) > max_bitstrings:
        sorted_counts = dict(list(sorted_counts.items())[:max_bitstrings])

    # Extract keys (bitstrings) and values (counts)
    bitstrings = list(sorted_counts.keys())
    values = list(sorted_counts.values())

    # Create a list of default colors for all bars
    colors = [bar_color] * len(sorted_counts)

    # Assign custom colors based on conditions
    for i, bitstring in enumerate(bitstrings):
        if bitstring_rankings and bitstring in bitstring_rankings:
            # print(bitstring_rankings)
            # print(bitstring)
            if bitstring_rankings[bitstring] == 0:
                colors[i] = 'gray'  # Change color to gray if the rank is 0
            elif bitstring == highlighted_outcome:
                colors[i] = highlight_color  # Change the color for the highlighted outcome

    # Create the bar plot
    fig, ax = plt.subplots(figsize=figsize)

    bar_positions = np.arange(len(bitstrings))
    bars = ax.bar(bar_positions, values, color=colors, edgecolor='black', linewidth=1.2)

    plt.xticks(rotation=60, ha='right')

    # Add bar labels to show counts on top of each bar
    if display_text:
        for i, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()
            # Display the count at the top of each bar
            ax.text(bar.get_x() + bar.get_width() / 2., height + 10, f'{value}', 
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

            # Add the bitstring performance ranking at the bottom inside the bars, only if rank is not 0
            if bitstring_rankings and bitstrings[i] in bitstring_rankings:
                rank = bitstring_rankings[bitstrings[i]]
                if rank != 0:  # Skip writing rank inside the bar if the rank is 0
                    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_y() + 5, 
                            f'{rank}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')

    # Add labels, title, and customize the plot
    if remove_xticks==True:
        ax.set_xticks([])  # Removes both the ticks and their labels
    else:
        ax.set_xticks(bar_positions)
        ax.set_xticklabels(bitstrings, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')



    # Add gridlines for better readability
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.6)

    # Create legend if a highlighted outcome is provided
    if highlighted_outcome:
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=highlight_color, label='Classical Solution'),
            plt.Rectangle((0, 0), 1, 1, color=bar_color, label='Other Counts'),
            plt.Rectangle((0, 0), 1, 1, color='gray', label='Invalid Solution')
        ]
        ax.legend(handles=handles, loc='upper right', fontsize=12, frameon=True)

    # Adjust spacing for better readability
    plt.tight_layout()

    # Show the plot
    plt.show()



def plot_value_distribution(data, optimal_value=None, best_val_found=None, figsize=(10, 6),
                            bar_colors={'unbalanced': 'skyblue', 'slack': 'orange'},
                            title='Value Distribution', 
                            xlabel='Value', ylabel='Counts'):
    """
    Plots a histogram of counts vs values.
    
    Parameters:
    data: List of tuples [(value1, counts1, bitstring), ...]
    optimal_value: Optional value to show as optimal line
    figsize: Tuple for figure size
    bar_colors: Dictionary of colors for different types of values
    title: Plot title
    xlabel, ylabel: Axis labels
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract values and counts from data
    values = [val for val, _, _ in data]
    counts = [cnt for _, cnt, _ in data]
    
    # Create bars for the histogram
    bars = ax.bar(values, counts, width=10,
                  color=bar_colors.get('slack', 'orange'), 
                  label='Counts')
    
    # If optimal value is provided, add it as a dashed line
    if optimal_value is not None:
        ax.axvline(x=optimal_value, color='black', linestyle='--', 
                   label='Optimal', zorder=5)
        
    # If optimal value is provided, add it as a dashed line
    if optimal_value is not None:
        ax.axvline(x=best_val_found, color='black', linestyle='--', 
                   label='Best value')
    
    # Customize the plot
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, pad=20)
    
    # Add legend
    ax.legend()
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    # plt.tight_layout()
    
    return fig, ax



def visualize_admm_details(dict_ADMM, save_path=None, format='pdf', dpi=300, 
                     combine_plots=False, filename_suffix=''):
    """
    Creates five plots showing different metrics from ADMM algorithm results with enhanced visuals.
    
    Parameters:
    -----------
    dict_ADMM : dict
        Dictionary where keys are labels (str) and values are ADMM result objects
        containing state attributes: cost_iterates, lambdas, lambda_mult, 
        residuals, and dual_residuals
    save_path : str, optional
        Directory path where to save the figures. If None, figures are not saved.
    format : str, optional
        Format to save the figures. Default is 'pdf'. Can be 'png', 'jpg', etc.
    dpi : int, optional
        Resolution for saved figures (dots per inch). Default is 300.
    combine_plots : bool, optional
        If True, saves all plots in a single figure. Default is False.
    filename_suffix : str, optional
        Suffix to add to the filename (e.g., '_rho1.0_eps1e-6'). Default is ''.
        
    Returns:
    --------
    figs : list
        List of the figure objects (either separate figures or single combined figure)
    """
    # Define common style parameters
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['legend.fontsize'] = 12
    
    # Define colors for consistency
    colors = plt.cm.tab10(np.linspace(0, 1, len(dict_ADMM)))
    
    if combine_plots:
        # Create a single figure with subplots
        fig, axs = plt.subplots(3, 2, figsize=(15, 20))
        axs = axs.flatten()  # Flatten for easier indexing
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        
        # Plot cost iterates
        for (label, result), color in zip(dict_ADMM.items(), colors):
            axs[0].plot(np.arange(1, len(result.state.cost_iterates)+1), 
                       result.state.cost_iterates, 
                       label=label, color=color, marker='o',
                       markevery=max(1, len(result.state.cost_iterates)//20),
                       markersize=6)
        axs[0].set_xlabel('Iterations', fontweight='bold')
        axs[0].set_ylabel('Cost Iterates', fontweight='bold')
        axs[0].set_title('Convergence of Cost Function', pad=15)
        axs[0].grid(True, linestyle='--', alpha=0.7)
        axs[0].legend(frameon=True, fancybox=True, shadow=True)
        
        # Plot lambdas
        for (label, result), color in zip(dict_ADMM.items(), colors):
            axs[1].plot(np.arange(1, len(result.state.lambdas)+1), 
                       result.state.lambdas, 
                       label=label, color=color, marker='s',
                       markevery=max(1, len(result.state.lambdas)//20),
                       markersize=6)
        axs[1].set_xlabel('Iterations', fontweight='bold')
        axs[1].set_ylabel('Lambda (Constraint Violation)', fontweight='bold')
        axs[1].set_title('Evolution of Constraint Violation', pad=15)
        axs[1].grid(True, linestyle='--', alpha=0.7)
        axs[1].legend(frameon=True, fancybox=True, shadow=True)
        
        # Plot lambda multipliers
        for (label, result), color in zip(dict_ADMM.items(), colors):
            axs[2].plot(np.arange(1, len(result.state.lambda_mult)+1), 
                       result.state.lambda_mult, 
                       label=label, color=color, marker='^',
                       markevery=max(1, len(result.state.lambda_mult)//20),
                       markersize=6)
        axs[2].set_xlabel('Iterations', fontweight='bold')
        axs[2].set_ylabel('Lambda Multipliers', fontweight='bold')
        axs[2].set_title('Evolution of Lambda Multipliers', pad=15)
        axs[2].grid(True, linestyle='--', alpha=0.7)
        axs[2].legend(frameon=True, fancybox=True, shadow=True)
        
        # Plot residuals
        for (label, result), color in zip(dict_ADMM.items(), colors):
            axs[3].semilogy(np.arange(1, len(result.state.residuals)+1), 
                          result.state.residuals, 
                          label=label, color=color, marker='d',
                          markevery=max(1, len(result.state.residuals)//20),
                          markersize=6)
        axs[3].set_xlabel('Iterations', fontweight='bold')
        axs[3].set_ylabel('Primal Residual (log scale)', fontweight='bold')
        axs[3].set_title('Evolution of Primal Residuals', pad=15)
        axs[3].grid(True, linestyle='--', alpha=0.7)
        axs[3].legend(frameon=True, fancybox=True, shadow=True)
        
        # Plot dual residuals
        for (label, result), color in zip(dict_ADMM.items(), colors):
            axs[4].semilogy(np.arange(1, len(result.state.dual_residuals)+1), 
                          result.state.dual_residuals, 
                          label=label, color=color, marker='v',
                          markevery=max(1, len(result.state.dual_residuals)//20),
                          markersize=6)
        axs[4].set_xlabel('Iterations', fontweight='bold')
        axs[4].set_ylabel('Dual Residual (log scale)', fontweight='bold')
        axs[4].set_title('Evolution of Dual Residuals', pad=15)
        axs[4].grid(True, linestyle='--', alpha=0.7)
        axs[4].legend(frameon=True, fancybox=True, shadow=True)
        
        # Remove the empty subplot
        axs[5].remove()
        
        fig.suptitle('ADMM Algorithm Convergence Metrics', fontsize=20, y=0.95)
        plt.tight_layout()
        
        figures = [fig]
        
    else:
        # Create separate figures
        # Cost iterates
        fig_cost = plt.figure()
        for (label, result), color in zip(dict_ADMM.items(), colors):
            plt.plot(np.arange(1, len(result.state.cost_iterates)+1), 
                    result.state.cost_iterates, 
                    label=label, color=color, marker='o',
                    markevery=max(1, len(result.state.cost_iterates)//20),
                    markersize=6)
        plt.xlabel('Iterations', fontweight='bold')
        plt.ylabel('Cost Iterates', fontweight='bold')
        plt.title('Convergence of Cost Function', pad=15)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(frameon=True, fancybox=True, shadow=True)
        plt.tight_layout()
        
        # Lambda plot
        fig_lambda = plt.figure()
        for (label, result), color in zip(dict_ADMM.items(), colors):
            plt.plot(np.arange(1, len(result.state.lambdas)+1), 
                    result.state.lambdas, 
                    label=label, color=color, marker='s',
                    markevery=max(1, len(result.state.lambdas)//20),
                    markersize=6)
        plt.xlabel('Iterations', fontweight='bold')
        plt.ylabel('Lambda (Constraint Violation)', fontweight='bold')
        plt.title('Evolution of Constraint Violation', pad=15)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(frameon=True, fancybox=True, shadow=True)
        plt.tight_layout()
        
        # Lambda multipliers plot
        fig_lambda_mult = plt.figure()
        for (label, result), color in zip(dict_ADMM.items(), colors):
            plt.plot(np.arange(1, len(result.state.lambda_mult)+1), 
                    result.state.lambda_mult, 
                    label=label, color=color, marker='^',
                    markevery=max(1, len(result.state.lambda_mult)//20),
                    markersize=6)
        plt.xlabel('Iterations', fontweight='bold')
        plt.ylabel('Lambda Multipliers', fontweight='bold')
        plt.title('Evolution of Lambda Multipliers', pad=15)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(frameon=True, fancybox=True, shadow=True)
        plt.tight_layout()
        
        # Residuals plot
        fig_residuals = plt.figure()
        for (label, result), color in zip(dict_ADMM.items(), colors):
            plt.semilogy(np.arange(1, len(result.state.residuals)+1), 
                        result.state.residuals, 
                        label=label, color=color, marker='d',
                        markevery=max(1, len(result.state.residuals)//20),
                        markersize=6)
        plt.xlabel('Iterations', fontweight='bold')
        plt.ylabel('Primal Residual (log scale)', fontweight='bold')
        plt.title('Evolution of Primal Residuals', pad=15)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(frameon=True, fancybox=True, shadow=True)
        plt.tight_layout()
        
        # Dual residuals plot
        fig_dual_residuals = plt.figure()
        for (label, result), color in zip(dict_ADMM.items(), colors):
            plt.semilogy(np.arange(1, len(result.state.dual_residuals)+1), 
                        result.state.dual_residuals, 
                        label=label, color=color, marker='v',
                        markevery=max(1, len(result.state.dual_residuals)//20),
                        markersize=6)
        plt.xlabel('Iterations', fontweight='bold')
        plt.ylabel('Dual Residual (log scale)', fontweight='bold')
        plt.title('Evolution of Dual Residuals', pad=15)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(frameon=True, fancybox=True, shadow=True)
        plt.tight_layout()
        
        figures = [fig_cost, fig_lambda, fig_lambda_mult, fig_residuals, fig_dual_residuals]
        figs_dict = {
            'cost': (fig_cost, 'Cost Function'),
            'lambda': (fig_lambda, 'Constraint Violation'),
            'lambda_mult': (fig_lambda_mult, 'Lambda Multipliers'),
            'residuals': (fig_residuals, 'Primal Residuals'),
            'dual_residuals': (fig_dual_residuals, 'Dual Residuals')
        }
    
    # Save figures if save_path is provided
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        
        # Check for existing files and get user confirmation once
        proceed, existing_files = check_existing_files(save_path, format, filename_suffix, combine_plots)
        
        if proceed:
            if combine_plots:
                filename = os.path.join(save_path, f'ADMM_all_metrics{filename_suffix}.{format}')
                fig.savefig(filename, format=format, dpi=dpi, bbox_inches='tight')
                print(f'Saved combined figure to: {filename}')
            else:
                for name, (fig, _) in figs_dict.items():
                    filename = os.path.join(save_path, f'ADMM_{name}{filename_suffix}.{format}')
                    fig.savefig(filename, format=format, dpi=dpi, bbox_inches='tight')
                    print(f'Saved figure to: {filename}')
        else:
            print("File saving cancelled by user.")
    
    plt.show()
    return figures





def visualize_admm_performance(dict_solvers, runtime_gurobi=None, cost_gurobi=None,
                             save_path=None, format='pdf', dpi=300, filename_suffix=''):
    """
    Generates performance comparison plots for different ADMM solvers.

    Parameters:
    -----------
    dict_solvers : dict
        Dictionary where keys are solver labels (str) and values are tuples of (admm_instance, result)
        containing the ADMM instance and its corresponding result
    runtime_gurobi : float, optional
        Execution time for Gurobi benchmark (if available)
    cost_gurobi : float, optional
        Cost result for Gurobi benchmark (if available)
    save_path : str, optional
        Directory path where to save the figures. If None, figures are displayed only.
    format : str, optional
        Format to save the figures. Default is 'pdf'.
    dpi : int, optional
        Resolution for saved figures (dots per inch). Default is 300.
    filename_suffix : str, optional
        Suffix to add to the filename. Default is ''.

    Returns:
    --------
    figs : list
        List of the figure objects
    """
    # Define common style parameters
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['legend.fontsize'] = 12

    # Define colors for consistency
    num_colors = len(dict_solvers) + (1 if runtime_gurobi is not None else 0)
    colors = plt.cm.tab10(np.linspace(0, 1, num_colors))
    markers = ['s', '^', 'D', 'x', 'o', 'v', '<', '>', 'p', '*']  # Various marker styles

    # Plot 1: Execution Time vs. Cost
    fig_time = plt.figure()
    
    # Plot Gurobi benchmark if provided
    if runtime_gurobi is not None and cost_gurobi is not None:
        plt.scatter(runtime_gurobi, cost_gurobi, 
                   label='Gurobi', marker='o', 
                   color=colors[0], s=100)
        color_start = 1
    else:
        color_start = 0

    # Plot ADMM solvers
    for (label, (admm, result)), color, marker in zip(dict_solvers.items(), 
                                                     colors[color_start:], 
                                                     markers):
        plt.scatter(admm._state.exec_time, result.fval, 
                   label=label, marker=marker, 
                   color=color, s=100)

    plt.xlabel('Execution Time (s)', fontweight='bold')
    plt.ylabel('Cost', fontweight='bold')
    plt.title('Execution Time vs. Cost for Different Solvers', pad=15)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.tight_layout()

    # Plot 2: Iterations vs. Cost
    fig_iter = plt.figure()
    
    # Plot ADMM solvers
    for (label, (admm, result)), color, marker in zip(dict_solvers.items(), 
                                                     colors[color_start:], 
                                                     markers):
        plt.scatter(len(admm._state.x0_saved), result.fval, 
                   label=label, marker=marker, 
                   color=color, s=100)

    # Set integer ticks for x-axis based on max iterations
    max_iterations = max(len(admm._state.x0_saved) for admm, _ in dict_solvers.values())
    # plt.xticks(range(0, max_iterations + 3))  # Add buffer

    plt.xlabel('Number of Iterations', fontweight='bold')
    plt.ylabel('Cost', fontweight='bold')
    plt.title('Iterations vs. Cost for Different Solvers', pad=15)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.tight_layout()

    figures = [fig_time, fig_iter]
    
    # Save figures if save_path is provided
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        
        # Save execution time vs cost plot
        filename_time = os.path.join(save_path, 
                                   f'execution_time_vs_cost{filename_suffix}.{format}')
        fig_time.savefig(filename_time, format=format, dpi=dpi, bbox_inches='tight')
        print(f'Saved figure to: {filename_time}')
        
        # Save iterations vs cost plot
        filename_iter = os.path.join(save_path, 
                                   f'iterations_vs_cost{filename_suffix}.{format}')
        fig_iter.savefig(filename_iter, format=format, dpi=dpi, bbox_inches='tight')
        print(f'Saved figure to: {filename_iter}')

    plt.show()
    return figures



def plot_value_history(value_history, sample_rate=10):
    """
    Plots the optimization progress by displaying the mean values 
    of the objective function over iterations, sampling the data 
    to maintain readability for large datasets.
    
    Args:
        value_history (list): List of mean values from the optimization process.
        sample_rate (int): Number of data points to skip for clarity.
    """
    # Ensure we don't exceed the available indices
    if sample_rate <= 0:
        sample_rate = 1  # Default to sampling every point if sample_rate is non-positive
    
    # Sample the value history for better visualization
    sampled_indices = np.arange(0, len(value_history), sample_rate)
    sampled_values = [value_history[i] for i in sampled_indices]

    plt.figure(figsize=(12, 6))
    plt.plot(sampled_indices, sampled_values, marker='o', linestyle='-', color='b', markersize=4)
    
    plt.title('Optimization Progress: Objective Function Mean Values')
    plt.xlabel('Iteration (Sampled every {} points)'.format(sample_rate))
    plt.ylabel('Mean Objective Value')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()




def plot_qaoa_optimization_results(optimization_results, gamma0=None, beta0=None, show_legend=False):
    """
    Visualize QAOA parameters (γ and β) from multiple SciPy optimization runs.
    
    Args:
        optimization_results: List of SciPy optimization results.
        only_final: Boolean, if True, only plots the final values from each optimization.
        p: Number of QAOA layers (if None, inferred from the data).
        gamma0: NumPy array of initial γ values, length must match p (optional).
        beta0: NumPy array of initial β values, length must match p (optional).
    
    Returns:
        None. Displays the plots of the optimization results.
    """
    # Extract parameters from optimization results
    parameters = np.array([result.x for result in optimization_results])

    p = len(parameters[0]) // 2

    # Validate input length for gamma0 and beta0
    if gamma0 is not None and len(gamma0) != p:
        raise ValueError(f"Length of gamma0 ({len(gamma0)}) must match the number of layers p ({p}).")
    if beta0 is not None and len(beta0) != p:
        raise ValueError(f"Length of beta0 ({len(beta0)}) must match the number of layers p ({p}).")

    # Extract interleaved gamma and beta values
    betas = parameters[:, 1::2]  # Odd-indexed values
    gammas = parameters[:, 0::2]   # Even-indexed values

    plt.style.use('default')

    # Plot γ Parameters
    fig_gamma, ax_gamma = plt.subplots(figsize=(12, 6))
    ax_gamma.set_facecolor('white')


    for idx, i in enumerate(range(len(gammas))):
        print(idx,i)
        ax_gamma.plot(np.arange(1, p+1), gammas[idx], label=f'$\gamma$ {idx+1}')

    # Plot initial guess for gamma0, if provided
    if gamma0 is not None:
        ax_gamma.plot(np.arange(1, p+1), gamma0, color='black',
                      linestyle='dashed', linewidth=2.5, label='Initial $\gamma_0$')

    ax_gamma.set_xlabel('p', fontsize=14)
    ax_gamma.set_title('$\\gamma$ Parameters Distribution', fontsize=16, fontweight='bold')
    ax_gamma.set_ylabel('$\\gamma$ Values', fontsize=14)
    ax_gamma.set_xticks(np.arange(1, p+1))
    ax_gamma.tick_params(axis='both', labelsize=12)
    if show_legend:
        ax_gamma.legend(title='Parameters', fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_gamma.grid(True, alpha=0.3)
    fig_gamma.patch.set_facecolor('white')
    plt.tight_layout()


    # Plot β Parameters
    fig_beta, ax_beta = plt.subplots(figsize=(12, 6))
    ax_beta.set_facecolor('white')

    for idx, i in enumerate(range(len(gammas))):
        ax_beta.plot(np.arange(1, p+1), betas[idx], label=f'$\\beta$ {idx+1}')

    # Plot initial guess for beta0, if provided
    if beta0 is not None:
        ax_beta.plot(np.arange(1, p+1), beta0, color='black',
                     label='Initial $\\beta_0$', linestyle='dashed', linewidth=2.5, )

    ax_beta.set_xlabel('p', fontsize=14)
    ax_beta.set_title('$\\beta$ Parameters Distribution', fontsize=16, fontweight='bold')
    ax_beta.set_ylabel('$\\beta$ Values', fontsize=14)
    ax_beta.tick_params(axis='both', labelsize=12)
    ax_gamma.set_xticks(np.arange(1, p+1))
    if show_legend:
        ax_beta.legend(title='Parameters', fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_beta.grid(True, alpha=0.3)
    fig_beta.patch.set_facecolor('white')
    plt.tight_layout()

    plt.show()


def plot_admm_cost(result, rho_init, beta, factor_c, max_iter, three_block, tol):
    """
    Plot the cost vs iteration for ADMM optimization with annotated parameters.

    Args:
        cost_it (list or np.array): Cost values for each iteration.
        rho_init (int, optional): Initial value of rho. Default is 650.
        rho (float, optional): Final value of rho. Default is None.
        beta (int, optional): Beta parameter. Default is 1.
        factor_c (int, optional): Factor c parameter. Default is 100.
        max_iter (int, optional): Maximum number of iterations. Default is 100.
        three_block (bool, optional): Whether to use the three-block method. Default is False.
        tol (float, optional): Tolerance for convergence. Default is 1e-9.
    """

    cost_it = result.state.cost_iterates
    rho = result.state.rho

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(len(cost_it)), cost_it, marker='o', linestyle='-', 
             color='b', markersize=5, label='Cost per iteration')

    # Titles and labels
    plt.title("Cost vs Iteration for ADMM Optimization", fontsize=16, fontweight='bold')
    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Cost", fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)

    # Add parameter info as a text box inside the plot
    params_text = (
        f"$\\rho_{{init}}$: {rho_init}\n"
        f"$\\rho$: {rho}\n"
        f"$\\beta$: {beta}\n"
        f"Factor c: {factor_c}\n"
        f"Max Iter: {max_iter}\n"
        f"Three-block: {three_block}\n"
        f"Tolerance: {tol}"
    )
    
    plt.text(
        0.95, 0.45, params_text, transform=plt.gca().transAxes,
        fontsize=12, va='top', ha='right', bbox=dict(boxstyle='round,pad=0.3', 
        edgecolor='black', facecolor='lightgray')
    )

    # Add legend and adjust layout
    plt.legend(loc='upper right', fontsize=12)
    plt.tight_layout()

    # Display the plot
    plt.show()




def plot_qaoa_parameter_evolution(gammas_history, betas_history, p, init_gamma=None, init_beta=None):
    """
    Visualize the evolution of QAOA variational parameters (γ and β) throughout the optimization process.
    
    Args:
        gammas_history: List of lists containing gamma parameters for each iteration.
        betas_history: List of lists containing beta parameters for each iteration.
        p: Number of layers (depth) in the QAOA circuit.
        init_gamma: Optional numpy array of initial gamma values to show as reference.
        init_beta: Optional numpy array of initial beta values to show as reference.
        
    Returns:
        None. Displays two plots showing the evolution of gamma and beta parameters.
    """
    plt.style.use('default')
    # Set classic color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Create a figure for gamma parameters
    fig_gamma = plt.figure(figsize=(12, 6))
    ax_gamma = fig_gamma.add_subplot(111)
    
    iterations = range(len(gammas_history))
    
    for i in range(p):
        # Plot each gamma parameter over iterations with consistent color
        line_color = colors[i % len(colors)]  # Cycle through colors if p > len(colors)
        ax_gamma.plot(iterations,
                     [g[i] for g in gammas_history],
                     marker='o',
                     label=f'$\\gamma_{{{i+1}}}$',
                     linewidth=2,
                     color=line_color)
        
        # Plot initial gamma if provided with matching color
        if init_gamma is not None:
            ax_gamma.axhline(y=init_gamma[i],
                           color=line_color,
                           linestyle='-',
                           alpha=1,
                           label=f'Initial $\\gamma_{{{i+1}}}$')
    
    ax_gamma.set_title('$\\gamma$ Parameters Evolution', fontsize=16)
    ax_gamma.set_xlabel('Optimization Iterations', fontsize=14)
    ax_gamma.set_ylabel('$\\gamma$ Values', fontsize=14)
    ax_gamma.tick_params(axis='both', labelsize=12)
    ax_gamma.legend(title='Parameters', fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_gamma.grid(True, alpha=0.6)
    plt.tight_layout()
    
    # Create a figure for beta parameters
    fig_beta = plt.figure(figsize=(12, 6))
    ax_beta = fig_beta.add_subplot(111)
    
    for i in range(p):
        # Plot each beta parameter over iterations with consistent color
        line_color = colors[i % len(colors)]  # Cycle through colors if p > len(colors)
        ax_beta.plot(iterations,
                    [b[i] for b in betas_history],
                    marker='o',
                    label=f'$\\beta_{{{i+1}}}$',
                    linewidth=2,
                    color=line_color)
        
        # Plot initial beta if provided with matching color
        if init_beta is not None:
            ax_beta.axhline(y=init_beta[i],
                          color=line_color,
                          linestyle='-',
                          alpha=1,
                          label=f'Initial $\\beta_{{{i+1}}}$')
    
    ax_beta.set_title('$\\beta$ Parameters Evolution', fontsize=16)
    ax_beta.set_xlabel('Optimization Iterations', fontsize=14)
    ax_beta.set_ylabel('$\\beta$ Values', fontsize=14)
    ax_beta.tick_params(axis='both', labelsize=12)
    ax_beta.legend(title='Parameters', fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_beta.grid(True, alpha=0.6)
    plt.tight_layout()
    
    plt.show()


def plot_cost_comparison_param(param_range, qaoa_cost, cost_classical, 
                         param_name, filename='', 
                         title='Cost Comparison: QAOA vs Classical'):
    """
    Plots the cost comparison between QAOA and classical cost, filtering out zero values in the QAOA cost.
    
    Args:
        param_range (array-like): The range of parameter values (e.g., p or lambda).
        qaoa_cost (array-like): The cost values for QAOA corresponding to each parameter.
        cost_classical (float): The classical cost value.
        param_name (str): Name of the parameter being plotted (default is 'p (Depth)').
        title (str): Title for the plot (default is 'Cost Comparison: QAOA vs Classical').
        filename (str): If provided, saves the plot as a file with this name (default is '').
    """
    
    # If param_range is a scalar, convert it to a list for uniform iteration handling
    if isinstance(param_range, (int, float)):
        param_range = np.array([param_range])

    # Convert qaoa_cost to a NumPy array if not already
    qaoa_cost = np.array(qaoa_cost)

    # Create a mask to filter out zero values in qaoa_cost
    mask = qaoa_cost > 0

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(param_range[mask], qaoa_cost[mask], label='Cost QAOA', color='royalblue', marker='o', linestyle='-', markersize=8)
    plt.axhline(cost_classical, label='Cost Classical', color='crimson', linestyle='--', linewidth=2)

    # Add a horizontal line for the minimum QAOA cost (if applicable)
    if np.any(mask):  # Ensure there are non-zero values in qaoa_cost
        min_cost_qaoa = np.min(qaoa_cost[mask])  # Min value of the QAOA cost
        plt.axhline(min_cost_qaoa, color='royalblue', linestyle='--', linewidth=2, label='Min QAOA Cost')

    # Add labels and title
    plt.xlabel(f'{param_name}', fontsize=14)
    plt.ylabel('Cost', fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylim(bottom=0)  # Ensure the y-axis starts from zero

    # Set x-ticks to be integers if applicable (only works for numeric ranges like depth or lambda)
    if np.issubdtype(type(param_range[0]), np.integer) or param_range.dtype.kind in 'iu':
        plt.xticks(param_range)

    # Add legend
    plt.legend(loc='upper right', fontsize=12)
    plt.tight_layout()

    # Save or show the plot
    if filename:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        plt.show()



def visualize_optimal_power_distribution(param_exec, show_details_cost=True, **power_distributions):
    """
    Plots the optimal power distributions from the provided dictionaries.
    Optionally shows unit parameters A, B, C for each unit under the bars.

    Args:
        param_exec (dict): Contains problem parameters like 'p_min', 'p_max', 'A', 'B', 'C'.
        show_details_cost (bool): If True, prints detailed cost calculations.
        **power_distributions: Arbitrary number of power distribution dictionaries with labels as keys.
    """
    
    # Ensure p_min and p_max are numpy arrays for consistent indexing
    p_min = np.array(param_exec['p_min'])
    p_max = np.array(param_exec['p_max'])
    A = param_exec['A']
    B = param_exec['B']
    C = param_exec['C']

    num_units = len(p_min)  # Number of power units
    unit_indices = np.arange(num_units)  # Indices for the units

    # Create the bar plot for the range from p_min to p_max
    plt.figure(figsize=(10, 6))

    # Create bars for the range from p_min to p_max
    bars = plt.bar(unit_indices, p_max - p_min, bottom=p_min, color='lightgrey',
                   edgecolor='black', linewidth=1.2, label='Power Range')

    # Initialize total cost storage for each power distribution
    total_costs = {label: 0 for label in power_distributions}

    # Plot the power distributions
    colors = ['royalblue', 'crimson', 'green', 'purple', 'orange']  # Add more colors if needed
    for idx, (label, power_array) in enumerate(power_distributions.items()):
        power_array = np.array(power_array['power'])  # Ensure it's a numpy array
        print("power array", power_array)
        units_on = (power_array > 0).astype(int)  # 1 if on, 0 if off

        # Plot the optimal power output lines
        for i in range(num_units):
            plt.hlines(y=power_array[i], xmin=bars[i].get_x(),
                       xmax=bars[i].get_x() + bars[i].get_width(),
                       color=colors[idx % len(colors)], linestyle='-', linewidth=3,
                       label=label if i == 0 else "")

        # Calculate and print detailed costs (if enabled)
        if show_details_cost:
            print(f'\n{label}')
            for i in range(num_units):
                cost = A[i] * units_on[i] + B[i] * power_array[i] + C[i] * (power_array[i] ** 2)
                total_costs[label] += cost

                # Format the output for better alignment
                print(f"Unit {i+1}: {A[i]:>6.1f} x {units_on[i]} + "
                      f"{B[i]:>6.2f} x {power_array[i]:>8.2f} + "
                      f"{C[i]:>8.3f} x {power_array[i]**2:>8.2f} = {cost:>30.2f}")

            print(f"{'Total ' + label + ' Cost:':>43} {total_costs[label]:.2f}")

    # # Annotate the bars with their values for the first power distribution
    # first_label, first_array = next(iter(power_distributions.items()))
    # for i, bar in enumerate(bars):
    #     plt.text(bar.get_x() + bar.get_width() / 2, first_array[i] + 0.5,
    #              f'{first_array[i]:.2f}', ha='center', va='bottom',
    #              color='royalblue', fontsize=11)

    #     if A and B and C:
    #         # Display A, B, C values just under the bar
    #         plt.text(bar.get_x() + bar.get_width() / 2, p_min[i] - 2,
    #                  f'A={A[i]:.1f}, B={B[i]:.1f}, C={C[i]:.2f}',
    #                  ha='center', va='top', color='black', fontsize=9)

    # Customize the plot
    plt.xlabel('Power Units', fontsize=14)
    plt.ylabel('Power Output', fontsize=14)
    plt.title('Optimal Power Distribution', fontsize=16, fontweight='bold')

    # # Set limits for y-axis to ensure space for the legend and annotations
    # max_power = max(np.max(p_max), max(np.max(arr) for arr in power_distributions.values()))
    # plt.ylim(bottom=0, top=max_power + 10)  # Adding space for annotations
    plt.ylim(bottom=0)

    plt.xticks(unit_indices, [f'Unit {i+1}' for i in unit_indices], fontsize=12, rotation=45)
    plt.yticks(fontsize=12)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show the plot
    plt.tight_layout()
    plt.show()

