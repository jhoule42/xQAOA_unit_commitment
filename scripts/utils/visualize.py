""" Functions to visualize results."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_rank_and_ratio(results, methods=None, labels=None):
    # Extract distributions
    distributions = list(results['very_greedy'].keys())
    distributions_cleaned = [d.replace('generate_', '') for d in distributions]

    # Initialize data for plotting
    if methods == None:
        methods = ['lazy_greedy', 'very_greedy', 'hourglass', 'copula', 'X']
        labels = ['LG', 'VG', r'$QKP_{H}$', r'$QKP_{COP}$', r'$QKP_{X}$']
    num_methods = len(methods)
    bar_width = 0.15  # Adjusted width for better spacing

    # rank_data = {method: [results[method][dist]['rank_solution'] for dist in distributions] for method in methods}
    ratio_data = {method: [results[method][dist]['ratio_optim'] for dist in distributions] for method in methods}

    x = np.arange(len(distributions_cleaned))  # X-axis positions for distributions

    # Updated professional color palette
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', 'k']  # Blue, Green, Red, Purple

    # ### Plot 1: Rank of Each Distribution
    # fig1, ax1 = plt.subplots(figsize=(10, 5))

    # for i, (method, label) in enumerate(zip(methods, labels)):
    #     bars = ax1.bar(
    #         x + i * bar_width,
    #         rank_data[method],
    #         width=bar_width,
    #         label=label,
    #         alpha=0.85,
    #         color=colors[i],
    #         edgecolor='black'
    #     )
        
    #     # Add text labels above each bar for rank
    #     for bar, rank in zip(bars, rank_data[method]):
    #         ax1.text(
    #             bar.get_x() + bar.get_width() / 2,
    #             bar.get_height() + 0.1,
    #             str(rank),
    #             ha='center',
    #             va='bottom',
    #             fontsize=10,
    #             color='black'
    #         )

    # ax1.set_title('Rank of Each Distribution', fontsize=14, fontweight='bold')
    # ax1.set_xlabel('Distribution', fontsize=12)
    # ax1.set_ylabel('Rank', fontsize=12)
    # ax1.set_xticks(x + bar_width * (num_methods - 1) / 2)
    # ax1.set_xticklabels(distributions_cleaned, rotation=0, ha='center')
    # ax1.grid(True, axis='y', linestyle='--', alpha=0.6)
    # ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    # plt.tight_layout()

    # # Show the first plot
    # plt.show()

    ### Plot 2: Ratio to Optimal for Each Distribution
    fig2, ax2 = plt.subplots(figsize=(10, 5))

    for i, (method, label) in enumerate(zip(methods, labels)):
        bars = ax2.bar(
            x + i * bar_width,
            ratio_data[method]*100,
            width=bar_width,
            label=label,
            alpha=0.85,
            color=colors[i],
            edgecolor='black'
        )
        
        # Add text labels above each bar for ratio
        for bar, ratio in zip(bars, ratio_data[method]):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.03,
                f'{ratio*100:.2f}',
                ha='center',
                va='bottom',
                fontsize=10,
                color='black'
            )

    ax2.set_title('Ratio to Optimal for Each Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Distribution', fontsize=12)
    ax2.set_ylabel('Ratio to Optimal', fontsize=12)
    ax2.set_ylim(0, 1.1)
    ax2.set_xticks(x + bar_width * (num_methods - 1) / 2)
    ax2.set_xticklabels(distributions_cleaned, rotation=0, ha='center')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    plt.tight_layout()

    # Show the second plot
    plt.show()


def plot_histogram_with_vlines(values_unbalanced, min_cost, values_slack=None,
                               bins_width=50,log=True, output_file=None, return_bin_height=False):
    """
    Plots histograms of two datasets with vertical lines indicating an optimal value.
    
    Parameters:
        values_unbalanced (dict): Data for the unbalanced histogram (keys as bins, values as weights).
        values_slack (dict): Data for the slack histogram (keys as bins, values as weights).
        min_cost (float): The value at which to draw the vertical line.
        output_file (str, optional): File path to save the plot (e.g., "output.png"). Defaults to None.
    """
    fig, ax = plt.subplots(figsize=(10, 6))  # Larger figure size for better readability

    # Plot unbalanced histogram and get counts
    unbalanced_counts, unbalanced_bins, _ = ax.hist(
        list(values_unbalanced.keys()),
        weights=list(values_unbalanced.values()),
        bins=bins_width,
        edgecolor="black",
        label="Unbalanced",
        align="right",
        alpha=0.7,
        color="steelblue"
    )

    max_height = unbalanced_counts.max()
    print(f"max height: {max_height}")
    if return_bin_height:
        fig.clear()
        plt.close()
        return max_height

    else:
        if values_slack:
            ax.hist(
                values_slack.keys(),
                weights=values_slack.values(),
                bins=bins_width,
                edgecolor="black",
                label="Slack",
                align="left",
                alpha=0.7,
                color="orange"
            )

            # Get the maximum height for slack histogram
            max_slack_height = values_slack.max()
            print(f"max height: {max_slack_height}")

        # Add vertical line
        ax.axvline(-min_cost, linestyle="--", color="red", label="Optimal", linewidth=2)

        # Set log scale for y-axis
        if log:
            ax.set_yscale("log")

        # Add labels, title, and legend
        ax.set_ylabel("Counts", fontsize=14)
        ax.set_xlabel("Values", fontsize=14)
        ax.set_title("Comparison of Values Distributions", fontsize=16)
        ax.legend(fontsize=12)


        # Add gridlines for clarity
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

        # Show or save plot
        if output_file:
            plt.savefig(output_file, bbox_inches="tight")
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



def plot_heatmap(data_dict, beta_params, gamma_params, best_value, vmax=20000, cmap='YlGnBu'):
    """
    Create a heatmap from a dictionary of data.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary with (beta, gamma) tuples as keys and values as counts
    beta_params : list
        List of unique beta parameter values
    gamma_params : list
        List of unique gamma parameter values
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing the heatmap
    """
    # Create a 2D array to hold the data
    heatmap_data = np.zeros((len(beta_params), len(gamma_params)))
    
    # Fill the heatmap with values from the dictionary
    for (beta, gamma), value in data_dict.items():
        if beta in beta_params and gamma in gamma_params:
            beta_idx = beta_params.index(beta)
            gamma_idx = gamma_params.index(gamma)
            if value > best_value:
                heatmap_data[beta_idx, gamma_idx] = 0
            else:
                heatmap_data[beta_idx, gamma_idx] = value
    # Create the plot
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, 
                annot=False, 
                cmap=cmap,
                vmin=0,
                vmax=vmax,
                cbar_kws={'label': 'Knapsack Value'}
                )
    
    plt.title('Heatmap of Parameter Counts')
    plt.xlabel('Gamma Parameters')
    plt.ylabel('Beta Parameters')
    plt.tight_layout()
    
    return plt.gcf()




def plot_best_values(results, methods=None, labels=None):
    """
    Plots the best values for each method for 'generate_profit_spanner'.
    """
    # Ensure we're looking at 'generate_profit_spanner'
    distribution = 'generate_profit_spanner'
    
    # Initialize methods and labels
    if methods is None:
        methods = list(results.keys())
    if labels is None:
        labels = methods

    num_methods = len(methods)
    bar_width = 0.15

    # Extract best values for each method
    best_value_data = [
        results.get(method, {}).get(distribution, {}).get('best_value', 0) 
        for method in methods
    ]

    x = np.arange(len(methods))
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', 'k']

    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(
        x,
        best_value_data,
        width=bar_width,
        alpha=0.85,
        color=[colors[i % len(colors)] for i in range(len(methods))],
        edgecolor='black'
    )
    
    # Add text labels above each bar
    for bar, value in zip(bars, best_value_data):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f'{value:.0f}',
            ha='center',
            va='bottom',
            fontsize=10,
            color='black'
        )

    ax.set_title('Best Values for Profit Spanner Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Methods', fontsize=12)
    ax.set_ylabel('Best Value', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    plt.show()


def plot_method_comparison(results, optimal_value,  methods=None, labels=None, title=None,
                           bar_width=0.15):
    """
    Plot a histogram comparing method performance across distributions.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing results for different methods and distributions
    optimal_value : float
        The optimal value to compare against
    methods : list, optional
        List of methods to plot (default is predefined set)
    labels : list, optional
        Custom labels for methods (default is predefined set)
    title : str, optional
        Custom title for the plot
    """
    # Extract distributions
    distributions = list(results['very_greedy'].keys())
    distributions_cleaned = [d.replace('generate_', '') for d in distributions]
    
    # Set default methods and labels if not provided
    if methods is None:
        methods = ['lazy_greedy', 'very_greedy', 'copula']
    if labels is None:
        labels = ['LG', 'VG', r'$QKP_{COP}$']
    
    # Professional color palette
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', 'k']
    
    # Prepare data
    best_values = {method: [results[method][dist]['value']\
                            for dist in distributions] for method in methods}
    
    min_values = {method: min(values) for method, values in best_values.items()}
    print(min_values)

    min_method = min(min_values.values())
    min_value = min(best_values.values())
    print(min_value[0])
    print(min_method)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))  # Increase the figure height
    # fig.subplots_adjust(top=0.)  # Adjust the top margin
    
    # Bar width and positioning
    bar_width = bar_width
    x = np.arange(len(distributions_cleaned))
    
    # Plot bars for each method
    for i, (method, label) in enumerate(zip(methods, labels)):
        bars = ax.bar(
            x + i * bar_width, 
            best_values[method], 
            width=bar_width,
            label=label,
            alpha=0.95,
            color=colors[i],
            edgecolor='black'
        )
        
        # Add text labels with performance relative to optimal
        for bar, value in zip(bars, best_values[method]):
            # Calculate percentage of optimal value
            perf_percentage = (value / optimal_value) * 100
            
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f'{value}\n ({perf_percentage:.2f}%)',
                ha='center',
                va='bottom',
                fontsize=12,
                color='black',
                fontweight='bold'
            )

    # Customize plot
    ax.set_title(title or 'Performance Comparison Across Distributions', fontsize=14, fontweight='bold')
    ax.set_xlabel('Distribution', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    
    # Set x-ticks
    ax.set_xticks(x + bar_width * (len(methods) - 1) / 2)
    ax.set_xticklabels(distributions_cleaned, rotation=0, ha='right')
    
    # # Add horizontal line for optimal value
    # ax.axhline(y=optimal_value, color='r', linestyle='--', label='Optimal Value')
    
    # Customize grid and legend
    # ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)

    ax.set_yscale('log')
    ax.set_ylim(top=optimal_value * 1.001)
    
    plt.tight_layout()
    plt.show()



def plot_optimal_parameters(parameters):
    """
    Plot the optimal parameters (beta, gamma) as a scatter plot.
    
    Args:
        parameters (list of tuple): A list of (beta, gamma) pairs.
    """
    # Separate the beta and gamma values for plotting
    beta_values = [beta for beta, gamma in parameters]
    gamma_values = [gamma for beta, gamma in parameters]

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(beta_values, gamma_values, color="blue", alpha=0.7, edgecolor="black")
    
    # Add labels and a grid
    plt.title("Optimal Parameters (β, γ)", fontsize=14)
    plt.xlabel("Beta (β)", fontsize=12)
    plt.ylabel("Gamma (γ)", fontsize=12)
    plt.grid(alpha=0.5)
    plt.xlim(0, np.pi)
    plt.ylim(0, 2*np.pi)
    
    # Show the plot
    plt.tight_layout()
    plt.show()
