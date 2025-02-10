import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def create_performance_figure():
    # Define tasks and their categories
    tasks = {
        'Visual': [
            'frame_brightness', 'global_flow', 'local_flow', 
            'global_flow_angle', 'local_flow_angle', 'face_num'
        ],
        'Auditory': [
            'volume', 'pitch', 'delta_volume', 'delta_pitch'
        ],
        'Language': [
            'speech', 'onset', 'gpt2_surprisal', 'word_length',
            'word_gap', 'word_index', 'word_head_pos', 'word_part_speech'
        ],
        'Multimodal': [
            'speaker'
        ]
    }

    # Define models
    models = ['Linear', 'Linear_Spectrogram', 'CNN', 'CNN_Spectrogram']

    # Create placeholder performance data (random values between 0.5 and 0.9)
    np.random.seed(42)  # For reproducibility
    performance_data = {
        task: {
            model: {
                'mean': np.random.uniform(0.5, 0.9),
                'std': np.random.uniform(0.02, 0.1)
            } for model in models
        } for category in tasks.values() for task in category
    }

    # Create figure with 4x5 grid - reduced size
    fig, axs = plt.subplots(4, 5, figsize=(12, 10))
    fig.suptitle('Performance Across Different Tasks', fontsize=16, y=0.98)

    # Flatten axs for easier iteration
    axs_flat = axs.flatten()

    # Plot counter - start from 1 to skip first box
    plot_idx = 1

    # Color palette
    colors = sns.color_palette("husl", 4)
    category_colors = {
        'Visual': colors[0],
        'Auditory': colors[1],
        'Language': colors[2],
        'Multimodal': colors[3]
    }

    # Model color palette
    model_colors = sns.color_palette("Set2", 4)

    # Bar width
    bar_width = 0.2
    
    # Add methodology text in the first plot (top left)
    first_ax = axs_flat[0]
    first_ax.axis('off')
    first_ax.text(0.5, 0.5, 
                 "Train/Test Split Methodology\n(Details to be added)",
                 ha='center', va='center',
                 fontsize=8, wrap=True)

    # Create plots for each task
    for category, task_list in tasks.items():
        for task in task_list:
            ax = axs_flat[plot_idx]
            
            # Plot bars for each model
            x = np.arange(len(models))
            for i, model in enumerate(models):
                perf = performance_data[task][model]
                ax.bar(i * bar_width, perf['mean'], bar_width,
                      yerr=perf['std'], 
                      color=model_colors[i],
                      capsize=3)
            
            # Customize plot
            ax.set_title(task.replace('_', ' ').title(), fontsize=8, pad=5)  # increased pad
            ax.set_ylim(0.4, 1.0)
            ax.set_xticks([])
            if plot_idx % 5 == 0:  # Left-most plots
                ax.set_ylabel('Accuracy', fontsize=8)
            
            # Add category label in top left corner - moved up slightly
            ax.text(0.02, 0.95, category, transform=ax.transAxes,
                   fontsize=6, alpha=0.7, ha='left', va='top')
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Make tick labels smaller
            ax.tick_params(axis='y', labelsize=6)
            
            plot_idx += 1

    # Add legend below the title
    fig.legend(models, 
              loc='upper center', 
              bbox_to_anchor=(0.5, 0.94),
              ncol=4,  # Display models in one row
              fontsize=8, 
              title='Models', 
              title_fontsize=9)

    # Adjust layout with more space at the top for title and legend
    plt.tight_layout(rect=[0, 0, 1, 0.92])  # adjusted rect parameter to leave more space at top
    
    # Show the plot
    plt.show()
    
    # Save figure
    plt.savefig('task_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    create_performance_figure() 