import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
import os
import glob
import matplotlib.font_manager as fm
font_path = 'font_arial.ttf'
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams.update({'font.size': 12})

def create_performance_figure():
    # Define tasks and their categories
    tasks = {
        'Language': [
            'speech', 'onset', 'gpt2_surprisal', 'word_length',
            'word_gap', 'word_index', 'word_head_pos', 'word_part_speech'
        ],
        'Auditory': [
            'volume', 'pitch', 'delta_volume', 'delta_pitch'
        ],
        'Visual': [
            'frame_brightness', 'global_flow', 'local_flow', 
            'global_flow_angle', 'local_flow_angle', 'face_num'
        ],
        'Multimodal': [
            'speaker'
        ]
    }
    task_list = {
        'onset': 0.5, 
        'speech': 0.5,
        'volume': 0.5,
        'pitch': 0.5,
        'speaker': 0.25,
        'delta_volume': 0.5,
        'delta_pitch': 0.5,
        'gpt2_surprisal': 0.5,
        'word_length': 0.5,
        'word_gap': 0.5,
        'word_index': 0.25,
        'word_head_pos': 0.5,
        'word_part_speech': 0.25,
        'frame_brightness': 0.5,
        'global_flow': 0.5,
        'local_flow': 0.5,
        'global_flow_angle': 0.25,
        'local_flow_angle': 0.25,
        'face_num': 0.33,
    }
    # Define models
    models = ['linear (raw voltage)', 'linear (spectrogram)']

    metric = 'auroc' # 'accuracy' or 'auroc'

    # Load performance data from JSON files
    performance_data = {}
    for category in tasks.values():
        for task in category:
            performance_data[task] = {}
            for model in models:
                model_prefix = 'linear' if 'linear' in model else 'cnn'
                spec_suffix = '_spectrogram' if 'spectrogram' in model else '_voltage'
                
                # Find matching files for this task/model combination
                pattern = f'eval_results/{model_prefix}{spec_suffix}_*_subject?_trial?_{task}.json'
                files = glob.glob(pattern)
                
                if files:
                    # Average results across all matching files
                    accuracies = []
                    stds = []
                    for f in files:
                        with open(f, 'r') as json_file:
                            data = json.load(json_file)

                            if metric == 'accuracy':
                                accuracies.append(data['mean_accuracy'])
                            else:
                                accuracies.append(np.nanmean([fold_result['auroc'] for fold_result in data['fold_results']]))
                    performance_data[task][model] = {
                        'mean': np.nanmean(accuracies),
                        'sem': np.nanstd(accuracies) / np.sqrt(len(accuracies)) if len(accuracies) > 0 else 0
                    }
                else:
                    # Use placeholder data if no files found
                    performance_data[task][model] = {
                        'mean': 0,#np.random.uniform(0.5, 0.9),
                        'sem': 0#np.random.uniform(0.02, 0.1)
                    }

    # Create figure with 4x5 grid - reduced size
    fig, axs = plt.subplots(4, 5, figsize=(8, 6))

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
    model_colors = sns.color_palette("Set2", 3)

    # Bar width
    bar_width = 0.2
    
    # Add methodology text in the first plot (top left)
    first_ax = axs_flat[0]
    first_ax.axis('off')
    # first_ax.text(0.5, 0.5, 
    #              "5-fold cross validation\nMean accuracy across subjects",
    #              ha='center', va='center', wrap=True)

    for task, chance_level in task_list.items():
        ax = axs_flat[plot_idx]
        
        # Plot bars for each model
        x = np.arange(len(models))
        for i, model in enumerate(models):
            perf = performance_data[task][model]
            ax.bar(i * bar_width, perf['mean'], bar_width,
                    yerr=perf['sem'], 
                    color=model_colors[i],
                    capsize=6)
        
        # Customize plot
        ax.set_title(task, fontsize=12, pad=10)
        if metric == 'accuracy':
            ax.set_ylim(0.2, 1.0)
        else:
            ax.set_ylim(0.4, 1.0)
            ax.set_yticks([0.4, 0.6, 0.8, 1.0])
        ax.set_xticks([])
        if (plot_idx % 5 == 0) or (plot_idx == 1):  # Left-most plots
            ax.set_ylabel(metric)

        # Add horizontal line at chance level
        if metric == 'auroc':
            chance_level = 0.5
        ax.axhline(y=chance_level, color='black', linestyle='--', alpha=0.5)
        
        # Add category label in top left corner
        #ax.text(0.02, 0.95, category, transform=ax.transAxes, alpha=0.7, ha='left', va='top')
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Make tick labels smaller
        ax.tick_params(axis='y')
        
        plot_idx += 1

    # Create a proxy artist for the chance line with the correct style
    chance_line = plt.Line2D([0], [0], color='black', linestyle='--', alpha=0.5)

    # Add legend at the bottom with custom handles
    handles = [plt.Rectangle((0,0),1,1, color=model_colors[i]) for i in range(len(models))]
    handles.append(chance_line)
    fig.legend(handles, models + ["chance"],
              loc='lower center', 
              bbox_to_anchor=(0.5, 0.05),
              ncol=3,
              frameon=False)

    # Adjust layout with space at the bottom for legend
    plt.tight_layout(rect=[0, 0.1, 1, 1], w_pad=0.4)
    
    # Save figure
    plt.savefig('figures/ss_st_task_performance.pdf', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    create_performance_figure()