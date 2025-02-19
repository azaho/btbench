import os
os.chdir('..') # go back to the root directory

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
import os
import glob
import matplotlib.font_manager as fm
font_path = 'figures/font_arial.ttf'
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
                pattern = f'eval_results_ss_sm/{model_prefix}{spec_suffix}_*_subject[0-9]*_trial?_{task}.json'
                files = glob.glob(pattern)
                
                if files:
                    # First average within subjects across trials
                    subject_averages = {}
                    for f in files:
                        with open(f, 'r') as json_file:
                            data = json.load(json_file)
                            subject_id = data['subject_id']
                            
                            if metric == 'accuracy':
                                value = data['mean_accuracy']
                            else:
                                value = np.nanmean([fold_result['auroc'] for fold_result in data['fold_results']])
                            
                            if subject_id not in subject_averages:
                                subject_averages[subject_id] = []
                            subject_averages[subject_id].append(value)
                    
                    # Store individual subject means
                    performance_data[task][model] = {
                        'subject_means': {subj: np.mean(trials) for subj, trials in subject_averages.items()}
                    }
                else:
                    # Use placeholder data if no files found
                    performance_data[task][model] = {
                        'subject_means': {}
                    }

    # Create figure with 4x5 grid - reduced size
    fig, axs = plt.subplots(4, 5, figsize=(8, 6))

    # Flatten axs for easier iteration
    axs_flat = axs.flatten()

    # Color palette
    colors = sns.color_palette("husl", 4)
    category_colors = {
        'Visual': colors[0],
        'Auditory': colors[1],
        'Language': colors[2],
        'Multimodal': colors[3]
    }

    # Model color palette using viridis
    model_colors = sns.color_palette("viridis", 2)

    # Bar width and spacing
    bar_width = 0.2
    
    # Calculate overall performance for each subject and model
    overall_performance = {model: {} for model in models}
    all_subjects = set()
    
    for task in task_list.keys():
        for model in models:
            for subject, value in performance_data[task][model]['subject_means'].items():
                all_subjects.add(subject)
                if subject not in overall_performance[model]:
                    overall_performance[model][subject] = []
                overall_performance[model][subject].append(value)
    
    # Plot overall performance in first axis
    first_ax = axs_flat[0]
    for i, model in enumerate(models):
        # Ensure we have all 10 subjects by filling missing ones with NaN
        all_subjects_sorted = sorted(all_subjects)
        subject_means = []
        for subject in all_subjects_sorted:
            if subject in overall_performance[model]:
                subject_means.append(np.nanmean(overall_performance[model][subject]))
            else:
                subject_means.append(np.nan)
        
        # Create evenly spaced x positions for all 10 subjects
        x_positions = np.linspace(-bar_width/2, bar_width/2, len(all_subjects_sorted))
        x_positions = x_positions + (i * bar_width)  # Offset for each model
        
        # Plot individual subject points
        first_ax.scatter(x_positions, subject_means, color=model_colors[i], alpha=0.5, s=20)
        
        # Plot mean and SEM
        mean = np.nanmean(subject_means)
        sem = np.nanstd(subject_means) / np.sqrt(np.sum(~np.isnan(subject_means)))
        first_ax.bar(i * bar_width, mean, bar_width,
                    yerr=sem,
                    color=model_colors[i],
                    alpha=0.3,
                    capsize=6)
    
    first_ax.set_title('overall', fontsize=12, pad=10, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    if metric == 'accuracy':
        first_ax.set_ylim(0.2, 1.0)
    else:
        first_ax.set_ylim(0.45, .8)
        first_ax.set_yticks([0.5, 0.6, 0.7, 0.8])
    first_ax.set_xticks([])
    first_ax.set_ylabel(metric)
    first_ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
    first_ax.spines['top'].set_visible(False)
    first_ax.spines['right'].set_visible(False)
    first_ax.tick_params(axis='y')

    # Plot counter - start from 1 for remaining plots
    plot_idx = 1

    for task, chance_level in task_list.items():
        ax = axs_flat[plot_idx]
        
        # Plot bars and points for each model
        for i, model in enumerate(models):
            # Ensure we have all 10 subjects by filling missing ones with NaN
            subject_means = []
            for subject in sorted(all_subjects):
                if subject in performance_data[task][model]['subject_means']:
                    subject_means.append(performance_data[task][model]['subject_means'][subject])
                else:
                    subject_means.append(np.nan)
            
            # Plot individual subject points
            if any(~np.isnan(subject_means)):
                # Create evenly spaced x positions for all 10 subjects
                x_positions = np.linspace(-bar_width/2, bar_width/2, len(all_subjects))
                x_positions = x_positions + (i * bar_width)  # Offset for each model
                ax.scatter(x_positions, subject_means, color=model_colors[i], alpha=0.5, s=20)
            
            # Plot mean and SEM
            mean = np.nanmean(subject_means)
            sem = np.nanstd(subject_means) / np.sqrt(np.sum(~np.isnan(subject_means)))
            ax.bar(i * bar_width, mean, bar_width,
                    yerr=sem,
                    color=model_colors[i],
                    alpha=0.3,
                    capsize=6)
        
        # Customize plot
        ax.set_title(task, fontsize=12, pad=10)
        if metric == 'accuracy':
            ax.set_ylim(0.2, 1.0)
        else:
            ax.set_ylim(0.45, .8)
            ax.set_yticks([0.5, 0.6, 0.7, 0.8])
        ax.set_xticks([])
        if (plot_idx % 5 == 0):  # Left-most plots
            ax.set_ylabel(metric)

        # Add horizontal line at chance level
        if metric == 'auroc':
            chance_level = 0.5
        ax.axhline(y=chance_level, color='black', linestyle='--', alpha=0.5)
        
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
    plt.savefig('figures/ss_sm_task_performance_per_subject.pdf', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    create_performance_figure()