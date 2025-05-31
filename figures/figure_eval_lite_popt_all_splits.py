import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
import os
import glob, math
import matplotlib.font_manager as fm
font_path = 'figures/font_arial.ttf'
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams.update({'font.size': 12})

import argparse
# Parse command line arguments
parser = argparse.ArgumentParser(description='Create performance figure for BTBench evaluation')
parser.add_argument('--split_type', type=str, default='SS_SM', 
                    help='Split type to use (SS_SM or SS_DM or DS_DM)')
args = parser.parse_args()
split_type = args.split_type

BTBENCH_LITE_SUBJECT_TRIALS = [
    (1, 1), (1, 2), 
    (2, 0), (2, 4),
    (3, 0), (3, 1),
    (4, 0), (4, 1),
    (7, 0), (7, 1),
    (10, 0), (10, 1)
]

def create_performance_figure():
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
    task_name_mapping = {
        'onset': 'Sentence Onset',
        'speech': 'Speech',
        'volume': 'Voice Volume', 
        'pitch': 'Voice Pitch',
        'speaker': 'Speaker Identity',
        'delta_volume': 'Delta Volume',
        'delta_pitch': 'Delta Pitch',
        'gpt2_surprisal': 'GPT-2 Surprisal',
        'word_length': 'Word Length',
        'word_gap': 'Inter-word Gap',
        'word_index': 'Word Position',
        'word_head_pos': 'Head Word Position',
        'word_part_speech': 'Part of Speech',
        'frame_brightness': 'Frame Brightness',
        'global_flow': 'Global Optical Flow',
        'local_flow': 'Local Optical Flow',
        'global_flow_angle': 'Global Flow Angle',
        'local_flow_angle': 'Local Flow Angle',
        'face_num': 'Number of Faces',
    }

    # optional: filter tasks
    #tasks_filter = ['onset', 'speech', 'volume', 'pitch', 'speaker', 'global_flow', 'local_flow', 'global_flow_angle', 'local_flow_angle']
    #tasks_filter = ['onset', 'speech', 'volume', 'speaker']
    #task_list = {task: task_list[task] for task in tasks_filter}

    nperseg = 256
    
    models = ['Linear',
              'BrainBERT', 'PopT']
    
    popt_models = [model for model in models if model.startswith('PopT')]  # Get the last model which is POPT
    non_popt_models = [model for model in models if not model.startswith('PopT')]
    popt_csv_paths = [f'eval_results_popt/population_frozen_{split_type}_results.csv', f'eval_results_popt/popt_{split_type}_results.csv'][1:]

    # Define models
    # models = [f'Linear (spectrogram, {ms_per_seg}ms)', 
    #           'BrainBERT (granularity=1)', 'BrainBERT (granularity=4)', 'BrainBERT (granularity=16)', 'BrainBERT (granularity=-1)']
    
    subject_trials = BTBENCH_LITE_SUBJECT_TRIALS
    metric = 'AUROC' # 'AUROC'
    assert metric == 'AUROC', 'Metric must be AUROC; no other metric is supported'

    performance_data = {}
    for task in task_list.keys():
        performance_data[task] = {}
        for model in models:
            performance_data[task][model] = {}

    for model_idx, model in enumerate(non_popt_models): # Andrii's model format
        for task in task_list.keys():
            subject_trial_means = []
            for subject_id, trial_id in subject_trials:
                nperseg_suffix = f'_nperseg{nperseg}' if nperseg != 256 else ''
                if subject_id == 2 and split_type == 'DS_DM':
                    continue
                if model == ('Linear'):
                    if split_type != 'DS_DM':
                        filename = f'/om2/user/zaho/btbench/eval_results_lite_{split_type}/linear_remove_line_noise/population_btbank{subject_id}_{trial_id}_{task}.json'
                    else:
                        # Find the first file that matches the pattern
                        pattern = f'/om2/user/hmor/btbench/eval_results_ds_dt_lite_desikan_killiany/DS-DT-FixedTrain-Lite_{task}_test_S{subject_id}T{trial_id}_*.json'
                        matching_files = glob.glob(pattern)
                        if matching_files:
                            filename = matching_files[0]  # Take the first matching file
                            #print(f"Found file: {filename}")
                        else:
                            print(f"Warning: No matching file found for pattern {pattern}, skipping...")
                            continue
                elif model.startswith('Linear (<200Hz)'):
                    filename = f'/om2/user/zaho/btbench/eval_results_lite_{split_type}/linear_downsample_200/population_btbank{subject_id}_{trial_id}_{task}.json'
                elif model.startswith('Linear (<200Hz, -line noise)'):
                    filename = f'/om2/user/zaho/btbench/eval_results_lite_{split_type}/linear_downsample_200-remove_line_noise/population_btbank{subject_id}_{trial_id}_{task}.json'
                elif model.startswith('Linear (FFT - abs'):
                    filename = f'/om2/user/zaho/btbench/eval_results_lite_{split_type}/linear_fft_abs{nperseg_suffix}/population_btbank{subject_id}_{trial_id}_{task}.json'
                elif model.startswith('Linear (FFT - real+imag'):
                    filename = f'/om2/user/zaho/btbench/eval_results_lite_{split_type}/linear_fft_realimag{nperseg_suffix}/population_btbank{subject_id}_{trial_id}_{task}.json'
                elif model.startswith('BrainBERT'):
                    granularity = -1
                    filename = f'/om2/user/zaho/BrainBERT/eval_results_lite_{split_type}/brainbert_frozen_mean_granularity_{granularity}/population_btbank{subject_id}_{trial_id}_{task}.json'
                if not os.path.exists(filename):
                    print(f"Warning: File {filename} not found, skipping...")
                    continue

                with open(filename, 'r') as json_file:
                    data = json.load(json_file)
                if split_type == 'DS_DM' and model.startswith('Linear'): # Hara's results
                    data = data['final_auroc']
                    value = data
                else:
                    if 'one_second_after_onset' in data['evaluation_results'][f'btbank{subject_id}_{trial_id}']['population']:
                        data = data['evaluation_results'][f'btbank{subject_id}_{trial_id}']['population']['one_second_after_onset'] 
                    else:
                        data = data['evaluation_results'][f'btbank{subject_id}_{trial_id}']['population']['whole_window'] # for BrainBERT only
                    value = np.nanmean([fold_result['test_roc_auc'] for fold_result in data['folds']])
                subject_trial_means.append(value)
            subject_trial_means = [x for x in subject_trial_means if not np.isnan(x)]
            performance_data[task][model] = {
                'mean': np.mean(subject_trial_means),
                'sem': np.std(subject_trial_means) / np.sqrt(len(subject_trial_means))
            }

    import pandas as pd
    # Load and process data for the POPT model
    for popt_model_i, popt_model in enumerate(popt_models):
        popt_csv_path = popt_csv_paths[popt_model_i]
        if os.path.exists(popt_csv_path):
            # Read the CSV file
            popt_data = pd.read_csv(popt_csv_path)
            # Group by subject_id, trial_id, and task_name to calculate mean across folds
            for task in task_list.keys():
                subject_trial_means = []
                
                for subject_id, trial_id in subject_trials:
                    # Filter data for current subject, trial, and task
                    task_data = popt_data[(popt_data['subject_id'] == subject_id) & 
                                        (popt_data['trial_id'] == trial_id) & 
                                        ((popt_data['task_name'] == task) | (popt_data['task_name'] == task + '_frozen_True'))]

                    if subject_id == 2 and split_type == 'DS_DM':
                        continue
                    
                    if not task_data.empty:
                        # Calculate mean ROC AUC across folds
                        value = task_data['test_roc_auc'].mean()
                        subject_trial_means.append(value)
                    else:
                        print(f"Warning: No data found for subject {subject_id}, trial {trial_id}, task {task} in POPT results ({popt_csv_path})")
                
                if subject_trial_means:
                    performance_data[task][popt_model] = {
                        'mean': np.mean(subject_trial_means),
                        'sem': np.std(subject_trial_means) / np.sqrt(len(subject_trial_means))
                    }
                else:
                    performance_data[task][popt_model] = {
                        'mean': np.nan,
                        'sem': np.nan
                    }
        else:
            print(f"Warning: POPT results file {popt_csv_path} not found")
            # Set NaN values for POPT model if file doesn't exist
            for task in task_list.keys():
                performance_data[task][popt_model] = {
                    'mean': np.nan,
                    'sem': np.nan
                }

    # Create figure with 3 columns for the three split types
    fig, axs = plt.subplots(1, 3, figsize=(8, 3))

    # Color palette
    colors = sns.color_palette("husl", 4)

    # Model color palette using two different colormaps
    linear_models = [model for model in models if model.startswith('Linear')]
    brainbert_models = [model for model in models if model.startswith('BrainBERT')]
    popt_model = [model for model in models if model.startswith('PopT')]
    
    linear_colors = sns.color_palette("viridis", len(linear_models))
    brainbert_colors = sns.color_palette("plasma", len(brainbert_models))
    popt_colors = sns.color_palette("magma", len(popt_model)+1)[1:] # to remove PopT (frozen) since it doesnt exist here

    # Combine colors in the correct order
    model_colors = linear_colors + brainbert_colors + popt_colors

    # Bar width
    bar_width = 0.2
    
    # Split types to plot
    split_types = ['SS_SM', 'SS_DM', 'DS_DM']
    split_titles = {
        'SS_SM': 'Same Subject,\nSame Movie',
        'SS_DM': 'Same Subject,\nDifferent Movie',
        'DS_DM': 'Different Subject,\nDifferent Movie'
    }
    
    for ax_idx, curr_split_type in enumerate(split_types):
        # Skip recomputation if this is the current split type
        if curr_split_type == split_type:
            curr_performance_data = performance_data
        else:
            # Recalculate performance data for this split type
            curr_performance_data = {}
            for task in task_list.keys():
                curr_performance_data[task] = {}
                for model in models:
                    curr_performance_data[task][model] = {}

            for model_idx, model in enumerate(non_popt_models):
                for task in task_list.keys():
                    subject_trial_means = []
                    for subject_id, trial_id in subject_trials:
                        if subject_id == 2 and curr_split_type == 'DS_DM':
                            continue
                        nperseg_suffix = f'_nperseg{nperseg}' if nperseg != 256 else ''
                        if model == ('Linear'):
                            if curr_split_type != 'DS_DM':
                                filename = f'/om2/user/zaho/btbench/eval_results_lite_{curr_split_type}/linear_remove_line_noise/population_btbank{subject_id}_{trial_id}_{task}.json'
                            else:
                                # Find the first file that matches the pattern
                                pattern = f'/om2/user/hmor/btbench/eval_results_ds_dt_lite_desikan_killiany/DS-DT-FixedTrain-Lite_{task}_test_S{subject_id}T{trial_id}_*.json'
                                matching_files = glob.glob(pattern)
                                if matching_files:
                                    filename = matching_files[0]  # Take the first matching file
                                    #print(f"Found file: {filename}")
                                else:
                                    print(f"Warning: No matching file found for pattern {pattern}, skipping...")
                                    continue
                        elif model.startswith('Linear (-line noise)'):
                            filename = f'/om2/user/zaho/btbench/eval_results_lite_{curr_split_type}/linear_remove_line_noise/population_btbank{subject_id}_{trial_id}_{task}.json'
                        elif model.startswith('Linear (<200Hz)'):
                            filename = f'/om2/user/zaho/btbench/eval_results_lite_{curr_split_type}/linear_downsample_200/population_btbank{subject_id}_{trial_id}_{task}.json'
                        elif model.startswith('Linear (<200Hz, -line noise)'):
                            filename = f'/om2/user/zaho/btbench/eval_results_lite_{curr_split_type}/linear_downsample_200-remove_line_noise/population_btbank{subject_id}_{trial_id}_{task}.json'
                        elif model.startswith('Linear (FFT - abs'):
                            filename = f'/om2/user/zaho/btbench/eval_results_lite_{curr_split_type}/linear_fft_abs{nperseg_suffix}/population_btbank{subject_id}_{trial_id}_{task}.json'
                        elif model.startswith('Linear (FFT - real+imag'):
                            filename = f'/om2/user/zaho/btbench/eval_results_lite_{curr_split_type}/linear_fft_realimag{nperseg_suffix}/population_btbank{subject_id}_{trial_id}_{task}.json'
                        elif model.startswith('BrainBERT'):
                            granularity = -1
                            filename = f'/om2/user/zaho/BrainBERT/eval_results_lite_{curr_split_type}/brainbert_frozen_mean_granularity_{granularity}/population_btbank{subject_id}_{trial_id}_{task}.json'
                        if not os.path.exists(filename):
                            print(f"Warning: File {filename} not found, skipping...")
                            continue

                        with open(filename, 'r') as json_file:
                            data = json.load(json_file)
                        if curr_split_type == 'DS_DM' and model.startswith('Linear'): # Hara's results
                            data = data['final_auroc']
                            value = data
                        else:
                            if 'one_second_after_onset' in data['evaluation_results'][f'btbank{subject_id}_{trial_id}']['population']:
                                data = data['evaluation_results'][f'btbank{subject_id}_{trial_id}']['population']['one_second_after_onset'] 
                            else:
                                data = data['evaluation_results'][f'btbank{subject_id}_{trial_id}']['population']['whole_window'] # for BrainBERT only
                            value = np.nanmean([fold_result['test_roc_auc'] for fold_result in data['folds']])
                        subject_trial_means.append(value)
                    curr_performance_data[task][model] = {
                        'mean': np.mean(subject_trial_means) if subject_trial_means else np.nan,
                        'sem': np.std(subject_trial_means) / np.sqrt(len(subject_trial_means)) if subject_trial_means else np.nan
                    }

            # Load and process data for the POPT models for this split type
            for popt_model_i, popt_model in enumerate(popt_models):
                curr_popt_csv_path = f'eval_results_popt/{"population_frozen" if "frozen" in popt_model.lower() else "popt"}_{curr_split_type}_results.csv'
                if os.path.exists(curr_popt_csv_path):
                    # Read the CSV file
                    popt_data = pd.read_csv(curr_popt_csv_path)
                    # Group by subject_id, trial_id, and task_name to calculate mean across folds
                    for task in task_list.keys():
                        subject_trial_means = []
                        
                        for subject_id, trial_id in subject_trials:
                            # Filter data for current subject, trial, and task
                            task_data = popt_data[(popt_data['subject_id'] == subject_id) & 
                                                (popt_data['trial_id'] == trial_id) & 
                                                ((popt_data['task_name'] == task) | (popt_data['task_name'] == task + '_frozen_True'))]
                            
                            if subject_id == 2 and curr_split_type == 'DS_DM':
                                continue
                            
                            if not task_data.empty:
                                # Calculate mean ROC AUC across folds
                                value = task_data['test_roc_auc'].mean()
                                subject_trial_means.append(value)
                            else:
                                print(f"Warning: No data found for subject {subject_id}, trial {trial_id}, task {task} in POPT results ({curr_popt_csv_path})")
                        
                        if subject_trial_means:
                            curr_performance_data[task][popt_model] = {
                                'mean': np.mean(subject_trial_means),
                                'sem': np.std(subject_trial_means) / np.sqrt(len(subject_trial_means))
                            }
                        else:
                            curr_performance_data[task][popt_model] = {
                                'mean': np.nan,
                                'sem': np.nan
                            }
                else:
                    print(f"Warning: POPT results file {curr_popt_csv_path} not found")
                    # Set NaN values for POPT model if file doesn't exist
                    for task in task_list.keys():
                        curr_performance_data[task][popt_model] = {
                            'mean': np.nan,
                            'sem': np.nan
                        }
        
        # Calculate overall performance for each model for this split type
        overall_performance = {}
        for model in models:
            means = [curr_performance_data[task][model]['mean'] for task in task_list.keys()]
            sems = [curr_performance_data[task][model]['sem'] for task in task_list.keys()]
            overall_performance[model] = {
                'mean': np.nanmean(means),
                'sem': np.sqrt(np.nansum(np.array([s**2 for s in sems if not np.isnan(s)]))) / np.sum(~np.isnan(np.array(sems)))  # Combined SEM on non-NaN values
            }
        
        # Print performance metrics for this split type
        print(f"\nPerformance for {split_titles[curr_split_type]}:")
        for model in models:
            perf = overall_performance[model]
            print(f"{model}: {perf['mean']:.3f} ± {perf['sem']:.3f}")
        
        # Plot overall performance in the current axis
        ax = axs[ax_idx]
        for i, model in enumerate(models):
            perf = overall_performance[model]
            ax.bar(i * bar_width, perf['mean'], bar_width,
                  yerr=perf['sem'],
                  color=model_colors[i],
                  capsize=6)
        
        ax.set_title(split_titles[curr_split_type], fontsize=14, pad=10)
        if metric == 'accuracy':
            ax.set_ylim(0.2, 1.0)
        else:
            ax.set_ylim(0.49, .6)
            ax.set_yticks([0.5, 0.55, 0.6])
        ax.set_xticks([])
        if ax_idx == 0:  # Only add y-label to the first axis
            ax.set_ylabel(metric, fontsize=14)
        ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='y', labelsize=12)

    # Create a proxy artist for the chance line with the correct style
    chance_line = plt.Line2D([0], [0], color='black', linestyle='--', alpha=0.5)

    # Add legend at the bottom with custom handles
    handles = [plt.Rectangle((0,0),1,1, color=model_colors[i]) for i in range(len(models))]
    handles.append(chance_line)
    fig.legend(handles, models + ["Chance"],
              loc='lower center', 
              bbox_to_anchor=(0.5, 0.02),
              ncol=4,
              frameon=False,
              fontsize=12)

    # Adjust layout with space at the bottom for legend
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    
    # Add padding to the left and right of each axis
    for ax in axs:
        box = ax.get_position()
        width = box.width * 0.8  # Reduce width to 80% of original
        ax.set_position([box.x0 + (box.width - width)/2, box.y0, width, box.height])
    
    # Save figure
    plt.savefig(f'figures/eval_lite_comparison_all_splits{nperseg_suffix}.pdf', dpi=300, bbox_inches='tight')
    print(f'Saved figure to figures/eval_lite_comparison_all_splits{nperseg_suffix}.pdf')
    plt.close()

if __name__ == "__main__":
    create_performance_figure()