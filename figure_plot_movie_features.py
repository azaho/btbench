import h5py
import os
import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import signal, stats
from matplotlib.patches import Patch
from btbench_config import ROOT_DIR
import matplotlib.font_manager as fm
font_path = 'font_arial.ttf'
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams.update({'font.size': 12})

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_and_save_features(features_df, title, output_dir):
    """Plot and save individual feature plots"""
    
    # Get the actual time range
    time_min = features_df['start'].min()
    time_max = features_df['start'].max()
    
    # 1. Time series plots
    plt.figure(figsize=(15, 6))
    plt.plot(features_df['start'], features_df['pitch'], label='Pitch', alpha=0.3, color='lightblue')
    plt.plot(features_df['pitch_ma'], label='Pitch (Moving Avg)', alpha=1, linewidth=2, color='blue')
    plt.title(f'Pitch Over Time - {title}')
    plt.ylabel('Pitch (Hz)')
    plt.xlabel('Time (seconds)')
    plt.xlim(time_min, time_max)  # Set proper x-axis limits
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'pitch_over_time.png'))
    plt.close()

    plt.figure(figsize=(15, 6))
    
    # Debug print to check data
    print(f"Time range: {features_df['start'].min()} to {features_df['start'].max()}")
    print(f"Number of valid RMS values: {features_df['rms'].notna().sum()}")
    print(f"Number of values after 6000s: {len(features_df[features_df['start'] > 6000])}")
    
    # Plot raw RMS values first
    valid_mask = features_df['rms'].notna()
    plt.plot(features_df.loc[valid_mask, 'start'], 
             features_df.loc[valid_mask, 'rms'], 
             label='RMS', alpha=0.3, color='lightgreen')
    
    # Plot moving average
    valid_ma_mask = features_df['rms_ma'].notna()
    plt.plot(features_df.loc[valid_ma_mask, 'start'], 
             features_df.loc[valid_ma_mask, 'rms_ma'], 
             label='RMS (Moving Avg)', alpha=1, linewidth=2, color='green')
    
    plt.title(f'RMS Over Time - {title}')
    plt.ylabel('RMS')
    plt.xlabel('Time (seconds)')
    plt.xlim(time_min, time_max)  # Set proper x-axis limits
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'rms_over_time.png'))
    plt.close()

    # 2. Speech density plot
    plt.figure(figsize=(15, 6))
    sentence_starts = features_df[features_df['idx_in_sentence'] == 0]['start']
    kernel = stats.gaussian_kde(sentence_starts)
    x_eval = np.linspace(features_df['start'].min(), features_df['start'].max(), 1000)
    plt.plot(x_eval, kernel(x_eval), color='red', label='Speech Density')
    plt.fill_between(x_eval, kernel(x_eval), alpha=0.3, color='red')
    plt.title(f'Speech Density - {title}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Speech Density')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'speech_density.png'))
    plt.close()

    # 3. Correlation plot
    plt.figure(figsize=(8, 8))
    scatter_data = pd.DataFrame({
        'Pitch': features_df['pitch'],
        'RMS': features_df['rms']
    })
    sns.scatterplot(data=scatter_data, x='RMS', y='Pitch', alpha=0.1)
    corr = features_df['pitch'].corr(features_df['rms'])
    plt.title(f'Pitch vs RMS Correlation\n{corr:.3f}')
    plt.savefig(os.path.join(output_dir, 'pitch_rms_correlation.png'))
    plt.close()

    # 4. Loud/Quiet sections analysis
    plt.figure(figsize=(15, 6))
    
    # Calculate thresholds using quartiles
    quiet_threshold = features_df['rms'].quantile(0.25)  # 25th percentile
    loud_threshold = features_df['rms'].quantile(0.75)   # 75th percentile
    median_rms = features_df['rms'].median()
    
    plt.plot(features_df['start'], features_df['rms'], color='gray', alpha=0.7, label='RMS')
    
    # Highlight loud/quiet sections using quartile thresholds
    loud_mask = features_df['rms'] > loud_threshold
    quiet_mask = features_df['rms'] < quiet_threshold
    
    plt.fill_between(features_df['start'], features_df['rms'], 
                     where=loud_mask, color='red', alpha=0.3, label='Loud Sections (>75th percentile)')
    plt.fill_between(features_df['start'], features_df['rms'], 
                     where=quiet_mask, color='blue', alpha=0.3, label='Quiet Sections (<25th percentile)')
    
    plt.axhline(y=median_rms, color='black', linestyle='--', alpha=0.5, label='Median RMS')
    plt.axhline(y=loud_threshold, color='red', linestyle='--', alpha=0.3)
    plt.axhline(y=quiet_threshold, color='blue', linestyle='--', alpha=0.3)
    
    plt.title(f'Loud and Quiet Sections - {title}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('RMS')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'loud_quiet_sections.png'))
    plt.close()

def plot_movie_features(movie_id, transcript_file, title):
    """Plot key movie features and events"""
    # Load and clean data
    features_df = pd.read_csv(transcript_file)
    features_df = features_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['start', 'pitch', 'rms'])
    
    # Create output directory
    output_dir = os.path.join('movie_features', movie_id)
    ensure_dir(output_dir)
    
    # 1. Audio Features Plot (Pitch and RMS)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # Pitch subplot - raw values only
    ax1.plot(features_df['start'], features_df['pitch'], color='blue', alpha=0.7, label='Pitch')
    ax1.set_ylabel('Pitch (Hz)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # RMS subplot - raw values only
    ax2.plot(features_df['start'], features_df['rms'], color='green', alpha=0.7, label='RMS')
    ax2.set_ylabel('RMS')
    ax2.set_xlabel('Time (seconds)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Audio Features Over Time - {title}')
    plt.savefig(os.path.join(output_dir, '1_audio_features.png'))
    plt.close()
    
    # 2. Loud/Quiet Sections Analysis using raw RMS
    plt.figure(figsize=(15, 6))
    
    # Calculate thresholds using quartiles
    quiet_threshold = features_df['rms'].quantile(0.25)  # 25th percentile
    loud_threshold = features_df['rms'].quantile(0.75)   # 75th percentile
    median_rms = features_df['rms'].median()
    
    plt.plot(features_df['start'], features_df['rms'], color='gray', alpha=0.7, label='RMS')
    
    # Highlight loud/quiet sections using quartile thresholds
    loud_mask = features_df['rms'] > loud_threshold
    quiet_mask = features_df['rms'] < quiet_threshold
    
    plt.fill_between(features_df['start'], features_df['rms'], 
                     where=loud_mask, color='red', alpha=0.3, label='Loud Sections (>75th percentile)')
    plt.fill_between(features_df['start'], features_df['rms'], 
                     where=quiet_mask, color='blue', alpha=0.3, label='Quiet Sections (<25th percentile)')
    
    plt.axhline(y=median_rms, color='black', linestyle='--', alpha=0.5, label='Median RMS')
    plt.axhline(y=loud_threshold, color='red', linestyle='--', alpha=0.3)
    plt.axhline(y=quiet_threshold, color='blue', linestyle='--', alpha=0.3)
    
    plt.title(f'Loud and Quiet Sections - {title}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('RMS')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, '2_loud_quiet_sections.png'))
    plt.close()
    
    # Update analysis summary with times and RMS values
    summary = {
        'title': title,
        'duration_seconds': features_df['end'].max(),
        'total_words': len(features_df),
        'median_rms': float(median_rms),
        'quiet_threshold': float(quiet_threshold),
        'loud_threshold': float(loud_threshold),
        'loud_sections_percentage': float(loud_mask.mean() * 100),
        'quiet_sections_percentage': float(quiet_mask.mean() * 100),
        'times': features_df['start'].tolist(),  # Add time points
        'rms': features_df['rms'].tolist()       # Add RMS values
    }
    
    with open(os.path.join(output_dir, 'analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

def plot_all_movies_volume(movies_dir='movie_features'):
    """Create a publication figure showing loud/quiet sections for all movies"""
    # Set up the figure with a smaller size
    fig = plt.figure(figsize=(12, 14))
    plt.style.use('seaborn-paper')
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12
    
    # Get all movie directories
    movie_dirs = [d for d in os.listdir(movies_dir) 
                 if os.path.isdir(os.path.join(movies_dir, d))]
    movie_dirs.sort()
    n_movies = len(movie_dirs)
    n_rows = (n_movies + 2) // 3
    
    # Create GridSpec with adjusted spacing
    gs = plt.GridSpec(n_rows * 3, 3,  # Use 3x the rows to add space between movie pairs
                     height_ratios=[0.4, 0.1, 0.5] * n_rows,  # [volume plot, color plot, spacing] for each movie
                     hspace=0.0,   # No space between paired plots
                     wspace=0.3)   # Space between columns
    
    for idx, movie in enumerate(movie_dirs):
        # Calculate row and column position
        base_row = (idx // 3) * 3  # Multiply by 3 because each movie takes 3 rows
        col = idx % 3
        
        # Load movie data
        with open(os.path.join(movies_dir, movie, 'analysis_summary.json'), 'r') as f:
            data = json.load(f)
        
        # Create two subplots for each movie
        ax1 = plt.subplot(gs[base_row, col])     # Volume line plot
        ax2 = plt.subplot(gs[base_row+1, col])   # Loud/quiet sections
        
        # Get data
        times = np.array(data['times'])
        rms = np.array(data['rms'])
        
        # Top plot: Volume line
        ax1.plot(times/60, rms, color='black', alpha=0.7, linewidth=0.5)
        movie_title = movie.replace('-', ' ').title()
        movie_duration = f"{data['duration_seconds']/60:.0f}"
        
        # Place title at the top of the volume plot
        ax1.set_title(f"{movie_title} - {movie_duration} min", 
                     pad=5)
        
        ax1.set_ylim(0, rms.max() * 1.1)
        ax1.set_xlim(0, data['duration_seconds']/60)
        
        # Bottom plot: Loud/quiet sections
        loud_mask = rms >= data['loud_threshold']
        quiet_mask = rms <= data['quiet_threshold']
        
        ax2.fill_between(times/60, 0, 1, 
                        where=loud_mask,
                        color='red', alpha=0.3)
        ax2.fill_between(times/60, 0, 1, 
                        where=quiet_mask,
                        color='blue', alpha=0.3)
        
        # Customize appearance
        ax2.set_ylim(0, 1)
        ax2.set_xlim(0, data['duration_seconds']/60)
        
        # Remove all axis elements from volume plot
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.grid(False)
        
        # Only show x-axis on bottom plot for last row
        if idx >= len(movie_dirs)-3:
            ax2.set_xlabel('Time (minutes)')
        else:
            ax2.set_xticks([])
        
        # Remove y-axis ticks
        ax2.set_yticks([])
        
        # Remove spines from both plots
        for spine in ax1.spines.values():
            spine.set_visible(False)
        for spine in ax2.spines.values():
            spine.set_visible(False)
            
        # Only add grid to bottom plot
        ax2.grid(True, alpha=0.2)
    
    # Move legend to bottom
    legend_elements = [
        plt.Line2D([0], [0], color='black', alpha=0.7, linewidth=0.5, label='Volume'),
        Patch(facecolor='red', alpha=0.3, label='Loud'),
        Patch(facecolor='blue', alpha=0.3, label='Quiet')
    ]
    fig.legend(handles=legend_elements, 
              loc='lower center',
              bbox_to_anchor=(0.5, 0.02),
              ncol=3)
    
    # Adjust layout
    plt.subplots_adjust(top=0.95, bottom=0.08)
    
    plt.savefig('figures/movie_volumes_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = ROOT_DIR
    metadata_dir = os.path.join(root_dir, "subject_metadata")
    
    # Find all metadata files
    movies_to_analyze = []
    for filename in os.listdir(metadata_dir):
        if filename.startswith('sub_') and filename.endswith('_metadata.json'):
            # Parse subject and trial IDs from filename
            # Format: sub_X_trialYYY_metadata.json
            parts = filename.split('_')
            sub_id = int(parts[1])
            trial_id = int(parts[2].replace('trial', ''))
            
            # Load metadata to get title
            with open(os.path.join(metadata_dir, filename), 'r') as f:
                meta_dict = json.load(f)
                movies_to_analyze.append({
                    'sub_id': sub_id,
                    'trial_id': trial_id,
                    'expected_title': meta_dict['title']
                })
    
    print(f"Found {len(movies_to_analyze)} movies to analyze")
    
    # Process each movie
    for movie in movies_to_analyze:
        print(f"\nProcessing {movie['expected_title']}...")
        
        # Load metadata to get movie info
        metadata_file = os.path.join(root_dir, 
                                   f'subject_metadata/sub_{movie["sub_id"]}_trial{movie["trial_id"]:03}_metadata.json')
        print(f"Looking for metadata file at: {metadata_file}")
        
        try:
            with open(metadata_file, 'r') as f:
                meta_dict = json.load(f)
                title = meta_dict['title']
                movie_id = meta_dict['filename']
                
            # Get transcript file path
            transcript_file = os.path.join(root_dir, f'transcripts/{movie_id}/features.csv')
            print(f"Looking for transcript file at: {transcript_file}")
            
            # Create plots
            plot_movie_features(movie_id, transcript_file, title)
            print(f"Successfully processed {title}")
            
        except FileNotFoundError as e:
            print(f"Error: Could not find file - {e}")
        except Exception as e:
            print(f"Error processing {movie['expected_title']}: {e}")

    plot_all_movies_volume() 