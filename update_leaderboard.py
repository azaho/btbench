import json
import os

# Paths
LEADERBOARD_JSON_PATH = "leaderboard/public/leaderboard_data.json"
BTBENCH_EVALUATION_PATH = "evaluation/"

def load_leaderboard():
    """Load the existing leaderboard data."""
    if not os.path.exists(LEADERBOARD_JSON_PATH):
        return {"tasks": [], "sentence-onset": [], "yelling-detection": [], "name-detection": [], "pitch": [], "rms": []}

    with open(LEADERBOARD_JSON_PATH, "r") as f:
        return json.load(f)

def save_leaderboard(data):
    """Save the updated leaderboard data."""
    with open(LEADERBOARD_JSON_PATH, "w") as f:
        json.dump(data, f, indent=2)

def is_duplicate(existing_entries, new_entry):
    """Check if new_entry already exists in the leaderboard."""
    for entry in existing_entries:
        if all(entry[key] == new_entry[key] for key in ["name", "rocAuc", "accuracy", "org", "date"]):
            return True
    return False

def update_leaderboard():
    """Read submissions from btbench/evaluation and update leaderboard_data.json."""
    leaderboard = load_leaderboard()
    

    
    for task in leaderboard["tasks"]:  # Iterate through defined tasks
        task_dir = os.path.join(BTBENCH_EVALUATION_PATH, task)
        print('-'*80)
        print(task_dir)
        
        if not os.path.isdir(task_dir):  # Skip if task folder doesn't exist
            continue

        for filename in os.listdir(task_dir):  # Iterate through JSON files
            print(filename)
            if filename.endswith(".json"):
                file_path = os.path.join(task_dir, filename)

                with open(file_path, "r") as f:
                    new_submission = json.load(f)  # Load submission
                    print(new_submission)
                    print('-'*80)

                # Append only if it is not already in the leaderboard
                if not is_duplicate(leaderboard[task], new_submission):
                    leaderboard[task].append(new_submission)

    save_leaderboard(leaderboard)  # Save the updated leaderboard
    

if __name__ == "__main__":
    update_leaderboard()
    
