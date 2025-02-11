# 📋 Evaluation Submissions

To submit a model evaluation, follow these steps:

1️⃣ **Fork this repository.**  
2️⃣ **Clone your fork** to your local machine.  
3️⃣ **Navigate to the `evaluation/` directory.**  
4️⃣ **Inside the relevant task folder (`evaluation/<task_name>/`), create a new JSON 
file** following this format:

```
evaluation/
├── sentence-onset/
│   ├── 20250211_modelA.json
├── yelling-detection/
│   ├── 20250212_modelB.json
├── name-detection/
├── pitch/
├── rms/
```

5️⃣ **File Format Example (JSON)**  
Each submission must be a single JSON file named `YYYYMMDD_modelX.json`, structured 
like this:

```json
{
  "name": "Decoder A",
  "rocAuc": 0.95,
  "accuracy": 95,
  "org": "MIT",
  "date": "2025-02-11"
}
```

6️⃣ **Open a Pull Request (PR) to submit your results.**  
7️⃣ Your submission will be reviewed and added to the leaderboard.

🔹 **For more details, check `checklist.md`.**

