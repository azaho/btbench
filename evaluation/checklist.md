# ✅ Submission Checklist

Before submitting your model evaluation, ensure you have completed the following 
steps:

### 📁 Directory Structure  
☑️ Placed your JSON file inside the correct task folder under `evaluation/`.  
☑️ Named your file correctly using the format: `YYYYMMDD_modelX.json`.  

**Example:**
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

### 📜 JSON File Format  
☑️ Your JSON file follows this structure:

```json
{
  "name": "Your Model Name",
  "rocAuc": 0.90,
  "accuracy": 92,
  "org": "Your Organization",
  "date": "YYYY-MM-DD"
}
```

### 🔄 Final Checks  
☑️ Verified that your model name is unique and descriptive.  
☑️ Ensured the ROC AUC and accuracy values are correctly computed.  
☑️ Checked that the date format is `YYYY-MM-DD`.  

### 📤 Submission  
☑️ Opened a **Pull Request (PR)** to submit your results.  

---

⚠️ **Note:** Incomplete or incorrectly formatted submissions may not be accepted.  
For additional help, refer to `README.md` or open an issue. 🚀

