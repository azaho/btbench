# ✅ Submission Checklist

Before submitting your model evaluation, ensure you have included the following 
required files in your submission directory:

### **1️⃣ Required Files**
- [ ] `all_preds.json` → Contains the model's predictions in the required format.
- [ ] `metadata.yaml` → Includes details about the model, author, task, and dataset.
- [ ] `report.json` → Summary of evaluation outcomes.
- [ ] `logs/` → Directory containing evaluation logs.
- [ ] `trajs/` → (Optional) Directory with reasoning traces.

### **2️⃣ File Format**
#### **all_preds.json**
Example format:
```json
{
  "predictions": [
    {"input": "The dog ran.", "prediction": "sentence_start"},
    {"input": "It was raining.", "prediction": "not_start"}
  ]
}

