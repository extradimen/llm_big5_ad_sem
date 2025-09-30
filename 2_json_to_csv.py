# This is the fixed code that can be directly copied into your notebook

import os
import json
import csv
import re

# Automatically scan the json_output directory
JSON_OUTPUT_BASE_DIR = "./json_output"
CSV_OUTPUT_BASE_DIR = "./csv_output"

# Likert mapping
LIKERT_MAP = {
    "Strongly Disagree": 1,
    "Disagree": 2,
    "Neutral": 3,
    "Agree": 4,
    "Strongly Agree": 5
}

# Reverse-scored item indexes (3rd item in each group)
REVERSE_INDEXES = [2, 5, 8, 11, 14]

# BFI standard labels mapped to questions (order must correspond)
BFI_LABELS = [
    "BFI_11", "BFI_12", "BFI_13",
    "BFI_21", "BFI_22", "BFI_23",
    "BFI_31", "BFI_32", "BFI_33",
    "BFI_41", "BFI_42", "BFI_43",
    "BFI_51", "BFI_52", "BFI_53"
]

BFI_QUESTIONS = [
    ("Extraversion", "I am talkative."),
    ("Extraversion", "I am outgoing, sociable."),
    ("Extraversion", "I am reserved."),  # reverse
    ("Agreeableness", "I am helpful and unselfish with others."),
    ("Agreeableness", "I am generally trusting."),
    ("Agreeableness", "I am sometimes rude to others."),  # reverse
    ("Conscientiousness", "I do a thorough job."),
    ("Conscientiousness", "I make plans and follow through with them."),
    ("Conscientiousness", "I tend to be careless."),  # reverse
    ("Neuroticism", "I get nervous easily."),
    ("Neuroticism", "I worry a lot."),
    ("Neuroticism", "I am emotionally stable."),  # reverse
    ("Openness", "I have an active imagination."),
    ("Openness", "I am original, come up with new ideas."),
    ("Openness", "I have few artistic interests.")  # reverse
]

def extract_model_info_from_folder(folder_name):
    """Extract model info and run index from folder name"""
    # Match format: experiment_outputs_{model}_{run_index}
    match = re.match(r"experiment_outputs_(.+)_(\d+)", folder_name)
    if match:
        model_name = match.group(1)
        run_index = int(match.group(2))
        return model_name, run_index
    return None, None

# Auto traverse json_output directory
if not os.path.exists(JSON_OUTPUT_BASE_DIR):
    print(f"❌ Directory not found: {JSON_OUTPUT_BASE_DIR}")
    print("Please make sure you have run the JSON parsing step first.")
else:
    # Collect all experiment folders
    experiment_folders = [f for f in os.listdir(JSON_OUTPUT_BASE_DIR) 
                         if os.path.isdir(os.path.join(JSON_OUTPUT_BASE_DIR, f)) 
                         and f.startswith("experiment_outputs_")]
    
    if not experiment_folders:
        print(f"❌ No experiment folders found in {JSON_OUTPUT_BASE_DIR}")
    else:
        print(f"🔍 Found {len(experiment_folders)} experiment folders")
        
        for folder_name in sorted(experiment_folders):
            model_name, run_index = extract_model_info_from_folder(folder_name)
            
            if model_name is None:
                print(f"⚠️ Cannot parse folder name: {folder_name}")
                continue
                
            folder_path = os.path.join(JSON_OUTPUT_BASE_DIR, folder_name)
            
            # Create CSV output directory
            os.makedirs(CSV_OUTPUT_BASE_DIR, exist_ok=True)
            output_csv = os.path.join(CSV_OUTPUT_BASE_DIR, f"{model_name}_{run_index}.csv")

            print(f"📁 Processing folder: {folder_name} → {output_csv}")

            data_rows = []
            skipped_files = []

            for file in os.listdir(folder_path):
                if (file.endswith(".json") 
                    and not file.startswith(".") 
                    and "-checkpoint" not in file):
                    filepath = os.path.join(folder_path, file)
                    try:
                        with open(filepath, "r") as f:
                            data = json.load(f)

                        flat = {
                            "sample_id": data.get("sample_id"),
                            "ad_type": data.get("ad_type")
                        }
                        # Auto-expand profile fields
                        profile = data.get("profile", {})
                        for key, value in profile.items():
                            # Normalize column name: spaces and special chars -> underscore
                            col_name = key.strip().lower().replace(" ", "_").replace("-", "_")
                            flat[col_name] = value
                        # Map BFI questions to standard labels + Likert mapping + reverse-scoring
                        for i, (_, q) in enumerate(BFI_QUESTIONS):
                            label = BFI_LABELS[i]
                            val = data["traits"].get(q, "")
                            numeric = LIKERT_MAP.get(val, "")
                            if numeric != "":
                                if i in REVERSE_INDEXES:
                                    numeric = 6 - numeric  # reverse scoring
                            flat[label] = numeric

                        # Extract ad attitude scores - keep only first 4
                        ad_scores = data.get("ad_attitude_scores", [])
                        for i, score in enumerate(ad_scores[:4], 1):  # only first 4
                            flat[f"ad_att_{i}"] = score

                        # Handle misspelled field name - keep only first 3
                        intent_scores = data.get("purchase_intention_scores") or data.get("purchase_intension_scores") or []
                        for i, score in enumerate(intent_scores[:3], 1):  # only first 3
                            flat[f"intent_{i}"] = score

                        data_rows.append(flat)

                    except Exception as e:
                        skipped_files.append((file, str(e)))
                        print(f"⚠️ Skipped {file}: {e}")

            # Write CSV - note indentation, this should be inside the for-loop
            if data_rows:
                fieldnames = ["sample_id", "ad_type", "gender", "age", "education", "occupation", "marital_status", "monthly_income", "region"] + BFI_LABELS + \
                             [f"ad_att_{i}" for i in range(1, 5)] + \
                             [f"intent_{i}" for i in range(1, 4)]

                data_rows.sort(key=lambda x: (x["sample_id"], x["ad_type"]))

                with open(output_csv, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(data_rows)

                print(f"✅ CSV saved: {output_csv} with {len(data_rows)} records.")
            else:
                print(f"⚠️ No valid data in {folder_name}")
