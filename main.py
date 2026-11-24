# Main script. Loads data and prompts user for model choice, then trains and tests the model from another script.
from typing import Dict

from gaussian import main as bayesian

from itertools import islice
import random
import csv
import numpy as np

# Hyperparameters
DATA_FILE = "student_grade_ses_dataset.csv"     # Filepath to csv of raw data
DATA_SIZE = 1000                                # Total number of records to use, split between training and testing.
DATA_SIZE_FULL = True                           # Boolean; if true, above value is ignored and full dataset is used.
TEST_RATIO = 0.80                               # Percentage of records to be put into testing. 0.5 is 50%.

# Features to be ignored, whether they are irrelevant dates, redundant features, or not meant to be known at prediction time.
IGNORE = {"student_id", "academic_year", "survey_date", "family_income_bracket", "final_grade_pct", "grade_class"}

# Definitions for non-numerical features; ranked means values are sequential (i.e. scholarship is none->partial->full)
ENUMS_UNRANKED = {
    "gender": ["Male", "Female", "Other"],
    "school_type": ["Government", "Government-Aided", "Private-Unaided"],
    "residence_type": ["Urban", "Semi-Urban", "Rural"],
    "father_occupation": ["Unemployed", "Skilled Labor / Trade", "Professional", "Clerical / Service",
                          "Business / Self-Employed", "Managerial / Executive", "Unskilled Labor"],
    "mother_occupation": ["Homemaker", "Unemployed", "Skilled Labor / Trade", "Professional", "Clerical / Service",
                          "Business / Self-Employed", "Managerial / Executive", "Unskilled Labor"]
}
ENUMS_RANKED = {
    "father_education_level": {
        "No Formal Education": 0,
        "Primary": 1,
        "Secondary": 2,
        "Higher Secondary": 3,
        "Bachelor's Degree": 4,
        "Master's Degree": 5,
        "Doctorate": 6
    },
    "mother_education_level": {
        "No Formal Education": 0,
        "Primary": 1,
        "Secondary": 2,
        "Higher Secondary": 3,
        "Bachelor's Degree": 4,
        "Master's Degree": 5,
        "Doctorate": 6
    },
    "scholarship_status": {
        "No Scholarship": 0,
        "Partial": 1,
        "Full": 2
    }}


# Randomize row order, then place first 75% of rows into training data and last 25% as test data.
def split(data):
    random.shuffle(data)
    cut = int(len(data) * (1-TEST_RATIO))
    trainData = data[:cut]
    testData = data[cut:]
    return trainData, testData

# Process non-numerical data to be represented numerically, whether it be boolean (unranked) or integer (ranked)
# AI DISCLOSURE: This function was heavily AI-assisted
def convert(data):
    for row in data:
        for key in list(row.keys()):
            # Replace missing values with None and convert to int/float
            if row[key] == '': row[key] = None
            else:
                try: row[key] = float(row[key])
                except ValueError: pass

            # Remove ignored columns
            if key in IGNORE:
                del row[key]

            # Convert ranked enums
            elif key in ENUMS_RANKED:
                if row[key] is not None:
                    row[key] = ENUMS_RANKED[key][row[key]]

            # Convert unranked enums via onehot
            elif key in ENUMS_UNRANKED:
                original_value = row[key]
                del row[key]

                for category in ENUMS_UNRANKED[key]:
                    row[f"{key}_{category}"] = int(original_value == category)
    return data

# Z-Standardize all values
def normalize(data):
    medians = {}
    iqrs = {}

    for key in data[0].keys():
        values = [row[key] for row in data if row[key] is not None]
        medians[key] = np.median(values)
        iqrs[key] = np.percentile(values, 75) - np.percentile(values, 25)

    for row in data:
        for key in row.keys():
            if key != "at_risk_flag":
                if row[key] is not None:
                    if iqrs[key] != 0:
                        row[key] = (row[key] - medians[key]) / iqrs[key]
                    else:
                        row[key] = 0.0

    return data

# Loads data, processes and normalizes it, then splits into train/test.
# The same train/test split is used across all models during one run of main.py for better comparison.
def loadDataset():
    with open(DATA_FILE, newline="") as f:
        reader = csv.DictReader(f)
        if not DATA_SIZE_FULL: allData = list(islice(reader, DATA_SIZE))       # List of dictionaries
        else: allData = list(reader)
        allData = convert(allData)
        normData = normalize(allData)
        trainData, testData = split(allData)
        normTrainData, normTestData = split(normData)
        return trainData, testData, normTrainData, normTestData

def main():
    print("Loading dataset...")
    trainData, testData, nTrainData, nTestData = loadDataset()
    print("Loaded.")
    while True:
        print("Choose model to train and test: ")
        print("1. Gaussian Naive Bayes Classifier")
        print("2. Adaboost (Stump Forest)")
        print("3. Gradient Boosting (Forest)")
        print("4. Schapire Boosting (Forest)")
        print("5. Quit")
        i = input("Type selection and press enter: ")

        match i.strip().lower():
            case "1":
                bayesian(testData, trainData)
            case "2":
                print("Not yet implemented")
            case "3":
                print("Not yet implemented")
            case "4":
                print("Not yet implemented")
            case "5" | "q" | "quit":
                return
            case _:
                print("Input not understood, choose a number 1-5")

main()