import re

def split_log_sections(filepath):
    """
    Reads a text file and splits its content into sections based on separator lines (e.g., '###').

    Args:
        filepath (str): Path to the log/text file.

    Returns:
        List[str]: A list where each element is a section of the file as a string.
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()

    # Split by the separator line
    sections = content.split('#' * 73)  # 50 or more #s
    # Remove leading/trailing whitespace and filter empty strings
    sections = [section.strip() for section in sections if section.strip()]

    #find numbers from each section
    all_numbers = []

    for section in sections:
        # Find all numbers including decimals and negative values
        numbers = re.findall(r'-?\d+\.\d+|-?\d+', section)
        # Convert to float
        all_numbers.append([float(num) for num in numbers])

    
    #runs are all sections except for first one
    runs = all_numbers[1:]

    #from each run remove 1 from f-score (garbage from regex)
    for run in runs:
        run.pop(4)
    
    #get ten runs for each classifier
    model_metrics = {}

    model_metrics["Euclidean_MDC"] = runs[:10]
    model_metrics["Mahalanobis MDC"] = runs[10:20]
    model_metrics["LDA Fisher MDC"] = runs [20:30]
    model_metrics["Bayesian"] = runs[30:40]
    model_metrics["KNN"] = runs[40:50]
    model_metrics["SVM"] = runs[50:60]

    for model_name, metrics in model_metrics.items():
        accuracies = [run[0] for run in metrics]
        specificities = [run[2] for run in metrics]
        f1_scores = [run[4] for run in metrics]
        sensitivities = [run[6] for run in metrics]

        print(model_name)
        print(f"Average accuracy: {sum(accuracies)/len(accuracies)}")
        print(f"Average specificity: {sum(specificities)/len(specificities)}")
        print(f"Average f1-score: {sum(f1_scores)/len(f1_scores)}")
        print(f"Average sensitivities: {sum(sensitivities)/len(sensitivities)}")
        print("#######################################\n")


split_log_sections("logs/KW_Natural.log")