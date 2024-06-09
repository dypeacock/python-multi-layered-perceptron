import pandas as pd
from sklearn.preprocessing import StandardScaler

def standardize_data(input_file, output_file):
    # Read data from Excel file
    data = pd.read_excel(input_file, usecols=range(9))  # Read only the first 9 columns
    
    # Extract numeric columns for standardization
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    
##    # Standardize the numeric data
##    scaler = StandardScaler()
##    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

##    # Normalize the numeric data
##    for col in numeric_columns:
##        mean = data[col].mean()
##        std = data[col].std()
##        data[col] = 0.8 * ((data[col] - mean) / std) + 0.1

    # Find global minimum and maximum values
    global_min = data[numeric_columns].min().min()
    global_max = data[numeric_columns].max().max()
    
    # Normalize the numeric data
    for col in numeric_columns:
        data[col] = 0.8 * ((data[col] - global_min) / (global_max - global_min)) + 0.1
    
    # Save the normalized data to a new Excel file
    data.to_excel(output_file, index=False)
    
    # Save the standardized data to a new Excel file
    data.to_excel(output_file, index=False)
    
    print("Standardized data saved to", output_file)

# Example usage
input_file = "FEHDataStudent.xlsx"
output_file = "standardized_data.xlsx"

standardize_data(input_file, output_file)
