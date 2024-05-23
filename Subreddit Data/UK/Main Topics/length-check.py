import pandas as pd

def count_data_points(csv_path):
    """Count the number of data points in a CSV file, excluding the header."""
    try:
        data = pd.read_csv(csv_path)
        data_points_count = len(data)  # Get the number of rows, which equals the number of data points
        return data_points_count
    except FileNotFoundError:
        return "The file does not exist."
    except Exception as e:
        return f"An error occurred: {e}"

# Specify the path to your CSV file
csv_file_path = '/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/UK/keyword collection/ukpolitics.csv'
data_points = count_data_points(csv_file_path)
print(f"The number of data points in the file is: {data_points}")
