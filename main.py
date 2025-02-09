
from src.Data_Preprocessing import DataPreprocessing
from Models.Models import Models
import src.Plots as plots
import pandas as pd
import  sys

def main():
    with open('./Reports/report.txt', 'w') as f:
        sys.stdout = f
        dataPreprocessing = DataPreprocessing() # Create an instance of the DataProcessing class
        exams_data, homework_data, compulsory_activities_data, optional_activities_data, original_data = dataPreprocessing.get_formatted_data() # Get the formatted data
        exams_scaled, homework_scaled, compulsory_activities_scaled, optional_activities_scaled, original_scaled = dataPreprocessing.normalize_data() # Get the Normalized data
        # print("Exams data:\n", exams_data.head(5)) # Print the first 5 rows of the exams data
        # print("\nHomework data:\n", homework_data.head(5)) # Print the first 5 rows of the homework data
        # print("\nCompulsory activities data:\n", compulsory_activities_data.head(5)) # Print the first 5 rows of the compulsory activities data
        # print("\nOptional activities data:\n", optional_activities_data.head(5)) # Print the first 5 rows of the optional activities data
        print("\nOriginal data:\n", original_data.head(5)) # Print the first 5 rows of the original data
        original_scaled_df = pd.DataFrame(original_scaled, columns = original_data.columns) # convert scaled data to dataframe for printing
        print("\nScaled original data:\n", original_scaled_df.head(5)) # Print the first 5 rows of scaled data

        dataPreprocessing.summarize_data() # Show some information of the dataset

        # make some general plots of the data
        plots.histograms(original_data, f'./Plots/Original_data_histograms.png')
        plots.scatter_plot_matrix(original_data, f'./Plots/Original_data_scatter_plot_matrix.png')

        models = Models(original_scaled, original_data) # create instance of the Models class which contains all the models
        models.KMeansClustering() # perform kmeans clustering
        models.HierarchicalClustering() # perform Hierarchical clustering
        models.DBScanClustering() # perform DBScan clustering

if __name__ == '__main__':
    main()