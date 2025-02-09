import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataPreprocessing:
    def __init__(self):
        exam_columns = 'A:B' # Define the columns of the data
        homework_columns = 'C:F'
        compulsory_activities_columns = 'G:N'
        optional_activities_columns = 'O:X'

        self.exams_data = pd.read_excel('./Data/grades.xlsx', skiprows = 1, usecols = exam_columns) # Load the data from the excel file
        self.homework_data = pd.read_excel('./Data/grades.xlsx', skiprows = 1, usecols = homework_columns, names = ['hw 1', 'hw 2', 'hw 3', 'hw 4']) # Load the data from the excel file
        self.compulsory_activities_data = pd.read_excel('./Data/grades.xlsx', skiprows = 1, usecols = compulsory_activities_columns, names = ['ca 1', 'ca 2', 'ca 3', 'ca 4', 'ca 5', 'ca 6', 'ca 7', 'ca 8']) # Load the data from the excel file
        self.optional_activities_data = pd.read_excel('./Data/grades.xlsx', skiprows = 1, usecols = optional_activities_columns, names = ['oa 1', 'oa 2', 'oa 3', 'oa 4', 'oa 5', 'oa 6', 'oa 7', 'oa 8', 'oa 9', 'oa 10']) # Load the data from the excel file
        self.original_data = pd.read_excel(
            './Data/grades.xlsx', skiprows = 1,
            names = ['Final Exam', 'Repeat Exam', 'Homework 1', 'Homework 2', 'Homework 3', 'Homework 4', 'Compulsory Activity 1', 'Compulsory Activity 2', 'Compulsory Activity 3', 'Compulsory Activity 4', 'Compulsory Activity 5', 'Compulsory Activity 6', 'Compulsory Activity 7', 'Compulsory Activity 8', 'Optional Activity 1', 'Optional Activity 2', 'Optional Activity 3', 'Optional Activity 4', 'Optional Activity 5', 'Optional Activity 6', 'Optional Activity 7', 'Optional Activity 8', 'Optional Activity 9', 'Optional Activity 10']
        ) # Load the data from the excel file

    def normalize_data(self):
        self.convert_to_numeric()
        scaler = StandardScaler()
        return scaler.fit_transform(self.exams_data), scaler.fit_transform(self.homework_data), scaler.fit_transform(self.compulsory_activities_data), scaler.fit_transform(self.optional_activities_data), scaler.fit_transform(self.original_data)

    def return_original_data(self):
        return self.original_data

    def summarize_data(self):
        print("Shape of the dataset: ", self.original_data.shape)
        print("Dataset description: ", self.original_data.describe())
        print("Class Distribution: ", self.original_data.groupby('Final Exam').size())

    def convert_to_numeric(self):
        self.exams_data.replace({'-': 0, -1: 0}, inplace = True) # Replace the missing values with 0
        self.homework_data.replace({'-': 0, -1: 0}, inplace = True)
        self.compulsory_activities_data.replace({'-': 0, -1: 0}, inplace = True)
        self.optional_activities_data.replace({'-': 0, -1: 0}, inplace = True)
        self.original_data.replace({'-': 0, -1: 0}, inplace = True)

    def get_formatted_data(self):
        self.convert_to_numeric()
        return self.exams_data, self.homework_data, self.compulsory_activities_data, self.optional_activities_data, self.original_data

    def pass_the_exams(self):
        self.original_data['Pass Actual'] = self.original_data.apply(lambda row: 0 if row['Final Exam'] >= 5 or row['Repeat Exam'] >= 5 else 1, axis = 1) # Create a new column 'Pass' based on the final and repeat exam grades
        return self.original_data

    def create_hypothesis_testing_dataset(self):
        mean_homework_mark = self.homework_data.mean(axis = 1) # Calculate the mean homework mark
        exam_result = self.exams_data[['Final', 'Repeat']].max(axis = 1) # Get the maximum exam result
        return mean_homework_mark, exam_result
