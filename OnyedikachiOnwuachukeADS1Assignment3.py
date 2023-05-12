# Importing the necessary Libraries needed for this Assignment
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import matplotlib.pyplot as plt
import cluster_tools as ct
import scipy.optimize as opt
from scipy.optimize import curve_fit
import errors as err

"""
Creating a def function to read and clean the world bank dataset 
on Agriculture, forestry, and fishing, value added (% of GDP) 
and return a clean dataframe:
"""

def read_and_clean_data(file_name):
    """
      A function that reads in world bank data on 
      Agriculture, forestry, and fishing, value added (% of GDP)  various 
      indicators from and return both the original and transposed version of
      the dataset
    
    Args:
        filename: the name of the world bank dataset that will be read for analysis 
        and manupulation
        
            
    Returns: 
        The dataset as agric_forest years as columns and countries as rows for analysis
    """    
    agric_forest = pd.read_csv(file_name, skiprows=4)
     
    agric_forest = agric_forest.drop(['Indicator Code', 'Country Code', 'Indicator Name', 'Unnamed: 66'], axis=1)

    agric_forest.set_index('Country Name', drop=True, inplace=True)

    agric_forest = agric_forest.dropna()

    return agric_forest

# Use the function
file_name = 'Agricandforest%GDPworldbankdata.csv'
agric_forest = read_and_clean_data(file_name)
print(agric_forest)


"""
Creating a def function to select random countries and specified years to be used
for futher analysis
"""
def select_countries_and_years(df, countries, years):
    """
    Selects specified countries and years from the given DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to select data from.
    countries (list of str): The list of countries to select.
    years (list of str): The list of years to select.

    Returns:
    pd.DataFrame: A DataFrame containing data for the specified countries and years.
    """
    # Selecting random countries from the DataFrame for the analysis
    selected_countries = df.loc[countries]

    # Selecting the years at ten years intervals from 1970 to 2020 for more comprehensive analysis
    selected_data = selected_countries[years]

    return selected_data

# Define the list of countries
countries = ['Benin', 'Burkina Faso', 'Bangladesh', 'Brazil', 'Botswana', 'Chile',
   'China', "Cote d'Ivoire", 'Congo, Rep.', 'Costa Rica', 'Ecuador',
   'Egypt, Arab Rep.', 'France', 'Gabon', 'Ghana', 'Guyana', 'Honduras',
   'India', 'Iran, Islamic Rep.', 'Kenya', 'Korea, Rep.', 'Sri Lanka',
   'Lesotho', 'Malawi', 'Malaysia', 'Niger', 'Pakistan', 'Philippines',
   'Sudan', 'Senegal', 'Singapore', 'Suriname', 'Eswatini', 'Chad', 'Togo',
   'Thailand', 'Turkiye', 'Uganda', 'South Africa', 'Zambia']

# Define the years of interest
years = ["1970", "1980", "1990", "2000", "2010", "2020"]

# Use the function
agric_forest_countries = select_countries_and_years(agric_forest, countries, years)
print(agric_forest_countries)

#Plotting a scatter matrix on the dataframe to show the least correlated years for further analysis
pd.plotting.scatter_matrix(agric_forest_countries, figsize=(9, 9), s=5, alpha=0.8)
plt.show()
