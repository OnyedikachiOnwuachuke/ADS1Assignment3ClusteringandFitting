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

    agric_forest = agric_forest.drop(
        ['Indicator Code', 'Country Code', 'Indicator Name', 'Unnamed: 66'], axis=1)

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
agric_forest_countries = select_countries_and_years(
    agric_forest, countries, years)
print(agric_forest_countries)


def fit_agricforest(df, columns, cluster_range):
    """
    Fits a KMeans model to the given DataFrame and calculates silhouette scores.

    Parameters:
    df (pd.DataFrame): The DataFrame to fit the model to.
    columns (list of str): The columns to use for fitting.
    cluster_range (range): The range of cluster numbers to try.

    Returns:
    None
    """
    # Extract columns for fitting.
    # .copy() prevents changes in df_fit to affect the original DataFrame.
    fit_df = df[columns].copy()

    # Normalize DataFrame and print result
    # Normalization is done only on the extract columns. .copy() prevents
    # changes in fit_df to affect the original DataFrame.
    # Replace ct.scaler with your function
    fit_df, df_min, df_max = ct.scaler(fit_df)
    print(fit_df.describe())
    print()

    print("n   score")
    # Loop over trial numbers of clusters calculating the silhouette
    for ic in cluster_range:
        # Set up KMeans and fit
        kmeans = cluster.KMeans(n_clusters=ic)
        kmeans.fit(fit_df)

        # Extract labels and calculate silhouette score
        labels = kmeans.labels_
        print(ic, skmet.silhouette_score(fit_df, labels))


# Define the columns of interest
columns = ["1980", "2020"]

# Define the range of cluster numbers to try
cluster_range = range(2, 7)

# Use the function
fit_agricforest(agric_forest_countries, columns, cluster_range)

#Plotting the Fits and Kmeans and producing 4 clusters
# Fit k-means with 4 clusters
# Make sure ct.scaler function returns a DataFrame
fit_agricforest, df_min, df_max = ct.scaler(agric_forest_countries[["1980", "2020"]].copy())

# Fit k-means with 4 clusters
kmeans = cluster.KMeans(n_clusters=4)
kmeans.fit(fit_agricforest)

# Add cluster label column to the original dataframe
agric_forest_countries["cluster_label"] = kmeans.labels_

# Group countries by cluster label
grouped = agric_forest_countries.groupby("cluster_label")

# Print countries in each cluster
for label, group in grouped:
    print("Cluster", label)
    print(group.index.tolist())
    print()

# Plot clusters with labels
plt.scatter(fit_agricforest["1980"], fit_agricforest["2020"], c=kmeans.labels_, cmap="Set1")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='k', marker="d", s=80)
plt.xlabel("1980")
plt.ylabel("2020")
plt.title("Clusters Agriculture, forestry, and fishing, value added (% of GDP)")

plt.show()