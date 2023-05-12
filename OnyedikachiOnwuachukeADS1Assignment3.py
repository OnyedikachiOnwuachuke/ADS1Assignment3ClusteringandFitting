#Importing the necessary Libraries needed for this Assignment
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
from sklearn.cluster import KMeans
import matplotlib.lines as mlines

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


# Using this function
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

#Plotting a scatter matrix to show the least correlated data from the selected years and countries
pd.plotting.scatter_matrix(agric_forest_countries, figsize=(9, 9), s=5, alpha=0.8)
plt.show()

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

# Fit k-means with 4 clusters
kmeans = KMeans(n_clusters=4)
kmeans.fit(agric_forest_countries.copy())

# Add cluster label column to the original dataframe
agric_forest_countries["cluster_label"] = kmeans.labels_

# Group countries by cluster label
grouped = agric_forest_countries.groupby("cluster_label")

# Get the centroid points
cen = kmeans.cluster_centers_

# Plot clusters with labels and centroids
scatter = plt.scatter(agric_forest_countries["1980"], agric_forest_countries["2020"], c=kmeans.labels_, cmap="Set1")
plt.scatter(cen[:, 0], cen[:, 1], color='k', marker="d", s=80)
plt.xlabel("1980")
plt.ylabel("2020")
plt.title("Clusters Agriculture, forestry, and fishing, value added (% of GDP)")

# Get the unique colors from the scatter plot
unique_colors = np.unique(scatter.get_array())

# Filter out NaN values from unique_colors
unique_colors = unique_colors[~np.isnan(unique_colors)]

# Define custom colors for each cluster
cluster_colors = ["red", "grey", "purple", "brown"]

# Create custom legend handles for each cluster with corresponding colors
legend_handles = []
for label, color in zip(grouped.groups.keys(), cluster_colors):
    legend_handles.append(mlines.Line2D([], [], color=color, marker='o', linestyle='None', label=f"Cluster {label}"))

# Add legend
plt.legend(handles=legend_handles)

plt.show()
"""
Creating a def function to read and clean the world bank dataset 
on Population total and return a clean dataframe:
"""
def pop_totC(file_name, countries):
    """
    Loads a CSV file, processes it, and selects specified countries.

    Parameters:
    file_name (str): The name of the CSV file to load.
    countries (list of str): The list of countries to select.

    Returns:
    pd.DataFrame: The processed DataFrame with selected countries.
    """
    df = pd.read_csv(file_name, skiprows=4)

    df = df.drop(['Indicator Code', 'Country Code', 'Indicator Name'], axis=1)

    df.set_index('Country Name', drop=True, inplace=True)

    # Select specified countries and transpose the DataFrame
    df_selected = df.loc[countries].transpose()

    # Convert all values to numeric, coercing errors to NaN
    df_selected = df_selected.applymap(lambda x: pd.to_numeric(x, errors='coerce'))

    # Rename the index to 'Year'
    df_selected = df_selected.rename_axis('Year')

    return df_selected

# Use the function
file_name = 'populationwbdata.csv'
countries = ['South Africa', 'Kenya', 'Ghana', 'Niger']
pop_totC = pop_totC(file_name, countries)
print(pop_totC)


"""
A def function in Plotting a subpplot for fitting the 
data sets with curve_fit for each country selected from
each cluster for comparison for similarities or differencies using polymonial model
"""

def plot_polyfit(dataframe, countries, degree=3):
    """
    This function generates a subplot grid and plots the population data for the specified countries,
    along with a polynomial fit.

    Parameters:
    dataframe (pd.DataFrame): The dataframe containing population data.
    countries (list): The list of countries to generate the plots for.
    degree (int): The degree of the polynomial to fit.

    Returns:
    None
    """
    from numpy.polynomial.polynomial import Polynomial
    

    # Determine subplot grid size based on number of countries
    grid_size = int(np.ceil(np.sqrt(len(countries))))

    fig, axs = plt.subplots(grid_size, grid_size, figsize=(12, 12))

    # Flatten axs to iterate easily in the case of multiple subplots
    axs = axs.flatten()

    # Iterate over each country and each subplot
    for ax, country in zip(axs, countries):
        # Get the data for the current country
        x_data = dataframe.index.values.astype(int)
        y_data = dataframe[country].values

        # Fit the polynomial to the data
        p = Polynomial.fit(x_data, y_data, degree)

        # Generate x values for the fitted function
        x_fit = np.linspace(x_data.min(), x_data.max(), 1000)

        # Calculate the fitted y values
        y_fit = p(x_fit)

        # Plot the original data and the fitted function
        ax.plot(x_data, y_data, 'bo', label='Data')
        ax.plot(x_fit, y_fit, 'r-', label='Fit')
        ax.set_title(country)
        ax.set_xlabel('Year')
        ax.set_ylabel('Population')

        # Add a legend
        ax.legend()
        
    # Set a common title for the entire plot
    fig.suptitle('Population Trends and Polynomial Fits for Selected Countries', fontsize=16)

    plt.tight_layout()
    plt.show()

plot_polyfit(pop_totC, ['South Africa', 'Kenya', 'Ghana', 'Niger'], degree=3)

"""
A def function in Plotting a subpplot for fitting the data sets with curve_fit 
and the error rangefor each country selected fromeach cluster for comparison 
for similarities or differencies using polymonial model
"""
def poly(t, c0, c1, c2, c3):
    """ Computes a polynomial c0 + c1*t + c2*t^2 + c3*t^3 """
    t = t - 1950
    f = c0 + c1*t + c2*t**2 + c3*t**3
    return f

def err_ranges(x, func, popt, perr):
    """ Calculate upper and lower errors """
    popt_up = popt + perr
    popt_dw = popt - perr
    fit = func(x, *popt)
    fit_up = func(x, *popt_up)
    fit_dw = func(x, *popt_dw)
    return fit_up, fit_dw

def plot_population_projection(dataframe, countries):
    """
    This function generates a subplot grid and plots the population data for the specified countries,
    along with a polynomial fit and its error ranges.

    Parameters:
    dataframe (pd.DataFrame): The dataframe containing population data.
    countries (list): The list of countries to generate the plots for.

    Returns:
    None
    """
    # Ensure index is of integer type
    dataframe.index = dataframe.index.astype(int)

    # Initialize a figure
    fig, axs = plt.subplots(2, 2, figsize=(10,10))

    # Flattening axs for easy iterating
    axs = axs.ravel()

    # Loop over the countries list
    for i, country in enumerate(countries):
        popt, pcorr = curve_fit(poly, dataframe.index, dataframe[country])
        print(f"Fit parameters for {country}: ", popt)
        # extract variances and calculate sigmas
        sigmas = np.sqrt(np.diag(pcorr))
        # call function to calculate upper and lower limits with extrapolation
        # create extended year range
        years = np.arange(1950, 2051)
        lower, upper = err_ranges(years, poly, popt, sigmas)
        axs[i].plot(dataframe.index, dataframe[country], label="data")
        axs[i].plot(years, poly(years, *popt), label="fit")
        # plot error ranges with transparency
        axs[i].fill_between(years, lower, upper, alpha=0.5)
        axs[i].set_title(f"Polynomial Fit for {country}")
        axs[i].legend(loc="upper left")

    # Adjust layout for neatness
    plt.tight_layout()
    plt.show()

plot_population_projection(pop_totC, ['South Africa', 'Kenya', 'Ghana', 'Niger'])





