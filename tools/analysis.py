# egctools/analysis.py

# This file contains functions for exploratory data analysis (EDA), data visualization, and statistical analysis.
# Edward was here

# Packages & Libraries
import os;
from math import pi;
import warnings;

import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
import seaborn as sns;
import squarify;

import scipy.stats as stats;
from scipy.stats import chi2_contingency, f_oneway;

import statsmodels.api as sm;
from statsmodels.formula.api import ols;
from statsmodels.stats.multicomp import pairwise_tukeyhsd;

from sklearn.metrics import brier_score_loss;

from IPython.display import display, HTML;
from pandas.plotting import register_matplotlib_converters;

import holoviews as hv;
import itertools;

hv.extension('bokeh');

# Register matplotlib converters and suppress warnings
register_matplotlib_converters()
warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')
warnings.filterwarnings("ignore", category=FutureWarning, module='pandas')

def single_variable_EDA(df: pd.DataFrame, column: str=None, analysis_type: str=None, bins=None, output_dir: str=None, **kwargs):
    """
    Performs single variable EDA on a column in a pandas DataFrame.
    
    Parameters:
    - df: pandas DataFrame
    - column: str, column name
    - analysis_type: str, "categorical" or "numerical". If None, it will be determined automatically.
    - bins: int or list, binning for numerical columns
    - output_dir: str, directory to save output plots
    - kwargs: additional keyword arguments for customizing the plots
    """
    
    def get_aggregate_statistics(df, column):
        """Returns aggregate statistics of the column as a DataFrame."""
        stats = df[column].describe().to_frame().T
        stats['missing'] = df[column].isnull().sum()
        return stats

    def plot_categorical(df, column, output_dir, **kwargs):
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # Create a 1x3 grid of subplots

        # Count Plot
        plt.subplot(1, 3, 1)
        sns.countplot(x=column, data=df, **kwargs)
        plt.title(f'Count Plot of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        
        # Treemap
        plt.subplot(1, 3, 2)
        sizes = df[column].value_counts()
        labels = sizes.index
        squarify.plot(sizes=sizes, label=labels, **kwargs)
        plt.title(f'Treemap of {column}')
        plt.axis('off')
        
        # Radar/Spider Chart
        sizes = df[column].value_counts()
        labels = sizes.index
        values = sizes.values
        num_vars = len(labels)
        
        if num_vars < 3:
            plt.subplot(1, 3, 3)
            plt.text(0.5, 0.5, 'Not enough categories for Radar Chart', ha='center', va='center', fontsize=12)
            plt.axis('off')
        else:
            angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
            angles += angles[:1]
            values = list(values) + [values[0]]
            
            ax = plt.subplot(1, 3, 3, polar=True)
            ax.fill(angles, values, color='blue', alpha=0.25)
            ax.plot(angles, values, color='blue', linewidth=2)
            ax.set_yticklabels([])
            plt.title(f'Radar Chart of {column}')
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels)

        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, f"{column}_categorical.png"))
        plt.show();
        
        # Display value counts and percentages as HTML tables
        value_counts = df[column].value_counts().to_frame().reset_index()
        value_counts.columns = [column, 'count']
        value_counts['percentage'] = (value_counts['count'] / len(df) * 100).round(2).astype(str) + '%'
        
        value_counts_sorted_by_percentage = value_counts.sort_values(by='percentage', ascending=False)
        value_counts_sorted_by_value = value_counts.sort_values(by=column)
        
        html_table_percentage = value_counts_sorted_by_percentage.to_html(index=False)
        html_table_value = value_counts_sorted_by_value.to_html(index=False)
        
        display(HTML(f"""
        <div style="display: flex;">
            <div style="margin-right: 20px;">
                <h3>Ordered by Percentage</h3>
                {html_table_percentage}
            </div>
            <div>
                <h3>Ordered by Value</h3>
                {html_table_value}
            </div>
        </div>
        """))
        
    def freedman_diaconis_bins(data):
        """Calculate optimal number of bins using the Freedman-Diaconis rule."""
        data = data.dropna()
        q25, q75 = np.percentile(data, [25, 75])
        bin_width = 2 * (q75 - q25) * len(data) ** (-1/3)
        return max(1, int((data.max() - data.min()) / bin_width))

    def plot_numerical(df, column, output_dir, bins, **kwargs):
        # Ensure the column is numeric
        df[column] = pd.to_numeric(df[column], errors='coerce')
        
        if bins is None:
            bins = freedman_diaconis_bins(df[column])
        
        plt.figure(figsize=(24, 6))
        
        # Box Plot
        plt.subplot(1, 4, 1)
        sns.boxplot(x=df[column].dropna(), **kwargs)
        plt.title(f'Box Plot of {column}')
        plt.xlabel(column)
        
        # Histogram with Density Line
        plt.subplot(1, 4, 2)
        sns.histplot(df[column].dropna(), kde=True, bins=bins, **kwargs)
        plt.title(f'Histogram with Density Line of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        
        # Violin Plot
        plt.subplot(1, 4, 3)
        sns.violinplot(x=df[column].dropna(), **kwargs)
        plt.title(f'Violin Plot of {column}')
        plt.xlabel(column)
        
        # KDE Plot
        plt.subplot(1, 4, 4)
        sns.kdeplot(df[column].dropna(), **kwargs)
        plt.title(f'KDE Plot of {column}')
        plt.xlabel(column)
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, f"{column}_numerical.png"))
        plt.show();
        
    def identify_outliers(df, column):
        """Identify outliers in a numerical column using the IQR method."""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        return outliers

    def determine_analysis_type(df, column):
        """Automatically determine if a column is categorical or numerical."""
        if df[column].dtype == 'object' or df[column].nunique() < 20:
            return "categorical"
        else:
            return "numerical"

    # Validate inputs
    if df is None:
        raise ValueError("No DataFrame provided.")
    if column is None:
        raise ValueError("No column provided.")

    # Automatically determine the analysis type if not provided
    if analysis_type is None:
        analysis_type = determine_analysis_type(df, column)
    
    if analysis_type not in ["categorical", "numerical"]:
        raise ValueError("Invalid type. Must be 'categorical' or 'numerical'.")

    # Ensure the output directory exists
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Set color-blind safe palette
    sns.set_palette("colorblind")
    
    # Perform the appropriate analysis
    if analysis_type == "categorical":
        plot_categorical(df, column, output_dir, **kwargs)
    elif analysis_type == "numerical":
        plot_numerical(df, column, output_dir, bins, **kwargs)
        
    # Display aggregate statistics
    stats = get_aggregate_statistics(df, column)
    print("Aggregate Statistics of {}:".format(column))
    display(HTML(stats.to_html(index=False)))

    # Outlier analysis for numerical data
    if analysis_type == "numerical":
        outliers = identify_outliers(df, column)
        if not outliers.empty:
            print("Outlier examples (using IQR method) of {}:".format(column))
            display(HTML(outliers.head(25).to_html(index=False)))
            print("Note: only up to the first 25 outliers are displayed.")
            print("Total number of potential outliers:", outliers.shape[0])
            IQR_lower = df[column].quantile(0.25) - 1.5 * (df[column].quantile(0.75) - df[column].quantile(0.25))
            IQR_upper = df[column].quantile(0.75) + 1.5 * (df[column].quantile(0.75) - df[column].quantile(0.25))
            # Show IQR bounds, with comma and up to 4 decimal places
            print(f"Lower Bound (Q1 - 1.5*IQR): {IQR_lower:,.4f} / Upper Bound (Q3 + 1.5*IQR): {IQR_upper:,.4f}")
        else:
            print("No outliers detected.")
  
def pairwise_EDA(df: pd.DataFrame, column1: str, column2: str, analysis_type: str = None, bins1=None, bins2=None, output_dir: str = None, **kwargs):
    """
    Performs pairwise EDA on two columns in a pandas DataFrame.

    Parameters:
    - df: pandas DataFrame
    - column1: str, first column name
    - column2: str, second column name
    - analysis_type: str, type of analysis ("cat_to_cat", "cat_to_num", "num_to_num", "num_to_cat"). If None, it will be determined automatically.
    - bins1: int or list, binning for the first column if it's numerical
    - bins2: int or list, binning for the second column if it's numerical
    - output_dir: str, directory to save output plots
    - kwargs: additional keyword arguments for customizing the plots
    """
    
    def describe_correlation(correlation_value):
        """Describes the strength and direction of the correlation."""
        abs_value = abs(correlation_value)
        if abs_value > 0.8:
            strength = 'Strong'
        elif abs_value > 0.5:
            strength = 'Moderate'
        elif abs_value > 0.3:
            strength = 'Weak'
        else:
            strength = 'Very weak or no'

        if correlation_value > 0:
            direction = 'positive'
        elif correlation_value < 0:
            direction = 'negative'
        else:
            direction = 'no'

        return f'{strength} {direction} correlation'

    def encode_binary(column):
        """Encodes binary categorical column to 0 and 1."""
        unique_values = column.unique()
        if len(unique_values) == 2:
            mapping = {unique_values[0]: 0, unique_values[1]: 1}
            return column.map(mapping)
        else:
            raise ValueError("Column is not binary and cannot be encoded for point biserial correlation.")

    def correlation_ratio(categories, measurements):
        """Calculates the correlation ratio (eta coefficient) for categorical to numerical association."""
        fcat, _ = pd.factorize(categories)
        cat_num = np.max(fcat) + 1
        y_avg_array = np.zeros(cat_num)
        n_array = np.zeros(cat_num)

        for i in range(cat_num):
            cat_measures = measurements[fcat == i]
            n_array[i] = len(cat_measures)
            y_avg_array[i] = np.average(cat_measures)
        
        y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
        numerator = np.sum(np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2)))
        denominator = np.sum(np.power(np.subtract(measurements, y_total_avg), 2))
        eta = np.sqrt(numerator / denominator)
        return eta

    def calculate_correlations(df, column1, column2, analysis_type):
        """Function to calculate different types of correlations between two columns."""
        correlations = []
        if analysis_type == "num_to_num":
            pearson_corr = df[column1].corr(df[column2], method='pearson')
            spearman_corr = df[column1].corr(df[column2], method='spearman')
            kendall_corr = df[column1].corr(df[column2], method='kendall')
            correlations = [
                ('Pearson', pearson_corr, describe_correlation(pearson_corr), 'Linear relationship. Sensitive to outliers and non-linearity.'),
                ('Spearman', spearman_corr, describe_correlation(spearman_corr), 'Monotonic relationship. Differs from Pearson in handling outliers and non-linearity.'),
                ('Kendall', kendall_corr, describe_correlation(kendall_corr), 'Ordinal (rank) relationship. Differs from Spearman in handling ties.')
            ]
        elif analysis_type == "cat_to_cat":
            contingency_table = pd.crosstab(df[column1], df[column2])
            chi2 = stats.chi2_contingency(contingency_table)[0]
            n = df.shape[0]
            r, k = contingency_table.shape
            cramers_v = (chi2 / (n * (min(k, r) - 1)))**0.5
            correlations = [
                ("Cramér's V", cramers_v, describe_correlation(cramers_v), 'Measures association between two categorical variables.')
            ]
        elif analysis_type == "cat_to_num" or analysis_type == "num_to_cat":
            if df[column1].nunique() == 2:
                col1_encoded = encode_binary(df[column1])
                col2 = df[column2].astype(float)
                point_biserial_corr = stats.pointbiserialr(col1_encoded, col2)[0]
                correlations = [
                    ('Point Biserial', point_biserial_corr, describe_correlation(point_biserial_corr), 'Measures correlation between a binary and a continuous variable.')
                ]
            else:
                col1 = df[column1]
                col2 = df[column2].astype(float)
                eta = correlation_ratio(col1, col2)
                correlations = [
                    ('Correlation Ratio (Eta)', eta, describe_correlation(eta), 'Measures correlation between a categorical and a continuous variable.')
                ]
        return correlations

    def display_correlation_table(correlations):
        """Displays nicely formatted correlation table in HTML, specifically for Jupyter notebooks."""
        correlation_table = pd.DataFrame(correlations, columns=['Correlation Measure', 'Correlation Value', 'Description', 'Explanation'])
        display(HTML(correlation_table.to_html(index=False)))

    def determine_analysis_type(df, column1, column2):
        """Automatic determination of analysis type based on column data types and unique values."""
        col1_type = 'cat' if df[column1].dtype == 'object' or df[column1].nunique() < 10 else 'num'
        col2_type = 'cat' if df[column2].dtype == 'object' or df[column2].nunique() < 10 else 'num'
        return f"{col1_type}_to_{col2_type}"

    def bin_numerical_column(df, column, bins):
        """Function to bin a numerical column based on a specified number of bins or bin edges."""
        if isinstance(bins, int):
            df[column + '_binned'] = pd.cut(df[column], bins=bins)
        elif isinstance(bins, list):
            df[column + '_binned'] = pd.cut(df[column], bins=bins)
        return df

    def plot_categorical_to_categorical(df, column1, column2, output_dir=None, **kwargs):
        plt.figure(figsize=(18, 6))
        
        # Contingency Table Heatmap
        plt.subplot(1, 2, 1)
        contingency_table = pd.crosstab(df[column1], df[column2])
        sns.heatmap(contingency_table, annot=True, fmt="d", cmap="Blues", **kwargs)
        plt.title(f'Contingency Table Heatmap of {column1} and {column2}')
        plt.xlabel(column2)
        plt.ylabel(column1)
        
        # Clustered Bar Plot with distinct colors
        plt.subplot(1, 2, 2)
        sns.countplot(x=column1, hue=column2, data=df, palette="tab10", **kwargs)
        plt.title(f'Clustered Bar Plot of {column1} and {column2}')
        plt.xlabel(column1)
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, f"{column1}_vs_{column2}_cat_to_cat.png"))
        plt.show()

    def plot_categorical_to_numerical(df, column1, column2, output_dir=None, **kwargs):
        plt.figure(figsize=(14, 12))
        # Box Plot
        plt.subplot(2, 2, 1)
        sns.boxplot(x=column1, y=column2, data=df, **kwargs)
        plt.title(f'Box Plot of {column2} by {column1}')
        plt.xlabel(column1)
        plt.ylabel(column2)
        
        # Violin Plot
        plt.subplot(2, 2, 2)
        sns.violinplot(x=column1, y=column2, data=df, **kwargs)
        plt.title(f'Violin Plot of {column2} by {column1}')
        plt.xlabel(column1)
        plt.ylabel(column2)
        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, f"{column1}_vs_{column2}_cat_to_num.png"))
        plt.show()

    def plot_numerical_to_numerical(df, column1, column2, output_dir=None, **kwargs):
        plt.figure(figsize=(14, 12))
        # Scatter Plot with LOWESS
        plt.subplot(2, 2, 1)
        sns.scatterplot(x=column1, y=column2, data=df, **kwargs)
        lowess = sm.nonparametric.lowess(df[column2], df[column1])
        plt.plot(lowess[:, 0], lowess[:, 1], color='red', lw=2, label='LOWESS')
        plt.legend()
        plt.title(f'Scatter Plot with LOWESS of {column1} vs {column2}')
        plt.xlabel(column1)
        plt.ylabel(column2)
        
        # Hexbin Plot
        plt.subplot(2, 2, 2)
        plt.hexbin(df[column1], df[column2], gridsize=30, cmap='Blues')
        plt.colorbar(label='Counts')
        plt.title(f'Hexbin Plot of {column1} vs {column2}')
        plt.xlabel(column1)
        plt.ylabel(column2)
        
        # Add a regression line if requested
        if kwargs.get('regression', False):
            plt.subplot(2, 2, 1)
            sns.regplot(x=column1, y=column2, data=df, scatter=False, **kwargs)
        
        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, f"{column1}_vs_{column2}_num_to_num.png"))
        plt.show()

    # Validate inputs
    if df is None:
        raise ValueError("No DataFrame provided.")
    if column1 is None or column2 is None:
        raise ValueError("Both column names must be provided.")
    
    # Automatically determine the analysis type if not provided
    if analysis_type is None:
        analysis_type = determine_analysis_type(df, column1, column2)
    
    if analysis_type not in ["cat_to_cat", "cat_to_num", "num_to_num", "num_to_cat"]:
        raise ValueError("Invalid type. Must be 'cat_to_cat', 'cat_to_num', 'num_to_num', or 'num_to_cat'.")
    
    # Binning if needed
    if bins1 and 'num' in analysis_type.split('_')[0]:
        df = bin_numerical_column(df, column1, bins1)
        column1 = column1 + '_binned'
    if bins2 and 'num' in analysis_type.split('_')[1]:
        df = bin_numerical_column(df, column2, bins2)
        column2 = column2 + '_binned'
    
    # Ensure the output directory exists
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Set color-blind safe palette
    sns.set_palette("colorblind")
    
    # Perform the appropriate analysis
    if analysis_type == "cat_to_cat":
        plot_categorical_to_categorical(df, column1, column2, output_dir, **kwargs)
    elif analysis_type == "cat_to_num":
        plot_categorical_to_numerical(df, column1, column2, output_dir, **kwargs)
    elif analysis_type == "num_to_num":
        plot_numerical_to_numerical(df, column1, column2, output_dir, **kwargs)
    elif analysis_type == "num_to_cat":
        plot_categorical_to_numerical(df, column2, column1, output_dir, **kwargs)

    # Display correlation table
    correlations = calculate_correlations(df, column1, column2, analysis_type)
    if correlations:
        display_correlation_table(correlations)

def cramers_v(confusion_matrix):
    """Calculate Cramér's V statistic for categorical-categorical association."""
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    return np.sqrt(chi2 / (n * (min(confusion_matrix.shape) - 1)))

def correlation_ratio(categories, measurements):
    """Calculate the correlation ratio for categorical-numerical association."""
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat) + 1
    y_avg = np.mean(measurements)
    numerator = np.sum([np.sum(measurements[fcat == i]) * (np.mean(measurements[fcat == i]) - y_avg) ** 2 for i in range(cat_num)])
    denominator = np.sum((measurements - y_avg) ** 2)
    return numerator / denominator

def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k-1, r-1))

def correlation_ratio(categories, measurements):
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2)))
    denominator = np.sum(np.power(np.subtract(measurements, y_total_avg), 2))
    return np.sqrt(numerator / denominator)

def correlation_analysis(df: pd.DataFrame, target: str = None):
    """
    Perform correlation analysis on a given DataFrame.
    This function includes visualizations and statistical tests to understand the correlation
    between numerical and categorical variables, with a focus on the target variable if specified.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame for analysis.
    target (str): Target variable for analysis. Default is None.
    """
    
    # Ensure DataFrame is valid
    if df.empty:
        print("The provided DataFrame is empty.")
        return
    
    # Identify numerical and categorical columns
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    
    if target:
        if target not in df.columns:
            print(f"The target variable '{target}' is not in the DataFrame.")
            return
        if np.issubdtype(df[target].dtype, np.number):
            target_type = 'numerical'
        else:
            target_type = 'categorical'
    else:
        target_type = None
    
    # Set the style of the visualization
    sns.set(style="whitegrid")

    # Correlation Heatmap for numerical columns
    if num_cols:
        plt.figure(figsize=(10, 8))
        corr_matrix = df[num_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap for Numerical Columns')
        plt.show();
    
    # Association Heatmap for categorical columns
    if cat_cols:
        cat_corr = pd.DataFrame(index=cat_cols, columns=cat_cols, data=0.0)
        
        for col1 in cat_cols:
            for col2 in cat_cols:
                if col1 != col2:
                    contingency_table = pd.crosstab(df[col1], df[col2])
                    cat_corr.loc[col1, col2] = cramers_v(contingency_table)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cat_corr.astype(float), annot=True, cmap='coolwarm', center=0)
        plt.title('Association Heatmap for Categorical Columns (Cramér\'s V)')
        plt.show();
    
    # Heatmap for correlations between numerical and categorical features
    if num_cols and cat_cols:
        num_cat_corr = pd.DataFrame(index=num_cols, columns=cat_cols, data=0.0)
        
        for num in num_cols:
            for cat in cat_cols:
                num_cat_corr.loc[num, cat] = correlation_ratio(df[cat], df[num])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(num_cat_corr.astype(float), annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap between Numerical and Categorical Features (Correlation Ratio)')
        plt.show();
    
    # Analysis with respect to the target variable
    if target:
        if target_type == 'numerical':
            # Correlation of numerical target with numerical features
            if num_cols:
                target_corr = df[num_cols].corrwith(df[target]).sort_values(ascending=False)
                plt.figure(figsize=(10, 6))
                sns.barplot(x=target_corr.index, y=target_corr.values, hue=target_corr.index, palette='coolwarm', legend=False)
                plt.title(f'Correlation of Numerical Features with Target: {target}')
                plt.xlabel('Numerical Features')
                plt.ylabel('Correlation Coefficient')
                plt.xticks(rotation=45)
                plt.show();
            
            # Association of numerical target with categorical features
            if cat_cols:
                cat_pvalues = []
                for cat in cat_cols:
                    contingency_table = pd.crosstab(df[cat], df[target])
                    chi2, p, _, _ = chi2_contingency(contingency_table)
                    cat_pvalues.append(p)
                
                cat_pvalues = pd.Series(cat_pvalues, index=cat_cols).sort_values()
                plt.figure(figsize=(10, 6))
                sns.barplot(x=cat_pvalues.index, y=cat_pvalues.values, hue=cat_pvalues.index, palette='coolwarm', legend=False)
                plt.title(f'P-Values of Categorical Features with Target: {target}')
                plt.xlabel('Categorical Features')
                plt.ylabel('P-Value')
                plt.xticks(rotation=45)
                plt.show();
        
        elif target_type == 'categorical':
            # Association of categorical target with numerical features
            if num_cols:
                num_pvalues = []
                for num in num_cols:
                    fvalue, p = f_oneway(*[df[df[target] == cat][num] for cat in df[target].unique()])
                    num_pvalues.append(p)
                
                num_pvalues = pd.Series(num_pvalues, index=num_cols).sort_values()
                plt.figure(figsize=(10, 6))
                sns.barplot(x=num_pvalues.index, y=num_pvalues.values, hue=num_pvalues.index, palette='coolwarm', legend=False)
                plt.title(f'P-Values of Numerical Features with Target: {target}')
                plt.xlabel('Numerical Features')
                plt.ylabel('P-Value')
                plt.xticks(rotation=45)
                plt.show();
        
        # Association of categorical target with categorical features
        if cat_cols:
            target_corr = pd.Series(index=cat_cols, data=0.0)
            for cat in cat_cols:
                if cat != target:
                    contingency_table = pd.crosstab(df[target], df[cat])
                    chi2, p, _, _ = chi2_contingency(contingency_table)
                    target_corr[cat] = p
            
            target_corr = target_corr.sort_values()
            plt.figure(figsize=(10, 6))
            sns.barplot(x=target_corr.index, y=target_corr.values, hue=target_corr.index, palette='coolwarm', legend=False)
            plt.title(f'P-Values of Categorical Features with Target: {target}')
            plt.xlabel('Categorical Features')
            plt.ylabel('P-Value')
            plt.xticks(rotation=45)
            plt.show();

        # Correlation of categorical target with numerical features using correlation ratio
        if num_cols:
            target_corr_ratio = pd.Series(index=num_cols, data=0.0)
            for num in num_cols:
                target_corr_ratio[num] = correlation_ratio(df[target], df[num])
            
            target_corr_ratio = target_corr_ratio.sort_values(ascending=False)
            plt.figure(figsize=(10, 6))
            sns.barplot(x=target_corr_ratio.index, y=target_corr_ratio.values, hue=target_corr_ratio.index, palette='coolwarm', legend=False)
            plt.title(f'Correlation Ratio of Numerical Features with Categorical Target: {target}')
            plt.xlabel('Numerical Features')
            plt.ylabel('Correlation Ratio')
            plt.xticks(rotation=45)
            plt.show();

                                                    
def statistical_tests(df: pd.DataFrame, target: str, p_value_threshold: float = 0.05):
    """
    Perform statistical tests on a given DataFrame with a focus on the target variable.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame for statistical tests.
    target (str): Target variable for analysis.
    p_value_threshold (float): Custom p-value threshold for statistical tests. Default is 0.05.
    """
    
    # Ensure DataFrame is valid
    if df.empty:
        print("The provided DataFrame is empty.")
        return
    
    # Identify numerical and categorical columns
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    
    if target:
        if target not in df.columns:
            print(f"The target variable '{target}' is not in the DataFrame.")
            return
        if np.issubdtype(df[target].dtype, np.number):
            target_type = 'numerical'
        else:
            target_type = 'categorical'
    else:
        target_type = None
    
    # Helper function to display test results
    def display_test_results(test_name, results, interpretation, strength_note):
        display(HTML(f"<h3>{test_name} (p-value threshold: {p_value_threshold})</h3>"))
        display(HTML(pd.DataFrame(results).to_html(index=False)))
        print(f"Explanation: {interpretation}")
        print(f"How to interpret: {strength_note}\n")
    
    # Strength interpretation
    def interpret_strength(p_value):
        if p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < p_value_threshold:
            return '*'
        else:
            return ''
    
    # ANOVA Test
    def perform_anova():
        anova_results = []
        for cat in cat_cols:
            model = ols(f'{target} ~ C({cat})', data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            p_value = anova_table["PR(>F)"].iloc[0]
            anova_results.append({
                'Feature': cat,
                'Statistic': anova_table["F"].iloc[0],
                'p-value': p_value,
                'Strength': interpret_strength(p_value)
            })
        display_test_results(f"ANOVA Results for {target}", anova_results, 
                            "Tests if different groups have different averages.",
                            "*: Significant difference. None: No significant difference.")

    # Tukey HSD Test
    def perform_tukey_hsd():
        tukey_results = []
        for cat in cat_cols:
            tukey = pairwise_tukeyhsd(endog=df[target], groups=df[cat], alpha=p_value_threshold)
            tukey_p_value = np.max(tukey.pvalues)
            tukey_results.append({
                'Feature': cat,
                'p-value': tukey_p_value,
                'Strength': interpret_strength(tukey_p_value)
            })
        display_test_results(f"Tukey HSD Results for {target}", tukey_results, 
                             "Compares all pairs of group averages.",
                             "*: Significant difference. None: No significant difference.")

    # Kruskal-Wallis Test
    def perform_kruskal_wallis():
        kruskal_results = []
        for cat in cat_cols:
            groups = [df[target][df[cat] == level] for level in df[cat].unique()]
            kruskal_stat, kruskal_p = stats.kruskal(*groups)
            kruskal_results.append({
                'Feature': cat,
                'Statistic': kruskal_stat,
                'p-value': kruskal_p,
                'Strength': interpret_strength(kruskal_p)
            })
        display_test_results(f"Kruskal-Wallis Test Results for {target}", kruskal_results, 
                             "Non-parametric test for group differences.",
                             "*: Significant difference. None: No significant difference.")
    
    # Pearson and Spearman Correlation
    def perform_correlations():
        correlation_results = []
        for num in num_cols:
            if num != target:
                # Pearson Correlation
                pearson_stat, pearson_p = stats.pearsonr(df[target], df[num])
                correlation_results.append({
                    'Feature': num,
                    'Test': 'Pearson',
                    'Statistic': pearson_stat,
                    'p-value': pearson_p,
                    'Strength': interpret_strength(pearson_p)
                })

                # Spearman Correlation
                spearman_stat, spearman_p = stats.spearmanr(df[target], df[num])
                correlation_results.append({
                    'Feature': num,
                    'Test': 'Spearman',
                    'Statistic': spearman_stat,
                    'p-value': spearman_p,
                    'Strength': interpret_strength(spearman_p)
                })
        display_test_results(f"Correlation Results for {target}", correlation_results, 
                             "Measures relationship between two variables.",
                             "*: Significant correlation. None: No significant correlation.")
    
    # Levene's Test and T-test
    def perform_levene_ttest():
        levene_ttest_results = []
        for num in num_cols:
            if num != target:
                # Levene's Test for homogeneity of variances
                levene_stat, levene_p = stats.levene(df[target], df[num])
                levene_ttest_results.append({
                    'Feature': num,
                    'Test': 'Levene',
                    'Statistic': levene_stat,
                    'p-value': levene_p,
                    'Strength': interpret_strength(levene_p)
                })

                # T-test
                t_stat, t_p = stats.ttest_ind(df[target], df[num])
                levene_ttest_results.append({
                    'Feature': num,
                    'Test': 'T-test',
                    'Statistic': t_stat,
                    'p-value': t_p,
                    'Strength': interpret_strength(t_p)
                })
        display_test_results(f"Levene's Test and T-test Results for {target}", levene_ttest_results, 
                             "Levene's: Tests if variances are equal. T-test: Tests if means are different.",
                             "Levene's: *: Unequal variances. None: Equal variances.\nT-test: *: Significant difference. None: No significant difference.")

    # Chi-Square Test and Fisher's Exact Test
    def perform_chi_square_fisher():
        chi2_results = []
        for cat in cat_cols:
            if cat != target:
                contingency_table = pd.crosstab(df[target], df[cat])
                chi2, p, dof, ex = stats.chi2_contingency(contingency_table)
                chi2_results.append({
                    'Feature': cat,
                    'Test': 'Chi-Square',
                    'Statistic': chi2,
                    'p-value': p,
                    'Strength': interpret_strength(p)
                })

                # Fisher's Exact Test (for 2x2 tables)
                if contingency_table.shape == (2, 2):
                    fisher_stat, fisher_p = stats.fisher_exact(contingency_table)
                    chi2_results.append({
                        'Feature': cat,
                        'Test': "Fisher's Exact",
                        'Statistic': fisher_stat,
                        'p-value': fisher_p,
                        'Strength': interpret_strength(fisher_p)
                    })
        display_test_results(f"Chi-Square and Fisher's Exact Test Results for {target}", chi2_results, 
                             "Chi-Square: Tests association between two categorical variables. Fisher's Exact: Tests association in 2x2 table.",
                             "*: Significant association. None: No significant association.")
    
    # Additional tests for categorical target
    def perform_logistic_regression():
        logistic_results = []
        for num in num_cols:
            if num != target:
                model = sm.Logit(df[target], df[num]).fit(disp=False)
                p_value = model.pvalues[0]
                logistic_results.append({
                    'Feature': num,
                    'p-value': p_value,
                    'Strength': interpret_strength(p_value)
                })
        display_test_results(f"Logistic Regression Results for {target}", logistic_results, 
                             "Tests relationship between a categorical target and numerical features.",
                             "*: Significant relationship. None: No significant relationship.")

    def perform_mcnemar_test():
        mcnemar_results = []
        for cat in cat_cols:
            if cat != target:
                contingency_table = pd.crosstab(df[target], df[cat])
                if contingency_table.shape == (2, 2):
                    mcnemar_stat, mcnemar_p = stats.mcnemar(contingency_table)
                    mcnemar_results.append({
                        'Feature': cat,
                        'Statistic': mcnemar_stat,
                        'p-value': mcnemar_p,
                        'Strength': interpret_strength(mcnemar_p)
                    })
        display_test_results(f"McNemar Test Results for {target}", mcnemar_results, 
                             "Tests for changes in categorical data.",
                             "*: Significant change. None: No significant change.")
    
    def perform_brier_score():
        brier_results = []
        for cat in cat_cols:
            if cat != target:
                y_true = (df[target] == cat).astype(int)
                y_prob = df[cat]
                brier_score = brier_score_loss(y_true, y_prob)
                brier_results.append({
                    'Feature': cat,
                    'Brier Score': brier_score,
                    'Strength': interpret_strength(brier_score)
                })
        display_test_results(f"Brier Score Results for {target}", brier_results, 
                             "Measures accuracy of probabilistic predictions.",
                             "*: High accuracy. None: Low accuracy.")

    # Execute relevant tests based on target type
    if target_type == 'numerical':
        perform_anova()
        perform_tukey_hsd()
        perform_kruskal_wallis()
        perform_correlations()
        perform_levene_ttest()
    elif target_type == 'categorical':
        perform_chi_square_fisher()
        perform_logistic_regression()
        perform_mcnemar_test()
        perform_brier_score()

def chord_diagram(df, threshold=0.3, strong_threshold=0.5, output_file=None):
    """Create a chord diagram to visualize the associations in the DataFrame."""

    def cramers_v_corrected(confusion_matrix):
        """Calculate corrected Cramér's V for a given confusion matrix."""
        if confusion_matrix.size == 0:
            return 0.0
        chi2 = stats.chi2_contingency(confusion_matrix, correction=False)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1)**2) / (n - 1)
        kcorr = k - ((k - 1)**2) / (n - 1)
        return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

    def correlation_ratio(categories, measurements):
        """Calculate the correlation ratio (eta squared) for a given set of categories and measurements."""
        fcat, _ = pd.factorize(categories)
        cat_num = np.max(fcat) + 1
        y_avg_array = np.zeros(cat_num)
        n_array = np.zeros(cat_num)
        for i in range(cat_num):
            cat_measures = measurements[fcat == i]
            n_array[i] = len(cat_measures)
            y_avg_array[i] = np.average(cat_measures)
        y_total_avg = np.sum(y_avg_array * n_array) / np.sum(n_array)
        numerator = np.sum(n_array * (y_avg_array - y_total_avg) ** 2)
        denominator = np.sum((measurements - y_total_avg) ** 2)
        return np.sqrt(numerator / denominator) if numerator != 0 else 0.0

    def compute_association_matrix_corrected(df):
        """Compute the corrected association matrix for the given DataFrame."""
        columns = df.columns
        size = len(columns)
        association_matrix = pd.DataFrame(np.zeros((size, size)), columns=columns, index=columns)

        for i, j in itertools.combinations(range(size), 2):
            col_i = df.columns[i]
            col_j = df.columns[j]
            if df[col_i].dtype == 'object' and df[col_j].dtype == 'object':
                confusion_matrix = pd.crosstab(df[col_i], df[col_j])
                association_matrix.iloc[i, j] = cramers_v_corrected(confusion_matrix)
                association_matrix.iloc[j, i] = association_matrix.iloc[i, j]
            elif df[col_i].dtype == 'object' or df[col_j].dtype == 'object':
                categories = df[col_i] if df[col_i].dtype == 'object' else df[col_j]
                measurements = df[col_i].astype(float) if df[col_j].dtype == 'object' else df[col_j].astype(float)
                association_matrix.iloc[i, j] = correlation_ratio(categories, measurements)
                association_matrix.iloc[j, i] = association_matrix.iloc[i, j]
            else:
                correlation = df[[col_i, col_j]].corr().iloc[0, 1]
                association_matrix.iloc[i, j] = correlation
                association_matrix.iloc[j, i] = correlation

        return association_matrix

    association_matrix_corrected = compute_association_matrix_corrected(df)

    # Filter significant relationships based on threshold
    significant_pairs = (association_matrix_corrected.abs() >= threshold).stack()
    significant_pairs = significant_pairs[significant_pairs].index.tolist()
    significant_pairs = [(var1, var2, association_matrix_corrected.loc[var1, var2], 
                          'positive' if association_matrix_corrected.loc[var1, var2] > 0 else 'negative')
                         for var1, var2 in significant_pairs if var1 != var2]

    # Prepare data for chord diagram
    edges = pd.DataFrame(significant_pairs, columns=['from', 'to', 'value', 'type'])
    edges['abs_value'] = edges['value'].abs()

    # Highlight strong correlations
    edges['line_width'] = np.where(edges['abs_value'] >= strong_threshold, 7, edges['abs_value'] * 10)  # Adjusted line thickness

    chord_data = hv.Dataset(edges, ['from', 'to'], ['value', 'type', 'abs_value', 'line_width'])
    chord = hv.Chord(chord_data).opts(
        node_color='index',
        edge_color=hv.dim('abs_value'),
        edge_cmap='RdYlBu',  # Heatmap color scheme for edges
        labels='index',
        edge_line_width=hv.dim('line_width'),
        label_text_font_size='10pt',  # Increase the font size of feature names
        bgcolor='white',
        width=500,  # Set the width of the plot
        height=500,  # Set the height of the plot
        node_cmap='Category20'  # Different color for each feature
    )

    if output_file:
        hv.save(chord, output_file, backend='bokeh')
    
    return chord