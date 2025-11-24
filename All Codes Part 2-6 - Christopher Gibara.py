# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 11:05:57 2025

@author: Chris
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df=pd.read_csv("C:/Users/Chris/Downloads/archive/anime_dataset.csv")

#Part 2: Preliminary Steps
    
#a) Initial data inspection
    
#Basic data inspection
print(df.info())        # Dataset structure: types, nulls, memory
print(df.head())        # First 5 rows - data preview
print(df.shape)         # Dimensions: (rows, columns)
print(df.describe())    # Statistical summary for numerical columns
print(df.columns.tolist())  # List of all column names
print(df.sample(5))     # Random sample to check data variety
    
#b) Handle duplicate entries
print(df.duplicated().sum()) #Counting for duplicates (if any)
df= df.drop_duplicates() #Removing all duplicates (if any) then saving it
#There are no duplicates in my dataset
    
#c) Identify and manage missing values
print(df.isnull().sum())
df= df.dropna(subset=['episodes']) #Picking option A) Dropping some rows since 3 is not much
print(df["episodes"].isnull().sum())
df['year'] = df['year'].fillna("Not Announced") #Picking option C) Filling values with "Not Announced"
#Justification: Dropping 3 values is not significant and makes it easy to clean up while dropping 168 rows would be somewhat significant so filling the option with "Not Announced" was the better option
print(df.isnull().sum())
    
#d) Correct data types and formats
    
# Convert numerical columns to fitting data types for analysis
# Using errors='coerce' to handle any invalid values by converting them to NaN values
df['score']= pd.to_numeric(df['score'], errors='coerce') 
df['episodes']= pd.to_numeric(df['episodes'], errors='coerce') 
df['popularity']= pd.to_numeric(df['popularity'], errors='coerce') 
df['members']= pd.to_numeric(df['members'], errors='coerce') 
print(df.dtypes)

#Part 3: Univariate non-graphical EDA
#Numerical Variables univariate analysis

#Calculating score univariate anaylsis with required metrics for analysis
print(df['score'].describe()) # Basic stats including quatiles
print(df['score'].mode()) # Mode
print(df['score'].var())  # Variance
print(df['score'].skew()) # Skewness
print(df['score'].kurtosis()) # Kurtosis
print(df['score'].quantile(0.25)) # Q1
print(df['score'].quantile(0.50)) # Q2
print(df['score'].quantile(0.75)) # Q3

#Calculating episodes univariate anaylsis with required metrics for analysis
print(df['episodes'].describe()) # Basic stats including quatiles
print(df['episodes'].mode()) # Mode
print(df['episodes'].var())  # Variance
print(df['episodes'].skew()) # Skewness
print(df['episodes'].kurtosis()) # Kurtosis
print(df['episodes'].quantile(0.25)) # Q1
print(df['episodes'].quantile(0.50)) # Q2
print(df['episodes'].quantile(0.75)) # Q3

#Calculating popularity univariate anaylsis with required metrics for analysis
print(df['popularity'].describe()) # Basic stats including quatiles
print(df['popularity'].mode()) # Mode
print(df['popularity'].var())  # Variance
print(df['popularity'].skew()) # Skewness
print(df['popularity'].kurtosis()) # Kurtosis
print(df['popularity'].quantile(0.25)) # Q1
print(df['popularity'].quantile(0.50)) # Q2
print(df['popularity'].quantile(0.75)) # Q3

#Calculating members univariate anaylsis with required metrics for analysis
print(df['members'].describe()) # Basic stats including quatiles
print(df['members'].mode()) # Mode
print(df['members'].var())  # Variance
print(df['members'].skew()) # Skewness
print(df['members'].kurtosis()) # Kurtosis
print(df['members'].quantile(0.25)) # Q1
print(df['members'].quantile(0.50)) # Q2
print(df['members'].quantile(0.75)) # Q3

#Categorial variables univariate analysis

#Calculating genres for frequency counts, proportion, mode and unique categories
print(df['genres'].value_counts().head(5)) # Top 5 frequency counts
print(df['genres'].value_counts(normalize=True).round(2)) # Proportions of all genres
print(df['genres'].mode()) # Most frequent genre
print(df['genres'].nunique()) # Number of unique genres

#Calculating studios for frequency counts, proportion, mode and unique categories
print(df['studios'].value_counts().head(5)) # Top 5 frequency counts
print(df['studios'].value_counts(normalize=True).round(2)) # Proportions of all studios
print(df['studios'].mode()) # Most appearing studio
print(df['studios'].nunique()) # Number of different studios

#Calculating year for frequency counts, proportion, mode and unique categories
print(df['year'].value_counts().head(5)) # Top 5 frequency counts
print(df['year'].value_counts(normalize=True).round(2)) # Proportions of all years
print(df['year'].mode()) # Most released animes during the year
print(df['year'].nunique()) # Number of different years

#Part 4: Univariate graphical EDA
# Distribution plots using features: bins, conditioning, histrogram, KDE

# Distribution plot for score 

# Conditioning for popular genres in dataset (Action & Comedy)
df['is_genre_popular'] = df['genres'].str.contains('Action|Comedy', na=False)

sns.histplot(df, x='score',
             bins = 25, # a) Custom bins amount e.g. 25
             hue='is_genre_popular', # b) Conditioning on popular genres
             stat='density', # c) Normalized histogram to compare sizes
             kde = True,  # D) KDE
             kde_kws={'bw_adjust' : 1.2}) # Smoothing bandwidth
# Titles on score distribution graph
plt.title('Action/Comedy vs Other Genres Distribution')
plt.xlabel('Score')
plt.ylabel('Density')
plt.legend(title='Popular Genre?', labels=['Other Genres', 'Action/Comedy'])
plt.show()

# Distribution plot for episodes 

# Conditioning for episodes greater than 12 since a season contains 12 episodes
df['is_long_series'] = df['episodes'] > 12

sns.histplot(df[df['episodes'] <= 100], x='episodes',
             bins=25,  # a) Custom bins amount e.g. 25
             hue='is_long_series', # b) Conditioning on series length
             stat='density', # c) Normalized histogram to compare sizes
             kde=True,   # d) KDE
             kde_kws={'bw_adjust': 1.2}) # d) Smoothing bandwidth
# Titles on episodes distribution graph
plt.title('Episode Distribution: Short vs Long Series')
plt.xlabel('Number of Episodes')
plt.ylabel('Density')
plt.legend(title='Long Series?', labels=['Short (1-12)', 'Long (13+)'])
plt.show()

# Distribution plot for popularity 

# Conditioning high vs low popularity (less than or equal to 7.5 is popular)
df['is_high_rated'] = df['score'] >= 7.5

sns.histplot(df, x='popularity', 
             bins=25, # a) Custom bins amount e.g. 25
             hue='is_high_rated', # b) Conditioning on series rating score
             stat='density', # c) Normalized histogram to compare size
             kde=True,  # d) KDE
             kde_kws={'bw_adjust': 1.2}) # d) Smoothing bandwidth
# Titles on popularity distribution graph
plt.title('Popularity Distribution: High-Rated vs Low-Rated Anime')
plt.xlabel('Popularity Rank (lower = more popular)')
plt.ylabel('Density')
plt.legend(title='High Rated?', labels=['Score < 7.5', 'Score ≥ 7.5'])
plt.show()

#Distribution plot for member count

# Conditioning for high member count against low member count
df['is_popular'] = df['members'] > df['members'].median()

sns.histplot(df, x='members',
             bins=25, # a) Custom bins amount e.g. 25
             hue='is_popular', # b) Conditioning on member count
             stat='density', # c) Normalized histogram to compare size
             kde=True, # d) KDE
             kde_kws={'bw_adjust': 1.2})  # d) Smoothing bandwidth
# Titles on member count distribution graph
plt.title('Member Distribution: Popular vs Niche Anime')
plt.legend(title='Many Members?', labels=['Below Median', 'Above Median'])
plt.show()

#Part 5: Multivariate non-graphical EDA
# Creating conditions that are meaningful categories to prepare for questions to be answered

#Filtering for animes with release data 2015+
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df['era_is_contemporary'] = df['year'] >= 2015

#Foraging for only comedy and action genres
df['is_comedy_action'] = df['genres'].str.contains('Action|Comedy', na=False)

#Filtering for top 5 studios by their count
top_5_studios = df['studios'].value_counts().head(5).index
df['is_big_studio'] = df['studios'].isin(top_5_studios)


# a) & b) Modern era vs action/comedy genres (with percentages)
era_is_modern = pd.crosstab(df['era_is_contemporary'], df['is_comedy_action'], normalize=True) * 100
print(era_is_modern.round(2))

# a) & b) Contemporary era vs Major Studios (with percentages)
modern_studio = pd.crosstab(df['era_is_contemporary'], df['is_big_studio'], normalize=True) * 100
print(modern_studio.round(2))

# c) Three-way: Era, Genre, Studio assiciations
three_way_table = pd.crosstab([df['era_is_contemporary'], df['is_comedy_action']], df['is_big_studio'])
print(three_way_table)

# Part 6: Multivariate graphical EDA

# 6.1 Visualizing Statistical Relationships

# 6.1a: Faceting feature using col parameter in relplot()
sns.relplot(df[df['members'] <= 500000], 
            x='score', y='members', 
            col='era_is_contemporary')
plt.suptitle("6.1a: Score vs Members by Era (Faceted)")
plt.show()

# 6.1b: Plot representing 5 variables at once (x, y, hue, size, col)
sns.relplot(df[df['members'] <= 500000], 
            x='score', y='members', 
            hue='is_comedy_action', 
            size='episodes', 
            col='era_is_contemporary')
plt.suptitle("6.1b: Five Variables - Score vs Members by Genre, Episodes, and Era")
plt.show()

# 6.1c: Linear regression plot
filtered_df = df[df['episodes'] > 10]
sns.lmplot(filtered_df, x='score', y='episodes')
plt.title("6.1c: Linear Regression - Score vs Episodes")
plt.show()

# 6.2 Visualizing Categorical Data

# 6.2a: Categorical scatter plot with jitter disabled
sns.catplot(df, x='era_is_contemporary', y='score', jitter=False, kind='strip')
plt.title("6.2a: Score by Era (Jitter Disabled)")
plt.show()

# 6.2b: Beeswarm plot representing 3 variables
sns.catplot(df, x='era_is_contemporary', y='score', hue='is_comedy_action', kind='swarm')
plt.title("6.2b: Beeswarm Plot - Score by Era and Genre")
plt.show()

# 6.2c: Boxen plot showing the shape of the distribution
sns.catplot(df, x='is_big_studio', y='popularity', kind='boxen')
plt.title("6.2c: Boxen Plot - Popularity by Studio Size")
plt.show()

# 6.2d: Split violin plot representing 3 variables with bandwidth adjusted
sns.catplot(df, x='is_big_studio', y='popularity', hue='era_is_contemporary', split=True, kind='violin')
plt.title("6.2d: Split Violin Plot - Popularity by Studio and Era")
plt.show()

# 6.2e: Violin plot with scatter points inside the violin shapes
sns.catplot(df, x='is_big_studio', y='episodes', inner='stick', kind='violin')
plt.title("6.2e: Violin Plot with Stick Points - Episodes by Studio")
plt.show()

# 6.2f: Point plot representing 3 variables showing 90% confidence intervals and lines in dashed style
sns.catplot(df, x='is_big_studio', y='popularity', hue='era_is_contemporary',
            kind='point', errorbar=('ci', 90), linestyle='--')
plt.title("6.2f: Point Plot - Popularity by Studio and Era (90% CI)")
plt.show()

# 6.2g: Bar plot showing the number of observations in each category
sns.catplot(df, x='is_comedy_action', kind='count')
plt.title("6.2g: Bar Plot - Count of Action/Comedy vs Other Genres")
plt.show()

# 6.3 Visualizing Bivariate Distributions

# 6.3a: Heatmap plot representing 2 variables with color intensity bar and adjusted bin width
sns.histplot(data=df, x='popularity', y='score', bins=30, cmap='coolwarm', cbar=True)
plt.title("6.3a: Heatmap - Popularity vs Score Distribution")
plt.show()

# 6.3b: Distribution plot with 2 variables using bivariate density contours 
sns.displot(df, x='popularity', y='score',  
            levels=10,                    # Adjusted amount of curves
            thresh=0.1,                   # Adjusted lowest level
            bw_adjust=0.8,                # Smoothing bandwidth adjustment
            kind='kde')
plt.title("6.3b: KDE Plot - Popularity vs Score with Adjusted Contours")
plt.show()
