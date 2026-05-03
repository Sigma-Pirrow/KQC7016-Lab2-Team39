import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

DATA_FILE = "WorldEnergy.csv"

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 10

df = pd.read_csv(DATA_FILE)
df.columns = [str(col).strip() for col in df.columns]

entity_col = None
for col in ["country", "Country", "Entity", "entity"]:
    if col in df.columns:
        entity_col = col
        break

year_col = None
for col in ["year", "Year", "YEAR"]:
    if col in df.columns:
        year_col = col
        break

if entity_col is None:
    raise ValueError("No entity column found")

if year_col is None:
    raise ValueError("No year column found")

selected_entities = {
    'World',
    'Asia',
    'Europe',
    'North America',
    'South America',
    'Africa',
    'Oceania',
    'European Union (27)',
    'Low-income countries',
    'Lower-middle-income countries',
    'Upper-middle-income countries',
    'High-income countries'
}

selected_numeric = [
    'electricity_demand',
    'fossil_energy_per_capita',
    'low_carbon_share_energy',
    'solar_electricity',
    'wind_electricity',
    'greenhouse_gas_emissions'
]

available_numeric = [col for col in selected_numeric if col in df.columns]

df = df[df[entity_col].isin(selected_entities)]
df = df[(df[year_col] >= 2000) & (df[year_col] <= 2022)]
df = df[[entity_col, year_col] + available_numeric].copy()

print(df.head())
print(df.isnull().sum())
print(df.describe(include="all").transpose())

if 'electricity_demand' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x=year_col, y='electricity_demand', hue=entity_col, marker="o")
    plt.title("Electricity Demand Trend by Region / Income Group")
    plt.xlabel("Year")
    plt.ylabel("Electricity Demand")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

if 'low_carbon_share_energy' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x=year_col, y='low_carbon_share_energy', hue=entity_col, marker="o")
    plt.title("Low-Carbon Share of Energy Trend")
    plt.xlabel("Year")
    plt.ylabel("Low-Carbon Share Energy")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

if 'greenhouse_gas_emissions' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x=year_col, y='greenhouse_gas_emissions', hue=entity_col, marker="o")
    plt.title("Greenhouse Gas Emissions Trend")
    plt.xlabel("Year")
    plt.ylabel("Greenhouse Gas Emissions")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

if 'solar_electricity' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x=year_col, y='solar_electricity', hue=entity_col, marker="o")
    plt.title("Solar Electricity Trend by Region / Income Group")
    plt.xlabel("Year")
    plt.ylabel("Solar Electricity")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

if 'wind_electricity' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x=year_col, y='wind_electricity', hue=entity_col, marker="o")
    plt.title("Wind Electricity Trend by Region / Income Group")
    plt.xlabel("Year")
    plt.ylabel("Wind Electricity")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

if 'fossil_energy_per_capita' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x=entity_col, y='fossil_energy_per_capita')
    plt.title("Distribution of Fossil Energy Per Capita")
    plt.xlabel("Entity")
    plt.ylabel("Fossil Energy Per Capita")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

if len(available_numeric) >= 2:
    corr = df[available_numeric].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Selected Variables")
    plt.tight_layout()
    plt.show()

print("Final columns used:")
print([entity_col, year_col] + available_numeric)

print("Final entities used:")
print(sorted(df[entity_col].dropna().unique()))

print("Year range:")
print(df[year_col].min(), df[year_col].max())