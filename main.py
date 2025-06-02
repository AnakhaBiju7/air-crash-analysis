import pandas as pd
df = pd.read_csv("air crash data.csv", encoding='ISO-8859-1')
df

import pandas as pd

# Load the dataset
df = pd.read_csv("air crash data.csv", encoding='ISO-8859-1')

# Rename columns for consistency
df.columns = ['accident_date', 'aircraft_type', 'registration', 'operator', 'fatalities', 'location']

# Convert accident_date to datetime
df['accident_date'] = pd.to_datetime(df['accident_date'], format='%d-%b-%y', errors='coerce')

# Convert fatalities to numeric (if not already)
df['fatalities'] = pd.to_numeric(df['fatalities'], errors='coerce')

# Extract Year and Month
df['year'] = df['accident_date'].dt.year
df['month'] = df['accident_date'].dt.month

# Check basic info
df.info()

import matplotlib.pyplot as plt
import seaborn as sns

# Annual crash and fatality counts
annual_stats = df.groupby('year').agg({
    'accident_date': 'count',
    'fatalities': 'sum'
}).rename(columns={'accident_date': 'crashes'})

# Plot
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.lineplot(data=annual_stats, x=annual_stats.index, y='crashes', marker='o')
plt.title('Crashes per Year')
plt.xlabel('Year')
plt.ylabel('Number of Crashes')

plt.subplot(1, 2, 2)
sns.lineplot(data=annual_stats, x=annual_stats.index, y='fatalities', marker='o', color='red')
plt.title('Fatalities per Year')
plt.xlabel('Year')
plt.ylabel('Number of Fatalities')

plt.tight_layout()
plt.show()

# Monthly analysis (across all years)
monthly_stats = df.groupby('month').agg({
    'accident_date': 'count',
    'fatalities': 'sum'
}).rename(columns={'accident_date': 'crashes'})

# Month name mapping
import calendar
monthly_stats.index = [calendar.month_abbr[m] for m in monthly_stats.index]

# Plot
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.barplot(x=monthly_stats.index, y='crashes', data=monthly_stats)
plt.title('Crashes by Month')
plt.xlabel('Month')
plt.ylabel('Number of Crashes')

plt.subplot(1, 2, 2)
sns.barplot(x=monthly_stats.index, y='fatalities', data=monthly_stats, color='salmon')
plt.title('Fatalities by Month')
plt.xlabel('Month')
plt.ylabel('Number of Fatalities')

plt.tight_layout()
plt.show()

# Group by aircraft type
aircraft_stats = df.groupby('aircraft_type').agg({
    'accident_date': 'count',
    'fatalities': 'mean'
}).rename(columns={'accident_date': 'crash_count', 'fatalities': 'avg_fatalities'})

# Sort descending by crash count
aircraft_stats_sorted = aircraft_stats.sort_values(by='crash_count', ascending=False).head(15)

# Plot crash counts
plt.figure(figsize=(14, 6))
sns.barplot(y=aircraft_stats_sorted.index, x='crash_count', data=aircraft_stats_sorted, palette='Blues_d')
plt.title('Top 15 Aircraft Types by Crash Count')
plt.xlabel('Number of Crashes')
plt.ylabel('Aircraft Type')
plt.tight_layout()
plt.show()

# Plot avg fatalities
plt.figure(figsize=(14, 6))
sns.barplot(y=aircraft_stats_sorted.index, x='avg_fatalities', data=aircraft_stats_sorted, palette='Reds_d')
plt.title('Top 15 Aircraft Types by Average Fatalities per Crash')
plt.xlabel('Average Fatalities')
plt.ylabel('Aircraft Type')
plt.tight_layout()
plt.show()

# Group by operator
operator_stats = df.groupby('operator').agg({
    'accident_date': 'count',
    'fatalities': 'mean'
}).rename(columns={'accident_date': 'crash_count', 'fatalities': 'avg_fatalities'})

# Sort descending by crash count
operator_stats_sorted = operator_stats.sort_values(by='crash_count', ascending=False).head(15)

# Plot crash counts by operator
plt.figure(figsize=(14, 6))
sns.barplot(y=operator_stats_sorted.index, x='crash_count', data=operator_stats_sorted, palette='Greens_d')
plt.title('Top 15 Operators by Crash Count')
plt.xlabel('Number of Crashes')
plt.ylabel('Operator')
plt.tight_layout()
plt.show()

# Plot avg fatalities by operator
plt.figure(figsize=(14, 6))
sns.barplot(y=operator_stats_sorted.index, x='avg_fatalities', data=operator_stats_sorted, palette='Oranges_d')
plt.title('Top 15 Operators by Average Fatalities per Crash')
plt.xlabel('Average Fatalities')
plt.ylabel('Operator')
plt.tight_layout()
plt.show()

from scipy.stats import kruskal

# Prepare list of fatalities arrays by aircraft type
groups = [group['fatalities'].dropna().values for name, group in df.groupby('aircraft_type') if len(group) > 5]

# Perform Kruskal-Wallis test (non-parametric)
stat, p = kruskal(*groups)
print(f"Kruskal-Wallis H-test statistic: {stat:.3f}, p-value: {p:.3e}")

if p < 0.05:
    print("There is a statistically significant difference in fatality rates between aircraft types.")
else:
    print("No statistically significant difference in fatality rates between aircraft types.")

df['fatal_crash'] = (df['fatalities'] > 0).astype(int)

print(df['fatal_crash'].value_counts())
print(df.groupby('fatal_crash')['fatalities'].describe())

plt.figure(figsize=(14,6))
sns.countplot(data=df, y='aircraft_type', hue='fatal_crash', order=df['aircraft_type'].value_counts().iloc[:15].index)
plt.title('Top 15 Aircraft Types in Fatal vs Non-Fatal Crashes')
plt.xlabel('Count')
plt.ylabel('Aircraft Type')
plt.legend(title='Fatal Crash', labels=['No', 'Yes'])
plt.tight_layout()
plt.show()

plt.figure(figsize=(14,6))
top_operators = df['operator'].value_counts().iloc[:15].index
sns.countplot(data=df[df['operator'].isin(top_operators)], y='operator', hue='fatal_crash')
plt.title('Top 15 Operators: Fatal vs Non-Fatal Crash Counts')
plt.xlabel('Count')
plt.ylabel('Operator')
plt.legend(title='Fatal Crash', labels=['No', 'Yes'])
plt.tight_layout()
plt.show()

from scipy.stats import chi2_contingency

# Contingency table for aircraft type and fatal_crash (limit to top 10 types to reduce sparsity)
top_aircraft = df['aircraft_type'].value_counts().iloc[:10].index
contingency = pd.crosstab(df[df['aircraft_type'].isin(top_aircraft)]['aircraft_type'], df['fatal_crash'])

chi2, p, dof, ex = chi2_contingency(contingency)

print(f"Chi-square statistic: {chi2:.2f}")
print(f"P-value: {p:.3e}")
if p < 0.05:
    print("Significant association between aircraft type and fatal crash status")
else:
    print("No significant association between aircraft type and fatal crash status")

# Aggregate crashes per month-year
monthly_crashes = df.groupby(pd.Grouper(key='accident_date', freq='M')).size().reset_index(name='crash_count')

# Check data
monthly_crashes.head()

!pip install prophet --quiet
from prophet import Prophet

prophet_df = monthly_crashes.rename(columns={'accident_date': 'ds', 'crash_count': 'y'})

model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
model.fit(prophet_df)

# Forecast next 24 months
future = model.make_future_dataframe(periods=24, freq='M')
forecast = model.predict(future)

# Check forecast head
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

fig1 = model.plot(forecast)
plt.title('Monthly Crash Count Forecast')
plt.xlabel('Date')
plt.ylabel('Crash Count')
plt.show()

fig2 = model.plot_components(forecast)
plt.show()

# Monthly aggregation of total fatalities
monthly_fatalities = df.groupby(pd.Grouper(key='accident_date', freq='M'))['fatalities'].sum().reset_index()

# Monthly aggregation of crash counts (for average)
monthly_counts = df.groupby(pd.Grouper(key='accident_date', freq='M')).size().reset_index(name='crash_count')

# Merge to calculate average fatalities per crash per month
fatality_df = monthly_fatalities.merge(monthly_counts, on='accident_date')
fatality_df['avg_fatalities_per_crash'] = fatality_df['fatalities'] / fatality_df['crash_count']

fatality_df.head()

import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))
plt.plot(fatality_df['accident_date'], fatality_df['fatalities'], label='Total Fatalities per Month')
plt.plot(fatality_df['accident_date'], fatality_df['avg_fatalities_per_crash'], label='Average Fatalities per Crash')
plt.title('Monthly Fatalities and Average Fatalities per Crash')
plt.xlabel('Date')
plt.ylabel('Fatalities')
plt.legend()
plt.show()

!pip install lifelines

import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

df = pd.read_csv("air crash data.csv", encoding='ISO-8859-1')

# Convert 'acc. date' to datetime format (adjust format as needed)
df['acc. date'] = pd.to_datetime(df['acc. date'], format='%d-%b-%y')

# Sort by accident date
df = df.sort_values('acc. date')

# Calculate time intervals between consecutive accidents (in days)
df['time_to_next'] = df['acc. date'].diff().dt.days.shift(-1)

# Drop the last row (no next accident to compare)
df = df.dropna(subset=['time_to_next'])

# Fit the Kaplan-Meier estimator
kmf = KaplanMeierFitter()
kmf.fit(durations=df['time_to_next'], event_observed=[1]*len(df))  # All intervals are "events"

# Plot the survival function
kmf.plot_survival_function()
plt.title('Survival Function: Time Between Accidents')
plt.xlabel('Time (days)')
plt.ylabel('Probability of No Accident')
plt.show()

