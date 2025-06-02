# air-crash-analysis
A comprehensive data analysis of global air crashes using Python. Includes trend analysis, forecasting with Prophet, and survival modeling with Lifelines.

##  About the Dataset

The dataset used in this project was collected from the Aviation Safety Network, a publicly available source of aviation accident and incident information.

###  Time Period

- **Coverage:** January 2000 to December 2024  

###  Data Cleaning and Exclusions

A total of **59 observations were excluded** from the analysis due to missing or unknown event dates. These entries could not be included in time-dependent analyses like trend forecasting and survival modeling.

Details of excluded records:

- **Most had zero reported fatalities**
- **1 event** recorded **5 fatalities**
- **7 events** had **unknown fatality numbers**

These entries were excluded to maintain the **temporal accuracy** of the analysis, which relies on precise date information for time series forecasting and survival analysis.

###  Final Dataset Columns

| Column           | Description |
|------------------|-------------|
| `accident_date`  | Date of the crash (converted to datetime) |
| `aircraft_type`  | Type or model of the aircraft involved |
| `registration`   | Registration number of the aircraft |
| `operator`       | Airline or operator of the aircraft |
| `fatalities`     | Number of deaths reported in the crash |
| `location`       | Geographical location of the crash |
| `year` / `month` | Extracted from `accident_date` for analysis |
| `fatal_crash`    | Boolean: 1 if fatalities > 0, otherwise 0 |

##  Methodology and Analysis Overview

This project performs a comprehensive exploratory data analysis (EDA) and modeling of air crash data using Python. The steps below summarize the methods and techniques applied:

---

### 1.  Data Loading and Preprocessing

- The dataset was loaded using `pandas.read_csv()` with `ISO-8859-1` encoding.
- Column names were standardized for readability and consistency.
- The `accident_date` column was converted to `datetime` format using `pd.to_datetime()`.
- The `fatalities` column was converted to numeric using `pd.to_numeric()`.
- New columns `year` and `month` were extracted from the date for temporal analysis.

---

### 2.  Annual Trends in Crashes and Fatalities

- Crashes and total fatalities were aggregated by year using `groupby()`.
- A dual line plot was created using `seaborn` to visualize:
  - Crashes per year
  - Fatalities per year

---

### 3.  Monthly Crash and Fatality Patterns

- Crashes and fatalities were aggregated by **month (across all years)**.
- A bar plot was used to show:
  - Number of crashes by month
  - Number of fatalities by month
- Month numbers were mapped to abbreviated names using the `calendar` module.

---

### 4.  Aircraft Type Analysis

- Data was grouped by `aircraft_type` to calculate:
  - Total crash count
  - Average fatalities per crash
- The top 15 aircraft types by crash count were visualized using horizontal bar plots for:
  - Crash frequency
  - Average fatalities

---

### 5.  Operator Analysis

- Similar to aircraft analysis, data was grouped by `operator` to compute:
  - Crash count
  - Average fatalities
- The top 15 operators were visualized using bar plots.

---

### 6.  Statistical Testing: Kruskal-Wallis Test

- The **Kruskal-Wallis H-test** (non-parametric) was used to compare **fatality distributions across aircraft types** with more than 5 records.
- This tests whether the median fatality count differs significantly between aircraft types.

---

### 7.  Fatal vs Non-Fatal Crash Analysis

- A new binary column `fatal_crash` was created: 1 if fatalities > 0, else 0.
- Counts of fatal vs non-fatal crashes were calculated.
- Count plots (`seaborn.countplot`) were created to visualize:
  - Fatal vs non-fatal crashes across top 15 aircraft types
  - Fatal vs non-fatal crashes across top 15 operators

---

### 8.  Chi-Square Test for Association

- A **Chi-Square Test** of independence was conducted between `aircraft_type` and `fatal_crash` status.
- A contingency table was created using `pd.crosstab()` limited to the top 10 aircraft types.
- This test determines whether certain aircraft types are more likely to have fatal crashes.

---

### 9.  Time Series Forecasting with Prophet

- Monthly crash counts were aggregated and fed into Facebook's `Prophet` model.
- The model was trained to forecast crashes for the next 24 months.
- Components like **seasonality** and **trend** were plotted to understand crash behavior over time.

---

### 10.  Fatality Trends Over Time

- Monthly total fatalities and average fatalities per crash were calculated.
- A line plot was created to display:
  - Total fatalities per month
  - Average fatalities per crash per month

---

### 11.  Survival Analysis: Time Between Crashes

- The dataset was sorted by date, and the time between consecutive accidents was computed using `.diff()`.
- The **Kaplan-Meier Estimator** from the `lifelines` library was used to model the **survival probability** (i.e., the probability that no accident occurs over a time period).
- A survival function plot shows how frequently crashes occur over time.

---

###  Libraries Used

- `pandas`, `numpy` – data manipulation
- `matplotlib`, `seaborn` – data visualization
- `prophet` – time series forecasting
- `lifelines` – survival analysis
- `scipy.stats` – statistical testing

---


