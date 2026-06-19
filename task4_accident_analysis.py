import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium

# ==========================================
# LOAD DATASET
# ==========================================

print("Loading dataset...")

df = pd.read_csv("US_Accidents_March23.csv")

# Take sample for faster processing
df = df.sample(100000, random_state=42)

print("Dataset Shape:", df.shape)

# ==========================================
# DATA CLEANING
# ==========================================

df = df.drop_duplicates()

# Handle mixed datetime formats
df['Start_Time'] = pd.to_datetime(
    df['Start_Time'],
    format='mixed',
    errors='coerce'
)

# Remove invalid dates
df = df.dropna(subset=['Start_Time'])

# Extract useful features
df['Hour'] = df['Start_Time'].dt.hour
df['Month'] = df['Start_Time'].dt.month
df['Day'] = df['Start_Time'].dt.day_name()

print("\nMissing Values:")
print(df.isnull().sum().sort_values(ascending=False).head(10))

sns.set_style("whitegrid")

# ==========================================
# 1. SEVERITY DISTRIBUTION
# ==========================================

plt.figure(figsize=(8, 5))
sns.countplot(x='Severity', data=df)
plt.title("Accident Severity Distribution")
plt.savefig("severity_distribution.png")
plt.show()

# ==========================================
# 2. ACCIDENTS BY HOUR
# ==========================================

plt.figure(figsize=(12, 6))
sns.histplot(df['Hour'], bins=24, kde=True)
plt.title("Accidents by Hour")
plt.xlabel("Hour")
plt.savefig("accidents_by_hour.png")
plt.show()

# ==========================================
# 3. ACCIDENTS BY DAY
# ==========================================

plt.figure(figsize=(10, 6))

day_order = [
    'Monday',
    'Tuesday',
    'Wednesday',
    'Thursday',
    'Friday',
    'Saturday',
    'Sunday'
]

sns.countplot(
    x='Day',
    data=df,
    order=day_order
)

plt.title("Accidents by Day of Week")
plt.xticks(rotation=30)
plt.savefig("accidents_by_day.png")
plt.show()

# ==========================================
# 4. ACCIDENTS BY MONTH
# ==========================================

plt.figure(figsize=(10, 5))
sns.countplot(x='Month', data=df)
plt.title("Accidents by Month")
plt.savefig("accidents_by_month.png")
plt.show()

# ==========================================
# 5. WEATHER CONDITIONS
# ==========================================

weather = df['Weather_Condition'].value_counts().head(10)

plt.figure(figsize=(12, 6))
sns.barplot(
    x=weather.values,
    y=weather.index
)

plt.title("Top 10 Weather Conditions")
plt.xlabel("Number of Accidents")
plt.savefig("weather_conditions.png")
plt.show()

# ==========================================
# 6. DAY VS NIGHT
# ==========================================

plt.figure(figsize=(8, 5))
sns.countplot(
    x='Sunrise_Sunset',
    data=df
)

plt.title("Day vs Night Accidents")
plt.savefig("day_night_accidents.png")
plt.show()

# ==========================================
# 7. TOP STATES
# ==========================================

top_states = df['State'].value_counts().head(10)

plt.figure(figsize=(12, 6))
sns.barplot(
    x=top_states.values,
    y=top_states.index
)

plt.title("Top 10 States with Most Accidents")
plt.xlabel("Accident Count")
plt.savefig("top_states.png")
plt.show()

# ==========================================
# 8. TEMPERATURE DISTRIBUTION
# ==========================================

plt.figure(figsize=(10, 5))
sns.histplot(
    df['Temperature(F)'].dropna(),
    bins=50,
    kde=True
)

plt.title("Temperature Distribution")
plt.savefig("temperature_distribution.png")
plt.show()

# ==========================================
# 9. VISIBILITY DISTRIBUTION
# ==========================================

plt.figure(figsize=(10, 5))
sns.histplot(
    df['Visibility(mi)'].dropna(),
    bins=30,
    kde=True
)

plt.title("Visibility During Accidents")
plt.savefig("visibility_distribution.png")
plt.show()

# ==========================================
# 10. CORRELATION HEATMAP
# ==========================================

numeric_cols = [
    'Severity',
    'Distance(mi)',
    'Temperature(F)',
    'Humidity(%)',
    'Pressure(in)',
    'Visibility(mi)',
    'Wind_Speed(mph)',
    'Precipitation(in)'
]

corr = df[numeric_cols].corr(numeric_only=True)

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.show()

# ==========================================
# 11. HOTSPOT MAP
# ==========================================

print("\nCreating hotspot map...")

map_df = df[['Start_Lat', 'Start_Lng']].dropna()

map_df = map_df.sample(
    min(3000, len(map_df)),
    random_state=42
)

accident_map = folium.Map(
    location=[
        map_df['Start_Lat'].mean(),
        map_df['Start_Lng'].mean()
    ],
    zoom_start=4
)

for lat, lng in zip(
        map_df['Start_Lat'],
        map_df['Start_Lng']):

    folium.CircleMarker(
        location=[lat, lng],
        radius=2,
        fill=True
    ).add_to(accident_map)

accident_map.save("Accident_Hotspots.html")

print("Hotspot map saved successfully!")

# ==========================================
# RESULTS
# ==========================================

print("\n========== RESULTS ==========")

print("Total Records Analysed:", len(df))

print(
    "Most Common Severity:",
    df['Severity'].mode()[0]
)

print(
    "Peak Accident Hour:",
    df['Hour'].mode()[0]
)

print(
    "Most Accident-Prone State:",
    df['State'].value_counts().idxmax()
)

print(
    "Most Common Weather:",
    df['Weather_Condition'].mode()[0]
)

print("\nAnalysis Completed Successfully!")