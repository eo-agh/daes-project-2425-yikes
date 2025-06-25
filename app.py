import streamlit as st
import pandas as pd
import h3
import pydeck as pdk
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Vehicle Speed Hexagonal Heatmap")
st.markdown(
    """
This application visualizes the average speed of vehicles across a hexagonal grid on a map.
It reads data from selected local Parquet files and converts coordinates to hex indices.

- **Green hexagons** represent higher average speeds.
- **Red hexagons** represent lower average speeds.

**Speeds are shown in meters per second (m/s).**
    """
)

# --- File Paths ---
DATA_SOURCES = {
    'med2med': r"C:\Users\bugaj\OneDrive\Pulpit\Studia\Analiza-danych-w-naukach-o-Ziemi\daneKaretki\final\med2med.parquet",
    'med2nonmed': r"C:\Users\bugaj\OneDrive\Pulpit\Studia\Analiza-danych-w-naukach-o-Ziemi\daneKaretki\final\med2nonmed.parquet",
    'nonmed2med': r"C:\Users\bugaj\OneDrive\Pulpit\Studia\Analiza-danych-w-naukach-o-Ziemi\daneKaretki\final\nonmed2med.parquet",
}

# --- Sidebar: Data Sources ---
st.sidebar.header("Select Data Sources")
selected_keys = [
    k for k in DATA_SOURCES
    if st.sidebar.checkbox(f"Include {k}.parquet", value=True)
]
if not selected_keys:
    st.error("Please select at least one data source.")
    st.stop()

# --- Cache load and parse all data ---
@st.cache_data
def load_all_sources(sources):
    cols = [
        'SegmentPoints', 'SegmentSpeeds_mps', 'Czas wezwania', 'Powód wezwania',
        'Kod pilności', 'Identyfikator pojazdu', 'Rodzaj wyjazdu 0- na sygnale, 1 -zwykly',
        'Typ zespolu', 'Określenie wieku pacjenta 0- dziecko, 1 - dorosly'
    ]
    frames = []
    for key, path in sources.items():
        try:
            df = pd.read_parquet(path, columns=cols)
            df['source'] = key
            # parse timestamp
            df['Czas wezwania'] = pd.to_datetime(df['Czas wezwania'], format="%Y-%m-%d %H:%M:%S", errors='coerce')
            frames.append(df)
        except Exception as e:
            st.warning(f"Error loading {key}: {e}")
    if not frames:
        return pd.DataFrame(columns=cols + ['source'])
    return pd.concat(frames, ignore_index=True)

# --- Load Data ---
combined_raw = load_all_sources(DATA_SOURCES)
if combined_raw.empty:
    st.error("No data loaded. Check file paths and formats.")
    st.stop()

# --- Sidebar: Advanced Filters ---
st.sidebar.header("Filters")
# Date range (year-month)
date_range = st.sidebar.date_input(
    label="Select date range", min_value=combined_raw['Czas wezwania'].min().date(),
    max_value=combined_raw['Czas wezwania'].max().date(),
    value=(combined_raw['Czas wezwania'].min().date(), combined_raw['Czas wezwania'].max().date())
)
# Day of week selection
days = st.sidebar.multiselect(
    "Day of Week", ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],
    default=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
)
# Hour of day range
hour_range = st.sidebar.slider(
    "Hour of Day", min_value=0, max_value=23, value=(0,23)
)

# Categorical filters with <=3 unique (checkboxes)
small_filters = {
    'Kod pilności': [],
    'Rodzaj wyjazdu 0- na sygnale, 1 -zwykly': [],
    'Typ zespolu': [],
    'Określenie wieku pacjenta 0- dziecko, 1 - dorosly': []
}
for col in small_filters:
    st.sidebar.subheader(col)
    for val in combined_raw[col].dropna().unique():
        if st.sidebar.checkbox(f"{col}: {val}", value=True):
            small_filters[col].append(val)

# Multi-select dropdown for Powód wezwania
reason_filter = st.sidebar.multiselect(
    "Powód wezwania", combined_raw['Powód wezwania'].dropna().unique(),
    default=combined_raw['Powód wezwania'].dropna().unique()
)

# --- Apply Filters ---
df = combined_raw.copy()
# date range
start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
df = df[(df['Czas wezwania'] >= start_date) & (df['Czas wezwania'] <= end_date)]
# day of week
if days:
    df = df[df['Czas wezwania'].dt.day_name().isin(days)]
# hour
df = df[(df['Czas wezwania'].dt.hour >= hour_range[0]) & (df['Czas wezwania'].dt.hour <= hour_range[1])]
# small categorical
for col, allowed in small_filters.items():
    df = df[df[col].isin(allowed)]
# large categorical
if reason_filter:
    df = df[df['Powód wezwania'].isin(reason_filter)]
# source filter
filtered = df[df['source'].isin(selected_keys)]

if filtered.empty:
    st.warning("No data after filtering. Adjust filters.")
    st.stop()

# --- Cache and Process Hexagons ---
@st.cache_data
def process_data_into_hexagons(_df, hex_resolution):
    exploded = _df.explode(['SegmentPoints','SegmentSpeeds_mps'])
    exploded['SegmentSpeeds_mps'] = pd.to_numeric(exploded['SegmentSpeeds_mps'], errors='coerce')
    exploded = exploded.dropna(subset=['SegmentSpeeds_mps'])
    exploded = exploded[(exploded['SegmentSpeeds_mps']>0)&(exploded['SegmentSpeeds_mps']<=60)]
    if exploded.empty:
        return pd.DataFrame(columns=['h3_hexagon','avg_speed'])
    exploded['h3_hexagon'] = exploded.apply(
        lambda r: h3.latlng_to_cell(r['SegmentPoints'][1], r['SegmentPoints'][0], hex_resolution),
        axis=1
    )
    hex_df = exploded.groupby('h3_hexagon')['SegmentSpeeds_mps'].mean().reset_index()
    hex_df.rename(columns={'SegmentSpeeds_mps':'avg_speed'}, inplace=True)
    return hex_df

hexagon_resolution = st.sidebar.slider(
    label="Hexagon Resolution", min_value=5, max_value=10, value=9,
    help="Higher values create smaller hexagons."
)

hex_speed_df = process_data_into_hexagons(filtered, hexagon_resolution)
if hex_speed_df.empty:
    st.warning("No hexagon data generated. Check resolution or data.")
    st.stop()

# --- Visualization ---
hex_speed_df['avg_speed_rounded'] = hex_speed_df['avg_speed'].round(2)
min_speed, max_speed = hex_speed_df['avg_speed'].min(), hex_speed_df['avg_speed'].max()
def speed_color(speed):
    norm = (speed - min_speed) / (max_speed - min_speed + 1e-5)
    red = int((1 - norm) * 255)
    green = int(norm * 255)
    return [red, green, 0, 180]
hex_speed_df['fill_color'] = hex_speed_df['avg_speed'].apply(speed_color)

view = pdk.ViewState(latitude=50.0647, longitude=19.9450, zoom=8)
layer = pdk.Layer(
    'H3HexagonLayer', data=hex_speed_df,
    get_hexagon='h3_hexagon', get_fill_color='fill_color', pickable=True, auto_highlight=True
)
tooltip = {"html": "<b>Avg Speed:</b> {avg_speed_rounded} m/s", "style": {"backgroundColor": "steelblue", "color": "white"}}
st.pydeck_chart(pdk.Deck(map_style='mapbox://styles/mapbox/light-v9', initial_view_state=view, layers=[layer], tooltip=tooltip))

# Histograms
c1, c2 = st.columns(2)
with c1:
    st.subheader("Avg Speeds (hexagons)")
    fig, ax = plt.subplots()
    ax.hist(hex_speed_df['avg_speed'], bins=30, edgecolor='black')
    ax.set(xlabel="Average Speed (m/s)",ylabel="Hexagon Count",title="Avg Speed Distribution")
    st.pyplot(fig, use_container_width=True)
with c2:
    st.subheader("Raw Segment Speeds")
    all_sp = [s for speeds in filtered['SegmentSpeeds_mps'] for s in speeds if 0< s<=60]
    fig2, ax2 = plt.subplots()
    ax2.hist(all_sp, bins=30, edgecolor='black')
    ax2.set(xlabel="Speed (m/s)",ylabel="Count",title="Raw Speeds Distribution")
    st.pyplot(fig2, use_container_width=True)

# Data Inspector
st.sidebar.subheader("Data Inspector")
if st.sidebar.checkbox("Show Raw Data"):
    st.dataframe(filtered)
if st.sidebar.checkbox("Show Hexagon Data"):
    st.dataframe(hex_speed_df)
