import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import calendar
import os
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Energy Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide default Streamlit UI elements
hide_streamlit_style = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    .stAlert {display:none;}
    div[data-testid="stToolbar"] {display: none;}
    div[data-testid="stDecoration"] {display: none;}
    div[data-testid="stStatusWidget"] {display: none;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Define color scheme
app_colors = {
    "background": "#f5f7fa",
    "header": "#222831",
    "text": "#393e46",
    "card_bg": "#ffffff",
    "dropdown_bg": "#f8f9fa",
    "accent": "#4ecca3",
    "highlight": "#ff5722"
}

# MPAN to Hotel mapping
mpan_to_hotel = {
 
 "1200050416050": "CIER", "1200031051738": "CIV", "2500021277783": "Westin",
    "2500021348111": "Westin", "1200050416078": "CIER", "1200050416069": "CIER",
    "1200010120786": "Canopy", "1200051315859": "Camden", "2500021281362": "Canopy",
    "1200031051630": "CIV", "1200029368322": "CIER", "1200029368721": "CIER",
    "1200029368759": "CIER", "1200029368712": "CIER", "1200029368698": "CIER",
    "1200029368703": "CIER", "1200029368340": "CIER", "1200029368642": "CIER",
    "1200029368651": "CIER", "1200029368633": "CIER", "1200061897284": "Canopy",
    "1200029368350": "CIER", "1200052502710": "EH", "1200031051640": "CIV",
    "1050000997145": "St Albans", "2500021277890": "Westin", "1200050416087": "CIER",
    "1200029368270": "CIER", "2500021281371": "Canopy"
}

@st.cache_data(show_spinner=False)
def load_data():
    """Load and process data from Excel files"""
    data_path = 'data'
    if not os.path.exists(data_path):
        st.error(f"Data directory not found at: {data_path}")
        return None
        
    excel_files = [f for f in os.listdir(data_path) if f.endswith('.xlsx')]
    if not excel_files:
        st.error("No Excel files found in data directory")
        return None
        
    df_list = []
    with st.spinner('Loading data...'):
        for file in excel_files:
            try:
                filepath = os.path.join(data_path, file)
                data = pd.read_excel(
                    filepath, 
                    sheet_name='D0036IndividualAll', 
                    skiprows=1,
                    engine='openpyxl'
                )
                
                # Basic data processing
                if "Reading Date" in data.columns:
                    data = data.rename(columns={"Reading Date": "Date", "Total": "Total Usage"})
                
                # Process half-hourly columns
                hh_columns = [f'HH{i}' for i in range(1, 49)]
                time_labels = []
                for i in range(48):
                    hour = i // 2
                    minute = "30" if i % 2 else "00"
                    time_labels.append(f"{str(hour).zfill(2)}:{minute}")
                
                # Safe column renaming
                data = data.copy()
                data.loc[:, "Date"] = pd.to_datetime(data["Date"])
                data.loc[:, "Total Usage"] = pd.to_numeric(data["Total Usage"], errors='coerce')
                data.loc[:, "MPAN"] = data["MPAN"].astype(str).str.strip()
                data.loc[:, "Hotel"] = data["MPAN"].map(mpan_to_hotel).fillna("Unknown")
                
                # Rename HH columns
                rename_dict = dict(zip(hh_columns, time_labels))
                data = data.rename(columns=rename_dict)
                
                # Keep only necessary columns
                columns_to_keep = ["Date", "MPAN", "Total Usage", "Hotel"] + time_labels
                df_list.append(data[columns_to_keep])
                
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
                continue
    
    if df_list:
        combined_data = pd.concat(df_list, ignore_index=True)
        combined_data = combined_data.dropna(subset=["Date", "Total Usage", "MPAN"])
        
        # Add time-based columns
        combined_data.loc[:, "Year"] = combined_data["Date"].dt.year
        combined_data.loc[:, "Month"] = combined_data["Date"].dt.month
        combined_data.loc[:, "Day of Week"] = combined_data["Date"].dt.day_name()
        
        # Get time columns for grouping
        time_columns = [f"{str(hour).zfill(2)}:{minute}" 
                       for hour in range(24) 
                       for minute in ['00', '30']]
        
        # Group by Date and Hotel to sum values for multiple MPANs
        agg_columns = ['Total Usage'] + time_columns
        grouped_data = combined_data.groupby(['Date', 'Hotel'], as_index=False)[agg_columns].sum()
        
        # Reapply time-based columns after grouping
        grouped_data.loc[:, "Year"] = grouped_data["Date"].dt.year
        grouped_data.loc[:, "Month"] = grouped_data["Date"].dt.month
        grouped_data.loc[:, "Day of Week"] = grouped_data["Date"].dt.day_name()
        
        return grouped_data
    return None

def create_heatmap(data, selected_hotel, selected_year, selected_month):
    """Create enhanced heatmap from half-hourly data"""
    # Filter data
    mask = (
        (data["Hotel"] == selected_hotel) & 
        (data["Year"] == selected_year) & 
        (data["Month"] == selected_month)
    )
    filtered_data = data.loc[mask].copy()
    
    # Get half-hourly columns
    time_cols = [f"{str(hour).zfill(2)}:{'00' if i == 0 else '30'}"
                 for hour in range(24) for i in range(2)]
    
    # Ensure numeric values
    for col in time_cols:
        filtered_data[col] = pd.to_numeric(filtered_data[col], errors='coerce')
    
    # Add day of week if not present
    if "Day of Week" not in filtered_data.columns:
        filtered_data["Day of Week"] = filtered_data["Date"].dt.day_name()
    
    # Calculate average usage for each time period and day
    pivot_data = pd.DataFrame()
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    for day in days_order:
        day_data = filtered_data[filtered_data["Day of Week"] == day]
        if not day_data.empty:
            pivot_data[day] = day_data[time_cols].mean()
    
    # Transpose to get days as rows and times as columns
    pivot_data = pivot_data.T
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=time_cols,
        y=days_order,
        colorscale="Blues",
        colorbar=dict(title="Usage (kWh)"),
        hoverongaps=False,
        hovertemplate="Day: %{y}<br>Time: %{x}<br>Usage: %{z:.2f} kWh<extra></extra>"
    ))
    
    # Update layout
    fig.update_layout(
        title=f"Average Half-Hourly Usage by Day - {calendar.month_name[selected_month]} {selected_year}",
        xaxis_title="Time of Day",
        yaxis_title="Day of Week",
        xaxis=dict(
            tickangle=-45,
            tickmode="array",
            ticktext=[f"{hour:02d}:00" for hour in range(24)],
            tickvals=[f"{hour:02d}:00" for hour in range(24)]
        ),
        height=500
    )
    
    return fig
def main():
    st.title("Electricity Usage Dashboard by Hotel")
    
    # Load data
    data = load_data()
    if data is None:
        st.stop()
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Hotel selection
    selected_hotel = st.sidebar.selectbox(
        "Select Hotel",
        options=sorted(data["Hotel"].unique())
    )
    
    # Filter data for selected hotel
    hotel_data = data.loc[data["Hotel"] == selected_hotel].copy()
    
    # Year and Month selection
    years = sorted(hotel_data["Year"].unique())
    months = sorted(hotel_data["Month"].unique())
    
    selected_year = st.sidebar.selectbox("Select Year", years, index=len(years)-1)
    selected_month = st.sidebar.selectbox(
        "Select Month",
        months,
        format_func=lambda x: calendar.month_name[x]
    )
    
    # Filter for selected month
    mask = (hotel_data["Year"] == selected_year) & (hotel_data["Month"] == selected_month)
    monthly_data = hotel_data.loc[mask].copy()
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    # Monthly Total
    monthly_total = monthly_data["Total Usage"].sum()
    col1.metric(
        "Monthly Total",
        f"{monthly_total:,.2f} kWh"
    )
    
    # YTD Total
    ytd_mask = (hotel_data["Year"] == selected_year) & (hotel_data["Date"] <= monthly_data["Date"].max())
    ytd_total = hotel_data.loc[ytd_mask, "Total Usage"].sum()
    col2.metric(
        "Year-to-Date Total",
        f"{ytd_total:,.2f} kWh"
    )
    
# In the metrics section, update the year-over-year comparison:

    # Get the latest date and current day of month
    latest_date = monthly_data["Date"].max()
    current_month_day = latest_date.day

    # Calculate current month's total up to current day
    current_month_mask = (
        (monthly_data["Date"].dt.day <= current_month_day)
    )
    monthly_total = monthly_data.loc[current_month_mask, "Total Usage"].sum()

    # Calculate last year's total for the exact same days
    last_year_mask = (
        (hotel_data["Year"] == selected_year - 1) & 
        (hotel_data["Month"] == selected_month) &
        (hotel_data["Date"].dt.day <= current_month_day)
    )

    last_year_data = hotel_data.loc[last_year_mask, "Total Usage"].sum()

    # Print debug information
    print(f"Current year days: {len(monthly_data[current_month_mask])}")
    print(f"Last year days: {len(hotel_data[last_year_mask])}")
    print(f"Current total: {monthly_total}")
    print(f"Last year total: {last_year_data}")

    # Calculate and display the comparison
    if last_year_data > 0:
        yoy_change = ((monthly_total - last_year_data) / last_year_data) * 100
        delta_color = "inverse" if yoy_change < 0 else "normal"  # Green for decrease, red for increase
        col3.metric(
            "vs Last Year",
            f"{monthly_total:,.2f} kWh",
            f"{yoy_change:+.1f}%",
            delta_color=delta_color,
            help=f"Comparing first {current_month_day} days of {calendar.month_name[selected_month]}. Green indicates decrease (improvement), Red indicates increase"
        )
    else:
        col3.metric(
            "vs Last Year",
            f"{monthly_total:,.2f} kWh",
            "N/A",
            help="No data available for comparison from last year"
        )

# Peak/Off-Peak Analysis
    st.subheader("Peak and Off-Peak Usage")
    peak_col1, peak_col2 = st.columns(2)
    
    with peak_col1:
        # Peak hours analysis (7 AM to 10 PM)
        peak_intervals = [f"{str(hour).zfill(2)}:{'00' if i == 0 else '30'}"
                        for hour in range(7, 22) for i in range(2)]
        off_peak_intervals = [f"{str(hour).zfill(2)}:{'00' if i == 0 else '30'}"
                            for hour in list(range(22, 24)) + list(range(0, 7)) for i in range(2)]
        
        peak_usage = monthly_data.loc[:, peak_intervals].sum().sum()
        off_peak_usage = monthly_data.loc[:, off_peak_intervals].sum().sum()
        
        peak_data = pd.DataFrame({
            "Period": ["Peak Hours (7AM-10PM)", "Off-Peak Hours (10PM-7AM)"],
            "Usage": [peak_usage, off_peak_usage]
        })
        
        fig_peak = px.pie(
            peak_data,
            values="Usage",
            names="Period",
            title="Peak vs Off-Peak Usage Distribution",
            color_discrete_sequence=[app_colors["highlight"], app_colors["accent"]]
        )
        fig_peak.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate="<b>%{label}</b><br>Usage: %{value:,.2f} kWh<br>Percentage: %{percent}<extra></extra>"
        )
        st.plotly_chart(fig_peak, key="peak_off_peak_chart")
    
    with peak_col2:
        # Weekly pattern with enhanced formatting
        weekly_avg = monthly_data.groupby("Day of Week")["Total Usage"].mean().reset_index()
        
        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        weekly_avg["Day of Week"] = pd.Categorical(weekly_avg["Day of Week"], categories=days_order, ordered=True)
        weekly_avg = weekly_avg.sort_values("Day of Week")
        
        fig_weekly = px.bar(
            weekly_avg,
            x="Day of Week",
            y="Total Usage",
            title="Average Daily Usage by Day of Week",
            color_discrete_sequence=[app_colors["accent"]]
        )
        fig_weekly.update_traces(
            hovertemplate="<b>%{x}</b><br>Average Usage: %{y:,.2f} kWh<extra></extra>"
        )
        fig_weekly.update_layout(
            xaxis_title="Day of Week",
            yaxis_title="Average Usage (kWh)",
            yaxis_tickformat=",."
        )
        st.plotly_chart(fig_weekly, key="weekly_pattern_chart")    

    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Daily Trend",
        "Monthly Totals",
        "Half-Hourly Usage",
        "Usage by Day",
        "Heatmap",
        "Projected vs Actual"
    ])
    
    with tab1:
        # Daily Trend with enhanced comparison
        daily_data = monthly_data.groupby("Date")["Total Usage"].sum().reset_index()
        
        # Get last year's data
        last_year_mask = (hotel_data["Year"] == selected_year - 1) & (hotel_data["Month"] == selected_month)
        last_year_daily = hotel_data.loc[last_year_mask].groupby("Date")["Total Usage"].sum().reset_index()
        
        fig_daily = go.Figure()
        fig_daily.add_trace(go.Scatter(
            x=daily_data["Date"],
            y=daily_data["Total Usage"],
            name=f"{calendar.month_name[selected_month]} {selected_year}",
            line=dict(color=app_colors["accent"])
        ))
        fig_daily.add_trace(go.Scatter(
            x=last_year_daily["Date"],
            y=last_year_daily["Total Usage"],
            name=f"{calendar.month_name[selected_month]} {selected_year-1}",
            line=dict(color=app_colors["highlight"], dash="dash")
        ))
        fig_daily.update_layout(
            title="Daily Usage Comparison",
            xaxis_title="Date",
            yaxis_title="Usage (kWh)",
            hovermode="x unified",
            yaxis_tickformat=",."
        )
        fig_daily.update_traces(
            hovertemplate="%{y:,.2f} kWh"
        )
        st.plotly_chart(fig_daily, use_container_width=True, key="daily_trend_chart")
    
    with tab2:
        # Get the latest date for partial month comparison
        latest_date = hotel_data["Date"].max()
        current_month_day = latest_date.day
        
        # Monthly totals with proper grouping and partial month handling
        def get_monthly_data(data):
            monthly = []
            for (year, month), group in data.groupby(['Year', 'Month']):
                # For current month, only use data up to the latest available day
                if year == latest_date.year and month == latest_date.month:
                    usage = group[group['Date'].dt.day <= current_month_day]['Total Usage'].sum()
                else:
                    # For past months in current year, use data up to same day as current month
                    if year == latest_date.year and month < latest_date.month:
                        usage = group[group['Date'].dt.day <= current_month_day]['Total Usage'].sum()
                    else:
                        usage = group['Total Usage'].sum()
                
                monthly.append({
                    'Year': year,
                    'Month': month,
                    'Month Name': calendar.month_name[month],
                    'Total Usage': usage
                })
            return pd.DataFrame(monthly)

        monthly_totals = get_monthly_data(hotel_data)
        
        # Create the visualization
        fig_monthly = px.bar(
            monthly_totals,
            x="Month Name",
            y="Total Usage",
            color="Year",
            title=f"Monthly Usage by Year (Up to Day {current_month_day} for Current Month)",
            barmode="group",  # This ensures bars are grouped rather than stacked
             color_discrete_sequence={
                "2023": "#4ecca3",
                "2024": "#ff5722",
                "2025": "#1e90ff",  # Add a distinct color for 2025
                # Continue adding unique colors for additional years as needed
                "2026": "#f4d03f",
                "2027": "#9b59b6",
                # Add more colors up to the required number, e.g., up to 226 as needed
            },  # Use consistent colors
            category_orders={
                "Month Name": list(calendar.month_name)[1:]  # Ensure correct month order
            }
        )
        
        # Update layout
        fig_monthly.update_layout(
            xaxis_title="Month",
            yaxis_title="Usage (kWh)",
            yaxis_tickformat=",.",
            showlegend=True,
            legend_title="Year",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            bargap=0.2,        # Gap between bars in a group
            bargroupgap=0.1    # Gap between bar groups
        )
        
        # Add hover information
        fig_monthly.update_traces(
            hovertemplate="<b>%{x}</b><br>Year: %{marker.color}<br>Usage: %{y:,.2f} kWh<extra></extra>"
        )
        
        # Display chart with unique key
        st.plotly_chart(fig_monthly, use_container_width=True, key="monthly_totals_chart")
        
        # Add note about partial month comparison
        if latest_date.day < calendar.monthrange(latest_date.year, latest_date.month)[1]:
            st.caption(f"Note: Current month data is partial (up to day {current_month_day}). " +
                    "All monthly comparisons show usage up to this same day of the month for fair comparison.")


    with tab3:
        # Enhanced half-hourly usage visualization
        last_date = monthly_data["Date"].max()
        last_date_data = monthly_data[monthly_data["Date"] == last_date]
        
        time_cols = [f"{str(hour).zfill(2)}:{'00' if i == 0 else '30'}"
                    for hour in range(24) for i in range(2)]
        
        half_hour_data = pd.DataFrame({
            'Time': time_cols,
            'Usage': last_date_data[time_cols].iloc[0]
        })
        
        fig_hourly = px.line(
            half_hour_data,
            x='Time',
            y='Usage',
            title=f"Half-Hourly Usage Pattern ({last_date.strftime('%Y-%m-%d')})",
            markers=True
        )
        fig_hourly.update_layout(
            xaxis_title="Time of Day",
            yaxis_title="Usage (kWh)",
            yaxis_tickformat=",.",
            xaxis=dict(tickangle=-45)
        )
        fig_hourly.update_traces(
            line_color=app_colors["accent"],
            hovertemplate="<b>%{x}</b><br>Usage: %{y:.2f} kWh<extra></extra>"
        )
        st.plotly_chart(fig_hourly, use_container_width=True, key="hourly_usage_chart")
    
    with tab4:
        # Enhanced daily usage pattern
        fig_daily_pattern = px.line(
            monthly_data,
            x="Date",
            y="Total Usage",
            title="Daily Usage Pattern",
            line_shape="spline"
        )
        fig_daily_pattern.update_layout(
            xaxis_title="Date",
            yaxis_title="Usage (kWh)",
            yaxis_tickformat=",."
        )
        fig_daily_pattern.update_traces(
            line_color=app_colors["accent"],
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Usage: %{y:,.2f} kWh<extra></extra>"
        )
        st.plotly_chart(fig_daily_pattern, use_container_width=True, key="daily_pattern_chart")
    
    with tab5:
        # Enhanced heatmap using the create_heatmap function
        fig_heatmap = create_heatmap(monthly_data, selected_hotel, selected_year, selected_month)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab6:
        # Projected vs Actual with proper data handling
        monthly_totals = hotel_data.groupby(["Year", "Month"], as_index=False)["Total Usage"].sum()
        monthly_totals["Month Name"] = monthly_totals["Month"].apply(lambda x: calendar.month_name[x])
        
        # Get current year data
        current_year_mask = monthly_totals["Year"] == selected_year
        current_year_data = monthly_totals[current_year_mask].copy()
        current_year_data["Type"] = "Actual"
        
        # Get last year data and calculate target
        last_year_mask = monthly_totals["Year"] == (selected_year - 1)
        last_year_data = monthly_totals[last_year_mask].copy()
        last_year_data["Total Usage"] = last_year_data["Total Usage"] * 0.9  # 90% target
        last_year_data["Type"] = "Target (90% of Last Year)"
        last_year_data["Year"] = selected_year  # Set year to current year for comparison
        
        # Combine data
        comparison_data = pd.concat([current_year_data, last_year_data])
        
        # Create visualization
        fig_comparison = px.bar(
            comparison_data,
            x="Month Name",
            y="Total Usage",
            color="Type",
            title="Actual vs Target Usage (10% Reduction Goal)",
            barmode="group",
            color_discrete_sequence=[app_colors["accent"], app_colors["highlight"]]
        )
        
        # Update layout
        fig_comparison.update_layout(
            xaxis_title="Month",
            yaxis_title="Usage (kWh)",
            yaxis_tickformat=",.",
            showlegend=True,
            legend_title="Type"
        )
        
        # Add custom hover template
        fig_comparison.update_traces(
            hovertemplate="<b>%{x}</b><br>%{data.name}<br>Usage: %{y:,.2f} kWh<extra></extra>"
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True, key="comparison_chart")
    
    # Enhanced download functionality
    st.sidebar.download_button(
        label="📥 Download Data as CSV",
        data=monthly_data.to_csv(index=False).encode('utf-8'),
        file_name=f'energy_data_{selected_hotel}_{selected_year}_{selected_month}.csv',
        mime='text/csv',
        help="Download the current selection's data as a CSV file"
    )

if __name__ == "__main__":
    main()

