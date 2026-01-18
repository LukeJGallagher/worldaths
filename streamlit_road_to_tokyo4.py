import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
import glob
import re

# Set page config
st.set_page_config(
    page_title="Road to Tokyo 2025",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }
    
    .event-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    
    .qualification-text {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load all CSV files from the road_to folder"""
    try:
        # Look for CSV files in the world_athletics/Data/road_to folder
        data_path = os.path.join("world_athletics", "Data", "road_to")
        csv_files = glob.glob(os.path.join(data_path, "road_to_tokyo_batch_*.csv"))
        
        if not csv_files:
            st.error(f"No CSV files found in {data_path}. Please check the folder structure.")
            return pd.DataFrame()
        
        # Load and combine all CSV files
        all_data = []
        file_info = []
        
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                all_data.append(df)
                file_info.append({
                    'file': os.path.basename(file),
                    'rows': len(df),
                    'events': df['Actual_Event_Name'].nunique() if 'Actual_Event_Name' in df.columns else 0
                })
                st.sidebar.success(f"✅ {os.path.basename(file)}: {len(df)} rows")
            except Exception as e:
                st.sidebar.error(f"❌ Error loading {os.path.basename(file)}: {e}")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Debug info in sidebar
            st.sidebar.write("**📊 File Loading Summary:**")
            total_events = combined_df['Actual_Event_Name'].nunique() if 'Actual_Event_Name' in combined_df.columns else 0
            st.sidebar.write(f"Total files: {len(csv_files)}")
            st.sidebar.write(f"Total rows: {len(combined_df)}")
            st.sidebar.write(f"Total unique events: {total_events}")
            
            # Show events by file
            with st.sidebar.expander("📋 Events by File"):
                for info in file_info:
                    st.write(f"**{info['file']}**: {info['rows']} rows, {info['events']} events")
            
            return combined_df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_qualification_metadata():
    """Load qualification process metadata if available"""
    try:
        metadata_path = os.path.join("world_athletics", "Data", "qualification_processes", "qualification_processes.csv")
        if os.path.exists(metadata_path):
            return pd.read_csv(metadata_path)
        else:
            # Try alternative path
            alt_path = os.path.join("world_athletics", "Data", "event_metadata.csv")
            if os.path.exists(alt_path):
                return pd.read_csv(alt_path)
    except Exception as e:
        st.sidebar.warning(f"Could not load qualification metadata: {e}")
    
    return pd.DataFrame()

@st.cache_data
def load_athlete_performance_data():
    """Load athlete performance data from modal scraper if available"""
    try:
        # Look for athlete performance files
        performance_path = os.path.join("world_athletics", "Data", "athlete_performances")
        if os.path.exists(performance_path):
            csv_files = glob.glob(os.path.join(performance_path, "performances_*.csv"))
            
            if csv_files:
                all_performance_data = []
                for file in csv_files:
                    try:
                        df = pd.read_csv(file)
                        all_performance_data.append(df)
                        st.sidebar.info(f"📊 Loaded {os.path.basename(file)}")
                    except Exception as e:
                        st.sidebar.warning(f"⚠️ Error loading {os.path.basename(file)}: {e}")
                
                if all_performance_data:
                    combined_df = pd.concat(all_performance_data, ignore_index=True)
                    st.sidebar.success(f"🏃 Performance data: {len(combined_df)} records")
                    return combined_df
                    
    except Exception as e:
        st.sidebar.warning(f"Could not load athlete performance data: {e}")
    
    return pd.DataFrame()

def parse_column6_data(value):
    """Parse Column_6 data to extract numeric values and rankings"""
    if pd.isna(value) or value == '':
        return None, None, None
    
    try:
        value_str = str(value).strip()
        
        # Check if this contains ranking points (has 'p' in it)
        if 'p' in value_str.lower():
            # Extract ranking points (e.g., "1312p", "1182p") and remove 'p'
            points_match = re.search(r'(\d+)p', value_str, re.IGNORECASE)
            points_value = int(points_match.group(1)) if points_match else None
            
            # Estimate ranking from points
            ranking = None
            if points_value:
                if points_value > 1300:
                    ranking = f"{min(20, max(1, int((1400 - points_value) / 10)))}th"
                elif points_value > 1000:
                    ranking = f"{min(100, max(21, int((1300 - points_value) / 5) + 20))}th"
                else:
                    ranking = f"{min(200, max(101, int((1000 - points_value) / 2) + 100))}th"
            
            return None, points_value, ranking
        
        else:
            # Look for time/distance values (e.g., "10.72", "11.77", "Qualified by Entry Standard=  10.72")
            # Try to find decimal numbers that look like times/distances
            time_patterns = [
                r'(\d+\.\d{2})',  # Match X.XX format (e.g., 10.72, 11.77)
                r'(\d+:\d{2}\.\d{2})',  # Match M:SS.XX format (e.g., 1:59.00)
                r'(\d+:\d{2}:\d{2})',  # Match H:MM:SS format (marathon times)
                r'(\d+\.\d{1,3})'  # Match general decimal format
            ]
            
            time_value = None
            for pattern in time_patterns:
                time_match = re.search(pattern, value_str)
                if time_match:
                    time_str = time_match.group(1)
                    try:
                        # Convert time formats to seconds for comparison
                        if ':' in time_str:
                            parts = time_str.split(':')
                            if len(parts) == 2:  # M:SS.XX format
                                minutes = float(parts[0])
                                seconds = float(parts[1])
                                time_value = minutes * 60 + seconds
                            elif len(parts) == 3:  # H:MM:SS format
                                hours = float(parts[0])
                                minutes = float(parts[1])
                                seconds = float(parts[2])
                                time_value = hours * 3600 + minutes * 60 + seconds
                        else:
                            # Direct decimal value
                            time_value = float(time_str)
                        break
                    except:
                        continue
            
            return time_value, None, None
        
    except Exception as e:
        return None, None, None

def parse_performance_result(result):
    """Parse performance result from the detailed performance data"""
    if pd.isna(result):
        return None
    
    try:
        # If it's already a number, return it
        if isinstance(result, (int, float)):
            return float(result)
        
        # If it's a string, try to parse it
        result_str = str(result).strip()
        
        # Handle time formats
        if ':' in result_str:
            parts = result_str.split(':')
            if len(parts) == 2:  # M:SS.XX format
                minutes = float(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
            elif len(parts) == 3:  # H:MM:SS format
                hours = float(parts[0])
                minutes = float(parts[1])
                seconds = float(parts[2])
                return hours * 3600 + minutes * 60 + seconds
        else:
            # Try to parse as a decimal number
            return float(result_str)
    except:
        return None

def create_box_plot(df, event_filter, y_column, title, color_column=None, overlay_performance=False, 
                   performance_df=None, selected_country="All Countries", selected_athlete="All Athletes",
                   perf_data_column="Result", show_ranking_score=False, show_pfsc_scores=True, 
                   overlay_color_by="None"):
    """Create an interactive box plot filtered by event with all points shown and optional performance overlay"""
    
    # Filter by selected event first
    if event_filter and event_filter != "All Events":
        df_filtered = df[df['Actual_Event_Name'] == event_filter].copy()
    else:
        df_filtered = df.copy()
    
    if df_filtered.empty:
        st.warning(f"No data available for event: {event_filter}")
        return None
    
    # Parse Column_6 data
    df_filtered[['time_value', 'points_value', 'ranking']] = df_filtered['Column_6'].apply(
        lambda x: pd.Series(parse_column6_data(x))
    )
    
    # Determine which value to use for plotting
    if y_column == 'time_value':
        plot_data = df_filtered.dropna(subset=['time_value'])
        y_values = 'time_value'
        y_title = "Time/Distance Value (seconds/meters)"
        hover_data = ['Athlete', 'time_value', 'Status', 'Actual_Event_Name']
    elif y_column == 'points_value':
        plot_data = df_filtered.dropna(subset=['points_value'])
        y_values = 'points_value'
        y_title = "Ranking Points"
        hover_data = ['Athlete', 'points_value', 'ranking', 'Status', 'Actual_Event_Name']
    else:
        st.warning("No valid data found for plotting")
        return None
    
    if plot_data.empty:
        st.warning(f"No {y_title.lower()} data available for the selected filters")
        return None
    
    # Create box plot with points
    fig = go.Figure()
    
    # Group by the color column if specified
    if color_column and color_column in plot_data.columns:
        groups = plot_data[color_column].unique()
        colors = px.colors.qualitative.Set3[:len(groups)]
        
        for i, group in enumerate(groups):
            group_data = plot_data[plot_data[color_column] == group]
            
            # Add box plot
            fig.add_trace(go.Box(
                y=group_data[y_values],
                name=str(group),
                boxpoints='all',
                pointpos=0,
                jitter=0.3,
                marker=dict(color=colors[i % len(colors)]),
                text=group_data['Athlete'],
                customdata=group_data[hover_data].values,
                hovertemplate="<b>%{text}</b><br>" +
                            f"{y_title}: %{{y}}<br>" +
                            "Status: %{customdata[2]}<br>" +
                            "Event: %{customdata[3]}<br>" +
                            ("<br>Estimated Ranking: %{customdata[2]}" if y_column == 'points_value' else "") +
                            "<extra></extra>"
            ))
    else:
        # Single box plot
        fig.add_trace(go.Box(
            y=plot_data[y_values],
            name=title,
            boxpoints='all',
            pointpos=0,
            jitter=0.3,
            text=plot_data['Athlete'],
            customdata=plot_data[hover_data].values,
            hovertemplate="<b>%{text}</b><br>" +
                        f"{y_title}: %{{y}}<br>" +
                        "Status: %{customdata[2]}<br>" +
                        "Event: %{customdata[3]}<br>" +
                        ("<br>Estimated Ranking: %{customdata[2]}" if y_column == 'points_value' else "") +
                        "<extra></extra>"
        ))
    
    # Add performance overlay if requested and available
    if overlay_performance and performance_df is not None and not performance_df.empty:
        # Filter performance data for the same event
        perf_filtered = performance_df.copy()
        
        # Filter by country if specified
        if selected_country != "All Countries" and 'Country_Code' in perf_filtered.columns:
            perf_filtered = perf_filtered[
                perf_filtered['Country_Code'].astype(str).str.lower() == selected_country.lower()
            ]
        
        # Filter by athlete if specified
        if selected_athlete != "All Athletes" and 'Athlete_Name' in perf_filtered.columns:
            perf_filtered = perf_filtered[
                perf_filtered['Athlete_Name'].astype(str).str.lower() == selected_athlete.lower()
            ]
        
        # Filter by event if specified
        if event_filter != "All Events":
            # More flexible event matching
            event_keywords = event_filter.lower().split()
            
            # Try to match by Event_Name or Event columns
            if 'Event_Name' in perf_filtered.columns:
                mask1 = perf_filtered['Event_Name'].str.lower().str.contains('|'.join(event_keywords), case=False, na=False)
            else:
                mask1 = pd.Series([False] * len(perf_filtered))
                
            if 'Event' in perf_filtered.columns:
                mask2 = perf_filtered['Event'].str.lower().str.contains('|'.join(event_keywords), case=False, na=False)
            else:
                mask2 = pd.Series([False] * len(perf_filtered))
            
            perf_filtered = perf_filtered[mask1 | mask2]
        
        # Handle different overlay types based on y_column
        if y_column == 'time_value' and perf_data_column in perf_filtered.columns:
            # Time/Distance overlay (existing logic)
            perf_filtered = perf_filtered.copy()
            perf_filtered['parsed_result'] = perf_filtered[perf_data_column].apply(parse_performance_result)
            perf_valid = perf_filtered.dropna(subset=['parsed_result'])
            
            if not perf_valid.empty:
                # Group by overlay_color_by if specified
                if overlay_color_by != "None" and overlay_color_by in df.columns:
                    # Map performance data to main data to get grouping info
                    perf_valid_with_groups = perf_valid.copy()
                    
                    # Try to match athletes between datasets
                    for idx, perf_row in perf_valid.iterrows():
                        athlete_name = perf_row.get('Athlete_Name', '')
                        # Find matching athlete in main dataset
                        matching_rows = df[df['Athlete'].str.contains(athlete_name.split()[-1] if athlete_name else '', case=False, na=False)]
                        if not matching_rows.empty:
                            # Get the group value for this athlete
                            group_value = matching_rows.iloc[0][overlay_color_by]
                            perf_valid_with_groups.loc[idx, 'overlay_group'] = group_value
                        else:
                            perf_valid_with_groups.loc[idx, 'overlay_group'] = 'Unknown'
                    
                    # Plot by groups - overlay directly on main box plot positions
                    groups = perf_valid_with_groups['overlay_group'].unique()
                    colors = px.colors.qualitative.Set1[:len(groups)]
                    
                    for i, group in enumerate(groups):
                        group_data = perf_valid_with_groups[perf_valid_with_groups['overlay_group'] == group]
                        if not group_data.empty:
                            # Create hover data
                            perf_hover_data = []
                            for _, row in group_data.iterrows():
                                hover_info = [
                                    row.get('Athlete_Name', 'Unknown'),
                                    row.get('parsed_result', 0),
                                    row.get('Competition', 'Unknown Competition'),
                                    row.get('Date', 'Unknown Date'),
                                    row.get('Event', 'Unknown Event'),
                                    row.get('Pl', 'N/A'),
                                    group
                                ]
                                perf_hover_data.append(hover_info)
                            
                            # Overlay directly on the main box plot (x=0 for main data)
                            x_positions = [0] * len(group_data)  # Same x as main box plot
                            
                            fig.add_trace(go.Scatter(
                                y=group_data['parsed_result'],
                                x=x_positions,
                                mode='markers',
                                name=f'Overlay: {group}',
                                marker=dict(
                                    color=colors[i % len(colors)],
                                    size=10,
                                    symbol='diamond',
                                    line=dict(width=2, color='white'),
                                    opacity=0.8
                                ),
                                text=group_data['Athlete_Name'],
                                customdata=perf_hover_data,
                                hovertemplate="<b>%{text}</b><br>" +
                                            "Performance: %{y}<br>" +
                                            f"{overlay_color_by}: %{{customdata[6]}}<br>" +
                                            "Event: %{customdata[4]}<br>" +
                                            "Place: %{customdata[5]}<br>" +
                                            "Competition: %{customdata[2]}<br>" +
                                            "Date: %{customdata[3]}<br>" +
                                            f"<extra>Performance Overlay</extra>"
                            ))
                    
                    st.info(f"📊 Overlaid {len(perf_valid)} competition performances grouped by {overlay_color_by}")
                
                else:
                    # Single group overlay directly on main data
                    perf_hover_data = []
                    for _, row in perf_valid.iterrows():
                        hover_info = [
                            row.get('Athlete_Name', 'Unknown'),
                            row.get('parsed_result', 0),
                            row.get('Competition', 'Unknown Competition'),
                            row.get('Date', 'Unknown Date'),
                            row.get('Event', 'Unknown Event'),
                            row.get('Pl', 'N/A')
                        ]
                        perf_hover_data.append(hover_info)
                    
                    # Overlay directly on the main box plot
                    x_positions = [0] * len(perf_valid)  # Same x as main box plot
                    
                    fig.add_trace(go.Scatter(
                        y=perf_valid['parsed_result'],
                        x=x_positions,
                        mode='markers',
                        name='Performance Overlay',
                        marker=dict(
                            color='red',
                            size=10,
                            symbol='diamond',
                            line=dict(width=2, color='white'),
                            opacity=0.8
                        ),
                        text=perf_valid['Athlete_Name'],
                        customdata=perf_hover_data,
                        hovertemplate="<b>%{text}</b><br>" +
                                    "Performance: %{y}<br>" +
                                    "Event: %{customdata[4]}<br>" +
                                    "Place: %{customdata[5]}<br>" +
                                    "Competition: %{customdata[2]}<br>" +
                                    "Date: %{customdata[3]}<br>" +
                                    "<extra>Performance Overlay</extra>"
                    ))
                    
                    st.info(f"📊 Overlaid {len(perf_valid)} competition performances on the main visualization")
            else:
                st.warning("No valid performance data found for this event")
                
        elif y_column == 'points_value':
            # Ranking points overlay - show both Ranking_Score and PfSc directly on main data
            overlay_added = False
            
            # Add Ranking_Score overlay (white border) - overlaid on main data
            if 'Ranking_Score' in perf_filtered.columns and show_ranking_score:
                ranking_valid = perf_filtered.dropna(subset=['Ranking_Score'])
                if not ranking_valid.empty:
                    # Map ranking data to main data for grouping
                    if overlay_color_by != "None" and overlay_color_by in df.columns:
                        ranking_valid_with_groups = ranking_valid.copy()
                        
                        # Try to match athletes between datasets
                        for idx, perf_row in ranking_valid.iterrows():
                            athlete_name = perf_row.get('Athlete_Name', '')
                            # Find matching athlete in main dataset
                            matching_rows = df[df['Athlete'].str.contains(athlete_name.split()[-1] if athlete_name else '', case=False, na=False)]
                            if not matching_rows.empty:
                                group_value = matching_rows.iloc[0][overlay_color_by]
                                ranking_valid_with_groups.loc[idx, 'overlay_group'] = group_value
                            else:
                                ranking_valid_with_groups.loc[idx, 'overlay_group'] = 'Unknown'
                        
                        # Plot by groups - overlay directly on main box plot
                        groups = ranking_valid_with_groups['overlay_group'].unique()
                        colors = px.colors.qualitative.Set2[:len(groups)]
                        
                        for i, group in enumerate(groups):
                            group_data = ranking_valid_with_groups[ranking_valid_with_groups['overlay_group'] == group]
                            if not group_data.empty:
                                ranking_hover_data = []
                                for _, row in group_data.iterrows():
                                    hover_info = [
                                        row.get('Athlete_Name', 'Unknown'),
                                        row.get('Ranking_Score', 0),
                                        row.get('Competition', 'Unknown Competition'),
                                        row.get('Date', 'Unknown Date'),
                                        row.get('Event', 'Unknown Event'),
                                        row.get('Average_Performance_Score', 'N/A'),
                                        group
                                    ]
                                    ranking_hover_data.append(hover_info)
                                
                                # Overlay directly on main box plot (x=0)
                                x_positions = [0] * len(group_data)  # Same x as main box plot
                                
                                fig.add_trace(go.Scatter(
                                    y=group_data['Ranking_Score'],
                                    x=x_positions,
                                    mode='markers',
                                    name=f'Ranking ({group})',
                                    marker=dict(
                                        color=colors[i % len(colors)],
                                        size=12,
                                        symbol='circle',
                                        line=dict(width=3, color='white'),  # White border
                                        opacity=0.9
                                    ),
                                    text=group_data['Athlete_Name'],
                                    customdata=ranking_hover_data,
                                    hovertemplate="<b>%{text}</b><br>" +
                                                "Ranking Score: %{y}<br>" +
                                                f"{overlay_color_by}: %{{customdata[6]}}<br>" +
                                                "Event: %{customdata[4]}<br>" +
                                                "Avg Performance Score: %{customdata[5]}<br>" +
                                                "Latest Competition: %{customdata[2]}<br>" +
                                                "Date: %{customdata[3]}<br>" +
                                                f"<extra>Current Ranking</extra>"
                                ))
                        
        elif y_column == 'points_value':
            # Ranking points overlay - show both Ranking_Score and PfSc directly on main data
            overlay_added = False
            
            # Add Ranking_Score overlay (white border) - overlaid on main data
            if 'Ranking_Score' in perf_filtered.columns and show_ranking_score:
                ranking_valid = perf_filtered.dropna(subset=['Ranking_Score'])
                if not ranking_valid.empty:
                    # Map ranking data to main data for grouping
                    if overlay_color_by != "None" and overlay_color_by in df.columns:
                        ranking_valid_with_groups = ranking_valid.copy()
                        
                        # Try to match athletes between datasets
                        for idx, perf_row in ranking_valid.iterrows():
                            athlete_name = perf_row.get('Athlete_Name', '')
                            # Find matching athlete in main dataset
                            matching_rows = df[df['Athlete'].str.contains(athlete_name.split()[-1] if athlete_name else '', case=False, na=False)]
                            if not matching_rows.empty:
                                group_value = matching_rows.iloc[0][overlay_color_by]
                                ranking_valid_with_groups.loc[idx, 'overlay_group'] = group_value
                            else:
                                ranking_valid_with_groups.loc[idx, 'overlay_group'] = 'Unknown'
                        
                        # Plot by groups - overlay directly on main box plot
                        groups = ranking_valid_with_groups['overlay_group'].unique()
                        colors = px.colors.qualitative.Set2[:len(groups)]
                        
                        for i, group in enumerate(groups):
                            group_data = ranking_valid_with_groups[ranking_valid_with_groups['overlay_group'] == group]
                            if not group_data.empty:
                                ranking_hover_data = []
                                for _, row in group_data.iterrows():
                                    hover_info = [
                                        row.get('Athlete_Name', 'Unknown'),
                                        row.get('Ranking_Score', 0),
                                        row.get('Competition', 'Unknown Competition'),
                                        row.get('Date', 'Unknown Date'),
                                        row.get('Event', 'Unknown Event'),
                                        row.get('Average_Performance_Score', 'N/A'),
                                        group
                                    ]
                                    ranking_hover_data.append(hover_info)
                                
                                # Overlay directly on main box plot (x=0)
                                x_positions = [0] * len(group_data)  # Same x as main box plot
                                
                                fig.add_trace(go.Scatter(
                                    y=group_data['Ranking_Score'],
                                    x=x_positions,
                                    mode='markers',
                                    name=f'Ranking ({group})',
                                    marker=dict(
                                        color=colors[i % len(colors)],
                                        size=12,
                                        symbol='circle',
                                        line=dict(width=3, color='white'),  # White border
                                        opacity=0.9
                                    ),
                                    text=group_data['Athlete_Name'],
                                    customdata=ranking_hover_data,
                                    hovertemplate="<b>%{text}</b><br>" +
                                                "Ranking Score: %{y}<br>" +
                                                f"{overlay_color_by}: %{{customdata[6]}}<br>" +
                                                "Event: %{customdata[4]}<br>" +
                                                "Avg Performance Score: %{customdata[5]}<br>" +
                                                "Latest Competition: %{customdata[2]}<br>" +
                                                "Date: %{customdata[3]}<br>" +
                                                f"<extra>Current Ranking</extra>"
                                ))
                        
                        st.info(f"📊 Overlaid {len(ranking_valid)} current ranking scores grouped by {overlay_color_by}")
                    else:
                        # Single group - overlay directly on main data
                        ranking_hover_data = []
                        for _, row in ranking_valid.iterrows():
                            hover_info = [
                                row.get('Athlete_Name', 'Unknown'),
                                row.get('Ranking_Score', 0),
                                row.get('Competition', 'Unknown Competition'),
                                row.get('Date', 'Unknown Date'),
                                row.get('Event', 'Unknown Event'),
                                row.get('Average_Performance_Score', 'N/A')
                            ]
                            ranking_hover_data.append(hover_info)
                        
                        # Overlay directly on main box plot
                        x_positions = [0] * len(ranking_valid)  # Same x as main box plot
                        
                        fig.add_trace(go.Scatter(
                            y=ranking_valid['Ranking_Score'],
                            x=x_positions,
                            mode='markers',
                            name='Current Ranking Score',
                            marker=dict(
                                color='blue',
                                size=12,
                                symbol='circle',
                                line=dict(width=3, color='white'),  # White border
                                opacity=0.9
                            ),
                            text=ranking_valid['Athlete_Name'],
                            customdata=ranking_hover_data,
                            hovertemplate="<b>%{text}</b><br>" +
                                        "Ranking Score: %{y}<br>" +
                                        "Event: %{customdata[4]}<br>" +
                                        "Avg Performance Score: %{customdata[5]}<br>" +
                                        "Latest Competition: %{customdata[2]}<br>" +
                                        "Date: %{customdata[3]}<br>" +
                                        "<extra>Current Ranking</extra>"
                        ))
                        
                        st.info(f"📊 Overlaid {len(ranking_valid)} current ranking scores (blue circles with white border)")
                    
                    overlay_added = True
            
            # Add PfSc overlay (Performance Scores) - overlaid on main data
            if 'PfSc' in perf_filtered.columns and show_pfsc_scores:
                pfsc_valid = perf_filtered.dropna(subset=['PfSc'])
                if not pfsc_valid.empty:
                    # Map PfSc data to main data for grouping
                    if overlay_color_by != "None" and overlay_color_by in df.columns:
                        pfsc_valid_with_groups = pfsc_valid.copy()
                        
                        # Try to match athletes between datasets
                        for idx, perf_row in pfsc_valid.iterrows():
                            athlete_name = perf_row.get('Athlete_Name', '')
                            # Find matching athlete in main dataset
                            matching_rows = df[df['Athlete'].str.contains(athlete_name.split()[-1] if athlete_name else '', case=False, na=False)]
                            if not matching_rows.empty:
                                group_value = matching_rows.iloc[0][overlay_color_by]
                                pfsc_valid_with_groups.loc[idx, 'overlay_group'] = group_value
                            else:
                                pfsc_valid_with_groups.loc[idx, 'overlay_group'] = 'Unknown'
                        
                        # Plot by groups - overlay directly on main box plot
                        groups = pfsc_valid_with_groups['overlay_group'].unique()
                        colors = px.colors.qualitative.Set3[:len(groups)]
                        
                        for i, group in enumerate(groups):
                            group_data = pfsc_valid_with_groups[pfsc_valid_with_groups['overlay_group'] == group]
                            if not group_data.empty:
                                pfsc_hover_data = []
                                for _, row in group_data.iterrows():
                                    hover_info = [
                                        row.get('Athlete_Name', 'Unknown'),
                                        row.get('PfSc', 0),
                                        row.get('Competition', 'Unknown Competition'),
                                        row.get('Date', 'Unknown Date'),
                                        row.get('Event', 'Unknown Event'),
                                        row.get('Pl', 'N/A'),
                                        row.get('Result', 'N/A'),
                                        group
                                    ]
                                    pfsc_hover_data.append(hover_info)
                                
                                # Overlay directly on main box plot (x=0)
                                x_positions = [0] * len(group_data)  # Same x as main box plot
                                
                                fig.add_trace(go.Scatter(
                                    y=group_data['PfSc'],
                                    x=x_positions,
                                    mode='markers',
                                    name=f'PfSc ({group})',
                                    marker=dict(
                                        color=colors[i % len(colors)],
                                        size=10,
                                        symbol='square',
                                        line=dict(width=2, color='black'),
                                        opacity=0.8
                                    ),
                                    text=group_data['Athlete_Name'],
                                    customdata=pfsc_hover_data,
                                    hovertemplate="<b>%{text}</b><br>" +
                                                "Performance Score: %{y}<br>" +
                                                f"{overlay_color_by}: %{{customdata[7]}}<br>" +
                                                "Event: %{customdata[4]}<br>" +
                                                "Place: %{customdata[5]}<br>" +
                                                "Result: %{customdata[6]}<br>" +
                                                "Competition: %{customdata[2]}<br>" +
                                                "Date: %{customdata[3]}<br>" +
                                                f"<extra>Performance Scores</extra>"
                                ))
                        
                        st.info(f"📊 Overlaid {len(pfsc_valid)} performance scores grouped by {overlay_color_by}")
                    else:
                        # Single group - overlay directly on main data
                        pfsc_hover_data = []
                        for _, row in pfsc_valid.iterrows():
                            hover_info = [
                                row.get('Athlete_Name', 'Unknown'),
                                row.get('PfSc', 0),
                                row.get('Competition', 'Unknown Competition'),
                                row.get('Date', 'Unknown Date'),
                                row.get('Event', 'Unknown Event'),
                                row.get('Pl', 'N/A'),
                                row.get('Result', 'N/A')
                            ]
                            pfsc_hover_data.append(hover_info)
                        
                        # Overlay directly on main box plot
                        x_positions = [0] * len(pfsc_valid)  # Same x as main box plot
                        
                        fig.add_trace(go.Scatter(
                            y=pfsc_valid['PfSc'],
                            x=x_positions,
                            mode='markers',
                            name='Performance Scores (PfSc)',
                            marker=dict(
                                color='orange',
                                size=10,
                                symbol='square',
                                line=dict(width=2, color='black'),
                                opacity=0.8
                            ),
                            text=pfsc_valid['Athlete_Name'],
                            customdata=pfsc_hover_data,
                            hovertemplate="<b>%{text}</b><br>" +
                                        "Performance Score: %{y}<br>" +
                                        "Event: %{customdata[4]}<br>" +
                                        "Place: %{customdata[5]}<br>" +
                                        "Result: %{customdata[6]}<br>" +
                                        "Competition: %{customdata[2]}<br>" +
                                        "Date: %{customdata[3]}<br>" +
                                        "<extra>Performance Scores</extra>"
                        ))
                        
                        st.info(f"📊 Overlaid {len(pfsc_valid)} performance scores (orange squares)")
                    
                    overlay_added = True
            
            if not overlay_added:
                st.warning("No ranking score or performance score data available for overlay")
                                group_value = matching_rows.iloc[0][overlay_color_by]
                                pfsc_valid_with_groups.loc[idx, 'overlay_group'] = group_value
                            else:
                                pfsc_valid_with_groups.loc[idx, 'overlay_group'] = 'Unknown'
                        
                        # Plot by groups
                        groups = pfsc_valid_with_groups['overlay_group'].unique()
                        colors = px.colors.qualitative.Set3[:len(groups)]
                        
                        for i, group in enumerate(groups):
                            group_data = pfsc_valid_with_groups[pfsc_valid_with_groups['overlay_group'] == group]
                            if not group_data.empty:
                                pfsc_hover_data = []
                                for _, row in group_data.iterrows():
                                    hover_info = [
                                        row.get('Athlete_Name', 'Unknown'),
                                        row.get('PfSc', 0),
                                        row.get('Competition', 'Unknown Competition'),
                                        row.get('Date', 'Unknown Date'),
                                        row.get('Event', 'Unknown Event'),
                                        row.get('Pl', 'N/A'),
                                        row.get('Result', 'N/A'),
                                        group
                                    ]
                                    pfsc_hover_data.append(hover_info)
                                
                                fig.add_trace(go.Scatter(
                                    y=group_data['PfSc'],
                                    x=[1.3 + i * 0.05] * len(group_data),  # Further offset by group
                                    mode='markers',
                                    name=f'PfSc ({overlay_color_by}): {group}',
                                    marker=dict(
                                        color=colors[i % len(colors)],
                                        size=8,
                                        symbol='square',
                                        line=dict(width=1, color='black')
                                    ),
                                    text=group_data['Athlete_Name'],
                                    customdata=pfsc_hover_data,
                                    hovertemplate="<b>%{text}</b><br>" +
                                                "Performance Score: %{y}<br>" +
                                                f"{overlay_color_by}: %{{customdata[7]}}<br>" +
                                                "Event: %{customdata[4]}<br>" +
                                                "Place: %{customdata[5]}<br>" +
                                                "Result: %{customdata[6]}<br>" +
                                                "Competition: %{customdata[2]}<br>" +
                                                "Date: %{customdata[3]}<br>" +
                                                f"<extra>Performance by {overlay_color_by}</extra>"
                                ))
                        
                        st.info(f"📊 Added {len(pfsc_valid)} performance scores grouped by {overlay_color_by}")
                    else:
                        # Single group
                        pfsc_hover_data = []
                        for _, row in pfsc_valid.iterrows():
                            hover_info = [
                                row.get('Athlete_Name', 'Unknown'),
                                row.get('PfSc', 0),
                                row.get('Competition', 'Unknown Competition'),
                                row.get('Date', 'Unknown Date'),
                                row.get('Event', 'Unknown Event'),
                                row.get('Pl', 'N/A'),
                                row.get('Result', 'N/A')
                            ]
                            pfsc_hover_data.append(hover_info)
                        
                        fig.add_trace(go.Scatter(
                            y=pfsc_valid['PfSc'],
                            x=[1.3] * len(pfsc_valid),
                            mode='markers',
                            name='Performance Scores (PfSc)',
                            marker=dict(
                                color='orange',
                                size=8,
                                symbol='square',
                                line=dict(width=1, color='darkorange')
                            ),
                            text=pfsc_valid['Athlete_Name'],
                            customdata=pfsc_hover_data,
                            hovertemplate="<b>%{text}</b><br>" +
                                        "Performance Score: %{y}<br>" +
                                        "Event: %{customdata[4]}<br>" +
                                        "Place: %{customdata[5]}<br>" +
                                        "Result: %{customdata[6]}<br>" +
                                        "Competition: %{customdata[2]}<br>" +
                                        "Date: %{customdata[3]}<br>" +
                                        "<extra>Performance Scores</extra>"
                        ))
                        
                        st.info(f"📊 Added {len(pfsc_valid)} performance scores (orange squares)")
                    
                    overlay_added = True
            
            if not overlay_added:
                st.warning("No ranking score or performance score data available for overlay")
        else:
            st.warning("No performance data available for overlay")
    
    # Update layout
    fig.update_layout(
        title=f"{title} - {event_filter}" if event_filter != "All Events" else title,
        yaxis_title=y_title,
        showlegend=True,
        height=600,
        hovermode='closest'
    )
    
    return fig

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<div class="main-header">🏃‍♂️ Road to Tokyo 2025 - Athletics Qualification Dashboard</div>', 
                unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
        qualification_metadata = load_qualification_metadata()
        performance_df = load_athlete_performance_data()
    
    if df.empty:
        st.error("No data loaded. Please check your file structure and ensure CSV files are in the correct location.")
        st.info("Expected path: `world_athletics/Data/road_to/road_to_tokyo_batch_*.csv`")
        return
    
    # Sidebar filters
    st.sidebar.header("🎯 Filters")
    
    # Add "Select All" options
    st.sidebar.subheader("Quick Selections")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Select All Events", key="select_all_events"):
            st.session_state.selected_events_state = sorted(df['Actual_Event_Name'].unique())
    with col2:
        if st.button("Clear Events", key="clear_events"):
            st.session_state.selected_events_state = []
    
    # Event filter
    events = sorted(df['Actual_Event_Name'].unique())
    st.sidebar.info(f"📊 Available Events: {len(events)}")
    
    # Initialize session state for events if not exists
    if 'selected_events_state' not in st.session_state:
        # Start with more events by default
        default_events = events[:15] if len(events) > 15 else events  
        st.session_state.selected_events_state = default_events
    
    selected_events = st.sidebar.multiselect(
        "Select Events",
        events,
        default=st.session_state.selected_events_state,
        help="Choose which events to include in the analysis. Use buttons above for quick selection."
    )
    
    # Update session state
    st.session_state.selected_events_state = selected_events
    
    # Show selected events count
    st.sidebar.success(f"✅ Selected: {len(selected_events)} events")
    
    # Federation filter
    federations = sorted(df['Federation'].unique())
    st.sidebar.info(f"🌍 Available Federations: {len(federations)}")
    
    # Add quick selection for federations
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Select All Federations", key="select_all_feds"):
            st.session_state.selected_federations_state = federations
    with col2:
        if st.button("Clear Federations", key="clear_feds"):
            st.session_state.selected_federations_state = []
    
    # Initialize session state for federations
    if 'selected_federations_state' not in st.session_state:
        st.session_state.selected_federations_state = federations[:10] if len(federations) > 10 else federations
    
    selected_federations = st.sidebar.multiselect(
        "Select Federations",
        federations,
        default=st.session_state.selected_federations_state,
        help="Choose which federations to include"
    )
    
    st.session_state.selected_federations_state = selected_federations
    st.sidebar.success(f"✅ Selected: {len(selected_federations)} federations")
    
    # Qualification Status filter
    qual_statuses = sorted(df['Qualification_Status'].unique())
    st.sidebar.info(f"🎯 Available Statuses: {len(qual_statuses)}")
    
    selected_statuses = st.sidebar.multiselect(
        "Select Qualification Status",
        qual_statuses,
        default=qual_statuses,
        help="Choose which qualification statuses to include"
    )
    
    st.sidebar.success(f"✅ Selected: {len(selected_statuses)} qualification statuses")
    
    # Filter data
    filtered_df = df[
        (df['Actual_Event_Name'].isin(selected_events)) &
        (df['Federation'].isin(selected_federations)) &
        (df['Qualification_Status'].isin(selected_statuses))
    ]
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Analysis", "📋 Qualification Process", "📈 Detailed Statistics", "🔍 Data Debug"])
    
    with tab4:
        st.header("Data Debug Information")
        
        if not df.empty:
            st.subheader("📊 Data Overview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Dataset Shape:**")
                st.write(f"Rows: {len(df)}")
                st.write(f"Columns: {len(df.columns)}")
                
                st.write("**Column Names:**")
                for col in df.columns:
                    st.write(f"• {col}")
            
            with col2:
                st.write("**Unique Events:**")
                if 'Actual_Event_Name' in df.columns:
                    unique_events = df['Actual_Event_Name'].unique()
                    st.write(f"Total: {len(unique_events)}")
                    for event in sorted(unique_events):
                        count = len(df[df['Actual_Event_Name'] == event])
                        st.write(f"• {event}: {count} records")
                else:
                    st.error("'Actual_Event_Name' column not found!")
                
                st.write("**Sample Column_6 Values:**")
                if 'Column_6' in df.columns:
                    sample_values = df['Column_6'].dropna().head(10)
                    for i, val in enumerate(sample_values):
                        st.write(f"• {val}")
                else:
                    st.error("'Column_6' column not found!")
            
            st.subheader("📋 Raw Data Sample")
            st.dataframe(df.head(20), use_container_width=True)
            
            # Show data types
            st.subheader("📊 Data Types")
            st.write(df.dtypes)
            
        else:
            st.error("No data loaded!")
    
    with tab1:
        st.header("Performance Analysis")
        
        if not filtered_df.empty:
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Athletes", len(filtered_df))
            
            with col2:
                st.metric("Events", filtered_df['Actual_Event_Name'].nunique())
            
            with col3:
                st.metric("Federations", filtered_df['Federation'].nunique())
            
            with col4:
                qualified_count = len(filtered_df[filtered_df['Qualification_Status'].str.contains('Qualified', na=False)])
                st.metric("Qualified Athletes", qualified_count)
            
            # Event-specific visualization section
            st.subheader("📊 Performance Visualization")
            
            # Event filter for visualization (separate from sidebar)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                events_for_viz = ["All Events"] + sorted(filtered_df['Actual_Event_Name'].unique())
                selected_event_viz = st.selectbox(
                    "Select Event for Visualization",
                    events_for_viz,
                    key="viz_event_select"
                )
            
            with col2:
                plot_type = st.selectbox(
                    "Select Data Type",
                    ["Time/Distance Values", "Ranking Points"],
                    key="plot_type"
                )
                
                y_column = 'time_value' if plot_type == "Time/Distance Values" else 'points_value'
            
            with col3:
                color_by = st.selectbox(
                    "Color/Group By",
                    ["None", "Qualification_Status", "Federation"],
                    key="color_by"
                )
                
                color_column = None if color_by == "None" else color_by
            
            # Performance overlay options
            st.subheader("🎯 Performance Overlay Options")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                overlay_performance = st.checkbox(
                    "📊 Overlay Competition Performances", 
                    value=False,
                    help="Add individual athlete competition results from detailed performance data"
                )
            
            with col2:
                if not performance_df.empty:
                    st.success(f"✅ {len(performance_df)} performance records available")
                    
                    # Country selection for performance overlay - handle mixed data types
                    if 'Country_Code' in performance_df.columns:
                        # Clean country codes - handle NaN and mixed types
                        country_codes = performance_df['Country_Code'].dropna().astype(str).unique()
                        perf_countries = sorted([code for code in country_codes if code.lower() not in ['nan', 'none', '']])
                    else:
                        perf_countries = []
                        
                    selected_perf_country = st.selectbox(
                        "Select Country for Overlay",
                        ["All Countries"] + perf_countries,
                        key="perf_country_select"
                    )
                else:
                    st.info("ℹ️ No performance data loaded")
                    overlay_performance = False
                    selected_perf_country = "All Countries"
            
            with col3:
                # Athlete selection for performance overlay
                if overlay_performance and not performance_df.empty:
                    # Filter athletes by selected country
                    if selected_perf_country != "All Countries":
                        country_athletes_df = performance_df[
                            performance_df['Country_Code'].astype(str).str.lower() == selected_perf_country.lower()
                        ]
                    else:
                        country_athletes_df = performance_df
                    
                    if not country_athletes_df.empty and 'Athlete_Name' in country_athletes_df.columns:
                        # Clean athlete names
                        athlete_names = country_athletes_df['Athlete_Name'].dropna().astype(str).unique()
                        perf_athletes = sorted([name for name in athlete_names if name.lower() not in ['nan', 'none', '']])
                        selected_perf_athlete = st.selectbox(
                            "Select Athlete for Overlay",
                            ["All Athletes"] + perf_athletes,
                            key="perf_athlete_select"
                        )
                    else:
                        selected_perf_athlete = "All Athletes"
                        st.info("No athletes found")
                else:
                    selected_perf_athlete = "All Athletes"d_perf_country != "All Countries":
                        country_athletes_df = performance_df[
                            performance_df['Country_Code'].astype(str).str.lower() == selected_perf_country.lower()
                        ]
                    else:
                        country_athletes_df = performance_df
                    
                    if not country_athletes_df.empty and 'Athlete_Name' in country_athletes_df.columns:
                        # Clean athlete names
                        athlete_names = country_athletes_df['Athlete_Name'].dropna().astype(str).unique()
                        perf_athletes = sorted([name for name in athlete_names if name.lower() not in ['nan', 'none', '']])
                        selected_perf_athlete = st.selectbox(
                            "Select Athlete for Overlay",
                            ["All Athletes"] + perf_athletes,
                            key="perf_athlete_select"
                        )
                    else:
                        selected_perf_athlete = "All Athletes"
                        st.info("No athletes found")
                else:
                    selected_perf_athlete = "All Athletes"
            
            with col4:
                # Overlay grouping option
                if overlay_performance:
                    overlay_color_options = ["None", "Qualification_Status", "Federation", "Status"]
                    # Add any other categorical columns from the main dataset
                    if not filtered_df.empty:
                        categorical_cols = filtered_df.select_dtypes(include=['object']).columns
                        additional_cols = [col for col in categorical_cols if col not in overlay_color_options and col not in ['Athlete', 'Event_Type', 'Actual_Event_Name']]
                        overlay_color_options.extend(additional_cols[:3])  # Limit to avoid overwhelming UI
                    
                    overlay_color_by = st.selectbox(
                        "Group Overlay By",
                        overlay_color_options,
                        key="overlay_color_by",
                        help="Color-code overlay points by this category"
                    )
                else:
                    overlay_color_by = "None"d_perf_country != "All Countries":
                        country_athletes_df = performance_df[
                            performance_df['Country_Code'].astype(str).str.lower() == selected_perf_country.lower()
                        ]
                    else:
                        country_athletes_df = performance_df
                    
                    if not country_athletes_df.empty and 'Athlete_Name' in country_athletes_df.columns:
                        # Clean athlete names
                        athlete_names = country_athletes_df['Athlete_Name'].dropna().astype(str).unique()
                        perf_athletes = sorted([name for name in athlete_names if name.lower() not in ['nan', 'none', '']])
                        selected_perf_athlete = st.selectbox(
                            "Select Athlete for Overlay",
                            ["All Athletes"] + perf_athletes,
                            key="perf_athlete_select"
                        )
                    else:
                        selected_perf_athlete = "All Athletes"
                        st.info("No athletes found")
                else:
                    selected_perf_athlete = "All Athletes"
            
            # Performance data type selection
            if overlay_performance and not performance_df.empty:
                if y_column == 'time_value':
                    # Time/Distance data options
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        performance_data_type = st.selectbox(
                            "Performance Data to Plot",
                            ["Result", "PfSc (Performance Score)"],
                            key="perf_data_type"
                        )
                    
                    with col2:
                        show_ranking_score = st.checkbox(
                            "Show Current Ranking Score",
                            value=True,
                            help="Display Ranking_Score as additional information"
                        )
                
                elif y_column == 'points_value':
                    # Ranking Points data options
                    st.subheader("📊 Ranking Points Overlay Options")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        show_ranking_score = st.checkbox(
                            "Show Current Ranking Score",
                            value=True,
                            help="Display current Ranking_Score (blue circles with white border)"
                        )
                    
                    with col2:
                        show_pfsc_scores = st.checkbox(
                            "Show Performance Scores (PfSc)",
                            value=True,
                            help="Display individual Performance Scores from competitions (orange squares)"
                        )
                    
                    # Set performance_data_type for consistency
                    performance_data_type = "Both Scores"
                    
                    st.info("💡 **Legend**: Blue circles (white border) = Current Ranking Score | Orange squares = Individual Performance Scores")
                
                else:
                    # Default fallback
                    performance_data_type = "Result"
                    show_ranking_score = True
            
            # Show selection summary
            if overlay_performance:
                summary_parts = []
                if selected_perf_country != "All Countries":
                    summary_parts.append(f"Country: {selected_perf_country}")
                if selected_perf_athlete != "All Athletes":
                    summary_parts.append(f"Athlete: {selected_perf_athlete}")
                if 'performance_data_type' in locals():
                    summary_parts.append(f"Data: {performance_data_type}")
                if overlay_color_by != "None":
                    summary_parts.append(f"Grouped by: {overlay_color_by}")
                
                if summary_parts:
                    st.info(f"🎯 Performance overlay: {' | '.join(summary_parts)}")
                
                # Show grouping legend if active
                if overlay_color_by != "None" and overlay_color_by in filtered_df.columns:
                    unique_groups = filtered_df[overlay_color_by].unique()
                    if len(unique_groups) <= 10:  # Only show if not too many groups
                        st.info(f"🎨 **{overlay_color_by} Groups**: {', '.join(map(str, unique_groups))}")
                    else:
                        st.info(f"🎨 **{overlay_color_by}**: {len(unique_groups)} different groups in dataset")
            
            # Create and display plot
            perf_data_column = "Result" if 'performance_data_type' not in locals() else ("Result" if performance_data_type == "Result" else "PfSc")
            show_rankings = 'show_ranking_score' in locals() and show_ranking_score
            show_pfsc = 'show_pfsc_scores' in locals() and show_pfsc_scores
            overlay_group_by = overlay_color_by if overlay_performance else "None"
            
            fig = create_box_plot(
                df=filtered_df, 
                event_filter=selected_event_viz,
                y_column=y_column, 
                title=f"{plot_type} Distribution",
                color_column=color_column,
                overlay_performance=overlay_performance,
                performance_df=performance_df,
                selected_country=selected_perf_country,
                selected_athlete=selected_perf_athlete,
                perf_data_column=perf_data_column,
                show_ranking_score=show_rankings,
                show_pfsc_scores=show_pfsc,
                overlay_color_by=overlay_group_by
            )
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
                # Show summary statistics for the selected event
                if selected_event_viz != "All Events":
                    event_data = filtered_df[filtered_df['Actual_Event_Name'] == selected_event_viz]
                    
                    # Parse the data for statistics
                    event_data_copy = event_data.copy()
                    event_data_copy[['time_value', 'points_value', 'ranking']] = event_data_copy['Column_6'].apply(
                        lambda x: pd.Series(parse_column6_data(x))
                    )
                    
                    if y_column == 'time_value':
                        valid_data = event_data_copy.dropna(subset=['time_value'])
                        if not valid_data.empty:
                            st.subheader(f"📈 {selected_event_viz} - Time/Distance Statistics")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Best Performance", f"{valid_data['time_value'].min():.2f}")
                            with col2:
                                st.metric("Average", f"{valid_data['time_value'].mean():.2f}")
                            with col3:
                                st.metric("Median", f"{valid_data['time_value'].median():.2f}")
                            with col4:
                                st.metric("Worst Performance", f"{valid_data['time_value'].max():.2f}")
                    
                    elif y_column == 'points_value':
                        valid_data = event_data_copy.dropna(subset=['points_value'])
                        if not valid_data.empty:
                            st.subheader(f"📈 {selected_event_viz} - Ranking Points Statistics")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Highest Points", f"{valid_data['points_value'].max()}")
                            with col2:
                                st.metric("Average Points", f"{valid_data['points_value'].mean():.0f}")
                            with col3:
                                st.metric("Median Points", f"{valid_data['points_value'].median():.0f}")
                            with col4:
                                st.metric("Lowest Points", f"{valid_data['points_value'].min()}")
            
            # Data table
            st.subheader("📋 Filtered Data")
            
            # Filter table data by the selected event for visualization
            if selected_event_viz != "All Events":
                table_df = filtered_df[filtered_df['Actual_Event_Name'] == selected_event_viz]
                st.info(f"Showing data for: **{selected_event_viz}** ({len(table_df)} athletes)")
            else:
                table_df = filtered_df
                st.info(f"Showing data for: **All Events** ({len(table_df)} athletes)")
            
            # Show relevant columns
            display_columns = [
                'Actual_Event_Name', 'Federation', 'Qualification_Status', 
                'Athlete', 'Status', 'Column_6'
            ]
            
            available_columns = [col for col in display_columns if col in table_df.columns]
            
            # Add a search filter for the data table
            if 'Athlete' in table_df.columns and not table_df.empty:
                athlete_search = st.text_input("🔍 Search Athletes", placeholder="Enter athlete name...")
                if athlete_search:
                    display_df = table_df[table_df['Athlete'].str.contains(athlete_search, case=False, na=False)]
                    st.info(f"Found {len(display_df)} athletes matching '{athlete_search}'")
                else:
                    display_df = table_df
            else:
                display_df = table_df
            
            # Display the table
            if not display_df.empty:
                st.dataframe(
                    display_df[available_columns],
                    use_container_width=True,
                    height=400
                )
                
                # Show summary of what's displayed
                st.caption(f"Displaying {len(display_df)} rows with {len(available_columns)} columns")
            else:
                st.warning("No data to display with current filters")
            
        else:
            st.warning("No data matches the selected filters. Please adjust your selections.")
    
    with tab2:
        st.header("Qualification Process Information")
        
        if not qualification_metadata.empty:
            # Event selection for qualification details
            if 'Display_Name' in qualification_metadata.columns:
                qual_events = sorted(qualification_metadata['Display_Name'].unique())
                selected_qual_event = st.selectbox(
                    "Select Event for Qualification Details",
                    qual_events,
                    key="qual_event_select"
                )
                
                # Display qualification details
                event_data = qualification_metadata[
                    qualification_metadata['Display_Name'] == selected_qual_event
                ].iloc[0]
                
                # Create qualification info card
                st.markdown(f"""
                <div class="event-card">
                    <h3>{selected_qual_event}</h3>
                    <p><strong>Event ID:</strong> {event_data.get('Event_ID', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Qualification details
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📊 Entry Information")
                    if 'entry_number' in event_data and pd.notna(event_data['entry_number']):
                        st.info(f"**Entry Number:** {event_data['entry_number']}")
                    
                    if 'entry_standard' in event_data and pd.notna(event_data['entry_standard']):
                        st.info(f"**Entry Standard:** {event_data['entry_standard']}")
                    
                    if 'maximum_quota' in event_data and pd.notna(event_data['maximum_quota']):
                        st.info(f"**Maximum Quota per Federation:** {event_data['maximum_quota']}")
                
                with col2:
                    st.markdown("### 📅 Important Dates")
                    if 'qualification_period' in event_data and pd.notna(event_data['qualification_period']):
                        st.info(f"**Qualification Period:** {event_data['qualification_period']}")
                    
                    if 'world_rankings_period' in event_data and pd.notna(event_data['world_rankings_period']):
                        st.info(f"**World Rankings Period:** {event_data['world_rankings_period']}")
                
                # Athletes breakdown
                st.markdown("### 👥 Athletes Breakdown")
                breakdown_cols = st.columns(3)
                
                breakdown_fields = [
                    ('athletes_by_entry_standard', 'By Entry Standard'),
                    ('athletes_by_finishing_position', 'By Finishing Position'),
                    ('athletes_by_world_rankings', 'By World Rankings'),
                    ('athletes_by_top_list', 'By Top List'),
                    ('athletes_by_universality', 'By Universality Places')
                ]
                
                for i, (field, label) in enumerate(breakdown_fields):
                    with breakdown_cols[i % 3]:
                        if field in event_data and pd.notna(event_data[field]):
                            value = event_data[field]
                            st.metric(label, value)
                
                # Additional notes
                if 'additional_notes' in event_data and pd.notna(event_data['additional_notes']):
                    st.markdown("### 📝 Additional Notes")
                    st.markdown(f"""
                    <div class="qualification-text">
                    {event_data['additional_notes']}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Full qualification text
                if 'full_qualification_text' in event_data and pd.notna(event_data['full_qualification_text']):
                    with st.expander("📖 Full Qualification Text"):
                        st.markdown(f"""
                        <div class="qualification-text">
                        {event_data['full_qualification_text']}
                        </div>
                        """, unsafe_allow_html=True)
            
            else:
                st.info("Qualification metadata is available but in a different format.")
                st.dataframe(qualification_metadata.head())
        
        else:
            st.warning("No qualification process metadata available.")
            st.info("To load qualification data, run the qualification process scraper first.")
    
    with tab3:
        st.header("Detailed Statistics")
        
        if not filtered_df.empty:
            # Event-wise statistics
            st.subheader("📊 Event-wise Statistics")
            
            event_stats = filtered_df.groupby('Actual_Event_Name').agg({
                'Athlete': 'count',
                'Federation': 'nunique'
            }).rename(columns={'Athlete': 'Total_Athletes', 'Federation': 'Federations_Count'})
            
            st.dataframe(event_stats, use_container_width=True)
            
            # Federation-wise statistics
            st.subheader("🌍 Federation-wise Statistics")
            
            fed_stats = filtered_df.groupby('Federation').agg({
                'Athlete': 'count',
                'Actual_Event_Name': 'nunique'
            }).rename(columns={'Athlete': 'Total_Athletes', 'Actual_Event_Name': 'Events_Count'})
            
            st.dataframe(fed_stats, use_container_width=True)
            
            # Qualification status distribution
            st.subheader("🎯 Qualification Status Distribution")
            
            status_counts = filtered_df['Qualification_Status'].value_counts()
            
            fig_pie = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Qualification Status Distribution"
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
        else:
            st.warning("No data available for detailed statistics.")

if __name__ == "__main__":
    main()