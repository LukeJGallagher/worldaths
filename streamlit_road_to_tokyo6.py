# Replace these functions in your existing streamlit app

@st.cache_data
def load_data():
    """Enhanced data loading function - finds ALL qualification data files"""
    try:
        all_data = []
        file_info = []
        
        # Multiple search locations
        search_locations = [
            "world_athletics/Data/road_to",
            "world_athletics/Data", 
            "Data/road_to",
            "Data",
            ".",  # current directory
        ]
        
        # Multiple file patterns
        file_patterns = [
            "road_to_tokyo_batch_*.csv",
            "road_to_batch_*.csv",
            "batch_*.csv", 
            "*tokyo*.csv",
            "*road_to*.csv",
            "*qualification*.csv"
        ]
        
        st.sidebar.write("🔍 **Enhanced Data Loading:**")
        files_found = 0
        
        for location in search_locations:
            if os.path.exists(location):
                st.sidebar.write(f"📁 Searching: {location}")
                
                for pattern in file_patterns:
                    csv_files = glob.glob(os.path.join(location, pattern))
                    
                    for file in csv_files:
                        filename = os.path.basename(file)
                        
                        # Avoid duplicates
                        if not any(info['file'] == filename for info in file_info):
                            try:
                                df = pd.read_csv(file)
                                all_data.append(df)
                                
                                file_info.append({
                                    'file': filename,
                                    'rows': len(df),
                                    'events': df['Actual_Event_Name'].nunique() if 'Actual_Event_Name' in df.columns else 0
                                })
                                files_found += 1
                                st.sidebar.success(f"✅ {filename}: {len(df)} rows")
                                
                            except Exception as e:
                                st.sidebar.error(f"❌ Error loading {filename}: {e}")
        
        # Also check current directory for any CSV that looks like qualification data
        current_csvs = glob.glob("*.csv")
        for csv_file in current_csvs:
            filename = os.path.basename(csv_file)
            if not any(info['file'] == filename for info in file_info):
                try:
                    df = pd.read_csv(csv_file)
                    # Check if it looks like qualification data
                    if any(col in df.columns for col in ['Athlete', 'Federation', 'Event', 'Qualification_Status']):
                        all_data.append(df)
                        file_info.append({
                            'file': filename,
                            'rows': len(df),
                            'events': df['Actual_Event_Name'].nunique() if 'Actual_Event_Name' in df.columns else 0
                        })
                        files_found += 1
                        st.sidebar.success(f"✅ {filename}: {len(df)} rows (current dir)")
                except:
                    pass
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Enhanced summary
            st.sidebar.write("📊 **Final Summary:**")
            st.sidebar.write(f"Files loaded: {files_found}")
            st.sidebar.write(f"Total rows: {len(combined_df):,}")
            st.sidebar.write(f"Total columns: {len(combined_df.columns)}")
            
            if 'Actual_Event_Name' in combined_df.columns:
                unique_events = combined_df['Actual_Event_Name'].nunique()
                st.sidebar.write(f"Unique events: {unique_events}")
            
            if 'Federation' in combined_df.columns:
                unique_feds = combined_df['Federation'].nunique()
                st.sidebar.write(f"Unique federations: {unique_feds}")
            
            if 'Athlete' in combined_df.columns:
                unique_athletes = combined_df['Athlete'].nunique()
                st.sidebar.write(f"Unique athletes: {unique_athletes}")
            
            # Detailed breakdown
            with st.sidebar.expander("📋 Detailed File Info"):
                for info in file_info:
                    st.write(f"**{info['file']}**: {info['rows']} rows, {info['events']} events")
            
            return combined_df
        else:
            st.sidebar.error("❌ No qualification data files found!")
            # Show current directory for debugging
            st.sidebar.write(f"Current directory: {os.getcwd()}")
            current_files = os.listdir(".")
            csv_files = [f for f in current_files if f.endswith('.csv')]
            if csv_files:
                st.sidebar.write("CSV files in current directory:")
                for f in csv_files:
                    st.sidebar.write(f"  • {f}")
            else:
                st.sidebar.write("No CSV files found in current directory")
            
            return pd.DataFrame()
            
    except Exception as e:
        st.sidebar.error(f"Error loading data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_athlete_performance_data():
    """Enhanced performance data loading"""
    try:
        all_performance_data = []
        file_info = []
        
        # Search locations for performance data
        search_locations = [
            "world_athletics/Data/athlete_performances",
            "world_athletics/Data",
            "Data/athlete_performances", 
            "Data",
            ".",
        ]
        
        # Performance file patterns
        performance_patterns = [
            "performances_*.csv",
            "athlete_*.csv",
            "*performance*.csv",
            "*modal*.csv"
        ]
        
        st.sidebar.write("🏃 **Performance Data Loading:**")
        
        for location in search_locations:
            if os.path.exists(location):
                for pattern in performance_patterns:
                    csv_files = glob.glob(os.path.join(location, pattern))
                    
                    for file in csv_files:
                        filename = os.path.basename(file)
                        if not any(info['file'] == filename for info in file_info):
                            try:
                                df = pd.read_csv(file)
                                all_performance_data.append(df)
                                
                                file_info.append({
                                    'file': filename,
                                    'rows': len(df),
                                    'athletes': df['Athlete_Name'].nunique() if 'Athlete_Name' in df.columns else 0
                                })
                                st.sidebar.success(f"✅ {filename}: {len(df)} records")
                                
                            except Exception as e:
                                st.sidebar.error(f"❌ {filename}: {e}")
        
        # Check current directory
        current_csvs = glob.glob("*.csv")
        for csv_file in current_csvs:
            filename = os.path.basename(csv_file)
            if filename.startswith('performances_') and not any(info['file'] == filename for info in file_info):
                try:
                    df = pd.read_csv(csv_file)
                    all_performance_data.append(df)
                    file_info.append({
                        'file': filename,
                        'rows': len(df),
                        'athletes': df['Athlete_Name'].nunique() if 'Athlete_Name' in df.columns else 0
                    })
                    st.sidebar.success(f"✅ {filename}: {len(df)} records (current dir)")
                except:
                    pass
        
        if all_performance_data:
            combined_df = pd.concat(all_performance_data, ignore_index=True)
            
            st.sidebar.write("🏃 **Performance Summary:**")
            st.sidebar.write(f"Files: {len(file_info)}")
            st.sidebar.write(f"Records: {len(combined_df):,}")
            
            if 'Athlete_Name' in combined_df.columns:
                st.sidebar.write(f"Athletes: {combined_df['Athlete_Name'].nunique()}")
            if 'Country_Code' in combined_df.columns:
                st.sidebar.write(f"Countries: {combined_df['Country_Code'].nunique()}")
            
            return combined_df
        else:
            st.sidebar.warning("⚠️ No performance data found")
            return pd.DataFrame()
            
    except Exception as e:
        st.sidebar.warning(f"Performance data error: {e}")
        return pd.DataFrame()

# Add this diagnostic function to your app
def show_data_diagnostics():
    """Add this as a new tab to help diagnose data loading issues"""
    st.header("🔧 Data Loading Diagnostics")
    
    st.subheader("Current Directory Structure")
    current_dir = os.getcwd()
    st.write(f"Working directory: `{current_dir}`")
    
    # Show all CSV files found
    all_csvs = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                try:
                    size = os.path.getsize(file_path) / 1024  # KB
                    all_csvs.append({
                        'path': file_path,
                        'name': file,
                        'size_kb': size
                    })
                except:
                    pass
    
    if all_csvs:
        st.write(f"Found {len(all_csvs)} CSV files:")
        df_files = pd.DataFrame(all_csvs)
        st.dataframe(df_files)
        
        # Quick peek at each file
        st.subheader("File Contents Preview")
        for csv_info in all_csvs[:5]:  # Show first 5 files
            with st.expander(f"📄 {csv_info['name']} ({csv_info['size_kb']:.1f} KB)"):
                try:
                    df = pd.read_csv(csv_info['path'], nrows=3)
                    st.write(f"Columns: {list(df.columns)}")
                    st.dataframe(df)
                except Exception as e:
                    st.error(f"Error reading file: {e}")
    else:
        st.error("No CSV files found!")
