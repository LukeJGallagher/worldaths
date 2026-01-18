import streamlit as st
import pandas as pd
import os
import glob
from pathlib import Path

def comprehensive_data_scan():
    """Scan entire directory structure for any CSV files that might contain athletics data"""
    
    st.write("## 🔍 Comprehensive Data Discovery")
    st.write("Scanning for ALL CSV files that might contain athletics qualification data...")
    
    discovered_files = []
    
    # Scan multiple directory levels
    search_patterns = [
        "**/*.csv",  # All CSV files recursively
        "*.csv",     # CSV files in current directory
        "**/road_to*.csv",
        "**/batch*.csv", 
        "**/tokyo*.csv",
        "**/performance*.csv",
        "**/athlete*.csv"
    ]
    
    for pattern in search_patterns:
        try:
            files = list(Path(".").glob(pattern))
            for file in files:
                if file not in [d['path'] for d in discovered_files]:
                    try:
                        # Quick peek at the file
                        df = pd.read_csv(file, nrows=5)  # Just read first 5 rows
                        
                        # Classify file type based on columns
                        file_type = "Unknown"
                        if any(col in df.columns for col in ['Athlete', 'Federation', 'Event', 'Qualification_Status']):
                            file_type = "Qualification Data"
                        elif any(col in df.columns for col in ['Athlete_Name', 'Competition', 'Result', 'Performance']):
                            file_type = "Performance Data"
                        elif any(col in df.columns for col in ['Event_ID', 'entry_standard', 'qualification_period']):
                            file_type = "Qualification Metadata"
                        
                        discovered_files.append({
                            'path': file,
                            'name': file.name,
                            'size_mb': file.stat().st_size / (1024*1024),
                            'type': file_type,
                            'columns': list(df.columns),  
                            'sample_data': df.head(2).to_dict('records') if len(df) > 0 else []
                        })
                        
                    except Exception as e:
                        # File exists but can't read it
                        discovered_files.append({
                            'path': file,
                            'name': file.name, 
                            'size_mb': file.stat().st_size / (1024*1024),
                            'type': "Error reading file",
                            'columns': [],
                            'sample_data': [],
                            'error': str(e)
                        })
        except:
            continue
    
    if discovered_files:
        st.success(f"🎉 Found {len(discovered_files)} CSV files!")
        
        # Group by type
        qualification_files = [f for f in discovered_files if f['type'] == 'Qualification Data']
        performance_files = [f for f in discovered_files if f['type'] == 'Performance Data'] 
        metadata_files = [f for f in discovered_files if f['type'] == 'Qualification Metadata']
        unknown_files = [f for f in discovered_files if f['type'] not in ['Qualification Data', 'Performance Data', 'Qualification Metadata']]
        
        # Display results by category
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Qualification Data Files")
            if qualification_files:
                for file in qualification_files:
                    with st.expander(f"📄 {file['name']} ({file['size_mb']:.1f}MB)"):
                        st.write(f"**Path:** `{file['path']}`")
                        st.write(f"**Columns:** {', '.join(file['columns'][:10])}")
                        if len(file['columns']) > 10:
                            st.write(f"... and {len(file['columns']) - 10} more columns")
                        
                        if file['sample_data']:
                            st.write("**Sample Data:**")
                            st.json(file['sample_data'][0] if file['sample_data'] else {})
            else:
                st.info("No qualification data files found")
            
            st.subheader("📊 Performance Data Files")  
            if performance_files:
                for file in performance_files:
                    with st.expander(f"📄 {file['name']} ({file['size_mb']:.1f}MB)"):
                        st.write(f"**Path:** `{file['path']}`")
                        st.write(f"**Columns:** {', '.join(file['columns'])}")
                        
                        if file['sample_data']:
                            st.write("**Sample Data:**")
                            st.json(file['sample_data'][0] if file['sample_data'] else {})
            else:
                st.info("No performance data files found")
        
        with col2:
            st.subheader("📋 Metadata Files")
            if metadata_files:
                for file in metadata_files:
                    with st.expander(f"📄 {file['name']} ({file['size_mb']:.1f}MB)"):
                        st.write(f"**Path:** `{file['path']}`")
                        st.write(f"**Columns:** {', '.join(file['columns'])}")
            else:
                st.info("No metadata files found")
            
            st.subheader("❓ Unknown/Other Files")
            if unknown_files:
                for file in unknown_files:
                    with st.expander(f"📄 {file['name']} ({file['size_mb']:.1f}MB)"):
                        st.write(f"**Path:** `{file['path']}`")
                        st.write(f"**Type:** {file['type']}")
                        if 'error' in file:
                            st.error(f"Error: {file['error']}")
                        else:
                            st.write(f"**Columns:** {', '.join(file['columns'])}")
            else:
                st.info("No unknown files found")
        
        # Generate loading code
        st.subheader("🔧 Generated Loading Code")
        st.write("Based on discovered files, here's the code to load all your data:")
        
        loading_code = "# Load all discovered data files\nimport pandas as pd\nimport glob\n\n"
        
        if qualification_files:
            loading_code += "# Qualification data files\nqualification_files = [\n"
            for file in qualification_files:
                loading_code += f"    r'{file['path']}',\n"
            loading_code += "]\n\nqualification_dfs = []\nfor file in qualification_files:\n    df = pd.read_csv(file)\n    qualification_dfs.append(df)\n    print(f'Loaded {file}: {len(df)} rows')\n\nall_qualification_data = pd.concat(qualification_dfs, ignore_index=True)\nprint(f'Total qualification rows: {len(all_qualification_data)}')\n\n"
        
        if performance_files:
            loading_code += "# Performance data files\nperformance_files = [\n"
            for file in performance_files:
                loading_code += f"    r'{file['path']}',\n"
            loading_code += "]\n\nperformance_dfs = []\nfor file in performance_files:\n    df = pd.read_csv(file)\n    performance_dfs.append(df)\n    print(f'Loaded {file}: {len(df)} rows')\n\nall_performance_data = pd.concat(performance_dfs, ignore_index=True)\nprint(f'Total performance rows: {len(all_performance_data)}')\n\n"
        
        st.code(loading_code, language='python')
        
        # Summary statistics
        st.subheader("📈 Summary Statistics")
        total_size = sum(f['size_mb'] for f in discovered_files)
        st.metric("Total Data Size", f"{total_size:.1f} MB")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Qualification Files", len(qualification_files))
        with col2:
            st.metric("Performance Files", len(performance_files))
        with col3:
            st.metric("Metadata Files", len(metadata_files))
        with col4:
            st.metric("Other Files", len(unknown_files))
            
    else:
        st.error("❌ No CSV files found in the current directory or subdirectories!")
        st.write("**Troubleshooting:**")
        st.write("1. Make sure you're running this from the correct directory")
        st.write("2. Check if your data files have .csv extension")
        st.write("3. Verify file permissions")

def manual_file_loader():
    """Allow manual file selection and loading"""
    st.write("## 📁 Manual File Loader")
    st.write("Can't find your files automatically? Upload them manually:")
    
    uploaded_files = st.file_uploader(
        "Upload CSV files",
        type=['csv'],
        accept_multiple_files=True,
        help="Upload all your athletics data CSV files"
    )
    
    if uploaded_files:
        st.success(f"Uploaded {len(uploaded_files)} files!")
        
        for uploaded_file in uploaded_files:
            try:
                df = pd.read_csv(uploaded_file)
                
                with st.expander(f"📄 {uploaded_file.name} - {len(df)} rows"):
                    st.write("**Columns:**", list(df.columns))
                    st.write("**Sample data:**")
                    st.dataframe(df.head())
                    
                    # Provide download button for processed file
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label=f"Download {uploaded_file.name}",
                        data=csv,
                        file_name=uploaded_file.name,
                        mime='text/csv'
                    )
                    
            except Exception as e:
                st.error(f"Error reading {uploaded_file.name}: {e}")

# Main diagnostic app
def main_diagnostic():
    st.title("🔍 Athletics Data Discovery Tool")
    st.write("This tool will help you find and load all your athletics qualification data.")
    
    tab1, tab2 = st.tabs(["🔍 Auto Discovery", "📁 Manual Upload"])
    
    with tab1:
        comprehensive_data_scan()
    
    with tab2:
        manual_file_loader()

if __name__ == "__main__":
    main_diagnostic()
