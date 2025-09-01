import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks
import json
import io

# Page configuration
st.set_page_config(
    page_title="Rowing Force Data Analyzer",
    page_icon="🚣",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styles
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    /* Reduce top spacing */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Reduce header spacing */
    .stHeader {
        margin-top: 0;
        padding-top: 0;
    }
    
    /* Reduce first element spacing */
    .main .block-container > div:first-child {
        margin-top: 0;
        padding-top: 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for real-time updates
if 'peak_threshold' not in st.session_state:
    st.session_state.peak_threshold = 0.7
if 'valley_threshold' not in st.session_state:
    st.session_state.valley_threshold = 0.7
if 'prominence' not in st.session_state:
    st.session_state.prominence = 0.1

@st.cache_data
def load_and_process_data(uploaded_file):
    """Load and preprocess JSON data"""
    try:
        # Read JSON file
        if uploaded_file is not None:
            data = json.load(uploaded_file)
        else:
            # If no file uploaded, use sample data
            st.warning("No file uploaded, using sample data")
            return None, "Please upload a JSON file to view actual data"
        
        # Extract data
        left_stroke_time = data['strokeForce']['left']['time']
        right_stroke_time = data['strokeForce']['right']['time']
        
        left_forward = data['lapForceTime']['left']['forward']
        right_forward = data['lapForceTime']['right']['forward']
        time = data['lapForceTime']['time']
        
        # Create DataFrame
        df = pd.DataFrame({
            'time': time, 
            'left_forward': left_forward, 
            'right_forward': right_forward
        })
        df.set_index('time', inplace=True)
        
        # Mark stroke cycles
        df['left_stroke_time'] = 0
        df['right_stroke_time'] = 0
        
        # Mark left hand stroke cycles
        for i in range(len(left_stroke_time) - 1):
            start = left_stroke_time[i]
            end = left_stroke_time[i + 1]
            df.loc[start:end, 'left_stroke_time'] = i + 1
        
        if len(left_stroke_time) > 0:
            df.loc[left_stroke_time[-1]:, 'left_stroke_time'] = len(left_stroke_time)
        
        # Mark right hand stroke cycles
        for i in range(len(right_stroke_time) - 1):
            start = right_stroke_time[i]
            end = right_stroke_time[i + 1]
            df.loc[start:end, 'right_stroke_time'] = i + 1
        
        if len(right_stroke_time) > 0:
            df.loc[right_stroke_time[-1]:, 'right_stroke_time'] = len(right_stroke_time)
        
        # Fill NaN values
        df['left_stroke_time'].fillna(0, inplace=True)
        df['right_stroke_time'].fillna(0, inplace=True)
        
        return df, None
        
    except Exception as e:
        return None, f"Data processing error: {str(e)}"

def create_force_plot(df, direction, stroke_number, peak_threshold, valley_threshold, prominence):
    """Create force curve plot"""
    stroke_col = f'{direction}_stroke_time'
    power_col = f'{direction}_forward'
    
    # Filter data for specified stroke cycle
    df_stroke = df[df[stroke_col] == stroke_number]
    
    if df_stroke.empty:
        fig = go.Figure()
        fig.add_annotation(
            text=f"No data found for {direction} direction stroke {stroke_number}",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="red")
        )
        fig.update_layout(
            title=f"{direction.capitalize()} Direction Stroke {stroke_number} - No Data",
            xaxis_title="Time", yaxis_title="Power (W)"
        )
        return fig, [] # Return empty list for warnings
    
    # Calculate statistics
    local_maxima = df_stroke[power_col].max()
    local_minima = df_stroke[power_col].min()
    
    # Detect peaks
    peaks, _ = find_peaks(
        df_stroke[power_col], 
        height=local_maxima * peak_threshold,
        # prominence=prominence * local_maxima
    )
    
    # Detect troughs
    if len(peaks) > 0:
        lowest_peak = df_stroke[power_col].iloc[peaks].min()
    else:
        lowest_peak = local_minima
    
    troughs, _ = find_peaks(
        -df_stroke[power_col], 
        height=-lowest_peak * valley_threshold,
        # height=-local_maxima * valley_threshold,
        prominence=prominence * local_maxima
    )
    
    # Analyze trough-peak relationships for double-peak pattern detection
    valid_troughs = []
    invalid_troughs = []
    pattern_warnings = []
    
    if len(peaks) >= 2 and len(troughs) > 0:
        # Check each trough to see if it's sandwiched between two peaks
        for trough_idx in troughs:
            trough_time = df_stroke.index[trough_idx]
            
            # Find peaks before and after this trough
            peaks_before = [p for p in peaks if df_stroke.index[p] < trough_time]
            peaks_after = [p for p in peaks if df_stroke.index[p] > trough_time]
            
            if len(peaks_before) > 0 and len(peaks_after) > 0:
                # Trough is sandwiched between peaks - valid double-peak pattern
                valid_troughs.append(trough_idx)
            else:
                # Trough is not sandwiched between peaks
                invalid_troughs.append(trough_idx)
                pattern_warnings.append(f"Trough at {trough_time}ms not sandwiched between peaks")
    
    elif len(peaks) == 1:
        # Only one peak - no double-peak pattern possible
        pattern_warnings.append("Only one peak detected - no double-peak pattern possible")
        if len(troughs) > 0:
            invalid_troughs.extend(troughs)
            pattern_warnings.append("All troughs are invalid due to single peak")
    
    elif len(peaks) == 0:
        # No peaks detected
        pattern_warnings.append("No peaks detected - cannot form double-peak pattern")
        if len(troughs) > 0:
            invalid_troughs.extend(troughs)
    
    # Create chart
    fig = go.Figure()
    
    # Main curve
    fig.add_trace(go.Scatter(
        x=df_stroke.index, 
        y=df_stroke[power_col],
        mode='lines', 
        name=f'{direction.capitalize()} Forward Power',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Peak markers
    if len(peaks) > 0:
        fig.add_trace(go.Scatter(
            x=df_stroke.index[peaks], 
            y=df_stroke[power_col].iloc[peaks],
            mode='markers', 
            marker=dict(color='red', size=12, symbol='x', line=dict(width=2)),
            name=f'Peaks ({len(peaks)})'
        ))
    
    # Valid trough markers (green circles)
    if len(valid_troughs) > 0:
        fig.add_trace(go.Scatter(
            x=df_stroke.index[valid_troughs], 
            y=df_stroke[power_col].iloc[valid_troughs],
            mode='markers', 
            marker=dict(color='green', size=12, symbol='circle', line=dict(width=2)),
            name=f'Valid Troughs ({len(valid_troughs)})'
        ))
    
    # Invalid trough markers (orange triangles)
    if len(invalid_troughs) > 0:
        fig.add_trace(go.Scatter(
            x=df_stroke.index[invalid_troughs], 
            y=df_stroke[power_col].iloc[invalid_troughs],
            mode='markers', 
            marker=dict(color='orange', size=10, symbol='triangle-down', line=dict(width=1)),
            name=f'Invalid Troughs ({len(invalid_troughs)})'
        ))
    
    # Threshold lines
    fig.add_hline(
        y=local_maxima * peak_threshold, 
        line=dict(color="red", dash="dash", width=2),
        annotation_text=f"Peak Threshold: {local_maxima * peak_threshold:.1f}W",
        annotation_position="top right"
    )
    
    fig.add_hline(
        y=lowest_peak * valley_threshold, 
        line=dict(color="green", dash="dash", width=2),
        annotation_text=f"Trough Threshold: {lowest_peak * valley_threshold:.1f}W",
        annotation_position="bottom right"
    )
    
    # Add pattern analysis to title
    title_suffix = ""
    if len(pattern_warnings) > 0:
        if len(valid_troughs) > 0:
            title_suffix = f" - Valid Double-Peak Pattern ({len(valid_troughs)} troughs)"
        else:
            title_suffix = " - No Valid Double-Peak Pattern"
    
    # Update layout
    fig.update_layout(
        title=f"{direction.capitalize()} Direction Stroke {stroke_number} - Power Curve Analysis{title_suffix}",
        xaxis_title="Time (ms)",
        yaxis_title="Power (W)",
        height=600,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1
        ),
        hovermode='x unified',
        template='plotly_white'
    )
    
    # Disable grid
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    
    return fig, pattern_warnings

def main():
    # Sidebar - File upload
    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Select JSON data file",
            type=['json'],
            help="Upload a JSON file containing rowing force data"
        )
        
        # Sample data download
        if st.button("📥 Download Sample Data"):
            sample_data = {
                "strokeForce": {
                    "left": {"time": [2780, 4310, 5270, 6370, 7450]},
                    "right": {"time": [2800, 3860, 4870, 5930, 7010]}
                },
                "lapForceTime": {
                    "time": list(range(2780, 8000, 10)),
                    "left": {"forward": [np.random.normal(50, 20) for _ in range(522)]},
                    "right": {"forward": [np.random.normal(50, 20) for _ in range(522)]}
                }
            }
            
            json_str = json.dumps(sample_data, indent=2)
            st.download_button(
                label="Download Sample JSON",
                data=json_str,
                file_name="sample_data.json",
                mime="application/json"
            )
    
    # Load and preprocess data
    df, error = load_and_process_data(uploaded_file)
    
    if error:
        st.error(error)
        return
    
    if df is not None:
        # Parameter control panel
        st.header("Double-Peak Pattern Detection")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # st.write("**Direction Selection**")
            direction = st.radio(
                "Choose hand:",
                ["left", "right"],
                format_func=lambda x: "Left Hand" if x == "left" else "Right Hand",
                horizontal=True
            )
        
        with col2:
            max_stroke = df[f'{direction}_stroke_time'].max()
            # st.write(f"**Stroke Selection (1-{int(max_stroke)})**")
            
            # Initialize stroke_number in session state if not exists
            if 'current_stroke' not in st.session_state:
                st.session_state.current_stroke = 1
            
            # Create a row for stroke navigation with buttons and slider
            stroke_col1, stroke_col2, stroke_col3 = st.columns([1, 3, 1])
            
            with stroke_col1:
                if st.session_state.current_stroke > 1:
                    if st.button("◀ Prev", key="prev_stroke"):
                        st.session_state.current_stroke = max(1, st.session_state.current_stroke - 1)
                        st.rerun()
                else:
                    st.button("◀ Prev", disabled=True, key="prev_stroke_disabled")
            
            with stroke_col2:
                stroke_number = st.slider(
                    "Select stroke number:",
                    min_value=1,
                    max_value=int(max_stroke),
                    value=st.session_state.current_stroke,
                    step=1,
                    help=f"Choose stroke number from 1 to {int(max_stroke)}",
                    key="stroke_slider",
                    on_change=lambda: setattr(st.session_state, 'current_stroke', st.session_state.stroke_slider)
                )
            
            with stroke_col3:
                if st.session_state.current_stroke < int(max_stroke):
                    if st.button("Next ▶", key="next_stroke"):
                        st.session_state.current_stroke = min(int(max_stroke), st.session_state.current_stroke + 1)
                        st.rerun()
                else:
                    st.button("Next ▶", disabled=True, key="next_stroke_disabled")
        
        with col3:
            st.write(" ")
        
        # Threshold sliders with real-time updates
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("")

            peak_threshold = st.slider(
                "Peak Threshold Ratio",
                min_value=0.1,
                max_value=1.0,
                value=st.session_state.peak_threshold,
                step=0.05,
                help="Peak needs to be higher than this threshold * max power",
                key="peak_slider",
                on_change=lambda: setattr(st.session_state, 'peak_threshold', st.session_state.peak_slider)
            )
        
        with col2:
            st.write("")


            valley_threshold = st.slider(
                "Trough Threshold Ratio",
                min_value=0.1,
                max_value=1.0,
                value=st.session_state.valley_threshold,
                step=0.05,
                help="Trough needs to be lower than this threshold * lowest peak",
                key="valley_slider",
                on_change=lambda: setattr(st.session_state, 'valley_threshold', st.session_state.valley_slider)
            )
        
        with col3:
            st.write("")

            prominence = st.slider(
                "Prominence Threshold",
                min_value=0.01,
                max_value=0.5,
                value=st.session_state.prominence,
                step=0.01,
                help="A higher prominence threshold will omit less significant troughs",
                key="prominence_slider",
                on_change=lambda: setattr(st.session_state, 'prominence', st.session_state.prominence_slider)
            )
        
        # # Create visualization with real-time updates
        # st.header("📊 Force Curve Analysis")
        
        # Create two columns: left for plot, right for text
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Create placeholder for the plot that updates in real-time
            plot_placeholder = st.empty()
            
            # Update the plot with current parameters
            fig, pattern_warnings = create_force_plot(
                df, 
                direction, 
                st.session_state.current_stroke, 
                st.session_state.peak_threshold, 
                st.session_state.valley_threshold, 
                st.session_state.prominence
            )
            plot_placeholder.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 Definitions and Additional Rules")
            st.markdown('- Peaks need to exceed the **peak threshold** * **max power**')
            st.markdown('- Troughs need to exceed the **trough threshold** * **power of the lowest peak**')
            
            st.markdown('This algorithm detects if **a trough that satisfies the trough threshold** is sandwiched by two peaks that satisfy the peak threshold') 
            st.markdown('**Valid Double-Peak Pattern**:')
            st.markdown('- There are at least two peaks in the stroke')
            st.markdown('- The trough is sandwiched between two peaks')
            st.markdown('**Not a Valid Double-Peak Pattern**:')
            st.markdown('- There is only one peak in the stroke')
            st.markdown('- The trough is not sandwiched by two peaks')

    else:
        st.info("Please upload a JSON data file to begin analysis")

if __name__ == "__main__":
    main() 