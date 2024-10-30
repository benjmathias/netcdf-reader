#!/usr/bin/env python3

import streamlit as st
import netCDF4 as nc
import os
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import tempfile

class NetCDFParser:
    """A class to handle NetCDF file parsing and visualization in Streamlit.
    
    This class provides methods to read, parse, and visualize NetCDF files through
    a Streamlit interface. It handles large datasets by implementing automatic
    downsampling and provides various visualization options.
    
    Attributes:
        MAX_PARSED_VARIABLESIZE_MB (float): Maximum size in MB for parsing variables
    """
    
    MAX_PARSED_VARIABLESIZE_MB = 1

    def __init__(self):
        """Initialize the NetCDFParser."""
        self.lat_names = ['lat', 'latitude', 'LATITUDE']
        self.lon_names = ['lon', 'longitude', 'LONGITUDE']

    @staticmethod
    def read_netcdf(file_path):
        """Read a NetCDF file and extract its metadata.
        
        Args:
            file_path (str): Path to the NetCDF file
            
        Returns:
            tuple: (file_info, data) containing file metadata and variable information
            
        Raises:
            FileNotFoundError: If the specified file doesn't exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        dataset = nc.Dataset(file_path, 'r')
        try:
            file_info = {
                'path': file_path,
                'size': f"{os.path.getsize(file_path) / (1024*1024):.2f} MB",
                'last_modified': datetime.fromtimestamp(os.path.getmtime(file_path)),
                'dimensions': list(dataset.dimensions.keys()),
                'dim_details': {dim_name: {'size': len(dim),
                                         'unlimited': dim.isunlimited()}
                               for dim_name, dim in dataset.dimensions.items()},
                'global_attributes': {attr: getattr(dataset, attr)
                                    for attr in dataset.ncattrs()}
            }

            data = {}
            for var_name, var in dataset.variables.items():
                data[var_name] = {
                    'attributes': {attr: getattr(var, attr) for attr in var.ncattrs()},
                    'shape': var.shape,
                    'dtype': var.dtype,
                    'dimensions': var.dimensions,
                    'variable': var
                }

            return file_info, data

        finally:
            dataset.close()

    @staticmethod
    def get_variable_data(var_obj, max_size_mb=MAX_PARSED_VARIABLESIZE_MB):
        """Safely load variable data with size checking and automatic downsampling.
        
        Args:
            var_obj: NetCDF variable object
            max_size_mb (float): Maximum size in MB to load
            
        Returns:
            np.ndarray: The loaded (potentially downsampled) data
        """
        size_mb = np.prod(var_obj.shape) * var_obj.dtype.itemsize / (1024 * 1024)
        
        if size_mb > max_size_mb:
            downsample_factor = max(1, int(np.ceil(np.sqrt(size_mb / (max_size_mb/4)))))
            
            if len(var_obj.shape) == 1:
                return var_obj[::downsample_factor]
            elif len(var_obj.shape) == 2:
                return var_obj[::downsample_factor, ::downsample_factor]
            else:
                return var_obj[0, ::downsample_factor, ::downsample_factor]
        else:
            return var_obj[:]

    def plot_variable(self, var_data, var_name, data):
        """Create appropriate plots for the selected variable.
        
        Args:
            var_data (dict): Variable metadata and data
            var_name (str): Name of the variable
            data (dict): Complete dataset information
            
        Returns:
            plotly.graph_objects.Figure: The generated plot
        """
        var_obj = var_data['variable']
        dims = var_data['dimensions']

        lat_dim = next((dim for dim in dims if dim.lower() in self.lat_names), None)
        lon_dim = next((dim for dim in dims if dim.lower() in self.lon_names), None)

        if lat_dim and lon_dim and len(var_obj.shape) == 2:
            return self._plot_geographic(var_obj, var_name, data, lat_dim, lon_dim)
        elif len(var_obj.shape) == 1:
            return self._plot_1d(var_obj, var_name)
        elif len(var_obj.shape) > 2:
            return self._plot_multidimensional(var_obj, var_name)
        else:
            return self._plot_2d_default(var_obj, var_name)

    def _plot_geographic(self, var_obj, var_name, data, lat_dim, lon_dim):
        """Create a geographic plot for 2D variables with lat/lon dimensions."""
        lats = self.get_variable_data(data[lat_dim]['variable'])
        lons = self.get_variable_data(data[lon_dim]['variable'])
        values = self.get_variable_data(var_obj)

        if values.shape[0] != len(lats) or values.shape[1] != len(lons):
            downsample_factor_lat = max(1, int(np.ceil(len(lats) / values.shape[0])))
            downsample_factor_lon = max(1, int(np.ceil(len(lons) / values.shape[1])))
            lats = lats[::downsample_factor_lat]
            lons = lons[::downsample_factor_lon]
            values = values[::downsample_factor_lat, ::downsample_factor_lon]

        df = pd.DataFrame({
            'Latitude': np.repeat(lats, len(lons)),
            'Longitude': np.tile(lons, len(lats)),
            'Value': values.flatten()
        })

        fig = px.scatter(
            df,
            x='Longitude',
            y='Latitude',
            color='Value',
            color_continuous_scale='Viridis',
            title=f"{var_name} Geographic Distribution",
            labels={'Value': var_name},
            width=800,
            height=600
        )

        fig.update_layout(
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            margin=dict(l=40, r=40, t=50, b=40)
        )
        return fig

    def _plot_1d(self, var_obj, var_name):
        """Create a line plot for 1D variables."""
        values = self.get_variable_data(var_obj)
        fig = px.line(y=values, title=f"{var_name} Values")
        fig.update_layout(
            yaxis_title=var_name,
            xaxis_title="Index"
        )
        return fig

    def _plot_multidimensional(self, var_obj, var_name):
        """Create a heatmap for multidimensional variables."""
        st.warning(f"Displaying the first slice of a {len(var_obj.shape)}D array.")
        first_slice = self.get_variable_data(var_obj)[0]
        fig = px.imshow(
            first_slice,
            title=f"{var_name} Heatmap (First Slice)",
            labels={'color': var_name},
            x=range(first_slice.shape[1]),
            y=range(first_slice.shape[0])
        )
        fig.update_layout(
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            margin=dict(l=40, r=40, t=50, b=40)
        )
        return fig

    def _plot_2d_default(self, var_obj, var_name):
        """Create a default plot for 2D variables without lat/lon dimensions."""
        values = self.get_variable_data(var_obj)
        x = np.arange(values.shape[1])
        y = np.arange(values.shape[0])
        df = pd.DataFrame(values, index=y, columns=x).reset_index().melt(
            id_vars='index', var_name='Longitude', value_name='Value'
        )

        fig = px.scatter(
            df,
            x='Longitude',
            y='index',
            color='Value',
            color_continuous_scale='Viridis',
            title=f"{var_name} Scatter Plot",
            labels={'index': 'Latitude', 'Value': var_name},
            width=800,
            height=600
        )

        fig.update_layout(
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            margin=dict(l=40, r=40, t=50, b=40)
        )
        return fig
    
    def run(self):
        """Run the Streamlit application."""
        st.set_page_config(page_title="NetCDF Reader", layout="wide")
        st.title("NetCDF File Reader")
        st.write("Upload and explore NetCDF files interactively")

        uploaded_file = st.file_uploader("Choose a NetCDF file", type=['nc'])
        
        if uploaded_file:
            self._handle_uploaded_file(uploaded_file)

    def _handle_uploaded_file(self, uploaded_file):
        """Process the uploaded NetCDF file.
        
        Args:
            uploaded_file: Streamlit's UploadedFile object
        """
        with tempfile.NamedTemporaryFile(suffix='.nc') as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_file.flush()
            
            with nc.Dataset(tmp_file.name, 'r') as dataset:
                try:
                    file_info = {
                        'size': f"{os.path.getsize(tmp_file.name) / (1024*1024):.2f} MB",
                        'last_modified': datetime.fromtimestamp(os.path.getmtime(tmp_file.name)),
                        'dimensions': list(dataset.dimensions.keys()),
                        'dim_details': {dim_name: {'size': len(dim),
                                                 'unlimited': dim.isunlimited()}
                                     for dim_name, dim in dataset.dimensions.items()},
                        'global_attributes': {attr: getattr(dataset, attr)
                                            for attr in dataset.ncattrs()}
                    }

                    data = {}
                    for var_name, var in dataset.variables.items():
                        data[var_name] = {
                            'attributes': {attr: getattr(var, attr) for attr in var.ncattrs()},
                            'shape': var.shape,
                            'dtype': var.dtype,
                            'dimensions': var.dimensions,
                            'variable': var
                        }

                    self._display_file_info(file_info, data)
                    self._display_variables(data)
                except Exception as e:
                    st.error(f"Error reading NetCDF file: {str(e)}")

    def _display_file_info(self, file_info, data):
        """Display file information in the Streamlit interface.
        
        Args:
            file_info (dict): File metadata
            data (dict): Variable data
        """
        st.header("File Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Size", file_info['size'])
        with col2:
            st.metric("Number of Variables", len(data))
        with col3:
            st.metric("Number of Dimensions", len(file_info['dimensions']))

        # Dimensions
        st.subheader("Dimensions")
        dim_df = pd.DataFrame([
            {'Dimension': dim,
             'Size': details['size'],
             'Unlimited': '✓' if details['unlimited'] else '✗'}
            for dim, details in file_info['dim_details'].items()
        ])
        st.dataframe(dim_df, use_container_width=True)

        # Global Attributes
        if file_info['global_attributes']:
            st.subheader("Global Attributes")
            attr_df = pd.DataFrame([
                {'Attribute': attr, 'Value': value}
                for attr, value in file_info['global_attributes'].items()
            ])
            st.dataframe(attr_df, use_container_width=True)

    def _display_variables(self, data):
        """Display variable information and visualizations.
        
        Args:
            data (dict): Variable data
        """
        st.header("Variables")
        selected_var = st.selectbox("Select a variable to explore:", list(data.keys()))

        if selected_var:
            var_data = data[selected_var]
            var_obj = var_data['variable']

            size_mb = np.prod(var_obj.shape) * var_obj.dtype.itemsize / (1024 * 1024)
            st.write(f"Variable size: {size_mb:.1f} MB")
            
            if size_mb > self.MAX_PARSED_VARIABLESIZE_MB:
                st.warning(f"Large variable detected. Data will be downsampled to stay under {self.MAX_PARSED_VARIABLESIZE_MB} MB.")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Variable Information")
                st.write(f"**Shape:** {var_data['shape']}")
                st.write(f"**Data Type:** {var_data['dtype']}")
                st.write(f"**Dimensions:** {var_data['dimensions']}")

            with col2:
                st.subheader("Variable Attributes")
                if var_data['attributes']:
                    for attr, value in var_data['attributes'].items():
                        st.write(f"**{attr}:** {value}")
                else:
                    st.write("No attributes found")

            st.subheader("Data Visualization")
            values = self.get_variable_data(var_obj)
            
            if np.issubdtype(values.dtype, np.number):
                stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                with stats_col1:
                    st.metric("Mean", f"{np.mean(values):.2e}")
                with stats_col2:
                    st.metric("Std Dev", f"{np.std(values):.2e}")
                with stats_col3:
                    st.metric("Min", f"{np.min(values):.2e}")
                with stats_col4:
                    st.metric("Max", f"{np.max(values):.2e}")

            try:
                fig = self.plot_variable(var_data, selected_var, data)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Could not create plot: {str(e)}")

            self._display_raw_data_preview(var_data, var_obj, data)

    def _display_raw_data_preview(self, var_data, var_obj, data):
        """Display a preview of the raw data.
        
        Args:
            var_data (dict): Variable metadata
            var_obj: NetCDF variable object
            data (dict): Complete dataset information
        """
        st.header("Raw Data Preview")
        st.markdown("Below is a preview of the raw data for the selected variable. Only a subset is shown to maintain performance.")
        preview = self.get_variable_data(var_obj, max_size_mb=0.5)

        dimensions = var_data['dimensions']
        lat_dim = next((dim for dim in dimensions if dim.lower() in self.lat_names), None)
        lon_dim = next((dim for dim in dimensions if dim.lower() in self.lon_names), None)

        if len(preview.shape) == 1:
            st.markdown("**First 100 Values:**")
            df_preview = pd.DataFrame(preview[:100], columns=['Value'])
            st.dataframe(df_preview)
        else:
            st.markdown("**First 50x50 Grid:**")
            df_preview = pd.DataFrame(preview[:50, :50])

            if lat_dim and lon_dim:
                lats = data[lat_dim]['variable'][:50]
                lons = data[lon_dim]['variable'][:50]
                
                df_preview.columns = [f"Longitude {lon}" for lon in lons]
                df_preview.index = [f"Latitude {lat}" for lat in lats]
            else:
                df_preview.columns = [f"Longitude {i}" for i in range(1, df_preview.shape[1] + 1)]
                df_preview.index = [f"Latitude {i}" for i in range(1, df_preview.shape[0] + 1)]

            st.dataframe(df_preview)

def main():
    """Entry point for the application."""
    parser = NetCDFParser()
    parser.run()

if __name__ == "__main__":
    main()
