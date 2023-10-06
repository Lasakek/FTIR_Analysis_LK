import streamlit as st
from bs4 import BeautifulSoup
from scipy.signal import savgol_filter, find_peaks
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
from collections import Counter
from scipy.optimize import minimize, least_squares, Bounds
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from scipy.special import wofz
from PIL import Image
from io import BytesIO
import xlsxwriter
from xlsxwriter import Workbook

#--------------------------------Defaults------------------------------#
# Default data


#---------Class for storing/handling Sample Information-----#
class FTIR:
   def __init__(self, sample_name):
      self.sample_name = sample_name
      self.data = {}  # You can use a dictionary to store different types of data

   def add_file_name(self, file_name):
      self.file_name = file_name

   def give_file_name(self):
      return self.file_name

   def add_data(self, key, value):
      # Store data in the dictionary
      self.data[key] = value

   def get_data(self, key):
      # Retrieve data from the dictionary
      return self.data[key]


#---------------Baselinge Correction---------------------#
def iterative_average(spectrum, threshold=0.001):
    smoothed = savgol_filter(spectrum, window_length=5, polyorder=1)
    diff = np.abs(spectrum - smoothed).sum()

    while diff > threshold * len(spectrum):
        spectrum -= smoothed
        smoothed = savgol_filter(spectrum, window_length=5, polyorder=1)
        diff = np.abs(spectrum - smoothed).sum()

    return smoothed

#-------------Read in Data from .csv file---------------#
def csv_to_data(uploaded_file):
    if uploaded_file is not None:
        file_names = []
        sample_names = []
        sample_objects = []
        file_idx = 0
        for file in uploaded_file:
            df = pd.read_csv(file)

            try:

                # Extract 'x' values (excluding the first row)
                x_values = df.iloc[0:, 0]
                # st.write("this is x", x_values)

                # Extract 'y' values (excluding the first row)
                y_values = df.iloc[0:, 1]
                # print(y_values)
            except Exception as e:
                st.warning("Please ensure the correct structure of the .csv file: \n x-values, y-values\n x1, y1\n x2, y2")

            sample_name = f"Sample_{file_idx}"

            sample = FTIR(sample_name)
            sample.add_data('X', x_values)
            # st.write(x_values)
            # print("This is Y_mean", np.mean(y_values))
            sample.add_data('Y', y_values)
            # st.write(y_values)
            sample.add_file_name(file.name)

            if file.name in file_names and not error:
                error = st.error(f"Please don't upload duplicate files", icon="🚨")
            else:
                sample_objects.append(sample)
                file_names.append(file.name)
                sample_names.append(sample_name)
            file_idx = file_idx + 1

    return sample_names, file_names, sample_objects #returns a np.array of sample_name, each sample_name is a FTIR object and contains x and y data


#----------Reading in Data from .xml files---------------#
def xml_to_data(uploaded_file):
    if uploaded_file is not None:
        file_idx = 0
        file_names = []
        sample_names = []
        sample_objects = []
        error = None
        selected_samples = []
        for file in uploaded_file:

            content = file.read()

            soup = BeautifulSoup(content, 'html5lib')

            values_tag = soup.find('values')
            if values_tag:
                y_values_text = values_tag.get_text()
                y_values = [float(value) for value in y_values_text.strip().split()]
            else:
                st.error(f"Could not find Y-values in {file}", icon="🚨")

            # y_values = savgol_filter(y_values, window_length=15, polyorder=4)
            y_values = iterative_average((y_values))

            fxv_tag = soup.find('parameter', {'name': 'FXV'})
            lxv_tag = soup.find('parameter', {'name': 'LXV'})

            if fxv_tag and lxv_tag:
                max_x_value = float(lxv_tag.text)
                min_x_value = float(fxv_tag.text)
            else:
                st.error(f"Could not find X-values in {file}", icon="🚨")

            num_y_values = len(y_values)
            x_values = [min_x_value + i * ((max_x_value - min_x_value) / (num_y_values - 1)) for i in
                        range(num_y_values)]

            sample_name = f"Sample_{file_idx}"

            sample = FTIR(sample_name)
            sample.add_data('X', x_values)
            # st.write(x_values)
            # print("This is Y_mean", np.mean(y_values))
            sample.add_data('Y', y_values)
            # st.write(y_values)
            sample.add_file_name(file.name)

            if file.name in file_names and not error:
                error = st.error(f"Please don't upload duplicate files", icon="🚨")
            else:
                sample_objects.append(sample)
                file_names.append(file.name)
                sample_names.append(sample_name)
            file_idx = file_idx + 1

    return sample_names, file_names, sample_objects #returns a np.array of sample_name, each sample_name is a FTIR object and contains x and y data


#----------Creating Raw_Data Dataframe------------------#
def short_norm_df(sample_objects, selected_samples, x_min, x_max, normalization):
    sample_data = {}
    for sample in sample_objects:
        if sample.sample_name in selected_samples:
            sample_data[sample.sample_name] = sample.get_data('Y')

    # Create the DataFrame with 'x' as the first column
    data = pd.DataFrame({'x': sample_objects[0].get_data('X')})

    # Add 'y' columns for selected samples
    for sample_name, y_values in sample_data.items():
        data[sample_name] = y_values



    # Filter data to be within the specified x-range
    data = data[(data['x'] >= x_min) & (data['x'] <= x_max)]

    if normalization:
        for sample_name in selected_samples:
            data[sample_name] = minmax_scale(data[sample_name])

    # Reset the index to start from 0 to the end
    data.reset_index(drop=True, inplace=True)

    return data



#----------Creating PCA Dataframe-----------------------#
#--still has an issue when not all samples are selected
def pca_df(data):

    del data['x']
    data = data.T

    # Create a PCA model with the number of components you desire
    n_components = 2  # You can change this to the number of components you want
    pca = PCA(n_components=n_components)
    # st.write(len(data))


    # Fit the PCA model to your 'y' values
    pca_result = pca.fit_transform(data)

    # Create a DataFrame for the PCA results
    pca_df = pd.DataFrame(data=pca_result, columns=[f"PC{i + 1}" for i in range(n_components)])

    return pca_df
#---Problem with amount of samples = 1
def PCA_Plot(data, selected_samples):
    if len(data.keys()) <= 3:
        st.warning("Please select at least 3 Samples to Perform PCA", icon="🚨")

    else:
        # Create a Dataframe, containing the PC1/PC2
        PCA = pca_df(data)

        # Create a scatter plot using Plotly Express
        fig = px.scatter(
            PCA,
            x='PC1',
            y='PC2',
            title='PCA Results',
            labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
            color=selected_samples  # Use sample names as colors

        )

        # Customize the plot as needed (e.g., add labels, adjust plot settings)
        fig.update_traces(marker=dict(size=15, opacity=0.8))
        # Add vertical gridlines
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='gray')

        fig.update_layout(width=600, height=600,
                          title_font=dict(size=20),
                          xaxis=dict(title_font=dict(size=25)),
                          yaxis=dict(title_font=dict(size=25)),
                          legend_font=dict(size=20))

        # Show the plot
        st.plotly_chart(fig)

#----------Creating the bar Plot for Peak Locations-------#

def bar_plot(data):
    peak_index_all = []

    # st.write(data['x'])
    for column in data.columns[1:]:
        y = data[column]
        x = data['x']

        # Calculate the second derivative using numpy's gradient function
        # first_derivate = np.gradient(y,x)
        # second_derivative = -np.gradient(first_derivate, x)
        second_derivative = -savgol_filter(y, window_length=15, polyorder=4, deriv=2)

        # print(d2y_dx2)
        peak_index, _ = find_peaks(second_derivative, prominence=0.0001)

        # print(peak_index)
        peak_index_all.append(peak_index)
    # st.write(peak_index_all)

    # Flatten the list of arrays into a single list
    flattened_list = [item for sublist in peak_index_all for item in sublist]

    # Count the occurrences of each peak index
    peak_counts = Counter(flattened_list)
    # print(peak_counts)

    # Step 3: Create a bar plot
    counts = list(peak_counts.values())
    x_values = [data['x'][peak_idx] for peak_idx in peak_counts.keys()]

    # Create a bar chart with thinner bars
    plt.figure(figsize=(8, 4))  # Adjust the width and height as needed
    plt.bar(x_values, counts)
    # Customize plot appearance
    plt.xlabel("Wavenumber [cm$^-$$^1$]")
    plt.ylabel("Number of Peaks")
    plt.title("Number of Peak Occurences in Inverted Second Derivatives")
    plt.xticks(rotation=80, fontsize=8)  # Adjust x-axis ticks
    plt.tight_layout()

    # Show the plot
    st.pyplot(plt)
    # Round x_values to 1 digit after the decimal point
    x_values = [round(x, 1) for x in x_values]
    # Create a DataFrame
    peak_df = pd.DataFrame({
        "Number of Occurrences:": counts,
        "Corresponding Wavenumber:": x_values
    })

    # Sort the DataFrame by 'Number of Occurrences' in descending order
    peak_df = peak_df.sort_values(by='Number of Occurrences:', ascending=False)

    # Transpose the DataFrame
    peak_df = peak_df.T

    # Display the DataFrame
    st.write(peak_df)

#----------Creating all the 2nd derivate Plots-----------#
# is fucked up right now
def second_der_plots(data, show_plots):
    # Define the number of columns for the layout
    num_columns = 4
    # print(data)
    if show_plots:
        st.subheader("Sample Plots")
        # Calculate the second derivative of each sample
        second_derivative_data = data.copy()
        sample_columns = data.columns[1:]
        x_values = data['x']

        for sample_col in sample_columns:
            y_values = data[sample_col]
            # first_derivate = np.gradient(y_values,x_values)
            # second_derivative = np.gradient(first_derivate, x_values)
            second_derivative = savgol_filter(y_values, window_length=15, polyorder=4, deriv=2)

            second_derivative_data[sample_col] = -second_derivative

        # Get the list of sample columns for the second derivative data
        second_derivative_sample_columns = second_derivative_data.columns[1:]

        # Define the number of plots per row
        plots_per_row = 4

        # Calculate the number of rows needed
        num_rows = (len(second_derivative_sample_columns) - 1) // plots_per_row + 1

        # Create a Streamlit app
        # st.title('Sample Line Charts with Second Derivative')

        # Define a fixed width for each plot
        plot_width = 400  # You can adjust this width as needed

        # Loop through the rows and create a grid of line charts for the second derivative data
        for i in range(num_rows):
            row_data = second_derivative_data[
                ['x'] + list(second_derivative_sample_columns[i * plots_per_row:(i + 1) * plots_per_row])]

            # Create a row with beta_columns
            cols = st.columns(plots_per_row)

            # Create line charts for each sample in this row
            for j, sample_col in enumerate(row_data.columns[1:]):
                with cols[j % plots_per_row]:
                    fig = px.line(row_data, x='x', y=sample_col, title=f'Second Derivative of {sample_col}')

                    # Set the width of the plot
                    fig.update_layout(width=plot_width, xaxis_title="Wavenumber [cm^-1]", yaxis_title="2nd Derivate Intensity")

                    # Adjust the margins to prevent x-axis cutoff
                    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))  # You can adjust these values as needed

                    st.plotly_chart(fig)




#---------Peak Deconvolution Functions------------------#

#------LEV-------------------#
def V_lev(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
def composite_function_lev(x, params):
    num_peaks = len(params) // 3
    total = np.zeros_like(x)
    for i in range(num_peaks):
        mu = params[i]
        A = params[num_peaks + i]
        sigma = params[num_peaks * 2 + i]
        total += V_lev(x, A, mu, sigma)
    return total



def peak_fit_lev(data, initial_guess, selected_samples):

    # drop empty rows in initial guesses
    initial_guess = initial_guess.dropna()

    # getting fitting parameters from the initial guesses from the input
    mu = initial_guess['mu']
    A = initial_guess['A']
    sigma = initial_guess['sigma']

    initial_params_lev = np.concatenate((mu, A, sigma))


    # Combine all bounds into a single list
    upper_bounds = [center + 10 for center in mu] + [np.inf] * len(mu) + [np.inf] * len(mu)
    lower_bounds = [center - 10 for center in mu] + [0] * len(mu) + [0] * len(mu)
    bounds = Bounds(lower_bounds, upper_bounds)


    parameter_result = {}

    progress_bar = st.progress(0, text="Fitting the Peaks of each Sample. Please wait...")

    for idx, sample in enumerate(selected_samples):
        y_data = data[sample]
        x_data = data['x']

        progress = (idx+1) / len(selected_samples)
        progress_bar.progress(progress, text=f"Fitting the Peaks of each Sample. Please wait...   Sample   {idx+1} of {len(selected_samples)} Samples in calculation")

        def objective_lev(params):
            # st.write(1)
            y_fitted = composite_function_lev(x_data, params)
            # st.write(2)
            fit_quality = y_fitted - y_data
            RMSE = (np.sqrt(np.mean((y_fitted - y_data) ** 2)) / max(y_data) * 100)
            # print(RMSE)
            return np.array(fit_quality)



        initial_params_lev = np.array(initial_params_lev)
        # st.write(type(initial_params_lev[0]))

        result = least_squares(objective_lev, initial_params_lev, bounds=bounds, method="trf", gtol=1e-5, xtol=1e-5, ftol=1e-5)

        # Extract optimized parameters
        optimized_params = result.x
        parameter_result[sample] = optimized_params

    progress_bar.empty()
    return parameter_result


def plot_fitted_spec_lev(x_data, y_raw, params, initial_guess, sample):
    # drop empty rows
    initial_guess = initial_guess.dropna()
    total = np.zeros_like(x_data)
    num_peaks = len(initial_guess['peak'])
    peak_funcs = {'X': x_data}
    key_list = []
    sample_color = []

    for i, peak in enumerate(initial_guess['peak']):
        mu = params[i]
        A = params[num_peaks + i]
        sigma = params[num_peaks * 2 + i]
        F_peak = V_lev(x_data, A, mu, sigma)
        total += F_peak
        peak_funcs[peak] = F_peak
        key_list.append(peak)


    key_list.append('Fitted Spectrum')
    key_list.append('Original Data')
    peak_funcs['Fitted Spectrum'] = total
    peak_funcs['Original Data'] = y_raw


    RMSE = round((np.sqrt(np.mean((total - y_raw)**2))/max(y_raw)*100),2)

    title = "Fitted Spectrum " + str(sample)
    fig = px.line(peak_funcs, x='X', y=key_list, title=title)
    fig.add_annotation(text=f'RMSE = {RMSE} %', x=0.9, y=0.9, xref='paper', yref='paper', showarrow=False)

    # Update the figure's layout to set the width and height
    fig.update_layout(xaxis_title="Wavenumber [cm^-1]", yaxis_title="Intensity",
                      title_font=dict(size=20), width= 500)
    st.plotly_chart(fig)

def plot_peak_areas_lev(x_data, y_raw, params, initial_guess, sample):
    # drop empty rows in initial_guess
    initial_guess = initial_guess.dropna()
    total = np.zeros_like(x_data)
    num_peaks = len(initial_guess['peak'])
    peak_funcs = {}


    for i, peak in enumerate(initial_guess['peak']):
        mu = params[i]
        A = params[num_peaks + i]
        sigma = params[num_peaks * 2 + i]
        F_peak = V_lev(x_data, A, mu, sigma)
        total += F_peak
        peak_funcs[peak] = F_peak




    # Calculate the area under each peak
    # Calculate the total area under the fitted composite spectrum
    total_area = -np.trapz(total, x_data)
    # print("This is total_area: ", total_area)
    peak_percentages = []
    labels = []
    for peak, peak_values in peak_funcs.items():
        area = -np.trapz(peak_values, x_data)
        # print("This is ", peak, " Area", area)
        # Calculate the percentage of each peak's area relative to the total area
        peak_percentages.append(round(area / total_area * 100,2))
        # print("This is Peak_percentages", peak_percentages)
        labels.append(peak)



    title = "Peak Percentages " + str(sample)

    fig = px.bar(peak_percentages, labels, peak_percentages, title=title, color=labels)

    # Update the figure's layout to set the width and height
    fig.update_layout(title=title, xaxis_title="Wavenumber [cm^-1]", yaxis_title="Area Percentage",
                      title_font=dict(size=20), width= 500)

    st.plotly_chart(fig)

    return peak_percentages




#-------------NELDER-MEAD--------------------------#
# Define the Voigt function
def V(x, center, amplitude, alpha, gamma):
    sigma = alpha / np.sqrt(2 * np.log(2))
    return np.real(wofz((x - center + 1j * gamma) / (sigma * np.sqrt(2)))) / sigma / np.sqrt(2 * np.pi) * amplitude



# Define the composite function
def composite_function(x, params):
    num_peaks = len(params) // 4
    total = np.zeros_like(x)
    for i in range(num_peaks):
        center = params[i]
        amplitude = params[num_peaks + i]
        alpha = params[num_peaks * 2 + i]
        gamma = params[num_peaks * 3 + i]
        total += V(x, center, amplitude, alpha, gamma)
    return total

def peak_fit(data, initial_guess, selected_samples):

    #drop empty rows in initial guesses
    initial_guess = initial_guess.dropna()
    # print("This is init:", initial_guess)


    # getting fitting parameters from the initial guesses from the input
    centers = initial_guess['center']
    amplitudes = initial_guess['amplitude']
    alphas = initial_guess['alpha']
    gammas = initial_guess['gamma']
    initial_params = np.concatenate((centers, amplitudes, alphas, gammas))

    # Define bounds for parameter optimization
    center_bounds = [(center - 10, center + 10) for center in centers]
    amplitude_bounds = (0.01, None)  # Allow positive amplitudes only
    alpha_bounds = (0.01, None)  # Allow positive alpha values only
    gamma_bounds = (0.01, None)  # Allow positive gamma values only
    bounds = center_bounds + [amplitude_bounds] * 4 + [alpha_bounds] * 4 + [gamma_bounds] * 4

    # Optimization settings
    max_iterations = 100000000
    convergence_tolerance = 1e-10


    parameter_result = {}

    progress_bar = st.progress(0, text="Fitting the Peaks of each Sample. Please wait...")

    for idx, sample in enumerate(selected_samples):
        y_data = data[sample]
        # print("This is Y_mean", np.mean(y_data))
        x_data = data['x']
        # st.write("We are at this sample: ", sample)
        # Define the objective function (sum of squared differences)

        progress = (idx+1)/len(selected_samples)
        progress_bar.progress(progress, text=f"Fitting the Peaks of each Sample. Please wait...   Sample   {idx+1} of {len(selected_samples)} Samples in calculation")


        def objective(params):
            y_fitted = composite_function(x_data, params)
            # y_fitted_2 = savgol_filter(y_fitted,window_length=15, polyorder=4, deriv=2)
            fit_quality = np.sqrt(np.mean((y_fitted - y_data) ** 2))
            # fit_quality_2 = np.sqrt(np.mean((np.array(minmax_scale(y_fitted_2)) - np.array(minmax_scale(Y_2)))**2))
            return fit_quality

        # # Perform optimization
        result = minimize(objective, initial_params, bounds=bounds, method="L-BFGS-B",
                          options={'maxiter': max_iterations, 'gtol': convergence_tolerance})





        # Extract optimized parameters
        optimized_params = result.x
        parameter_result[sample] = optimized_params
    progress_bar.empty()
    return parameter_result

def plot_fitted_spec(x_data, y_raw, params, initial_guess, sample):
    # drop empty rows
    initial_guess = initial_guess.dropna()
    total = np.zeros_like(x_data)
    num_peaks = len(initial_guess['peak'])
    peak_funcs = {'X': x_data}
    key_list = []
    sample_color = []

    for i, peak in enumerate(initial_guess['peak']):
        center = params[i]
        amplitude = params[num_peaks + i]
        alpha = params[num_peaks * 2 + i]
        gamma = params[num_peaks * 3 + i]
        F_peak = V(x_data, center, amplitude, alpha, gamma)
        total += F_peak
        peak_funcs[peak] = F_peak
        key_list.append(peak)


    key_list.append('Fitted Spectrum')
    key_list.append('Original Data')
    peak_funcs['Fitted Spectrum'] = total
    peak_funcs['Original Data'] = y_raw


    RMSE = round((np.sqrt(np.mean((total - y_raw)**2))/max(y_raw)*100),2)

    title = "Fitted Spectrum " + str(sample)
    fig = px.line(peak_funcs, x='X', y=key_list, title=title)
    fig.add_annotation(text=f'RMSE = {RMSE} %', x=0.9, y=0.9, xref='paper', yref='paper', showarrow=False)

    # Update the figure's layout to set the width and height
    fig.update_layout(xaxis_title="Wavenumber [cm^-1]", yaxis_title="Intensity",
                      title_font=dict(size=20), width= 500)
    st.plotly_chart(fig)




def plot_peak_areas(x_data, y_raw, params, initial_guess, sample):
    # drop empty rows in initial_guess
    initial_guess = initial_guess.dropna()
    total = np.zeros_like(x_data)
    num_peaks = len(initial_guess['peak'])
    peak_funcs = {}


    for i, peak in enumerate(initial_guess['peak']):
        center = params[i]
        amplitude = params[num_peaks + i]
        alpha = params[num_peaks * 2 + i]
        gamma = params[num_peaks * 3 + i]
        F_peak = V(x_data, center, amplitude, alpha, gamma)
        total += F_peak
        peak_funcs[peak] = F_peak




    # Calculate the area under each peak
    # Calculate the total area under the fitted composite spectrum
    total_area = -np.trapz(total, x_data)
    # print("This is total_area: ", total_area)
    peak_percentages = []
    labels = []
    for peak, peak_values in peak_funcs.items():
        area = -np.trapz(peak_values, x_data)
        # print("This is ", peak, " Area", area)
        # Calculate the percentage of each peak's area relative to the total area
        peak_percentages.append(round(area / total_area * 100,2))
        # print("This is Peak_percentages", peak_percentages)
        labels.append(peak)



    title = "Peak Percentages " + str(sample)

    fig = px.bar(peak_percentages, labels, peak_percentages, title=title, color=labels)

    # Update the figure's layout to set the width and height
    fig.update_layout(title=title, xaxis_title="Wavenumber [cm^-1]", yaxis_title="Area Percentage",
                      title_font=dict(size=20), width= 500)

    st.plotly_chart(fig)

    return peak_percentages


def residual_err(x, y, params, sample, algorithm):
    if algorithm:
        y_fit = composite_function(x,params)
    else:
        y_fit = composite_function_lev(x, params)

    residual = y-y_fit
    res = pd.DataFrame()
    res['x'] = x
    res['residuals'] = residual
    title = "Residual Error Behaviour " + str(sample)
    fig = px.scatter(res, x='x', y='residuals', title=title)
    fig.update_layout(xaxis_title="Wavenumber [cm^-1]", yaxis_title="Residual", width= 500)
    st.plotly_chart(fig)



def plot_heatmap(parameters):
    state_df = st.session_state['heatmap_df']

    param_add = st.text_input('Additional Parameters - Parameters are seperated by commas (,)',
                              'Param1, Param2, Param3')
    empty_columns = param_add.split(', ')
    heatmap_df = pd.concat([state_df, pd.DataFrame(columns=empty_columns)], axis=1)
    st.markdown("*Enter the specific Parameter Values for each Sample. A column must be filled out completely to be plotted.*")
    heatmap_df_edit = st.data_editor(heatmap_df, num_rows='dynamic', disabled=parameters)
    dropped = heatmap_df_edit.dropna(axis=1, how='any')
    # st.write(dropped)


    miami = dropped.drop('Sample', axis=1)
    for_real = round(miami.corr(),2)
    # print(for_real)

    fig = px.imshow(for_real, text_auto=True, aspect="auto")
    fig.update_layout(width=1000, height=1000, title="Correlation Heatmap",
                      title_font=dict(size=20),
                      xaxis=dict(title_font=dict(size=50), tickfont=dict(size=20)),
                      yaxis=dict(title_font=dict(size=50), tickfont=dict(size=20)),
                      legend_font=dict(size=40),
                      coloraxis_colorbar=dict(
                          title_font=dict(size=20),
                          tickfont=dict(size=20)))

    st.plotly_chart(fig)
    return for_real


def main():
    st.set_page_config(layout="wide")
    # Define content for the Analysis Page (you can create another tab for analysis)

    #-----------------all of the Sidebar Input--------------------------#
    st.sidebar.header("Upload your .xml Files here:")



    csv_toggle = st.sidebar.toggle("Turn on to upload .csv files.")

    if not csv_toggle:
        uploaded_file = st.sidebar.file_uploader("Choose a file", type=['xml'], help='You are only able to upload .xml files.', accept_multiple_files=True)
        sample_names, file_names, sample_objects = xml_to_data(uploaded_file)
        data_table = pd.DataFrame({"Sample Names": sample_names, "File Names": file_names})
    else:
        uploaded_file = st.sidebar.file_uploader("You are only able to upload .csv files with x-values in first columns and y-values in the second column.", type=['csv'],
                                                 accept_multiple_files=True)
        sample_names, file_names, sample_objects = csv_to_data(uploaded_file)
        data_table = pd.DataFrame({"Sample Names": sample_names, "File Names": file_names})
       
    st.sidebar.markdown("**Your Data:**")
    st.sidebar.dataframe(data_table, use_container_width=True, height=250, hide_index=True)



    #---Chose a Selection of samples---#
    st.sidebar.divider()
    st.sidebar.markdown("**Chose the to be Plotted Samples**")
    all = st.sidebar.checkbox("Select all")

    if all:
      select_samples = st.sidebar.multiselect("Select one or more samples:",
                                               sample_names, sample_names)
      selected_samples = select_samples
    else:
      select_samples = st.sidebar.multiselect("Select one or more samples:",
                                               sample_names)
      selected_samples = select_samples

    st.sidebar.divider()

    #---Input of x_Range---#
    st.sidebar.markdown("**Select the Wavenumber Range [cm$^-$ $^1$]**")
    cols = st.sidebar.columns(2)
    with cols[0]:
      x_min = st.number_input('Insert a X_min', min_value=899, max_value=4000, value=1600)
    with cols[1]:
      x_max = st.number_input('Insert a X_max', min_value=899, max_value=4000, value=1700)
    if x_min > x_max:
      st.sidebar.write('''Please have X_min smaller than X_max :hugging_face:''')

    #---Input Normalization---#
    st.sidebar.divider()
    st.sidebar.markdown("**Normalization of Raw Data**")
    normalization = st.sidebar.toggle(":last_quarter_moon: Should really be turned on for further Analysis")

    #initializing Heatmap dataframe
    heatmap_df = []

    #----------Plotting in different Tabs--------------------#
    tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["\u2001 \u2001\u2001 Mainpage \u2001 \u2001 \u2001 ", "Plot Raw Data", "\u2001PCA\u2001", "Peak Identification", "Peak Deconvolution", "Correlation Heatmap", "Downloading Results"])

    with tab0:
        # Set the title and brief overview of the topic
        st.title("FTIR Data Analysis for Protein Secondary Structure")
        st.write("\n\n\n")  # Adds three lines of vertical space

        # Brief Overview
        st.markdown("**This Streamlit app is designed to analyze FTIR (Fourier-Transform Infrared Spectroscopy)"
                    " data to extract information about the secondary structure of proteins."
                    " This App is made for scientific purposes.**")
        st.write("\n\n\n\n\n\n")  # Adds three lines of vertical space

        # Display the flowchart using Streamlit
        # Display an image from a file
       
        # st.image("X:\FB4\BIO_VT\06_FG_Spadiut\4_Personal_folders\Interns and Students\LKE\Pictures\StreamLitApps\FTIR\Flowchart.drawio.png", caption="Your Image", use_column_width=True)

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            # Preprocessing Section
            st.header("What is happening? And how to use it.")

            image = Image.open(
                'Flowchart.drawio.png')

            st.image(image, )

        with col2:
            st.header("Data Preprocessing")

            # Explanation of data extraction step
            st.subheader("**Extracting the data from the given .xml files**")
            st.markdown(
                "**Each .xml file is dedicated to one Sample. The intensity values, can directly be read out. The correlating wavenumber is calculated between 900 and 4000 cm$^-$ $^1$.**")
            st.write("\n")
            st.subheader("**Baseline Correction**")

            st.markdown(
                "**The Baseline Correction is a critical Preprocessing Step and is necessary to enhance data quality, remove systematic noise, and improve the accuracy of subsequent analyses.**")
            st.markdown(
                "**In this App, the Baseline Correction is an iterative averaging approach published by XXX. Iterative averaging - Iteratively smooths intensities to minimize peaks, subtracts smoothed spectrum as baseline. Repeats until baseline converges.**")

            st.write("\n")

            st.subheader("Normalization")
            st.markdown("**The Normalization of the data takes place in the set Wavenumber-Range.**")

            st.write("\n\n\n\n\n\n")

            st.divider()

            # Data Analysis Section
            st.header("Data Analysis")
            st.subheader("Principal Component Analysis (PCA)")

            st.markdown(
                '**Principal Component Analysis (PCA) is a statistical technique used for dimensionality reduction and data analysis. It simplifies complex data by transforming it into a new coordinate system, where the axes (principal components) are orthogonal and ranked by their importance in capturing data variance. PCA helps identify patterns, reduce noise, and visualize data in a more concise manner.**')
            st.write("\n")

            st.subheader("Peak Identification")
            st.markdown(
                "**An identification of the most prominent peaks location over all the samples can be done using the inverted second derivatives of the spectral data. These are used as initial guesses for the peak locations, which is used by the fitting algorithm.**")
            st.write("\n")

            st.subheader("Peak Fitting")
            st.markdown("**Peak fitting, also known as deconvolution, is a data analysis technique used to separate and quantify individual peaks or components within a complex dataset. This technique is particularly useful when multiple overlapping peaks are present in the data, making it challenging to extract meaningful information.\n The first step in this process is to provide reasonable choice of the amount of peaks, their location and shape. These are used to construct a initial comosite function of all the peaks. The objective is to minimize the difference between the original data and the predicted function. This is achieved, by iteratively adjusting the used peak parameters. Different minimization problems can be defined and various algorithmsfor the parameter optimization, be used. Also can different peak shapes be applied. These options can result in fairly different outcomes when there is a high degree of overlapping present. The residuals between original and fitted data can give some kind of direction of how good a fit is. A high randomnes of these residuals indicates a good fit, a high grade of order (function-like/wavy shapes) indicates a bad fit.**")

            st.write("\n")

            # Display Heatmap
            st.subheader("Correlation Heatmap")
            st.markdown(
                "**In the final step the results of the Peak Fitting will be plotted in a correlation Heatmap. This provides valuable insight into how secondary structure are influenced by any other given parameters.**")

    with tab1:
        st.header("Here you can visually observe your Raw Data in one Plot")
        st.divider()
        if not select_samples:
            st.warning("Please upload files and select at least one Sample.")
        else:
            #getting the dataframe for plotting
            data = short_norm_df(sample_objects, selected_samples, x_min, x_max, normalization)



            # Plot the selected samples using Plotly Express
            fig = px.line(data, x='x', y=selected_samples,
                          title='Y-values of Selected Samples (Normalized)' if normalization else 'Y-values of Selected Samples')
            # Update the figure's layout to set the width and height
            fig.update_layout(width=1200, height=600, xaxis_title="Wavenumber [cm^-1]", yaxis_title="Intensity",
                              title_font=dict(size=20),
                              xaxis=dict(title_font=dict(size=25)),
                              yaxis=dict(title_font=dict(size=25)),
                              legend_font=dict(size=20))
            st.plotly_chart(fig)
            # Create a button for downloading the DataFrame as an Excel file
            # Option to download using Streamlit's built-in function
            csv = data.to_csv().encode('utf-8')

            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='large_df.csv',
                mime='text/csv',
            )
            st.write(data)



    with tab2:
        st.header("Principal Component Analysis (PCA)")
        # st.markdown(
        #     '''*Principal Component Analysis (PCA) is a statistical technique used for dimensionality reduction and data analysis. It simplifies complex data by transforming it into a new coordinate system, where the axes (principal components) are orthogonal and ranked by their importance in capturing data variance. PCA helps identify patterns, reduce noise, and visualize data in a more concise manner.*''')
        st.divider()
        st.markdown("**Try switching on/off Normalization.**	:chart_with_downwards_trend: 	:chart_with_upwards_trend:")

        if not select_samples:
            st.warning("Please upload files and select at least one Sample.")

        else:

            PCA_Plot(data, selected_samples)


    with tab3:
        st.header("Peak Identification")
        # st.markdown(
        #     '''*Peak deconvolution is a process of decomposing overlapping peaks to extract information about the hidden peak. Peak deconvolution can provide more accurate and detailed information about overlapping peaks, allowing for better analysis and interpretation of the data.*''')
        st.divider()
        if not select_samples:
            st.warning("Please upload files and select at least one Sample.")
        else:
            data = short_norm_df(sample_objects, selected_samples, x_min, x_max, normalization)
            # st.write(data)
            if not normalization:
                st.warning("Please Turn on Normalization for this :angel:")
            else:

                bar_plot(data)

                st.divider()
                st.subheader("Here you can look at the Second Derivate Plots")

                # Create a toggle to show/hide the plots
                show_plots = st.checkbox("Show Second Derivate Plots")

                second_der_plots(data, show_plots)




    with tab4:

        st.header("Peak Deconvolution")
        st.markdown('Fitting Peaks to a strongly overlapping composite function can be achieved by different approaches. Two different algorithms and minimization function are being used here. The first method uses an objective function to minimize the least squares between predicted and original data by a Trust-Region-Reflective (trf) algorithm. The objective function of the second approach is to minimize the RMSE between original and predicted data, based on the L-BFGS-B algorithm. Try out both.')

        st.divider()
        st.subheader('You can adjust the Initial Parameter Guesses and Number of Peaks. :hatching_chick:')
        st.write("\n\n\n\n")
        cola1, cola2 = st.columns(2)

        with cola1:
            st.markdown('**Least Squares Fitting Parameters**')
            default_data_lev = {
                'peak': ['β-Sheet', 'α-Helix', 'β-Turn', 'β-Sheet_2'],
                'mu': [1623, 1652, 1670, 1683],
                'A': [0.6, 0.2, 0.1, 0.1],
                'sigma': [10, 6, 4, 4]
            }
            df_lev = pd.DataFrame(default_data_lev)

            initial_guess_lev = st.data_editor(df_lev, num_rows='dynamic', hide_index=True)
            parameters_lev = ["Sample"] + initial_guess_lev['peak'].tolist()

        with cola2:
            st.markdown('**RMSE Fitting Parameters**')
            default_data = {
                'peak': ['β-Sheet', 'α-Helix', 'β-Turn', 'β-Sheet_2'],
                'center': [1623, 1652, 1670, 1683],
                'amplitude': [20, 8, 6, 6],
                'alpha': [6, 4, 4, 4],
                'gamma': [6, 4, 4, 4]
            }
            df = pd.DataFrame(default_data)

            initial_guess = st.data_editor(df, num_rows='dynamic', hide_index=True)
            parameters = ["Sample"] + initial_guess['peak'].tolist()




        if not select_samples:
            st.warning("Please upload files and select at least one Sample.")
        else:
            if not normalization:

                st.warning("Please turn on Normalization for this Step.")

            else:

                algorithm = st.toggle("Turn Off for Least Squares Fit | Turn on for RMSE Fit")
                start_decon = st.button("Press to Start Deconvolution")
                if start_decon:

                    if algorithm:
                        # defining the columns of the heatmap_df for future use
                        heatmap_df = pd.DataFrame(columns=parameters).dropna(axis=1, how='any')
                        optimized_parameters = peak_fit(data, initial_guess, selected_samples)

                        # st.write(optimized_parameters)

                        col1, col2, col3 = st.columns(3)

                        for sample in selected_samples:

                            with col1:

                                sample_color = plot_fitted_spec(data['x'], data[sample], optimized_parameters[sample], initial_guess, sample)

                            with col2:
                                peak_percentage = plot_peak_areas(data['x'], data[sample], optimized_parameters[sample], initial_guess, sample)
                                heatmap_row = [sample] + peak_percentage
                                heatmap_df.loc[len(heatmap_df)] = heatmap_row
                                if 'heatmap_df' not in st.session_state:
                                    st.session_state['heatmap_df'] = heatmap_df
                                else:
                                    st.session_state['heatmap_df'] = heatmap_df

                            with col3:
                                residual_err(data['x'], data[sample], optimized_parameters[sample], sample, algorithm)


                        # heat_bool = True
                        if 'heat_bool' not in st.session_state:
                            st.session_state['heat_bool'] = True
                        else:
                            st.session_state['heat_bool'] = True
                    else:
                        # defining the columns of the heatmap_df for future use
                        heatmap_df = pd.DataFrame(columns=parameters_lev).dropna(axis=1, how='any')
                        optimized_parameters_lev = peak_fit_lev(data, initial_guess_lev, selected_samples)


                        col1, col2, col3 = st.columns(3)

                        for sample in selected_samples:

                            with col1:

                                sample_color = plot_fitted_spec_lev(data['x'], data[sample], optimized_parameters_lev[sample],
                                                                initial_guess_lev, sample)

                            with col2:
                                peak_percentage = plot_peak_areas_lev(data['x'], data[sample], optimized_parameters_lev[sample],
                                                                  initial_guess_lev, sample)
                                heatmap_row = [sample] + peak_percentage
                                heatmap_df.loc[len(heatmap_df)] = heatmap_row
                                if 'heatmap_df' not in st.session_state:
                                    st.session_state['heatmap_df'] = heatmap_df
                                else:
                                    st.session_state['heatmap_df'] = heatmap_df

                            with col3:

                                residual_err(data['x'], data[sample], optimized_parameters_lev[sample], sample, algorithm)


                        # heat_bool = True
                        if 'heat_bool' not in st.session_state:
                            st.session_state['heat_bool'] = True
                        else:
                            st.session_state['heat_bool'] = True


    with tab5:


        st.title('Correlation Heatmap  	:hot_face: :cold_face:')


        if 'heat_bool' not in st.session_state:
                st.session_state['heat_bool'] = False
                st.warning("The Heatmap is based on the Peak Deconvolution Values, so please do this first. :fox_face:")

        else:
            if st.session_state['heat_bool'] == True:
                # Add 10 empty columns using a list comprehension
                if 'heatmap_df' in st.session_state:
                    start_the_heat = st.button("Calculate and Display Heatmap")
                    if start_the_heat:
                        corr_matrix = plot_heatmap(parameters)
                        st.session_state['corr_matrix'] = corr_matrix


            else:
                st.warning("The Heatmap is based on the Peak Deconvolution Values, so please do this first. :fox_face:")

    with tab6:
        st.title("Here you can download your Results :open_hands:")
        st.divider()

        download_button = st.button("Download Results as Excel File")
        if download_button:
            if st.session_state['corr_matrix'] is not None:
                corr = st.session_state['corr_matrix']
            else:
                corr = []

            if st.session_state['heatmap_df'] is not None:
                peak_percentage = st.session_state['heatmap_df']
            else:
                peak_percentage = []

            if data is not None:
                raw_data = data
            else:
                raw_data = []


            new_column_names = {'x':'Wavenumber [cm$^-$$^1$]'}
            new_row_names = []

            for sample in selected_samples:
                index = int(sample[7:])
                sample_object = sample_objects[index]
                file_name = sample_object.give_file_name()
                # item = {sample: file_name}
                # st.write(item)
                new_column_names[sample] = file_name

                new_row_names.append(file_name)
            raw_data.rename(columns=new_column_names, inplace=True)
            # st.write(peak_percentage)
            # st.write(type(peak_pe rcentage))
            peak_percentage[0] = new_row_names
            st.write(raw_data, peak_percentage, corr)
            output = BytesIO()

            # Write files to in-memory strings using BytesIO and xlsxwriter
            # See: https://xlsxwriter.readthedocs.io/workbook.html?highlight=BytesIO#constructor
            with xlsxwriter.Workbook(output, {'in_memory': True}) as workbook:
                with pd.ExcelWriter(workbook, engine='xlsxwriter') as writer:
                    raw_data.to_excel(writer, sheet_name='Raw_Data', index=False)
                    peak_percentage.to_excel(writer, sheet_name='Peak_Percentages', index=False)
                    corr.to_excel(writer, sheet_name='Correlation_Matrix', index=False)

            # Reset the position to the beginning of the stream
            output.seek(0)

            # Create the download button
            st.download_button(
                label="Please confirm",
                data=output.getvalue(),
                file_name="Results.xlsx",
                mime="application/vnd.ms-excel",
                key="download_button"
            )



if __name__ == "__main__":
    main()
