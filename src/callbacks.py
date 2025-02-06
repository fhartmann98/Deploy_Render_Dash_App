from dash import callback,dash_table
from dash.dependencies import Input, Output, State
from dash import html

import dash
import anems_data_analysis as anems
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

from skrf import Network
from scipy.signal import find_peaks
import tempfile
import os
import io


def register_callbacks(app):
    # Define column name aliases
    COLUMN_ALIASES = {
        "frequency": ["frequency", "freq", "f"],
    }

    def resolve_column_name(df, target):
        """Find the best match for a given column alias in the DataFrame."""
        for alias in COLUMN_ALIASES.get(target, []):
            if alias in df.columns:
                return alias
        return None  # Return None if no match is found

    def parse_contents(contents, filename,size):
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            if 'csv' in filename:
                # Assume that the user uploaded a CSV file
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                # Auto-detect column names
                for key in COLUMN_ALIASES.keys():
                    col_name = resolve_column_name(df, key)
                    if col_name:
                        df.rename(columns={col_name: key}, inplace=True)

            elif 'xls' in filename:
                # Assume that the user uploaded an Excel file
                df = pd.read_excel(io.BytesIO(decoded))
            elif 's1p' in filename or 's2p' in filename:

                # Create a temporary file with the correct extension
                file_extension = os.path.splitext(filename)[1]  # Get the file extension from the uploaded filename
                if file_extension.lower() not in ['.s1p', '.s2p', '.ts']:
                    raise ValueError(f"Unsupported file extension: {file_extension}. Expected '.s1p', '.s2p', or '.ts'.")

                with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
                    temp_file.write(decoded)
                    temp_file_path = temp_file.name

                # Load the Touchstone file using skrf.Network
                network = Network(temp_file_path)

                if 's1p' in filename:
                    print("1-ports")
                    s_params11_real = network.s_re[:, 0, 0]  # S11 real part
                    s_params11_imag = network.s_im[:, 0, 0]  # S11 imaginary part
                    y_params11_real = network.y_re[:, 0, 0]  # S11 real part
                    y_params11_imag = network.y_im[:, 0, 0]  # S11 imaginary part
                    # Prepare DataFrame
                    df = pd.DataFrame({
                        'Frequency': network.f,
                        'S11_re': s_params11_real,
                        'S11_im': s_params11_imag,
                        'Y11_re': y_params11_real,
                        'Y11_im': y_params11_imag})

                elif 's2p' in filename :
                    print("2-ports")
                    # Extracting real and imaginary parts for S11, S12, S21, and S22
                    s_params11_real = network.s_re[:, 0, 0]  # S11 real part
                    s_params11_imag = network.s_im[:, 0, 0]  # S11 imaginary part

                    s_params12_real = network.s_re[:, 0, 1]  # S12 real part
                    s_params12_imag = network.s_im[:, 0, 1]  # S12 imaginary part

                    s_params21_real = network.s_re[:, 1, 0]  # S21 real part
                    s_params21_imag = network.s_im[:, 1, 0]  # S21 imaginary part

                    s_params22_real = network.s_re[:, 1, 1]  # S22 real part
                    s_params22_imag = network.s_im[:, 1, 1]  # S22 imaginary part

                    y_params11_real = network.y_re[:, 0, 0]  # S11 real part
                    y_params11_imag = network.y_im[:, 0, 0]  # S11 imaginary part

                    y_params12_real = network.y_re[:, 0, 1]  # S12 real part
                    y_params12_imag = network.y_im[:, 0, 1]  # S12 imaginary part

                    y_params21_real = network.s_re[:, 1, 0]  # S21 real part
                    y_params21_imag = network.s_im[:, 1, 0]  # S21 imaginary part

                    y_params22_real = network.y_re[:, 1, 1]  # S22 real part
                    y_params22_imag = network.y_im[:, 1, 1]  # S22 imaginary part

                    # Prepare DataFrame
                    df = pd.DataFrame({
                        'Frequency': network.f,
                        'S11_re': s_params11_real,
                        'S11_im': s_params11_imag,
                        'S12_re': s_params12_real,
                        'S12_im': s_params12_imag,
                        'S21_re': s_params21_real,
                        'S21_im': s_params21_imag,
                        'S22_re': s_params22_real,
                        'S22_im': s_params22_imag,
                        'Y11_re': y_params11_real,
                        'Y11_im': y_params11_imag,
                        'Y12_re': y_params12_real,
                        'Y12_im': y_params12_imag,
                        'Y21_re': y_params21_real,
                        'Y21_im': y_params21_imag,
                        'Y22_re': y_params22_real,
                        'Y22_im': y_params22_imag})
            else:
                return html.Div(['Unsupported file format.']), None

        except Exception as e:
            print(e)
            return html.Div(['There was an error processing this file.']), None

        return html.Div([
            html.H5(filename),
            html.H6("File uploaded successfully"),
            dash_table.DataTable(
                data=df.head().to_dict('records'),
                columns=[{'name': i, 'id': i} for i in df.columns],
                style_table={'width': '{}px'.format(size), 'overflowX': 'auto'}
            ),
            html.Hr(),  # horizontal line
        ]), df.to_dict('records')

    @app.callback(
        [Output('stored-data', 'data'),
         Output('output-data-upload', 'children')],
        [Input('upload-data', 'contents')],
        [State('upload-data', 'filename'),
         State('upload-data', 'last_modified')]
    )
    def update_output_store(contents, filename, date):
        if contents is not None:
            children, data = parse_contents(contents, filename,2200)

            return data, children
        return None, html.Div(['No file uploaded yet.'])


    @callback(
        Output('text-input', 'style'),
        Input('enable-input', 'value'))

    def toggle_input_if_param_sweep(checkbox_value):
        if 'enabled' in checkbox_value:
            return {'display': 'inline-block'}
        else:
            return {'display': 'none'}

    # Callback to display the graph when the button is clicked
    @app.callback(
        Output('admittance_graph', 'style'),
        [Input('show-graph-button', 'n_clicks'),
        Input('enable-input', 'value')]
    )
    def display_graph(n_clicks,checkbox_value):
        if n_clicks > 0:
            return {'width': '100%', 'display': 'inline-block'}
        return {'width': '100%','display': 'none'}  # Hide the graph


    @app.callback(
        [Output('admittance_graph', 'figure'),
        Output('param_message', 'children'),
         Output('dropdown-options', 'data')],
        [ Input('show-graph-button', 'n_clicks'),
            Input('stored-data', 'data'),
          Input('enable-input', 'value')],
          [State('text-input', 'value')]
    )
    def update_graph(n_clicks, data,checkbox_value,param):
        if n_clicks > 0:
            if data is None:
                return {},{},{}

            # Convert the stored data back to a DataFrame
            df = pd.DataFrame(data)

            if 'enabled' in checkbox_value and param !=None:
                y_abs=[]

                list_of_param = (pd.unique(df['{}'.format(param)]))
                # Split the input values by comma and strip any surrounding whitespace
                data_list = list_of_param.tolist()
                print("data is : {}".format(data_list))
                message = f'Values of {param} : {list_of_param}'

                freq = df['frequency'].loc[df['{}'.format(param)] == list_of_param[0]]
                for i in range(len(list_of_param)):
                    parameter = list_of_param[i]
                    y_abs.append(df['y_abs'].loc[df['{}'.format(param)] == parameter])

                fig = make_subplots(specs=[[{"secondary_y": False}]])
                for j in range(len(list_of_param)):
                    fig.add_trace(go.Scatter(x=freq / 10 ** 9, y=y_abs[j],
                                             mode='lines',
                                             name='Y11 Param_{}'.format(j)), secondary_y=False)
                fig.update_layout(xaxis_title='Frequency (GHz)')
                fig.update_yaxes(title_text="Admittance <b>[S]</b>", secondary_y=False)
                fig.update_layout(
                    font_family="Times New Roman",
                    font_color="black",
                    title_font_family="Times New Roman",
                    title_font_color="black",
                    legend_title_font_color="black",
                    paper_bgcolor='white',
                    plot_bgcolor="white",
                    height=1000
                )
                fig.update_xaxes(showgrid=True, linecolor='black')
                fig.update_yaxes(showgrid=True, linecolor='black', type="log", exponentformat="power", dtick="D1")
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

            elif 'enabled' in checkbox_value and param =='':
                message=f'Input value is empty! Please enter a value and submit'
                return {}, message,[]

            else:
                message=f'Single parameter'
                data_list = []
                f = df['frequency']
                y_abs = df['y_abs']
                fig = make_subplots(specs=[[{"secondary_y": False}]])
                fig.add_trace(go.Scatter(x=f/10**9, y=y_abs,
                                         mode='lines',
                                         name='Y11 Param_{}'), secondary_y=False)

                fig.update_layout(xaxis_title='Frequency (GHz)')
                fig.update_yaxes(title_text="Admittance <b>[S]</b>", secondary_y=False)
                fig.update_layout(
                    font_family="Times New Roman",
                    font_color="black",
                    title_font_family="Times New Roman",
                    title_font_color="black",
                    legend_title_font_color="black",
                    paper_bgcolor='white',
                    plot_bgcolor="white",
                    height=1000
                )
                fig.update_xaxes(showgrid=True, linecolor='black')
                fig.update_yaxes(showgrid=True, linecolor='black', type="log", exponentformat="power", dtick="D2")
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

            return fig,message,data_list
        else:
            return {},{},{}


    @app.callback(
        [Output('fit-dropdown', 'options'),
        Output('fit-dropdown', 'style'),
         Output('text-dropdown', 'style')],
        Input('show-graph-button', 'n_clicks'),
        Input('dropdown-options', 'data'),
    Input('enable-input', 'value')
    )
    def update_dropdown_options(n_clicks, input_values,checkbox):
        if n_clicks > 0 and 'enabled' in checkbox and len(input_values)>1:
            # Split the input values by comma and strip any surrounding whitespace
            options = input_values

            return options,{ 'display': 'block'},{'font-weight': 'bold', 'display': 'block'}
        return [],{ 'display': 'none'},{ 'display': 'none'}

    #If crop is selected it will show a frequency range
    @callback(
        Output('crop-range', 'className'),
        Input('crop_input', 'value'))
    def toggle_input_if_param_sweep(checkbox_value):
        if 'enabled' in checkbox_value:
            return 'visible'
        else:
            return 'hidden'


    @app.callback(
         [Output('plot-image', 'src'),
                Output('refresh-div', 'children'),
          Output('output-perf-fit', 'children')],
        [ Input('show-graph-button', 'n_clicks'),
            Input('stored-data', 'data'),
            Input('refresh-button', 'n_clicks'),
          Input('fit-dropdown', 'value'),
          Input('minimizer', 'value'),
          Input('k2', 'value'),
          Input('frequency-range', 'value'),
          Input('crop_input', 'value'),
          Input('crop-range', 'value'),
          Input('freq_sweep_type', 'value')],
          [State('prominence_value', 'value'),
           State('text-input', 'value'),
           State('graph-title', 'value'),
           State('x-coord', 'value'),
           State('y-coord', 'value')]
    )
    def update_graph_fit(n_clicks, data,refresh,dropdown_value,minimizer,k2,selected_range,crop,crop_range,freq_sweep_type,value_prom,text_param,graph_title,x_coord,y_coord):
        if n_clicks > 0 or refresh>0 :
            if data is None:
                return dash.no_update,{},{}

            # Convert the stored data back to a DataFrame
            df_init = pd.DataFrame(data)

            if 'enabled' in crop and crop_range!=selected_range:
                df= df_init[(df_init['frequency'] >= crop_range[0]*1e9) & (df_init['frequency'] <= crop_range[1]*1e9)]
                df=df.reset_index(drop=True)
            else :
                df=df_init
            enable_graph=0

            if dropdown_value == None:

                enable_graph=1
                freq = df['frequency']

                y_vals = []

                #Case where data is stored with y_abs, y_re and y_im
                if freq_sweep_type == 'Y parameters':
                    y_current_re = []
                    y_current_im = []

                    y_current_re.append(df['y_re'])
                    y_current_im.append(df['y_im'])

                    for j in range(len(freq)):
                        y_vals.append(complex(y_current_re[0][j], y_current_im[0][j]))

                # Case with data stored as "s11"

                #elif freq_sweep_type == 'S parameters':

                message = f'Graph updated with prominence value: {value_prom}'

            elif dropdown_value != None:
                enable_graph=1
                freq = df['frequency'].loc[df['{}'.format(text_param)] == dropdown_value].reset_index(drop=True)

                if freq_sweep_type == 'Y parameters':
                    y_vals = []
                    y_current_re = []
                    y_current_im = []

                    parameter = dropdown_value

                    y_current_re.append(df['y_re'].loc[df['{}'.format(text_param)] == parameter].reset_index(drop=True))
                    print(y_current_re)
                    y_current_im.append(df['y_im'].loc[df['{}'.format(text_param)] == parameter].reset_index(drop=True))
                    print("y_current_re 0:", y_current_re[0][0])


                    if len(freq) == len(y_current_re[0]) == len(y_current_im[0]):
                        for j in range(len(freq)):
                            print("j is :{}".format(j))
                            y_vals.append(complex(y_current_re[0][j], y_current_im[0][j]))

                message = f'Graph updated with prominence value: {value_prom} and param value {dropdown_value}'

            if enable_graph:
            # Now let's detect the resonance peaks. We are setting some restrictions on peak prominence and peak height to only get the most significant peaks
                prominence = value_prom * np.abs(y_vals)
                peaks, props = find_peaks(np.abs(y_vals), prominence=prominence, height=6e-5)

                # Define the equivalent circuit model
                # For now, let's assume there is no series resistance
                mbvd_model = anems.mBVDAdmittanceModel(static_branch_type='series', r_series=False)
                # Let's constrain some of the parameters
                # The mBVD model for resonators typically assumes these two relations to hold, so we force them in the fit
                constraints = {'L_m': '1 / (C_m*(2*pi*f_r)**2)',
                               'R_m': '1 / y_real'}
                # Make an initial guess for all the parameters
                guess_params = mbvd_model.guess(freq, y_vals, peaks=peaks, peak_props=props, constraints=constraints,
                                                verbose=True, c0_guess_region='all')

                mbvd_model.fit(freq, y_vals, minimize='{}'.format(minimizer))
                # Evaluate the fitted model at the frequencies of the measurement
                y_fit = mbvd_model.eval(freq)

                fit_df = mbvd_model.fitted_params_dataframe()
                # Extract the main mode using the magnitude of C_m as a criterion
                main_mode = anems.utils.extract_main_mode(fit_df, crit='C_m')
                print("The main mode is labelled {}".format(main_mode))

                fig, ax = anems.utils.plot_fitting(freq, y_vals, data_fit=y_fit, plot_re=True, peaks=peaks, scale_db=False,
                                                   y_label='Admittance [S]',
                                                   title=f"{graph_title}")
                ax.legend(loc=3)
                print(fit_df)
                # Add the inset
                # The textcoords argument specifies the relative location of the upper left corner
                # Check the docstring for additional arguments
                if x_coord != None and y_coord != None:
                    x_val= x_coord
                    y_val =y_coord


                anems.utils.insert_mode_inset(ax, data=fit_df, kt2_def='{}'.format(k2),mode=main_mode, textcoords=(x_val,y_val), fontsize=11)
                ax.set_xlim([selected_range[0]*1e9, selected_range[1]*1e9])
                # ax.set_ylim([10e-6,10e0])
                perf_fit_df = anems.utils.convert_to_performance_params(fit_df, only_main=False)


                # Convert Matplotlib figure to PNG image
                buffer = BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                plot_data = buffer.getvalue()

                # Encode PNG image to base64 string
                plot_base64 = base64.b64encode(plot_data).decode('utf-8')


                # Return the extracted data as the figure property of dcc.Graph
                return f'data:image/png;base64,{plot_base64}',message,html.Div([html.H6("Performance Fit"),dash_table.DataTable(data=perf_fit_df.head().to_dict('records'),
                columns=[{'name': i, 'id': i} for i in perf_fit_df.columns],style_table={'width': '1000px', 'overflowX': 'auto'}),html.Hr(),])
        else :
            return dash.no_update,dash.no_update,dash.no_update

    @app.callback(
        [Output('stored-meas', 'data'),
            Output('stored-short', 'data'),
            Output('stored-open', 'data'),
         Output('output-meas-upload', 'children'),
         Output('output-short-upload', 'children'),
         Output('output-open-upload', 'children')],
        [Input('upload-meas', 'contents'),
         Input('upload-short', 'contents'),
         Input('upload-open', 'contents')],
        [State('upload-meas', 'filename'),
        State('upload-short', 'filename'),
        State('upload-open', 'filename')]
    )
    def update_output_measurement(contents_meas,contents_short,contents_open,filename_meas, filename_short ,filename_open):

        if contents_meas is not None:
            children_meas, data_meas = parse_contents(contents_meas, filename_meas,600)
        else :
            children_meas=html.Div(['No file uploaded yet.'])
            data_meas = None

        if contents_short is not None:
            children_short, data_short = parse_contents(contents_short, filename_short,600)
        else:
            children_short = html.Div(['No file uploaded yet.'])
            data_short = None

        if contents_open is not None:
            children_open, data_open = parse_contents(contents_open, filename_open,600)
        else:
            children_open = html.Div(['No file uploaded yet.'])
            data_open = None


        return data_meas, data_short, data_open, children_meas,children_short,children_open

    # If crop is selected it will show a frequency range
    @callback(
        Output('crop-range-meas', 'className'),
        Input('crop_input_meas', 'value'))
    def toggle_input_if_param_sweep(checkbox_value):
        if 'enabled' in checkbox_value:
            return 'visible'
        else:
            return 'hidden'

    @app.callback(
        [Output('measurement_graph', 'figure') ,
        Output('short_graph', 'figure') ,
        Output('open_graph', 'figure')],
            [Input('stored-meas', 'data'),
            Input('stored-short', 'data'),
             Input('stored-open', 'data'),
            Input('meas_ys', 'value'),
             Input('short_ys', 'value'),
             Input('open_ys', 'value')])
    def update_graph(data_meas,data_short,data_open,meas_ys,short_ys,open_ys):

        if data_meas is None:
            fig_meas={}
        else:
            # Convert the stored data back to a DataFrame
            df_meas = pd.DataFrame(data_meas)
            nb_ports = 1
            if "S12" in df_meas.columns:
                nb_ports = 2

            f = df_meas['Frequency']

            if nb_ports == 1:
                if meas_ys == "Y-param":
                    y_abs = np.sqrt(df_meas['Y11_re'] ** 2 + df_meas['Y11_im'] ** 2)

                elif meas_ys == "S-param":
                    y_abs = np.sqrt(df_meas['S11_re'] ** 2 + df_meas['S11_im'] ** 2)
            elif nb_ports == 2:
                if meas_ys == "Y-param":
                    y_abs = -np.sqrt(df_meas['Y12_re'] ** 2 + df_meas['Y12_im'] ** 2)

                elif meas_ys == "S-param":
                    y_abs = -np.sqrt(df_meas['S12_re'] ** 2 + df_meas['S12_im'] ** 2)


            fig_meas = make_subplots(specs=[[{"secondary_y": False}]])
            fig_meas.add_trace(go.Scatter(x=f/10**9, y=y_abs,
                                     mode='lines',
                                     name='{}'.format(meas_ys)), secondary_y=False)

            fig_meas.update_layout(xaxis_title='Frequency (GHz)')
            if meas_ys == "Y-param":
                fig_meas.update_yaxes(title_text="Y11 <b>[S]</b>", secondary_y=False)
            elif meas_ys == "S-param":
                fig_meas.update_yaxes(title_text="S11 <b>[S]</b>", secondary_y=False)
            fig_meas.update_layout(
                font_family="Times New Roman",
                font_color="black",
                title_font_family="Times New Roman",
                title_font_color="black",
                legend_title_font_color="black",
                paper_bgcolor='white',
                plot_bgcolor="white",
                height=400
            )
            fig_meas.update_xaxes(showgrid=True, linecolor='black')
            fig_meas.update_yaxes(showgrid=True, linecolor='black',type="log")
            fig_meas.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
            fig_meas.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

        if data_short is None:
            fig_short={}
        else:
            # Convert the stored data back to a DataFrame
            df_short = pd.DataFrame(data_short)
            nb_ports=1
            if "S12" in df_short.columns:
                nb_ports = 2

            f = df_short['Frequency']

            if nb_ports == 1:
                if short_ys == "Y-param":
                    y_abs = np.sqrt(df_short['Y11_re']**2 + df_short['Y11_im']**2)

                elif short_ys == "S-param":
                    y_abs = np.sqrt(df_short['S11_re']**2 + df_short['S11_im']**2)
            elif nb_ports == 2:
                if short_ys == "Y-param":
                    y_abs = -np.sqrt(df_short['Y12_re']**2 + df_short['Y12_im']**2)

                elif short_ys == "S-param":
                    y_abs = -np.sqrt(df_short['S12_re']**2 + df_short['S12_im']**2)

            fig_short = make_subplots(specs=[[{"secondary_y": False}]])
            fig_short.add_trace(go.Scatter(x=f/10**9, y=y_abs,
                                     mode='lines',
                                     name='{}'.format(short_ys)), secondary_y=False)

            fig_short.update_layout(xaxis_title='Frequency (GHz)')
            if short_ys == "Y-param":
                fig_short.update_yaxes(title_text="Y11 <b>[S]</b>", secondary_y=False)
            elif short_ys == "S-param":
                fig_short.update_yaxes(title_text="S11 <b>[S]</b>", secondary_y=False)
            fig_short.update_layout(
                font_family="Times New Roman",
                font_color="black",
                title_font_family="Times New Roman",
                title_font_color="black",
                legend_title_font_color="black",
                paper_bgcolor='white',
                plot_bgcolor="white",
                height=400
            )
            fig_short.update_xaxes(showgrid=True, linecolor='black')
            fig_short.update_yaxes(showgrid=True, linecolor='black',type="log")
            fig_short.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
            fig_short.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

        if data_open is None:
            fig_open={}
        else:
            # Convert the stored data back to a DataFrame
            df_open = pd.DataFrame(data_open)

            nb_ports = 1
            if "S12_re" in df_open.columns:
                nb_ports = 2

            f = df_open['Frequency']

            if nb_ports == 1:
                if short_ys == "Y-param":
                    y_abs = np.sqrt(df_open['Y11_re'] ** 2 + df_open['Y11_im'] ** 2)

                elif short_ys == "S-param":
                    y_abs = np.sqrt(df_open['S11_re'] ** 2 + df_open['S11_im'] ** 2)
            elif nb_ports == 2:
                if short_ys == "Y-param":
                    y_abs = -np.sqrt(df_open['Y12_re'] ** 2 + df_open['Y12_im'] ** 2)

                elif short_ys == "S-param":
                    y_abs = -np.sqrt(df_open['S12_re'] ** 2 + df_open['S12_im'] ** 2)

            fig_open = make_subplots(specs=[[{"secondary_y": False}]])
            fig_open.add_trace(go.Scatter(x=f/10**9, y=y_abs,
                                     mode='lines',
                                     name='{}'.format(open_ys)), secondary_y=False)

            fig_open.update_layout(xaxis_title='Frequency (GHz)')
            if open_ys =="Y-param":
                fig_open.update_yaxes(title_text="Y11 <b>[S]</b>", secondary_y=False)
            elif open_ys =="S-param":
                fig_open.update_yaxes(title_text="S11 <b>[S]</b>", secondary_y=False)

            fig_open.update_layout(
                font_family="Times New Roman",
                font_color="black",
                title_font_family="Times New Roman",
                title_font_color="black",
                legend_title_font_color="black",
                paper_bgcolor='white',
                plot_bgcolor="white",
                height=400
            )
            fig_open.update_xaxes(showgrid=True, linecolor='black')
            fig_open.update_yaxes(showgrid=True, linecolor='black',type="log")
            fig_open.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
            fig_open.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

        return fig_meas,fig_short,fig_open

    @app.callback(
        [Output('plot-image-meas', 'src'),
         Output('refresh-div-meas', 'children'),
         Output('output-perf-fit-meas', 'children')],
        [Input('show-meas-button', 'n_clicks'),
         Input('stored-meas', 'data'),
         Input('refresh-button-meas', 'n_clicks'),
         Input('minimizer-meas', 'value'),
         Input('k2-meas', 'value'),
         Input('frequency-range-meas', 'value'),
         Input('crop_input_meas', 'value'),
         Input('crop-range-meas', 'value')],
        [State('prominence_value_meas', 'value'),
         State('graph-title-meas', 'value'),
         State('x-coord-meas', 'value'),
         State('y-coord-meas', 'value')]
    )
    def update_fitting_measurement(n_clicks, data, refresh, minimizer, k2, selected_range, crop, crop_range,
                          value_prom, graph_title, x_coord, y_coord):
        if n_clicks > 0 or refresh > 0:
            if data is None:
                return dash.no_update, {}, {}

            # Convert the stored data back to a DataFrame
            df_init = pd.DataFrame(data)

            if 'enabled' in crop and crop_range != selected_range:
                print("cropped")
                df = df_init[
                    (df_init['Frequency'] >= crop_range[0] * 1e9) & (df_init['Frequency'] <= crop_range[1] * 1e9)]
                df = df.reset_index(drop=True)
            else:
                df = df_init

            freq = df['Frequency']

            nb_ports=1
            y_meas=[]

            if "S12_re" in df.columns:
                nb_ports = 2

            if nb_ports == 1:
                for j in range(len(freq)):
                    y_meas.append(complex(df['Y11_re'][j], df['Y11_im'][j]))
            elif nb_ports==2 :
                for j in range(len(freq)):
                    y_meas.append(-complex(df['Y12_re'][j], df['Y12_im'][j]))


            # Now let's detect the resonance peaks. We are setting some restrictions on peak prominence and peak height to only get the most significant peaks
            prominence = value_prom * np.abs(y_meas)
            assert len(np.abs(y_meas)) == len(prominence), "Lengths do not match!"
            print("y_vals length:", len(np.abs(y_meas)))
            print("prominence length:", len(prominence))
            print(value_prom)

            if len(np.abs(y_meas)) == len(prominence) :
                peaks, props = find_peaks(np.abs(y_meas), prominence=prominence, height=1e-5)

            # Define the equivalent circuit model
            # For now, let's assume there is no series resistance
            mbvd_model = anems.mBVDAdmittanceModel(static_branch_type='series', r_series=False)
            # Let's constrain some of the parameters
            # The mBVD model for resonators typically assumes these two relations to hold, so we force them in the fit
            constraints = {'L_m': '1 / (C_m*(2*pi*f_r)**2)',
                           'R_m': '1 / y_real'}
            # Make an initial guess for all the parameters
            guess_params = mbvd_model.guess(freq, y_meas, peaks=peaks, peak_props=props, constraints=constraints,
                                            verbose=True, c0_guess_region='all')

            mbvd_model.fit(freq, y_meas, minimize='{}'.format(minimizer))
            # Evaluate the fitted model at the frequencies of the measurement
            y_fit = mbvd_model.eval(freq)

            fit_df = mbvd_model.fitted_params_dataframe()
            # Extract the main mode using the magnitude of C_m as a criterion
            main_mode = anems.utils.extract_main_mode(fit_df, crit='C_m')
            print("The main mode is labelled {}".format(main_mode))

            fig, ax = anems.utils.plot_fitting(freq, y_meas, data_fit=y_fit, plot_re=True, peaks=peaks,
                                               scale_db=False,
                                               y_label='Admittance [S]',
                                               title=f"{graph_title}")
            ax.legend(loc=3)
            print(fit_df)
            # Add the inset
            # The textcoords argument specifies the relative location of the upper left corner
            # Check the docstring for additional arguments
            if x_coord != None and y_coord != None:
                x_val = x_coord
                y_val = y_coord

            anems.utils.insert_mode_inset(ax, data=fit_df, kt2_def='{}'.format(k2), mode=main_mode,
                                          textcoords=(x_val, y_val), fontsize=11)
            ax.set_xlim([selected_range[0] * 1e9, selected_range[1] * 1e9])
            # ax.set_ylim([10e-6,10e0])
            perf_fit_df = anems.utils.convert_to_performance_params(fit_df, only_main=False)

            # Convert Matplotlib figure to PNG image
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            plot_data = buffer.getvalue()
            message = f'Graph updated with prominence value: {value_prom}'
            # Encode PNG image to base64 string
            plot_base64 = base64.b64encode(plot_data).decode('utf-8')

            # Return the extracted data as the figure property of dcc.Graph
            return f'data:image/png;base64,{plot_base64}', message, html.Div(
                [html.H6("Performance Fit"), dash_table.DataTable(data=perf_fit_df.head().to_dict('records'),
                                                                  columns=[{'name': i, 'id': i} for i in
                                                                           perf_fit_df.columns],
                                                                  style_table={'width': '1000px',
                                                                               'overflowX': 'auto'}), html.Hr(), ])
        else:
            return dash.no_update, dash.no_update, dash.no_update