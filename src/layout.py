from dash import html, dcc

layout = html.Div(style={'margin': '20px'}, children=[
    # Main title
    html.H1("Fitting Web Interface", style={'textAlign': 'center'}),
    dcc.Tabs([
        dcc.Tab(label='Simulations', children=[
            html.Div([
                # Introduction
                html.Hr(),
                html.Div('This interface can be used to fit simulations, using the fitting code'),
                html.Hr(),

                # File upload
                dcc.Upload(
                    id='upload-data',
                    children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
                    style={
                        'backgroundColor': '#CFE2F3',  # Button background color
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px'
                    },
                    multiple=False
                ),
                dcc.Store(id='stored-data'),  # Component to store the DataFrame
                html.Div(id='output-data-upload'),

                html.Hr(),
                # Adaptive or normal frequency sweep
                html.Div(
                    'Select the way the data is stored :'),
                dcc.RadioItems(
                            id='freq_sweep_type',
                            options=[{'label': option, 'value': option} for option in ['Y parameters', 'S parameters']],
                            value='Y parameters',
                            inline=True,
                            labelStyle={'display': 'inline-block', 'margin-right': '20px'},
                            style={'marginBottom': '20px'}
                        ),
                # Parametric Sweep Section
                html.Div(
                    'If you performed a parametric sweep, please tick the box and enter the variable name and click on submit'),
                html.Br(),
                dcc.Checklist(
                    id='enable-input',
                    options=[{'label': 'Parametric Sweep', 'value': 'enabled'}],
                    value=[]
                ),
                dcc.Input(
                    id='text-input',
                    type='text',
                    placeholder='Enter param name',
                    style={'display': 'none', 'marginTop': '10px'}
                ),
                html.Div(id='param_message', style={'marginTop': '10px'}),

                html.Br(),
                html.Button(
                    id='show-graph-button',
                    children='SHOW GRAPH',
                    n_clicks=0,
                    style={'marginTop': '20px'}
                ),
                html.Br(),
                # Graphs and Controls
                html.Div([
                    html.Div([
                        html.H3('Admittance Graphs'),
                        dcc.Graph(id='admittance_graph')
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                    html.Div([
                        html.H3('Fit Graph'),
                        html.Div('Prominence value:', style={'fontWeight': 'bold'}),
                        dcc.Input(
                            id='prominence_value',
                            type='number',
                            value=0.8,
                            placeholder="Prominence",
                            style={'width': '100px', 'marginBottom': '10px'}
                        ),
                        html.Button('Refresh Graphs', id='refresh-button', n_clicks=0, style={'marginLeft': '10px'}),
                        dcc.Store(id='dropdown-options'),
                        html.Div('Cases to fit:', id='text-dropdown', style={'fontWeight': 'bold', 'display': 'none'}),
                        dcc.Dropdown(id='fit-dropdown', placeholder='Case to fit:', options=[], value=None,
                                     style={'display': 'none', 'marginBottom': '10px'}),

                        html.Div('Minimizer choice:', style={'fontWeight': 'bold'}),
                        dcc.RadioItems(
                            id='minimizer',
                            options=[{'label': option, 'value': option} for option in
                                     ['reim', 'abs', 'abs_db', 'reim_db']],
                            value='abs_db',
                            inline=True,
                            labelStyle={'display': 'inline-block', 'margin-right': '20px'},
                            style={'marginBottom': '20px'}
                        ),

                        html.Div('Coupling Definition:', style={'fontWeight': 'bold'}),
                        dcc.RadioItems(
                            id='k2',
                            options=[{'label': option, 'value': option} for option in ['k2_eff', 'rar', 'fbw']],
                            value='k2_eff',
                            inline=True,
                            labelStyle={'display': 'inline-block', 'margin-right': '20px'},
                            style={'marginBottom': '20px'}
                        ),

                        html.Div('Frequency range:', style={'fontWeight': 'bold'}),
                        dcc.RangeSlider(0, 10, marks={i: f'{i} GHz' for i in range(0, 11)}, value=[1, 9],
                                        id='frequency-range'),
                        dcc.Checklist(
                            id='crop_input',
                            options=[{'label': 'Crop dataset', 'value': 'enabled'}],
                            value=[]
                        ),
                        dcc.RangeSlider(0, 10,
                                        marks={i: f'{i} GHz' for i in range(0, 11)},
                                        value=[1, 9],
                                        id='crop-range',
                                        className = 'hidden'),
                        html.Div('Inset coordinates:', style={'fontWeight': 'bold', 'marginTop': '10px'}),
                        html.Div([
                            html.Div([
                                html.Label('X Coordinate:'),
                                dcc.Input(id='x-coord', type='number', value=0.8, style={'width': '100px'}),
                            ], style={'display': 'inline-block', 'marginRight': '10px'}),

                            html.Div([
                                html.Label('Y Coordinate:'),
                                dcc.Input(id='y-coord', type='number', value=0.5, style={'width': '100px'}),
                            ], style={'display': 'inline-block'})
                        ], style={'marginBottom': '20px'}),

                        html.Div('Graph Title:', style={'fontWeight': 'bold'}),
                        dcc.Input(id='graph-title', type='text', placeholder='Graph title',
                                  style={'display': 'block', 'width': '400px', 'marginBottom': '20px'}),
                        html.Img(id='plot-image'),
                        html.Div(id='output-perf-fit'),
                        html.Div(id='refresh-div', style={'marginTop': '20px'}),
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '2%'}),
                ]),
            ], style={'padding': '20px'})
        ]),
        dcc.Tab(label='Measurements', children=[
            html.Div([
                html.Hr(),
                html.Div('This interface can be used to fit measurements, using the fitting code'),
                html.Hr(),

                html.Div([
                    html.Label('Measurement',style={'fontWeight': 'bold'}),
                    dcc.Upload(
                        id='upload-meas',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select Measurement File')
                        ]),
                        style={
                            'backgroundColor': '#CFE2F3',
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px'
                        }
                    ),
                dcc.Store(id='stored-meas'),  # Component to store the DataFrame
                html.Div(id='output-meas-upload'),
                html.Hr(),
                html.H3('Measurement Graph'),
                dcc.RadioItems(
                            id='meas_ys',
                            options=[{'label': option, 'value': option} for option in ['S-param', 'Y-param',]],
                            value='S-param',
                            inline=True,
                            labelStyle={'display': 'inline-block', 'margin-right': '20px'},
                            style={'marginBottom': '20px'}
                        ),
                dcc.Graph(id='measurement_graph')
                ], style={'display': 'inline-block', 'width': '30%', 'padding': '10px'}),

                html.Div([
                    html.Label('Short',style={'fontWeight': 'bold'}),
                    dcc.Upload(
                        id='upload-short',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select Short File')
                        ]),
                        style={
                            'width': '100%',
                            'backgroundColor': '#CFE2F3',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px'
                        }
                    ),
                dcc.Store(id='stored-short'),  # Component to store the DataFrame
                html.Div(id='output-short-upload'),
                html.Hr(),
                html.H3('Short Graph'),
                dcc.RadioItems(
                            id='short_ys',
                            options=[{'label': option, 'value': option} for option in ['S-param', 'Y-param',]],
                            value='S-param',
                            inline=True,
                            labelStyle={'display': 'inline-block', 'margin-right': '20px'},
                            style={'marginBottom': '20px'}
                        ),
                dcc.Graph(id='short_graph')
                ], style={'display': 'inline-block', 'width': '30%', 'padding': '10px'}),

                html.Div([
                    html.Label('Open',style={'fontWeight': 'bold'}),
                    dcc.Upload(
                        id='upload-open',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select Open File')
                        ]),
                        style={
                            'backgroundColor': '#CFE2F3',
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px'
                        }
                    ),
                dcc.Store(id='stored-open'),  # Component to store the DataFrame
                html.Div(id='output-open-upload'),
                html.Hr(),
                html.H3('Open Graph'),
                dcc.RadioItems(
                            id='open_ys',
                            options=[{'label': option, 'value': option} for option in ['S-param', 'Y-param',]],
                            value='S-param',
                            inline=True,
                            labelStyle={'display': 'inline-block', 'margin-right': '20px'},
                            style={'marginBottom': '20px'}
                        ),
                dcc.Graph(id='open_graph')
                ], style={'display': 'inline-block', 'width': '30%', 'padding': '10px'}),
            ]),
            html.Hr(),
            html.H3('De-embedding'),
            html.Hr(),
            html.Button(
                    id='show-meas-button',
                    children='SHOW FITTING',
                    n_clicks=0,
                    style={'marginTop': '20px'}),
            html.Hr(),
            html.Div([
                html.Div([
                    html.H3('Fitted measurement', style={'textAlign': 'center'}),
                    html.Img(id='plot-image-meas', style={'display': 'block', 'margin': '0 auto'}),
                    html.Div(id='output-perf-fit-meas', style={'display': 'block', 'margin': '0 auto'}),
                    html.Div(id='refresh-div-meas', style={'marginTop': '20px'}),
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'textAlign': 'center'}),

                html.Div([
                    html.H3('Parameters'),
                    html.Div('Prominence value:', style={'fontWeight': 'bold'}),
                    dcc.Input(
                        id='prominence_value_meas',
                        type='number',
                        value=0.8,
                        placeholder="Prominence",
                        style={'width': '100px', 'marginBottom': '10px'}
                    ),
                    html.Button('Refresh Graphs', id='refresh-button-meas', n_clicks=0, style={'marginLeft': '10px'}),
                    html.Div('Minimizer choice:', style={'fontWeight': 'bold'}),
                    dcc.RadioItems(
                        id='minimizer-meas',
                        options=[{'label': option, 'value': option} for option in
                                 ['reim', 'abs', 'abs_db', 'reim_db']],
                        value='abs_db',
                        inline=True,
                        labelStyle={'display': 'inline-block', 'margin-right': '20px'},
                        style={'marginBottom': '20px'}
                    ),
                    html.Div('Coupling Definition:', style={'fontWeight': 'bold'}),
                    dcc.RadioItems(
                        id='k2-meas',
                        options=[{'label': option, 'value': option} for option in ['k2_eff', 'rar', 'fbw']],
                        value='k2_eff',
                        inline=True,
                        labelStyle={'display': 'inline-block', 'margin-right': '20px'},
                        style={'marginBottom': '20px'}
                    ),
                    html.Div('Frequency range:', style={'fontWeight': 'bold'}),
                    dcc.RangeSlider(0, 10, marks={i: f'{i} GHz' for i in range(0, 11)}, value=[1, 9],
                                    id='frequency-range-meas'),
                    dcc.Checklist(
                        id='crop_input_meas',
                        options=[{'label': 'Crop dataset', 'value': 'enabled'}],
                        value=[]
                    ),
                    dcc.RangeSlider(0, 10,
                                    marks={i: f'{i} GHz' for i in range(0, 11)},
                                    value=[1, 9],
                                    id='crop-range-meas',
                                    className = 'hidden'),
                    html.Div('Inset coordinates:', style={'fontWeight': 'bold', 'marginTop': '10px'}),
                    html.Div([
                        html.Div([
                            html.Label('X Coordinate:'),
                            dcc.Input(id='x-coord-meas', type='number', value=0.8, style={'width': '100px'}),
                        ], style={'display': 'inline-block', 'marginRight': '10px'}),

                        html.Div([
                            html.Label('Y Coordinate:'),
                            dcc.Input(id='y-coord-meas', type='number', value=0.5, style={'width': '100px'}),
                        ], style={'display': 'inline-block'})
                    ], style={'marginBottom': '20px'}),

                    html.Div('Graph Title:', style={'fontWeight': 'bold'}),
                    dcc.Input(id='graph-title-meas', type='text', placeholder='Graph title',
                              style={'display': 'block', 'width': '400px', 'marginBottom': '20px'})
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '2%'}),
            ]),
            ], style={'padding': '20px'}),

        ]),

])