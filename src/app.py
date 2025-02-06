from dash import Dash
import dash_bootstrap_components as dbc  # Optional, if using Bootstrap for styling

# Initialize the app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

from layout import layout
from callbacks import register_callbacks

# Set the layout from the layout.py file
app.layout = layout
register_callbacks(app)
# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)