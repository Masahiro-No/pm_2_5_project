#171

from dash import Dash, html, dcc, Input, Output, Patch, clientside_callback, callback
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import pandas as pd
from datetime import datetime, timedelta

# adds templates to plotly.io
load_figure_template(["minty", "minty_dark"])

df = px.data.gapminder()

app = Dash(__name__, external_stylesheets=[dbc.themes.MINTY, dbc.icons.FONT_AWESOME])

color_mode_switch = html.Span(
    [
        dbc.Label(className="fa fa-moon", html_for="color-mode-switch"),
        dbc.Switch(id="color-mode-switch", value=False, className="d-inline-block ms-1", persistence=True),
        dbc.Label(className="fa fa-sun", html_for="color-mode-switch"),
    ]
)

# Create initial PM2.5 prediction figure showing only today and 7 days ahead
def create_prediction_figure(template_name="minty"):
    # Generate only 2 dates: today and 7 days ahead
    today = datetime.today().strftime('%Y-%m-%d')
    day_7 = (datetime.today() + timedelta(days=6)).strftime('%Y-%m-%d')
    dates = [today, day_7]
    
    # Sample values for today and day 7
    pm25_values = [25, 25]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=pm25_values,
        mode='lines+markers',
        name='PM2.5 Prediction'
    ))
    
    fig.update_layout(
        title='PM2.5 Prediction: Today vs. 7 Days Ahead',
        xaxis_title='Date',
        yaxis_title='PM2.5 (μg/m³)',
        template=template_name
    )
    
    return fig

prediction_fig = create_prediction_figure()

# Layout with sidebar and main content
app.layout = dbc.Container([
    html.H3(["PM2.5 Prediction Dashboard: วันนี้ vs 7 วันข้างหน้า"], className="bg-primary text-white p-2 mb-4"),
    dbc.Row([
        # Sidebar with inputs
        dbc.Col([
            html.Div([
                html.H4("Input Parameters", className="mb-3"),
                
                html.Label("วันเริ่มต้น:", className="fw-bold mt-3"),
                dcc.DatePickerSingle(
                    id='date-picker',
                    date=datetime.today().date(),
                    display_format='YYYY-MM-DD',
                    className="mb-3 w-100"
                ),
                
                html.Label("PM2.5 ปัจจุบัน (μg/m³):", className="fw-bold"),
                dcc.Input(
                    id='current-pm25-input',
                    type='number',
                    placeholder='Enter current PM2.5...',
                    value=25,
                    min=0,
                    className="mb-3 w-100"
                ),
                
                html.Label("อุณหภูมิ (°C):", className="fw-bold"),
                dcc.Input(
                    id='temperature-input',
                    type='number',
                    placeholder='Enter temperature...',
                    value=25,
                    className="mb-3 w-100"
                ),
                
                html.Label("ความชื้น (%):", className="fw-bold"),
                dcc.Input(
                    id='humidity-input',
                    type='number',
                    placeholder='Enter humidity...',
                    value=60,
                    min=0,
                    max=100,
                    className="mb-3 w-100"
                ),
                
                html.Div([
                    dbc.Button("สร้างการพยากรณ์", id="predict-button", color="primary", className="w-100 mt-3"),
                ]),
                
                html.Div([
                    html.H5("สถานะโมเดล:", className="mt-4"),
                    html.Div(id="model-status", children="พร้อมทำนาย", className="fst-italic")
                ]),
                
                html.Div([
                    color_mode_switch
                ], className="mt-4 pt-2 border-top")
            ], className="p-3 border rounded")
        ], width=4),
        
        # Main content with prediction graph
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("ผลการทำนาย PM2.5: วันนี้ vs 7 วันข้างหน้า"),
                dbc.CardBody([
                    dcc.Graph(id="prediction-graph", figure=prediction_fig)
                ])
            ])
        ], width=8)
    ])
], fluid=True)

# Callback to update prediction graph based on inputs
@callback(
    [Output("prediction-graph", "figure"),
     Output("model-status", "children")],
    [Input("predict-button", "n_clicks"),
     Input("date-picker", "date"),
     Input("current-pm25-input", "value"),
     Input("temperature-input", "value"),
     Input("humidity-input", "value"),
     Input("color-mode-switch", "value")]
)
def update_prediction(n_clicks, date, current_pm25, temperature, humidity, dark_mode):
    if n_clicks is None:
        # Initial load
        template_name = "minty" if not dark_mode else "minty_dark"
        return create_prediction_figure(template_name), "พร้อมทำนาย"
    
    # This is a placeholder for your model integration
    def predict_pm25(date, current_pm25, temp, humidity):
        import random
        
        # Start date (today)
        start_date = datetime.strptime(date, '%Y-%m-%d')
        
        # Only predict today and 7 days ahead
        today = start_date.strftime('%Y-%m-%d')
        day_7 = (start_date + timedelta(days=6)).strftime('%Y-%m-%d')
        dates = [today, day_7]
        
        # Today's value is the current PM2.5
        today_value = current_pm25
        
        # Calculate day 7 value based on inputs
        # Temperature effect: higher temps might increase PM2.5
        temp_effect = (temp - 25) * 0.2
        
        # Humidity effect: higher humidity might decrease PM2.5 (precipitation)
        humidity_effect = (humidity - 50) * -0.1
        
        # Random variation for forecast
        variation = random.uniform(-8, 8)
        
        # Day 7 prediction with some relation to today's value
        day_7_value = today_value * 0.6 + 0.4 * (today_value + temp_effect + humidity_effect + variation)#แก้ตรงนี้
        day_7_value = max(5, min(80, day_7_value))  # Keep within reasonable bounds
        
        values = [today_value, day_7_value]
        
        return dates, values
    
    # Get predictions for today and day 7 only
    dates, pm25_values = predict_pm25(date, current_pm25, temperature, humidity)
    
    # Create figure with appropriate template
    template_name = "minty" if not dark_mode else "minty_dark"
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=pm25_values,
        mode='lines+markers',
        name='PM2.5 Prediction'
    ))
    
    # Add point annotations to show exact values
    fig.add_trace(go.Scatter(
        x=dates,
        y=pm25_values,
        mode='text',
        text=[f"{val:.1f}" for val in pm25_values],
        textposition="top center",
        textfont=dict(size=14),
        showlegend=False
    ))
    
    # Format date for title
    display_date = datetime.strptime(date, '%Y-%m-%d').strftime('%d/%m/%Y')
    
    fig.update_layout(
        title=f'การทำนาย PM2.5: {display_date} vs 7 วันข้างหน้า',
        xaxis_title='วันที่',
        yaxis_title='PM2.5 (μg/m³)',
        template=template_name
    )
    
    # Format x-axis to show only the 2 dates
    fig.update_xaxes(
        tickmode='array',
        tickvals=dates,
        ticktext=["วันเริ่มต้น", "7 วันข้างหน้า"]
    )
    
    # Add colored regions for PM2.5 levels
    fig.add_hrect(y0=0, y1=12, line_width=0, fillcolor="green", opacity=0.1)
    fig.add_hrect(y0=12, y1=35.4, line_width=0, fillcolor="yellow", opacity=0.1)
    fig.add_hrect(y0=35.4, y1=55.4, line_width=0, fillcolor="orange", opacity=0.1)
    fig.add_hrect(y0=55.4, y1=150.4, line_width=0, fillcolor="red", opacity=0.1)
    fig.add_hrect(y0=150.4, y1=250.4, line_width=0, fillcolor="purple", opacity=0.1)
    
    return fig, f"ทำนายเสร็จสิ้น: วันที่={date}, PM2.5 ปัจจุบัน={current_pm25}, อุณหภูมิ={temperature}°C, ความชื้น={humidity}%"

# Dark mode toggle
clientside_callback(
    """
    (switchOn) => {
       document.documentElement.setAttribute('data-bs-theme', switchOn ? 'light' : 'dark');  
       return window.dash_clientside.no_update
    }
    """,
    Output("color-mode-switch", "id"),
    Input("color-mode-switch", "value"),
)

if __name__ == "__main__":
    app.run_server(debug=True)