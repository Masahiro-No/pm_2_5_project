from dash import Dash, html, dcc, Input, Output, clientside_callback, callback
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import pandas as pd
from datetime import datetime, timedelta
import joblib
import numpy as np
import os


app = Dash(__name__, external_stylesheets=[dbc.themes.MINTY, dbc.icons.FONT_AWESOME])

# Load the model (or create a placeholder if not found)
model_path = 'pm25_prediction_model.joblib'
try:
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        model_loaded = True
    else:
        model_loaded = False
except:
    model_loaded = False

color_mode_switch = html.Span(
    [
        dbc.Label(className="fa fa-moon", html_for="color-mode-switch"),
        dbc.Switch(id="color-mode-switch", value=False, className="d-inline-block ms-1", persistence=True),
        dbc.Label(className="fa fa-sun", html_for="color-mode-switch"),
    ]
)

# Create initial PM2.5 prediction figure showing only today and 7 days ahead
def create_prediction_figure(dark_mode=False):
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
    
    # Use built-in plotly templates
    template_name = "plotly_dark" if dark_mode else "plotly"
    
    # Set background color based on dark mode
    bg_color = "#333" if dark_mode else "#fff"
    text_color = "#fff" if dark_mode else "#333"
    
    fig.update_layout(
        title='PM2.5 Prediction: Today vs. 7 Days Ahead',
        xaxis_title='Date',
        yaxis_title='PM2.5 (μg/m³)',
        template=template_name,
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=text_color)
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
    
    return fig

prediction_fig = create_prediction_figure()

# Layout with sidebar and main content
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H3(["PM2.5 Prediction Dashboard: วันนี้ vs 7 วันข้างหน้า"], className="bg-primary text-white p-2 mb-4"),
        ])
    ]),
    dbc.Row([
        # Sidebar with inputs
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Input Parameters", className="fw-bold"),
                dbc.CardBody([
                    html.Label("วันเริ่มต้น:", className="fw-bold mt-2"),
                    dcc.DatePickerSingle(
                        id='date-picker',
                        date=datetime.today().date(),
                        display_format='DD/MM/YYYY',
                        className="mb-3 w-100"
                    ),
                    
                    html.Label("PM2.5 ปัจจุบัน (μg/m³):", className="fw-bold"),
                    dbc.Input(
                        id='current-pm25-input',
                        type='number',
                        placeholder='Enter current PM2.5...',
                        value=25,
                        min=0,
                        className="mb-3"
                    ),
                    
                    html.Label("อุณหภูมิ (°C):", className="fw-bold"),
                    dbc.Input(
                        id='temperature-input',
                        type='number',
                        placeholder='Enter temperature...',
                        value=25,
                        className="mb-3"
                    ),
                    
                    html.Label("ความชื้น (%):", className="fw-bold"),
                    dbc.Input(
                        id='humidity-input',
                        type='number',
                        placeholder='Enter humidity...',
                        value=60,
                        min=0,
                        max=100,
                        className="mb-3"
                    ),
                    
                    dbc.Button("สร้างการพยากรณ์", id="predict-button", color="primary", className="w-100 mt-3"),
                    
                    html.Hr(),
                    
                    html.Div([
                        html.H5("สถานะโมเดล:", className="mb-2"),
                        html.Div(id="model-status", 
                                children=f"{'โมเดลพร้อมทำนาย' if model_loaded else 'ไม่พบโมเดล (ใช้การจำลองแทน)'}", 
                                className="fst-italic")
                    ]),
                    
                    html.Div([
                        color_mode_switch
                    ], className="mt-4 pt-2 border-top")
                ])
            ])
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
    # Handle the case when inputs are None
    if None in [date, current_pm25, temperature, humidity]:
        dark_mode_value = dark_mode if dark_mode is not None else False
        return create_prediction_figure(dark_mode_value), f"{'โมเดลพร้อมทำนาย' if model_loaded else 'ไม่พบโมเดล (ใช้การจำลองแทน)'}"
    
    # Parse date
    try:
        start_date = datetime.strptime(date, '%Y-%m-%d')
    except:
        # Use today's date if parsing fails
        start_date = datetime.today()
        
    day_7_date = start_date + timedelta(days=6)
    
    # Format dates for display
    today_str = start_date.strftime('%Y-%m-%d')
    day_7_str = day_7_date.strftime('%Y-%m-%d')
    
    # Use the model if loaded, otherwise use simulated prediction
    if model_loaded:
        try:
            # Extract month and day features
            dayofweek = start_date.weekday()
            month = start_date.month
            day = start_date.day
            
            # Prepare input for the model
            # Adjust these features based on your actual model's requirements
            X = pd.DataFrame({
                'humidity': [humidity],
                'temperature': [temperature],
                'dayofweek': [dayofweek],
                'month': [month],
                'day': [day],
                'pm25': [current_pm25],
            })
            
            # Predict PM2.5 for day 7
            day_7_value = float(model.predict(X)[0])
            
            # Keep within reasonable bounds
            day_7_value = max(5, min(80, day_7_value))
            
            model_status = f"ทำนายเสร็จสิ้นด้วยโมเดล: วันที่={date}, PM2.5 ปัจจุบัน={current_pm25}, อุณหภูมิ={temperature}°C, ความชื้น={humidity}%"
        except Exception as e:
            # Fallback to simulation if prediction fails
            day_7_value = simulate_prediction(current_pm25, temperature, humidity)
            model_status = f"โมเดลทำนายล้มเหลว (ใช้การจำลองแทน): {str(e)}"
    else:
        # Use simulated prediction
        day_7_value = simulate_prediction(current_pm25, temperature, humidity)
        model_status = f"ทำนายด้วยการจำลอง: วันที่={date}, PM2.5 ปัจจุบัน={current_pm25}, อุณหภูมิ={temperature}°C, ความชื้น={humidity}%"
    
    # Use built-in plotly templates
    template_name = "plotly_dark" if dark_mode else "plotly"
    
    # Set background color based on dark mode
    bg_color = "#333" if dark_mode else "#fff"
    text_color = "#fff" if dark_mode else "#333"
    
    fig = go.Figure()
    
    # Add line and markers
    dates = [today_str, day_7_str]
    pm25_values = [current_pm25, day_7_value]
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=pm25_values,
        mode='lines+markers',
        name='PM2.5 Prediction',
        marker=dict(size=10)
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
    display_date = start_date.strftime('%d/%m/%Y')
    display_future_date = day_7_date.strftime('%d/%m/%Y')
    
    fig.update_layout(
        title=f'การทำนาย PM2.5: {display_date} vs {display_future_date}',
        xaxis_title='วันที่',
        yaxis_title='PM2.5 (μg/m³)',
        template=template_name,
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=text_color)
    )
    
    # Format x-axis to show only the 2 dates
    fig.update_xaxes(
        tickmode='array',
        tickvals=dates,
        ticktext=[display_date, display_future_date]
    )
    
    # Add colored regions for PM2.5 levels
    fig.add_hrect(y0=0, y1=12, line_width=0, fillcolor="green", opacity=0.1)
    fig.add_hrect(y0=12, y1=35.4, line_width=0, fillcolor="yellow", opacity=0.1)
    fig.add_hrect(y0=35.4, y1=55.4, line_width=0, fillcolor="orange", opacity=0.1)
    fig.add_hrect(y0=55.4, y1=150.4, line_width=0, fillcolor="red", opacity=0.1)
    fig.add_hrect(y0=150.4, y1=250.4, line_width=0, fillcolor="purple", opacity=0.1)
    
    return fig, model_status

def simulate_prediction(current_pm25, temperature, humidity):
    """Simulate a prediction when the model is not available"""
    import random
    
    # Temperature effect: higher temps might increase PM2.5
    temp_effect = (temperature - 25) * 0.2
    
    # Humidity effect: higher humidity might decrease PM2.5 (precipitation)
    humidity_effect = (humidity - 50) * -0.1
    
    # Random variation for forecast
    variation = random.uniform(-8, 8)
    
    # Day 7 prediction with some relation to today's value
    day_7_value = current_pm25 * 0.6 + 0.4 * (current_pm25 + temp_effect + humidity_effect + variation)
    day_7_value = max(5, min(80, day_7_value))  # Keep within reasonable bounds
    
    return day_7_value

# Dark mode toggle - fix the implementation
clientside_callback(
    """
    function(switchOn) {
       document.documentElement.setAttribute('data-bs-theme', switchOn ? 'dark' : 'light');  
       return window.dash_clientside.no_update
    }
    """,
    Output("color-mode-switch", "id"),
    Input("color-mode-switch", "value"),
)

if __name__ == "__main__":
    app.run_server(debug=True)