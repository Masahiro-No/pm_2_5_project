from dash import Dash, html, dcc, Input, Output, clientside_callback, callback
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import pandas as pd
from datetime import datetime, timedelta
import joblib
import numpy as np
import os
import random


app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG, dbc.icons.FONT_AWESOME])

# ใส่โมเดลที่นี้
model_path = 'pm25_prediction_model.joblib'
try:
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        model_loaded = True
    else:
        model_loaded = False
except:
    model_loaded = False

# Custom CSS for animations and styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>PM2.5 Prediction</title>
        {%favicon%}
        {%css%}
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;600&display=swap');
            
            body {
                font-family: 'Kanit', sans-serif;
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                background-attachment: fixed;
            }
            
            .dashboard-header {
                background: linear-gradient(90deg, #4a00e0, #8e2de2);
                border-radius: 15px;
                padding: 15px 20px;
                box-shadow: 0 10px 20px rgba(0,0,0,0.3);
                margin-bottom: 25px;
                position: relative;
                overflow: hidden;
            }
            
            .dashboard-header::after {
                content: "";
                position: absolute;
                top: 0;
                left: 0;
                width: 200%;
                height: 100%;
                background: linear-gradient(90deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0.1) 50%, rgba(255,255,255,0) 100%);
                transform: translateX(-100%);
                animation: shimmer 3s infinite;
            }
            
            @keyframes shimmer {
                100% {
                    transform: translateX(50%);
                }
            }
            
            .main-card {
                border-radius: 15px;
                overflow: hidden;
                box-shadow: 0 5px 15px rgba(0,0,0,0.3);
                transition: transform 0.3s, box-shadow 0.3s;
                border: none;
                background: rgba(30, 30, 60, 0.7);
                backdrop-filter: blur(10px);
            }
            
            .main-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 10px 25px rgba(0,0,0,0.4);
            }
            
            .card-header {
                background: linear-gradient(90deg, #00b4db, #0083b0);
                color: white;
                font-weight: bold;
                border: none;
            }
            
            .input-container {
                position: relative;
                margin-bottom: 20px;
            }
            
            .custom-input {
                background: rgba(255,255,255,0.1);
                border: 2px solid rgba(255,255,255,0.1);
                border-radius: 10px;
                color: white;
                transition: all 0.3s;
            }
            
            .custom-input:focus {
                background: rgba(255,255,255,0.15);
                border-color: #00b4db;
                box-shadow: 0 0 10px rgba(0,180,219,0.5);
            }
            
            .prediction-button {
                background: linear-gradient(90deg, #ff9966, #ff5e62);
                border: none;
                border-radius: 10px;
                padding: 12px;
                font-weight: bold;
                transition: all 0.3s;
                box-shadow: 0 5px 15px rgba(255,94,98,0.4);
            }
            
            .prediction-button:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 20px rgba(255,94,98,0.6);
                background: linear-gradient(90deg, #ff5e62, #ff9966);
            }
            
            .prediction-button:active {
                transform: translateY(1px);
            }
            
            .air-quality-indicator {
                border-radius: 10px;
                padding: 8px 15px;
                margin-bottom: 8px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                transition: transform 0.2s;
            }
            
            .air-quality-indicator:hover {
                transform: scale(1.02);
            }
            
            .color-mode-container {
                display: flex;
                align-items: center;
                justify-content: center;
                margin-top: 25px;
                padding-top: 15px;
                border-top: 1px solid rgba(255,255,255,0.1);
            }
            
            .fa-moon {
                color: #a0a0a0;
                margin-right: 10px;
            }
            
            .fa-sun {
                color: #ffb347;
                margin-left: 10px;
            }
            
            .animate-pulse {
                animation: pulse 2s infinite;
            }
            
            @keyframes pulse {
                0% {
                    opacity: 1;
                }
                50% {
                    opacity: 0.7;
                }
                100% {
                    opacity: 1;
                }
            }
            
            .floating {
                animation: floating 3s ease-in-out infinite;
            }
            
            @keyframes floating {
                0% {
                    transform: translateY(0px);
                }
                50% {
                    transform: translateY(-10px);
                }
                100% {
                    transform: translateY(0px);
                }
            }
            
            .level-description {
                display: flex;
                align-items: center;
            }
            
            .level-icon {
                margin-right: 10px;
                font-size: 18px;
            }
            
            .model-status-container {
                background: rgba(0,0,0,0.2);
                border-radius: 10px;
                padding: 15px;
                margin-top: 20px;
            }
            
            .tips-accordion .accordion-button {
                background: rgba(255,255,255,0.1);
                color: white;
            }
            
            .tips-accordion .accordion-body {
                background: rgba(0,0,0,0.2);
            }
            
            .date-picker {
                width: 100%;
                border-radius: 10px;
                overflow: hidden;
            }
            
            .weather-icon {
                font-size: 24px;
                margin-right: 10px;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

color_mode_switch = html.Div([
    dbc.Label(className="fa fa-moon", html_for="color-mode-switch"),
    dbc.Switch(id="color-mode-switch", value=True, className="d-inline-block mx-2", persistence=True),
    dbc.Label(className="fa fa-sun", html_for="color-mode-switch"),
], className="color-mode-container")

# Air quality descriptions and icons
air_quality_levels = [
    {"range": "0-12 μg/m³", "label": "ดี (ปลอดภัย)", "color": "rgba(0,128,0,0.2)", "icon": "fa-smile", "desc": "คุณภาพอากาศดี สามารถทำกิจกรรมกลางแจ้งได้ตามปกติ"},
    {"range": "12-35.4 μg/m³", "label": "ปานกลาง", "color": "rgba(255,255,0,0.2)", "icon": "fa-meh", "desc": "คุณภาพอากาศยอมรับได้ แต่ผู้ที่ไวต่อมลพิษควรระวัง"},
    {"range": "35.4-55.4 μg/m³", "label": "ไม่ดีต่อกลุ่มเสี่ยง", "color": "rgba(255,165,0,0.2)", "icon": "fa-frown", "desc": "ผู้ที่มีโรคระบบทางเดินหายใจควรลดการออกกำลังกายกลางแจ้ง"},
    {"range": "55.4-150.4 μg/m³", "label": "ไม่ดีต่อสุขภาพ", "color": "rgba(255,0,0,0.2)", "icon": "fa-angry", "desc": "ทุกคนควรลดกิจกรรมกลางแจ้ง โดยเฉพาะกลุ่มเสี่ยง"},
    {"range": "150.4-250.4 μg/m³", "label": "ไม่ดีต่อสุขภาพมาก", "color": "rgba(128,0,128,0.2)", "icon": "fa-dizzy", "desc": "หลีกเลี่ยงกิจกรรมกลางแจ้งทั้งหมด พิจารณาใช้หน้ากากอนามัย"}
]

# Create initial PM2.5 prediction figure
def create_prediction_figure(dark_mode=True):
    day_7 = (datetime.today() + timedelta(days=6)).strftime('%Y-%m-%d')
    predicted_pm25 = 20
    
    fig = go.Figure()
    
    # Add gauge chart instead of bar chart
    fig.add_trace(go.Indicator(
        mode = "gauge+number+delta",
        value = predicted_pm25,
        domain = {'x': [0, 1], 'y': [0, 1]},
        delta = {'reference': 12, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [None, 250], 'tickwidth': 1, 'tickcolor': "white" if dark_mode else "black"},
            'bar': {'color': "#6495ED"},
            'bgcolor': "rgba(0,0,0,0)" if dark_mode else "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 12], 'color': 'rgba(0,128,0,0.4)'},
                {'range': [12, 35.4], 'color': 'rgba(255,255,0,0.4)'},
                {'range': [35.4, 55.4], 'color': 'rgba(255,165,0,0.4)'},
                {'range': [55.4, 150.4], 'color': 'rgba(255,0,0,0.4)'},
                {'range': [150.4, 250.4], 'color': 'rgba(128,0,128,0.4)'},
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': predicted_pm25
            }
        }
    ))
    
    # Use built-in plotly templates
    template_name = "plotly_dark" if dark_mode else "plotly"
    
    # Set background color based on dark mode
    bg_color = "rgba(0,0,0,0)" if dark_mode else "#fff"
    text_color = "#fff" if dark_mode else "#333"
    
    fig.update_layout(
        title='PM2.5 Prediction for 7 Days Ahead',
        template=template_name,
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=text_color, family="Kanit"),
        margin=dict(l=20, r=20, t=60, b=20),
        height=400
    )
    
    return fig

prediction_fig = create_prediction_figure()

# Generate health tips based on PM2.5 level
def get_health_tips(pm25_level):
    general_tips = [
        "ติดตามคุณภาพอากาศเป็นประจำทุกวัน",
        "พกหน้ากากอนามัยติดตัวไว้เสมอ",
        "หลีกเลี่ยงการออกกำลังกายกลางแจ้งเมื่อคุณภาพอากาศไม่ดี",
        "ปิดหน้าต่างเมื่อคุณภาพอากาศภายนอกแย่",
        "ใช้เครื่องฟอกอากาศในบ้าน"
    ]
    
    if pm25_level <= 12:
        specific_tips = [
            "สามารถทำกิจกรรมกลางแจ้งได้ตามปกติ",
            "เป็นช่วงเวลาที่ดีในการเปิดหน้าต่างระบายอากาศ",
            "ควรใช้โอกาสนี้ในการออกกำลังกายกลางแจ้ง"
        ]
    elif pm25_level <= 35.4:
        specific_tips = [
            "กลุ่มเสี่ยงควรพิจารณาลดการออกกำลังกายกลางแจ้งที่หนัก",
            "ควรพกหน้ากากอนามัยเมื่อออกนอกบ้านเป็นเวลานาน",
            "ระวังอาการผิดปกติของระบบทางเดินหายใจ"
        ]
    elif pm25_level <= 55.4:
        specific_tips = [
            "กลุ่มเสี่ยงควรหลีกเลี่ยงการออกกำลังกายกลางแจ้ง",
            "ควรใช้หน้ากาก N95 เมื่อต้องอยู่กลางแจ้งเป็นเวลานาน",
            "ปิดหน้าต่างเพื่อป้องกันฝุ่นเข้าบ้าน",
            "ใช้เครื่องฟอกอากาศในห้องนอน"
        ]
    elif pm25_level <= 150.4:
        specific_tips = [
            "ทุกคนควรหลีกเลี่ยงการออกกำลังกายกลางแจ้ง",
            "สวมหน้ากาก N95 เมื่อออกนอกบ้าน",
            "ลดการออกนอกบ้านโดยไม่จำเป็น",
            "ใช้เครื่องฟอกอากาศตลอดเวลาที่อยู่ในบ้าน"
        ]
    else:
        specific_tips = [
            "หลีกเลี่ยงการออกนอกบ้านโดยเด็ดขาด",
            "สวมหน้ากาก N95 ตลอดเวลาที่อยู่นอกบ้าน",
            "ปิดประตูหน้าต่างให้สนิท",
            "ใช้เครื่องฟอกอากาศในทุกห้อง",
            "พิจารณาอพยพไปยังพื้นที่ที่มีคุณภาพอากาศดีกว่า"
        ]
    
    # Combine and randomize tips
    all_tips = specific_tips + random.sample(general_tips, min(3, len(general_tips)))
    random.shuffle(all_tips)
    
    return all_tips[:5]  # Return 5 tips

# Create animated dust particle background effect
dust_particles = dbc.Container([
    html.Div("", className="dust-particle", 
             style={"left": f"{random.randint(0, 100)}%", 
                    "top": f"{random.randint(0, 100)}%",
                    "width": f"{random.randint(2, 8)}px",
                    "height": f"{random.randint(2, 8)}px",
                    "animation-delay": f"{random.random()*5}s",
                    "animation-duration": f"{random.randint(10, 20)}s"})
    for _ in range(20)
], fluid=True, className="dust-container")

# Create animated weather icon based on humidity and temperature
def get_weather_icon(temperature, humidity):
    if temperature > 30 and humidity < 40:
        return html.I(className="fas fa-sun weather-icon text-warning")
    elif temperature > 25 and humidity > 70:
        return html.I(className="fas fa-cloud-sun-rain weather-icon text-info")
    elif humidity > 80:
        return html.I(className="fas fa-cloud-rain weather-icon text-primary")
    elif temperature < 20:
        return html.I(className="fas fa-snowflake weather-icon text-info")
    else:
        return html.I(className="fas fa-cloud-sun weather-icon text-warning")

# Layout with sidebar and main content
app.layout = dbc.Container([
    # Animated header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H2([
                    html.I(className="fas fa-wind mr-2"), 
                    " PM2.5 PREDICTION ", 
                    html.Span("SYSTEM", className="text-warning")
                ], className="mb-0"),
                html.P("ระบบพยากรณ์คุณภาพอากาศล่วงหน้า 7 วัน", className="mb-0 mt-2 text-light-emphasis")
            ], className="dashboard-header text-white text-center")
        ])
    ]),
    
    dbc.Row([
        # Sidebar with inputs
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-sliders-h mr-2"),
                    "ตั้งค่าพารามิเตอร์",
                ], className="fw-bold"),
                dbc.CardBody([
                    html.Div([
                        html.Label([
                            html.I(className="fas fa-calendar-alt mr-2"), 
                            "วันที่เริ่มต้น:"
                        ], className="fw-bold mt-2"),
                        dcc.DatePickerSingle(
                            id='date-picker',
                            date=datetime.today().date(),
                            display_format='DD/MM/YYYY',
                            className="date-picker mb-3"
                        ),
                    ], className="input-container"),
                    
                    html.Div([
                        html.Label([
                            html.I(className="fas fa-temperature-high mr-2"), 
                            "อุณหภูมิ (°C):"
                        ], className="fw-bold"),
                        dbc.Input(
                            id='temperature-input',
                            type='number',
                            placeholder='Enter temperature...',
                            value=25,
                            min=0,
                            max=50,
                            className="custom-input mb-3"
                        ),
                    ], className="input-container"),
                    
                    html.Div([
                        html.Label([
                            html.I(className="fas fa-tint mr-2"), 
                            "ความชื้น (%):"
                        ], className="fw-bold"),
                        dbc.Input(
                            id='humidity-input',
                            type='number',
                            placeholder='Enter humidity...',
                            value=60,
                            min=0,
                            max=100,
                            className="custom-input mb-3"
                        ),
                    ], className="input-container"),
                    
                    dbc.Button([
                        html.I(className="fas fa-magic mr-2"), 
                        "สร้างการพยากรณ์"
                    ], id="predict-button", color="primary", className="prediction-button w-100 mt-3"),
                    
                    html.Div([
                        html.H5([
                            html.I(className="fas fa-info-circle mr-2"),
                            "สถานะโมเดล:"
                        ], className="mb-2"),
                        html.Div(id="model-status", 
                                className="fst-italic animate-pulse")
                    ], className="model-status-container"),
                    
                    html.Div([
                        dbc.Accordion([
                            dbc.AccordionItem([
                                html.Div(id="health-tips-content", className="mt-2")
                            ], title="คำแนะนำสุขภาพ", item_id="health-tips")
                        ], start_collapsed=True, id="tips-accordion", className="tips-accordion mt-3")
                    ]),
                    
                    html.Div([
                        color_mode_switch
                    ])
                ])
            ], className="main-card h-100")
        ], width=3),
        
        # Main content with prediction graph
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-chart-line mr-2"),
                    "ผลการทำนาย PM2.5: 7 วันข้างหน้า",
                    html.Div(id="weather-indicator", className="float-end")
                ]),
                dbc.CardBody([
                    dcc.Graph(id="prediction-graph", figure=prediction_fig, className="mb-3"),
                    
                    html.Div(id="air-quality-status", className="text-center mt-3 mb-4"),
                ])
            ], className="main-card mb-4"),
            
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-lungs mr-2"),
                    "ระดับคุณภาพอากาศและผลกระทบ"
                ]),
                dbc.CardBody([
                    html.Div([
                        html.Div([
                            html.Div([
                                html.I(className=f"fas {level['icon']} level-icon"),
                                level['range'] + ": " + level['label']
                            ], className="level-description"),
                            html.Small(level['desc'], className="text-light-emphasis")
                        ], className="air-quality-indicator py-2 px-3 mb-2", 
                          style={"backgroundColor": level['color']})
                        for level in air_quality_levels
                    ])
                ])
            ], className="main-card")
        ], width=9)
    ])
    
], fluid=True, className="mt-3")

# Callback to update prediction graph and other UI elements based on inputs
@callback(
    [Output("prediction-graph", "figure"),
     Output("model-status", "children"),
     Output("air-quality-status", "children"),
     Output("health-tips-content", "children"),
     Output("weather-indicator", "children")
     ],
    [Input("predict-button", "n_clicks"),
     Input("date-picker", "date"),
     Input("temperature-input", "value"),
     Input("humidity-input", "value"),
     Input("color-mode-switch", "value")]
)
def update_prediction(n_clicks, date, temperature, humidity, dark_mode):
    # Default values for countdown

    
    # Handle the case when inputs are None
    if None in [date, temperature, humidity]:
        dark_mode_value = dark_mode if dark_mode is not None else True
        default_weather = get_weather_icon(25, 60)
        default_air_status = html.Div([
            html.H4("กรุณากรอกข้อมูลและกดปุ่มพยากรณ์", className="text-secondary"),
        ])
        default_health_tips = [html.Li("กรุณากดปุ่มพยากรณ์เพื่อดูคำแนะนำสุขภาพ")]
        return (
            create_prediction_figure(dark_mode_value), 
            f"{'โมเดลพร้อมทำนาย' if model_loaded else 'ไม่พบโมเดล (ใช้การจำลองแทน)'}", 
            default_air_status,
            html.Ul(default_health_tips),
            default_weather,
            days,
            hours,
            minutes
        )
    
    # Parse date
    try:
        start_date = datetime.strptime(date, '%Y-%m-%d')
    except:
        # Use today's date if parsing fails
        start_date = datetime.today()
        
    # Calculate days until prediction date
    day_7_date = start_date + timedelta(days=6)
    
    # Format date for display
    display_future_date = day_7_date.strftime('%d/%m/%Y')
    
    # Get weather icon
    weather_icon = get_weather_icon(temperature, humidity)
    
    # Use the model if loaded, otherwise use simulated prediction
    if model_loaded:
        try:
            # Extract month, day features
            dayofweek = start_date.weekday()
            month = start_date.month
            day = start_date.day
            
            # Prepare input for the model
            X = pd.DataFrame({
                'humidity': [humidity],
                'temperature': [temperature],
                'dayofweek': [dayofweek],
                'month': [month],
                'day': [day]
            })
            
            # Predict PM2.5 for day 7
            predicted_value = float(model.predict(X)[0])
            
            # Keep within reasonable bounds
            predicted_value = max(5, min(250, predicted_value))
            
            model_status = f"ทำนายเสร็จสิ้นด้วยโมเดล: วันที่={date}, อุณหภูมิ={temperature}°C, ความชื้น={humidity}%"
        except Exception as e:
            # Fallback to simulation if prediction fails
            predicted_value = simulate_prediction(temperature, humidity, start_date)
            model_status = f"โมเดลทำนายล้มเหลว (ใช้การจำลองแทน): {str(e)}"
    else:
        # Use simulated prediction
        predicted_value = simulate_prediction(temperature, humidity, start_date)
        model_status = f"ทำนายด้วยการจำลอง: วันที่={date}, อุณหภูมิ={temperature}°C, ความชื้น={humidity}%"
    
    # Determine air quality level
    air_quality_level = None
    air_quality_color = None
    air_quality_icon = None
    air_quality_label = None
    air_quality_desc = None
    
    if predicted_value <= 12:
        air_quality_level = "ดี"
        air_quality_color = "green"
        air_quality_icon = "fa-smile"
        air_quality_label = "คุณภาพอากาศดี"
        air_quality_desc = "ปลอดภัยสำหรับทุกคน"
    elif predicted_value <= 35.4:
        air_quality_level = "ปานกลาง"
        air_quality_color = "yellow"
        air_quality_icon = "fa-meh"
        air_quality_label = "คุณภาพอากาศปานกลาง"
        air_quality_desc = "กลุ่มเสี่ยงควรระวัง"
    elif predicted_value <= 55.4:
        air_quality_level = "ไม่ดีต่อกลุ่มเสี่ยง"
        air_quality_color = "orange"
        air_quality_icon = "fa-frown"
        air_quality_label = "คุณภาพอากาศไม่ดีต่อกลุ่มเสี่ยง"
        air_quality_desc = "ผู้ที่มีโรคระบบทางเดินหายใจควรหลีกเลี่ยงกิจกรรมกลางแจ้ง"
    elif predicted_value <= 150.4:
        air_quality_level = "ไม่ดีต่อสุขภาพ"
        air_quality_color = "red"
        air_quality_icon = "fa-angry"
        air_quality_label = "คุณภาพอากาศไม่ดีต่อสุขภาพ"
        air_quality_desc = "ทุกคนควรลดกิจกรรมกลางแจ้ง"
    else:
        air_quality_level = "ไม่ดีต่อสุขภาพมาก"
        air_quality_color = "purple"
        air_quality_icon = "fa-dizzy"
        air_quality_label = "คุณภาพอากาศไม่ดีต่อสุขภาพมาก"
        air_quality_desc = "หลีกเลี่ยงกิจกรรมกลางแจ้งทั้งหมด"
    
    # Create air quality status display
    air_quality_status = html.Div([
        html.Div([
            html.I(className=f"fas {air_quality_icon} fa-3x text-{air_quality_color}"),
            html.H3(air_quality_label, className=f"text-{air_quality_color} mt-2"),
            html.P(air_quality_desc, className="text-light-emphasis")
        ], className="text-center p-3 floating")
    ], className="bg-dark bg-opacity-50 rounded p-2")
    
    # Get health tips based on PM2.5 level
    health_tips = get_health_tips(predicted_value)
    health_tips_elements = [html.Li(tip) for tip in health_tips]
    
    # Create gauge chart
    fig = go.Figure()
    
    # Add gauge chart 
    fig.add_trace(go.Indicator(
        mode = "gauge+number",
        value = predicted_value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"การทำนาย PM2.5 สำหรับวันที่ {display_future_date}"},
        gauge = {
            'axis': {'range': [None, 250], 'tickwidth': 1},
            'bar': {'color': air_quality_color},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 12], 'color': 'rgba(0,128,0,0.4)'},
                {'range': [12, 35.4], 'color': 'rgba(255,255,0,0.4)'},
                {'range': [35.4, 55.4], 'color': 'rgba(255,165,0,0.4)'},
                {'range': [55.4, 150.4], 'color': 'rgba(255,0,0,0.4)'},
                {'range': [150.4, 250.4], 'color': 'rgba(128,0,128,0.4)'},
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': predicted_value
            }
        }
    ))
    
    # Use built-in plotly templates
    template_name = "plotly_dark" if dark_mode else "plotly"
    
    # Set background color based on dark mode
    bg_color = "rgba(0,0,0,0)" if dark_mode else "#fff"
    text_color = "#fff" if dark_mode else "#333"
    
    fig.update_layout(
        template=template_name,
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=text_color, family="Kanit"),
        margin=dict(l=20, r=20, t=60, b=20),
        height=400
    )
    
    # Add annotation for the air quality level
    fig.add_annotation(
        x=0.5,
        y=0.25,
        text=f"ระดับอากาศ: {air_quality_level}",
        showarrow=False,
        font=dict(size=16)
    )
    
    return fig, model_status, air_quality_status, html.Ul(health_tips_elements), weather_icon

def simulate_prediction(temperature, humidity, date):
    """Simulate a prediction when the model is not available"""
    # Temperature effect: higher temps might increase PM2.5
    temp_effect = (temperature - 25) * 0.3
    
    # Humidity effect: higher humidity might decrease PM2.5 (precipitation)
    humidity_effect = (humidity - 50) * -0.15
    
    # Seasonal effect (higher in winter, lower in summer)
    month = date.month
    seasonal_effect = 0
    if 11 <= month <= 12 or 1 <= month <= 2:  # Winter
        seasonal_effect = 20
    elif 3 <= month <= 5:  # Spring
        seasonal_effect = 10
    elif 6 <= month <= 8:  # Summer
        seasonal_effect = -10
    
    # Day of week effect (higher on weekdays due to traffic)
    day_of_week = date.weekday()
    weekday_effect = 5 if day_of_week < 5 else -5
    
    # Base value (around 25)
    base_value = 25
    
    # Random variation
    variation = random.uniform(-8, 8)
    
    # Predicted value with environmental factors
    predicted_value = base_value + temp_effect + humidity_effect + seasonal_effect + weekday_effect + variation
    
    # Add some randomized extreme events
    if random.random() < 0.05:  # 5% chance of extreme event
        extreme_factor = random.uniform(1.5, 2.5)
        predicted_value *= extreme_factor
    
    predicted_value = max(5, min(250, predicted_value))  # Keep within reasonable bounds
    
    return predicted_value

# Dark mode toggle
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