from dash import Dash, html, dcc, Input, Output, clientside_callback, callback,State
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import pandas as pd
from datetime import datetime, timedelta
import joblib
import numpy as np
import os
import random
from pycaret.regression import *
from pycaret import *

app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG, dbc.icons.FONT_AWESOME])
# ใส่โมเดลที่นี้
model_path = 'models/Update_Fisrt_models.pkl'
try:
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        model_loaded = True
        print('success')
    else:
        model_loaded = False
        print('false')
except Exception as e:
    print(f"Error loading model: {e}")
    model_loaded = False

df = pd.read_csv("clean_data.csv", parse_dates=["timestamp"], index_col="timestamp")
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
            
            .dust-container {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
                z-index: 0;
            }

            .dust-particle {
                position: absolute;
                background-color: rgba(255, 255, 255, 0.3);
                border-radius: 50%;
                animation: float linear infinite;
                pointer-events: none;
            }

            @keyframes float {
                0% {
                    transform: translateY(0) rotate(0deg);
                    opacity: 0.4;
                }
                25% {
                    opacity: 0.8;
                }
                50% {
                    transform: translateY(-100vh) rotate(360deg);
                    opacity: 0.4;
                }
                75% {
                    opacity: 0.8;
                }
                100% {
                    transform: translateY(-200vh) rotate(720deg);
                    opacity: 0.4;
                }
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


# Air quality descriptions and icons
air_quality_levels = [
    {"range": "0-12 μg/m³", "label": "ดี (ปลอดภัย)", "color": "rgba(0,128,0,0.2)", "icon": "fa-smile", "desc": "คุณภาพอากาศดี สามารถทำกิจกรรมกลางแจ้งได้ตามปกติ"},
    {"range": "12-35.4 μg/m³", "label": "ปานกลาง", "color": "rgba(255,255,0,0.2)", "icon": "fa-meh", "desc": "คุณภาพอากาศยอมรับได้ แต่ผู้ที่ไวต่อมลพิษควรระวัง"},
    {"range": "35.4-55.4 μg/m³", "label": "ไม่ดีต่อกลุ่มเสี่ยง", "color": "rgba(255,165,0,0.2)", "icon": "fa-frown", "desc": "ผู้ที่มีโรคระบบทางเดินหายใจควรลดการออกกำลังกายกลางแจ้ง"},
    {"range": "55.4-150.4 μg/m³", "label": "ไม่ดีต่อสุขภาพ", "color": "rgba(255,0,0,0.2)", "icon": "fa-angry", "desc": "ทุกคนควรลดกิจกรรมกลางแจ้ง โดยเฉพาะกลุ่มเสี่ยง"},
    {"range": "150.4-250.4 μg/m³", "label": "ไม่ดีต่อสุขภาพมาก", "color": "rgba(128,0,128,0.2)", "icon": "fa-dizzy", "desc": "หลีกเลี่ยงกิจกรรมกลางแจ้งทั้งหมด พิจารณาใช้หน้ากากอนามัย"}
]

# Create initial PM2.5 prediction figure
def create_prediction_figure():
    predicted_pm25 = 20
    fig = go.Figure()
    
    # Add gauge chart instead of bar chart
    fig.add_trace(go.Indicator(
        mode = "gauge+number+delta",
        value = predicted_pm25,
        domain = {'x': [0, 1], 'y': [0, 1]},
        delta = {'reference': 12, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [None, 250], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#6495ED"},
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
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': predicted_pm25
            }
        }
    ))
    
    # Use built-in plotly templates
    template_name = "plotly_dark"
    
    # Set background color based on dark mode
    bg_color = "rgba(0,0,0,0)"
    text_color = "#fff"
    
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
    dust_particles,
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
                            html.Small(level['desc'])
                        ], className="air-quality-indicator py-2 px-3 mb-2", 
                          style={"backgroundColor": level['color']})
                        for level in air_quality_levels
                    ])
                ])
            ], className="main-card")
        ], width=9)
    ])

], fluid=True, className="mt-3")
@callback(
    [Output("prediction-graph", "figure"),
     Output("model-status", "children"),
     Output("air-quality-status", "children"),
     Output("health-tips-content", "children"),
     Output("weather-indicator", "children")
     ],
    [Input("predict-button", "n_clicks")],
    [State("date-picker", "date"),
     State("temperature-input", "value"),
     State("humidity-input", "value")]  
    )
def update_prediction(n_clicks, date, temperature, humidity):
    # เช็คว่ามีการกดปุ่มจริงๆ หรือไม่
    if n_clicks is None or None in [date, temperature, humidity]:
        # สร้างค่าเริ่มต้นเมื่อยังไม่มีการกดปุ่ม

        default_weather = get_weather_icon(25, 60)
        default_air_status = html.Div([
            html.H4("กรุณากรอกข้อมูลและกดปุ่มพยากรณ์", className="text-secondary"),
        ])
        default_health_tips = [html.Li("กรุณากดปุ่มพยากรณ์เพื่อดูคำแนะนำสุขภาพ")]
        return (
            create_prediction_figure(), 
            "รอการทำนายจากโมเดล", 
            default_air_status,
            html.Ul(default_health_tips),
            default_weather
        )
        
    # Parse date
    try:
        start_date = datetime.strptime(date, '%Y-%m-%d')
    except:
        # Use today's date if parsing fails
        start_date = datetime.today()
    
    # เก็บข้อมูลปัจจุบันและข้อมูลในอดีต
    latest_input = pd.DataFrame({
        'temperature': [temperature],
        'humidity': [humidity],
        'dayofweek': [start_date.weekday()],
        'month': [start_date.month],
        'day': [start_date.day]
    })
    
    # ใช้ข้อมูลในอดีตสำหรับอ้างอิง
    historical_df = df.copy()
    
    # สร้างช่วงเวลาในอนาคต 7 วัน (169 ชั่วโมง)
    future_dates = pd.date_range(start=start_date, periods=169, freq='H')[1:]
    future_predictions = []
    
    # สร้าง lag features จากการหาค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐานจากข้อมูลในอดีต
    for lag in range(1, 8):
        lag_col = f'pm_2_5_lag_{lag}'
        
        # หาข้อมูลที่มีวันและเดือนเดียวกัน
        historical_data = historical_df[(historical_df.index.month == start_date.month) & 
                                        (historical_df.index.day == start_date.day)]
        
        # ถ้าไม่มีข้อมูลในวันเดียวกัน ให้ใช้ข้อมูลในเดือนเดียวกัน
        if len(historical_data) == 0:
            historical_data = historical_df[historical_df.index.month == start_date.month]
        
        # ถ้าไม่มีข้อมูลในเดือนเดียวกัน ให้ใช้ข้อมูลทั้งหมด
        if len(historical_data) == 0:
            historical_data = historical_df
        
        # คำนวณค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐาน
        if lag_col in historical_data.columns:
            mean_val = historical_data[lag_col].mean()
            std_val = historical_data[lag_col].std() if not pd.isna(historical_data[lag_col].std()) else 2.0
            
            # สุ่มค่าจากค่าเฉลี่ย +/- ส่วนเบี่ยงเบนมาตรฐาน
            random_val = mean_val + random.uniform(-1, 1) * std_val
            latest_input[lag_col] = random_val
        else:
            # ถ้าไม่มีคอลัมน์นี้ในข้อมูลประวัติ ใช้ค่าเริ่มต้น
            latest_input[lag_col] = 0
    
    model_status = ""
    
    # ทำนายค่า PM2.5 สำหรับแต่ละชั่วโมงในอนาคต
    for date in future_dates:
        # อัปเดตคุณลักษณะตามเวลา
        latest_input['dayofweek'] = date.dayofweek
        latest_input['month'] = date.month
        latest_input['day'] = date.day
        
        # อัปเดตคุณลักษณะสภาพอากาศโดยใช้ค่าเฉลี่ยจากข้อมูลในอดีต
        latest_input['temperature'] = get_estimated_value(historical_df, date, 'temperature')
        latest_input['humidity'] = get_estimated_value(historical_df, date, 'humidity')
        
        # ทำนายค่า PM2.5
        if model_loaded:
            try:
                pred = float(predict_model(model, data=latest_input)['prediction_label'][0])
                model_status = f"ทำนายเสร็จสิ้นด้วยโมเดล: วันที่={date.strftime('%Y-%m-%d')}, อุณหภูมิ={latest_input['temperature'].values[0]:.1f}°C, ความชื้น={latest_input['humidity'].values[0]:.1f}%"
            except Exception as e:
                # Fallback หากการทำนายล้มเหลว
                pred = get_estimated_value(historical_df, date, 'pm_2_5')
                model_status = f"โมเดลทำนายล้มเหลว (ใช้ข้อมูลในอดีต): {str(e)}"
        else:
            # ใช้ข้อมูลในอดีตหากไม่มีโมเดล
            pred = get_estimated_value(historical_df, date, 'pm_2_5')
            model_status = f"ทำนายด้วยข้อมูลในอดีต: วันที่={date.strftime('%Y-%m-%d')}, อุณหภูมิ={latest_input['temperature'].values[0]:.1f}°C, ความชื้น={latest_input['humidity'].values[0]:.1f}%"
        
        # เก็บค่าทำนาย
        future_predictions.append((date, pred))
        
        # อัปเดต lag features
        for lag in range(7, 1, -1):
            latest_input[f'pm_2_5_lag_{lag}'] = latest_input[f'pm_2_5_lag_{lag-1}']
        latest_input['pm_2_5_lag_1'] = pred
    
    # แปลงเป็น DataFrame
    future_df = pd.DataFrame(future_predictions, columns=['timestamp', 'pm_2_5'])
    future_df.set_index('timestamp', inplace=True)
    
    # เฉลี่ยเป็นรายวัน
    daily_future_df = future_df.resample('D').mean()
    
    # แสดงเฉพาะวันที่ 7 (วันสุดท้าย) ของการทำนาย
    seventh_day_prediction = daily_future_df.iloc[6:7]  # ดัชนี 6 คือวันที่ 7 (เริ่มจาก 0)
    predicted_value = float(seventh_day_prediction['pm_2_5'].values[0])
    
    # Calculate days until prediction date
    day_7_date = start_date + timedelta(days=6)
    
    # Format date for display
    display_future_date = day_7_date.strftime('%d/%m/%Y')
    
    # Get weather icon
    weather_icon = get_weather_icon(latest_input['temperature'].values[0], latest_input['humidity'].values[0])
    
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
    template_name = "plotly_dark"
    
    bg_color = "rgba(0,0,0,0)"
    text_color = "#fff"
    
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

# ฟังก์ชันช่วยสำหรับการประมาณค่าจากข้อมูลในอดีต
def get_estimated_value(df, date, column):
    """
    หาค่าเฉลี่ยของคอลัมน์ที่ระบุจากข้อมูลในอดีต โดยใช้เงื่อนไขวันและเดือนเดียวกัน
    
    Parameters:
    df (DataFrame): DataFrame ที่มีข้อมูลในอดีต
    date (Timestamp): วันที่ต้องการประมาณค่า
    column (str): ชื่อคอลัมน์ที่ต้องการหาค่าเฉลี่ย
    
    Returns:
    float: ค่าเฉลี่ยของคอลัมน์ที่ระบุ +/- ค่าสุ่มจากส่วนเบี่ยงเบนมาตรฐาน
    """
    # หาข้อมูลที่มีเดือนและวันเดียวกัน
    historical_data = df[(df.index.month == date.month) & (df.index.day == date.day)]
    
    if len(historical_data) > 0:
        # ถ้ามีข้อมูลในวันและเดือนเดียวกัน
        mean_val = historical_data[column].mean()
        std_val = historical_data[column].std() if not pd.isna(historical_data[column].std()) else 2.0
        return mean_val + random.uniform(-1, 1) * std_val
    
    elif len(df[df.index.month == date.month]) > 0:
        # ถ้าไม่มีข้อมูลในวันเดียวกัน ให้ใช้ข้อมูลในเดือนเดียวกัน
        month_data = df[df.index.month == date.month]
        mean_val = month_data[column].mean()
        std_val = month_data[column].std() if not pd.isna(month_data[column].std()) else 2.0
        return mean_val + random.uniform(-1, 1) * std_val
    
    else:
        # ถ้าไม่มีข้อมูลที่เกี่ยวข้องเลย ให้ใช้ค่าเฉลี่ยทั้งหมด
        mean_val = df[column].mean()
        std_val = df[column].std() if not pd.isna(df[column].std()) else 2.0
        return mean_val + random.uniform(-1, 1) * std_val
    
def simulate_prediction_from_historical(temperature, humidity, date, df):
    """
    Simulate a prediction using historical data for the same day/month
    """
    # Find data from the same day and month in historical records
    month = date.month
    day = date.day
    # Try to get historical records from the same day and month
    historical_data = df[(df.index.month == month) & (df.index.day == day)]
    
    # If no exact matches, use data from the same month
    if len(historical_data) == 0:
        historical_data = df[df.index.month == month]
    
    # If still no data, use all historical data
    if len(historical_data) == 0:
        historical_data = df
    
    # Get the average PM2.5 value from historical data
    base_pm25 = historical_data['pm_2_5'].mean()
    std_pm25 = historical_data['pm_2_5'].std() if len(historical_data) > 1 else 5.0
    
    # Adjust for temperature difference (comparing to historical average)
    hist_temp_avg = historical_data['temperature'].mean()
    temp_factor = (temperature - hist_temp_avg) * 0.3  # Adjust this coefficient as needed
    
    # Adjust for humidity difference (comparing to historical average)
    hist_humidity_avg = historical_data['humidity'].mean()
    humidity_factor = (humidity - hist_humidity_avg) * -0.15  # Negative because higher humidity often reduces PM2.5
    
    # Random variation to make predictions more realistic
    random_factor = random.uniform(-1, 1) * std_pm25 * 0.5
    
    # Calculate final prediction
    predicted_value = base_pm25 + temp_factor + humidity_factor + random_factor
    
    # Ensure the value is within reasonable bounds
    predicted_value = max(5.0, min(250.0, predicted_value))
    
    return predicted_value




if __name__ == "__main__":
    app.run_server(debug=True)