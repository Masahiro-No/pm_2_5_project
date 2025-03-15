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
from plotly.subplots import make_subplots
import ephem
from pycaret.time_series import *

app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG, dbc.icons.FONT_AWESOME])
# โมเดลแรก
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
#โมเดลสอง
tide_model_path = 'models/Second_models.pkl'
try:
    if os.path.exists(tide_model_path):
        tide_model = joblib.load(tide_model_path)
        tide_model_loaded = True
        print('Tide model loaded successfully')
    else:
        tide_model_loaded = False
        print('Tide model not found')
except Exception as e:
    print(f"Error loading tide model: {e}")
    tide_model_loaded = False

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
def get_moon_phase(date):
    moon = ephem.Moon(date)
    phase = moon.phase / 100  # Adjust the value to be between 0-1
    return phase

def get_thai_season(month):
    if month in [3, 4, 5]:
        return 'summer'
    elif month in [6, 7, 8, 9, 10]:
        return 'rainy'
    else:
        return 'winter'

def prepare_tide_features(date):
    """Prepare features for tide prediction model"""
    features = {
        'day': date.day,
        'month': date.month,
        'year': date.year,
        'moon_phase': get_moon_phase(date),
        'full_moon_days': 1 if get_moon_phase(date) >= 0.98 else 0,
        'dark_moon_days': 1 if get_moon_phase(date) <= 0.02 else 0
    }
    
    # Add season features
    season = get_thai_season(date.month)
    features['season_rainy'] = 1 if season == 'rainy' else 0
    features['season_summer'] = 1 if season == 'summer' else 0
    features['season_winter'] = 1 if season == 'winter' else 0
    
    return features

# Create initial PM2.5 prediction figure
def create_prediction_figure():
    predicted_pm25 = 20
    
    # Create a figure with subplots - same structure as in update_prediction
    fig = make_subplots(
        rows=2, 
        cols=1,
        row_heights=[0.65, 0.35],
        specs=[[{"type": "indicator"}], [{"type": "xy"}]],
        vertical_spacing=0.15
    )
    
    # Add gauge chart to first row
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=predicted_pm25,
            domain={'y': [0, 1], 'x': [0.05, 0.95]},
            title={'text': f"การทำนาย PM2.5 สำหรับ 7 วันข้างหน้า"},
            gauge={
                'axis': {'range': [None, 250], 'tickwidth': 1},
                'bar': {'color': "yellow"},  # Default color for initial display
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 12], 'color': 'rgba(0,128,0,0.4)'},
                    {'range': [12, 35.4], 'color': 'rgba(255,255,0,0.4)'},
                    {'range': [35.4, 55.4], 'color': 'rgba(255,165,0,0.4)'},
                    {'range': [55.4, 150.4], 'color': 'rgba(255,0,0,0.4)'},
                    {'range': [150.4, 250.4], 'color': 'rgba(128,0,128,0.4)'}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': predicted_pm25
                }
            }
        ),
        row=1, col=1
    )
    
    # Create sample data for line chart
    start_date = datetime.today().date()
    daily_dates = [start_date + timedelta(days=i) for i in range(7)]
    daily_values = [20, 22, 25, 21, 18, 23, 20]  # Sample values for initial display
    
    # Add line chart of daily predictions to second row
    fig.add_trace(
        go.Scatter(
            x=[d.strftime('%d/%m') for d in daily_dates],
            y=daily_values,
            mode='lines+markers',
            name='PM2.5',
            line=dict(
                color='rgba(78, 115, 223, 1)',
                width=3,
                shape='spline',
                dash='solid'
            ),
            marker=dict(
                size=8,
                symbol='circle',
                color=daily_values,
                colorscale=[
                    [0, 'rgba(0,128,0,1)'],      # Green
                    [0.14, 'rgba(255,255,0,1)'],  # Yellow
                    [0.28, 'rgba(255,165,0,1)'],  # Orange
                    [0.42, 'rgba(255,0,0,1)'],    # Red
                    [0.56, 'rgba(128,0,128,1)']   # Purple
                ],
                line=dict(width=1, color='rgba(255, 255, 255, 0.8)')
            ),
            hovertemplate='วันที่: %{x}<br>PM2.5: %{y:.1f} μg/m³<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Add reference lines for air quality levels
    fig.add_shape(
        type="line",
        x0=0,
        y0=12,
        x1=1,
        y1=12,
        line=dict(color="rgba(0,128,0,1)", width=1, dash="dash"),
        row=2, col=1,
        xref="paper"
    )
    
    fig.add_shape(
        type="line",
        x0=0,
        y0=35.4,
        x1=1,
        y1=35.4,
        line=dict(color="rgba(255,255,0,1)", width=1, dash="dash"),
        row=2, col=1,
        xref="paper"
    )
    
    fig.add_shape(
        type="line",
        x0=0,
        y0=55.4,
        x1=1,
        y1=55.4,
        line=dict(color="rgba(255,165,0,1)", width=1, dash="dash"),
        row=2, col=1,
        xref="paper"
    )
    
    # Set template and colors
    template_name = "plotly_dark"
    bg_color = "rgba(0,0,0,0)"
    text_color = "#fff"
    
    fig.update_layout(
    template=template_name,
    paper_bgcolor=bg_color,
    plot_bgcolor=bg_color,
    font=dict(color=text_color, family="Kanit"),
    margin=dict(l=20, r=20, t=50, b=20),
    height=550,  # Slightly increase overall height
    showlegend=False
)
    
    # Update x-axis and y-axis of the line chart
    fig.update_xaxes(
        title_text="วันที่",
        gridcolor="rgba(255, 255, 255, 0.1)",
        row=2, col=1
    )
    
    fig.update_yaxes(
        title_text="PM2.5 (μg/m³)",
        gridcolor="rgba(255, 255, 255, 0.1)",
        row=2, col=1
    )
    
    return fig

prediction_fig = create_prediction_figure()

def create_tide_chart(start_date):
    """สร้างกราฟทำนายระดับน้ำขึ้น-น้ำลง 4 วัน (วันปัจจุบัน + 3 วันข้างหน้า)"""
    
    # สร้างวันที่สำหรับวันปัจจุบัน + 3 วันข้างหน้า
    dates = [start_date + timedelta(days=i) for i in range(4)]
    
    # สร้างการทำนายสำหรับแต่ละวัน (1 ค่าต่อวัน)
    tide_predictions = []
    
    for date in dates:
        features = prepare_tide_features(date)
        features_df = pd.DataFrame([features])
        # ตรวจสอบว่าโมเดลถูกโหลด และใช้ได้หรือไม่
        if  tide_model_loaded:
            try:
                # ทำนายระดับน้ำ
                prediction = predict_model(tide_model, features_df)
                tide_level = float(prediction['prediction_label'][0])
                print("สำเร็จละน้อง")
            except Exception as e:
                # ถ้าการทำนายล้มเหลว ให้ใช้ค่าสุ่ม
                print(f"Tide prediction failed: {e}")
                tide_level = random.uniform(0.5, 2.5)
        else:
            # สร้างข้อมูลระดับน้ำแบบสุ่มถ้าไม่มีโมเดล
            tide_level = random.uniform(0.5, 2.5)
            
        tide_predictions.append(tide_level)
    
    # สร้างกราฟ
    fig = go.Figure()
    
    # สีสำหรับเส้นการทำนาย
    color = 'rgba(0, 191, 255, 1)'
    
    # เพิ่มเส้นสำหรับการทำนาย
    fig.add_trace(go.Scatter(
        x=[d.strftime('%d/%m') for d in dates],
        y=tide_predictions,
        mode='lines+markers',
        name='ระดับน้ำ',
        line=dict(
            color=color,
            width=3,
            shape='spline'
        ),
        marker=dict(
            size=8,
            symbol='circle',
            color=color,
            line=dict(width=1, color='rgba(255, 255, 255, 0.8)')
        ),
        hovertemplate='วันที่: %{x}<br>ระดับน้ำ: %{y:.2f} เมตร<extra></extra>'
    ))
    
    # เพิ่มพื้นที่เฉดสีที่สวยงามใต้เส้นกราฟ
    fig.add_trace(go.Scatter(
        x=[d.strftime('%d/%m') for d in dates],
        y=[0] * len(dates),
        fill='tonexty',
        fillcolor=f'rgba(0, 191, 255, 0.1)',
        line=dict(color='rgba(0,0,0,0)'),
        hoverinfo="skip",
        showlegend=False
    ))
    
    # เพิ่มสัญลักษณ์เฟสของดวงจันทร์
    for i, date in enumerate(dates):
        moon_phase = get_moon_phase(date)
        if moon_phase >= 0.98:  # พระจันทร์เต็มดวง
            fig.add_trace(go.Scatter(
                x=[date.strftime('%d/%m')],
                y=[tide_predictions[i] + 0.2],  # เหนือเส้นที่สูงที่สุดเล็กน้อย
                mode='markers',
                marker=dict(
                    symbol='circle',
                    size=15,
                    color='rgba(255, 255, 200, 1)',
                    line=dict(color='rgba(255, 255, 0, 1)', width=1)
                ),
                name='พระจันทร์เต็มดวง',
                showlegend=False,
                hoverinfo='text',
                hovertext='พระจันทร์เต็มดวง'
            ))
        elif moon_phase <= 0.02:  # เดือนมืด
            fig.add_trace(go.Scatter(
                x=[date.strftime('%d/%m')],
                y=[tide_predictions[i] + 0.2],  # เหนือเส้นที่สูงที่สุดเล็กน้อย
                mode='markers',
                marker=dict(
                    symbol='circle',
                    size=15,
                    color='rgba(30, 30, 30, 1)',
                    line=dict(color='rgba(100, 100, 100, 1)', width=1)
                ),
                name='เดือนมืด',
                showlegend=False,
                hoverinfo='text',
                hovertext='เดือนมืด'
            ))
    
    # ปรับแต่งเลย์เอาต์
    fig.update_layout(
        title={
            'text': 'การพยากรณ์ระดับน้ำขึ้น-น้ำลง 3 วัน',
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        template='plotly_dark',
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#fff", family="Kanit"),
        margin=dict(l=20, r=20, t=50, b=20),
        height=350,
        xaxis=dict(
            title='วันที่',
            gridcolor="rgba(255, 255, 255, 0.1)",
            tickmode='array',
            tickvals=[d.strftime('%d/%m') for d in dates],
            ticktext=["วันนี้", "พรุ่งนี้", "2 วันถัดไป", "3 วันถัดไป"]
        ),
        yaxis=dict(
            title='ระดับน้ำ (เมตร)',
            gridcolor="rgba(255, 255, 255, 0.1)"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0.3)",
            bordercolor="rgba(255,255,255,0.2)",
            borderwidth=1
        )
    )
    
    # เพิ่มคำอธิบายสำหรับการทำนาย
    fig.add_annotation(
        x=0.02,
        y=0.95,
        xref="paper",
        yref="paper",
        text="<b>การพยากรณ์ระดับน้ำ</b>",
        showarrow=False,
        font=dict(color=color),
        bgcolor="rgba(0,0,0,0.5)",
        bordercolor=color,
        borderwidth=1,
        borderpad=4,
        align="left"
    )
    
    return fig


    
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
                    dcc.Graph(id="prediction-graph", figure=prediction_fig, className="mb-5"),
                    
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
    ]),
    dbc.Row([
    dbc.Col([
        dbc.Card([
            dbc.CardHeader([
                html.I(className="fas fa-water mr-2"),
                "การพยากรณ์ระดับน้ำขึ้น-น้ำลง 4 วัน",
                html.Span(id="moon-phase-indicator", className="float-end")
            ]),
            dbc.CardBody([
                dcc.Graph(id="tide-chart", className="mb-3"),
                html.Div([
                    html.Div([
                        html.Span("ระดับน้ำ", className="badge bg-info me-2"),
                    ], className="mb-2"),
                    html.Small([
                        html.I(className="fas fa-info-circle me-2"),
                        "วงกลมสว่าง = พระจันทร์เต็มดวง, วงกลมมืด = เดือนมืด"
                    ], className="text-muted")
                ], className="text-center")
            ])
        ], className="main-card mb-4")
    ])
        ])
], fluid=True, className="mt-3")

@callback(
    [Output("prediction-graph", "figure"),
     Output("model-status", "children"),
     Output("air-quality-status", "children"),
     Output("health-tips-content", "children"),
     Output("weather-indicator", "children"),
     Output("tide-chart", "figure"),
     Output("moon-phase-indicator", "children")],
    [Input("predict-button", "n_clicks")],
    [State("date-picker", "date"),
     State("temperature-input", "value"),
     State("humidity-input", "value")]  
)
def update_prediction(n_clicks, date, temperature, humidity):
    # เช็คว่ามีการกดปุ่มจริงๆ หรือไม่
    if n_clicks is None:
    # Default values when button hasn't been clicked
        default_weather = get_weather_icon(25, 60)
        default_air_status = html.Div([
            html.H4("กรุณากรอกข้อมูลและกดปุ่มพยากรณ์", className="text-secondary"),
        ])
        default_health_tips = [html.Li("กรุณากดปุ่มพยากรณ์เพื่อดูคำแนะนำสุขภาพ")]
        default_tide_fig = create_tide_chart(datetime.today())
        default_moon = html.I(className="fas fa-moon text-secondary")
        
        return (
            create_prediction_figure(), 
            "รอการทำนายจากโมเดล", 
            default_air_status,
            html.Ul(default_health_tips),
            default_weather,
            default_tide_fig,
            default_moon
        )
        
    # Parse date
    try:
        start_date = datetime.strptime(date, '%Y-%m-%d')
    except:
        # Use today's date if parsing fails
        start_date = datetime.today()

    tide_fig = create_tide_chart(start_date)
    moon_phase = get_moon_phase(start_date)
    if moon_phase >= 0.98:
        moon_indicator = html.I(className="fas fa-moon text-warning", title="พระจันทร์เต็มดวง")
    elif moon_phase <= 0.02:
        moon_indicator = html.I(className="fas fa-moon text-secondary", title="เดือนมืด")
    else:
        # Show appropriate moon phase icon
        if moon_phase < 0.5:
            moon_indicator = html.I(className="fas fa-moon text-secondary", title=f"ข้างแรม {int(moon_phase * 100)}%")
        else:
            moon_indicator = html.I(className="fas fa-moon text-warning", title=f"ข้างขึ้น {int(moon_phase * 100)}%")

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
    air_quality_color = None
    air_quality_icon = None
    air_quality_label = None
    air_quality_desc = None
    
    if predicted_value <= 12:
        air_quality_color = "green"
        air_quality_icon = "fa-smile"
        air_quality_label = "คุณภาพอากาศดี"
        air_quality_desc = "ปลอดภัยสำหรับทุกคน"
    elif predicted_value <= 35.4:
        air_quality_color = "yellow"
        air_quality_icon = "fa-meh"
        air_quality_label = "คุณภาพอากาศปานกลาง"
        air_quality_desc = "กลุ่มเสี่ยงควรระวัง"
    elif predicted_value <= 55.4:
        air_quality_color = "orange"
        air_quality_icon = "fa-frown"
        air_quality_label = "คุณภาพอากาศไม่ดีต่อกลุ่มเสี่ยง"
        air_quality_desc = "ผู้ที่มีโรคระบบทางเดินหายใจควรหลีกเลี่ยงกิจกรรมกลางแจ้ง"
    elif predicted_value <= 150.4:
        air_quality_color = "red"
        air_quality_icon = "fa-angry"
        air_quality_label = "คุณภาพอากาศไม่ดีต่อสุขภาพ"
        air_quality_desc = "ทุกคนควรลดกิจกรรมกลางแจ้ง"
    else:
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

    
    fig = make_subplots(
    rows=2, 
    cols=1,
    row_heights=[0.65, 0.35],  # Give a bit more height to the gauge
    specs=[[{"type": "indicator"}], [{"type": "xy"}]],
    vertical_spacing=0.2  # Increase spacing between subplots
)

    # Add gauge chart to first row
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=predicted_value,
            domain={'y': [0, 1], 'x': [0.05, 0.95]},
            title={
                'text': f"การทำนาย PM2.5 สำหรับวันที่ {display_future_date}",
                # Remove the 'y' property - it's not valid
                'font': {'size': 16},
                'align': 'center'
            },
            gauge={
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
                    {'range': [150.4, 250.4], 'color': 'rgba(128,0,128,0.4)'}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': predicted_value
                }
            }
        ),
        row=1, col=1
    )


    # Prepare data for line chart
    daily_dates = [start_date + timedelta(days=i) for i in range(7)]
    daily_values = daily_future_df['pm_2_5'].values

    # Add line chart of daily predictions to second row
    fig.add_trace(
        go.Scatter(
            x=[d.strftime('%d/%m') for d in daily_dates],
            y=daily_values,
            mode='lines+markers',
            name='PM2.5',
            line=dict(
                color='rgba(78, 115, 223, 1)',
                width=3,
                shape='spline',
                dash='solid'
            ),
            marker=dict(
                size=8,
                symbol='circle',
                color=daily_values,
                colorscale=[
                    [0, 'rgba(0,128,0,1)'],      # Green
                    [0.14, 'rgba(255,255,0,1)'],  # Yellow
                    [0.28, 'rgba(255,165,0,1)'],  # Orange
                    [0.42, 'rgba(255,0,0,1)'],    # Red
                    [0.56, 'rgba(128,0,128,1)']   # Purple
                ],
                line=dict(width=1, color='rgba(255, 255, 255, 0.8)')
            ),
            hovertemplate='วันที่: %{x}<br>PM2.5: %{y:.1f} μg/m³<extra></extra>'
        ),
        row=2, col=1
    )

    # Add reference lines for air quality levels
    fig.add_shape(
        type="line",
        x0=0,
        y0=12,
        x1=1,
        y1=12,
        line=dict(color="rgba(0,128,0,1)", width=1, dash="dash"),
        row=2, col=1,
        xref="paper"
    )

    fig.add_shape(
        type="line",
        x0=0,
        y0=35.4,
        x1=1,
        y1=35.4,
        line=dict(color="rgba(255,255,0,1)", width=1, dash="dash"),
        row=2, col=1,
        xref="paper"
    )

    fig.add_shape(
        type="line",
        x0=0,
        y0=55.4,
        x1=1,
        y1=55.4,
        line=dict(color="rgba(255,165,0,1)", width=1, dash="dash"),
        row=2, col=1,
        xref="paper"
    )

    # Use built-in plotly templates
    template_name = "plotly_dark"
    bg_color = "rgba(0,0,0,0)"
    text_color = "#fff"

    fig.update_layout(
    template=template_name,
    paper_bgcolor=bg_color,
    plot_bgcolor=bg_color,
    font=dict(color=text_color, family="Kanit"),
    margin=dict(l=20, r=20, t=50, b=20),  # Increase top margin (t) value
    height=550,  # Slightly increase overall height
    showlegend=False
)


    # Update x-axis and y-axis of the line chart
    fig.update_xaxes(
        title_text="วันที่",
        gridcolor="rgba(255, 255, 255, 0.1)",
        row=2, col=1
    )

    fig.update_yaxes(
        title_text="PM2.5 (μg/m³)",
        gridcolor="rgba(255, 255, 255, 0.1)",
        row=2, col=1
    )
    
    return (fig, model_status, air_quality_status, html.Ul(health_tips_elements), 
            weather_icon, tide_fig, moon_indicator)

# ฟังก์ชันช่วยสำหรับการประมาณค่าจากข้อมูลในอดีต
def get_estimated_value(df, date, column):
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




if __name__ == "__main__":
    app.run_server(debug=True)