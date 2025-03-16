from datetime import *
import pandas as pd
import ephem
import os
from pycaret.time_series import *
import pycaret.time_series as pt

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
def prepare_tide_features(target_date):
    """
    สร้างตารางข้อมูลคุณลักษณะสำหรับการทำนายระดับน้ำ
    โดยเริ่มจาก 72 ชั่วโมงก่อนวันที่ที่ระบุไปจนถึงวันปัจจุบันที่เวลา 00:00 น.
    
    Args:
        target_date (datetime): วันที่เป้าหมาย (จะใช้เวลา 00:00 น. ของวันนี้)
    
    Returns:
        pd.DataFrame: ตารางข้อมูลคุณลักษณะสำหรับ 72 ชั่วโมงย้อนหลัง
    """
    # ปรับให้เป็นเวลา 00:00 น. ของวันที่ระบุ
    current_date = datetime(target_date.year, target_date.month, target_date.day, 0, 0, 0)
    
    # สร้างรายการวันที่ย้อนหลัง 72 ชั่วโมง
    all_dates = [current_date - timedelta(hours=72-i) for i in range(72)]  # รวมเวลา 00:00 ของวันปัจจุบันด้วย
    
    # สร้างรายการข้อมูลคุณลักษณะ
    features_list = []
    
    for date in all_dates:
        features = {
            'day': date.day,
            'month': date.month,
            'year': date.year,
            'moon_phase': get_moon_phase(date),
            'full_moon_days': 1 if get_moon_phase(date) >= 0.98 else 0,
            'dark_moon_days': 1 if get_moon_phase(date) <= 0.02 else 0
        }
        
        # เพิ่มคุณลักษณะของฤดูกาล
        season = get_thai_season(date.month)
        features['season_rainy'] = 1 if season == 'rainy' else 0
        features['season_summer'] = 1 if season == 'summer' else 0
        features['season_winter'] = 1 if season == 'winter' else 0
        
        features_list.append(features)
    
    # สร้าง DataFrame จากรายการคุณลักษณะ
    features_df = pd.DataFrame(features_list, index=all_dates)
    features_df.index = features_df.index.to_period("H")  # หรือ "D" ถ้าเป็น daily
    features_df.to_csv("AAAAA.csv")
    return features_df
tide_model_path = 'models/Second_models'
try:
    if os.path.exists(tide_model_path+'.pkl'):
        tide_model = pt.load_model(tide_model_path)
        tide_model_loaded = True
        print('Tide model loaded successfully')
    else:
        tide_model_loaded = False
        print('Tide model not found')
except Exception as e:
    print(f"Error loading tide model: {e}")
    tide_model_loaded = False
A = prepare_tide_features(datetime.now() + timedelta(days=5*365))
print(predict_model(tide_model,fh=72,X=A.iloc[:72]))