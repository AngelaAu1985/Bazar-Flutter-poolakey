#!/usr/bin/env python3
# pip install pandas numpy scikit-learn xgboost matplotlib seaborn plotly colorama joblib
"""
Poolakey Analyzer - تحلیل داده‌ها و آموزش مدل برای پرداخت‌های کافه‌بازار
بر اساس SDK FlutterPoolakey (Dart)
نسخه: 1.0.0 | آخرین به‌روزرسانی: 20 اکتبر 2025
"""

import pandas as pd
import numpy as np
import json
import os
import logging
import argparse
from datetime import datetime
from pathlib import Path

# مدل‌ها و ابزارها
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# نمودارها
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ذخیره‌سازی
import joblib
from colorama import init, Fore, Style

# تنظیمات
init(autoreset=True)  # رنگ در کنسول
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'

# تنظیم لاگ
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("poolakey_analysis.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# نقشه وضعیت خرید
STATE_MAP = {0: "خریداری‌شده", 1: "بازپرداخت", 2: "در حال انتظار", 3: "نامشخص"}
PRODUCT_MAP = {"premium": "پریمیوم", "pro": "حرفه‌ای", "basic": "پایه"}

# دایرکتوری خروجی
OUTPUT_DIR = Path("poolakey_analysis_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# -------------------------------
# 1. بارگذاری داده‌ها
# -------------------------------
def load_purchase_data(file_path=None):
    """بارگذاری داده‌ها از فایل یا تولید نمونه"""
    if file_path and os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        logger.info(f"داده‌ها از فایل {file_path} بارگذاری شد. تعداد رکوردها: {len(df)}")
    else:
        logger.info("فایل داده یافت نشد. تولید داده‌های نمونه...")
        np.random.seed(42)
        n = 500
        now = datetime.now()
        df = pd.DataFrame({
            'orderId': [f'ORD{i:05d}' for i in range(1, n+1)],
            'purchaseToken': [f'TOKEN{i:05d}' for i in range(1, n+1)],
            'payload': [f'payload_{i}' for i in range(1, n+1)],
            'packageName': ['com.example.app'] * n,
            'purchaseState': np.random.choice([0, 1, 2, 3], size=n, p=[0.7, 0.1, 0.15, 0.05]),
            'purchaseTime': [
                int((now - pd.Timedelta(days=np.random.randint(0, 60))).timestamp() * 1000)
                for _ in range(n)
            ],
            'productId': np.random.choice(['premium', 'pro', 'basic'], size=n, p=[0.4, 0.3, 0.3]),
            'originalJson': [json.dumps({'user': f'user_{i}'}) for i in range(1, n+1)],
            'dataSignature': [f'sig{i}' for i in range(1, n+1)]
        })
        logger.info(f"داده‌های نمونه تولید شد. تعداد رکوردها: {n}")

    df['purchaseTime'] = pd.to_datetime(df['purchaseTime'], unit='ms')
    df['purchaseState_str'] = df['purchaseState'].map(STATE_MAP)
    df['productId_fa'] = df['productId'].map(PRODUCT_MAP)
    return df

# -------------------------------
# 2. تحلیل داده‌ها
# -------------------------------
def analyze_data(df):
    logger.info("شروع تحلیل داده‌ها...")
    
    # 1. آمار کلی
    total_purchases = len(df)
    revenue_estimate = df[df['purchaseState'] == 0].shape[0] * 50000  # فرض: هر خرید 50,000 تومان
    refund_rate = (df['purchaseState'] == 1).mean() * 100

    summary = f"""
    تحلیل کلی:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    تعداد کل خریدها: {total_purchases:,}
    درآمد تخمینی: {revenue_estimate:,} تومان
    نرخ بازپرداخت: {refund_rate:.2f}%
    محصولات: {df['productId_fa'].unique().tolist()}
    دوره زمانی: {df['purchaseTime'].min().date()} تا {df['purchaseTime'].max().date()}
    """
    print(Fore.CYAN + summary)
    logger.info(summary.strip())

    # 2. توزیع وضعیت خرید
    fig1 = px.histogram(
        df, x='purchaseState_str', color='productId_fa',
        title="توزیع وضعیت خرید بر اساس محصول",
        labels={'purchaseState_str': 'وضعیت خرید', 'productId_fa': 'محصول'},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig1.update_layout(barmode='stack', xaxis={'categoryorder': 'total descending'})
    fig1.write_html(str(OUTPUT_DIR / "distribution.html"))
    logger.info("نمودار توزیع ذخیره شد: distribution.html")

    # 3. روند زمانی
    df_daily = df.set_index('purchaseTime').resample('D').size().reset_index()
    df_daily.columns = ['date', 'count']
    fig2 = px.line(df_daily, x='date', y='count', title="روند روزانه خریدها",
                   markers=True, line_shape='spline')
    fig2.update_layout(hovermode="x unified")
    fig2.write_html(str(OUTPUT_DIR / "trend.html"))
    logger.info("نمودار روند ذخیره شد: trend.html")

    # 4. نمودار ساعتی
    df['hour'] = df['purchaseTime'].dt.hour
    hourly = df.groupby('hour').size().reset_index(name='count')
    fig3 = px.bar(hourly, x='hour', y='count', title="توزیع خریدها در ساعات روز",
                  labels={'hour': 'ساعت', 'count': 'تعداد خرید'})
    fig3.write_html(str(OUTPUT_DIR / "hourly.html"))
    logger.info("نمودار ساعتی ذخیره شد: hourly.html")

    # 5. همبستگی
    corr = df[['purchaseState', 'hour', 'purchaseTime']].copy()
    corr['day'] = corr['purchaseTime'].dt.dayofweek
    corr_matrix = corr.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title("ماتریس همبستگی")
    plt.savefig(OUTPUT_DIR / "correlation.png", dpi=300, bbox_inches='tight')
    logger.info("ماتریس همبستگی ذخیره شد.")

# -------------------------------
# 3. مهندسی ویژگی
# -------------------------------
def feature_engineering(df):
    df = df.copy()
    df['hour'] = df['purchaseTime'].dt.hour
    df['day_of_week'] = df['purchaseTime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['month'] = df['purchaseTime'].dt.month
    df['is_night'] = df['hour'].between(0, 6).astype(int)
    df['time_since_first'] = (df['purchaseTime'] - df['purchaseTime'].min()).dt.total_seconds() / 3600

    # کدگذاری
    le = LabelEncoder()
    df['product_encoded'] = le.fit_transform(df['productId'])

    return df, le

# -------------------------------
# 4. آموزش مدل‌ها
# -------------------------------
def train_models(df):
    df_feat, le = feature_engineering(df)
    
    features = ['hour', 'day_of_week', 'is_weekend', 'month', 'is_night', 'product_encoded', 'time_since_first']
    X = df_feat[features]
    y = df_feat['purchaseState']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, class_weight='balanced'),
        'XGBoost': xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='mlogloss'),
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    }

    results = {}
    best_model = None
    best_acc = 0

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = {'model': model, 'accuracy': acc, 'predictions': y_pred}

        if acc > best_acc:
            best_acc = acc
            best_model = model

        logger.info(f"{name} - دقت: {acc:.4f}")

    # ذخیره بهترین مدل
    joblib.dump(best_model, OUTPUT_DIR / "best_model.pkl")
    joblib.dump(le, OUTPUT_DIR / "label_encoder.pkl")
    logger.info(f"بهترین مدل ذخیره شد با دقت: {best_acc:.4f}")

    # اهمیت ویژگی‌ها (فقط برای RF و XGB)
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        feat_imp = pd.DataFrame({'feature': features, 'importance': importances})
        feat_imp = feat_imp.sort_values('importance', ascending=False)

        fig = px.bar(feat_imp, x='importance', y='feature', orientation='h',
                     title="اهمیت ویژگی‌ها در پیش‌بینی وضعیت خرید")
        fig.write_html(str(OUTPUT_DIR / "feature_importance.html"))
        logger.info("اهمیت ویژگی‌ها ذخیره شد.")

    return results, X_test, y_test

# -------------------------------
# 5. گزارش نهایی
# -------------------------------
def generate_report(results, y_test):
    report = """
    گزارش نهایی تحلیل و مدل‌سازی
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    for name, res in results.items():
        acc = res['accuracy']
        report += f"{name}: دقت = {acc:.4f}\n"
    
    report += f"\nبهترین مدل ذخیره شد در: {OUTPUT_DIR}/best_model.pkl"
    print(Fore.GREEN + report)
    with open(OUTPUT_DIR / "summary.txt", "w", encoding='utf-8') as f:
        f.write(report)
    logger.info("گزارش نهایی ذخیره شد.")

# -------------------------------
# 6. تابع اصلی
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="تحلیل داده‌های Poolakey و آموزش مدل")
    parser.add_argument('--data', type=str, help='مسیر فایل JSON داده‌های خرید')
    parser.add_argument('--samples', type=int, default=500, help='تعداد نمونه در صورت عدم وجود فایل')
    args = parser.parse_args()

    print(Fore.MAGENTA + Style.BRIGHT + """
    Poolakey Analyzer
    تحلیل هوشمند پرداخت‌های کافه‌بازار
    """ + Style.RESET_ALL)

    try:
        df = load_purchase_data(args.data or None)
        analyze_data(df)
        results, X_test, y_test = train_models(df)
        generate_report(results, y_test)
        
        print(Fore.YELLOW + f"""
        خروجی‌ها در پوشه ذخیره شد:
        {OUTPUT_DIR.resolve()}
        شامل: نمودارها، مدل، گزارش
        """)
        
    except Exception as e:
        logger.error(f"خطا: {str(e)}")
        print(Fore.RED + f"خطا رخ داد: {e}")

if __name__ == "__main__":
    main()