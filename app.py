# -*- coding: utf-8 -*-
import os
import io
import base64
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from flask import Flask, request, render_template, jsonify, session, send_file
from werkzeug.utils import secure_filename
from scipy.stats import pearsonr
from sklearn.svm import OneClassSVM
import matplotlib
import pdfkit
import sqlite3
from scipy.stats import chisquare

# 显式设置不使用 GUI 渲染引擎
matplotlib.use('Agg')

import sys

# 获取程序运行路径
if getattr(sys, 'frozen', False):
    # 如果是打包后的路径
    BASE_DIR = sys._MEIPASS
else:
    # 如果是普通的 python 运行路径
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__,
            template_folder=os.path.join(BASE_DIR, 'templates'),
            static_folder=os.path.join(BASE_DIR, 'static'))

EXTERNAL_DIR = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, 'frozen', False) else __file__))
app.config['UPLOAD_FOLDER'] = os.path.join(EXTERNAL_DIR, 'uploads')
app.config['DATA_FOLDER'] = os.path.join(EXTERNAL_DIR, 'data')
app.config['DB_PATH'] = os.path.join(EXTERNAL_DIR, 'history.db')


# ==========================================
# 1. 全局配置与环境初始化
# ==========================================

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font='SimHei')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'benford_secure_key_2026'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DATA_FOLDER'] = 'data'
app.config['DB_PATH'] = 'history.db'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DATA_FOLDER'], exist_ok=True)


# --- 数据库初始化逻辑 ---
def init_db():
    conn = sqlite3.connect(app.config['DB_PATH'])
    cursor = conn.cursor()
    # 1. 先确保表存在
    cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_time TEXT,
                filename TEXT,
                baseline_info TEXT,
                window_size INTEGER,
                abnormal_rate REAL,
                avg_confidence REAL,
                conclusion TEXT
            )
        ''')
    conn.commit()

    # 2. 强制检查并添加 nu_param 列 (这是核心补丁)
    try:
        cursor.execute('ALTER TABLE test_history ADD COLUMN nu_param REAL DEFAULT 0.05')
        conn.commit()
        print(">>> 数据库结构已自动修复：已添加 nu_param 列")
    except sqlite3.OperationalError:
        # 如果列已经存在，它会报这个错，直接跳过即可
        print(">>> 数据库结构检查完毕：nu_param 列已存在")

    conn.close()


init_db()

# --- PDF 导出环境配置 ---
path_wkhtmltopdf = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
pdf_config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)

analysis_progress = {
    'status': '准备就绪',
    'percent': 0
}


# ==========================================
# 2. 核心算法逻辑 (Benford + OCSVM)
# ==========================================

def improved_benford_test(data):
    data = np.asarray(data)
    data = data[data > 0]
    if len(data) < 10:
        return {'error': '数据量过小'}

    first_digits = [int(str(x).replace('.', '').lstrip('0')[0]) for x in data if x > 0]
    obs = np.array([first_digits.count(i) for i in range(1, 10)])
    if obs.sum() == 0:
        return {'error': '无有效数字'}
    obs = obs / obs.sum()
    benford = np.array([np.log10(1 + 1 / i) for i in range(1, 10)])

    chi2_stat = np.sum((obs - benford) ** 2 / benford)
    chi2_p = 1 - np.exp(-chi2_stat)
    mad = np.mean(np.abs(obs - benford))
    mad_conf = max(0, 1 - mad / 0.015)
    corr, _ = pearsonr(obs, benford)
    corr_conf = (corr + 1) / 2
    consistency = (mad_conf + corr_conf) / 2
    combined_conf = (consistency + (1 - chi2_p)) / 2

    return {
        'chi2_stat': chi2_stat, 'chi2_p': chi2_p, 'mad_stat': mad,
        'mad_confidence': mad_conf, 'correlation': corr,
        'corr_confidence': corr_conf, 'consistency_score': consistency,
        'combined_confidence': combined_conf
    }


def generate_benford_feature_matrix(data, window=40, step=1):
    X = []
    try:
        data = np.array(data, dtype=float)
    except:
        return np.array([])

    for i in range(0, len(data) - window + 1, step):
        res = improved_benford_test(data[i:i + window])
        if 'error' not in res:
            X.append([
                res['chi2_stat'], res['chi2_p'], res['mad_stat'],
                res['mad_confidence'], res['correlation'],
                res['corr_confidence'], res['consistency_score'],
                res['combined_confidence']
            ])
    return np.array(X)


def run_analysis_pipeline(train_values, test_values, window_size=40, nu=0.05):
    X_train = generate_benford_feature_matrix(train_values, window=window_size)
    X_test = generate_benford_feature_matrix(test_values, window=window_size)

    if len(X_train) < 5:
        raise ValueError("基准数据有效样本过少，无法训练模型。")
    if len(X_test) < 1:
        raise ValueError("待测数据有效数值不足。")

    model = OneClassSVM(kernel='rbf', nu=nu, gamma='scale')
    model.fit(X_train)

    train_scores = model.decision_function(X_train)
    test_scores = model.decision_function(X_test)
    threshold = np.percentile(train_scores, 5)

    cols = ['chi2', 'chi2_p', 'mad', 'mad_conf', 'corr', 'corr_conf', 'consistency', 'combined_conf']
    results_df = pd.DataFrame(X_test, columns=cols)

    results_df['start_row'] = range(1, len(results_df) + 1)
    results_df['end_row'] = results_df['start_row'] + window_size - 1
    results_df['OCSVM_Score'] = test_scores
    results_df['Is_Abnormal'] = (test_scores < threshold).astype(int)

    return results_df, threshold, train_scores


# ==========================================
# 3. 报告生成与可视化
# ==========================================

def create_report_visuals(results, threshold, train_scores):
    plt.figure(figsize=(15, 12))
    plt.subplot(2, 2, 1)
    plt.plot(results['OCSVM_Score'], color='#2c3e50', linewidth=1.5, label='检测窗口得分')
    plt.axhline(y=threshold, color='#e74c3c', linestyle='--', label=f'异常阈值 ({threshold:.2f})')
    anomalies = results[results['Is_Abnormal'] == 1]
    if not anomalies.empty:
        plt.scatter(anomalies.index, anomalies['OCSVM_Score'], color='#e74c3c', s=20, label='可疑异常段')
    plt.title("异常评分趋势图 (点击可疑点定位数据)", fontsize=14)
    plt.legend()

    plt.subplot(2, 2, 2)
    sns.kdeplot(train_scores, fill=True, color="#3498db", label="基准分布 (Baseline)")
    sns.kdeplot(results['OCSVM_Score'], fill=True, color="#f1c40f", label="待测数据 (Test Data)")
    plt.axvline(threshold, color='#e74c3c', linestyle='--')
    plt.title("数据分布特征重叠度分析", fontsize=14)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.hist(results['combined_conf'], bins=20, color='#27ae60', alpha=0.7, edgecolor='white')
    plt.title("本福特综合置信度分布", fontsize=14)

    plt.subplot(2, 2, 4)
    corr = results[['chi2', 'mad', 'corr', 'OCSVM_Score']].corr()
    sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0)
    plt.title("特征相关性热力图", fontsize=14)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode()


def generate_text_summary(results):
    total = len(results)
    abnormal = int(results['Is_Abnormal'].sum())
    rate = abnormal / total if total > 0 else 0
    avg_conf = results['combined_conf'].mean()

    # 核心结论逻辑
    if rate > 0.20:
        conclusion = "警告：数据存在严重的统计逻辑冲突，高度疑似人为构造或不当修改。"
        level = "danger"
    elif rate > 0.05:
        conclusion = "提示：部分数据段落偏离统计常数，建议核查原始实验记录。"
        level = "warning"
    else:
        conclusion = "结论：数据分布符合科学统计规律，未见明显修饰痕迹。"
        level = "success"

    return {
        "分析样本量": total,
        "检出异常窗口": abnormal,
        "异常偏离率": f"{rate:.2%}",
        "raw_rate": rate,  # 传出原始数值用于判断
        "基准符合度均值": f"{avg_conf:.4f}",
        "核心判定结论": conclusion,
        "risk_level": level
    }


# ==========================================
# 4. Flask 路由
# ==========================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/history')
def history():
    conn = sqlite3.connect(app.config['DB_PATH'])
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM test_history ORDER BY test_time DESC')
    records = cursor.fetchall()
    conn.close()
    return render_template('history.html', records=records)


@app.route('/guidelines')
def guidelines():
    rules = [
        {"id": "01", "title": "原始性准则 (Originality)", "content": "科研数据必须直接源于实验器材记录、系统日志或受控调查问卷。"},
        {"id": "02", "title": "完整性准则 (Integrity)", "content": "不得为了获得显著性差异而选择性剔除“不合群”的数据点。"},
        {"id": "03", "title": "可重复性准则 (Reproducibility)", "content": "数据转换逻辑需保持一致，本平台旨在辅助识别非自然干预的痕迹。"},
        {"id": "04", "title": "存储规范 (Traceability)", "content": "原始数据应至少保存 5-10 年以备学术复核。"}
    ]
    return render_template('guidelines.html', rules=rules)


@app.route('/algorithm')
def algorithm():
    algo_configs = {
        'benford': {'chi_critical': 15.51, 'mad_threshold': 0.015, 'confidence_weights': [0.4, 0.4, 0.2]},
        'ocsvm': {'kernel': 'RBF (Gaussian)', 'nu': 0.08, 'gamma': 'scale', 'complexity': 'O(n_samples^2)'}
    }
    return render_template('algorithm.html', config=algo_configs)


@app.route('/about')
def about():
    changelog = [
        {"version": "V1.6.0", "date": "2026-02-05", "desc": "新增测试历史数据库记录与对比管理模块。"},
        {"version": "V1.5.0", "date": "2026-02-04", "desc": "新增 PDF 学术鉴定报告导出功能与 OCSVM 算法参数优化。"},
        {"version": "V1.0.0", "date": "2025-12-20", "desc": "完成核心 Benford 统计引擎开发。"}
    ]
    tech_stack = {
        "Backend": "Python 3.9+ / Flask / SQLite3",
        "Machine_Learning": "Scikit-Learn (One-Class SVM)",
        "Data_Processing": "Pandas / NumPy / SciPy",
        "Visualization": "Matplotlib / Seaborn"
    }
    return render_template('about.html', changelog=changelog, tech_stack=tech_stack)


@app.route('/quick_check', methods=['POST'])
def quick_check():
    file = request.files.get('file')
    if not file: return jsonify({"status": "error", "message": "未找到文件"})
    try:
        df = pd.read_excel(file)
        num_col = df.select_dtypes(include=[np.number]).columns[0]
        data = df[num_col].dropna().values
        first_digits = [int(str(abs(x)).replace('.', '').lstrip('0')[0]) for x in data if abs(x) > 0]
        if len(first_digits) < 50:
            return jsonify({"status": "warning", "message": "样本量过小（需>50条），预检可能不准。"})
        counts = np.bincount(first_digits, minlength=10)[1:]
        actual_dist = counts / len(first_digits)
        expected_dist = np.log10(1 + 1 / np.arange(1, 10))
        mae = np.mean(np.abs(actual_dist - expected_dist))
        if mae > 0.05:
            return jsonify({"status": "alert", "message": f"预检提示：偏差较大 (MAE: {mae:.4f})，建议核查。"})
        return jsonify({"status": "success", "message": "初步校验：数据分布符合统计规律。"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/progress')
def get_progress():
    return jsonify(analysis_progress)


@app.route('/delete_history/<int:record_id>', methods=['POST'])
def delete_history(record_id):
    try:
        conn = sqlite3.connect(app.config['DB_PATH'])
        cursor = conn.cursor()
        # 执行删除指令
        cursor.execute('DELETE FROM test_history WHERE id = ?', (record_id,))
        conn.commit()
        conn.close()
        return jsonify({"status": "success", "message": "记录已删除"})
    except Exception as e:
        # 如果数据库操作失败，返回错误信息
        print(f"删除失败: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route('/export_report', methods=['POST'])
def export_report():
    try:
        data_json = request.form.get('results_json')
        file_name = request.form.get('file_name', 'analysis_report')
        df = pd.read_json(io.StringIO(data_json))
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='异常分析定位')
        output.seek(0)
        return send_file(output, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                         attachment_filename=f"Benford_Report_{file_name}.xlsx", as_attachment=True)
    except Exception as e:
        return f"导出失败: {str(e)}", 400


@app.route('/export_pdf', methods=['POST'])
def export_pdf():
    try:
        report_data = request.form.get('report_json')
        plot_url = request.form.get('plot_url')
        file_name = request.form.get('file_name', '未知文件')
        summary = json.loads(report_data)
        rendered = render_template('pdf_template.html', summary=summary, plot_url=plot_url, file_name=file_name,
                                   timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        pdf = pdfkit.from_string(rendered, False, configuration=pdf_config,
                                 options={'encoding': "UTF-8", 'enable-local-file-access': None})
        return send_file(io.BytesIO(pdf), mimetype='application/pdf', attachment_filename=f"检测报告_{file_name}.pdf",
                         as_attachment=True)
    except Exception as e:
        return f"PDF生成失败: {str(e)}", 400


@app.route('/analyze', methods=['POST'])
def analyze():
    global analysis_progress
    try:
        analysis_progress['status'] = '正在解析上传请求...'
        analysis_progress['percent'] = 5
        data_source = request.form.get('data_source')
        preset_type = request.form.get('preset_type')
        train_file = request.files.get('train_file')
        test_file = request.files.get('test_file')
        window_size = int(request.form.get('window_size', 40))
        nu_val = float(request.form.get('nu_param', 0.05))

        preset_files = {'wdi_amount': 'wdi_baseline.xlsx', 'gdp_growth': 'gdp_baseline.xlsx',
                        'covid_data': 'covid_baseline.xlsx'}
        train_df = None
        baseline_info = ""

        if data_source == 'default':
            target = preset_files.get(preset_type, 'wdi_baseline.xlsx')
            train_df = pd.read_excel(os.path.join(app.config['DATA_FOLDER'], target))
            baseline_info = f"行业预设: {preset_type}"
        elif data_source == 'custom':
            path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(train_file.filename))
            train_file.save(path)
            train_df = pd.read_excel(path)
            baseline_info = f"自定义: {train_file.filename}"
        elif data_source == 'mixed':
            baseline_info = "增强混合模式"
            dfs = [pd.read_excel(
                os.path.join(app.config['DATA_FOLDER'], preset_files.get(preset_type, 'wdi_baseline.xlsx')))]
            if train_file:
                path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(train_file.filename))
                train_file.save(path)
                dfs.append(pd.read_excel(path))
            train_df = pd.concat(dfs, ignore_index=True)

        test_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(test_file.filename))
        test_file.save(test_path)
        test_df = pd.read_excel(test_path)

        def extract_numeric(df):
            nums = df.select_dtypes(include=[np.number])
            if nums.empty: return None, None
            best_col = nums.var().idxmax()
            return nums[best_col].dropna().values, best_col

        train_vals, _ = extract_numeric(train_df)
        test_vals, col_name = extract_numeric(test_df)

        analysis_progress['status'] = 'OCSVM 核心计算中...'
        analysis_progress['percent'] = 50
        results, threshold, train_scores = run_analysis_pipeline(train_vals, test_vals, window_size, nu=nu_val)

        report_summary = generate_text_summary(results)
        plot_url = create_report_visuals(results, threshold, train_scores)

        # --- 保存到数据库 ---
        try:
            conn = sqlite3.connect(app.config['DB_PATH'])
            cursor = conn.cursor()
            cursor.execute('''INSERT INTO test_history 
                    (test_time, filename, baseline_info, window_size, abnormal_rate, avg_confidence, conclusion, nu_param) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                           (datetime.now().strftime("%Y-%m-%d %H:%M"),
                            test_file.filename,
                            baseline_info,
                            window_size,
                            float(results['Is_Abnormal'].sum() / len(results)),
                            float(results['combined_conf'].mean()),
                            report_summary['核心判定结论'],
                            nu_val))
            conn.commit()
            conn.close()
        except Exception as db_err:
            print(f"DB Error: {db_err}")

        analysis_progress['status'] = '完成'
        analysis_progress['percent'] = 100
        return render_template('result.html', report=report_summary, detailed_results=results.to_dict('records'),
                               plot_url=plot_url, total_windows=len(results), data_column=col_name)
    except Exception as e:
        analysis_progress['percent'] = 0
        return jsonify({'error': str(e)})


@app.route('/api/search_suggestions')
def search_suggestions():
    # 1. 静态词库（算法百科与指南）
    suggestions = [
        {"title": "本福特定律 (Benford's Law) 原理", "type": "algo", "url": "/algorithm"},
        {"title": "OCSVM 异常检测模型说明", "type": "algo", "url": "/algorithm"},
        {"title": "学术论文数据真实性准则", "type": "guide", "url": "/guidelines"},
        {"title": "卡方检验 (Chi-Square) 在检测中的应用", "type": "algo", "url": "/algorithm"},
        {"title": "如何解读分析报告", "type": "help", "url": "/about"}
    ]

    # 2. 动态词库（从数据库读取历史记录的文件名）
    try:
        import sqlite3
        conn = sqlite3.connect('history.db')  # 确保数据库文件名正确
        cursor = conn.cursor()
        # 提取最近的 15 条不同的检测记录文件名
        cursor.execute('SELECT DISTINCT filename FROM test_history ORDER BY id DESC LIMIT 15')
        history_files = cursor.fetchall()
        conn.close()

        for file in history_files:
            suggestions.append({
                "title": f"历史文件: {file[0]}",
                "type": "history",
                "url": "/history"
            })
    except Exception as e:
        print(f"搜索联想数据库读取失败: {e}")
        # 如果数据库报错，依然返回静态词库，保证搜索框不“死掉”

    return jsonify(suggestions)


if __name__ == '__main__':
    # 打印数据库路径，方便演示时找文件
    db_path = os.path.abspath(app.config['DB_PATH'])
    print(f">>> 数据库位置: {db_path}")

    # debug=False: 必须关闭，否则打包后会反复自启
    # use_reloader=False: 必须关闭，否则 PyInstaller 无法正常运行
    # host='0.0.0.0': 允许局域网访问（可选）
    app.run(host='127.0.0.1', port=5001, debug=False, use_reloader=False)