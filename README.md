# Benford-OCSVM-Detection-System
1. 项目简介

本项目旨在利用统计学规律与机器学习算法，自动识别并检测科研论文中的伪造实验数据。通过结合 Benford's Law（本福特法则） 的统计特性与 OCSVM（一类支持向量机） 的异常检测能力，实现对论文数据真实性的初步筛查。

2. 核心功能

数据清洗与预处理： 支持医学/生物数据集（如 MIMIC-3）的自动化 ETL 流程，完成特征提取与标准化。

多维度异常检测： 融合 Benford 统计检验与 OCSVM 算法，实现数据真实性评估。

可视化部署： 基于 Flask 搭建后端分析平台，支持本地化封装（.exe/.jar），便于非开发人员便捷使用。

3. 技术栈

核心语言： Python

机器学习/统计： OCSVM, Benford's Law

后端框架： Flask

部署工具： PyInstaller / Launch4j

5. 如何运行

环境依赖：

pip install -r requirements.txt

启动服务：

python app.py

访问界面：

在浏览器中打开 http://127.0.0.1:5000 即可使用。

5. 项目成果
本项目已通过大学生研究训练计划（SRT）国家级初评。
实现了算法逻辑与业务功能的工程化闭环。
