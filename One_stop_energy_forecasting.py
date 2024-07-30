import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
import seaborn as sns

# 全局变量
PV_dataset = None

# 数据导入函数
def import_data():
    global PV_dataset
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx *.xls")])
    if file_path.endswith(".csv"):
        PV_dataset = pd.read_csv(file_path)
    else:
        PV_dataset = pd.read_excel(file_path)
    messagebox.showinfo("Data Import", "Data imported successfully!")

# 显示表头函数
def show_head():
    global PV_dataset
    if PV_dataset is not None:
        headers = PV_dataset.columns.tolist()
        text_widget.delete("1.0", tk.END)
        text_widget.insert(tk.END, str(headers))
    else:
        messagebox.showerror("Error", "No data imported")

# 数据清洗函数
def clean_data():
    global PV_dataset
    if PV_dataset is not None:
        PV_dataset.fillna(0, inplace=True)
        PV_dataset = PV_dataset.apply(lambda x: x.replace([np.inf, -np.inf], np.nan))
        PV_dataset.fillna(method='ffill', inplace=True)
        PV_dataset.fillna(method='bfill', inplace=True)
        plot_data()
    else:
        messagebox.showerror("Error", "No data imported")

# 绘图函数
def plot_data():
    global PV_dataset, plot_frame
    if PV_dataset is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        PV_dataset.plot(ax=ax)
        for widget in plot_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    else:
        messagebox.showerror("Error", "No data to plot")

#  互信息分析函数
def mutual_info():
    global PV_dataset, plot_frame
    if PV_dataset is not None:
        target = 'Power'  # 假设目标变量是'Power'
        X = PV_dataset.drop(columns=[target])
        y = PV_dataset[target]
        mi = mutual_info_regression(X, y)
        mi_series = pd.Series(mi, index=X.columns)
        fig, ax = plt.subplots(figsize=(10, 6))
        mi_series.sort_values(ascending=False).plot(kind='bar', ax=ax)
        ax.set_title("Mutual Information")
        for widget in plot_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    else:
        messagebox.showerror("Error", "No data imported")

# 相关性分析函数
def correlation_analysis():
    global PV_dataset, plot_frame
    if PV_dataset is not None:
        corr_matrix = PV_dataset.corr()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title("Correlation Matrix")
        for widget in plot_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    else:
        messagebox.showerror("Error", "No data imported")

# 特征工程函数
def feature_engineering():
    global PV_dataset
    if PV_dataset is not None:
        response = messagebox.askyesno("Feature Engineering", "Do you want to proceed with feature engineering?")
        if response:
            target = 'Power'  # 假设目标变量是'Power'
            X = PV_dataset.drop(columns=[target])
            y = PV_dataset[target]
            # 互信息和相关性分析
            mi = mutual_info_regression(X, y)
            corr = X.corrwith(y)
            # 去除关联最弱的2个变量
            weakest_features = mi.argsort()[:2]
            X = X.drop(columns=X.columns[weakest_features])
            # 主成分分析
            if X.shape[1] > 9:
                pca = PCA(n_components=6)
                X_pca = pca.fit_transform(X)
                X = pd.DataFrame(X_pca)
            # 数据归一化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            # 分割数据集
            X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, shuffle=False)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)
            messagebox.showinfo("Feature Engineering", "Processing complete!")
    else:
        messagebox.showerror("Error", "No data imported")

# PV二级界面函数
def show_pv_interface():
    pv_window = tk.Toplevel()
    pv_window.title("PV Data Processing")
    pv_window.geometry("1200x800")

    # 数据导入按钮
    tk.Button(pv_window, text="Import Data", command=import_data).pack(pady=10)

    # 显示表头按钮
    tk.Button(pv_window, text="Show Head", command=show_head).pack(pady=10)

    # 数据清洗按钮
    tk.Button(pv_window, text="Clean Data", command=clean_data).pack(pady=10)

    # 显示表头文本框
    global text_widget
    text_widget = tk.Text(pv_window, height=1, wrap='none')
    text_widget.pack(fill=tk.X, padx=10)
    x_scroll = tk.Scrollbar(pv_window, orient='horizontal', command=text_widget.xview)
    text_widget.configure(xscrollcommand=x_scroll.set)
    x_scroll.pack(fill=tk.X, padx=10)

    # Plot框
    global plot_frame
    plot_frame = tk.Frame(pv_window, width=800, height=400)
    plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # 底部按钮
    bottom_frame = tk.Frame(pv_window)
    bottom_frame.pack(fill=tk.X, padx=10, pady=10)

    tk.Button(bottom_frame, text="MI Analysis", command=mutual_info).pack(side=tk.LEFT, padx=5)
    tk.Button(bottom_frame, text="Correlation Analysis", command=correlation_analysis).pack(side=tk.LEFT, padx=5)
    tk.Button(bottom_frame, text="Feature Engineering", command=feature_engineering).pack(side=tk.LEFT, padx=5)

# 主界面函数
def show_main_interface():
    main = tk.Tk()
    main.title("Main Interface")
    main.geometry("600x400")
    main.grid_columnconfigure(0, weight=1)
    main.grid_columnconfigure(1, weight=1)
    main.grid_columnconfigure(2, weight=1)
    main.grid_columnconfigure(3, weight=1)
    main.grid_columnconfigure(4, weight=1)
    main.grid_rowconfigure(0, weight=1)
    main.grid_rowconfigure(1, weight=1)
    main.grid_rowconfigure(2, weight=1)
    main.grid_rowconfigure(3, weight=1)
    main.grid_rowconfigure(4, weight=1)

    tk.Label(main, text="Data Input").grid(row=0, column=1, columnspan=3, pady=10)

    # Data Input Buttons
    tk.Button(main, text="PV", command=show_pv_interface).grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
    tk.Button(main, text="Wind").grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
    tk.Button(main, text="Load").grid(row=1, column=2, padx=10, pady=10, sticky="nsew")
    tk.Button(main, text="Price").grid(row=1, column=3, padx=10, pady=10, sticky="nsew")

    tk.Label(main, text="Model Selection").grid(row=2, column=1, columnspan=3, pady=10)

    # Model Selection Buttons
    tk.Button(main, text="M1").grid(row=3, column=0, padx=10, pady=10, sticky="nsew")
    tk.Button(main, text="M2").grid(row=3, column=1, padx=10, pady=10, sticky="nsew")
    tk.Button(main, text="M3").grid(row=3, column=2, padx=10, pady=10, sticky="nsew")
    tk.Button(main, text="M4").grid(row=3, column=3, padx=10, pady=10, sticky="nsew")
    tk.Button(main, text="M5").grid(row=3, column=4, padx=10, pady=10, sticky="nsew")

    # Result Buttons
    tk.Button(main, text="View").grid(row=4, column=1, padx=10, pady=10, sticky="nsew")
    tk.Button(main, text="Open file").grid(row=4, column=3, padx=10, pady=10, sticky="nsew")

    main.mainloop()

# 登录界面函数
def show_login_interface():
    root = tk.Tk()
    root.title("Login")
    root.geometry("300x150")

    tk.Label(root, text="User").grid(row=0, column=0, padx=20, pady=10)
    tk.Label(root, text="Password").grid(row=1, column=0, padx=20, pady=10)

    entry_user = tk.Entry(root)
    entry_password = tk.Entry(root, show="*")
    entry_user.grid(row=0, column=1, padx=20, pady=10)
    entry_password.grid(row=1, column=1, padx=20, pady=10)

    def login():
        username = entry_user.get()
        password = entry_password.get()
        
        if username in users and users[username] == password:
            messagebox.showinfo("Login Status", "Login Successful!")
            root.destroy()
            show_main_interface()
        else:
            messagebox.showerror("Login Status", "Invalid Username or Password")

    tk.Button(root, text="Login", command=login).grid(row=2, column=1, pady=10)

    root.mainloop()

# 预设用户数据
users = {"chen": "123456", "user2": "password2"}

# 显示登录界面
show_login_interface()
