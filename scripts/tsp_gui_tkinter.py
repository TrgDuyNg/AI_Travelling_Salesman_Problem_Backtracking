"""
Travelling Salesman Problem - GUI Application with Tkinter and Matplotlib
Giao diện GUI Tkinter để so sánh Backtracking và ACO
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import numpy as np
from tsp_backtracking import TSPBacktracking
from tsp_aco import TSP_ACO
import csv
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class TSPGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("TSP: Backtracking vs ACO")
        self.root.geometry("1000x700")
        self.root.resizable(False, False)
        
        self.cities = ['Ha Noi', 'Hai Phong', 'Da Nang', 'TP.HCM', 'Can Tho']
        self.coordinates = [
            (21.0285, 105.8542), (20.8449, 106.6881), (16.0544, 108.2022),
            (10.7769, 106.6964), (10.0379, 105.7869)
        ]
        self.distance_matrix = np.array([
            [0, 120, 764, 1710, 1840],
            [120, 0, 840, 1830, 1960],
            [764, 840, 0, 964, 1094],
            [1710, 1830, 964, 0, 169],
            [1840, 1960, 1094, 169, 0]
        ])
        
        self.result_backtracking = None
        self.result_aco = None
        self.normalized_coordinates = None
        self.aco_solver = None
        
        self.create_widgets()
    
    def create_widgets(self):
        """Create main interface"""
        main_container = ttk.Frame(self.root)
        main_container.pack(fill='both', expand=True)
        
        # Left sidebar - Input & Parameters
        left_panel = ttk.Frame(main_container, width=350)
        left_panel.pack(side='left', fill='both', padx=5, pady=5)
        left_panel.pack_propagate(False)
        
        # Right panel - Results
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side='right', fill='both', expand=True, padx=5, pady=5)
        
        self.create_left_panel(left_panel)
        self.create_right_panel(right_panel)
    
    def create_left_panel(self, parent):
        """Create left sidebar with data input"""
        notebook = ttk.Notebook(parent)
        notebook.pack(fill='both', expand=True)
        
        # Data Input Tab
        input_frame = ttk.Frame(notebook)
        notebook.add(input_frame, text='Nhap du lieu')
        self.create_input_tab(input_frame)
        
        # Parameters Tab
        params_frame = ttk.Frame(notebook)
        notebook.add(params_frame, text='Tham so')
        self.create_params_tab(params_frame)
    
    def create_input_tab(self, parent):
        """Create data input tab"""
        sub_notebook = ttk.Notebook(parent)
        sub_notebook.pack(fill='both', expand=True, padx=3, pady=3)
        
        # Default Tab
        default_tab = ttk.Frame(sub_notebook)
        sub_notebook.add(default_tab, text='Mac dinh')
        ttk.Button(default_tab, text='Su dung 5 thanh pho', 
                  command=self.use_default_cities).pack(pady=5)
        default_text = scrolledtext.ScrolledText(default_tab, height=8, width=35)
        default_text.pack(fill='both', expand=True, padx=3, pady=3)
        default_text.insert('end', self.format_cities())
        default_text.config(state='disabled')
        
        # Manual Tab
        manual_tab = ttk.Frame(sub_notebook)
        sub_notebook.add(manual_tab, text='Nhap tay')
        
        frame_manual = ttk.Frame(manual_tab)
        frame_manual.pack(fill='x', padx=3, pady=3)
        
        ttk.Label(frame_manual, text='Ten:', width=8).pack(side='left')
        self.entry_city_name = ttk.Entry(frame_manual, width=18)
        self.entry_city_name.pack(side='left', padx=2)
        
        ttk.Label(frame_manual, text='Lon:', width=6).pack(side='left')
        self.entry_lon = ttk.Entry(frame_manual, width=10)
        self.entry_lon.pack(side='left', padx=2)
        
        ttk.Label(frame_manual, text='Lat:', width=6).pack(side='left')
        self.entry_lat = ttk.Entry(frame_manual, width=10)
        self.entry_lat.pack(side='left', padx=2)
        
        btn_frame = ttk.Frame(manual_tab)
        btn_frame.pack(fill='x', padx=3, pady=3)
        ttk.Button(btn_frame, text='Them', command=self.add_city, width=8).pack(side='left', padx=1)
        ttk.Button(btn_frame, text='Xoa', command=self.remove_city, width=8).pack(side='left', padx=1)
        ttk.Button(btn_frame, text='Xoa tat', command=self.clear_cities, width=8).pack(side='left', padx=1)
        
        self.text_manual_cities = scrolledtext.ScrolledText(manual_tab, height=6, width=35)
        self.text_manual_cities.pack(fill='both', expand=True, padx=3, pady=3)
        
        # CSV Tab
        csv_tab = ttk.Frame(sub_notebook)
        sub_notebook.add(csv_tab, text='CSV')
        
        frame_csv = ttk.Frame(csv_tab)
        frame_csv.pack(fill='x', padx=3, pady=3)
        
        self.entry_csv = ttk.Entry(frame_csv, width=25)
        self.entry_csv.pack(side='left', padx=2)
        ttk.Button(frame_csv, text='Browse', command=self.browse_csv, width=8).pack(side='left', padx=1)
        ttk.Button(frame_csv, text='Load', command=self.load_csv, width=8).pack(side='left', padx=1)
        
        self.text_csv_cities = scrolledtext.ScrolledText(csv_tab, height=10, width=35)
        self.text_csv_cities.pack(fill='both', expand=True, padx=3, pady=3)
        
        # Random Tab
        random_tab = ttk.Frame(sub_notebook)
        sub_notebook.add(random_tab, text='Random')
        
        frame_random = ttk.Frame(random_tab)
        frame_random.pack(fill='x', padx=3, pady=3)
        
        ttk.Label(frame_random, text='N (5-15):', width=10).pack(side='left')
        self.spin_n_cities = ttk.Spinbox(frame_random, from_=5, to=15, width=8)
        self.spin_n_cities.set(10)
        self.spin_n_cities.pack(side='left', padx=2)
        ttk.Button(frame_random, text='Random', command=self.generate_random_cities, width=10).pack(side='left', padx=2)
        
        self.text_random_cities = scrolledtext.ScrolledText(random_tab, height=10, width=35)
        self.text_random_cities.pack(fill='both', expand=True, padx=3, pady=3)
    
    def create_params_tab(self, parent):
        """Create parameters tab"""
        aco_frame = ttk.LabelFrame(parent, text='Thong so ACO', padding=5)
        aco_frame.pack(fill='both', expand=True, padx=3, pady=3)
        
        params = [
            ('Kien (5-50):', 20, 5, 50),
            ('Iterations (10-200):', 50, 10, 200),
            ('Alpha (0.1-3.0):', 1.0, 0.1, 3.0),
            ('Beta (0.1-5.0):', 2.0, 0.1, 5.0),
            ('Evaporation (0.1-0.9):', 0.5, 0.1, 0.9),
            ('Q constant (10-500):', 100, 10, 500),
        ]
        
        self.param_spinboxes = {}
        for label, default, from_val, to_val in params:
            frame = ttk.Frame(aco_frame)
            frame.pack(fill='x', pady=2)
            ttk.Label(frame, text=label, width=18).pack(side='left')
            spinbox = ttk.Spinbox(frame, from_=from_val, to=to_val, width=12)
            spinbox.set(default)
            spinbox.pack(side='left', padx=2)
            self.param_spinboxes[label] = spinbox
        
        # Main action buttons
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill='x', padx=3, pady=5)
        
        ttk.Button(button_frame, text='Giai bai toan', 
                  command=self.solve_problem).pack(fill='x', pady=2)
        ttk.Button(button_frame, text='Xem bieu do', 
                  command=self.show_comparison_chart).pack(fill='x', pady=2)
        ttk.Button(button_frame, text='In chi tiet', 
                  command=self.print_details).pack(fill='x', pady=2)
    
    def create_right_panel(self, parent):
        """Create right panel for results"""
        notebook = ttk.Notebook(parent)
        notebook.pack(fill='both', expand=True)
        
        # Results Tab
        results_frame = ttk.Frame(notebook)
        notebook.add(results_frame, text='Ket qua')
        
        self.text_results = scrolledtext.ScrolledText(results_frame, height=25, width=60)
        self.text_results.pack(fill='both', expand=True, padx=3, pady=3)
        
        # Chart Tab
        self.chart_frame = ttk.Frame(notebook)
        notebook.add(self.chart_frame, text='Bieu do')
    
    def use_default_cities(self):
        """Load default cities"""
        self.cities = ['Ha Noi', 'Hai Phong', 'Da Nang', 'TP.HCM', 'Can Tho']
        self.coordinates = [
            (21.0285, 105.8542), (20.8449, 106.6881), (16.0544, 108.2022),
            (10.7769, 106.6964), (10.0379, 105.7869)
        ]
        self.normalize_coordinates()
        self.distance_matrix = self.calculate_distance_matrix()
        messagebox.showinfo('Thanh cong', f'Da load {len(self.cities)} thanh pho')
    
    def add_city(self):
        """Add city to the list"""
        name = self.entry_city_name.get()
        lon = self.entry_lon.get()
        lat = self.entry_lat.get()
        
        if not name or not lon or not lat:
            messagebox.showerror('Loi', 'Nhap day du thong tin!')
            return
        
        try:
            lon, lat = float(lon), float(lat)
            self.cities.append(name)
            self.coordinates.append((lat, lon))
            self.normalize_coordinates()
            self.distance_matrix = self.calculate_distance_matrix()
            
            self.entry_city_name.delete(0, 'end')
            self.entry_lon.delete(0, 'end')
            self.entry_lat.delete(0, 'end')
            
            self.update_manual_cities_display()
            messagebox.showinfo('Thanh cong', f'Da them {name}')
        except ValueError:
            messagebox.showerror('Loi', 'Toa do phai la so!')
    
    def remove_city(self):
        """Remove last city from the list"""
        if self.cities:
            removed = self.cities.pop()
            self.coordinates.pop()
            self.normalize_coordinates()
            if self.cities:
                self.distance_matrix = self.calculate_distance_matrix()
            self.update_manual_cities_display()
    
    def clear_cities(self):
        """Clear all cities from the list"""
        self.cities = []
        self.coordinates = []
        self.text_manual_cities.config(state='normal')
        self.text_manual_cities.delete('1.0', 'end')
        self.text_manual_cities.config(state='disabled')
    
    def browse_csv(self):
        """Browse for a CSV file"""
        file = filedialog.askopenfilename(filetypes=[('CSV', '*.csv')])
        if file:
            self.entry_csv.delete(0, 'end')
            self.entry_csv.insert(0, file)
    
    def load_csv(self):
        """Load cities from a CSV file"""
        file = self.entry_csv.get()
        if not file or not os.path.exists(file):
            messagebox.showerror('Loi', 'Chon file CSV hop le!')
            return
        
        try:
            self.cities = []
            self.coordinates = []
            
            with open(file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if len(row) >= 3:
                        name, lon, lat = row[0], float(row[1]), float(row[2])
                        self.cities.append(name)
                        self.coordinates.append((lat, lon))
            
            if len(self.cities) > 15:
                messagebox.showwarning('Canh bao', 
                    f'CSV co {len(self.cities)} thanh pho. Chi su dung toi da 15.')
                self.cities = self.cities[:15]
                self.coordinates = self.coordinates[:15]
            
            self.normalize_coordinates()
            self.distance_matrix = self.calculate_distance_matrix()
            self.update_csv_cities_display()
            messagebox.showinfo('Thanh cong', f'Da load {len(self.cities)} thanh pho')
        except Exception as e:
            messagebox.showerror('Loi', f'Loi load CSV: {str(e)}')
    
    def generate_random_cities(self):
        """Generate random cities"""
        n = int(self.spin_n_cities.get())
        self.cities = [f'Thanh pho {i+1}' for i in range(n)]
        self.coordinates = [(np.random.uniform(16, 22), np.random.uniform(103, 108)) for _ in range(n)]
        
        self.normalize_coordinates()
        self.distance_matrix = self.calculate_distance_matrix()
        
        text = f'Sinh {n} thanh pho ngau nhien:\n\n'
        for i, city in enumerate(self.cities):
            lat, lon = self.coordinates[i]
            text += f'{i+1}. {city}\n'
        
        self.text_random_cities.config(state='normal')
        self.text_random_cities.delete('1.0', 'end')
        self.text_random_cities.insert('end', text)
        self.text_random_cities.config(state='disabled')
        
        messagebox.showinfo('Thanh cong', f'Sinh {n} thanh pho')
    
    def update_manual_cities_display(self):
        """Update display for manually added cities"""
        text = f'Danh sach ({len(self.cities)} thanh pho):\n\n'
        for i, city in enumerate(self.cities):
            text += f'{i+1}. {city}\n'
        
        self.text_manual_cities.config(state='normal')
        self.text_manual_cities.delete('1.0', 'end')
        self.text_manual_cities.insert('end', text)
        self.text_manual_cities.config(state='disabled')
    
    def update_csv_cities_display(self):
        """Update display for cities loaded from CSV"""
        text = f'Tong {len(self.cities)} thanh pho:\n\n'
        for i, city in enumerate(self.cities[:10]):
            text += f'{i+1}. {city}\n'
        if len(self.cities) > 10:
            text += f'... va {len(self.cities)-10} thanh pho khac\n'
        
        self.text_csv_cities.config(state='normal')
        self.text_csv_cities.delete('1.0', 'end')
        self.text_csv_cities.insert('end', text)
        self.text_csv_cities.config(state='disabled')
    
    def format_cities(self):
        """Format the city list for display"""
        text = 'Danh sach 5 thanh pho:\n\n'
        for i, city in enumerate(self.cities):
            text += f'{i+1}. {city}\n'
        return text
    
    def normalize_coordinates(self):
        """Normalize coordinates to a [0, 100] range"""
        if not self.coordinates:
            return
        
        coords = np.array(self.coordinates)
        lat_min, lat_max = coords[:, 0].min(), coords[:, 0].max()
        lon_min, lon_max = coords[:, 1].min(), coords[:, 1].max()
        
        lat_range = lat_max - lat_min if lat_max > lat_min else 1
        lon_range = lon_max - lon_min if lon_max > lon_min else 1
        
        self.normalized_coordinates = [
            ((lat - lat_min) / lat_range * 100 if lat_range > 0 else 50,
             (lon - lon_min) / lon_range * 100 if lon_range > 0 else 50)
            for lat, lon in self.coordinates
        ]
    
    def calculate_distance_matrix(self):
        """Calculate the distance matrix between cities"""
        n = len(self.cities)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                lat1, lon1 = self.normalized_coordinates[i]
                lat2, lon2 = self.normalized_coordinates[j]
                dist = np.sqrt((lat2-lat1)**2 + (lon2-lon1)**2)
                matrix[i][j] = dist
                matrix[j][i] = dist
        
        return matrix
    
    def solve_problem(self):
        """Solve TSP using both algorithms"""
        if len(self.cities) < 3:
            messagebox.showerror('Loi', 'Can it nhat 3 thanh pho!')
            return
        
        if len(self.cities) > 15:
            messagebox.showerror('Loi', f'Toi da 15 thanh pho (hien co {len(self.cities)})')
            return
        
        # Get ACO parameters
        n_ants = int(self.param_spinboxes['Kien (5-50):'].get())
        n_iter = int(self.param_spinboxes['Iterations (10-200):'].get())
        alpha = float(self.param_spinboxes['Alpha (0.1-3.0):'].get())
        beta = float(self.param_spinboxes['Beta (0.1-5.0):'].get())
        evap = float(self.param_spinboxes['Evaporation (0.1-0.9):'].get())
        q = float(self.param_spinboxes['Q constant (10-500):'].get())
        
        self.text_results.config(state='normal')
        self.text_results.delete('1.0', 'end')
        self.text_results.insert('end', 'Dang giai bai toan...\n')
        self.text_results.config(state='disabled')
        self.root.update()
        
        # Solve with Backtracking
        bt_solver = TSPBacktracking(self.cities, self.distance_matrix)
        self.result_backtracking = bt_solver.solve(verbose=False)
        
        # Solve with ACO
        aco_solver = TSP_ACO(self.cities, self.distance_matrix,
                            n_ants=n_ants, n_iterations=n_iter,
                            alpha=alpha, beta=beta, evaporation_rate=evap, q=q)
        self.result_aco = aco_solver.solve(verbose=False)
        
        self.aco_solver = aco_solver
        
        self.display_results()
    
    def display_results(self):
        """Display results"""
        text = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
        text += "KET QUA SO SANH BACKTRACKING VA ACO\n"
        text += "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n\n"
        
        text += f"So thanh pho: {len(self.cities)}\n"
        text += f"Danh sach: {', '.join(self.cities[:3])}"
        if len(self.cities) > 3:
            text += f", ..."
        text += "\n\n"
        
        # Backtracking results
        text += "1. BACKTRACKING (Quay lui)\n"
        text += "=" * 60 + "\n"
        text += f"Tuyen duong: {' THEN '.join(self.result_backtracking['route'][:5])}"
        if len(self.result_backtracking['route']) > 5:
            text += " ..."
        text += "\n"
        text += f"Khoang cach: {self.result_backtracking['distance']:.2f}\n"
        text += f"Thoi gian: {self.result_backtracking['time']:.6f} giay\n"
        text += f"Pham vi: O(n!)\n\n"
        
        # ACO results
        text += "2. ACO (Ant Colony Optimization)\n"
        text += "=" * 60 + "\n"
        text += f"Tuyen duong: {' THEN '.join(self.result_aco['route'][:5])}"
        if len(self.result_aco['route']) > 5:
            text += " ..."
        text += "\n"
        text += f"Khoang cach: {self.result_aco['distance']:.2f}\n"
        text += f"Thoi gian: {self.result_aco['time']:.6f} giay\n"
        text += f"Pham vi: O(n^2 x m x iterations)\n\n"
        
        text += "THONG SO ACO:\n"
        text += f"  Kien: {self.aco_solver.n_ants}, Lap: {self.aco_solver.n_iterations}\n"
        text += f"  Alpha: {self.aco_solver.alpha}, Beta: {self.aco_solver.beta}\n"
        text += f"  Bay hoi: {self.aco_solver.evaporation_rate}, Q: {self.aco_solver.q}\n\n"
        
        # Comparison
        text += "3. SO SANH\n"
        text += "=" * 60 + "\n"
        
        dist_diff = abs(self.result_backtracking['distance'] - self.result_aco['distance'])
        if self.result_backtracking['distance'] > 0:
            percent_diff = (dist_diff / self.result_backtracking['distance']) * 100
        else:
            percent_diff = 0
        
        text += f"Khoang cach: BT={self.result_backtracking['distance']:.2f} / ACO={self.result_aco['distance']:.2f}\n"
        text += f"Chenh lech: {dist_diff:.2f} ({percent_diff:.1f}%)\n\n"
        
        time_diff = abs(self.result_backtracking['time'] - self.result_aco['time'])
        if self.result_backtracking['time'] > 0:
            time_percent = (time_diff / self.result_backtracking['time']) * 100
        else:
            time_percent = 0
        
        text += f"Thoi gian: BT={self.result_backtracking['time']:.6f}s / ACO={self.result_aco['time']:.6f}s\n"
        text += f"Chenh lech: {time_diff:.6f}s\n"
        
        self.text_results.config(state='normal')
        self.text_results.delete('1.0', 'end')
        self.text_results.insert('end', text)
        self.text_results.config(state='disabled')
    
    def show_comparison_chart(self):
        """Show comparison charts"""
        if not self.result_backtracking or not self.result_aco:
            messagebox.showerror('Loi', 'Giai bai toan truoc!')
            return
        
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        
        fig = Figure(figsize=(10, 5), dpi=100)
        
        # Chart 1: Distance comparison
        ax1 = fig.add_subplot(121)
        algorithms = ['Backtracking', 'ACO']
        distances = [self.result_backtracking['distance'], self.result_aco['distance']]
        colors = ['#FF6B6B', '#4ECDC4']
        bars = ax1.bar(algorithms, distances, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax1.set_ylabel('Khoang cach', fontsize=11)
        ax1.set_title('So sanh khoang cach', fontsize=11, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        for bar, dist in zip(bars, distances):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{dist:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Chart 2: ACO convergence
        ax2 = fig.add_subplot(122)
        iterations = range(1, len(self.aco_solver.convergence_data) + 1)
        ax2.plot(iterations, self.aco_solver.convergence_data, 'o-', 
                color='#4ECDC4', linewidth=2, markersize=4)
        ax2.axhline(y=self.result_backtracking['distance'], color='#FF6B6B', 
                   linestyle='--', linewidth=2, label=f'BT: {self.result_backtracking["distance"]:.1f}')
        ax2.set_xlabel('Lap', fontsize=11)
        ax2.set_ylabel('Khoang cach', fontsize=11)
        ax2.set_title('Qua trinh hoi tu ACO', fontsize=11, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def print_details(self):
        """Print detailed steps"""
        if not self.result_backtracking or not self.result_aco:
            messagebox.showerror('Loi', 'Giai bai toan truoc!')
            return
        
        output = "XXXX SO CHI TIET CUA CAC THUAT TOAN XXXX\n\n"
        
        output += "BACKTRACKING - CAN PHAT HIEN:\n"
        output += "=" * 70 + "\n"
        if self.result_backtracking['steps']:
            for step in self.result_backtracking['steps'][:30]:
                output += step + "\n"
            if len(self.result_backtracking['steps']) > 30:
                output += f"... va {len(self.result_backtracking['steps'])-30} buoc khac ...\n"
        
        output += "\n" + "=" * 70 + "\n"
        output += "ACO - CAC BUOC HOI TU:\n"
        output += "=" * 70 + "\n"
        if self.result_aco['steps']:
            for step in self.result_aco['steps']:
                output += step + "\n"
        
        detail_window = tk.Toplevel(self.root)
        detail_window.title("Chi tiet tung buoc")
        detail_window.geometry("900x500")
        
        detail_text = scrolledtext.ScrolledText(detail_window, height=30, width=110)
        detail_text.pack(fill='both', expand=True, padx=3, pady=3)
        detail_text.insert('end', output)
        detail_text.config(state='disabled')

def main():
    root = tk.Tk()
    gui = TSPGUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()