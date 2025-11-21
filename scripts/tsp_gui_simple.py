import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import csv
import numpy as np
import math
from tsp_backtracking import TSPBacktracking
from tsp_aco import TSP_ACO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class TSPSimpleGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("TSP Solver: Backtracking vs ACO")
        self.root.geometry("1200x650")
        self.root.resizable(True, True)
        
        # Default data
        self.cities = ['Ha Noi', 'Hai Phong', 'Da Nang', 'TP.HCM', 'Can Tho']
        self.coordinates = [
            (21.0285, 105.8542), (20.8449, 106.6881), (16.0544, 108.2022),
            (10.7769, 106.6964), (10.0379, 105.7869)
        ]
        self.distance_matrix = self.calculate_distance_matrix()
        
        self.result_backtracking = None
        self.result_aco = None
        self.aco_solver = None
        
        self.create_ui()
    
    def create_ui(self):
        """Create simple and clean UI"""
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_container, text="Travelling Salesman Problem Solver", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)
        
        # Content area with 2 columns
        content = ttk.Frame(main_container)
        content.pack(fill='both', expand=True)
        
        # LEFT PANEL - Input & Config
        left_panel = self._create_left_panel(content)
        left_panel.pack(side='left', fill='both', padx=(0, 10))
        
        # RIGHT PANEL - Results
        right_panel = self._create_right_panel(content)
        right_panel.pack(side='right', fill='both', expand=True)
    
    def _create_left_panel(self, parent):
        """Create left panel for input"""
        left_frame = ttk.LabelFrame(parent, text='INPUT DATA & SOLVE', padding=10)
        
        # Section 1: City List
        ttk.Label(left_frame, text="Cities:", font=('Arial', 10, 'bold')).pack(anchor='w', pady=(10, 5))
        
        cities_frame = ttk.Frame(left_frame)
        cities_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        scrollbar = ttk.Scrollbar(cities_frame)
        scrollbar.pack(side='right', fill='y')
        
        self.text_cities = scrolledtext.ScrolledText(cities_frame, height=8, width=35, 
                                                     yscrollcommand=scrollbar.set)
        self.text_cities.pack(fill='both', expand=True)
        scrollbar.config(command=self.text_cities.yview)
        
        # Display default cities
        self._update_cities_display()
        
        # Section 2: Add/Remove Cities
        ttk.Label(left_frame, text="Add City:", font=('Arial', 9, 'bold')).pack(anchor='w', pady=(10, 5))
        
        input_frame = ttk.Frame(left_frame)
        input_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(input_frame, text='Name:', width=6).pack(side='left')
        self.entry_city_name = ttk.Entry(input_frame, width=15)
        self.entry_city_name.pack(side='left', padx=2)
        
        ttk.Label(input_frame, text='Lat:', width=4).pack(side='left')
        self.entry_lat = ttk.Entry(input_frame, width=10)
        self.entry_lat.pack(side='left', padx=2)
        
        ttk.Label(input_frame, text='Lon:', width=4).pack(side='left')
        self.entry_lon = ttk.Entry(input_frame, width=10)
        self.entry_lon.pack(side='left', padx=2)
        
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(btn_frame, text='Add', command=self.add_city, width=8).pack(side='left', padx=2)
        ttk.Button(btn_frame, text='Remove', command=self.remove_city, width=8).pack(side='left', padx=2)
        ttk.Button(btn_frame, text='Reset', command=self.reset_cities, width=8).pack(side='left', padx=2)

        # Import CSV and Random generation controls
        io_frame = ttk.Frame(left_frame)
        io_frame.pack(fill='x', padx=5, pady=(6,4))
        ttk.Button(io_frame, text='Import CSV', command=self.import_csv).pack(side='left', padx=2)
        ttk.Label(io_frame, text='  Or generate random:').pack(side='left', padx=(8,4))
        self.spin_random_n = ttk.Spinbox(io_frame, from_=3, to=15, width=5)
        self.spin_random_n.set(5)
        self.spin_random_n.pack(side='left')
        ttk.Label(io_frame, text='count').pack(side='left', padx=(4,6))
        ttk.Label(io_frame, text='Seed:').pack(side='left')
        self.entry_rand_seed = ttk.Entry(io_frame, width=8)
        self.entry_rand_seed.pack(side='left', padx=(4,6))
        ttk.Button(io_frame, text='Randomize', command=self.randomize).pack(side='left')
        
        # Section 3: ACO Parameters
        ttk.Label(left_frame, text="ACO Parameters:", font=('Arial', 9, 'bold')).pack(anchor='w', pady=(10, 5))
        
        params_frame = ttk.Frame(left_frame)
        params_frame.pack(fill='x', padx=5, pady=5)

        # split parameters into two columns for a more compact layout
        left_col = ttk.Frame(params_frame)
        right_col = ttk.Frame(params_frame)
        left_col.pack(side='left', fill='both', expand=True, padx=(0,6))
        right_col.pack(side='left', fill='both', expand=True)

        self.param_spinboxes = {}
        params = [
            ('Ants:', 20, 5, 100),
            ('Iterations:', 50, 10, 200),
            ('Alpha:', 1.0, 0.1, 3.0),
            ('Beta:', 2.0, 0.1, 5.0),
            ('Evaporation rate:', 0.5, 0.1, 0.9),
            ('Q:', 100, 10, 500),
        ]

        mid = (len(params) + 1) // 2
        left_params = params[:mid]
        right_params = params[mid:]

        def make_param_row(parent, label, default, from_val, to_val):
            row = ttk.Frame(parent)
            row.pack(fill='x', pady=2)
            ttk.Label(row, text=label, width=14).pack(side='left')
            spinbox = ttk.Spinbox(row, from_=from_val, to=to_val, width=10)
            spinbox.set(default)
            spinbox.pack(side='left', padx=2)
            self.param_spinboxes[label] = spinbox

        for label, default, from_val, to_val in left_params:
            make_param_row(left_col, label, default, from_val, to_val)
        for label, default, from_val, to_val in right_params:
            make_param_row(right_col, label, default, from_val, to_val)
        
        # Section 4: Solve Button
        # Sample dataset selector and prominent Solve button
        sample_frame = ttk.Frame(left_frame)
        sample_frame.pack(fill='x', padx=5, pady=(8,4))
        ttk.Label(sample_frame, text='Sample dataset:').pack(side='left')
        self.sample_var = tk.StringVar(value='5')
        self.sample_combo = ttk.Combobox(sample_frame, textvariable=self.sample_var, values=['5','10','15','20'], width=6, state='readonly')
        self.sample_combo.pack(side='left', padx=(6,8))
        ttk.Button(sample_frame, text='Load Sample', command=lambda: self.load_sample(int(self.sample_var.get()))).pack(side='left')

        solve_btn = ttk.Button(left_frame, text='SOLVE PROBLEM', command=self.solve)
        solve_btn.pack(fill='x', padx=5, pady=16)
        # make the button visually larger by configuring its font via tk
        try:
            import tkinter.font as tkfont
            f = tkfont.Font(solve_btn, solve_btn.cget('font'))
            f.configure(size=12, weight='bold')
            solve_btn.configure(style=None)
            solve_btn.configure(font=f)
        except Exception:
            pass

        # Extra controls: View charts and Save details
        extra_frame = ttk.Frame(left_frame)
        extra_frame.pack(fill='x', padx=5, pady=(6,8))
        ttk.Button(extra_frame, text='View Charts', command=self.view_charts).pack(side='left', padx=2)
        ttk.Button(extra_frame, text='Save Details', command=self.save_details).pack(side='left', padx=6)
        
        # Status
        self.status_label = ttk.Label(left_frame, text="Ready", relief='sunken')
        self.status_label.pack(fill='x', padx=5, pady=5)
        
        return left_frame
    
    def _create_right_panel(self, parent):
        """Create right panel for results"""
        right_frame = ttk.LabelFrame(parent, text='RESULTS', padding=10)
        
        # Create notebook for Results and Charts tabs
        notebook = ttk.Notebook(right_frame)
        notebook.pack(fill='both', expand=True)
        
        # Results Tab
        results_frame = ttk.Frame(notebook)
        notebook.add(results_frame, text='Results')
        
        self.text_results = scrolledtext.ScrolledText(results_frame, height=28, width=70, 
                                                      font=('Courier', 9))
        self.text_results.pack(fill='both', expand=True)
        
        # Initial message
        self.text_results.insert('end', "Press 'SOLVE PROBLEM' to start\n\n")
        self.text_results.insert('end', "Results will be displayed here:\n")
        self.text_results.insert('end', "- Best route found\n")
        self.text_results.insert('end', "- Total distance\n")
        self.text_results.insert('end', "- Execution time\n")
        self.text_results.insert('end', "- Comparison between algorithms\n")
        self.text_results.config(state='disabled')
        
        # Charts Tab
        self.charts_frame = ttk.Frame(notebook)
        notebook.add(self.charts_frame, text='Charts')
        # keep reference for View Charts button
        self.notebook = notebook
        
        return right_frame
    
    def _update_cities_display(self):
        """Update cities display"""
        self.text_cities.config(state='normal')
        self.text_cities.delete('1.0', 'end')
        self.text_cities.insert('end', f'Total: {len(self.cities)} cities\n\n')
        for i, city in enumerate(self.cities, 1):
            self.text_cities.insert('end', f'{i}. {city}\n')
        self.text_cities.config(state='disabled')
    
    def add_city(self):
        """Add city to the list"""
        name = self.entry_city_name.get().strip()
        lat_str = self.entry_lat.get().strip()
        lon_str = self.entry_lon.get().strip()
        
        if not name or not lat_str or not lon_str:
            messagebox.showerror('Error', 'Please fill all fields!')
            return
        
        try:
            lat, lon = float(lat_str), float(lon_str)
            if len(self.cities) >= 15:
                messagebox.showwarning('Warning', 'Maximum 15 cities allowed!')
                return
            
            self.cities.append(name)
            self.coordinates.append((lat, lon))
            self.distance_matrix = self.calculate_distance_matrix()
            
            self.entry_city_name.delete(0, 'end')
            self.entry_lat.delete(0, 'end')
            self.entry_lon.delete(0, 'end')
            
            self._update_cities_display()
            self.status_label.config(text=f"Added '{name}' ({len(self.cities)} cities)")
        except ValueError:
            messagebox.showerror('Error', 'Coordinates must be numbers!')
    
    def remove_city(self):
        """Remove last city"""
        if len(self.cities) <= 3:
            messagebox.showwarning('Warning', 'Keep at least 3 cities!')
            return
        
        removed = self.cities.pop()
        self.coordinates.pop()
        self.distance_matrix = self.calculate_distance_matrix()
        self._update_cities_display()
        self.status_label.config(text=f"Removed '{removed}' ({len(self.cities)} cities)")
    
    def reset_cities(self):
        """Reset to default cities"""
        self.cities = ['Ha Noi', 'Hai Phong', 'Da Nang', 'TP.HCM', 'Can Tho']
        self.coordinates = [
            (21.0285, 105.8542), (20.8449, 106.6881), (16.0544, 108.2022),
            (10.7769, 106.6964), (10.0379, 105.7869)
        ]
        self.distance_matrix = self.calculate_distance_matrix()
        self._update_cities_display()
        self.status_label.config(text="Reset to default cities")

    def import_csv(self):
        path = filedialog.askopenfilename(filetypes=[('CSV files','*.csv'), ('All files','*.*')])
        if not path:
            return
        try:
            with open(path, newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                rows = [r for r in reader if any(c.strip() for c in r)]
                if not rows:
                    messagebox.showerror('Error', 'CSV is empty')
                    return
                header = [h.strip().lower() for h in rows[0]]
                data_rows = rows[1:]
                names = []
                coords = []
                # try to detect lat/lon columns
                lat_idx = None; lon_idx = None; name_idx = None
                for i, h in enumerate(header):
                    if h in ('lat','latitude'):
                        lat_idx = i
                    if h in ('lon','lng','longitude'):
                        lon_idx = i
                    if h in ('name','city'):
                        name_idx = i
                if lat_idx is None or lon_idx is None:
                    # assume format name,longitude,latitude per docs
                    name_idx = 0
                    lon_idx = 1
                    lat_idx = 2

                for r in data_rows:
                    try:
                        nm = r[name_idx].strip() if name_idx < len(r) else f'C{len(names)}'
                        lat = float(r[lat_idx])
                        lon = float(r[lon_idx])
                        names.append(nm)
                        coords.append((lat, lon))
                    except Exception:
                        continue
            if not coords:
                messagebox.showerror('Error', 'No valid coordinates found in CSV')
                return
            self.cities = names
            self.coordinates = coords
            self.distance_matrix = self.calculate_distance_matrix()
            self._update_cities_display()
            self.status_label.config(text=f'Imported {len(self.cities)} cities from CSV')
        except Exception as e:
            messagebox.showerror('Error', f'Failed to import CSV: {e}')

    def load_sample(self, n=5):
        # Load a sample CSV from the bundled samples folder
        import os
        base = os.path.dirname(__file__)
        samp_dir = os.path.join(base, 'samples')
        fname = os.path.join(samp_dir, f'sample_{n}.csv')
        if not os.path.exists(fname):
            messagebox.showerror('Error', f'Sample file not found: {fname}')
            return
        try:
            with open(fname, newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = [r for r in reader if any(c.strip() for c in r)]
                if not rows:
                    messagebox.showerror('Error', 'Sample CSV is empty')
                    return
                header = [h.strip().lower() for h in rows[0]]
                data_rows = rows[1:]
                names = []
                coords = []
                # assume header name,longitude,latitude
                lat_idx = None; lon_idx = None; name_idx = None
                for i, h in enumerate(header):
                    if h in ('lat','latitude'):
                        lat_idx = i
                    if h in ('lon','lng','longitude'):
                        lon_idx = i
                    if h in ('name','city'):
                        name_idx = i
                if lat_idx is None or lon_idx is None:
                    # fallback to columns 1=longitude,2=latitude
                    name_idx = 0; lon_idx = 1; lat_idx = 2
                for r in data_rows:
                    try:
                        nm = r[name_idx].strip() if name_idx < len(r) else f'C{len(names)}'
                        lon = float(r[lon_idx]); lat = float(r[lat_idx])
                        names.append(nm); coords.append((lat, lon))
                    except Exception:
                        continue
            if not coords:
                messagebox.showerror('Error', 'No valid coordinates in sample CSV')
                return
            self.cities = names
            self.coordinates = coords
            self.distance_matrix = self.calculate_distance_matrix()
            self._update_cities_display()
            self.status_label.config(text=f'Loaded sample dataset with {n} cities')
        except Exception as e:
            messagebox.showerror('Error', f'Failed to load sample: {e}')

    def randomize(self):
        try:
            n = int(self.spin_random_n.get())
        except Exception:
            n = 5
        seed = self.entry_rand_seed.get().strip()
        import random as _rnd
        if seed:
            try:
                _rnd.seed(int(seed))
            except Exception:
                _rnd.seed(seed)
        else:
            _rnd.seed()
        names = [f'C{i}' for i in range(n)]
        coords = []
        for i in range(n):
            lat = _rnd.uniform(-90, 90)
            lon = _rnd.uniform(-180, 180)
            coords.append((lat, lon))
        self.cities = names
        self.coordinates = coords
        self.distance_matrix = self.calculate_distance_matrix()
        self._update_cities_display()
        self.status_label.config(text=f'Generated {n} random cities')
    
    def calculate_distance_matrix(self):
        """Calculate Euclidean distance matrix"""
        n = len(self.cities)
        matrix = np.zeros((n, n))
        
        coords = np.array(self.coordinates)
        lat_min, lat_max = coords[:, 0].min(), coords[:, 0].max()
        lon_min, lon_max = coords[:, 1].min(), coords[:, 1].max()
        
        lat_range = lat_max - lat_min if lat_max > lat_min else 1
        lon_range = lon_max - lon_min if lon_max > lon_min else 1
        
        # Normalize coordinates
        normalized = [
            ((lat - lat_min) / lat_range * 100 if lat_range > 0 else 50,
             (lon - lon_min) / lon_range * 100 if lon_range > 0 else 50)
            for lat, lon in self.coordinates
        ]
        
        # Calculate distances
        for i in range(n):
            for j in range(i + 1, n):
                lat1, lon1 = normalized[i]
                lat2, lon2 = normalized[j]
                dist = np.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)
                matrix[i][j] = dist
                matrix[j][i] = dist
        
        return matrix
    
    def solve(self):
        """Solve TSP with both algorithms"""
        if len(self.cities) < 3:
            messagebox.showerror('Error', 'Need at least 3 cities!')
            return
        
        self.status_label.config(text="Solving... Please wait...")
        self.root.update()
        
        try:
            # Get ACO parameters
            n_ants = int(self.param_spinboxes['Ants:'].get())
            n_iter = int(self.param_spinboxes['Iterations:'].get())
            alpha = float(self.param_spinboxes['Alpha:'].get())
            beta = float(self.param_spinboxes['Beta:'].get())
            
            # Solve with Backtracking
            bt_solver = TSPBacktracking(self.cities, self.distance_matrix)
            self.result_backtracking = bt_solver.solve(verbose=False)
            
            # Solve with ACO (use UI params including evaporation and Q)
            try:
                evaporation = float(self.param_spinboxes['Evaporation rate:'].get())
            except Exception:
                evaporation = 0.5
            try:
                q_const = float(self.param_spinboxes['Q:'].get())
            except Exception:
                q_const = 100

            aco_solver = TSP_ACO(self.cities, self.distance_matrix,
                                n_ants=n_ants, n_iterations=n_iter,
                                alpha=alpha, beta=beta, 
                                evaporation_rate=evaporation, q=q_const)
            self.result_aco = aco_solver.solve(verbose=False)
            self.aco_solver = aco_solver
            
            self._display_results()
            self.status_label.config(text="✓ Solved successfully")
        except Exception as e:
            messagebox.showerror('Error', f'Solve failed: {str(e)}')
            self.status_label.config(text="✗ Error during solving")
    
    def _display_results(self):
        """Display results in result panel"""
        bt = self.result_backtracking
        aco = self.result_aco
        
        # Format output
        output = f"""
╔════════════════════════════════════════════════════════════════════╗
║                 TRAVELLING SALESMAN PROBLEM RESULTS                 ║
╚════════════════════════════════════════════════════════════════════╝

     PROBLEM DATA
─────────────────────────────────────────────────────────────────────
Cities ({len(self.cities)}): {', '.join(self.cities)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    BACKTRACKING (Exhaustive Search)
─────────────────────────────────────────────────────────────────────
Best Route:     {' → '.join(bt['route'])} → {bt['route'][0]}
Total Distance: {bt['distance']:.2f} km
Execution Time: {bt['time']:.6f} seconds
Routes Explored: {bt['explored_routes']} paths
Complexity:     O(n!)

    ACO (Ant Colony Optimization)
─────────────────────────────────────────────────────────────────────
Best Route:     {' → '.join(aco['route'])} → {aco['route'][0]}
Total Distance: {aco['distance']:.2f} km
Execution Time: {aco['time']:.6f} seconds
Parameters:     Ants={self.aco_solver.n_ants}, Iterations={self.aco_solver.n_iterations}
                Alpha={self.aco_solver.alpha}, Beta={self.aco_solver.beta}
Complexity:     O(n² × m × iterations)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    COMPARISON
─────────────────────────────────────────────────────────────────────
Distance:
  • Backtracking: {bt['distance']:.2f} km
  • ACO:          {aco['distance']:.2f} km
  • Difference:   {abs(bt['distance'] - aco['distance']):.2f} km ({abs(bt['distance'] - aco['distance'])/bt['distance']*100:.1f}%)

Execution Time:
  • Backtracking: {bt['time']:.6f} seconds
  • ACO:          {aco['time']:.6f} seconds
  • Speedup:      {bt['time']/aco['time']:.2f}x

    ANALYSIS
─────────────────────────────────────────────────────────────────────
✓ Backtracking found the OPTIMAL solution (guaranteed)
✓ ACO found a {'NEAR-OPTIMAL' if abs(bt['distance'] - aco['distance'])/bt['distance']*100 < 5 else 'GOOD'} solution
✓ ACO is {bt['time']/aco['time']:.1f}x {'faster' if aco['time'] < bt['time'] else 'slower'} than Backtracking
✓ For {len(self.cities)} cities, Backtracking explored {bt['explored_routes']} routes
  (Theoretical maximum: {math.factorial(len(self.cities)-1)//2} routes)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        self.text_results.config(state='normal')
        self.text_results.delete('1.0', 'end')
        self.text_results.insert('end', output)
        self.text_results.config(state='disabled')
        
        # Display charts
        self._display_charts()
    
    def _display_charts(self):
        """Display comparison charts"""
        if not self.result_backtracking or not self.result_aco:
            return
        
        # Clear previous charts
        for widget in self.charts_frame.winfo_children():
            widget.destroy()
        
        bt = self.result_backtracking
        aco = self.result_aco
        
        # Create figure with 2 subplots
        fig = Figure(figsize=(10, 5), dpi=100)
        
        # Chart 1: Distance Comparison
        ax1 = fig.add_subplot(121)
        algorithms = ['Backtracking', 'ACO']
        distances = [bt['distance'], aco['distance']]
        colors = ['#FF6B6B', '#4ECDC4']
        
        bars = ax1.bar(algorithms, distances, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax1.set_ylabel('Distance (km)', fontsize=11, fontweight='bold')
        ax1.set_title('Distance Comparison', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, dist in zip(bars, distances):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{dist:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Chart 2: Execution Time Comparison
        ax2 = fig.add_subplot(122)
        times = [bt['time'], aco['time']]
        
        bars = ax2.bar(algorithms, times, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax2.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
        ax2.set_title('Execution Time Comparison', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time:.6f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        fig.tight_layout()
        
        # Display in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.charts_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def view_charts(self):
        try:
            self.notebook.select(self.charts_frame)
        except Exception:
            pass

    def save_details(self):
        try:
            content = self.text_results.get('1.0', 'end')
            path = filedialog.asksaveasfilename(defaultextension='.txt', filetypes=[('Text files','*.txt'), ('All files','*.*')])
            if not path:
                return
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            messagebox.showinfo('Saved', f'Results saved to {path}')
        except Exception as e:
            messagebox.showerror('Error', f'Failed to save: {e}')


def main():
    root = tk.Tk()
    gui = TSPSimpleGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()