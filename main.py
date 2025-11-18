import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import threading
import time
from backtracking import TSPBacktracking
from aco import ACO_TSP
from utils import create_distance_matrix

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

class TSPApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("TSP SOLVER 2025 ULTIMATE – Interactive Comparison")
        # Start with a window sized relative to the screen for better compatibility
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        self.geometry(f"{int(screen_w*0.9)}x{int(screen_h*0.85)}")
        self.minsize(int(screen_w*0.6), int(screen_h*0.6))

        self.coords = None
        self.names = None
        self.dist_matrix = None
        self.full_log = []
        self.show_full_log = False
        self._saved_log_view = None
        self.last_bt = None
        self.last_aco = None
        self.aco_full_log = []
        self.bt_full_log = []
        self.last_aco_history = None
        self.last_aco_history_times = None
        self.last_bt_time = None
        self.last_bt_nodes = None
        self.history = []  # Lưu tất cả lần chạy để so sánh

        self.create_ui()

    def create_ui(self):
        # Title & subtitle
        title = ctk.CTkLabel(self, text="TSP SOLVER 2025 ULTIMATE", font=ctk.CTkFont(size=40, weight="bold"))
        title.pack(pady=(20, 5))
        subtitle = ctk.CTkLabel(self, text="Backtracking vs ACO – So sánh trực quan & interactive", font=ctk.CTkFont(size=18), text_color="#00ffff")
        subtitle.pack(pady=(0, 20))

        main = ctk.CTkFrame(self)
        main.pack(fill="both", expand=True, padx=30, pady=10)

        # (Removed floating Run button — use the main Run button in the left panel)

        # LEFT PANEL
        left = ctk.CTkFrame(main, width=500)
        left.pack(side="left", fill="y", padx=(0, 20))

        ctk.CTkLabel(left, text="DỮ LIỆU & CẤU HÌNH", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=(20,10))
        
        self.file_label = ctk.CTkLabel(left, text="Chưa chọn file", text_color="gray")
        self.file_label.pack(pady=5)
        ctk.CTkButton(left, text="CHỌN FILE CSV", command=self.load_csv).pack(pady=5)

        ctk.CTkLabel(left, text="Số thành phố (8-50):").pack(pady=(15,5))
        self.n_slider = ctk.CTkSlider(left, from_=8, to=50, number_of_steps=42)
        self.n_slider.set(15)
        self.n_slider.pack(pady=5, padx=20, fill="x")
        self.n_label = ctk.CTkLabel(left, text="15")
        self.n_label.pack()
        self.n_slider.configure(command=lambda v: self.n_label.configure(text=f"{int(v)}"))

        ctk.CTkLabel(left, text="Seed (42 = so sánh công bằng):").pack(pady=(10,5))
        self.seed = ctk.CTkEntry(left)
        self.seed.insert(0, "42")
        self.seed.pack(pady=5, padx=20, fill="x")

        ctk.CTkLabel(left, text="THUẬT TOÁN:").pack(pady=(20,5))
        self.method = ctk.StringVar(value="both")
        ctk.CTkRadioButton(left, text="Cả hai", variable=self.method, value="both").pack(pady=3)
        ctk.CTkRadioButton(left, text="Backtracking", variable=self.method, value="backtracking").pack(pady=3)
        ctk.CTkRadioButton(left, text="ACO", variable=self.method, value="aco").pack(pady=3)

        ctk.CTkLabel(left, text="CẤU HÌNH ACO").pack(pady=(20,5))
        # Hide the detailed ACO entry fields by default; use a settings dialog instead.
        self.aco_params = {
            "n_ants": "100",
            "iterations": "1000",
            "alpha": "1.0",
            "beta": "5.0",
            "evaporation": "0.6",
            "Q": "200",
        }

        # Status row with a prominent settings button
        status_row = ctk.CTkFrame(left)
        status_row.pack(fill="x", padx=15, pady=(6,8))
        self.aco_status_label = ctk.CTkLabel(status_row, text="ACO: mặc định", text_color="gray")
        self.aco_status_label.pack(side="left")
        ctk.CTkButton(status_row, text="CÀI ĐẶT ACO", command=self.open_aco_settings, fg_color="#3a7bd5").pack(side="right")

        # Main run button (high contrast for dark background)
        ctk.CTkButton(left, text="GIẢI NGAY", command=self.run, height=60, font=ctk.CTkFont(size=24, weight="bold"), fg_color="#ffcc00").pack(pady=20, padx=40, fill="x")

        # (Batch comparison UI removed — use single-run interactive comparison only)

        # RIGHT TABS
        self.tabview = ctk.CTkTabview(main)
        self.tabview.pack(side="right", fill="both", expand=True)

        self.tabview.add("Kết quả")
        self.tabview.add("Backtracking Tour")
        self.tabview.add("ACO Tour")
        self.tabview.add("So sánh Interactive")

        # Log
        log_frame = ctk.CTkFrame(self.tabview.tab("Kết quả"))
        log_frame.pack(fill="both", expand=True, padx=15, pady=15)
        self.log_box = ctk.CTkTextbox(log_frame, font=("Consolas", 13))
        self.log_box.pack(fill="both", expand=True, side="left")
        # Use native tk.Scrollbar for stability (avoid customtkinter scrollbar recursion bug)
        scrollbar = tk.Scrollbar(log_frame, command=self.log_box.yview, orient='vertical')
        scrollbar.pack(side="right", fill="y")
        try:
            self.log_box.configure(yscrollcommand=scrollbar.set)
        except Exception:
            # fallback: ignore if CTk textbox doesn't accept standard yscrollcommand
            pass

        self.full_log_btn = ctk.CTkButton(self.tabview.tab("Kết quả"), text="Xem toàn bộ log", command=self.toggle_full_log)
        self.full_log_btn.pack(pady=5)

        # Tour plots
        self.fig_bt = Figure(facecolor='#0e1117')
        self.canvas_bt = FigureCanvasTkAgg(self.fig_bt, self.tabview.tab("Backtracking Tour"))
        self.canvas_bt.get_tk_widget().pack(fill="both", expand=True)

        self.fig_aco = Figure(facecolor='#0e1117')
        self.canvas_aco = FigureCanvasTkAgg(self.fig_aco, self.tabview.tab("ACO Tour"))
        self.canvas_aco.get_tk_widget().pack(fill="both", expand=True)

        # Interactive Comparison
        self.fig_comp = Figure(facecolor='#0e1117', figsize=(13, 8))

        # Controls for interactive plot mode (Iteration vs Batch)
        ctrl_parent = self.tabview.tab("So sánh Interactive")
        ctrl_frame = ctk.CTkFrame(ctrl_parent)
        ctrl_frame.pack(fill="x", padx=10, pady=(10, 4))
        ctk.CTkLabel(ctrl_frame, text="Chế độ hiển thị:", width=120, anchor="w").pack(side="left", padx=(6,8))
        self.interactive_mode = tk.StringVar(value="iteration")
        ctk.CTkRadioButton(ctrl_frame, text="Iteration", variable=self.interactive_mode, value="iteration", command=self._on_interactive_mode_changed).pack(side="left", padx=4)
        ctk.CTkRadioButton(ctrl_frame, text="Time", variable=self.interactive_mode, value="time", command=self._on_interactive_mode_changed).pack(side="left", padx=4)

        self.canvas_comp = FigureCanvasTkAgg(self.fig_comp, ctrl_parent)
        self.canvas_comp.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(6,10))
        self.toolbar = NavigationToolbar2Tk(self.canvas_comp, ctrl_parent)
        self.toolbar.update()
        self.toolbar.pack(side="bottom", fill="x")

        self.initial_comparison_plot()
        # Load a default Vietnam cities dataset so the initial comparisons use real cities
        try:
            self.load_vn_cities()
        except Exception:
            pass

        # Keep left panel responsive when window resizes
        self._left_frame = left
        self.bind('<Configure>', self._on_resize)

    def _on_resize(self, event):
        try:
            total_w = max(self.winfo_width(), 400)
            # keep left panel at about 23-30% of available width, clamp between 320 and 700
            target_w = int(max(320, min(700, total_w * 0.25)))
            self._left_frame.configure(width=target_w)
        except Exception:
            pass

    def initial_comparison_plot(self):
        ax = self.fig_comp.add_subplot(111)
        ax.text(0.5, 0.5, "Chưa có dữ liệu\nChạy thuật toán để so sánh", ha='center', va='center', fontsize=18, color='gray', transform=ax.transAxes)
        ax.set_title("SO SÁNH BACKTRACKING vs ACO", color='white', fontsize=18)
        ax.axis('off')
        self.canvas_comp.draw()

    def open_aco_settings(self):
        dlg = ctk.CTkToplevel(self)
        dlg.title("Cài đặt ACO")
        dlg.geometry("480x320")
        entries = {}

        fields = [("n_ants","Số kiến"), ("iterations","Vòng lặp"), ("alpha","α"), ("beta","β"), ("evaporation","ρ"), ("Q","Q")]
        for k, label in fields:
            f = ctk.CTkFrame(dlg); f.pack(fill="x", padx=14, pady=6)
            ctk.CTkLabel(f, text=label, width=160, anchor="w").pack(side="left")
            val = self.aco_params.get(k, "")
            if hasattr(val, 'get'):
                curr = val.get()
            else:
                curr = str(val)
            e = ctk.CTkEntry(f, width=140); e.insert(0, str(curr)); e.pack(side="right")
            entries[k] = e

        btn_frame = ctk.CTkFrame(dlg); btn_frame.pack(pady=12)
        ctk.CTkButton(btn_frame, text="Áp dụng", command=lambda: self._apply_aco_settings(entries, dlg)).pack(side="left", padx=8)
        ctk.CTkButton(btn_frame, text="Hủy", command=dlg.destroy).pack(side="right", padx=8)

    def _apply_aco_settings(self, entries, dlg):
        # Validate and copy values back to the main entries
        for k, e in entries.items():
            v = e.get()
            try:
                if k in ("n_ants", "iterations"):
                    _ = int(v)
                else:
                    _ = float(v)
            except Exception:
                messagebox.showerror("Lỗi", f"Giá trị không hợp lệ cho {k}: {v}")
                return
            # store back into the aco params dict (hidden in main UI)
            self.aco_params[k] = str(v)
        # update status label
        try:
            self.aco_status_label.configure(text="ACO: đã cài đặt", text_color="#00ff88")
        except Exception:
            pass
        dlg.destroy()

    def toggle_full_log(self):
        self.show_full_log = not self.show_full_log
        if self.show_full_log:
            # save current view so we can restore it later
            try:
                self._saved_log_view = self.log_box.get("0.0", "end")
            except Exception:
                self._saved_log_view = None
            self.full_log_btn.configure(text="Ẩn toàn bộ log")
            self.log_box.delete("0.0", "end")
            # Show detailed logs for both Backtracking and ACO
            try:
                combined = []
                if self.bt_full_log:
                    combined.append("--- Backtracking detailed ---")
                    combined.extend(self.bt_full_log)
                if self.aco_full_log:
                    if combined:
                        combined.append("")
                    combined.append("--- ACO detailed ---")
                    combined.extend(self.aco_full_log)
                if not combined:
                    combined = ["(Không có log chi tiết)"]
                self.log_box.insert("0.0", "\n".join(combined))
            except Exception:
                self.log_box.insert("0.0", "\n".join(self.full_log))
        else:
            self.full_log_btn.configure(text="Xem toàn bộ log")
            self.log_box.delete("0.0", "end")
            if self._saved_log_view is not None:
                # restore previous visible content exactly
                self.log_box.insert("0.0", self._saved_log_view)
            else:
                # fallback: show recent lines
                self.log_box.insert("0.0", "\n".join(self.full_log[-200:]))

    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files","*.csv"), ("All Files","*.*")])
        if not path:
            return
        coords = []
        names = []
        with open(path, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        # skip header if present
        start_idx = 0
        if lines:
            first = lines[0].split(',')
            try:
                float(first[0]); float(first[1])
            except Exception:
                start_idx = 1
        for line in lines[start_idx:]:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0]); y = float(parts[1])
                coords.append([x, y])
                names.append(parts[2] if len(parts) > 2 else f"C{len(names)}")
            except Exception:
                continue
        if not coords:
            messagebox.showerror("Lỗi", "Không đọc được dữ liệu tọa độ từ file.")
            return
        self.coords = np.array(coords)
        self.names = names
        self.dist_matrix = create_distance_matrix(self.coords)
        self.file_label.configure(text=path.split('/')[-1])
        self.log_box.insert("end", f"Đã tải {len(self.coords)} thành phố từ {path}\n")

    def load_vn_cities(self):
        # A small sample of Vietnamese cities with lat/lon
        cities = [
            ("Hanoi", 21.0278, 105.8342),
            ("Ho Chi Minh", 10.8231, 106.6297),
            ("Da Nang", 16.0544, 108.2022),
            ("Hai Phong", 20.8449, 106.6881),
            ("Can Tho", 10.0452, 105.7469),
            ("Nha Trang", 12.2388, 109.1967),
            ("Hue", 16.4637, 107.5909),
            ("Vung Tau", 10.4114, 107.1362),
            ("Quy Nhon", 13.7827, 109.2191),
            ("Buon Ma Thuot", 12.6667, 108.0500),
            ("Pleiku", 13.9833, 108.0000),
            ("Thanh Hoa", 19.8067, 105.7786),
            ("Vinh", 18.6796, 105.6814),
            ("Dong Hoi", 17.4683, 106.6000),
            ("Phu Quoc", 10.2906, 103.9844),
        ]
        names = [c[0] for c in cities]
        lats = np.array([c[1] for c in cities])
        lons = np.array([c[2] for c in cities])
        # project lat/lon to kilometers using equirectangular approx
        mean_lat = np.deg2rad(lats.mean())
        x = lons * (111.320 * np.cos(mean_lat))
        y = lats * 110.574
        coords = np.column_stack((x, y))
        self.coords = coords
        self.names = names
        self.dist_matrix = create_distance_matrix(self.coords)
        self.file_label.configure(text="Vietnam sample")
        self.log_box.insert("end", f"Đã tải bộ dữ liệu mẫu Việt Nam ({len(self.coords)} thành phố)\n")

    def run(self):
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def _run(self):
        # Prepare data
        if self.coords is None:
            try:
                n = int(self.n_slider.get())
            except Exception:
                n = 15
            try:
                seed = int(self.seed.get())
            except Exception:
                seed = None
            rng = np.random.RandomState(seed)
            self.coords = rng.rand(n, 2) * 100
            self.names = [f"C{i}" for i in range(n)]
            self.dist_matrix = create_distance_matrix(self.coords)

        method = self.method.get()

        record = {'n': len(self.coords)}

        # Run backtracking if requested
        if method in ("both", "backtracking"):
            self.log_box.insert("end", "Chạy Backtracking...\n")
            bt = TSPBacktracking(self.coords, self.names)
            path, cost, elapsed, log_lines, nodes = bt.solve()
            # store last backtracking meta info for plotting
            try:
                self.last_bt = (path, cost)
                self.last_bt_time = float(elapsed)
                self.last_bt_nodes = int(nodes)
            except Exception:
                pass
            # build per-edge detailed log for Backtracking best path
            try:
                bt_detailed = []
                for i in range(len(path) - 1):
                    a = path[i]
                    b = path[i+1]
                    c = float(self.dist_matrix[a][b]) if self.dist_matrix is not None else 0.0
                    name_a = self.names[a] if self.names and a < len(self.names) else str(a)
                    name_b = self.names[b] if self.names and b < len(self.names) else str(b)
                    bt_detailed.append(f"{name_a} -> {name_b}: cost={c:.3f}")
                # store for detailed view
                self.bt_full_log.extend(bt_detailed)
                # also append a summary line to full_log
                self.full_log.append(f"Backtracking: path={path}, cost={cost:.3f}, time={elapsed:.3f}s, nodes={nodes}")
            except Exception:
                pass
            # store and draw backtracking tour
            try:
                self.last_bt = (path, cost)
                self.draw_backtracking_tour(path, cost)
            except Exception:
                pass
            record['bt_cost'] = float(cost)
            record['bt_time'] = float(elapsed)
            record['bt_nodes'] = int(nodes)
            self.log_box.insert("end", f"Backtracking: cost={cost:.2f}, time={elapsed:.3f}s, nodes={nodes}\n")

        # Run ACO if requested
        if method in ("both", "aco"):
            self.log_box.insert("end", "Chạy ACO...\n")
            aco = ACO_TSP(self.dist_matrix)
            # apply params
            try:
                def _param(k):
                    # Simplified: redirect to interactive plot (Iteration vs Time modes only)
                    try:
                        self._on_interactive_mode_changed()
                    except Exception:
                        self.initial_comparison_plot()
            except Exception:
                # Ensure the try has an except to avoid syntax error; log and continue if parameter handling fails
                try:
                    self.log_box.insert("end", "Lỗi khi áp dụng tham số ACO (bỏ qua)\n")
                except Exception:
                    pass

        # Update comparison plot
        try:
            self.update_comparison_plot()
        except Exception as e:
            self.log_box.insert("end", f"Lỗi khi cập nhật biểu đồ: {e}\n")

    # Batch comparison removed: single-run interactive comparison remains

    # Các hàm log, load_csv, run, _run, draw_tours giữ nguyên như phiên bản trước (với log chi tiết đường đi)

    def update_comparison_plot(self):
        # Simplified: only keep the interactive Iteration ↔ Time view
        try:
            self.fig_comp.clear()
            # Delegate rendering to the interactive plot routine which handles iteration/time modes
            self.update_interactive_plot(None, self.last_aco_history_times)
        except Exception:
            # fallback: show a placeholder
            self.fig_comp.clear()
            ax = self.fig_comp.add_subplot(111)
            ax.text(0.5, 0.5, 'Chưa có dữ liệu so sánh (Chạy ACO trước)', ha='center', va='center', color='gray', transform=ax.transAxes)
            self.canvas_comp.draw()
        return
        # Old multi-panel comparison removed; interactive view used instead.

    def update_interactive_plot(self, aco_history_costs=None, aco_history_times=None, bt_cost: float = None):
        # Plot ACO cumulative time vs iteration (Iteration mode) or iteration vs time (Time mode)
        try:
            self.fig_comp.clear()
            ax = self.fig_comp.add_subplot(111)
            ax.set_facecolor('#0e1117')

            mode = getattr(self, 'interactive_mode', tk.StringVar(value='iteration')).get()

            # Prefer per-iteration cumulative times for plotting
            if aco_history_times:
                n = len(aco_history_times)
                iters = list(range(1, n + 1))
                if mode == 'iteration':
                    # x = iteration, y = cumulative time
                    ax.plot(iters, aco_history_times, '-o', color='#ff00ff', label='ACO time (cumulative)', linewidth=2)
                    ax.set_xlabel('Iteration', color='white')
                    ax.set_ylabel('Time (s)', color='white')
                else:
                    # time mode: x = time, y = iteration (plot as step-like)
                    ax.plot(aco_history_times, iters, '-o', color='#ff00ff', label='ACO iterations over time', linewidth=2)
                    ax.set_xlabel('Time (s)', color='white')
                    ax.set_ylabel('Iteration', color='white')
            else:
                # No per-iteration data; show placeholder or fallback to history-over-n if available
                if len(self.history) > 0:
                    n_vals = [d.get('n') for d in self.history]
                    if any('aco_time' in d and d['aco_time'] is not None for d in self.history):
                        aco_vals = [d.get('aco_time') for d in self.history]
                        ax.plot(n_vals, aco_vals, '-^', color='#ff00ff', label='ACO time (avg)')
                        ax.set_xlabel('Số thành phố (n)', color='white')
                        ax.set_ylabel('Time (s)', color='white')
                    else:
                        ax.text(0.5, 0.5, 'Chưa có dữ liệu thời gian ACO', ha='center', va='center', color='gray', transform=ax.transAxes)

            # Plot Backtracking as a single marker (nodes vs time)
            try:
                if self.last_bt_time is not None and self.last_bt_nodes is not None:
                    if mode == 'iteration':
                        # BT as a single point: x = node_count, y = time
                        ax.plot([self.last_bt_nodes], [self.last_bt_time], 's', color='#00ff88', markersize=10, label='Backtracking (nodes vs time)')
                    else:
                        # time mode: x = time, y = node_count
                        ax.plot([self.last_bt_time], [self.last_bt_nodes], 's', color='#00ff88', markersize=10, label='Backtracking (time vs nodes)')
            except Exception:
                pass

            ax.set_title('Interactive Comparison (Iteration ↔ Time)', color='white')
            leg = ax.legend()
            if leg:
                for text in leg.get_texts():
                    text.set_color('white')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_edgecolor('#444')
            ax.grid(True, alpha=0.3)
            self.canvas_comp.draw()
        except Exception as e:
            try:
                self.log_box.insert('end', f"Lỗi vẽ interactive plot: {e}\n")
            except Exception:
                pass

    def _on_interactive_mode_changed(self):
        mode = self.interactive_mode.get()
        try:
            bt_cost = None
            if self.last_bt and len(self.last_bt) >= 2:
                try:
                    bt_cost = float(self.last_bt[1])
                except Exception:
                    bt_cost = None

            if mode == 'iteration':
                # pass per-iteration times as second argument
                self.update_interactive_plot(None, self.last_aco_history_times, bt_cost=bt_cost)
            else:
                # time mode
                self.update_interactive_plot(None, self.last_aco_history_times, bt_cost=bt_cost)
        except Exception as e:
            try:
                self.log_box.insert('end', f"Lỗi khi chuyển chế độ interactive: {e}\n")
            except Exception:
                pass

    def draw_backtracking_tour(self, path, cost=None):
        try:
            self.fig_bt.clear()
            ax = self.fig_bt.add_subplot(111)
            ax.set_facecolor('#0e1117')
            coords = self.coords
            if coords is None or path is None:
                ax.text(0.5, 0.5, "Chưa có đường đi Backtracking", ha='center', va='center', color='gray')
                self.canvas_bt.draw()
                return
            path_nodes = path
            xs = [coords[i][0] for i in path_nodes]
            ys = [coords[i][1] for i in path_nodes]
            ax.plot(xs, ys, '-o', color='#00ff88', linewidth=2, markersize=6)
            for i, idx in enumerate(path_nodes[:-1]):
                ax.annotate(str(self.names[idx] if self.names else idx), (coords[idx][0], coords[idx][1]), color='white')
            ax.set_title(f"Backtracking Tour{(' - cost: %.2f' % cost) if cost is not None else ''}", color='white')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_edgecolor('#444')
            self.canvas_bt.draw()
        except Exception as e:
            try:
                self.log_box.insert('end', f"Lỗi vẽ Backtracking: {e}\n")
            except Exception:
                pass

    def draw_aco_tour(self, path, cost=None):
        try:
            self.fig_aco.clear()
            ax = self.fig_aco.add_subplot(111)
            ax.set_facecolor('#0e1117')
            coords = self.coords
            if coords is None or path is None:
                ax.text(0.5, 0.5, "Chưa có đường đi ACO", ha='center', va='center', color='gray')
                self.canvas_aco.draw()
                return
            path_nodes = path
            xs = [coords[i][0] for i in path_nodes]
            ys = [coords[i][1] for i in path_nodes]
            ax.plot(xs, ys, '-o', color='#ff00ff', linewidth=2, markersize=6)
            for i, idx in enumerate(path_nodes[:-1]):
                ax.annotate(str(self.names[idx] if self.names else idx), (coords[idx][0], coords[idx][1]), color='white')
            ax.set_title(f"ACO Tour{(' - cost: %.2f' % cost) if cost is not None else ''}", color='white')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_edgecolor('#444')
            self.canvas_aco.draw()
        except Exception as e:
            try:
                self.log_box.insert('end', f"Lỗi vẽ ACO: {e}\n")
            except Exception:
                pass

# Phần _run và draw_tours giữ nguyên như phiên bản trước (với log chi tiết đường đi)

if __name__ == "__main__":
    app = TSPApp()
    app.mainloop()