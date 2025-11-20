# 🗺️ Giải Bài Toán Người Du Lịch (Travelling Salesman Problem)

## 📋 Mô tả đồ án

Đồ án này giải quyết bài toán **Người du lịch (TSP - Travelling Salesman Problem)** bằng hai thuật toán:

1. **Backtracking (Quay lui)** - Sử dụng thư viện `simpleAI`
2. **ACO (Ant Colony Optimization)** - Thuật toán tối ưu hóa đàn kiến

## 🎯 Mục tiêu

- Tìm tuyến đường ngắn nhất đi qua tất cả các thành phố và quay về điểm xuất phát
- So sánh hiệu quả của hai thuật toán khác nhau
- Trực quan hóa kết quả và quá trình tìm kiếm

## 🛠️ Công nghệ sử dụng

- **Python 3.8+**
- **simpleAI** - Thư viện AI cho thuật toán Backtracking
- **NumPy** - Xử lý ma trận và tính toán
- **Matplotlib** - Trực quan hóa kết quả
- **Tkinter** - Giao diện người dùng

## 📦 Cài đặt

### Bước 1: Cài đặt Python

Đảm bảo bạn đã cài đặt Python 3.8 trở lên.

### Bước 2: Cài đặt thư viện

\`\`\`bash
pip install simpleai==0.8.3
pip install numpy==1.24.3
pip install matplotlib==3.7.1
\`\`\`

Hoặc sử dụng file requirements.txt:

\`\`\`bash
pip install -r requirements.txt
\`\`\`

## 🚀 Chạy chương trình

### Chạy ứng dụng chính với giao diện:

\`\`\`bash
python scripts/tsp_main.py
\`\`\`

### Chạy riêng từng thuật toán:

**Backtracking:**
\`\`\`bash
python scripts/tsp_backtracking.py
\`\`\`

**ACO:**
\`\`\`bash
python scripts/tsp_aco.py
\`\`\`

## 📊 Tính năng

### 1. Thuật toán Backtracking (simpleAI)

- ✅ Sử dụng CSP (Constraint Satisfaction Problem) framework
- ✅ Tìm kiếm đầy đủ không gian giải pháp
- ✅ Đảm bảo tìm được giải pháp tối ưu (nếu có đủ thời gian)
- ✅ Áp dụng các ràng buộc để cắt tỉa không gian tìm kiếm

**Cách hoạt động:**
- Định nghĩa bài toán như một CSP với các biến là vị trí trong tuyến đường
- Áp dụng ràng buộc: mỗi thành phố chỉ được thăm một lần
- Sử dụng backtracking để tìm kiếm giải pháp tối ưu

### 2. Thuật toán ACO (Ant Colony Optimization)

- ✅ Mô phỏng hành vi tìm đường của đàn kiến
- ✅ Sử dụng pheromone để hướng dẫn tìm kiếm
- ✅ Cân bằng giữa khai thác (exploitation) và khám phá (exploration)
- ✅ Phù hợp cho bài toán lớn với nhiều thành phố

**Tham số có thể điều chỉnh:**
- `n_ants`: Số lượng kiến (mặc định: 30)
- `n_iterations`: Số vòng lặp (mặc định: 100)
- `alpha`: Ảnh hưởng của pheromone (mặc định: 1.0)
- `beta`: Ảnh hưởng của heuristic (mặc định: 2.0)
- `evaporation_rate`: Tỷ lệ bay hơi pheromone (mặc định: 0.5)

### 3. Giao diện trực quan

- 🎨 Giao diện đẹp mắt với theme tối
- 📈 Hiển thị tuyến đường trên bản đồ
- 📊 Biểu đồ hội tụ của thuật toán ACO
- 📋 So sánh chi tiết kết quả hai thuật toán
- ⚙️ Điều chỉnh tham số thuật toán

## 📈 Kết quả mẫu

### Dữ liệu test

**8 thành phố Việt Nam:**
- Hà Nội
- Hải Phòng
- Đà Nẵng
- Huế
- TP. Hồ Chí Minh
- Cần Thơ
- Nha Trang
- Đà Lạt

### So sánh hiệu suất

| Tiêu chí | Backtracking | ACO |
|----------|--------------|-----|
| Độ chính xác | Tối ưu toàn cục | Gần tối ưu |
| Thời gian | Chậm với n lớn | Nhanh hơn |
| Khả năng mở rộng | Hạn chế (n < 15) | Tốt (n > 20) |
| Tính ổn định | Ổn định | Phụ thuộc tham số |

## 🎓 Lý thuyết

### Bài toán TSP

Cho n thành phố và khoảng cách giữa mỗi cặp thành phố, tìm tuyến đường ngắn nhất đi qua tất cả các thành phố đúng một lần và quay về điểm xuất phát.

**Độ phức tạp:** O(n!) - NP-hard

### Thuật toán Backtracking

Backtracking là kỹ thuật tìm kiếm đầy đủ với khả năng quay lui khi gặp ngõ cụt.

**Ưu điểm:**
- Tìm được giải pháp tối ưu
- Dễ hiểu và cài đặt

**Nhược điểm:**
- Chậm với bài toán lớn
- Độ phức tạp hàm mũ

### Thuật toán ACO

ACO mô phỏng hành vi tìm đường của đàn kiến trong tự nhiên.

**Nguyên lý:**
1. Kiến đi ngẫu nhiên và để lại pheromone
2. Đường ngắn hơn có pheromone đậm đặc hơn
3. Kiến sau ưu tiên đi theo đường có nhiều pheromone
4. Pheromone bay hơi theo thời gian

**Ưu điểm:**
- Hiệu quả với bài toán lớn
- Có khả năng thoát khỏi local optimum
- Dễ song song hóa

**Nhược điểm:**
- Không đảm bảo tối ưu toàn cục
- Phụ thuộc vào tham số

## 📝 Cấu trúc code

\`\`\`
tsp-solver/
├── scripts/
│   ├── tsp_main.py              # Ứng dụng chính với GUI
│   ├── tsp_backtracking.py      # Thuật toán Backtracking
│   └── tsp_aco.py               # Thuật toán ACO
├── requirements.txt             # Thư viện cần thiết
└── README.md                    # Tài liệu hướng dẫn
\`\`\`

## 🔬 Hướng phát triển

- [ ] Thêm thuật toán Genetic Algorithm
- [ ] Hỗ trợ nhập dữ liệu từ file
- [ ] Tối ưu hóa hiệu suất cho bài toán lớn (n > 50)
- [ ] Thêm animation cho quá trình tìm kiếm
- [ ] Export kết quả ra PDF

## 👨‍💻 Tác giả

Đồ án môn học - Trí tuệ nhân tạo

## 📄 License

MIT License - Tự do sử dụng cho mục đích học tập

## 🙏 Tham khảo

- simpleAI Documentation: https://simpleai.readthedocs.io/
- Dorigo, M., & Stützle, T. (2004). Ant Colony Optimization
- Russell, S., & Norvig, P. (2020). Artificial Intelligence: A Modern Approach

---

**Lưu ý:** Đây là đồ án học tập. Kết quả có thể khác nhau tùy thuộc vào tham số và dữ liệu đầu vào.