# Huong Dan Su Dung - Travelling Salesman Problem

## Cai Dat

### Yeu cau
- Python 3.8 tro len

### Cai dat thu vien

Chay lenh sau:

```bash
pip install -r requirements.txt
```

### Chay ung dung

```bash
python scripts/tsp_gui_simple.py
```

## Su dung co ban

1. Chọn phương pháp nhập dữ liệu: Mặc định, Nhập tay, Import CSV, hoặc Random (số thành phố 3-15).
2. Nếu dùng ACO, tuỳ chỉnh các tham số: số kiến (Ants), số iterations, alpha, beta, evaporation rate và Q constant.
3. Nhấn nút "🚀 SOLVE PROBLEM" để chạy cả hai thuật toán (Backtracking và ACO).
4. Xem kết quả so sánh trong tab "Results".
5. Nhấn "View Charts" để xem biểu đồ so sánh trong tab "Charts".
6. Nhấn "Save Details" để lưu kết quả chi tiết ra file văn bản.

## Import CSV

Tệp CSV có thể có header chứa `name`, `latitude`/`lat` và `longitude`/`lon` (không phân biệt chữ hoa thường). Nếu không có header, parser sẽ giả định thứ tự `name,longitude,latitude` theo ví dụ dưới.

Ví dụ:
```
name,longitude,latitude
Ha Noi,105.8,21.0
Hai Phong,106.7,20.8
Da Nang,108.2,16.0
```

## Tham so ACO

- So kien (n_ants): 5-50, mac dinh 20
- Iterations: 10-200, mac dinh 50
- Alpha: 0.1-3.0, mac dinh 1.0
- Beta: 0.1-5.0, mac dinh 2.0
- Evaporation rate: 0.1-0.9, mac dinh 0.5
- Q constant: 10-500, mac dinh 100

## Gioi han

- Backtracking: toi da 15 thanh pho (do do phuc tap O(n!))
- ACO: toi da 15 thanh pho (de so sanh voi Backtracking)