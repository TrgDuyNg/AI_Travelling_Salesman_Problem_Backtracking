# Huong Dan Su Dung - Travelling Salesman Problem

## Cai Dat

### Yeu cau
- Python 3.8 tro len

### Cai dat thu vien

Chay lenh sau:

\`\`\`bash
pip install -r requirements.txt
\`\`\`

### Chay ung dung

\`\`\`bash
python scripts/tsp_gui_tkinter.py
\`\`\`

## Su dung co ban

1. Chon phuong phap nhap du lieu: Mac dinh, Nhap tay, File CSV, hoac Random (5-15 thanh pho)
2. Neu dung ACO, tuy chinh cac tham so: so kien, so iterations, alpha, beta, evaporation rate, Q constant
3. Nhan nut "GIAI BAI TOAN" de chay hai thuat toan
4. Xem ket qua so sanh trong tab "Ket qua"
5. Nhan "Xem bieu do" de xem bieu do so sanh
6. Nhan "In chi tiet" de xem cac buoc chi tiet cua tung thuat toan

## Tep CSV

Tep CSV can co dung: name,longitude,latitude

Ví du:
\`\`\`
name,longitude,latitude
Ha Noi,105.8,21.0
Hai Phong,106.7,20.8
Da Nang,108.2,16.0
\`\`\`

## Cac tham so ACO

- So kien (n_ants): 5-50, mac dinh 20
- Iterations: 10-200, mac dinh 50
- Alpha: 0.1-3.0, mac dinh 1.0
- Beta: 0.1-5.0, mac dinh 2.0
- Evaporation rate: 0.1-0.9, mac dinh 0.5
- Q constant: 10-500, mac dinh 100

## Gioi han

- Backtracking: toi da 15 thanh pho (do do phuc tap O(n!))
- ACO: toi da 15 thanh pho (de so sanh voi Backtracking)