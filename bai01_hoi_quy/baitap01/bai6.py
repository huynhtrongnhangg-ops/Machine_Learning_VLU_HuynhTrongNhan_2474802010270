def du_doan_gia(dien_tich):
    w = 0.078367
    b = 0.401752

    if dien_tich < 35.5 or dien_tich > 117.5:
        print("Canh bao: dien tich nam ngoai khoang du lieu.")

    return w * dien_tich + b


gia_60 = du_doan_gia(60)
gia_80 = du_doan_gia(80)
gia_200 = du_doan_gia(200)

print(f"Du doan gia can 60 m2: {gia_60:.3f} ty dong")
print(f"Du doan gia can 80 m2: {gia_80:.3f} ty dong")
print(f"Du doan gia can 200 m2: {gia_200:.3f} ty dong")