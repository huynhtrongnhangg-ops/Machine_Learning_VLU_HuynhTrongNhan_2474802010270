# Bài 1: Bayes với ba nguyên nhân
#Ty le cua tung nhom sinh vien
hoc_deu = 0.50
hoc_don = 0.30
it_hoc = 0.20
#Ty le qua mon cua tung nhom
qua_deu = 0.95
qua_don = 0.70
qua_it = 0.30
#Xac suat qua mon cua ca lop
p_qua = hoc_deu * qua_deu + hoc_don * qua_don + it_hoc * qua_it
#Xac suat thuoc tung nhom khi da qua mon
p_deu = hoc_deu * qua_deu / p_qua
p_don = hoc_don * qua_don / p_qua
p_it = it_hoc * qua_it / p_qua
print("Xac suat thuoc tung nhom khi da qua mon:")
print(f"Hoc deu: {p_deu:.4f}")
print(f"Hoc don: {p_don:.4f}")
print(f"It hoc: {p_it:.4f}")
#Kiem tra tong ba xac suat
tong = p_deu + p_don + p_it
print(f"Tong ba xac suat: {tong:.4f}")
