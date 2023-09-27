import gspread

gc = gspread.service_account()

sh = gc.open("fault_detection_results")

print(sh.sheet1.get('A1'))