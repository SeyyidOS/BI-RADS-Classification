import pandas as pd
import tqdm

data = pd.read_excel(r"C:\Users\EMRE\Documents\Gits\BI-RADS-Classification\Codes\veribilgisi.xlsx")
data = data.reset_index()
new_data = pd.DataFrame(columns=['img_name','birads_class','kompozisyon','kadran','alt-iç','alt-dış','üst-iç','merkez','üst-dış'])

dict_birads = {"BI-RADS0":0, "BI-RADS1-2":1, "BI-RADS4-5":2}
dict_kompozisyon = {'A':0, 'B':1, 'C':2, 'D':3}

for index, row in tqdm(data.iterrows()):
    # kadran bilgisi olmayanlar, burada direkt birads 0 dedim ama değişebilir
    if len(row['KADRAN BİLGİSİ (SAĞ)']):
        right_class = dict_birads[row['BIRADS KATEGORİSİ']]
    else:
        right_class = 0

    if len(row['KADRAN BİLGİSİ (SOL)']):
        left_class = dict_birads[row['BIRADS KATEGORİSİ']]
    else:
        left_class = 0

    # CC ve MLO olanları aynı labelledim, değişebilir
    new_row = {'img_name':row['HASTANO'] + "_RCC", 
            'birads_class':right_class, 
            'kompozisyon':dict_kompozisyon[row['MEME KOMPOZİSYONU']], 
            'kadran':0, #sag icin
            'alt-iç': int('ALT İÇ' in row['KADRAN BİLGİSİ (SAĞ)']),
            'üst-iç': int('ÜST İÇ' in row['KADRAN BİLGİSİ (SAĞ)']),
            'merkez': int('MERKEZ' in row['KADRAN BİLGİSİ (SAĞ)']),
            'alt-dış': int('ALT DIŞ' in row['KADRAN BİLGİSİ (SAĞ)']),
            'üst-dış': int('ÜST DIŞ' in row['KADRAN BİLGİSİ (SAĞ)'])
            }

    new_row2 = {'img_name':row['HASTANO'] + "_RMLO", 
            'birads_class':right_class, 
            'kompozisyon':dict_kompozisyon[row['MEME KOMPOZİSYONU']], 
            'kadran':0, #sag icin
            'alt-iç': int('ALT İÇ' in row['KADRAN BİLGİSİ (SAĞ)']),
            'üst-iç': int('ÜST İÇ' in row['KADRAN BİLGİSİ (SAĞ)']),
            'merkez': int('MERKEZ' in row['KADRAN BİLGİSİ (SAĞ)']),
            'alt-dış': int('ALT DIŞ' in row['KADRAN BİLGİSİ (SAĞ)']),
            'üst-dış': int('ÜST DIŞ' in row['KADRAN BİLGİSİ (SAĞ)'])
            }

    new_row3 = {'img_name':row['HASTANO'] + "_LCC", 
            'birads_class':left_class, 
            'kompozisyon':dict_kompozisyon[row['MEME KOMPOZİSYONU']], 
            'kadran':1, #sol icin
            'alt-iç': int('ALT İÇ' in row['KADRAN BİLGİSİ (SOL)']),
            'üst-iç': int('ÜST İÇ' in row['KADRAN BİLGİSİ (SOL)']),
            'merkez': int('MERKEZ' in row['KADRAN BİLGİSİ (SOL)']),
            'alt-dış': int('ALT DIŞ' in row['KADRAN BİLGİSİ (SOL)']),
            'üst-dış': int('ÜST DIŞ' in row['KADRAN BİLGİSİ (SOL)'])
            }

    new_row4 = {'img_name':row['HASTANO'] + "_LMLO", 
            'birads_class':left_class, 
            'kompozisyon':dict_kompozisyon[row['MEME KOMPOZİSYONU']], 
            'kadran':1, #sol icin
            'alt-iç': int('ALT İÇ' in row['KADRAN BİLGİSİ (SOL)']),
            'üst-iç': int('ÜST İÇ' in row['KADRAN BİLGİSİ (SOL)']),
            'merkez': int('MERKEZ' in row['KADRAN BİLGİSİ (SOL)']),
            'alt-dış': int('ALT DIŞ' in row['KADRAN BİLGİSİ (SOL)']),
            'üst-dış': int('ÜST DIŞ' in row['KADRAN BİLGİSİ (SOL)'])
            }
        
    new_data = new_data.append(new_row, ignore_index=True)
    new_data = new_data.append(new_row2, ignore_index=True)
    new_data = new_data.append(new_row3, ignore_index=True)
    new_data = new_data.append(new_row4, ignore_index=True)

new_data.to_csv("images_with_labels.csv", index=False)