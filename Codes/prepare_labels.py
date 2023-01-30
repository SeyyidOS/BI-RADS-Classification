import pandas as pd
from tqdm import tqdm

data = pd.read_excel(r"C:\Users\EMRE\Documents\Gits\BI-RADS-Classification\Codes\veribilgisi.xlsx")
data = data.reset_index()
new_data = pd.DataFrame(columns=['img_name','birads_class','kompozisyon','kadran','alt-ic','alt-dis','ust-ic','merkez','ust-dis'])

# aşağıdakiler one-hot yapılacak
dict_birads = {"BI-RADS0":0, "BI-RADS1-2":1, "BI-RADS4-5":2}
dict_kompozisyon = {'A':0, 'B':1, 'C':2, 'D':3}

for index, row in tqdm(data.iterrows()):
    # kadran bilgisi olmayanlar, burada direkt birads 0 dedim ama değişebilir
    if str(row['KADRAN BİLGİSİ (SAĞ)']) != "nan":
        right_class = dict_birads[row['BIRADS KATEGORİSİ']]
    else:
        right_class = 0

    if str(row['KADRAN BİLGİSİ (SOL)']) != "nan":
        left_class = dict_birads[row['BIRADS KATEGORİSİ']]
    else:
        left_class = 0

    # CC ve MLO olanları aynı labelledim, değişebilir
    new_row = {'img_name':str(row['HASTANO']) + "_RCC", 
            'birads_class':right_class, 
            'kompozisyon':dict_kompozisyon[row['MEME KOMPOZİSYONU']], 
            'kadran':0, #sag icin
            'alt-ic': int('ALT İÇ' in str(row['KADRAN BİLGİSİ (SAĞ)'])),
            'ust-ic': int('ÜST İÇ' in str(row['KADRAN BİLGİSİ (SAĞ)'])),
            'merkez': int('MERKEZ' in str(row['KADRAN BİLGİSİ (SAĞ)'])),
            'alt-dis': int('ALT DIŞ' in str(row['KADRAN BİLGİSİ (SAĞ)'])),
            'ust-dis': int('ÜST DIŞ' in str(row['KADRAN BİLGİSİ (SAĞ)']))
            }

    new_row2 = {'img_name':str(row['HASTANO']) + "_RMLO", 
            'birads_class':right_class, 
            'kompozisyon':dict_kompozisyon[row['MEME KOMPOZİSYONU']], 
            'kadran':0, #sag icin
            'alt-ic': int('ALT İÇ' in str(row['KADRAN BİLGİSİ (SAĞ)'])),
            'ust-ic': int('ÜST İÇ' in str(row['KADRAN BİLGİSİ (SAĞ)'])),
            'merkez': int('MERKEZ' in str(row['KADRAN BİLGİSİ (SAĞ)'])),
            'alt-dis': int('ALT DIŞ' in str(row['KADRAN BİLGİSİ (SAĞ)'])),
            'ust-dis': int('ÜST DIŞ' in str(row['KADRAN BİLGİSİ (SAĞ)']))
            }

    new_row3 = {'img_name':str(row['HASTANO']) + "_LCC", 
            'birads_class':left_class, 
            'kompozisyon':dict_kompozisyon[row['MEME KOMPOZİSYONU']], 
            'kadran':1, #sol icin
            'alt-ic': int('ALT İÇ' in str(row['KADRAN BİLGİSİ (SOL)'])),
            'ust-ic': int('ÜST İÇ' in str(row['KADRAN BİLGİSİ (SOL)'])),
            'merkez': int('MERKEZ' in str(row['KADRAN BİLGİSİ (SOL)'])),
            'alt-dis': int('ALT DIŞ' in str(row['KADRAN BİLGİSİ (SOL)'])),
            'ust-dis': int('ÜST DIŞ' in str(row['KADRAN BİLGİSİ (SOL)']))
            }

    new_row4 = {'img_name':str(row['HASTANO']) + "_LMLO", 
            'birads_class':left_class, 
            'kompozisyon':dict_kompozisyon[row['MEME KOMPOZİSYONU']], 
            'kadran':1, #sol icin
            'alt-ic': int('ALT İÇ' in str(row['KADRAN BİLGİSİ (SOL)'])),
            'ust-ic': int('ÜST İÇ' in str(row['KADRAN BİLGİSİ (SOL)'])),
            'merkez': int('MERKEZ' in str(row['KADRAN BİLGİSİ (SOL)'])),
            'alt-dis': int('ALT DIŞ' in str(row['KADRAN BİLGİSİ (SOL)'])),
            'ust-dis': int('ÜST DIŞ' in str(row['KADRAN BİLGİSİ (SOL)']))
            }
        
    df_dictionary = pd.DataFrame([new_row,new_row2,new_row3,new_row4])
    new_data = pd.concat([new_data, df_dictionary], ignore_index=True)

new_data.to_csv(r"C:\Users\EMRE\Documents\Gits\BI-RADS-Classification\Codes\images_with_labels.csv", index=False)