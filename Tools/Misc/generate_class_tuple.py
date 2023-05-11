

CLASS_TXT_PATH = r'E:\PNID_Data\2023_0228\SymbolClass_Class.txt'

with open(CLASS_TXT_PATH, 'r') as f:
    lines = f.readlines()
    classes = [line.split('|')[1].replace('\n', '') for line in lines]
    
    result = 'classes = ('
    for cls in classes[:-1]:
        result += f"'{cls}', "
    
    result += f"'{classes[-1]}', 'text)"
    print(result)        
    