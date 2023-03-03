from pathlib import Path
from tqdm import tqdm
import random

XML_PATH = r'E:\PNID_Data\2023_0228\XML'

def split_tran_val_test(train_size:int, val_size:int, test_size:int):
    '''
    train_size, val_size, test_size: 전체 100 기준 각 셋의 비율. 총합은 100 이어야 함. ex) 90, 5, 5
    '''
    if train_size + val_size + test_size != 100: 
        print('Total of sizes must be 100.')
        return
    train_ratio = train_size / 100
    val_ratio = val_size / 100
    test_ratio = test_size / 100

    print(train_ratio, val_ratio, test_ratio)    

    # train, val, test = set(), set(), set()
    total_list = list(Path(XML_PATH).glob('*.xml'))
    total_count = len(total_list)
    train_count = int(train_ratio * total_count)
    val_count = int(val_ratio * total_count)
    test_count = int(test_ratio * total_count)

    print(f'Total : {total_count}')
    print(f'{train_count}:{val_count}:{test_count}')

    total_set = set(total_list)
    val = set(random.sample(total_set, val_count))
    no_val = total_set - val
    test = set(random.sample(no_val, test_count))
    train = no_val - test

    return train, val, test

def get_val_test_names(train, val, test):
    count = 0
    val_names = ''
    for i in sorted(list(val)):
        val_names += f'\'{i.stem}\', '
        count += 1
        if count % 4 == 0:
            val_names += '\n'
    
    count = 0
    test_names = ''
    for i in sorted(list(test)):
        test_names += f'\'{i.stem}\', '
        count += 1
        if count % 4 == 0:
            test_names += '\n'
    
    return val_names, test_names

if __name__ == '__main__':
    train, val, test = split_tran_val_test(90, 5, 5)

    val_names, test_names = get_val_test_names(val, test)

    print('Validation Set List : ')
    print(val_names)
    print('Test Set List : ')
    print(test_names)