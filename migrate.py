from test import FRUIT

def migrate_file():
    with open('label.txt', 'w', encoding='UTF-8') as myfile:
        for label in FRUIT.keys():
            myfile.writelines(label + ':' + FRUIT[label] + '\n')

def read_data():
    label = {}
    with open('label.txt', encoding='UTF-8') as myfile:
        for line in myfile.readlines():
            arg = line.replace('\n','').split(':')
            label[arg[0]] = arg[1]
    return label

migrate_file()
print(read_data())