import csv

import definitons

headers = [
    ['image', 'compare_with', 'result', 'is_same_person', 'time', 'dataset', 'type'],
    ['folder', 'image', 'was_found', 'time', 'dataset', 'type']
]


def write(data, file_name, header_type):
    header = headers[0]
    if header_type == 'complex':
        header = headers[1]
    with open(definitons.root_dir + '\\' + file_name + '.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)


def read(file_name):
    with open(definitons.root_dir + '\\' + file_name + '.csv', 'r', encoding='UTF8', newline='') as f:
        reader = csv.reader(f)
        for i, line in enumerate(reader):
            print('line[{}] = {}'.format(i, line))
