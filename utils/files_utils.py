import csv

headers = [
    ['image', 'compare_with', 'result', 'is_same_person', 'time', 'dataset', 'type1', 'type2'],
    ['folder', 'image', 'was_found', 'time', 'dataset', 'type']
]


def write(data, file_name, header_type, path):
    header = headers[0]
    if header_type == 'complex':
        header = headers[1]
    path = path.split('.txt')[0]
    with open(path + '_' + file_name + '.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
