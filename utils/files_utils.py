import csv

import definitons

headers = [
    ['image', 'compare_with', 'result', 'is_same_person', 'time', 'dataset', 'type1', 'type2'],
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


def read(file_name, file_type='usual'):
    correct = 0
    incorrect = 0
    time = 0.0
    with open(definitons.root_dir + '\\' + file_name + '.csv', 'r', encoding='UTF8', newline='') as f:
        reader = csv.reader(f)
        for i, line in enumerate(reader):
            if i != 0:
                if file_type == 'usual':
                    if line[2] == line[3]:
                        correct += 1
                    else:
                        incorrect += 1
                    time += float(line[4])
                else:
                    if line[2] == 'True':
                        correct += 1
                    else:
                        incorrect += 1
                    time += float(line[3])
                print('line[{}] = {}'.format(i, line))
    print('Correct ', correct)
    print('Incorrect ', incorrect)
    print('Total time ', time)


def main():
    read('vgg_model')


if __name__ == "__main__":
    main()
