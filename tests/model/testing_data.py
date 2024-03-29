blur_percents = [1, 2, 3, 4, 5]
labels = ['correct', 'incorrect']

datasets = [
    'images\\datasets\\usual\\first_set\\',  # 0
    'images\\datasets\\usual\\second_set\\Final Training Images',  # 1
    'images\\datasets\\usual\\second_set\\Final Testing Images',  # 2

    'images\\datasets\\noised\\first_set\\',  # 3
    'images\\datasets\\noised\\second_set\\Final Training Images',  # 4
    'images\\datasets\\noised\\second_set\\Final Testing Images',  # 5

    'images\\datasets\\blurred\\first_set\\',  # 6
    'images\\datasets\\blurred\\second_set\\Final Training Images',  # 7
    'images\\datasets\\blurred\\second_set\\Final Testing Images',  # 8
]

test_data = [
    # first_set (30)
    ['1\\1.jpg', 'usual', 'first_set', datasets[0]],
    ['1\\5.jpg', 'usual', 'first_set', datasets[0]],
    ['1\\9.jpg', 'usual', 'first_set', datasets[0]],
    ['1\\1.jpg', 'usual', 'first_set', datasets[0]],
    ['1\\2.jpg', 'usual', 'first_set', datasets[0]],
    ['1\\5.jpg', 'usual', 'first_set', datasets[0]],
    ['1\\6.jpg', 'usual', 'first_set', datasets[0]],
    ['1\\8.jpg', 'usual', 'first_set', datasets[0]],
    ['1\\9.jpg', 'usual', 'first_set', datasets[0]],
    ['1\\10.jpg', 'usual', 'first_set', datasets[0]],
    ['12\\7.jpg', 'usual', 'first_set', datasets[0]],
    ['12\\7.jpg', 'usual', 'first_set', datasets[0]],
    ['12\\5.jpg', 'usual', 'first_set', datasets[0]],
    ['12\\4.jpg', 'usual', 'first_set', datasets[0]],
    ['12\\3.jpg', 'usual', 'first_set', datasets[0]],
    ['12\\2.jpg', 'usual', 'first_set', datasets[0]],
    ['12\\1.jpg', 'usual', 'first_set', datasets[0]],
    ['12\\9.jpg', 'usual', 'first_set', datasets[0]],
    ['40\\8.jpg', 'usual', 'first_set', datasets[0]],
    ['40\\7.jpg', 'usual', 'first_set', datasets[0]],
    ['40\\6.jpg', 'usual', 'first_set', datasets[0]],
    ['40\\5.jpg', 'usual', 'first_set', datasets[0]],
    ['40\\4.jpg', 'usual', 'first_set', datasets[0]],
    ['40\\3.jpg', 'usual', 'first_set', datasets[0]],
    ['40\\2.jpg', 'usual', 'first_set', datasets[0]],
    ['40\\1.jpg', 'usual', 'first_set', datasets[0]],
    ['6\\7.jpg', 'usual', 'first_set', datasets[0]],
    ['6\\8.jpg', 'usual', 'first_set', datasets[0]],
    ['6\\9.jpg', 'usual', 'first_set', datasets[0]],
    ['6\\6.jpg', 'usual', 'first_set', datasets[0]],
    ['6\\5.jpg', 'usual', 'first_set', datasets[0]],
    ['6\\2.jpg', 'usual', 'first_set', datasets[0]],

    # second_set\\Final Training Image (15)
    ['face1\\1face1.jpg', 'usual', 'second_set', datasets[1]],
    ['face1\\2face1.jpg', 'usual', 'second_set', datasets[1]],
    ['face2\\10face2.jpg', 'usual', 'second_set', datasets[1]],
    ['face2\\12face2.jpg', 'usual', 'second_set', datasets[1]],
    ['face1\\1face1.jpg', 'usual', 'second_set', datasets[1]],
    ['face1\\2face1.jpg', 'usual', 'second_set', datasets[1]],
    ['face3\\10face3.jpg', 'usual', 'second_set', datasets[1]],
    ['face3\\12face3.jpg', 'usual', 'second_set', datasets[1]],
    ['face4\\1face4.jpg', 'usual', 'second_set', datasets[1]],
    ['face4\\2face4.jpg', 'usual', 'second_set', datasets[1]],
    ['face5\\10face5.jpg', 'usual', 'second_set', datasets[1]],
    ['face5\\12face5.jpg', 'usual', 'second_set', datasets[1]],
    ['face6\\1face6.jpg', 'usual', 'second_set', datasets[1]],
    ['face6\\2face6.jpg', 'usual', 'second_set', datasets[1]],
    ['face5\\10face5.jpg', 'usual', 'second_set', datasets[1]],
    ['face4\\12face4.jpg', 'usual', 'second_set', datasets[1]],

    # second_set\\Final Testing Image (15)
    ['face16\\1face16.jpg', 'usual', 'second_set', datasets[2]],
    ['face16\\3face16.jpg', 'usual', 'second_set', datasets[2]],
    ['face12\\1face12.jpg', 'usual', 'second_set', datasets[2]],
    ['face12\\2face12.jpg', 'usual', 'second_set', datasets[2]],
    ['face10\\2face10.jpg', 'usual', 'second_set', datasets[2]],
    ['face10\\2face10.jpg', 'usual', 'second_set', datasets[2]],
    ['face13\\1face13.jpg', 'usual', 'second_set', datasets[2]],
    ['face13\\2face13.jpg', 'usual', 'second_set', datasets[2]],
    ['face8\\3face8.jpg', 'usual', 'second_set', datasets[2]],
    ['face8\\4face8.jpg', 'usual', 'second_set', datasets[2]],
    ['face7\\1face7.jpg', 'usual', 'second_set', datasets[2]],
    ['face7\\2face7.jpg', 'usual', 'second_set', datasets[2]],
    ['face9\\1face9.jpg', 'usual', 'second_set', datasets[2]],
    ['face9\\2face9.jpg', 'usual', 'second_set', datasets[2]],
    ['face4\\1face4.jpg', 'usual', 'second_set', datasets[2]],
    ['face4\\2face4.jpg', 'usual', 'second_set', datasets[2]],

    # first_set MIXED (30)
    ['1\\1.jpg', 'noised', 'first_set', datasets[3]],
    ['1\\5.jpg', 'noised', 'first_set', datasets[3]],
    ['1\\9.jpg', 'noised', 'first_set', datasets[3]],
    ['5\\1.jpg', 'noised', 'first_set', datasets[3]],
    ['5\\2.jpg', 'noised', 'first_set', datasets[3]],
    ['6\\5.jpg', 'noised', 'first_set', datasets[3]],
    ['6\\6.jpg', 'noised', 'first_set', datasets[3]],
    ['7\\8.jpg', 'noised', 'first_set', datasets[3]],
    ['8\\9.jpg', 'blurred', 'first_set', datasets[6]],
    ['8\\10.jpg', 'noised', 'first_set', datasets[3]],
    ['12\\7.jpg', 'blurred', 'first_set', datasets[6]],
    ['12\\7.jpg', 'noised', 'first_set', datasets[3]],
    ['12\\5.jpg', 'blurred', 'first_set', datasets[6]],
    ['33\\4.jpg', 'noised', 'first_set', datasets[3]],
    ['33\\3.jpg', 'blurred', 'first_set', datasets[6]],
    ['23\\2.jpg', 'noised', 'first_set', datasets[3]],
    ['34\\1.jpg', 'blurred', 'first_set', datasets[6]],
    ['39\\9.jpg', 'noised', 'first_set', datasets[3]],
    ['40\\8.jpg', 'blurred', 'first_set', datasets[6]],
    ['26\\7.jpg', 'noised', 'first_set', datasets[3]],
    ['26\\6.jpg', 'blurred', 'first_set', datasets[6]],
    ['15\\5.jpg', 'noised', 'first_set', datasets[3]],
    ['15\\4.jpg', 'blurred', 'first_set', datasets[6]],
    ['19\\3.jpg', 'noised', 'first_set', datasets[3]],
    ['17\\2.jpg', 'blurred', 'first_set', datasets[6]],
    ['6\\1.jpg', 'noised', 'first_set', datasets[3]],
    ['6\\7.jpg', 'blurred', 'first_set', datasets[6]],
    ['13\\8.jpg', 'noised', 'first_set', datasets[3]],
    ['13\\9.jpg', 'blurred', 'first_set', datasets[6]],
    ['23\\6.jpg', 'noised', 'first_set', datasets[3]],
    ['22\\5.jpg', 'blurred', 'first_set', datasets[6]],
    ['22\\2.jpg', 'noised', 'first_set', datasets[3]],

    # second_set\\Final Training Image MIXED (15)
    ['face1\\1face1.jpg', 'noised', 'second_set', datasets[4]],
    ['face1\\2face1.jpg', 'blurred', 'second_set', datasets[7]],
    ['face2\\10face2.jpg', 'noised', 'second_set', datasets[4]],
    ['face2\\12face2.jpg', 'blurred', 'second_set', datasets[7]],
    ['face1\\1face1.jpg', 'noised', 'second_set', datasets[4]],
    ['face1\\2face1.jpg', 'blurred', 'second_set', datasets[7]],
    ['face3\\10face3.jpg', 'noised', 'second_set', datasets[4]],
    ['face3\\12face3.jpg', 'blurred', 'second_set', datasets[7]],
    ['face4\\1face4.jpg', 'noised', 'second_set', datasets[4]],
    ['face4\\2face4.jpg', 'blurred', 'second_set', datasets[7]],
    ['face5\\10face5.jpg', 'noised', 'second_set', datasets[4]],
    ['face5\\12face5.jpg', 'blurred', 'second_set', datasets[7]],
    ['face6\\1face6.jpg', 'noised', 'second_set', datasets[4]],
    ['face6\\2face6.jpg', 'blurred', 'second_set', datasets[7]],
    ['face5\\10face5.jpg', 'noised', 'second_set', datasets[4]],
    ['face4\\12face4.jpg', 'blurred', 'second_set', datasets[7]],

    # second_set\\Final Testing Image MIXED (15)
    ['face16\\1face16.jpg', 'noised', 'second_set', datasets[5]],
    ['face16\\3face16.jpg', 'blurred', 'second_set', datasets[8]],
    ['face12\\1face12.jpg', 'noised', 'second_set', datasets[5]],
    ['face12\\2face12.jpg', 'blurred', 'second_set', datasets[8]],
    ['face10\\2face10.jpg', 'noised', 'second_set', datasets[5]],
    ['face10\\2face10.jpg', 'blurred', 'second_set', datasets[8]],
    ['face13\\1face13.jpg', 'noised', 'second_set', datasets[5]],
    ['face13\\2face13.jpg', 'blurred', 'second_set', datasets[8]],
    ['face8\\3face8.jpg', 'noised', 'second_set', datasets[5]],
    ['face8\\4face8.jpg', 'blurred', 'second_set', datasets[8]],
    ['face7\\1face7.jpg', 'noised', 'second_set', datasets[5]],
    ['face7\\2face7.jpg', 'blurred', 'second_set', datasets[8]],
    ['face9\\1face9.jpg', 'noised', 'second_set', datasets[8]],
    ['face9\\2face9.jpg', 'blurred', 'second_set', datasets[8]],
    ['face4\\1face4.jpg', 'noised', 'second_set', datasets[5]],
    ['face4\\2face4.jpg', 'blurred', 'second_set', datasets[8]],
]
