base_map = {
    0: 'P',
    1: 'A',
    2: 'C',
    3: 'G',
    4: 'T',
    5: 'S',
    6: 'E'
}    

def convert_to_base_strings(base_lists):
    base_strings = []
    for base_list in base_lists:
        base_string = convert_to_base_string(base_list)
        base_strings.append(base_string)
    return base_strings

def convert_to_base_string(base_list):
    base_string = ''
    for base_token in base_list:
        base_string += base_map[base_token]
    return base_string
                