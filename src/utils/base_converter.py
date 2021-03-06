base_map = {
    0: 'P',
    1: 'A',
    2: 'C',
    3: 'G',
    4: 'T',
    5: 'S',
    6: 'E'
}    

"""
Param int_base_lists: list of int base lists (e.g. [[1,2,3],[1,1,1],[2,2,2]], corresponding to ['ACG',...])
Param skip_tokens: list of tokens to skip during conversion. Defualt S,E,P - skips start, end and padding tokens
Returns: list of base strings
"""
def convert_to_base_strings(int_base_lists, skip_tokens=['S', 'E', 'P']):
    base_strings = []
    for int_base_list in int_base_lists:
        base_string = convert_to_base_string(int_base_list)
        base_strings.append(base_string)
    return base_strings

def convert_to_base_string(int_base_list, skip_tokens=['S', 'E', 'P']):
    base_string = ''
    for int_base_token in int_base_list:
        str_base_token = base_map[int_base_token]
        if str_base_token not in skip_tokens:
            base_string += str_base_token
    return base_string
                