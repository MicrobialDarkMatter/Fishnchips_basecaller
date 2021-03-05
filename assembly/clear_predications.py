with open('./assembly/assembly2.txt', 'r') as f:
    data = f.read()

data = data.split('\n')
with open('./assembly/predictions.txt', 'w') as f:
    for line in data:
        pred = []
        for char in line:
            if char in 'actg':
                pred.append(char)
        f.write(f'{"".join(pred)}\n')
