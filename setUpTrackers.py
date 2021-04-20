from functools import partial

def func(runFunction, path, list):
    output = runFunction.run(path, None)
    for item in output:
        list.append([item])
    return list

def createFunction(N):
    d = {f'add{N}': partial(func)}