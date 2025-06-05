import pandas as pd

def stringFlatten(strList):
# takes a list elements that can be strings or list of strings or nan or a list of any of these things and returns a string
    if type(strList) != list and pd.isna(strList):
        return ''
    if not( type(strList) == str or type(strList) == list):
        raise(ValueError, "argument contains something else than strings")
    if type(strList) == str:
        return strList
    elif type(strList) == list:
        toReturn = ''
        for elem in strList :
            if stringFlatten(elem) == '':
                continue
            else:
                toReturn += "," + stringFlatten(elem)
        return toReturn
    