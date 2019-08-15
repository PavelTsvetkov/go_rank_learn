import sgf

import sgf
with open("C:\\Users\\adast\\Downloads\\Kennychan-Ephemere.sgf") as f:
    collection = sgf.parse(f.read())

print(collection)


