import os

path = r"C:\Users\GoldSpot_Cloudberry\OneDrive - Goldspot Discoveries Inc\Documents\Goldspot\Core Images\Digit Classification\train"

files = []
# r=root, d=directories, f=files
for r, d, f in os.walk(path):
    for file in f:
        if '.png' in file:
            files.append(os.path.join(r, file))

count = 1
for file in files:
    os.rename(file, path + "\Example " + str(count) + ".png")
    count += 1

print("Done process")
