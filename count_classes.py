# Count the number of different varieties

red = 0
dream = 0
poinsettia = 0

fr=open('VOCdevkit/VOC2007/ImageSets/Main/test.txt','r', encoding='UTF-8')
lines=fr.readlines()
for line in lines:
    if "Red" in line:
        red += 1
    elif "Dream" in line:
        dream += 1
    elif "Poinsettia" in line:
        poinsettia += 1

fr.close()
print("red:%s\ndream:%s\npoinsettia:%s\ntotal:%s" % (red, dream, poinsettia, (red+dream+poinsettia)))