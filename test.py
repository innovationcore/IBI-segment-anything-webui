import re
from PIL import Image
import numpy as np

compressed = '2438631F4T1733F7T1730F10T1728F11T1726F14T1725F14T1725F15T1723F16T1723F16T1723F16T1723F16T1723F16T1723F16T1724F15T1724F14T1726F13T1728F10T1730F9T1732F6T1734F4T275336F12T1725F16T1722F18T1720F21T1717F23T1716F24T1714F26T1713F27T1712F27T1711F28T1711F29T1711F28T1711F28T1711F28T1711F28T1712F26T1713F26T1713F26T1713F26T1714F25T1714F25T1715F23T1717F21T1719F18T1723F15T1727F10T1733F4T2424762F'
imgx = 1739
imgy = 3000

def generate_overlay(compressed, imgx, imgy, filename):
    counts = []
    values = []
    splits = re.split('(T|F)', compressed)
    #print(splits)
    pixel_color = []

    for each in splits:
        if each == 'T':
            values.append('T')
        elif each == 'F':
            values.append('F')
        elif each != 'T' and each != 'F' and each != '':
            counts.append(int(each))

    i = 0
    for each in counts:
        for pixel in range(each):
            if (i % 2 == 0):
                pixel_color.append(False)  # black
            else:
                pixel_color.append(True)  # white
        i += 1

    print(len(pixel_color))
    pixel_color = np.array(pixel_color)


    overlay = Image.fromarray(pixel_color.reshape((imgy, imgx)).astype('uint8') * 255)
    overlay.save('overlays/' + filename + '.png', 'PNG')
    return overlay

overlay = generate_overlay(compressed, imgx, imgy, 'test')

#print(overlay)