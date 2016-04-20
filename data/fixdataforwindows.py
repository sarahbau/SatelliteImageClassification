import os

for file in os.listdir(os.getcwd()):
    if file.endswith('.dat'):
        fi1 = open(file, 'rb')

        f1 = fi1.read()
        fi1.close()

        fi2 = open(file, 'wb')
        f1 = f1.replace(chr(13), '')
        fi2.write(f1)
        fi2.close()