import os
os.chdir("C:/Users/sarth/OneDrive/Desktop/Rutgers/Sem 2/ML/Project/gestures_2")
for i in range(10,36):
    os.rename(chr(i+87),str(i))