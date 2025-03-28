# HUFFMAN AND SHANNON_FANO
# Consider a discrete memoryless source with symbols and statistics {0.125, 0.0625, 0.25, 0.0625, 0.125, 0.125, 0.25} for its output. Apply the Huffman and Shannon-Fano to this source.

# AIM:
  To compute the Average Codeword Length, Entropy, Efficiency, Redundancy, and Variance for a discrete memoryless source 
using Huffman and Shannon-Fano coding based on the given probabilities and codeword lengths.

## TOOLS REQUIRED :

Python: A versatile language for scientific computing and signal processing. <br />
NumPy & Matplotlib: Libraries for numerical operations and high-quality visualizations, essential for demonstrating sampling.
      
# PROGRAM:
```
import numpy as np
import math 
L  = 0
hs = 0
p = []
lk = []
n = int(input("Enter the number of Samples : "))
for i in range (n): 
    pr = float(input(f"Enter the probability of sample values {i + 1}: "))  
    p.append(pr)
for j in range (n): 
    l = float(input(f"Enter the length of the sample values {j + 1}: "))  
    lk.append(l)

for k in range (n):
    Avg1 = p[k] * lk[k]
    L = L + Avg1

for k in range (n):
    e = p[k] * math.log(1 / p[k], 2)
    hs = hs + e
hs = round(hs,3)

eff = hs / L
eff = round(eff,3)

red =  round(1 - eff,3) 

var = 0
for k in range(n):
    var1 = p[k] * (lk[k]-L)**2
    var = var + var1
var = round(var,3)
print()
print(f"Average Codeword Length is : {L}")
print(f"Entropy is : {hs}")
print(f"Efficiency is : {eff*100} %")
print(f"Redudancy is : {red}")
print(f"Variance is : {var}")
```


## OUTPUT:  
![Screenshot 2025-03-25 191943](https://github.com/user-attachments/assets/efa46457-a3cc-4c5a-b3c2-4795c1ed616c)

# CALCULATIONS:
![calculation](https://github.com/user-attachments/assets/4e132d78-c5ff-47b3-a6bc-57693e09ee10)
![calculations2](https://github.com/user-attachments/assets/dfd29c37-4c4d-437a-a134-862f7fe71f17)

## RESULT: 
For the given probabilities 
0.125,0.0625,0.25,0.0625,0.125,0.125,0.25. <br />
Average Codeword Length is : 2.625 <br />
Entropy is : 2.625 <br />
Efficiency is : 100.0 % <br />
Redudancy is : 0.0 <br />
Variance is : 0.484. <br />
