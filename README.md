# 今回解く最適化問題  
・製品Pn（n=1~10）を生産，販売して利益を最大化する．  
・製品Pnを1つ生産するには，原料M1がn1 kg，原料M2がn2 kg，原料M3がn3 kg，生産コストCn1，販売コストCn2が必要．  
・原料M1はM1 kgまで，原料M2はM2 kgまで，原料M3はM3 kgまでしか利用できない．  
・利益を最大化するには，各製品をどの程度生産すれば良いか？  


```python
import warnings
warnings.simplefilter('ignore')
```


```python
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
from pulp import *
```


```python
df = pd.read_csv('../Data/Data_for_optimization_tutorial.csv')
df.set_index('Products', inplace=True)
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Material1</th>
      <th>Material2</th>
      <th>Material3</th>
      <th>Production_cost</th>
      <th>Sales_cost</th>
      <th>Sales</th>
    </tr>
    <tr>
      <th>Products</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>P01</th>
      <td>8</td>
      <td>5</td>
      <td>7</td>
      <td>100</td>
      <td>541</td>
      <td>2500</td>
    </tr>
    <tr>
      <th>P02</th>
      <td>35</td>
      <td>15</td>
      <td>8</td>
      <td>200</td>
      <td>222</td>
      <td>4500</td>
    </tr>
    <tr>
      <th>P03</th>
      <td>42</td>
      <td>25</td>
      <td>9</td>
      <td>100</td>
      <td>368</td>
      <td>6321</td>
    </tr>
    <tr>
      <th>P04</th>
      <td>36</td>
      <td>34</td>
      <td>5</td>
      <td>100</td>
      <td>487</td>
      <td>852</td>
    </tr>
    <tr>
      <th>P05</th>
      <td>12</td>
      <td>6</td>
      <td>11</td>
      <td>56</td>
      <td>956</td>
      <td>1024</td>
    </tr>
    <tr>
      <th>P06</th>
      <td>10</td>
      <td>15</td>
      <td>26</td>
      <td>44</td>
      <td>321</td>
      <td>678</td>
    </tr>
    <tr>
      <th>P07</th>
      <td>15</td>
      <td>8</td>
      <td>32</td>
      <td>77</td>
      <td>25</td>
      <td>444</td>
    </tr>
    <tr>
      <th>P08</th>
      <td>22</td>
      <td>4</td>
      <td>42</td>
      <td>87</td>
      <td>36</td>
      <td>4444</td>
    </tr>
    <tr>
      <th>P09</th>
      <td>8</td>
      <td>3</td>
      <td>26</td>
      <td>250</td>
      <td>654</td>
      <td>8951</td>
    </tr>
    <tr>
      <th>P10</th>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>222</td>
      <td>66</td>
      <td>3651</td>
    </tr>
  </tbody>
</table>
</div>




```python
M1 = 300
M2 = 100
M3 = 150

# 数理モデル
m = LpProblem('Model', sense=LpMaximize)
# 変数
x_1 = LpVariable('Num_of_P01', lowBound=0, cat=LpInteger )
x_2 = LpVariable('Num_of_P02', lowBound=0, cat=LpInteger )
x_3 = LpVariable('Num_of_P03', lowBound=0, cat=LpInteger )
x_4 = LpVariable('Num_of_P04', lowBound=0, cat=LpInteger )
x_5 = LpVariable('Num_of_P05', lowBound=0, cat=LpInteger )
x_6 = LpVariable('Num_of_P06', lowBound=0, cat=LpInteger )
x_7 = LpVariable('Num_of_P07', lowBound=0, cat=LpInteger )
x_8 = LpVariable('Num_of_P08', lowBound=0, cat=LpInteger )
x_9 = LpVariable('Num_of_P09', lowBound=0, cat=LpInteger )
x_10 = LpVariable('Num_of_P10', lowBound=0, cat=LpInteger )
# 目的関数
m += (df['Sales']['P01']-df['Production_cost']['P01']-df['Sales_cost']['P01']) * x_1 + (df['Sales']['P02']-df['Production_cost']['P02']-df['Sales_cost']['P02']) * x_2 + \
    (df['Sales']['P03']-df['Production_cost']['P03']-df['Sales_cost']['P03']) * x_3 + (df['Sales']['P04']-df['Production_cost']['P04']-df['Sales_cost']['P04']) * x_4 + \
    (df['Sales']['P05']-df['Production_cost']['P01']-df['Sales_cost']['P05']) * x_5 + (df['Sales']['P06']-df['Production_cost']['P06']-df['Sales_cost']['P06']) * x_6 + \
    (df['Sales']['P07']-df['Production_cost']['P07']-df['Sales_cost']['P07']) * x_7 + (df['Sales']['P08']-df['Production_cost']['P08']-df['Sales_cost']['P08']) * x_8 + \
    (df['Sales']['P09']-df['Production_cost']['P09']-df['Sales_cost']['P09']) * x_9 + (df['Sales']['P10']-df['Production_cost']['P10']-df['Sales_cost']['P10']) * x_10
# 制約条件
m += df['Material1']['P01'] * x_1 + df['Material1']['P02'] * x_2 + \
    df['Material1']['P03'] * x_3 + df['Material1']['P04'] * x_4 + \
    df['Material1']['P05'] * x_5 + df['Material1']['P06'] * x_6 + \
    df['Material1']['P07'] * x_7 + df['Material1']['P08'] * x_8 + \
    df['Material1']['P09'] * x_9 + df['Material1']['P10'] * x_10 <= M1
m += df['Material2']['P01'] * x_1 + df['Material2']['P02'] * x_2 + \
    df['Material2']['P03'] * x_3 + df['Material2']['P04'] * x_4 + \
    df['Material2']['P05'] * x_5 + df['Material2']['P06'] * x_6 + \
    df['Material2']['P07'] * x_7 + df['Material2']['P08'] * x_8 + \
    df['Material2']['P09'] * x_9 + df['Material2']['P10'] * x_10 <= M2
m += df['Material3']['P01'] * x_1 + df['Material3']['P02'] * x_2 + \
    df['Material3']['P03'] * x_3 + df['Material3']['P04'] * x_4 + \
    df['Material3']['P05'] * x_5 + df['Material3']['P06'] * x_6 + \
    df['Material3']['P07'] * x_7 + df['Material3']['P08'] * x_8 + \
    df['Material3']['P09'] * x_9 + df['Material3']['P10'] * x_10 <= M3

m.solve() # ソルバーの実行
print(m)
print('P01 : ', int(value(x_1)), \
      '\nP02 : ', int(value(x_2)), \
      '\nP03 : ', int(value(x_3)), \
      '\nP04 : ', int(value(x_4)), \
      '\nP05 : ', int(value(x_5)), \
      '\nP06 : ', int(value(x_6)), \
      '\nP07 : ', int(value(x_7)), \
      '\nP08 : ', int(value(x_8)), \
      '\nP09 : ', int(value(x_9)), \
      '\nP10 : ', int(value(x_10)), \
      '\nTotal sales : ', int(value(m.objective)))
```

    Welcome to the CBC MILP Solver 
    Version: 2.10.3 
    Build Date: Dec 15 2019 
    
    command line - /usr/local/lib/python3.8/site-packages/pulp/solverdir/cbc/linux/64/cbc /tmp/d9250458c68c4c4384378d9d263737bc-pulp.mps -max -timeMode elapsed -branch -printingOptions all -solution /tmp/d9250458c68c4c4384378d9d263737bc-pulp.sol (default strategy 1)
    At line 2 NAME          MODEL
    At line 3 ROWS
    At line 8 COLUMNS
    At line 69 RHS
    At line 73 BOUNDS
    At line 84 ENDATA
    Problem MODEL has 3 rows, 10 columns and 30 elements
    Coin0008I MODEL read with 0 errors
    Option for timeMode changed from cpu to elapsed
    Continuous objective value is 62403.1 - 0.00 seconds
    Cgl0004I processed model has 3 rows, 9 columns (9 integer (0 of which binary)) and 27 elements
    Cutoff increment increased from 1e-05 to 0.9999
    Cbc0012I Integer solution of -59813 found by DiveCoefficient after 0 iterations and 0 nodes (0.00 seconds)
    Cbc0038I Full problem 3 rows 9 columns, reduced to 2 rows 2 columns
    Cbc0012I Integer solution of -60261 found by DiveCoefficient after 106 iterations and 0 nodes (0.01 seconds)
    Cbc0031I 3 added rows had average density of 5
    Cbc0013I At root node, 3 cuts changed objective from -62403.106 to -60261.005 in 29 passes
    Cbc0014I Cut generator 0 (Probing) - 2 row cuts average 2.0 elements, 2 column cuts (2 active)  in 0.000 seconds - new frequency is 1
    Cbc0014I Cut generator 1 (Gomory) - 80 row cuts average 5.0 elements, 0 column cuts (0 active)  in 0.000 seconds - new frequency is 1
    Cbc0014I Cut generator 2 (Knapsack) - 0 row cuts average 0.0 elements, 0 column cuts (0 active)  in 0.000 seconds - new frequency is -100
    Cbc0014I Cut generator 3 (Clique) - 0 row cuts average 0.0 elements, 0 column cuts (0 active)  in 0.000 seconds - new frequency is -100
    Cbc0014I Cut generator 4 (MixedIntegerRounding2) - 0 row cuts average 0.0 elements, 0 column cuts (0 active)  in 0.000 seconds - new frequency is -100
    Cbc0014I Cut generator 5 (FlowCover) - 1 row cuts average 2.0 elements, 0 column cuts (0 active)  in 0.000 seconds - new frequency is -100
    Cbc0014I Cut generator 6 (TwoMirCuts) - 38 row cuts average 4.7 elements, 0 column cuts (0 active)  in 0.000 seconds - new frequency is 1
    Cbc0014I Cut generator 7 (ZeroHalf) - 1 row cuts average 9.0 elements, 0 column cuts (0 active)  in 0.000 seconds - new frequency is -100
    Cbc0001I Search completed - best objective -60261, took 106 iterations and 0 nodes (0.01 seconds)
    Cbc0035I Maximum depth 0, 0 variables fixed on reduced cost
    Cuts at root node changed objective from -62403.1 to -60261
    Probing was tried 29 times and created 4 cuts of which 0 were active after adding rounds of cuts (0.000 seconds)
    Gomory was tried 29 times and created 80 cuts of which 0 were active after adding rounds of cuts (0.000 seconds)
    Knapsack was tried 29 times and created 0 cuts of which 0 were active after adding rounds of cuts (0.000 seconds)
    Clique was tried 29 times and created 0 cuts of which 0 were active after adding rounds of cuts (0.000 seconds)
    MixedIntegerRounding2 was tried 29 times and created 0 cuts of which 0 were active after adding rounds of cuts (0.000 seconds)
    FlowCover was tried 29 times and created 1 cuts of which 0 were active after adding rounds of cuts (0.000 seconds)
    TwoMirCuts was tried 29 times and created 38 cuts of which 0 were active after adding rounds of cuts (0.000 seconds)
    ZeroHalf was tried 1 times and created 1 cuts of which 0 were active after adding rounds of cuts (0.000 seconds)
    
    Result - Optimal solution found
    
    Objective value:                60261.00000000
    Enumerated nodes:               0
    Total iterations:               106
    Time (CPU seconds):             0.00
    Time (Wallclock seconds):       0.01
    
    Option for printingOptions changed from normal to all
    Total time (CPU seconds):       0.00   (Wallclock seconds):       0.01
    
    Model:
    MAXIMIZE
    1859*Num_of_P01 + 4078*Num_of_P02 + 5853*Num_of_P03 + 265*Num_of_P04 + -32*Num_of_P05 + 313*Num_of_P06 + 342*Num_of_P07 + 4321*Num_of_P08 + 8047*Num_of_P09 + 3363*Num_of_P10 + 0
    SUBJECT TO
    _C1: 8 Num_of_P01 + 35 Num_of_P02 + 42 Num_of_P03 + 36 Num_of_P04
     + 12 Num_of_P05 + 10 Num_of_P06 + 15 Num_of_P07 + 22 Num_of_P08
     + 8 Num_of_P09 + 7 Num_of_P10 <= 300
    
    _C2: 5 Num_of_P01 + 15 Num_of_P02 + 25 Num_of_P03 + 34 Num_of_P04
     + 6 Num_of_P05 + 15 Num_of_P06 + 8 Num_of_P07 + 4 Num_of_P08 + 3 Num_of_P09
     + 7 Num_of_P10 <= 100
    
    _C3: 7 Num_of_P01 + 8 Num_of_P02 + 9 Num_of_P03 + 5 Num_of_P04 + 11 Num_of_P05
     + 26 Num_of_P06 + 32 Num_of_P07 + 42 Num_of_P08 + 26 Num_of_P09
     + 7 Num_of_P10 <= 150
    
    VARIABLES
    0 <= Num_of_P01 Integer
    0 <= Num_of_P02 Integer
    0 <= Num_of_P03 Integer
    0 <= Num_of_P04 Integer
    0 <= Num_of_P05 Integer
    0 <= Num_of_P06 Integer
    0 <= Num_of_P07 Integer
    0 <= Num_of_P08 Integer
    0 <= Num_of_P09 Integer
    0 <= Num_of_P10 Integer
    
    P01 :  0 
    P02 :  0 
    P03 :  1 
    P04 :  0 
    P05 :  0 
    P06 :  0 
    P07 :  0 
    P08 :  0 
    P09 :  3 
    P10 :  9 
    Total sales :  60261

