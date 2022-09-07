---
layout: single
title:  "Data Visualization 20220907"
---

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```


```python
key = pd.read_csv(r"C:\Users\윤철환\Desktop\Louis\VS_practice2\key_stats.csv")
key.head()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>player_name</th>
      <th>club</th>
      <th>position</th>
      <th>minutes_played</th>
      <th>match_played</th>
      <th>goals</th>
      <th>assists</th>
      <th>distance_covered</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Courtois</td>
      <td>Real Madrid</td>
      <td>Goalkeeper</td>
      <td>1230</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>64.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Vinícius Júnior</td>
      <td>Real Madrid</td>
      <td>Forward</td>
      <td>1199</td>
      <td>13</td>
      <td>4</td>
      <td>6</td>
      <td>133.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Benzema</td>
      <td>Real Madrid</td>
      <td>Forward</td>
      <td>1106</td>
      <td>12</td>
      <td>15</td>
      <td>1</td>
      <td>121.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Modrić</td>
      <td>Real Madrid</td>
      <td>Midfielder</td>
      <td>1077</td>
      <td>13</td>
      <td>0</td>
      <td>4</td>
      <td>124.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Éder Militão</td>
      <td>Real Madrid</td>
      <td>Defender</td>
      <td>1076</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>110.4</td>
    </tr>
  </tbody>
</table>
</div>




```python
data = {
    '이름' : ['채치수', '정대만', '송태섭', '서태웅', '강백호', '변덕규', '황태산', '윤대협'],
    '학교' : ['북산고', '북산고', '북산고', '북산고', '북산고', '능남고', '능남고', '능남고'],
    '키' : [197, 184, 168, 187, 188, 202, 188, 190],
    '국어' : [90, 40, 80, 40, 15, 80, 55, 100],
    '영어' : [85, 35, 75, 60, 20, 100, 65, 85],
    '수학' : [100, 50, 70, 70, 10, 95, 45, 90],
    '과학' : [95, 55, 80, 75, 35, 85, 40, 95],
    '사회' : [85, 25, 75, 80, 10, 80, 35, 95],
    'SW특기' : ['Python', 'Java', 'Javascript', '', '', 'C', 'PYTHON', 'C#']
}
student = pd.DataFrame(data,
                       index = range(1,9,1))
student['학년'] = [3,3,1,2,2,3,1,2]
student
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>이름</th>
      <th>학교</th>
      <th>키</th>
      <th>국어</th>
      <th>영어</th>
      <th>수학</th>
      <th>과학</th>
      <th>사회</th>
      <th>SW특기</th>
      <th>학년</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>채치수</td>
      <td>북산고</td>
      <td>197</td>
      <td>90</td>
      <td>85</td>
      <td>100</td>
      <td>95</td>
      <td>85</td>
      <td>Python</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>정대만</td>
      <td>북산고</td>
      <td>184</td>
      <td>40</td>
      <td>35</td>
      <td>50</td>
      <td>55</td>
      <td>25</td>
      <td>Java</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>송태섭</td>
      <td>북산고</td>
      <td>168</td>
      <td>80</td>
      <td>75</td>
      <td>70</td>
      <td>80</td>
      <td>75</td>
      <td>Javascript</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>서태웅</td>
      <td>북산고</td>
      <td>187</td>
      <td>40</td>
      <td>60</td>
      <td>70</td>
      <td>75</td>
      <td>80</td>
      <td></td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>강백호</td>
      <td>북산고</td>
      <td>188</td>
      <td>15</td>
      <td>20</td>
      <td>10</td>
      <td>35</td>
      <td>10</td>
      <td></td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>변덕규</td>
      <td>능남고</td>
      <td>202</td>
      <td>80</td>
      <td>100</td>
      <td>95</td>
      <td>85</td>
      <td>80</td>
      <td>C</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>황태산</td>
      <td>능남고</td>
      <td>188</td>
      <td>55</td>
      <td>65</td>
      <td>45</td>
      <td>40</td>
      <td>35</td>
      <td>PYTHON</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>윤대협</td>
      <td>능남고</td>
      <td>190</td>
      <td>100</td>
      <td>85</td>
      <td>90</td>
      <td>95</td>
      <td>95</td>
      <td>C#</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



Line Graph


```python
x = [-1,0,1]
y = [5,9,8]
plt.figure(figsize = (10,5), dpi = 100, facecolor = 'lightyellow') # dots per inch 해상도
plt.title('Line Graph')
plt.xlabel('X label', color = 'blue')
plt.ylabel('Y label', color = 'green')

plt.plot(x,y,label = 'Apple Received',
         marker = 'o',ms = 10,mec = 'black',mfc = 'white', # marker o,v,X / markersize, markeredgecolor, markerfacecolor
         linewidth = 3,linestyle = '--',color = 'pink', # linestyle :, -.
         alpha = 0.3) # 투명도
plt.xlim(-1.2,1.2) # x축 범위
plt.ylim(4.0,10.0) # y축 범위         
plt.legend(loc = (0.6,0.1)) # lower right
plt.grid(color = 'purple',ls = ':', alpha = 0.3, linewidth = 1.5)
plt.show()
# plt.savefig('graph_200.png',dpi = 200, facecolor = 'grey',edgecolor = 'black') # figure 저장하기
```



<center><img src ="https://user-images.githubusercontent.com/112631941/188785857-a7ab77c0-2524-449b-9db1-9f0d62e4dc6d.png"></center>



```python
apple = [3,5,4,9]
banana = [1,3,8,4]
cranberry = [10,6,7,9]
dragon_fruit = [1,4,3,2]
date = [1,2,3,4]

plt.plot(date,apple,'o-',label = 'apple')
plt.plot(date,banana,'v--',label = 'banana',)
plt.plot(date,cranberry,'x:',label = 'cranberry')
plt.plot(date,dragon_fruit,'.-.',label = 'dragonfruit')
plt.legend(loc = 'upper left',ncol = 2) # legend의 열
plt.show()
```


    
<center><img src = "https://user-images.githubusercontent.com/112631941/188785925-f4b26238-1195-4d2c-95a9-0298e4215a07.png"></center>
    


Bar Graph


```python
# Vertical Bar graph #

Player_name = ['Salah','Mane','Firmino','Jota']
Goals = [10,8,5,8]
colors = ['red','blue','green','black']
ticks = ['A','B','C','D']

plt.bar(Player_name, Goals, color = colors, width = 0.7)
plt.xticks(Player_name, ticks, rotation  = 90) # 항목
plt.ylim(3,12) # y축 범위
plt.show()
```


    
<center><img src = "https://user-images.githubusercontent.com/112631941/188785952-a8f8207a-838f-4d2e-8954-7d1cef020854.png"></center>
    



```python
# 누적 막대 그래프 #

plt.bar(student.index, student['국어'], label = 'Korean')
plt.bar(student.index, student['영어'], bottom = student['국어'], label = 'English')
plt.bar(student.index, student['수학'], bottom = student['국어'] + student['영어'], label = 'Math')
plt.legend()
plt.show()
```


    
<center><img src = "https://user-images.githubusercontent.com/112631941/188785975-1d6c69fd-b5de-4c1e-a39d-d672e68ddc57.png"></center>



```python
# 다중 막대 그래프 #

index = np.arange(student.shape[0])
w = 0.25
plt.figure(figsize = (10,5))
plt.title('Grade of students')
plt.bar(index-w, student['국어'],label = 'Korean', width = w)
plt.bar(index, student['영어'],label = 'English', width = w)
plt.bar(index+w, student['수학'],label = 'Math', width = w)
plt.xticks(index, student.index)
plt.legend()
plt.show()
```


    
<center><img src = "https://user-images.githubusercontent.com/112631941/188785992-2a5b3c33-5a12-447e-811d-edd3d136672b.png"></center>
    



```python
# Horizontal Bar graph #

plt.barh(Player_name, Goals, color = colors)
plt.gca().invert_yaxis() # y축 대칭
plt.show()
```


<center><img src = "https://user-images.githubusercontent.com/112631941/188786006-ec4aef32-cc3f-4c9a-9651-cf9cde0ae071.png"></center>
    


Histogram


```python
plt.hist(student['키'], bins = 7)
plt.title('Student Heights')
plt.xlabel('Heights')
plt.show()
```


<center><img src = "https://user-images.githubusercontent.com/112631941/188786028-4d13fa0a-9fca-47f3-bcdb-55c6e1168081.png"></center>
    


Pie Chart


```python
values = []
labels = []
for position in key['position'].unique():
  labels.append(position)
  values.append(len(key[key['position']==position]))
explode = [0.05] * len(key['position'].unique())

plt.title('Players in each Position')
plt.pie(values,labels = labels,explode = explode,autopct = '%.1f',startangle = 90,counterclock = False) # autopct : percentage
plt.legend(loc = (1.2,0.6),title = 'Positions')
plt.show()
```



<center><img src = "https://user-images.githubusercontent.com/112631941/188786041-42c57785-cef6-4457-bbf5-b5c62f9231f1.png"></center>
    



```python
def custom(pct):
  return "{:.0f}".format(pct) if pct >= 10 else '' # 정수형
  # return "{:.1f}".format(pct) if pct >= 10 else ''

wedgeprops = {'width' : 0.5, 'edgecolor' : 'w', 'linewidth' : 2}

plt.title('Players in each Position')
plt.pie(values,labels = labels,autopct = custom,startangle = 90,counterclock = False,wedgeprops = wedgeprops, pctdistance = 0.75)
plt.legend(loc = (1.2,0.6),title = 'Positions')
plt.show()
```


    
<center><img src = "https://user-images.githubusercontent.com/112631941/188786073-5700796b-0441-4dc4-a117-1a0d2dde643c.png"></center>



Scatterplot


```python
plt.figure(figsize = (6,7))
plt.scatter(student['영어'], student['수학'], s = student['학년'] * 300, c = student['학년'], cmap ='viridis')
plt.xlabel('English')
plt.ylabel('Math')
plt.colorbar(label = 'Year', ticks = [1,2,3],orientation = 'horizontal', shrink = 0.5)
plt.show()
```


    
<center><img src = "https://user-images.githubusercontent.com/112631941/188786101-f7d79e89-0fd0-41f1-969f-0b45e33eaf0c.png"></center>
    


Subplots


```python
# fig는 그래프 바깥 영역, axes는 그래프 안쪽 영역 #
fig, axs = plt.subplots(2,2, figsize = (15,10))
fig.suptitle('Student Scores')

axs[0,0].bar(student.index, student['국어'], label = 'Korean')
axs[0,0].set_title('Korean Scores')
axs[0,0].set(xlabel = 'Number', ylabel = 'Score')
axs[0,0].legend()
axs[0,0].grid(linestyle = '--', linewidth = 0.3)
axs[0,0].set_facecolor('lightyellow')

axs[0,1].plot(student.index, student['수학'], label = 'Math')
axs[0,1].plot(student.index, student['영어'], label = 'English')
axs[0,1].set_title('Math Scores')
axs[0,1].set(xlabel = 'Number', ylabel = 'Score')
axs[0,1].grid(linestyle = '--', linewidth = 0.3,color = 'lightyellow')
axs[0,1].set_facecolor('lightblue')
axs[0,1].legend()

axs[1,0].barh(student.index, student['키'], label = 'Height')
axs[1,0].set_title('Heights')
axs[1,0].set(xlabel = 'Height', ylabel = 'Number')
axs[1,0].legend()
axs[1,0].grid(linestyle = ':',linewidth = 0.3, color = 'grey')

axs[1,1].scatter(student['사회'],student['과학'],s = student['학년'] * 250,c = student['학년'], cmap = 'viridis')
axs[1,1].set_title('Correlation between Society and Science scores')
axs[1,1].grid(linestyle = '-.', linewidth = 0.3, color = 'lightgreen')

plt.show()
```


    
<center><img src = "https://user-images.githubusercontent.com/112631941/188786126-6b505e11-9f5c-4145-acaa-a349c7904856.png"></center>



    

