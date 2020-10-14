import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df=pd.read_csv('pima_indians_diabetes_miss.csv') 
df_org=pd.read_csv('pima_indians_diabetes_original.csv') 

#Question 1
print('Question 1:')
count_na=len(df)-df.describe().loc['count']
plt.bar(df.columns,count_na)
plt.xlabel('Attributes');plt.ylabel('Count')
plt.title('No. of missing values');plt.show()

#Question 2
print('Question 2:')
#a
print('a)\n')
del_tuple=[]
null_val=df.isna()
for i in range(len(df)):
    if sum(null_val.iloc[i])>=(1/3*len(df.columns)):
        del_tuple.append(i)
df.drop(del_tuple,inplace=True)
print('Total no. of tuples deleted:',len(del_tuple))
print('\nRow no. of deleted tuples:\n',*del_tuple)
#b
print('\nb)')
del_tuple=df[df['class'].isna()]
df.drop(del_tuple.index,inplace=True)
print('\nTotal no. of tuples deleted:',len(del_tuple.index))
print('\nRow no. of deleted tuples:\n',*del_tuple.index)

#Question3
print('\nQuestion 3:\n')
count_na=len(df)-df.describe().loc['count']
for i in range(len(df.columns)):
    print(df.columns[i],':',int(count_na[i]))
print('\nTotal no. of missing values:',int(sum(count_na)))

#Question 4
print('\nQuestion 4:')
def filling(df):
    data=pd.concat((df.mean(),df.median(),df.mode().loc[0],
            df.mode().loc[1],df.std()),axis=1,)
    index=[['Mean','Median','Mode1','Mode2','Statndard Dev']]
    data=data.T
    data.set_index(index,inplace=True)
    data=data.T
    print('New Data:\n')
    print(data)
    
    data_org=pd.concat((df_org.mean(),df_org.median(),df_org.mode().loc[0],
            df_org.mode().loc[1],df_org.std()),axis=1,)
    data_org=data_org.T
    data_org.set_index(index,inplace=True)
    data_org=data_org.T
    print('\nOriginal Data:\n')
    print(data_org)
def RMSE(df_):
    print('RMSE:\n')
    RMSE=[0]*len(df.columns)
    n=0
    for i in df.columns:
        null_index=df[i][df[i].isna()].index
        if len(null_index)==0:
            RMSE[n]=0
        else:
            for j in null_index:
                RMSE[n]+=(df_[i][j]-df_org[i][j])**2
                
            RMSE[n]/=len(null_index)
            RMSE[n]**=0.5
        print(i,':',RMSE[n])
        n+=1
    plt.bar(df.columns,RMSE)
    plt.xlabel('Attributes')
    plt.ylabel('RMSE')
    plt.show()

#a
print('a)')
df_a=df.fillna(df.mean())
#i
print('i)\n')
filling(df_a)
#ii
print('\nii)\n')
RMSE(df_a)

#b
print('\nb)')
df_b=df.interpolate()
#i
print('i)\n')
filling(df_b)
#ii
print('\nii)\n')
RMSE(df_b)

#Question 5
print('\nQuestion 5:')
df=df_b

def outliers(x):
    minimum=2.5*np.percentile(df[x],25)-1.5*np.percentile(df[x],75)
    maximum=2.5*np.percentile(df[x],75)-1.5*np.percentile(df[x],25)
    outliers_=pd.concat((df[x][df[x]< minimum],df[x][df[x]> maximum]))
    return outliers_

def boxplot():
    fig,axs=plt.subplots(1,2,figsize=(12,8))
    axs[0].boxplot(df['Age'],vert=True,patch_artist=True)
    axs[0].set_title('Age')
    axs[0].set_ylabel('Age')
    axs[1].boxplot(df['BMI'],vert=True,patch_artist=True)
    axs[1].set_title('BMI')
    axs[1].set_ylabel('BMI')
    plt.show()

#i
print('i)\n')
print('Outliers in Age:')
print(*outliers('Age').values)
print('\nOutliers in BMI:')
print(*outliers('BMI').values)
boxplot()
#ii
print('ii)')
outliers_age=outliers('Age')
df['Age'][outliers_age.index]=df['Age'].median()
outliers_bmi=outliers('BMI')
df['BMI'][outliers_bmi.index]=df['BMI'].median()
boxplot()
