#-*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math


def make_condition(df, start, end, gap):
    # np.select를 사용하기위해 '값이 어떠한 조건일 때'라는 condition을 만들어줌
    condition = []
    for i in np.arange(start, end+gap, gap):
        condition.append((df >= i) & (df < i+gap))
    return condition

def make_choices(start, end, gap):
    # np.selcet를 사용하기 위해 값이 어떠한 조건일 때, '해당하는 값'을 만들어줌
    choices = []
    precision = check_precision(gap)
    for i in np.arange(start, end+gap, gap):
        if precision == 0:
            choices.append(int(i))
        else:
            choices.append(round(i, precision))
    return choices

def check_precision(value):
    # 소수점 아래 몇자리까지 있는지 확인
    try:
        precision = len(str(value).split('.')[1])
    except IndexError:
        precision = 0
    return precision

def get_result_df(df, s_output, e_output):
    # 분석된 데이터를 확인할 범위를 구하고 
    # 범위 내의 결과를 return
    gap_precision = check_precision(GAP)    
    start_precision = check_precision(s_output)
    end_precision = check_precision(e_output)
    
    if start_precision > gap_precision:
        # start_precision이 0인 경우는 이 조건문에 들어올 수 없음
        s_outout = s_output.split('.')[0] + '.' + s_output.split('.')[1][ : gap_precision + 1]
        s_o = round(float(s_output), gap_precision)
    else:
        s_o = round(float(s_output), gap_precision)

    if end_precision > gap_precision:
        # start_precision이 0인 경우는 이 조건문에 들어올 수 없음
        e_o = s_output.split('.')[0] + '.' + s_output.split('.')[1][ : gap_precision + 1]
        e_o = round(float(e_o), gap_precision) + GAP     
    else:
        e_o = round(float(e_output), gap_precision)

    result_df = df[ (df['output_by10kg'] >= s_o) & (df['output_by10kg'] < e_o) ]
    return result_df  

def drawboxplot_sep(sepfig_width, sepfig_height, df, input_vals):
    # boxplot을 그려줌
    #pad = 0.3
    columns = df.columns
    len_columns = len(columns)
    fig = plt.figure(figsize=(sepfig_width, sepfig_height*len_columns))
    for i, col in enumerate(columns):
        ax = fig.add_subplot(len_columns, 1, i+1)
        sns.boxplot(x=col, data=df, palette='Set3')
        ax.set_xlabel('')
        ax.set_ylabel('')
        plt.subplots_adjust(hspace = 1)
        plt.title('<%s>' % str(col), fontsize=15, y=1.01)    
	
        info = df[col].describe()
        quartile_1 = round(info['25%'], 1)
        quartile_3 = round(info['75%'], 1)
        median = round(info['50%'], 1)

        string = str("Recommendation(25%%): %s\nRecommendation(75%%): %s" %(quartile_1, quartile_3)) #.decode('utf-8')

        if (i < 3) and (input_vals[i] != None) :
            input_v = float(input_vals[i])
            ax.vlines(input_v, -0.47, 0.47, colors='r', linestyles='solid', linewidth=7)
            
            present = input_v
            efficiency = round((median - input_v) / median * 100, 1)
            if efficiency >= 0:
                efficiency = 'Save(+) %s%%' % (efficiency)
            else:
                efficiency = 'Waste(-) %s%%' % (abs(efficiency))
              
            string += str('\n\nRecommendation(median): %s\nPresent: %s\n\nEfficiency -> %s' % (median, present, efficiency)) #.decode('utf-8')
        else:
            string += str('\n\nRecommendation(median): %s' % (median)).decode('utf-8')

        ax.text(1.05, 0.5, string, ha='left', va='center',transform=ax.transAxes, fontsize=10)            
        pad = 0.5
        fig.subplots_adjust(right=0.61)
    plt.show()


# 넥스지 CSV 데이터를 불러오고, 필요없는 컬럼을 삭제함
data = pd.read_csv('../../Raw_data/prod_out_4_180923-181005_hschoi.csv', encoding='utf-8')
del data['Unnamed: 25']
del data['epoch const']


# 단위시간당 생산량 = 전력(kw) / 톤당 전력(kwh/ton)
data['OUTPUT_FROM_PWR'] = data['DISP_PWR_M'] * 1.0 / data['DISP_ENG']


# 단위시간당 생산량을 계산할 때, 톤당 전력 값이 0인 경우
# 계산값이 NaN이나 inf 값이 나온 경우를 걸러주기 위해 처리해줌
row_index = data['DISP_ENG'] == 0
data.loc[row_index, 'OUTPUT_FROM_PWR'] = 0


# 단위시간당 생산량을 계산할 때, 톤당 전력 또는 전력 값이 NaN일 경우
# 계산값이 NaN이 나온 부분을 걸러주기 위해 처리해줌
data = data[data['OUTPUT_FROM_PWR'].notnull()]


max_output = data['OUTPUT_FROM_PWR'].max()
# GAP을 10kg으로 설정함
GAP = 0.01
condition = make_condition(data['OUTPUT_FROM_PWR'], 0, max_output, GAP)
choices = make_choices(0, max_output, GAP)


# condition(조건)과 조건에 해당하는 choice를 이용하여 해당하는 구간에 대한 Tag를 붙임
data['output_by10kg'] = np.select(condition, choices)


# 생산량 범위(s_output ~ e_output)에 해당하고
# 확인을 원하는 tags(checked_tag)의 데이터프레임만 뽑아낸다.
s_output = "3.21"
e_output = "3.22"
checked_tags = ["HS_STM", "DISP_PWR_M", "DISP_ENG"]
input_vals = ["1.1", "1.1", "1.1"]

result = get_result_df(data, s_output, e_output)
result = result[checked_tags]
result = result[result.notnull()]


# 그래프를 그리기
drawboxplot_sep(9, 2.5, result, input_vals)