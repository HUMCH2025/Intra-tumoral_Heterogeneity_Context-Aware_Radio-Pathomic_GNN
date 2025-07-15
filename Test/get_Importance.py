import pandas as pd
import numpy as np
from scipy.stats import ttest_ind,norm,f,mannwhitneyu,shapiro
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


if __name__ == '__main__':
    df = pd.read_excel('E:\\wsi\\test\\feature_IG.xlsx')
    df_low = df[df['IG_Value'] <= df['IG_Value'].quantile(0.1)]
    value = len(df_low) * [0]
    df_low.insert(loc=len(df_low.columns), columns='IG', value=value)
    df_low.drop('IG_Value',axis=1,inplace=True)
    df_high = df[df['IG_Value'] >= df['IG_Value'].quantile(0.9)]
    value = len(df_low) * [0]
    df_high.insert(loc=len(df_high.columns), columns='IG', value=value)
    df_high.drop('IG_Value', axis=1, inplace=True)
    df = pd.concat([df_low,df_high],axis=0)
    threshold = 0.75
    correlations = df.corr()['IG'].abs()
    df = df[correlations[(correlations <= threshold) & (correlations.index != 'IG')].index.tolist()]
    x,y = df.iloc[:,0:-1],df.loc[:,-1]
    data_0 = df.groupby(df.IG == 0).get_group(True)
    data_1 = df.groupby(df.IG == 0).get_group(False)
    list_t = []
    list_u = []
    df = df.drop('IG',axis=1)
    for i in range(len(df.columns)):
        shapiro_test = shapiro(df[df.columns[i]])
        if shapiro_test.pvalue >= 0.05:
            list_t.append(df.columns[i])
        else:
            list_u.append(df.columns[i])
    list_select = []
    for i in list_t:
        F = np.var(data_0[i].tolist()) / np.var(data_1[i].tolist())
        v1 = len(data_0[i].tolist()) - 1
        v2 = len(data_1[i].tolist()) - 1
        p_val = 1 - 2 * abs(0.5 - f.cdf(F, v1, v2))
        if p_val < 0.05:
            ttest, pval = ttest_ind(data_0[i].tolist(), data_1[i].tolist(), equal_var=False, alternative='two-sided')
        else:
            ttest, pval = ttest_ind(data_0[i].tolist(), data_1[i].tolist(), equal_var=True)
        if pval < 0.05:
            list_select.append(i)
    for i in list_u:
        s, pval = mannwhitneyu(data_0[i].tolist(), data_1[i].tolist(), alternative='two-sided')
        if pval < 0.05:
            list_select.append(i)
    data_x = x[list_select]
    scaler = StandardScaler()
    x = scaler.fit_transform(data_x)
    x = pd.DataFrame(x, columns=data_x.columns)
    model = LogisticRegression(C=0.001,max_iter=100,penalty='l2',solver='liblinear',random_state=42)
    model.fit(x,y)
    coef = model.coef_[0]
    features = x.columns
    coef_df = pd.DataFrame({'Feature':features,'Coefficient':coef})
    print(coef_df)