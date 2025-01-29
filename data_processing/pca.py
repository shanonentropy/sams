# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 19:33:20 2024

@author: zahmed
"""

fpath = 'C:\processed_data_sensor_2\data_matricies\svd_data_matrix\cycle1_week1\demeaned_data_matrix_first_cycle_week1'

df = pd.read_csv(fpath, sep=',', header=0)

X1, y1 = df.iloc[:, 1:], df.iloc[:, 0]

n_comps = 7

pcx = pc(n_components=n_comps)

pcx.fit_transform(X)

plt.plot(pcx.explained_variance_ratio_, 'x')

loadings = X1 @ pcx.components_.T

plt.plot(loadings[0])


lnr = LinearRegression()
x_t = loadings
lnr.fit(x_t, y1)
y_pd = lnr.predict(x_t)
tr = np.sqrt(mean_squared_error(y1, y_pd))
print(f'training error with 7 comps is {tr:3f}')

sns.regplot(x= y1, y = y_pd, order=1)

for i in range(n_comps):
    plt.plot(pcx.components_[i])
    plt.title(i)
    plt.show()
    
m0 = pcx.components_[0]
m1 = pcx.components_[1]
m2 = pcx.components_[2]
m3 = pcx.components_[3]    
m4 = pcx.components_[4]
m5 = pcx.components_[5]
m6 = pcx.components_[7]

    
plt.plot(m0); plt.plot(m1*0.2); #plt.plot(m2)
m1_2 = m0 - m1*0.35

plt.plot(m1_2)

for i in range(n_comps):
    sns.heatmap(pcx.components_[0].reshape(-1, 1) @ pcx.components_[i].reshape(1, -1))
    plt.title(i)
    plt.show()
