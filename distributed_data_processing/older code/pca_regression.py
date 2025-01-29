# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 19:54:16 2024

@author: zahmed

write print i.e. errors statements to a file

"""

#import modules
import dask.array as da
from dask_ml.model_selection import train_test_split
import dask.dataframe as dd
from dask_ml.decomposition import PCA   
import matplotlib.pyplot as plt
from dask.distributed import Client
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
 
#import data matrix
fpath = 'C:\processed_data_sensor_2\data_matricies\svd_data_matrix\week3_third_cycle\demeaned_data_matrix_third_cycle_week_3'
ddf = dd.read_csv(fpath, sep=',', header=0)

ddf = ddf.rename(columns={'Unnamed: 0': 'temperature'});
#ddf.head(2)

client = Client(n_workers=2, threads_per_worker=1, memory_limit='20GB')
col_list = ddf.columns
col_list[0]

# slice the ddf into X and y
X = ddf[col_list[1:]]
y = ddf[col_list[0]]; 

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=True, random_state=42)

X_train_array = X_train.to_dask_array(lengths=True)
X_test_array = X_test.to_dask_array(lengths=True)

# Create a Dask PCA object
n_comps = 20
pca = PCA(n_components=n_comps, svd_solver='randomized')

# Fit the PCA model
pca.fit_transform(X_train_array) 
pca.transform(X_test_array)
# Get explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# filename and path for figure
export_name = 'c:/sams//saved_data/sensor_2_week_3_third_cycle_screeplot.png'
# Create scree plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.xticks(range(1, len(explained_variance_ratio) + 1))
plt.grid()
plt.savefig(export_name, dpi=700)
plt.show()



print(pca.singular_values_)

# =============================================================================
# save pca modes
# =============================================================================

pca_export = 'c:/sams//saved_data/sensor_2_week_3_third_cycle_pca_modes.csv'

df_pca = pd.DataFrame(pca.components_)

df_pca.to_csv(pca_export)
# =============================================================================
# compute loadings
# =============================================================================

loadings_train = (X_train_array@pca.components_.T).compute()
loadings_test = (X_test_array@pca.components_.T).compute()
y_train_comp = y_train.compute()

# wavelength labels for x axis
x_label =ddf.columns[1:].astype('float')

#save figs for each mode
export_name = 'c:/sams//saved_data/sensor_2_week_3_third_cycle_eigenmode_{}.png'


# plot each of the components
for i in range(n_comps):
    plt.plot(x_label, pca.components_.T[:,i]); 
    plt.title('eigenmode {}'.format(i))
    plt.xlabel('Wavelength (nm)')
    plt.savefig(export_name.format(i), dpi=700)
    plt.show()


#save figs for each mode's laoding 
export_name = 'c:/sams/saved_data/sensor_2_week_3_third_cycle_eigenmode_{}_loading.png'

# make loading plots for each mode
for i in range(n_comps):
    plt.plot(loadings_train[:, i])
    plt.title('loading for {}-th eigemode'.format(i))
    plt.savefig(export_name.format(i), dpi=700)
    plt.show()



# plot components against first mode (zeroth mode)

for i in range(n_comps):
    if i ==0:
        pass
    else:
        plt.plot(loadings_train[:,0],loadings_train[:,i], '*' )
        plt.xlabel('first mode')
        plt.ylabel('mode {}'.format(i))
        plt.savefig('c:/sams/saved_data/sensor_2_week_3_eigenmode_{}_against_first_mode.png'.format(i))
        plt.show()







y_pred_test_comp = y_test.compute()

# =============================================================================
# test linear regression without reseting y index
# =============================================================================
lnr = LinearRegression()
x_t = loadings_train[:, 0:7]
lnr.fit(x_t, y_train)
y_pd = lnr.predict(x_t)
tr = np.sqrt(mean_squared_error(y_train, y_pd))
print(f'training error with 7 comps is {tr:3f}')
y_test_pd = lnr.predict(loadings_test[:, 0:7])
tr_t = np.sqrt(mean_squared_error(y_test, y_test_pd))
print(f'testing error with 7 comps is {tr_t:3f}')

#y_pred_test_comp = y_test_pd.compute()


plt.plot(y_pred_test_comp, y_test_pd, 'x')
plt.title('Linear regression')
plt.xlabel('Measured Temperature (deg C)')
plt.ylabel('Predicted Temperature (deg C)')
plt.savefig('c:/sams/saved_data/sensor_2_week_3_third_cycle_LINEAR_testing.png', dpi = 700)
plt.show()

plt.plot(y_pred_test_comp, y_pred_test_comp-y_test_pd, 'x')
plt.title('LINEAR regression')
plt.xlabel('Measured Temperature (deg C)')
plt.ylabel('Predicted Temperature (deg C)')
plt.savefig('c:/sams/saved_data/sensor_2_week_3_third_cycle_LINEAR_testing_residual.png', dpi = 700)
plt.show()


# =============================================================================
# linear regression search for optimal n_comps
# =============================================================================

ncomps =[]
coefs = []
training_error= []
testing_error =[]

#y_ = y_train_comp.reset_index()['temperature']
#y__ = y_test.compute().reset_index()['temperature']
ln = LinearRegression()

for i in range(n_comps):
    if i ==0:
        pass
    else:
        ncomps.append(i)
        x_ = (loadings_train[:, 0:i])
        ln.fit(x_, y_train)
        coefs.append(ln.coef_)
        y_pred_training = ln.predict(x_)
        tr_error = np.sqrt(mean_squared_error(y_train, y_pred_training))
        training_error.append(tr_error)
        print(f'training error with {i} comps is {tr_error:3f}')
        y_pred_testing =  ln.predict(loadings_test[:, 0:i])
        tt_error = np.sqrt(mean_squared_error(y_test, y_pred_testing))
        testing_error.append(tt_error)
        print(f'testing error with {i} comps is {tt_error:3f}')

df_ = pd.DataFrame(list(zip(ncomps,training_error, testing_error, coefs)))
df_.columns = ['components', 'training_error', 'testing_error', 'coefs']

plt.plot(df_.components, df_.training_error)
plt.plot(df_.components, df_.testing_error)
plt.show()

df_.to_csv('n_comp_evaluation_third_cycle.csv')


# =============================================================================
# Lasso regression
# =============================================================================

x_ = (loadings_train[:, 0:7])
y_ = y_train#_comp.reset_index()['temperature']
y__ = y_test#.compute().reset_index()['temperature']
ls = Lasso(alpha = 0.1)
ls.fit(x_, y_)
y_pred_training = ls.predict(x_)
print(f'training error is {mean_squared_error(y_train, y_pred_training):3f}')

y_pred_testing =  ls.predict(loadings_test[:, 0:7])
print(f'testing error is {mean_squared_error(y__, y_pred_testing):3f}')

plt.plot(y_pred_test_comp, y_pred_testing, 'x')
plt.title('LASSO regression')
plt.xlabel('Measured Temperature (deg C)')
plt.ylabel('Predicted Temperature (deg C)')
plt.savefig('c:/sams/saved_data/sensor_2_week_3_third_cycle_LASSO_testing.png', dpi = 700)
plt.show()

plt.plot(y_pred_test_comp, y_pred_test_comp-y_pred_testing, 'x')
plt.title('LASSO regression')
plt.xlabel('Measured Temperature (deg C)')
plt.ylabel('Predicted Temperature (deg C)')
plt.savefig('c:/sams/saved_data/sensor_2_week_3_third_cycle_LASSO_testing_residual.png', dpi = 700)
plt.show()




# set hyperparameter search for alpha

alphas = np.array([1e-6, 1e-5,1e-4, 1e-3, 1e-2, 0.005, 0.1, 0.5, 1])

alphas_l =[]
training_error_l= []
testing_error_l = []
coef_l = []

for a in alphas:
    alphas_l.append(a)
    ls_ = Lasso(alpha = a)
    ls_.fit(x_, y_)
    y_pred_training = ls_.predict(x_)
    training_error_l.append(mean_squared_error(y_train, y_pred_training))
    y_pred_testing =  ls_.predict(loadings_test[:, 0:7])
    testing_error_l.append(mean_squared_error(y__, y_pred_testing))
    coef_l.append(ls_.coef_)

plt.plot(alphas_l, training_error_l[:], 'x')
plt.title('LASSO regression optimization')
plt.xlabel('alphas')
plt.ylabel('training_error')
plt.savefig('c:/sams/saved_data/sensor_2_week_3_third_cycle_LASSO_alphas_training_error.png', dpi = 700)
plt.show()

plt.plot(alphas_l, testing_error_l, 'x')
plt.title('LASSO regression optimization testing error')
plt.xlabel('alphas')
plt.ylabel('testing_error')
plt.savefig('c:/sams/saved_data/sensor_2_week_3_third_cycle_LASSO_alphas_testing_error.png', dpi = 700)
plt.show()


# =============================================================================
# Ridge regression
# =============================================================================

x_ = (loadings_train[:, 0:7])
y_ = y_train#_comp.reset_index()['temperature']
y__ = y_test#.compute().reset_index()['temperature']
lrd = Ridge(alpha = 0.1)
lrd.fit(x_, y_)
y_pred_training = lrd.predict(x_)
print(f'training error is {mean_squared_error(y_train, y_pred_training):3f}')

y_pred_testing =  lrd.predict(loadings_test[:, 0:7])
print(f'testing error is {mean_squared_error(y__, y_pred_testing):3f}')



plt.plot(y_pred_test_comp, y_pred_testing, 'x')
plt.title('RIDGE regression')
plt.xlabel('Measured Temperature (deg C)')
plt.ylabel('Predicted Temperature (deg C)')
plt.savefig('c:/sams/saved_data/sensor_2_week_3_third_cycle_RIDGE_testing.png', dpi = 700)
plt.show()

plt.plot(y_pred_test_comp, y_pred_test_comp-y_pred_testing, 'x')
plt.title('RIDGE regression')
plt.xlabel('Measured Temperature (deg C)')
plt.ylabel('Predicted Temperature (deg C)')
plt.savefig('c:/sams/saved_data/sensor_2_week_3_third_cycle_RIDGE_testing_residual.png', dpi = 700)
plt.show()





# set hyperparameter search for alpha


alphas_ridge =[]
training_error_ridge= []
testing_error_ridge = []

for a in alphas:
    alphas_ridge.append(a)
    ridge_reg = Ridge(alpha = a)
    ridge_reg.fit(x_, y_)
    y_pred_training = ridge_reg.predict(x_)
    training_error_ridge.append(mean_squared_error(y_train, y_pred_training))
    y_pred_testing =  ridge_reg.predict(loadings_test[:, 0:7])
    testing_error_ridge.append(mean_squared_error(y__, y_pred_testing))

plt.plot(alphas_ridge, training_error_ridge, 'x')
plt.title('RIDGE regression optimization')
plt.xlabel('alphas')
plt.ylabel('training_error')
plt.savefig('c:/sams/saved_data/sensor_2_week_3_third_cycle_RIDGE_alphas_training_error.png', dpi = 700)
plt.show()

plt.plot(alphas_ridge, testing_error_ridge, 'x')
plt.title('RIDGE regression optimization testing error')
plt.xlabel('alphas')
plt.ylabel('testing_error')
plt.savefig('c:/sams/saved_data/sensor_2_week_3_third_cycle_RIDGE_alphas_testing_error.png', dpi = 700)
plt.show()





client.close()


# =============================================================================
# compare pca mode 0 against 1
# =============================================================================

mode_0 = pca.components_.T[:, 0]
mode_1 = pca.components_.T[:, 1]

a0 = -1
sub_01 = mode_0 - (a0 * mode_1)

plt.plot(sub_01)
plt.title('comp 0-1')
plt.xlabel('pixel number')
plt.ylabel('Intensity (AU)')
plt.savefig('c:/sams/saved_data/sensor_2_week_3_third_cycle_mode_zero_and_one.png', dpi = 700)
plt.show()

plt.plot(mode_0)
plt.plot(((mode_1*-1)+0.01 )*.62 )
plt.title('comp 0 and 1- w/1 scaled on zpl')
plt.xlabel('pixel number')
plt.ylabel('Intensity (AU)')
plt.savefig('c:/sams/saved_data/sensor_2_week_3_third_cycle_mode_zero_overlaid_scaled_one.png', dpi = 700)
plt.show()


plt.plot(mode_0 - ((mode_1*-1)+0.01 )*.62 )
plt.title('comp 0 sub 1- w/1 scaled on zpl')
plt.xlabel('pixel number')
plt.ylabel('Intensity (AU)')
plt.savefig('c:/sams/saved_data/sensor_2_week_3_third_cycle_mode_zero_sub_scaled_one.png', dpi = 700)
plt.show()






# =============================================================================
# batch processing of linear regression using recursive alogrithm
# =============================================================================
'''

ln_batch =LinearRegression()

batch_size = 21121

for i in range(0, loadings_train.shape[0], batch_size):
    start = i
    end = min(i + batch_size, loadings_train.shape[0])  # Ensure end doesn't exceed dataset length
    #plt.plot(y_train.compute()[start:end])
    print(i)
    ln_batch.fit(x_[start:end,:], y_[start:end])  # Fit model on current batch



y_pred_training = ln_batch.predict(x_)
print(f'training error in batch fitting is {mean_squared_error(y_train, y_pred_training):3f}')

y_pred_testing =  ln_batch.predict(loadings_test[:, 0:9])
print(f'testing error in batch fitting is {mean_squared_error(y__, y_pred_testing):3f}')




'''




# =============================================================================
# GP w/.o scaling using recursive alogrithm
# =============================================================================
'''
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel, DotProduct

kernel_ = (ConstantKernel()+RBF(length_scale = 1, length_scale_bounds = (1e-6, 1000)) ) + WhiteKernel()
gp_model = GaussianProcessRegressor(kernel=kernel_)

#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(loadings_train[:, 0:1])
#y_train_comp = y_train.compute()
batch_size = 21121

for i in range(0, loadings_train.shape[0], batch_size):
    start = i
    end = min(i + batch_size, loadings_train.shape[0])  # Ensure end doesn't exceed dataset length
    #plt.plot(y_train.compute()[start:end])
    print(i)
    gp_model.fit(x_[start:end,:], y_[start:end])  # Fit model on current batch


y_pred_train_, std_pred_train = gp_model.predict(x_, return_std=True)


y_pred_train = y_pred_train_.reset_index()['temperature']
plt.plot(y_, y_pred_train, 'x')





# fit model
gp_model.fit(X_scaled, y_train)

# predict
y_pred, std_pred = gp_model.predict(X_scaled, return_std=True)

plt.plot(X_scaled, y_pred, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
plt.scatter(X_scaled, y_train, label="Observations")
plt.plot(X_scaled, mean_prediction, label="Mean prediction")
plt.fill_between(
    X_scaled.ravel(),
    mean_prediction - 1.96 * std_prediction,
    mean_prediction + 1.96 * std_prediction,
    alpha=0.5,
    label=r"95% confidence interval",
)
plt.legend()
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
_ = plt.title("Gaussian process regression on noise-free dataset")
'''
