import torch
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import Lasso,Ridge,ElasticNet
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.interpolate import CubicSpline, UnivariateSpline

torch.manual_seed(0)
np.random.seed(0)

def compute_time_derivative(u, dt):
    if len(u.shape)==2:
        # Initialize derivative array
        dudt = torch.zeros_like(u)
        # Central differences for interior points
        dudt[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dt)
        # Forward difference for the first point
        dudt[:, 0] = (u[:, 1] - u[:, 0]) / dt
        # Backward difference for the last point
        dudt[:, -1] = (u[:, -1] - u[:, -2]) / dt
        return dudt
    
    if len(u.shape)==3:
        # Initialize derivative array
        dudt = torch.zeros_like(u)
        # Central differences for interior points
        dudt[:, :, 1:-1] = (u[:, :, 2:] - u[:, :, :-2]) / (2 * dt)
        # Forward difference for the first point
        dudt[:, :, 0] = (u[:, :, 1] - u[:, :, 0]) / dt
        # Backward difference for the last point
        dudt[:, :, -1] = (u[:, :, -1] - u[:, :, -2]) / dt
        return dudt
    
def compute_space_derivative(u, dx, order):

    dudx = torch.zeros_like(u)

    if order == 1:
        # First derivative
        dudx[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dx)  # Central difference
        dudx[0, :] = (u[1, :] - u[0, :]) / dx              # Forward difference
        dudx[-1, :] = (u[-1, :] - u[-2, :]) / dx           # Backward difference

    elif order == 2:
        # Second derivative
        dudx[1:-1, :] = (u[2:, :] - 2 * u[1:-1, :] + u[:-2, :]) / (dx ** 2)  # Central difference
        dudx[0, :] = (u[2, :] - 2 * u[1, :] + u[0, :]) / (dx ** 2)           # Forward difference
        dudx[-1, :] = (u[-1, :] - 2 * u[-2, :] + u[-3, :]) / (dx ** 2)       # Backward difference

    elif order == 3:
        dudx[2:-2, :] = (-u[0:-4, :] + 2 * u[1:-3, :] - 2 * u[3:-1, :] + u[4:, :]) / (2 * dx**3) # Central difference
        dudx[0, :] = (-3*u[0, :] + 3*u[1, :] - u[2, :]) / dx**3 # Forward difference 
        dudx[1, :] = (-3*u[1, :] + 3*u[2, :] - u[3, :]) / dx**3 # Forward difference 
        dudx[-1, :] = (-u[-4, :] + 3*u[-3, :] - 3*u[-2, :]) / dx**3  # Backward difference 
        dudx[-2, :] = (-u[-5, :] + 3*u[-4, :] - 3*u[-3, :]) / dx**3  # Backward difference 
    return dudx

def compute_time_derivative_spline(u, t, s):
    dudt = np.zeros_like(u)

    if len(u.shape)==2:
        for row in range(u.shape[0]):  
            u_np =  u[row,:].numpy()
            t_curr = t[row,:].numpy()
            spline = UnivariateSpline(t_curr, u_np, k=5, s=s)  
            
            dudt[row, :] = spline.derivative(1)(t_curr)  
        
        return torch.tensor(dudt, dtype=u.dtype)
    
    if len(u.shape)==3:
        dudt = np.zeros_like(u)
        # Iterate over all (x, y) slices
        for i in range(u.shape[0]):  
            for j in range(u.shape[1]):  
                u_slice = u[i, j, :]
                t_slice = t[i, j, :]

                spline = UnivariateSpline(t_slice, u_slice, k=5, s=0.1)

                # Compute the derivative 
                dudt[i, j, :] = spline.derivative(1)(t_slice)

        return torch.tensor(dudt, dtype=u.dtype, device=u.device)


def compute_space_derivative_spline(u, x, order, s):
    dudx = np.zeros_like(u)
    for col in range(u.shape[1]):  
        u_np =  u[:,col].numpy()
        x_curr = x[:,col]
        spline = UnivariateSpline(x_curr, u_np, k=5, s=s)  
        
     
    dudx[:, col] = spline.derivative(order)(x_curr) 

    return torch.tensor(dudx, dtype=u.dtype)


def compute_space_derivative_x(u, dx, order):
    dudx = torch.zeros_like(u)

    if order == 1:
        # First derivative
        dudx[1:-1, :, :] = (u[2:, : :] - u[:-2, :, :]) / (2 * dx)  # Central difference
        dudx[0, :, :] = (u[1, :, :] - u[0, :, :]) / dx              # Forward difference
        dudx[-1, :, :] = (u[-1, :, :] - u[-2, :, :]) / dx           # Backward difference

    elif order == 2:
        # Second derivative
        dudx[1:-1, :, :] = (u[2:, :, :] - 2 * u[1:-1, :, :] + u[:-2, :, :]) / (dx ** 2)  # Central difference
        dudx[0, :, :] = (u[2, :, :] - 2 * u[1, :, :] + u[0, :, :]) / (dx ** 2)           # Forward difference
        dudx[-1, :, :] = (u[-1, :, :] - 2 * u[-2, :, :] + u[-3, :, :]) / (dx ** 2)       # Backward difference

    elif order == 3:
        dudx[2:-2, :, :] = (-u[0:-4, :, :] + 2 * u[1:-3, :, :] - 2 * u[3:-1, :, :] + u[4:, :, :]) / (2 * dx**3) # Central difference
        dudx[0, :, :] = (-3*u[0, :, :] + 3*u[1, :, :] - u[2, :, :]) / dx**3 # Forward difference 
        dudx[1, :, :] = (-3*u[1, :, :] + 3*u[2, :, :] - u[3, :, :]) / dx**3 # Forward difference 
        dudx[-1, :, :] = (-u[-4, :, :] + 3*u[-3, :, :] - 3*u[-2, :, :]) / dx**3  # Backward difference 
        dudx[-2, :, :] = (-u[-5, :, :] + 3*u[-4, :, :] - 3*u[-3, :, :]) / dx**3  # Backward difference 
    return dudx


def compute_space_derivative_x_spline(u, x, order, s):
    dudx = np.zeros_like(u)
    for j in range(u.shape[1]):  
        for k in range(u.shape[2]):  
            u_slice = u[:, j, k]
            x_np = x[:,j,k]

            spline = UnivariateSpline(x_np, u_slice, k=5, s=s)

            # Compute the derivative
            dudx[:, j, k] = spline.derivative(order)(x_np)

    # Convert the result back to a PyTorch tensor
    return torch.tensor(dudx, dtype=u.dtype, device=u.device)


def compute_space_derivative_y(u, dy, order):

    dudx = torch.zeros_like(u)

    if order == 1:
        # First derivative
        dudx[:, 1:-1, :] = (u[:, 2:, :] - u[:, :-2, :]) / (2 * dy)  # Central difference
        dudx[:, 0, :] = (u[:, 1, :] - u[:, 0, :]) / dy             # Forward difference
        dudx[:, -1, :] = (u[:, -1, :] - u[:, -2, :]) / dy          # Backward difference

    elif order == 2:
        # Second derivative
        dudx[:, 1:-1, :] = (u[:, 2:, :] - 2 * u[:, 1:-1, :] + u[:, :-2, :]) / (dy ** 2)  # Central difference
        dudx[:, 0, :] = (u[:, 2, :] - 2 * u[:, 1, :] + u[:, 0, :]) / (dy ** 2)           # Forward difference
        dudx[:, -1, :] = (u[:, -1, :] - 2 * u[:, -2, :] + u[:, -3, :]) / (dy ** 2)       # Backward difference

    elif order == 3:
        dudx[:, 2:-2, :] = (-u[:, 0:-4, :] + 2 * u[:, 1:-3, :] - 2 * u[:, 3:-1, :] + u[:, 4:, :]) / (2 * dy**3) # Central difference
        dudx[:, 0, :] = (-3*u[:, 0, :] + 3*u[:, 1, :] - u[:, 2, :]) / dy**3 # Forward difference 
        dudx[:, 1, :] = (-3*u[:, 1, :] + 3*u[:, 2, :] - u[:, 3, :]) / dy**3 # Forward difference 
        dudx[:, -1, :] = (-u[:, -4, :] + 3*u[:, -3, :] - 3*u[:, -2, :]) / dy**3  # Backward difference 
        dudx[:, -2, :] = (-u[:, -5, :] + 3*u[:, -4, :] - 3*u[:, -3, :]) / dy**3  # Backward difference 
    return dudx


def compute_space_derivative_y_spline(u, y, order, s):
    dudy = np.zeros_like(u)
    for i in range(u.shape[0]):  
        for k in range(u.shape[2]):  
            u_slice = u[i, :, k]
            y_slice = y[i, :, k]

            spline = UnivariateSpline(y_slice, u_slice, k=5, s=s)

            # Compute the derivative 
            dudy[i, :, k] = spline.derivative(order)(y_slice)

    return torch.tensor(dudy, dtype=u.dtype, device=u.device)

def myregression(theta,target,names,threshold,alpha,test_size=0.2):
    n_iter = len(alpha)
    selected_names = names
    selected_coeff = [t for t in range(theta.shape[1])]
    for it in range(n_iter):
        print("Regression #",it+1)
        X_train, X_test, y_train, y_test = train_test_split(theta[:, selected_coeff], target, test_size=test_size, random_state=0)
        # Fit regression
        regression = Ridge(alpha=alpha[it])
        regression.fit(X_train, y_train)

        # Apply hard thresholding
        regression.coef_ = np.where(np.abs(regression.coef_) >= threshold, regression.coef_, 0).flatten()
        #print(regression.coef_)

        # Print coefficients
        for i in range(len(selected_coeff)):
            if regression.coef_[i] != 0.0:
                print(selected_names[i], " c= ", regression.coef_[i])

        # Predict and evaluate
        y_pred = regression.predict(X_test)
        print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
        print("###################### end of regression",it +1,"##############################")
        print()

        selected_names_new = []
        selected_coeff_new = []
        for j in range(len(regression.coef_)):
            if regression.coef_[j] != 0.0:
                selected_names_new.append(selected_names[j])
                selected_coeff_new.append(selected_coeff[j])
        selected_names = selected_names_new
        selected_coeff = selected_coeff_new
    
    return selected_names, regression.coef_

def myregression2(theta,target,names,threshold,alpha,test_size=0.2):
    threshold2 = 0.01
    n_iter = len(alpha)
    selected_names = names
    selected_coeff = [t for t in range(theta.shape[1])]
    for it in range(n_iter):
        print("Regression #",it+1)
        X_train, X_test, y_train, y_test = train_test_split(theta[:, selected_coeff], target, test_size=test_size, random_state=0)
        # Fit regression
        regression = Ridge(alpha=alpha[it])
        regression.fit(X_train, y_train)

        # Apply hard thresholding
        # Apply weighted thresholding based on norm of coefficients and term weights
        for i in range(len(selected_coeff)):
            weighted_norm = torch.norm(torch.tensor(regression.coef_[i]) * theta[:,selected_coeff[i]],p=2)
            if weighted_norm<=threshold:
                regression.coef_[i] = 0
        regression.coef_ = np.where(np.abs(regression.coef_) >= threshold2, regression.coef_, 0).flatten()

        # Print coefficients
        for i in range(len(selected_coeff)):
            if regression.coef_[i] != 0.0:
                print(selected_names[i], " c= ", regression.coef_[i])

        # Predict and evaluate
        y_pred = regression.predict(X_test)
        print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
        print("###################### end of regression",it +1,"##############################")
        print()

        selected_names_new = []
        selected_coeff_new = []
        for j in range(len(regression.coef_)):
            if regression.coef_[j] != 0.0:
                selected_names_new.append(selected_names[j])
                selected_coeff_new.append(selected_coeff[j])
        selected_names = selected_names_new
        selected_coeff = selected_coeff_new
    
    return selected_names, regression.coef_

def myregression3(theta,target,names,threshold,n_iter=3,test_size=0.2):
    threshold2 = 0.01
    selected_names = names
    selected_coeff = [t for t in range(theta.shape[1])]
    for it in range(n_iter):
        print("Regression #",it+1)
        X_train, X_test, y_train, y_test = train_test_split(theta[:, selected_coeff], target, test_size=test_size, random_state=0)
        # Fit regression
        X_numpy_train = X_train.detach().numpy()
        y_numpy_train = y_train.detach().numpy()
        X_numpy_test = X_test.detach().numpy()
        y_numpy_test = y_test.detach().numpy()
        regression = sm.OLS(y_numpy_train,X_numpy_train).fit()

        # Apply hard thresholding
        # Apply weighted thresholding based on pvalues
        print(regression.pvalues)
        for i in range(len(selected_coeff)):
            if regression.pvalues[i]>=threshold:
                regression.params[i] = 0
        regression.params = np.where(np.abs(regression.params) >= threshold2, regression.params, 0).flatten()

        # Print coefficients
        for i in range(len(selected_coeff)):
            if regression.params[i] != 0.0:
                print(selected_names[i], " c= ", regression.params[i])

        # Predict and evaluate
        y_pred = regression.predict(X_numpy_test)
        print("Mean Squared Error:", mean_squared_error(y_numpy_test, y_pred))
        print("###################### end of regression",it +1,"##############################")
        print()

        selected_names_new = []
        selected_coeff_new = []
        for j in range(len(regression.params)):
            if regression.params[j] != 0.0:
                selected_names_new.append(selected_names[j])
                selected_coeff_new.append(selected_coeff[j])
        selected_names = selected_names_new
        selected_coeff = selected_coeff_new
        threshold = threshold/2
    
    return selected_names, regression.params

def printPDE_in_u(selected_names, regression_coef_):
    print("dudt =", end=" " )
    for k in range(len(selected_names)):
        if k>0:
            print("      ", end=" " )
        if k==len(selected_names)-1:
            print(regression_coef_[k],"*",selected_names[k])
            break
        print(regression_coef_[k],"*",selected_names[k],"+")

def printPDE_in_v(selected_names, regression_coef_):
    print("dvdt =", end=" " )
    for k in range(len(selected_names)):
        if k>0:
            print("      ", end=" " )
        if k==len(selected_names)-1:
            print(regression_coef_[k],"*",selected_names[k])
            break
        print(regression_coef_[k],"*",selected_names[k],"+")

def printPDE(selected_names, regression_coef_):
    print("dudt =", end=" " )
    for k in range(len(selected_names)):
        if k>0:
            print("      ", end=" " )
        if k==len(selected_names)-1:
            print(regression_coef_[k],"*",selected_names[k])
            break
        print(regression_coef_[k],"*",selected_names[k],"+")
