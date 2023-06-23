# THz-STS-Algorithm
A compilation of functions to perform the steady-state terahertz scanning tunneling spectroscopy (THz-STS) algorithm described in Ammerman, S.E. et al. Nat Commun 12, 6794 (2021). https://doi.org/10.1038/s41467-021-26656-3 and functions to determine a true waveform from THz cross-correlation measurements described in ...

## Installation

Run the following to install

```python
pip install thzsts
```

## Usage

```python
import thzsts

...

```

See the jupyter notebook in example directory for a demonstration with real data.

## Documentation

All functions explained here can be found in the "thzsts.py" script. To use the functions in a jupyter notebook, you can place a copy of the script in the same folder and insert the import statement ```import thzsts as f``` (f can be replaced by whatever you want, you call functions with f.function_name).

### 1.1 Theory behind the inversion algorithm

This section gives a brief recap of the steady-state inverison algorithm described in "Algorithm for subcycle terahertz scanning tunneling spectroscopy" S. E. Ammerman et al., Phys. Rev. B 105, 115427 (2022).

When we measure "QE curves" we measure the rectified (integrated) charge across the junction for the whole waveform. The junction is described by the $I(V)$ conductance curve. The voltage pulse induced by the THz field is labeled as $V_\mathrm{THz}$. The rectified charge for a single pulse can be described as
$$Q_\mathrm{THz}= \int_{-\infty}^{+\infty} I(V_\mathrm{THz}(t))dt$$

To study physical phenomena, we would like to investigate features in the $I(V)$ curve. The goal of the inversion algorithm is to extract the $I(V)$ curve from the $Q(E)$ curve using the shape of the ingoing THz waveform. Mathematically, we can express the waveform as $V_\mathrm{THz}=V_\mathrm{pk}V_0(t)$ where $V_0(t)$ describes the temporal normalized shape and $V_\mathrm{pk}$ describes the scaling factor of the waveform that scales linearly with the THz field strength. 

To entangle the integral above, we approximate the $I(V)$ curve with a polynomial of order N. $I(V)=\sum_{n=1}^N A_n V^n$ where $A_n$ are constant coefficients.

Plugging the approximation and the mathematical description of the waveform into the integral for $Q_\mathrm{THz}$ yields
$$Q_\mathrm{THz}(V_{pk})= \int_{-\infty}^{+\infty}  \sum_{n=1}^N A_n (V_\mathrm{pk}V_0(t))^n dt = \sum_{n=1}^N A_n V_\mathrm{pk}^n \int_{-\infty}^{+\infty} V_0(t)^n dt \equiv \sum_{n=1}^N A_n V_\mathrm{pk}^n B_n$$


$B_n$ is the remaining integral after applying the approximation. If we know the waveform shape, this can easily be calculated. The waveform shape can be measured by EOS, PES or with our new method of THz-CC.

We have to determine the coefficients $A_n$ to extract the $I(V)$ curve. This is possible by determining the voltage calibration factor $\alpha$ which converts $V_\mathrm{pk}=\alpha E_\mathrm{THz}$ and fitting our $Q_\mathrm{THz}(V_{pk}=\alpha E_\mathrm{THz})$ data to the equation above. With this we know all other terms in the equation besides $A_n$.

Note that, we cannot determine the linear term of the $I(V)$ curve because  $B_1=\int_{-\infty}^{+\infty} V_0(t) dt = 0$.

### 1.2 Theory behind curve fitting

#### 1.2.1 Linear Regression

To make the fitting process as straight forward as possible we can simplify the fit equation to $$Q_\mathrm{THz}(V_\mathrm{pk})=\sum^{p_\mathrm{max}}_{p=2} C_p V_\mathrm{pk}^p$$

Fitting a polynomial sum to some input datapoints can be done with linear regression which can be solved analytically and has an exact solution.

The input datapoints are usually labeled as $\vec{x}$ and $\vec{y}$ (in our case $x=V_\mathrm{pk}$ and $y=Q_\mathrm{THz}(V_\mathrm{pk})$). 

First, we set up the so-called design matrix using an $\vec{x}$-value vector of length n ($x_0$ to $x_n$) and the polynomial sum with highest order $p$ and lowest order $2$. The design matrix can be written as
$$
\vec{X}=\begin{bmatrix} 
	x_0^2 & x_0^3 & ... & x_0^{p-1} & x_0^p \\
	x_1^2 & x_1^3 & ... & x_1^{p-1} & x_1^p\\
	... & ... & ... & ... & ... \\
    x_n^2 & x_n^3 & ... & x_n^{p-1} & x_n^p \\
	\end{bmatrix}
$$

The horizontal dimension (column count) corresponds to the number of terms in the polynomial. The vertical direction (row count) corresponds to the number of data points.

We can set up the linear regression as a linear algebra problem with $\vec{X} \cdot \vec{b} = \hat{f}(x)$. $\vec{b}$ corresponds to the fit coefficients $C_p$ in our problem. $\hat{f}(x)$ are the predicted values by our fit model. To put it in a simple way, we want to choose $\vec{b}$ such that $\hat{f}(x)$ matches $\vec{y}$ from our input data well.

The metric (also referred to as loss or loss function) used in linear regression to determine a good fit is mean squared error (MSE) which is described as
$$L_\mathrm{MSE}= \frac{1}{n}\sum_{i=0}^{n} (y_i-\hat{f}(x_i))^2$$

or writing this with matrix notation 
$$L_\mathrm{MSE}= \frac{1}{n}|| \vec{y}-\vec{X} \cdot \vec{b}||^2$$.

To minimize the MSE we take the deriative with respect to $\vec{b}$ and set it to zero and solve for $\vec{b}$.

$$\frac{\partial}{\partial b} L_\mathrm{MSE} = \frac{\partial}{\partial b} (\vec{y}-\vec{X} \cdot \vec{b})^T(\vec{y}-\vec{X} \cdot \vec{b})= - 2\vec{X}^T(\vec{y}-\vec{X} \cdot \vec{b})$$

$$\vec{X}^T\vec{y}-\vec{X}^T\vec{X}\vec{b}=0 \quad \Rightarrow \quad \vec{b}=(\vec{X}^T\vec{X})^{-1}\vec{X}^T\vec{y}
$$

The matrix $(\vec{X}^T\vec{X})$ must be invertible and positive definite in order for us to solve this equation.

#### 1.2.2 The problem of overfitting and how to counteract it

The challenge with fit models is that the MSE generally decreases with increasing polynomial order which makes it seem like the data fits better with a higher order.
![title](https://scikit-learn.org/0.15/_images/plot_underfitting_overfitting_0011.png)

The picture above portrays the problem pretty well. We would like to have a metric that helps us find the region of polynomial orders where we are not under- nor overfitting. 

One way to do this is to make the fitting procedure a two step process. 
- In step 1, we use a so-called training data set to find the fit parameters. 
- In step 2, we take an independent second testing dataset and calculate the MSE of these new datapoints to the fit model determined by the first dataset. 

The large oscillation that an overfitted model develops while minimizing the error of individual (outlier) data points will have a negative impact on the MSE of the testing data. Repeating this procedure for different polynomial orders will tell us at what point the fitted model represents the actual underlying model of both datasets and added model complexity does not just optimize the MSE of the training data.

<img src=https://forums.fast.ai/uploads/default/original/2X/d/db413a396fe0d3555f15e78e05bb36a2141bb8a4.jpg width="400">

#### 1.2.3 (Shuffle split) Cross validation method

A way to implement the train-test strategy dicussed above it to use the cross validation method. The idea here is to split up the measured data multiple times into test and training samples which is often more feasible than having multiple input data sets.

The input parameters of the cross validation method are 
- the ratio between test and train data, for example 80% of the data to train (fit) the model and 20% data to test the model
- the number of cross validations (iterations), e.g. 5 iterations of splitting up the data into fit and train samples 

Each iteration gives a MSE between the test data and the train data model. We can calculate the mean error over all iterations as well as the standard deviation of the error. This will give us the plot from the previous section and we can determine a polynomial order where we prevent under or overfitting. 

<img src=https://miro.medium.com/v2/resize:fit:1200/1*AAwIlHM8TpAVe4l2FihNUQ.png width="800">

In our case the input data is the QE curve. This data is ordered, meaning that the x-axis (THz field axis) is continuously increasing. It would not make much sense to split the ordered data in the way the graphic above depicts.

Instead, we can use shuffle split cross validation. The only difference here is that the train and test samples are split randomly along the data vectors. This gives us the possibility to make a huge number of splits and in practice the limiting factor for how many iterations to make is solely the time it takes to compute the fit model and errors. The fit coefficents are averaged over all splits.

<img src=https://inria.github.io/scikit-learn-mooc/_images/shufflesplit_diagram.png width="800">

### 1.3 Functions for applying the inversion algorithm and extracting the IV curve

#### 1.3.1 The main function

```python
def inversion_algorithm_splits(wf, qe, p_order, splits, train_test_ratio, rand_seed=2237, mse_metrics=False):
    """Main function to perform extraction algorithm using shuffle splits."""

    # Calculate Coefficient (ref. Algorithm paper)
    Bn = calculate_Bn(wf, p_order)   # calculate waveform integral up to order p
    Cn, test_loss_avg, test_loss_std, train_loss_avg, train_loss_std, fit_qe, mse_qe_fit = shuffle_splits_fit(
        qe, p_order, splits, train_test_ratio, rand_seed)
    
    ext_didv = 0
    ext_iv = 0
    # loop to calculate An (polynomial prefactors for the IV and dIdV curves)
    for i in range(len(Cn)):
        # calculate An
        An = Cn[i] / Bn[i]
        # add contributions to recovered didv
        ext_didv += (i+2)*An*qe[0]**(i-1+2)
        ext_iv += An*qe[0]**(i+2)

    # make I(E) and dIdE two column matrices 
    ext_didv = np.vstack((qe[0], ext_didv))
    ext_iv = np.vstack((qe[0], ext_iv))

    # calculate current pulses i(t) and and simulate the qe measurement 
    sim_qe, sim_it = rectify_QE(wf, ext_iv)
    
    # if interpolation range out of range set error to nan
    try:
        mse_qe_sim = mean_squared_error(qe[1], sim_qe[1])
    except:
        mse_qe_sim = np.nan
    
    if mse_metrics is True:
        return (ext_iv, ext_didv, fit_qe, sim_qe, sim_it, test_loss_avg, test_loss_std, 
               train_loss_avg, train_loss_std, mse_qe_fit, mse_qe_sim)
    else:
        return ext_iv, ext_didv, fit_qe, sim_qe, sim_it
```

The inputs of the main function are
- wf: two column array (time, norm. waveformshape), main peak normalized to +1 
- qe: two column array (E field or voltage, THz induced current)
- p_order: polynomial order of the fit (degree of highest term in the sum)
- splits: number of iterations for the shuffle split cross validation
- train_test_ratio: e.g. if 0.2 we use 20% of the data to train, 80% of the data to fit
- rand_seed: to make things repeatable set the seed to arbitrary number, but with high number of splits not a big influence

#### Step 1: calculating $B_n$

Function called from main function with line ```Bn = calculate_Bn(wf, p_order)```.

```python
def calculate_Bn(wf, p_order):
    """Calculate the B factor that only depend on the waveform."""
    Bn = np.zeros(p_order-1) # leave of constant and linear term
    for i in range(p_order-1):
        Bn[i] = np.real(integrate.simps(y=wf[1]**(i+2),x=wf[0]))        
    return Bn
```

This function simply creates an array of the length of polynomial terms and fills it with the integrated exponentiated waveform. 

#### Step 2: Performing the fit using shuffle split cross validation

Function called from main function with line ```Cn, loss_avg, loss_std, fit_qe = shuffle_splits_fit(wf, qe, p_order, splits, train_test_ratio, rand_seed)```. 

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error

def shuffle_splits_fit(qe, p_order, splits, train_test_ratio, rand_seed):
    """Perform inversion with shuffle split fit."""
    
    # Make a design matrix with polynomial features
    # start with quadratic term (constant term not included)
    poly = PolynomialFeatures((2, p_order), include_bias=False)   
    DM = poly.fit_transform(np.reshape(qe[0], (qe[0].size, 1)))  # qe needs specific shape
    
    # set up fit model for fit
    lin_mod = linear_model.LinearRegression

    # set up shuffle split sampling with given parameters, if random_state None different every time
    shuffle = ShuffleSplit(splits, test_size=train_test_ratio, random_state=rand_seed)

    # array to hold errors between train and test data sets
    test_losses = np.zeros(splits)
    train_losses = np.zeros(splits)
    # array to hold coefficients for polynomial terms
    coeffs = np.zeros((splits, p_order-1))
    
    idx = 0
    q = qe[1]
    # loop through split sets
    for train, test in shuffle.split(qe.T):
        
        # fit the polynomial model to the training data without an intercept (zero crossing at zero)
        reg = lin_mod(fit_intercept=False).fit(DM[train], q[train])
        # predict the QE curve for the test data with the fit model from the train data
        pred_q_test = reg.predict(DM[test])
        pred_q_train = reg.predict(DM[train])
        
        # calculate the mean squared error loss between the predicte Q from the train data and the actual Q
        test_losses[idx] = mean_squared_error(pred_q_test, q[test])
        train_losses[idx] = mean_squared_error(pred_q_train, q[train])
        # get fit coefficients from linear regression model  
        coeffs[idx, :] = reg.coef_
        idx += 1
        
    # calculate loss outputs
    test_loss_avg = np.mean(test_losses)
    test_loss_std = np.std(test_losses)
    train_loss_avg = np.mean(train_losses)
    train_loss_std = np.std(train_losses)
    
    # calculate the mean value for all coefficients over all splits
    Cn = np.mean(coeffs, axis=0)
    
    # calculate the fitted QE curve
    fit_qe = np.dot(DM, Cn)
    fit_qe = np.vstack((qe[0], fit_qe))
    
    # calculate more error statistics
    mse_qe_fit = mean_squared_error(qe[1], fit_qe[1])
    
    return Cn, test_loss_avg, test_loss_std, train_loss_avg, train_loss_std, fit_qe, mse_qe_fit
```

This function uses the scikit learn library to perform linear regression with shuffle split cross validation as explained in the previous section of this document. I find the scikit learn documentation to be very helpful most of the time https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression.

In this function the main steps are:
- setup the design matrix (DM) using the PolynomialFeatures. See example below for how this works
```python
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
x = np.arange(5).reshape(5, 1)
print("x:\n", x)
poly = PolynomialFeatures(2)
X = poly.fit_transform(x)
print("Design matrix:\n", X) 
```
Input:
```
x:
 [[0]
 [1]
 [2]
 [3]
 [4]]
```
Output:
```
Design matrix:
 [[ 1.  0.  0.]
 [ 1.  1.  1.]
 [ 1.  2.  4.]
 [ 1.  3.  9.]
 [ 1.  4. 16.]]
 ```

- setup a linear model object that we will use for linear regression later
- setup a shufflesplit model which takes the number of splits, the test_size (test_train_ratio) and the random_seed as input

- Shuffle split iteration loop
    - Apply the shufflesplit object to the design matrix and loop through the iterations. In each loop the rows of the designmatrix (datapoints in the QE curve) are distributed differently between train and test data according to the defined ratio (the variables train and test are arrays of datapoint indices).
    - For each split/iteration: apply the linear model to the training part of the data, i.e. perform the fit. The variable reg is a fit object and has various methods and properties.
    - Use the predict method from the fit object to get the $\hat{f}(x)$ values at the x points of the test data to compare to the actual measured $y$ values at those points.
    - Calculate the MSE for the predicted test points to the actual test points and get the fit coefficients from the fit object

- from all the iterations calculate the mean loss and loss standard deviation as well as the mean coefficients
- build a two column vector to output the fit curve for the QE

#### Step 3: Calculate the extracted $I(V)$ and $dI/dV$ curve

In the loop of the main function we calculate the coefficients $A_n$ for the extracted $I(V)$ and its derivative. 
```python
    for i in range(len(Cn)):
        # calculate An
        An = Cn[i] / Bn[i]
        # add contributions to recovered didv
        ext_didv += (i+2)*An*qe[0]**(i-1+2)
        ext_iv += An*qe[0]**(i+2)
```

#### Step 4: Calculate current pulses and simulated QE curve

Function called from the main function with line ```sim_qe, sim_it = rectify_QE(wf, ext_iv)``` .

```python
def rectify_QE(wf, ext_iv, bias=0):
    """Rectify the waveform on the recovered IV curve to get simulated QE curve."""
    
    # interpolate the extracted IV curve to be able to rectify on arbitrary points
    interp_IV = interp1d(x=ext_iv[0], y=ext_iv[1])

    # set up Vpk array, 1:-1 to avoid interpolation range error 
    Vpks = ext_iv[0] #, 1:-1]
    
    # set up array for rectified QE 
    sim_qe = np.zeros((2, len(Vpks)))
    sim_qe[0] = Vpks
    
    #  set up array for current pulses
    sim_it = np.zeros((len(Vpks), len(wf[0])))

    # loop to rectify waveform for each Vpk
    for i, Vpk in enumerate(Vpks):
        VTHz = Vpk * wf[1] + bias
        
        try:
            ITHz = interp_IV(x=VTHz)
        except:
            ITHz = [np.nan]*len(wf[0])
            
        sim_it[i, :] = ITHz
        sim_qe[1, i] = integrate.simps(y=ITHz, x=wf[0])
        
    return sim_qe, sim_it
```

Here, we use the input waveform to simulate the QE measurement using the extracted $I(V)$ curve. We have to use interpolation to create a function object which we can probe at arbitrary x points. We use the x-axis of the extracted $I(V)$ curve a the x-axis ($V_\mathrm{pk}$) of the simulated QE curve. 

We then loop over the $V_\mathrm{pk}$  and in each iteration scale the normalized waveform shape by $V_\mathrm{pk}$. Sometimes, we enounter waveforms that if normalized to their main peak have opposite sign peaks that are greater than 1. This is not a case we want to deal with necessarily but to be able to handle it, we can implement a "try: ... except: ..." statement. The interpolated function only exists over the range of the input QE curve. If the pulse is scaled to the max  $V_\mathrm{pk}$ but the normalized peak is not the maximum of the waveform shape we will get an error. The except statement catches that and writes "nan" into the rectified current. 

sim_it is a 2D array made of the current pulses for each $V_\mathrm{pk}$ waveform. Note that this represents current pulses without the THz-CC in place.

sim_qe is a two column array wih the same size as the input QE curve and they should look practically identical when they are plotted together.

Note that, if we want to simulate QE curves with a bias voltage $V_\mathrm{d.c.}\neq 0$, for example when simulating a voltage calibration, we can also use this function. Simply set the input parameter ```bias``` to the desired value. Make sure that it has the same units as the x-axis of the IV curve.

#### Step 5: Calculate errors for mean QE fit and QE sim

The last lines of the main function are used to calculate the MSE for the shuffle split averaged fit to the data and the simulated QE curve with respect to the data.

### 1.4 Simulating the THz-CC measurement

The inversion algorithm does not care at all how the waveform was measured that is used as the input, as long as it has the correct normalized shape. Our current options are EOS, PES and THz-CC measured waveforms. EOS is not ideal because we assume that the waveform in the STM junction looks different from free-space. 

The idea of THz-CC waveforms is that we split up the THz pulse into a big main pulse and a small probe pulse (because of the optical setup, the polarization of the probe pulse is opposite of the main pulse). The THz field strength of the main pulse can be scaled. The goal is to use the peak of the main pulse to drive the probe pulse (which is scanned across the main pulse) into a linear regime of the sample's $I(V)$ curve. By chopping the probe pulse and measuring the differential current the shape of the probe pulse is recorded.

The functions below recreate this process using the $I(V)$ extracted with the algorithm.

#### 1.4.1 Main function

The inputs of the main function are
- wf: the waveform as a two column array where the waveform is normalized to its main peak and the main peak is pointing up
- efield: the field at which the THz-CC waveform is simulated (size of the main pulse), this can also be a voltage if the input IV curve is in volts
- ext_iv: two column array IV curve can have field strength or volts as x-axis (has to match efield units)
- probe_size: relative size of the probe pulse compared to the main pulse at 100% (since in the experiment only the main pulse is scaled)
- t_min and t_max: the time range in which the pulse is supposed to be simulated, usually set to range of input waveform time axis
- delay_pts: number of points along time axis of simulated pulse

```python
def simulate_Thz_CC(wf, efield, ext_iv, probe_size=0.05, t_min=-5, t_max=5, delay_pts=400): # wf two cols norm., efield in %, ext_iv two cols
    '''Simulate the THz-CC waveform measurement using a waveform at a specific E_THz.'''  
    e_max = 100 # probe size refers to e_max, if e_max = 100% and probe_size = 0.05 => probe peak at 5%
    
    # Stationary (big) waveform
    wf_stat = np.vstack((wf[0], wf[1]*efield))
    # Small probe waveform has opposite sign of stationary wf
    wf_probe = np.vstack((wf[0], wf[1]*e_max*probe_size*(-1)*np.sign(efield)))
    
    # set up delay array
    cc_delay = np.linspace(t_min, t_max, delay_pts)
    
    # generate waveforms for each delay
    cc_wfs = generate_cc_waveforms(wf_stat, wf_probe, cc_delay)
    
    # Rectify and calculate waveform
    thz_cc_wf = np.zeros(delay_pts)
    for i in range(len(cc_wfs)):
        # Rectify each waveform on extracted Iv curve 
        thz_cc_wf[i] = rectify_Qt(np.vstack((wf[0], cc_wfs[i])), ext_iv)
        
    # Remove mean offset and normalize
    thz_cc_wf = np.subtract(thz_cc_wf, np.mean(thz_cc_wf)) 
    thz_cc_sim = np.vstack((cc_delay, thz_cc_wf)) 
    
    return thz_cc_sim
```

In the first lines of the function the stationary (main) and the probe waveforms are setup according to the input field strength and the time_axis is created.

#### 1.4.2 Generating the incoming waveforms

Next, we generate the overlapping waveforms at each delay. The function below takes car of that and is called from the main function with the line ```cc_wfs = generate_cc_waveforms(wf_stat, wf_probe, cc_delay)```.

```python
def generate_cc_waveforms(wf_stat, wf_probe, cc_delay):
    '''Generte overlapping waveform for stationary and probe pulse with each cc time delay.'''
    # Define min and max of the time delay
    tmin = cc_delay[0]
    tmax= cc_delay[-1]
    
    # Define time axes for data and simulated waveform
    dt = (wf_stat[0,-1]-wf_stat[0,0])/len(wf_stat[0]) # data time step
    delay_step = (tmax-tmin)/len(cc_delay)            # simulation delay-time step
    
    # add zeros of size half the scan range to each side as a fraction of Tpts  
    stat_wave = np.pad(wf_stat[1],[int(abs(tmin/dt)),int(abs(tmax/dt))]) 
    probe_wf = np.pad(wf_probe[1],[int(abs(tmin/dt)),int(abs(tmax/dt))]) 

    # adjustable wave starts rolled forward by haalf the data time range
    adj_wave = np.roll(probe_wf,int(abs(tmin/dt)))

    # loop through simulation cc time delay and append waveforms for each delay step
    waveforms = []
    for n in range(len(cc_delay)):
        # roll adj wave forward by one delay step each loop iteration
        wave = stat_wave + np.roll(adj_wave,-int(n*delay_step/dt))
        # append the waveforms after removing the padding
        waveforms.append(wave[int(abs(tmin/dt)):-int(abs(tmax/dt))])
        
    return waveforms
```

This function outputs a matrix of waveforms. The numpy documentation is helpful for understanding the details of each step. 
1. Define the time step along the actual time axis of the waveform ```dt``` and the ```delay step``` which are the steps for going from one overlap of main and probe pulses to the next.
2. Pad the waveform arrays with zeros to achieve a bigger range to overlap them.
3. Initialize the positions of the adjustable (probe) waveform relative to the stationary (main) waveform.
4. Loop through the overlap delays and add the adjustable and stationary waveform at each step and append them to the matrix. Before the waveforms are appended to the matrix they are trimmed back to their original size (the padding is removed).

The numpy function roll is very helpful here, e.g. np.roll([1, 2, 3, 4], 1) leads to [4, 1, 2, 3]. 

#### 1.4.3 Rectify each waveform

In the next part of the main function the actual measurement takes place. With ```thz_cc_wf[i] = rectify_Qt(np.vstack((wf[0], cc_wfs[i])), ext_iv)``` each waveform from the matrix is rectified on the $I(V)$ curve. The function used for this is very similar to ```rectify_QE(wf, ext_iv)``` however, here we only calculate a single point and do not sweep over a $V_\mathrm{pk}$ range. 

In this function the $I(V)$ curve is interpolated and the currents corresponding to the waveform values along its time axis are integrated. The output is a single point in the simulated THz-CC waveform.


```python
def rectify_Qt(wf, iv):
    IV_interp = interp1d(iv[0], iv[1])
    try:
        It = IV_interp(wf[1])
    except:
        It = [np.nan]*len(wf[0])
        #print("Out of interpolation range, set to zero.")
    else: 
        It = IV_interp(wf[1])
    qt = integrate.simps(It,x=(wf[0]))
    return qt
```

Note that, we have to catch the error of being outside of the $I(V)$ interpolation range. Imagine the range of the measured QE curve is [-100% to 100%]. The extracted $I(V)$ curve will have the same range. This limits us in the field range we can use to simulate the THz-CC waveforms. Assuming a probe size of 8% means that the simulation of a 100% main pulse will lead to a total pulse of over 100% at the constructive interference delays of both pulses. The simulation of the THz-CC waveforms is not possible in this case.

#### 1.4.4 Outputting the simulated waveform

In the last lines of the main function we subtract the mean value which can be seen as equivalent to a differential experiment. The waveform is output as a two column array and can for example be compared to the input waveform of the inversion algorithm or waveforms taken at different field strengths.