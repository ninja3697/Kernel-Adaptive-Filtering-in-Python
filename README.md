# Kernel-Adaptive-Filtering-in-Python


## 1. INTRODUCTION

Kernel method is one of the most popular method in the area of adaptive filtering and signal processing. The reason behind this is the combination of the famed kernel trick and the adaptive filters (like Least Mean Squares(LMS) and Recursive Least Squares(RLS)) algorithm provides an interesting sample-by-sample update for an adaptive filter in reproducing kernel Hilbert spaces (RKHS). Instead of calculating the higher dimensional values in the feature space, the Kernel methods (namely KLMS and KRLS) make use of a mathematical result more popularly known as Kernel Trick. The basic idea of Kernel Trick is that a Mercer kernel function which is applied to pairs of input vectors, can be interpreted as an inner product in a high-dimensional feature space (more formally called Hilbert space), thus allowing inner products in the feature space to be computed without making direct reference to feature vectors.

This idea has been used extensively in recent years, most notably in classification and regression. In this project, we will try to implement this kernel trick on adaptive filters to solve the non-linear prediction problems using linear regression based predictive filters. Also we will compare the effect of various parameters on the performance of kernel filters.


## 2. LITRATURE REVIEW

Machine learning is a branch of artificial intelligence that aims at solving real life engineering problems. It provides the opportunity to learn without being explicitly programmed and it is based on the concept of learning from data. The advantage of machine learning (ML) methods is that it uses mathematical models, heuristic learning, knowledge acquisitions and decision trees for decision making. Thus, it provides controllability, observability and stability. The application of machine learning models on human disease diagnosis aids medical experts based on the symptoms at an early stage, even though some diseases exhibit similar symptoms.

- **Supervised learning: -**

Supervised learning is the most common form of machine learning scheme used in solving the engineering problems. It can be thought as the most appropriate way of mapping a set of input variables with a set of output variables. The system learns to infer a function from a collection of labeled training data. The training dataset contains a set of input features and several instance values for respective features. The predictive performance accuracy of a machine learning algorithm depends on the supervised learning scheme. The aim of the inferred function may be to solve a regression or classification problem.

- **Sequence Learning: -**

Most machine learning algorithms are designed for independent, identically distributed (i.i.d.) data. But many interesting data types are not i.i.d. In particular, the successive points in sequential data are strongly correlated.

Sequence learning is the study of machine learning algorithms designed for sequential data. These algorithms should

1. not assume data points to be independent

2. be able to deal with sequential distortions

3. make use of context information

Sequence Learning Used for Time-Series Prediction, Sequence Labelling etc.

- **Kernel in machine learning**

The idea is to use a higher-dimension feature space to make the data almost linearly separable. There are plenty of higher dimensional spaces to make the data points separable. For instance, we have shown that the polynomial mapping is a great start.

We have also demonstrated that with lots of data, these transformation is not efficient. Instead, we can use a kernel function to modify the data without changing to a new feature plan. The magic of the kernel is to find a function that avoids all the trouble implied by the high-dimensional computation. The result of a kernel is a scalar, or said differently we are back to one-dimensional space.

After we found this function, we can plug it to the standard linear classifier.


## 3. LINEAR ADAPTIVE FILTERING

An _adaptive filter_ is a computational device that attempts to model the relationship between two signals in real time in an iterative manner. Adaptive filters are often realized either as a set of program instructions running on an arithmetical processing device such as a microprocessor or DSP chip, or as a set of logic operations implemented in a field-programmable gate array (FPGA) or in a semicustom or custom VLSI integrated circuit. However, ignoring any errors introduced by numerical precision effects in these implementations, the fundamental operation of an adaptive filter can be characterized independently of the specific physical realization that it takes. For this reason, we shall focus on the mathematical forms of adaptive filters as opposed to their specific realizations in software or hardware.

An adaptive filter is defined by four aspects:

1. the _signals_ being processed by the filter
2. the _structure_ that defines how the output signal of the filter is computed from its input signal
3. the _parameters_ within this structure that can be iteratively changed to alter the filter&#39;s input-output relationship
4. the _adaptive algorithm_ that describes how the parameters are adjusted from one time instant to the next


**3.2 TYPES OF ADAPTIVE FILTERS**

There are numerous types of adoptive filters available. The filters useful for our purpose are the following:

  1. Least Mean Squares (LMS) Filter
  2. Recursive Least Squares (RLS) Filter
  3. Kernel Least Mean Squares (KLMS) Filter
  4. Kernel Recursive Least Squares (KRLS) Filter



**3.3 APPLICATIONS OF ADAPTIVE FILTERS**

  1. **System Identification:**
  
  One common adaptive filter application is to use adaptive filters to identify an unknown system, such as the response of an unknown communications channel or the frequency response of an auditorium, to pick fairly divergent applications. Other applications include echo cancellation and channel identification.

  2. **Noise or Interference Cancellation:**

  In noise cancellation, adaptive filters let you remove noise from a signal in real time. Here, the desired signal, the one to clean up, combines noise and desired information. To remove the noise, feed a signal n&#39;(k) to the adaptive filter that is correlated to the noise to be removed from the desired signal.

  3. **Prediction:**

  Based on the data fed over time, an adaptive filter can predict future values accurately and get better with time.

## 4. Least Mean Squares (LMS) Filter

This is the most basic type of filter available today. It is a stochastic gradient descent method in that the filter is only adapted based on the error at the current time.
LMS has very low complexity due to its simplicity and also produces satisfactory results in most of the cases.

**4.1 CONVERGENCE CONSIDERATIONS OF LMS ALGORITHM**

The first criterion for convergence of the LMS algorithm is convergence in the mean, which is described by
However, this criterion is too weak to be of any practical value, because a sequence of zero - mean, but otherwise arbitrary random, vectors converges in this sense.
A more practical convergence criterion is convergence in the mean square, which is described by [Haykin, 2002]

**4.2 LEARNING CURVE**

Learning curve is an informative way of examining the convergence behavior of the LMS algorithm or in general any adaptive filter. We will use the learning curve a great deal in our experiments to compare the performance of different adaptive filters and effect of different parameters in them. The learning curve is a plot of the mean square error (MSE)  versus the number of iterations **i**. The two main ways to obtain the estimate of  include the ensemble - average approach and the testing mean - square - error approach.
To obtain the ensemble - averaged learning curve, we need an ensemble of adaptive filters, with each filter operating with the same configuration settings such as updating rule, step - size parameter, and initialization. The input and desired signals are independent for each filter. For each filter, we plot the sample learning curve, which is simply the squared value of the estimation error versus the number of iterations. The sample learning curve so obtained consists of noisy components because of the inherently stochastic nature of the adaptive filter. Then we take the average of these sample learning curves over the ensemble of adaptive filters used in the experiment, thereby smoothing out the effects of noise. The averaged learning curve is called the ensemble – averaged learning curve. This method is applicable for any environment, stationary or nonstationary.
The other approach is by setting aside a testing data set before the training. For each iteration, we have the weight estimate . We compute the mean square error on the testing data set by using . Then, we plot the testing MSE versus the number of iterations. This approach only needs one adaptive filter and is computationally cheaper comparing with the ensemble – average approach. However, this method does not apply in situations where the environment is nonstationary.


## 5. Recursive Least Squares (RLS) Filter

Recursive least squares (RLS) is an adaptive filter algorithm that recursively finds the coefficients that minimize a weighted linear least squares cost function relating to the input signals. This approach is in contrast to other algorithms such as the least mean squares (LMS) that aim to reduce the mean square error. In the derivation of the RLS, the input signals are considered deterministic, while for the LMS and similar algorithm they are considered stochastic. Compared to most of its competitors, the RLS exhibits extremely fast convergence. However, this benefit comes at the cost of high computational complexity.

## 6. Kernel Least Mean Squares (KLMS) Filter

A  **kernel adaptive filter**  is a type of nonlinear adaptive filter. Kernel adaptive filters implement a nonlinear transfer function using kernel methods. In these methods, the signal is mapped to a high-dimensional linear feature space and a nonlinear function is approximated as a sum over kernels, whose domain is the feature space. If this is done in a reproducing kernel Hilbert space, a kernel method can be a universal approximation for a nonlinear function. Kernel methods have the advantage of having convex loss functions, with no local minima, and of being only moderately complex to implement.

Because high-dimensional feature space is linear, kernel adaptive filters can be thought of as a generalization of linear adaptive filters. As with linear adaptive filters, there are two general approaches to adapting a filter: the least mean squares filter (LMS) and the recursive least squares filter (RLS).

In KLMS filter, we transform the input  into a high – dimensional feature space.

The new algorithm is computed without using the weights. Instead, we have the sum of all past errors multiplied by the kernel evaluations on the previously received data, which is equivalent to the weights. Therefore, having direct access to the weights enables the computation of the output with a single inner product, which is a huge time saving, but the two procedures are actually equivalent.

**6.1 KERNEL AND PARAMETER SELECTION**

- **Criteria for kernel**

The necessity of specifying the kernel and its parameter applies to all kernel methods, and it is reminiscent of nonparametric regression, where the weight function and its smoothing parameter must be chosen. The kernel is a crucial ingredient of any kernel method in the sense that it defines the similarity between data points.

Due to the richness for approximation and ability to reproduce Hilbert space with universal approximating capability, we will use Gaussian Kernel for our analysis.

- **Criteria for Step Size**

This criterion is same as for LMS. Practically, we find the best suitable step size by using Cross-Validation and testing method.

## 7. Kernel Recursive Least Squares (KRLS) Filter

To derive RLS in reproducing kernel Hilbert spaces (RKHS), we use the Mercer theorem to transform the data  into the feature space F as . The basic idea behind kernel methods is that a Mercer kernel function, which is applied to pairs of input vectors, can be interpreted as an inner product in a high-dimensional Hilbert space, thus allowing inner products in the feature space to be computed without making direct reference to feature vectors. This idea, which is commonly known as the &quot;kernel trick,&quot; has been used extensively in recent years, most notably in classification and regression [EMM,2004].

In the simplest form of the RLS algorithm, we minimize at each iteration _i_ as the sum of the squared errors.

needs to be solved recursively. As RKHS is a high-dimensional space, regularization here is necessary. Similar to KLMS, KRLS is also solved using Kernel trick.

## 8. TOOLS AND TECHNOLOGY USED

- PROGRAMMING LANGUAGE: Python 3.6
- LIBRARIES USED:
  - Numpy
  - Scipy
  - Pandas
  - Matplotlib
  - Padasip
  - Scikit – learn
  - Kaftools

- TOOLS USED:
  - Anaconda Distribution 3.6.6
  - Jupyter Ipython Notebook

## 9. REFERENCES

- Adaptive Filter Theory, 4th edition. Upper Saddle River, NJ - Prentice Hall 2002.
- Engel; S. Mannor; R.Meir. &quot;The kernel recursive least-squares algorithm.&quot; _IEEE Signal Processing Society._[[YMM 2004]](https://ieeexplore.ieee.org/document/1315946)
- Kernel Adaptive Filtering, By Weifeng Liu, José C. Príncipe, and Simon Haykin Copyright © 2010 John Wiley &amp; Sons, Inc.
- [https://en.wikipedia.org/wiki/Adaptive\_filter](https://en.wikipedia.org/wiki/Adaptive_filter)
- [https://en.wikipedia.org/wiki/Least\_mean\_squares\_filter](https://en.wikipedia.org/wiki/Least_mean_squares_filter)
- [https://en.wikipedia.org/wiki/Kernel\_adaptive\_filter](https://en.wikipedia.org/wiki/Kernel_adaptive_filter)
- [https://en.wikipedia.org/wiki/Recursive\_least\_squares\_filter](https://en.wikipedia.org/wiki/Recursive_least_squares_filter)
- Filters source:
  -  LMS and RLS: [https://pypi.org/project/padasip/](https://pypi.org/project/padasip/)
  - Kernel Version: [https://github.com/pin3da/kernel-adaptive filtering/blob/master/filters.py](https://github.com/pin3da/kernel-adaptive%20filtering/blob/master/filters.py)
