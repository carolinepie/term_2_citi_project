import numpy as np
# import tensorly as tl
from tensorly.regression.tucker_regression_changed import TuckerRegressor

from sklearn.linear_model import Lasso, Ridge
# from sklearn.svm import LinearSVR
# from sklearn.ensemble import AdaBoostRegressor

if __name__ == '__main__':
    shapes = (100, 4, 8, 3)
    SEED = 100
    # SEED = np.random.randint(10000)
    np.random.seed(SEED)
    X = np.arange(np.prod(shapes)).reshape(shapes) + 1
    np.random.seed(SEED)
    X = X * np.random.randn(*shapes)
    np.random.seed(SEED)
    y = 2 * X + np.random.randn(*X.shape)
    for _ in range(len(shapes) - 1):
        y = y.sum(axis=-1)

    print("original fit result:")
    # estimator = TuckerRegressor(weight_ranks=[1]*len(shapes))
    estimator = TuckerRegressor(weight_ranks=[1]*(len(shapes)-1))
    # estimator = TuckerRegressor(weight_ranks=[2,2])
    estimator.fit(X, y)
    y_pred = estimator.predict(X)
    print(y_pred[:4])

    # print("fit2 result:")
    # estimator4 = TuckerRegressor(weight_ranks=[1]*(len(shapes)-1))
    # estimator4.fit2(X, y, [1,1e-4,1e8])
    # y_pred4 = estimator4.predict(X)
    # print(y_pred4[:4])

    # print("original fit result (same fit):")
    # estimator.fit(X, y)
    # y_pred_ = estimator.predict(X)
    # print(y_pred_[:4])

    # print("original fit result (fit2):")
    # estimator.fit2(X, y, [1,0.1,10])
    # y_pred__ = estimator.predict(X)
    # print(y_pred__[:4])


    print("fit2 result2:")
    estimator5 = TuckerRegressor(weight_ranks=[1,2,3])
    estimator5.fit2(X, y)
    y_pred5 = estimator5.predict(X)
    print(y_pred5[:4])

    print("fit2 result2:")
    estimator7 = TuckerRegressor(weight_ranks=[1,2,3])
    estimator7.fit2(X, y)
    y_pred7 = estimator7.predict(X)
    print(y_pred7[:4])

    print("fit2 result2:")
    estimator6 = TuckerRegressor(weight_ranks=[1,2,4])
    estimator6.fit2(X, y)
    y_pred6 = estimator6.predict(X)
    print(y_pred6[:4])

    # print("new fit result:")
    # estimator_new = TuckerRegressor(weight_ranks=[1]*(len(shapes)-1))
    # # estimator_new = TuckerRegressor(weight_ranks=[2,2])
    # estimator_new.fit_new(X, y)
    # y_pred_new = estimator_new.predict(X)
    # print(y_pred_new[:4])

    # # print(np.allclose(y_pred, y_pred_new))

    # print("other regressor attempt:")
    # estimator1 = TuckerRegressor(weight_ranks=[1]*(len(shapes)-1))
    # estimator1.fit_new(X, y, Lasso(fit_intercept=False))
    # y_pred1 = estimator1.predict(X)
    # print(y_pred1[:4])

    # estimator2 = TuckerRegressor(weight_ranks=[1]*(len(shapes)-1))
    # estimator2.fit_new(X, y, LinearSVR(fit_intercept=False)) # convergence warning # TODO
    # y_pred2 = estimator2.predict(X)
    # print(y_pred2[:4])

    # 'AdaBoostRegressor' object has no attribute 'coef_' # TODO
    # estimator3 = TuckerRegressor(weight_ranks=[1]*(len(shapes)-1))
    # estimator3.fit_new(X, y, AdaBoostRegressor(Ridge(fit_intercept=False)))
    # y_pred3 = estimator3.predict(X)
    # print(y_pred3[:4])
    