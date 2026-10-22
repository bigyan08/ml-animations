"""Verified SVM fits and geometry, computed without rendering side effects."""
import numpy as np
from sklearn.svm import SVC

X = np.array([[1.,1.],[2.,2.],[3.,3.],[4.,3.],[4.,5.]])
Y = np.array([-1,-1,1,1,1])
OUTLIER = np.array([1.5,1.5])

def linear_model():
    return SVC(kernel="linear", C=1000, tol=1e-9).fit(X,Y)

def soft_model(c):
    return SVC(kernel="linear", C=c, tol=1e-9).fit(np.vstack([X,OUTLIER]),np.r_[Y,1])

def nonlinear_data():
    inner=np.array([[3.,3.],[2.5,3.],[3.5,3.],[3.,2.5],[3.,3.5]])
    angle=np.arange(12)*2*np.pi/12
    outer=np.c_[3+2*np.cos(angle),3+2*np.sin(angle)]
    return np.vstack([inner,outer]),np.r_[np.ones(len(inner)),-np.ones(len(outer))]

def nonlinear_model():
    x,y=nonlinear_data()
    return SVC(kernel="rbf",C=10,gamma=1,tol=1e-9).fit(x,y)

def candidate_clearance(w,b):
    """Smallest signed perpendicular distance to this separating line."""
    w=np.asarray(w,dtype=float)
    return float(np.min(Y*(X@w+b))/np.linalg.norm(w))

def verify_math():
    model=linear_model()
    assert np.allclose(model.coef_[0],[1,1],atol=1e-6)
    assert np.allclose(model.intercept_,[-5],atol=1e-6)
    assert set(model.support_)=={1,2}
    assert np.all(Y*model.decision_function(X)>=1-1e-6)
    assert model.predict([[2.8,2.6]])[0]==1
    assert abs(model.decision_function([[2.8,2.6]])[0]-.4)<1e-6
    assert np.isclose(2/np.linalg.norm(model.coef_[0]),np.sqrt(2))
    assert candidate_clearance([1,1],-5)>candidate_clearance([1,1],-4.2)
    assert candidate_clearance([1,1],-5)>candidate_clearance([1.5,.5],-5)
    # OUTLIER is a convex combination of two negative examples: no hard separator.
    assert np.allclose(OUTLIER,(X[0]+X[1])/2)
    nonlinear=nonlinear_model(); x,y=nonlinear_data()
    assert np.array_equal(nonlinear.predict(x),y)
    return {'weights':model.coef_[0].tolist(),'bias':float(model.intercept_[0]),
            'support_vectors':model.support_vectors_.tolist(),'prediction_score':.4,
            'margin_width':float(2/np.linalg.norm(model.coef_[0]))}

if __name__ == "__main__":
    import json
    print(json.dumps(verify_math(),indent=2))
