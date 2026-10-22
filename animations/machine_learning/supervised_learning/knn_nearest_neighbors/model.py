"""The full ten-student dataset, exact neighbors and real decision maps."""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

X=np.array([[2.8,3.0],[3.4,3.3],[3.6,3.2],[2.3,2.5],[2.15,2.75],
            [1.5,1.4],[1.1,2.3],[4.0,4.0],[4.6,3.8],[5.2,4.7]])
Y=np.array([0,1,1,0,0,0,0,1,1,1])
QUERY=np.array([3.,3.])

def neighbors(k,query=QUERY):
    distances=np.linalg.norm(X-np.asarray(query),axis=1)
    indices=np.argsort(distances,kind="stable")[:k]
    return indices,distances[indices]

def predict(k,query=QUERY):
    indices,_=neighbors(k,query)
    return int(np.bincount(Y[indices],minlength=2).argmax())

def classifier(k):
    return KNeighborsClassifier(n_neighbors=k,weights="uniform",p=2,algorithm="brute").fit(X,Y)

def region_map(k,resolution=160):
    """Vectorized exact voting grid; lower y first, ties resolved by input order."""
    values=(np.arange(resolution)+.5)*6/resolution
    gx,gy=np.meshgrid(values,values)
    queries=np.c_[gx.ravel(),gy.ravel()]
    d2=np.sum((queries[:,None,:]-X[None,:,:])**2,axis=2)
    indices=np.argsort(d2,axis=1,kind="stable")[:,:k]
    labels=(np.sum(Y[indices],axis=1)>k/2).astype(int)
    return labels.reshape(resolution,resolution)

def verify_math():
    i,d=neighbors(5)
    assert np.array_equal(i,[0,1,2,3,4])
    assert np.allclose(d,[.2,.5,np.sqrt(.4),np.sqrt(.74),np.sqrt(.785)])
    assert [predict(k) for k in [1,3,5]]==[0,1,0]
    for k in [1,3,5]:
        model=classifier(k); distances,indices=model.kneighbors([QUERY])
        assert np.array_equal(indices[0],neighbors(k)[0])
        assert model.predict([QUERY])[0]==predict(k)
    # Unit conversion changes raw distances but training-fitted standardization cancels it.
    minutes=X*np.array([60.,1.]); query_minutes=QUERY*np.array([60.,1.])
    hours_scaler=StandardScaler().fit(X); minute_scaler=StandardScaler().fit(minutes)
    assert np.allclose(hours_scaler.transform(X),minute_scaler.transform(minutes))
    assert np.allclose(hours_scaler.transform([QUERY]),minute_scaler.transform([query_minutes]))
    assert np.isclose(np.linalg.norm((X[1]-QUERY)*[60,1]),np.sqrt(576+.09))
    return {'nearest_five_distances':d.tolist(),'labels':Y[i].tolist(),
            'predictions':{str(k):predict(k) for k in [1,3,5]},
            'minutes_distance':float(np.sqrt(576+.09))}

if __name__=="__main__":
    import json
    print(json.dumps(verify_math(),indent=2))
