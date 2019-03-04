import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


# sciserver cas job to retrieve galaxy table
"""
select 
  phot.objID as objid, probPSF as star, fieldID as field, 
  psfMag_r, psfMagErr_r, cModelMag_r, cModelMagErr_r,
  petroR50_r, petroR50Err_r, petroR90_r, petroR90Err_r, 
  deVRad_r, deVRadErr_r, deVAB_r, deVABErr_r, 
  expRad_r, expRadErr_r, expAB_r, expABErr_r,
  gx.nvote, gx.p_el, gx.p_cw, gx.p_acw, gx.p_edge, gx.p_dk, gx.p_mg, gx.p_cs, gx.p_el_debiased, gx.p_cs_debiased, gx.spiral, gx.elliptical, gx.uncertain
into mydb.galaxies
from PhotoObjAll as phot
  join dbo.zooSpec as gx on phot.objID=gx.objid
"""

gxdata = pd.read_csv('galaxies.csv', na_values=('-1000', '-9999'))
gxdata = gxdata.dropna()


# sciserver star query
"""
select 
  top 100000
  objID as objid, probPSF as star, fieldID as field, 
  psfMag_r, psfMagErr_r, cModelMag_r, cModelMagErr_r,
  petroR50_r, petroR50Err_r, petroR90_r, petroR90Err_r, 
  deVRad_r, deVRadErr_r, deVAB_r, deVABErr_r, 
  expRad_r, expRadErr_r, expAB_r, expABErr_r
into mydb.stars
from PhotoObjAll
where probPSF = 1
"""

stdata = pd.read_csv('stars.csv', na_values=('-1000', '-9999'))
stdata = stdata.dropna()

print(data.columns)

data['petro_ratio'] = data.petroR90_r/data.petroR50_r
data['petro_deVRad'] = data.petroR50_r/data.deVRad_r


# featurelist = ['petroR50_r', 'petroR50Err_r', 'petroR90_r', 'petroR90Err_r',
#                'deVRad_r', 'deVRadErr_r', 'deVAB_r', 'deVABErr_r',
#                'expRad_r', 'expRadErr_r', 'expAB_r', 'expABErr_r']
featurelist = ['petro_ratio', 'petro_deVRad', 'deVAB_r']
data = data[np.isfinite(data['petro_deVRad'])]

# test for spiral; if not spiral, then elliptical
targetlist = ['spiral']
data = data[data.uncertain == 0]

X_train, X_test, y_train, y_test = train_test_split(
    data[featurelist],
    data[targetlist].values.ravel())

print(y_train.shape)


# model = LinearSVC(C=10, random_state=12345)
model = RandomForestClassifier(max_depth=3,
                               n_estimators=100,
                               n_jobs=100,
                               random_state=12345)

print(model.fit(X_train, y_train))

print('training data:', model.score(X_train, y_train))
print('test data:', model.score(X_test, y_test))

y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))

print(model.get_params())

print("\n".join([str(x) for x in zip(featurelist,
                                     model.feature_importances_)]))
