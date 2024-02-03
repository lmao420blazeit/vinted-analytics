import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
import matplotlib

from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.metrics import make_scorer, RocCurveDisplay
from scipy.stats import f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols

from skopt import BayesSearchCV
from skopt.space import Real

from imblearn.metrics import specificity_score

uri = 'postgresql://user:4202@localhost:5432/vinted-ai'
engine = create_engine(uri)

sql_query = "SELECT * FROM public.products_catalog LIMIT 15000"
data = pd.read_sql(sql_query, engine)

from sklearn import svm

clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(data["price"].values.reshape(-1, 1))
y_pred_train = clf.predict(data["price"].values.reshape(-1, 1))
data["svm"] = y_pred_train
data = data[data["svm"] == 1]

#data = data[(data["catalog_id"] == 1807) | (data["catalog_id"] == 221)]
data = data[data["catalog_id"] == 1231]

anova_result = f_oneway(*[group['price'] for name, group in data.groupby('brand_title')])
print(anova_result)

# Fit a linear model using statsmodels
print(data.isna().sum())
model = ols('price ~ C(brand_title) + C(status) + C(size_title)', data=data).fit()

# Perform ANOVA
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)


from statsmodels.multivariate.manova import MANOVA

from statsmodels.stats.multicomp import pairwise_tukeyhsd

#manova = MANOVA(endog=data[["brand_title", "status", "size_title", "catalog_id"]], 
#                exog=data["price"].astype(float))
#print(manova.mv_test())

#topbrands = data['brand_title'].value_counts().nlargest(30)
#filter = topbrands.index.tolist() 
#data = data[data['brand_title'].isin(filter)]

tukey_result = pairwise_tukeyhsd(endog=data['price'], groups=data['size_title'], alpha=0.05)
print(tukey_result.summary())

tukey_result.plot_simultaneous()
matplotlib.pyplot.show()

# Look at the confidence intervals and determine which pairs of groups have overlapping intervals. 
# If the intervals overlap, the groups are not significantly different. If the intervals do not overlap, the groups are significantly different.