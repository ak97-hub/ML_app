from scipy.sparse.construct import random
import streamlit as st
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.tree import ExtraTreeRegressor, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from category_encoders.ordinal import OrdinalEncoder
from category_encoders.target_encoder import TargetEncoder
from category_encoders.leave_one_out import LeaveOneOutEncoder
from category_encoders.james_stein import JamesSteinEncoder
from category_encoders.m_estimate import MEstimateEncoder
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
from streamlit.legacy_caching.caching import cache
from streamlit.state.session_state import SessionState

car_path = "/Users/angelokhan/Documents/Projects/streamlit-ML/CarPrice_Assignment.csv"
housing_path = "/Users/angelokhan/Documents/Projects/streamlit-ML/Housing_data.csv"
entire_form = st.form(key="form1")
header = st.container()
dataset = st.container()
dcol1, dcol2 = st.columns(2)
clean = st.container()
encode = st.container()
scalar = st.container()
stnd_col1, stnd_col2 = st.columns(2)
multicollinearity = st.container()
vif_col1, vif_col2 = st.columns(2)
outliers = st.container()
model = st.container()
model_col1, model_col2 = st.columns(2)

@st.cache
def load_data(path):
    return pd.read_csv(path)

@st.cache
def scale_data(data, scale_type):
    columns = data.columns
    if scale_type == "Standardize":
        ss = StandardScaler()
        df = pd.DataFrame(ss.fit_transform(data))
    elif scale_type == "Min-Max Normalization":
        mm = MinMaxScaler()
        df = pd.DataFrame(mm.fit_transform(data))

    i = 0
    for x in columns:
        df = df.rename(columns={i:x})
        i += 1
    return df.copy()

@st.cache
def apply_PCA(data, input):
    pca = PCA(n_components=2)
    X = data.drop("SalePrice", axis=1)
    y = data.loc[:, "SalePrice"]
    if input:
        princpal_comps = pca.fit_transform(X)
        principalDf = pd.DataFrame(data = princpal_comps
             , columns = ['principal component 1', 'principal component 2'])
        principalDf["SalePrice"] = y
        return principalDf.copy()
    else:
        return data.copy()

@st.cache
def multicollinearity_func(data, input, type):
    if input:
        if type == "Housing Dataset":
            all_num_cols = ["LotArea", "MasVnrArea", "BsmtUnfSF", "TotalBsmtSF",
                        "1stFlrSF", "2ndFlrSF","BsmtFullBath", "GarageArea",
                        "WoodDeckSF", "OpenPorchSF", "FullBath","HalfBath",
                        "TotRmsAbvGrd", "Fireplaces"]
        else:
            all_num_cols = ['car_ID', 'symboling', 'wheelbase', 'carlength', 'carwidth',
            'carheight', 'curbweight', 'enginesize', 'boreratio', 'stroke',
            'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg', 'price']
        numeric_cols = [x for x in data.columns if x in all_num_cols]
        num_df = data[numeric_cols]
        vif_info = pd.DataFrame()
        vif_info['VIF'] = [variance_inflation_factor(num_df.values, i) for i in
                          range(num_df.shape[1])]
        vif_info['Column'] = num_df.columns
    return vif_info.copy()

@st.cache
def calc_outliers(data, per_remove):
    data = data.copy()
    outliers_data = outlier_func(data,
     True).sort_values("Mahalanobis_Distance",ascending=False)
    tot_rows = len(outliers_data.index)
    outliers_data = outliers_data.reset_index(drop=True)
    total_per_to_remove = int((per_remove/100)*tot_rows)
    outliers_data = outliers_data.iloc[total_per_to_remove:,:]
    return outliers_data

@st.cache
def outlier_func(data, input):
    data = data.copy()
    if input:
      contamination = 0.18 
      el = EllipticEnvelope(store_precision=True, 
                                       assume_centered=False, 
                                       support_fraction=1, 
                                      contamination=contamination, 
                                       random_state=0)
    
      el.fit(data)
      data["Mahalanobis_Distance"] = el.mahalanobis(data)
      return data

@st.cache
def clean_M(data, num, mahalan):
    temp_df = data.copy()
    temp_df["M"] = mahalan
    condition = temp_df[temp_df["M"] > num].index
    temp_df = temp_df.drop(condition)
    temp_df.drop("M", axis=1)
    return temp_df.copy()


def reg_setup(regression, X, y, params):
    if regression == "Lasso":
        reg = Lasso(alpha = params["alpha"])
        return apply_reg(reg, X, y)
    elif regression == "Linear Regression":
        reg = LinearRegression()
        return apply_reg(reg, X, y)
    elif regression == "Extra Tree Regressor":
        reg = ExtraTreeRegressor(max_features=params["max_feat_E"])
        return apply_reg(reg, X, y)
    elif regression == "Decision Tree Regressor":
        reg = DecisionTreeRegressor(max_features=params["max_feat_D"])
        return apply_reg(reg, X, y)
    elif regression == "Random Forest Regressor":
        reg = RandomForestRegressor(n_estimators=params["n_est_R"])
        return apply_reg(reg, X, y)
    elif regression == "Gradient Boosting Regressor":
        reg = GradientBoostingRegressor(n_estimators=params["n_est_G"])
        return apply_reg(reg, X, y)

@st.cache
def apply_reg(reg, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
    random_state= 1234)
    reg.fit(X_train, y_train)
    try:
        feature_importance = reg.feature_importances_
    except:
        pass
    try:
        feature_importance = reg.coef_
    except:
        pass

    X_cols = X.columns
    y_predict = reg.predict(X_test)
    mse = mean_squared_error(y_test, y_predict)

    rmse = np.sqrt(mse)
    r2 = reg.score(X_test, y_test)
    scores = cross_val_score(reg,
                             X_test,
                             y_test,
                             scoring = "neg_mean_squared_error",
                             cv = 10)

    rmse_scores = np.sqrt(-scores)
    results_dict = {"RMSE" : "{:,.5f}".format(rmse),
                    "MSE": "{:,.5f}".format(mse),
                    "Mean RMSE Scores": "{:,.5f}".format(rmse_scores.mean()),
                    "Std RMSE Scores": "{:,.5f}".format(rmse_scores.std()),
                    "R2": "{:,.5f}".format(r2)
                    }

    results = pd.DataFrame.from_dict(results_dict, orient='index',
    columns=["Scoring"])
    ft_import_df = feature_import(ft_imports=feature_importance, ft_columns=X_cols)
    return results, ft_import_df

def feature_import(ft_imports, ft_columns):
    ft_imports_df = pd.DataFrame()
    ft_imports_df["Features"] = ft_columns
    ft_imports_df["Measure"] = ft_imports
    return ft_imports_df

@st.cache
def display_scores(scores):
    scores = ["{:,.2f}".format(x) for x in scores]
    print("Scores: ", scores)
    print("Mean RMSE scores: {:,.2f}".format(scores.mean()))
    print("Std RMSE Scores: {:,.2f}".format(scores.std()))

@st.cache
def encode_data(data, encode_selection, type):
    df = data.copy()
    if type == "Housing Dataset":
        cols = ['MSZoning', 'LandContour', 'LotConfig', 'Neighborhood',
                    'BldgType', 'YearBuilt', 'YearRemodAdd',
                    'RoofStyle', 'RoofMatl', 'Exterior1st', 
                    'Exterior2nd', 'Foundation', 'Heating']
        target = "SalePrice"
    else:
        cols = ['CarName','fueltype','aspiration','doornumber','carbody',
        'drivewheel','enginelocation','enginetype','cylindernumber','fuelsystem']
        target = "price"
    
    cat_cols = [x for x in data.columns if x in cols]
    if encode_selection == "JamesStein":
        js = JamesSteinEncoder()
        data = js.fit_transform(data[cat_cols], data[target])
    elif encode_selection == "LeaveOne":
        l1 = LeaveOneOutEncoder()
        data = l1.fit_transform(data[cat_cols], data[target])
    elif encode_selection == "M-estimate":
        m_e = MEstimateEncoder()
        data = m_e.fit_transform(data[cat_cols], data[target])
    elif encode_selection == "Target encoder":
        te = TargetEncoder()
        data = te.fit_transform(data[cat_cols], data[target])

    for x in data.columns:
        df[x] = data[x]

    return df.copy()

with entire_form:

    with st.sidebar:
        #choose dataset
        dataset_name = st.selectbox("Select Dataset", 
        ("Housing Dataset", "Car Dataset"))

        #remove features selectbox
        if dataset_name == "Housing Dataset":
            features = ['MSZoning', 'LotArea', 'LandContour', 'LotConfig', 'Neighborhood',
            'BldgType', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'RoofStyle',
            'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrArea', 'ExterQual',
            'Foundation', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'CentralAir',
            '1stFlrSF', '2ndFlrSF', 'BsmtFullBath', 'FullBath', 'HalfBath',
            'KitchenQual', 'TotRmsAbvGrd', 'Fireplaces', 'GarageArea', 'PavedDrive',
            'WoodDeckSF', 'OpenPorchSF']

        else:
            features = [
            'car_ID', 'symboling', 'CarName', 'fueltype', 'aspiration', 
            'doornumber', 'carbody', 'drivewheel', 'enginelocation', 'wheelbase', 
            'carlength', 'carwidth', 'carheight', 'curbweight', 'enginetype', 
            'cylindernumber', 'enginesize', 'fuelsystem', 'boreratio', 'stroke', 
            'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg', 
            'price']

        drop_cols = st.multiselect("Select Features to Remove", features)

        #encoding selectbox
        encoding_type = ["JamesStein", "LeaveOne", "M-estimate", "Target encoder"]
        selection_encode = st.selectbox("Choose Encoder for categorical data", encoding_type)
        
        #scalling selectbox
        scaling_type = ["Standardize", "Min-Max Normalization"]
        selection = st.selectbox("Choose Scaling", scaling_type)

        #outlier removal bar
        remove_perc = st.slider("Remove % Outliers", 0, 10)

        #mode type
        regression_type = ["Linear Regression", "Random Forest Regressor", "Lasso","Extra Tree Regressor", 
        "Decision Tree Regressor", "Gradient Boosting Regressor"]
        reg_selection = st.selectbox("Choose Regression Model", regression_type)
        params = dict()
        if reg_selection == "Lasso":
            alpha = st.slider("Manipulate alpha for Lasso Regression", 0.00, 1.00)
            params["alpha"] = alpha
        elif reg_selection == "Linear Regression":
            pass
        elif reg_selection == "Extra Tree Regressor":
            max_feat = st.slider("Manipulate max features of ETR", 1, 15)
            params["max_feat_E"] = max_feat
        elif reg_selection == "Decision Tree Regressor":
            max_feat = st.slider("Manipulate max features of Decision Tree", 1, 10)
            params["max_feat_D"] = max_feat
        elif reg_selection == "Random Forest Regressor":
            n_est = st.slider("Manipulate n estimators of RFR", 1, 100)
            params["n_est_R"] = n_est
        elif reg_selection == "Gradient Boosting Regressor":
            n_est = st.slider("Manipulate n estimators of GBR", 1, 100)
            params["n_est_G"] = n_est

        #submit button
        st.form_submit_button("Submit changes to see results")

    with header:
        st.title("Welcome to my awesome data science project!")
        st.title("")
        st.subheader("Introduction")
        st.text(
        """
        In this interactive application use the available datasets to create a 
        housing, or car, price prediction regression model. On the left side you'll 
        see you are able to drop columns, choose how you want to encode the data, 
        how you'd like to scale the data, and how much of the outlier data you want 
        removed. Finally choose your regression model and press the submit button 
        to see the performance! After, you can fine tune you models parameters, such 
        as max features for the Extra TreeRegressor, or even go back and make
        more changes! Take time to explore the effects of everything from encoding
        to outlier removal on the performance of the model. 
        Enjoy!
        """
)

    with dataset:
        dataset.header("Data")
        dataset.text(
        '''
        In this section you can change the encoder, and submit the change to see how 
        each categorical column is affected by each specific encoder. Notice that in 
        this section we only change categorical columns, that is simply because a 
        machine learning algorithm cannot interpret what these categorical columns 
        as strings, so we must convert these categorical columnns into numeric columns. 
        However, given that these categorical columns now have numeric values, they 
        could easily influence how the algorithm interprets connections. Therefore we 
        must becareful how we encode these categorical columns. For example, a column 
        that contains values bad, good, great can be denoted as an ordinal column 
        because we can assign numeric values, such that bad = 0, good = 1, 
        and great = 2. This works because we know that bad < good < great, and therefore
        0 < 1 < 2. Well what do we do when we havea column such as MSZoning  below? 
        We can't assign ordinal values because we do not know whether the categories RL,
        RM, RH, etc have some arrangement such that one is superior to the other. But 
        also we cannot just assign random variables to to each category, because what
        if we used a random number generator that produced the results RL = 100, RM = 5,
        & RH = 3, then we run into the same problem. This is where encoding comes in.
        Each encoding available in the selectbox is just one of few, each using different
        techniques to overcome the problem of assigning numeric values without having an
        affect on how the model would otherwise interpret them and their relationships 
        with other variables.  
        
        Change the encoder and see the effects!

        More information:
        https://www.kaggle.com/subinium/11-categorical-encoders-and-benchmark
        '''
        
        )
        with dcol1:
            dcol1.header("Original Dataset")
            if dataset_name == "Housing Dataset":
                data = load_data(housing_path)
                size = "Size: 1435 x 36"
            else:
                data = load_data(car_path)
                size = "Size: 205 x 26"
            data = data.drop(drop_cols, axis= 1)
            dcol1.write(data.head(6))
            dcol1.text(size)


        with dcol2:
            dcol2.header("Encoded Dataset")
            data = encode_data(data=data, encode_selection=selection_encode, type = dataset_name)
            dcol2.write(data.head(6))
            dcol2.text(size)


    with scalar:
        st.header("")
        st.header("Scaling the Data")
        scalar.text(
            """
            Now that our data is all encoded into numeric values, we need to move on to the next 
            step. That is scaling our data. As you can see in the bottom left graph, this is a 
            plot of the kernal density estimate(kde). This graph tells us the distribution of 
            observations in a dataset across all of its possible values. For example, the 
            minimum value of the column LotArea is 1300 and the max value is 215,245. Clearly a 
            huge range, yet the minimum value of OverallQual is 1 and the max is 5. We can 
            directly see this on the left graph below, when all variables are plotted on the
            same graph. How could we ever compare these variables when 215,245 is clearly way 
            larger than 5? The algorithm will definitely be biased towards using LotArea than 
            OverallQual when making predictions. This is where scaling the data comes in, again 
            like encoding there are many techniques, all of which also affect the outcome of 
            the model. The two main ones are available in the selection box, standardization and 
            minimum-maximum normalization. As you can probably already see, on the right hand graph 
            by scaling all of our variables we're able to represent them in a much tighter space, 
            leaving much less room for biases in the model!

            More information:
            https://towardsdatascience.com/scale-standardize-or-normalize-with-scikit-learn-6ccc7d176a02
            """)
        old_data = data.copy()
        data = scale_data(data, selection)
        st.write(data.head(6))
        #col 1
        fig1, ax1 = plt.subplots(figsize= (5, 6))
        for x in range(1, len(data.columns)):
            ax1 = sns.kdeplot(old_data.iloc[:, x])
        ax1.set(xlabel="Features & Target")
        plt.title("Before Scale")
        stnd_col1.pyplot(fig1)
        #col 2
        fig2, ax2 = plt.subplots(figsize= (5, 6))
        for x in range(1, len(data.columns)):
            ax2 = sns.kdeplot(data.iloc[:,x])
        ax2.set(xlabel="Features & Target")
        plt.title("After Scale")
        stnd_col2.pyplot(fig2)

    with multicollinearity:
        st.header("")
        st.header("Multicollinearity")
        st.text(
            """
            In this section we discuss multicollinearity. Multicollinearity is when there is 
            significant intercorrelation between two or more variables. Why is this bad? 
            When a variable is highly correaleted to one or many other variables this can make 
            it hard to distinguish the importance of individual variables, therefore giving the 
            model a hard time using the correct predictors. Getting rid of highly correlated values
            isn't necessarily a bad thing either because it's highly correlated counterparts are enough 
            to account for its absence. Below, we have calculated the Variance Inflation Factor, this 
            is a neat calculation used for determining the correlation of variables to other variables 
            within the same dataset. The values begin and 1 and can go however high, the larger the 
            value the higher the multicollinearity of that variable. There is no exact cut off VIF 
            value at which you should remove a feature, but some recommend all features with a VIF 
            value above 5. Notice also that there are much less columns in the table than in the 
            actual table. That is because for VIF we strictly use the numeric columns, not the 
            categorical columns. Try removing the column with the highest VIF, and click submit 
            to see how your model is affected!

            More information:
            https://towardsdatascience.com/how-to-remove-multicollinearity-using-python-4da8d9d8abb2 

            """)
        VIF_df = multicollinearity_func(data, input=True, type=dataset_name)
        #col 1
        vif_col1.write(VIF_df.sort_values("VIF",ascending=False))
        #col 2
        fig, ax = plt.subplots(figsize= (5, 5))
        ax = sns.barplot(data=VIF_df.sort_values("VIF", ascending=False),
        x="VIF", y = "Column")
        plt.title("Measure of Multicollinearity(VIF)")
        vif_col2.pyplot(fig)

    with outliers:
        st.header("")
        st.header("Outliers")
        outliers.text(
            """
            In this section we look at outlier data measured by the Mahalanobis Distance. 
            Given our data is greater than two dimensions, we must use a multivariate method of 
            measuring outliers. To quote from the article linked directly below: \"The Mahalanobis 
            distance can be effectively thought of a way to measure the distance between a 
            point and a distribution.\". Mahalanobis Distance is a complex yet very interesting 
            technique, and a great tool to have handy!\n
            For further details please read the awesome article below!

            More information:
            https://medium.com/analytics-vidhya/anomaly-detection-in-python-part-1-basics-code-and-standard-algorithms-37d022cdbcff
            """)
        outliers_data_col, outliers_chart = st.columns(2)
        outliers_data = calc_outliers(data, remove_perc)

        #col 1 
        outliers_data_col.write(outliers_data.iloc[:25, ::-1])
        outliers_data_col.text("The total # of rows removed are: {}".format(outliers_data.index[0]))

        #col 2
        fig, ax = plt.subplots(figsize= (5, 5))
        ax.hist(outliers_data["Mahalanobis_Distance"], color="c")
        plt.xlabel("Mahalanobis Distance")
        plt.ylabel("Count")
        plt.title("Mahalanobis Distance")
        outliers_chart.pyplot(fig)
        
        
    with model:
        st.header("")
        st.header("Model")
        st.text(
            """
            You're at the finish line! Below is some important information on how your model 
            performed. As you see below we have a chart with all the columns within the data 
            and their respective feature importance values. The feature importance values are 
            a measure of how much a feauture contributed to the prediction. However, beware 
            that does not mean the other features with low feature importance values do not 
            contribute. Remember these models are produced with a combination of variables, 
            a variables individual contribution does not exist in a vaccum, so while a 
            variable may seemingly have a low feature importance it could be contributing to
            a higher feature importance within another variable!

            To clarify a few definitions:
            RMSE- Root Mean Square Error
            MSE- Mean Square Error
            Mean RMSE- Mean of the Root Mean Square Error
            Std RMSE- Standard Deviation of the Root Mean Square Error
            R2- R-squared

            We first begin by first seeing how our model predicts on a set of data we set 
            aside called the test data. We make the model predict the result and compare it
            that result to the actual value. That is if we trained our model to predict 
            housing prices, our model would be given the data from all other columns and 
            asked to predict the  price of the homes based on the set of features it was 
            trained on. Since we know the  actual prices and have the predicted prices, we
            run some analysis on it. One such  analysis is the MSE. MSE is a measure of the
            average, squared difference between the observed(predicted values) and the 
            actual values. What the RMSE is the square root of  the MSE, and it essentially
            tells us is the standard deviation of the prediction errors. That is, how well 
            concentrated is the data around the best fit line. Therefore if an  RMSE of the 
            model is high, that doesn't mean it's doing so well. The mean and standard 
            deviation of the RMSE, just help us further understand the picture RMSE paints. 
            And lastly R-squared tells us how well our model fits the data. However, also 
            beware that a low R-squared does not necessarily mean a bad thing, nor does a 
            high R-squared always  mean the model is perfect.

            Enough reading time to play around with the parameters and see how the model
            performs!!!
            
            More information:
            https://medium.com/@amanbamrah/how-to-evaluate-the-accuracy-of-regression-results-b38e5512afd3
            """)
        if dataset_name == "Housing Dataset":
            target = "SalePrice"
        else:
            target = "price"
        drop_X = ["Mahalanobis_Distance", target]
        X = outliers_data.drop(drop_X, axis=1)
        y = outliers_data[target]
        coef = ["Lasso", "Linear Regression"]
        if reg_selection in coef: 
            measure = "Coefficient"
        else:
            measure = "Mean_Decrease_in_Impurity"

        results, ft_imp_df = reg_setup(regression=reg_selection, X=X, y=y, params=params)
        ft_imp_df = ft_imp_df.rename({"Measure": measure}, axis=1)

        with model_col1:
            model_col1.subheader("Important Features")
            model_col1.write(ft_imp_df.sort_values(measure, ascending = False))

        with model_col2:
            model_col2.subheader("Model Scores")
            model_col2.write(results)

        
        fig, ax = plt.subplots(figsize = (10, 7))
        ax = sns.barplot(data=ft_imp_df, 
        x=measure, y = "Features")
        measure = "{}".format(measure.replace("_", " "))
        plt.title("Feature importance using {}".format(measure))
        ax.set_xlabel = measure
        ax.set_ylabel = "Features"
        model.pyplot(fig)