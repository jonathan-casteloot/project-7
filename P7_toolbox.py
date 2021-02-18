from scipy import stats
from sklearn.preprocessing import RobustScaler
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st


def population_ratio(country_code_list):

    # data preparation : ratio total population - retained countries
    google_2010_total_population = 6.9 * 10**9

    population = pd.read_csv('./dataset/worldbank_gini_population.csv', 
                             usecols=['Series Name', 'Country Code', '2010 [YR2010]'])
                                      
    population = population[population['Series Name']=='Population, total']
    population = population[['Country Code', '2010 [YR2010]']]
    population.columns =['country_code', 'population']
    population = population.set_index('country_code')
    population = population.loc[country_code_list]
    population['population'] = population['population'].astype(int)
    print('data population / total population ',(population.sum()/google_2010_total_population)[0])

def generate_incomes(size, elasticity_coefficient):
    ln_y_parent = st.norm(0,1).rvs(size=size)
    epsilon = st.norm(0,1).rvs(size=size)
    y_child = np.exp(elasticity_coefficient * ln_y_parent + epsilon)
    
    return y_child, np.exp(ln_y_parent)


def quantiles(incomes, nb_quantiles):
    # sort incomes
    incomes_sorted = incomes[incomes.argsort()]
    
    # create quantile list
    quantiles = np.repeat(np.arange(1, nb_quantiles + 1), incomes.shape[0]/nb_quantiles)
    
    # matrix : incomes_sorted + quantiles
    matrix = np.concatenate([incomes_sorted.reshape(-1,1), quantiles.reshape(-1,1)], axis=1)
    matrix = pd.DataFrame(matrix)
    matrix = matrix.set_index(0)
    
    # create an index dataframe with incomes value
    quantiles = pd.DataFrame(incomes)
    quantiles = quantiles.set_index(0)
    
    # create quantile dataframe
    quantiles = quantiles.merge(matrix, left_index=True, right_index=True)
    quantiles = quantiles.reset_index(drop=True)
    quantiles = quantiles.astype(int)

    return quantiles


def compute_quantiles(y_child, y_parent, nb_quantiles):
    # create child quantiles
    c_i_child = quantiles(y_child, nb_quantiles)
    
    # create parent quantiles
    c_i_parent = quantiles(y_parent, nb_quantiles)
    
    # concatenate child and parent quantile
    sample = pd.concat([c_i_child, c_i_parent], axis=1)
    sample.columns = ["c_i_child","c_i_parent"]
    
    return sample


def conditional_distributions(sample):
    # counts for each couple of class child + class parents
    counts = sample.groupby(['c_i_child', 'c_i_parent']).apply(len)

    # create matrix c_i_child c_i_parent
    counts_matrix = counts.unstack(fill_value=0)

    # create numpy array
    counts_matrix = np.array(counts_matrix)
    
    # compute cardinal for each row
    cardinal = np.sum(counts_matrix, axis=0)

    # compute conditionnal probability
    counts_matrix = counts_matrix / cardinal
    
    return counts_matrix


def c_i_parent(elasticity_coefficient):
    # incomes generation
    y_child, y_parent = generate_incomes(size=100000, elasticity_coefficient=elasticity_coefficient)

    # sample creation
    sample = compute_quantiles(y_child, y_parent, nb_quantiles=100)

    # conditionnal distribution creation
    conditionnal_distribution = conditional_distributions(sample)
    
    return conditionnal_distribution


def elasticity_parent_class_dataframe(elasticity_coefficient):
    
    # make sample attribution 1000 individuals per child class
    sample_attribution = c_i_parent(elasticity_coefficient) * 1000
    
    # dataframe creation with conditionnal distribution
    elasticity_dataframe = pd.DataFrame(sample_attribution) 
    
    # country_code column creation
    elasticity_dataframe['elasticity_coefficient'] = elasticity_coefficient  
    
    # create child quantile
    elasticity_dataframe = elasticity_dataframe.reset_index()
    elasticity_dataframe = elasticity_dataframe.rename(columns={'index':'quantile'})
    elasticity_dataframe['quantile'] = elasticity_dataframe['quantile'] + 1
    
    return elasticity_dataframe


def shapiro_boolean(vector):
    '''
    return True if vector is gaussian-like with shapiro/wilk test.
    '''
    
    from scipy import stats
    from sklearn.preprocessing import RobustScaler
       
    critical_threshold = 0.05
    
    # standardization
    X = np.array(vector).reshape(-1,1)
    standardized_X = RobustScaler().fit_transform(X)
    
    # shapiro test
    small_size_standardized_X = np.random.choice(standardized_X.reshape(-1), size=10, replace=False)
    if stats.shapiro(small_size_standardized_X)[1] >= critical_threshold:
        return True
    else:
        return False
    

def regression_coeff_score(X , y):
    # model selection
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                        test_size=0.2,
                                                        random_state=0)

    # model train : AI part
    model = LinearRegression()
    model.fit(X_train,y_train)

    # model score
    print('R² train :', cross_val_score(LinearRegression(),
                                                   X_train, y_train,
                                                   cv=40).mean().round(2))

    print('R² test :', model.score(X_test, y_test).round(2))
    
    # model predict
    residuals = y_test - model.predict(X_test)

    # prediction coordinates
    return [model.coef_, model.intercept_, residuals]
