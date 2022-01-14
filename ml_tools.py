# Imports
import pandas as pd
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
os.environ['NUMEXPR_MAX_THREADS'] = '12'

import logging
logging.basicConfig(level=logging.INFO)

from tqdm import tqdm_notebook
tqdm_notebook().pandas()

from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, make_scorer
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, RandomForestRegressor, RandomForestClassifier, ExtraTreesClassifier,\
                             ExtraTreesRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import Lasso, LinearRegression, Ridge

import joblib

def save_model(model, filename):
    joblib.dump(model, filename)

def load_model(filename):
    return joblib.load(filename)

RND_STATE = 17

def read_data(RAW_FILENAME):
    df = pd.read_excel(RAW_FILENAME)
    return df

# Functions
def get_describe(data):
    res = data.describe(include='all').T.copy(deep=True)
    med = data.median()
    nul = data.isnull().sum()
    nul_percent = (nul * 100 / len(data)).astype(int)
    typ = data.dtypes
    res.insert(0, 'dtypes', typ)
    res.insert(1, 'nulls', nul)
    res.insert(1, 'nulls_percent', nul_percent)
    res.insert(5, 'median', med)
    
    return res

def drop_unnec_empty_const_cols(df, df_descr):
    cols_to_drop = ['Проект', 'Ссылка на планировку', 'Тип', 'Номер квартиры', 'Статус', 'Паркинг', 'Вид из окна', 'Тип планировки 2',
    'Дополнительный статус',
     'Дата старта продаж',
     'ProfitbaseId',
     'ProfitbaseHomeId',
     'ComplexId', 'Ценазакв,м,']
    
    cols_toilet = [col for col in list(df) if 'тип санузла' in col.lower()]
    cols_to_drop.extend(cols_toilet)
#     cols_to_drop.extend(df_descr[df_descr['nulls_percent']==100].index)
    df.drop(cols_to_drop, axis=1, inplace=True)
    return df

def drop_houses_without_data(df, houses_to_drop):
    df = df.loc[~df['Номер дома'].isin(houses_to_drop)].copy(deep=True)
    return df

def restore_studios(df):
    if df['Комнатность'].dtype != 'object':
        idx_studio = df['Комнатность'] == 0
        df.loc[idx_studio, 'Комнатность'] = 1
    else: 
        idx_studio = df['Комнатность'].str.contains('с')
        df['Комнатность'] = df['Комнатность'].map(lambda x: int(x[0]))
        
    df.loc[df['Студия'].isnull() & idx_studio, 'Студия'] = 'да'
    df.loc[df['Студия'].isnull(), 'Студия'] = 'нет'
    df['Студия'] = df['Студия'].replace({'да': 1, 'нет': 0})
    return df

def restore_toilets(df):
    cols = [col for col in list(df) if ('Размер' in col) & ('узла' in col)]
    df.loc[:, cols] = df[cols].fillna(0).astype(float)
    df.loc[(df['Размер сан узла №1'].notnull()), 'Количество сан узлов'] = 1
    df.loc[(df['Размер сан узла №2'].notnull()), 'Количество сан узлов'] = 2
    df.loc[(df['Размер сан узла №3'].notnull()), 'Количество сан узлов'] = 3
    df.loc[(df['Размер сан узла №4'].notnull()), 'Количество сан узлов'] = 4
    
#     df.loc[(df['Количество сан узлов'] == 1), 'Тип санузла 2'] = 'Санузел отсутствует'
    df.loc[(df['Количество сан узлов'] == 1), 'Размер сан узла №2'] = 0
    
#     df.loc[df['Тип санузла 1'].isnull(), 'Тип санузла 1'] = 'нет_данных'
#     df.loc[df['Тип санузла 2'].isnull(), 'Тип санузла 2'] = 'нет_данных'
    return df

def restore_wardrobe(df):
    df['Гардеробная'].fillna(0, inplace=True)
    if 'Площадь гардеробных' in list(df):
        df['Площадь гардеробных'].fillna(0, inplace=True)
    else:
        cols = [col for col in list(df) if 'Площадь гардеробной' in col]
        df['Площадь гардеробных'] = df[cols].fillna(0).sum(axis=1)
        df.drop(cols, axis=1, inplace=True)
    return df

def restore_balcony(df):
    df['Линия'] = df['Линия'].fillna(0)
    df['Количество балконов'] = df['Количество балконов'].fillna(0)
    df['Размер балкона №1'] = df['Размер балкона №1'].fillna(0).astype(str).str.replace(',','.').astype(float)
    df['Размер балкона №2'] = df['Размер балкона №2'].fillna(0).astype(str).str.replace(',','.').astype(float)
    df['Размер балкона №3'] = df['Размер балкона №3'].fillna(0).astype(str).str.replace(',','.').astype(float)
    df['Размер балкона №4'] = df['Размер балкона №4'].fillna(0).astype(str).str.replace(',','.').astype(float)
    df['Размер балкона №5'] = df['Размер балкона №5'].fillna(0).astype(str).str.replace(',','.').astype(float)
    
    df['Терраса №1'] = df['Терраса №1'].fillna(0).astype(str).str.replace(',','.').astype(float)
    df['Терраса №2'] = df['Терраса №2'].fillna(0).astype(str).str.replace(',','.').astype(float)
    df['Терраса №3'] = df['Терраса №3'].fillna(0).astype(str).str.replace(',','.').astype(float)
    df['Терраса №4'] = df['Терраса №4'].fillna(0).astype(str).str.replace(',','.').astype(float)
    df['Терраса №5'] = df['Терраса №5'].fillna(0).astype(str).str.replace(',','.').astype(float)
    return df

def restore_rooms_square(df):
    if df['Общая площадь, кв,м,'].dtype == 'object':
        df['Общая площадь, кв,м,'] = df['Общая площадь, кв,м,'].str.replace(',','.').astype(float)
    
    if df['Размер коридора'].dtype == 'object':
        df['Размер коридора'] = df['Размер коридора'].str.replace(',','.').astype(float)
    
    if df['Жилая площадь за кв,м,'].dtype == 'object':
        df['Жилая площадь за кв,м,'] = df['Жилая площадь за кв,м,'].str.replace(',','.').astype(float)
        
    if df['Кухня'].dtype == 'object':
        df['Кухня'] = df['Кухня'].fillna(0).astype(str).str.replace(',','.').astype(float)
    else:
        df['Кухня'] = df['Кухня'].fillna(0).astype(float)
    
    cols = [col for col in list(df) if ('Размер' in col) & ('комнаты' in col)]
    for col in cols: 
        df[col] = df[col].fillna(0).astype(str).str.replace(',','.').astype(float)
    
    if 'Размер первой комнаты' in list(df):
        df.drop('Размер первой комнаты', axis=1, inplace=True)
    
#     if 'Размер первой комнаты' in list(df):
#         df.loc[df['Размер первой комнаты'].isnull(), 'Размер первой комнаты'] = \
#         df.loc[df['Размер первой комнаты'].isnull(), 'Жилая площадь за кв,м,']
        
#         cols = [col for col in list(df) if ('Размер' in col) & ('комнаты' in col)]
#         for col in cols: 
#             df[col] = df[col].fillna(0).astype(str).str.replace(',','.').astype(float)
#     else:
#         cols = [col for col in list(df) if ('Размер' in col) & ('комнаты' in col)]
#         for col in cols: 
#             df[col] = df[col].fillna(0).astype(str).str.replace(',','.').astype(float)
#         df['Размер первой комнаты'] = df['Жилая площадь за кв,м,'] - df[cols].sum(axis=1)    
    
    return df

def restore_selling_date(df):
#     df['Дата продажи'].fillna(df['Дата продажи'].max(), inplace=True)
#     df['Месяц продажи'] = pd.to_datetime(df['Дата продажи']).dt.month
#     df['Год продажи'] = pd.to_datetime(df['Дата продажи']).dt.year
    df.drop('Дата продажи', axis=1, inplace=True)
    return df

def get_plan(x):
    if type(x) is str:
        if len(x) <= 3:
            if len(x) == 2:
                return x[1], '0'
            else:
                return x[1], x[2]
        else:
            tmp = x.split('-')
            return tmp[-2], tmp[-1].split('(')[0]
    
    elif math.isnan(x):
        return 'None', 'None'
    
def get_plans_cols(df):
#     df['планировка_1'], df['планировка_2'] = zip(*df['Тип планировки'].apply(get_plan))
    df.drop('Тип планировки', axis=1, inplace=True)
    return df

def get_one_hot_encoding(df, df_descr):
    to_dum = df_descr.loc[df_descr['dtypes'] == object].index
    dummies = pd.get_dummies(df[to_dum])
    df = pd.concat([df.drop(to_dum, axis=1), dummies], axis=1)
    return df

def filter_iqr(df, TARGET_COL, min_q=0.25, max_q=0.75):
#     sns.boxplot(df[TARGET_COL]);
#     plt.show()

    Q1 = df[TARGET_COL].quantile(min_q)
    Q3 = df[TARGET_COL].quantile(max_q)
    IQR = Q3 - Q1

    idx = (df[TARGET_COL] < (Q1 - 1.5 * IQR)) | (df[TARGET_COL] > (Q3 + 1.5 * IQR))

#     sns.boxplot(df.loc[~idx, TARGET_COL]);
#     plt.show()
    
    return df.loc[~idx].copy(deep=True)

def calc_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))#.astype(float)

def calc_mape(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100, 2)

def get_split(df, TARGET_COL, SPLIT=0.2):
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=SPLIT, 
                                                        shuffle=True, 
#                                                         stratify=X['Номер дома'],
                                                        random_state=2019)
    return X_train, X_test, y_train, y_test

def calc_cross_val_score(model, X_train, y_train, CV_NUM=5):
    model_name = model.__class__.__name__
    print(model_name)
    mape_scorer = make_scorer(calc_mape, greater_is_better=False)
    c = cross_val_score(model, X_train, y_train, cv=CV_NUM, n_jobs=-1, scoring='neg_mean_squared_error', verbose=0)
    c1 = cross_val_score(model, X_train, y_train, cv=CV_NUM, n_jobs=-1, scoring=mape_scorer, verbose=0)
    return {'model': model_name, 
            'rmse_train': np.mean(np.sqrt(c * (-1))), 
            'mape_train (%)': np.mean(c1 * (-1))}

def select_cv_best_model(X_train, y_train, CV_NUM):
    clf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=RND_STATE, max_features='log2')
    regr = AdaBoostRegressor(random_state=RND_STATE, n_estimators=100, learning_rate=1)
    tree = ExtraTreesRegressor(n_estimators=100, random_state=RND_STATE, n_jobs=-1, max_features='log2')
    gbr = GradientBoostingRegressor(n_estimators=100, random_state=RND_STATE, max_features='log2')
    lasso = Lasso(random_state=RND_STATE)
    linreg = LinearRegression(n_jobs=-1)
    ridge = Ridge(random_state=RND_STATE)    

    models = [clf, regr, tree, gbr, lasso, linreg, ridge]

    models_dict = {}
    results = []
    for m in tqdm_notebook(models):
        models_dict[m.__class__.__name__] = m
        try:
            results.append(calc_cross_val_score(m, X_train, y_train, CV_NUM))
        except:
            print('Ошибка с моделью {}'.format(m.__class__.__name__))
            continue
            
    results = pd.DataFrame.from_records(results).sort_values(by='rmse_train')

    selected_models = results.head(5)['model'].values

    return models_dict, selected_models, results

def train_final_model(X_train, X_test, y_train, y_test, models_dict, selected_models):
    final_models_dict = []
    final_models = {}

    for f in tqdm_notebook(selected_models):
        print(f)
        reg = models_dict[f]
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        rmse = calc_rmse(y_test, y_pred)
        mape = calc_mape(y_test, y_pred)
        final_models[f] = reg
        final_models_dict.append({'model': f, 'rmse_valid': rmse, 'mape_valid (%)': mape})

    final_model_results = pd.DataFrame.from_records(final_models_dict).sort_values(by='rmse_valid')

    return final_model_results, final_models

def train_model(df, TARGET_COL, SPLIT, NEED_FILTER_IQR, NEED_DROP_PRICE_PER_SQM, PRICE_PER_SQM_COL, CV_NUM):
    if NEED_FILTER_IQR:
        df = filter_iqr(df, TARGET_COL)
    if NEED_DROP_PRICE_PER_SQM:
        df = df.copy(deep=True)
#         df = df.drop(PRICE_PER_SQM_COL, axis=1).copy(deep=True)
        

    X_train, X_test, y_train, y_test = get_split(df, TARGET_COL, SPLIT)
    
#     scaler = RobustScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)
    
#     logging.info(y_train)
    models_dict, selected_models, results = select_cv_best_model(X_train, y_train, CV_NUM)
    final_model_results, final_models = train_final_model(X_train, X_test, y_train,
                                                          y_test, models_dict,
                                                          selected_models)
    return results, final_model_results, final_models

def prepare_data(df, HOUSES_TO_DROP):
    df_descr = get_describe(df)
    df = drop_unnec_empty_const_cols(df, df_descr)
    df = drop_houses_without_data(df, HOUSES_TO_DROP)
    df_descr = get_describe(df)
    df = restore_studios(df)
    df = restore_toilets(df)
    df = restore_wardrobe(df)
    df = restore_balcony(df)
    df = restore_rooms_square(df)
    df = restore_selling_date(df)
#     df.fillna('нет данных', inplace=True)
    df = get_plans_cols(df)
    df_descr = get_describe(df)
#     df = get_one_hot_encoding(df, df_descr)
    return df

def search_models(df, 
                  TARGET_COL, 
                  list_NEED_FILTER_IQR,
                  list_SPLIT, 
                  list_CV_NUM,
                  list_NEED_DROP_PRICE_PER_SQM, 
                  PRICE_PER_SQM_COL):
    search = pd.DataFrame()    
    for NEED_FILTER_IQR in tqdm_notebook(list_NEED_FILTER_IQR):
        for SPLIT in list_SPLIT:
            for CV_NUM in list_CV_NUM:
                for NEED_DROP_PRICE_PER_SQM in list_NEED_DROP_PRICE_PER_SQM:
                    logging.info("Начинаем тренировку модели...")
                    
                    results, final_model_results, final_models = train_model(df, 
                                                                             TARGET_COL, 
                                                                             SPLIT, 
                                                                             NEED_FILTER_IQR, 
                                                                             NEED_DROP_PRICE_PER_SQM, 
                                                                             PRICE_PER_SQM_COL, 
                                                                             CV_NUM
                                                                            )
                    
                    logging.info("Готово!")

                    all_results = pd.concat([results.head().reset_index(drop=True), 
                                             final_model_results.iloc[:, 1:]], axis=1).reset_index(drop=True)
                    all_results = all_results[np.r_[['model'], sorted(list(all_results)[1:])]].copy(deep=True)
                    all_results['NEED_FILTER_IQR'] = NEED_FILTER_IQR
                    all_results['SPLIT'] = SPLIT
                    all_results['CV_NUM'] = CV_NUM
                    all_results['NEED_DROP_PRICE_PER_SQM'] = NEED_DROP_PRICE_PER_SQM

                    search = pd.concat([search, all_results], axis=0, ignore_index=True)
                    
    return search

def get_best_params(search):
    search['train_valid_diff_mape'] = np.abs((search['mape_train (%)'] - search['mape_valid (%)']) \
                                         / search['mape_train (%)'])

    search['train_valid_diff_rmse'] = np.abs((search['rmse_train'] - search['rmse_valid']) \
                                             / search['rmse_train'])
    
    search['hmean_rmse'] = np.abs(2*search['rmse_train']*search['rmse_valid'] / (search['rmse_train'] + search['rmse_valid']))
    search['hmean_mape'] = np.abs(2*search['mape_train (%)']*search['mape_valid (%)'] / (search['mape_train (%)'] + search['mape_valid (%)']))
    search['custom_rating'] = search['hmean_rmse'] * search['hmean_mape']
    search['custom_rating'] = 1 - search['custom_rating'] / search['custom_rating'].max()

    search.sort_values(by=['custom_rating'], ascending=False, inplace=True)
    
    model_name = search.iloc[0]['model']
    NEED_FILTER_IQR = search.iloc[0]['NEED_FILTER_IQR']
    SPLIT = search.iloc[0]['SPLIT']
    CV_NUM = search.iloc[0]['CV_NUM']
    NEED_DROP_PRICE_PER_SQM = search.iloc[0]['NEED_DROP_PRICE_PER_SQM']
    
    return search, model_name, NEED_FILTER_IQR, SPLIT, CV_NUM, NEED_DROP_PRICE_PER_SQM

def preprocess_input(data, PRICE_PER_SQM_COL, NEED_DROP_PRICE_PER_SQM, TARGET_COL):
    if NEED_DROP_PRICE_PER_SQM:
        data = data.copy(deep=True)
#         data = data.drop(PRICE_PER_SQM_COL, axis=1).copy(deep=True)
    data.drop(TARGET_COL, axis=1, inplace=True)   
    
    return data