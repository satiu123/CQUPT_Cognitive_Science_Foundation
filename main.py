import CQUPT_Cognitive_Science_Foundation.preprocess as pp
def preprocess(
        data,
        channel_index = ['FC5', 'FC3', 'FC1', 'FC2', 'FC4','FC6','C5','C3','C1','Cz','C2','C4','C6','CP5','CP3','CP1','CPz','CP2','CP4','CP6'],
        band = [4,40],
        time_interval =[1000,3500],
        isFilter = True,
        ):
    #选取通道
    data_selec=pp.prep_selectChannels(data,channel_index)
    #滤波
    if isFilter:
        data_filt=pp.prep_bandFilter(data_selec,band,data['fs'])
        #分段
        data_segm=pp.prep_segmentation(data_filt,time_interval,data['fs'])
    else:
        data_segm=pp.prep_segmentation(data_selec,time_interval,data['fs'])
    return data_segm
import numpy as np
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

def split_SMT(SMT,index):
    new_SMT={}
    new_SMT['x']=SMT['x'][index]
    new_SMT['y_dec']=SMT['y_dec'][:,index]
    new_SMT['y_logic']=SMT['y_logic'][:,index]
    new_SMT['class']=SMT['class']

    return new_SMT
def conbine_SMT(all_SMT):
    new_SMT={}
    new_SMT['x']=np.concatenate([SMT['x'] for SMT in all_SMT],axis=0)
    new_SMT['y_dec']=np.concatenate([SMT['y_dec'] for SMT in all_SMT],axis=1)
    new_SMT['y_logic']=np.concatenate([SMT['y_logic'] for SMT in all_SMT],axis=1)
    new_SMT['class']=all_SMT[0]['class']

    return new_SMT
def eval_cross_validation(SMT, n_splits:int, csp_n_components:int, clf):
    """
    执行交叉验证并返回平均准确率。

    Args:
        SMT (dict): 包含 x, y_dec, y_logic, class 的字典。
        n_splits (int): KFold 的折数。
        csp_n_components (int): CSP 的 component 数量。
        clf (object): 分类器对象。

    Returns:
        float: 平均准确率。
    """
    import CQUPT_Cognitive_Science_Foundation.util as util
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    X=SMT['x']
    y=SMT['y_dec']
    for train_index, test_index in kf.split(X):
        X_train, X_test = split_SMT(SMT,train_index), split_SMT(SMT,test_index)
        y_train, y_test = X_train['y_dec'].flatten(), X_test['y_dec'].flatten()

        # 1. CSP 特征提取
        X_train_csp,train_CSP_W,train_CSP_D=util.func_csp(X_train,nPatterns=csp_n_components,policy='normal')
        #将测试集投影到CSP_W上
        X_test_projection=util.func_projection(X_test,train_CSP_W)
        # 2. 计算对数方差特征
        X_train_features = util.calculate_log_variance(X_train_csp['x'])
        X_test_features = util.calculate_log_variance(X_test_projection['x'])

        # 3. 训练分类器
        clf.fit(X_train_features, y_train)

        # 4. 预测
        y_pred = clf.predict(X_test_features)

        # 5. 计算准确率
        score = accuracy_score(y_test, y_pred)
        scores.append(score)
    return np.mean(scores)
#跨被试
def eval_cross_subject(all_SMT, csp_n_components:int, clf):
    """
    执行交叉验证并返回平均准确率。

    Args:
        all_SMT (list): 包含 SMT 的列表。
        csp_n_components (int): CSP 的 component 数量。
        clf (object): 分类器对象。

    Returns:
        float: 平均准确率。
    """
    import CQUPT_Cognitive_Science_Foundation.util as util
    scores = []
    for subject in range(10):
        train_SMT=conbine_SMT([all_SMT[i] for i in range(10) if i!=subject])
        test_SMT=all_SMT[subject]
        train_y=train_SMT['y_dec']
        test_y=test_SMT['y_dec']
        # 1. CSP 特征提取
        X_train_csp,train_CSP_W,train_CSP_D=util.func_csp(train_SMT,nPatterns=csp_n_components,policy='normal')
        #将测试集投影到CSP_W上
        X_test_projection=util.func_projection(test_SMT,train_CSP_W)
        # 2. 计算对数方差特征
        X_train_features = util.calculate_log_variance(X_train_csp['x'])
        X_test_features = util.calculate_log_variance(X_test_projection['x'])

        # 3. 训练分类器
        clf.fit(X_train_features, train_y.flatten())

        # 4. 预测
        y_pred = clf.predict(X_test_features)

        # 5. 计算准确率
        score = accuracy_score(test_y.flatten(), y_pred)
        scores.append(score)
        print(f"subject {subject+1} score: {score:.4f}")

    return np.mean(scores)

from sklearn.svm import SVC


def train_and_test(session:int,cross_subject:bool,cross_validation:bool,clf,csp_n_components=3):
    """
    Args:
        session (int): 1 或 2。
        cross_subject (bool): 是否进行跨被试的训练和测试。
        cross_validation (bool): 是否进行交叉验证。
        clf (str): 分类器名称，'LDA' 或 'SVM'。
    Returns:
        score (float): 准确率。
    """
    if session not in [1,2]:
        raise ValueError('session must be 1 or 2')
    if cross_subject and cross_validation:
        raise ValueError('cross_subject and cross_validation cannot be True at the same time')
    if clf not in ['LDA','SVM']:
        raise ValueError('clf must be LDA or SVM')
    
    filepath=[f'EEG/session0{session}/sess0{session}_subj{f"0{i}" if i<10 else str(i)}_EEG_MI.mat' for i in range(1,11)]
    field=['x','t','fs','y_dec','y_logic','y_class','class', 'chan']
    n_splits = 10
    all_SMT=[]
    scores=[]
    if cross_subject:
        for i in range(0,10):
            #读取数据
            train, test = pp.load_data(filepath[i])
            train_data=pp.prep_data(train,field,100)
            test_data=pp.prep_data(test,field,100)
            #预处理
            train_data_segm=preprocess(train_data,isFilter=True)
            test_data_segm=preprocess(test_data,isFilter=True)
            all_SMT.append(conbine_SMT([train_data_segm,test_data_segm]))
        clf =  LinearDiscriminantAnalysis() if clf=='LDA' else SVC(kernel='linear')
        score = eval_cross_subject(all_SMT, csp_n_components=csp_n_components, clf=clf)
        scores.append(score)
        return np.mean(scores),np.std(scores)
    elif cross_validation:
        for i in range(0,10):
            #读取数据
            train, test = pp.load_data(filepath[i])
            train_data=pp.prep_data(train,field,100)
            test_data=pp.prep_data(test,field,100)
            #预处理
            train_data_segm=preprocess(train_data,isFilter=True)
            test_data_segm=preprocess(test_data,isFilter=True)
            data=conbine_SMT([train_data_segm,test_data_segm])
            
            clf =  LinearDiscriminantAnalysis() if clf=='LDA' else SVC(kernel='linear')
            score = eval_cross_validation(data, n_splits, csp_n_components=csp_n_components, clf=clf)
            scores.append(score)
            print(f"subject {i+1} score: {score:.4f}")
        return np.mean(scores),np.std(scores)
    else:
        raise ValueError('cross_subject and cross_validation cannot be False at the same time')
"""
流程：
1. 读取数据
2. 预处理
    1）选取通道
    2）滤波
    3）分段
3. 特征提取
    1）CSP
    2）投影
    3）对数方差
4. 分类
5. 评估
    1）交叉验证
    2）跨被试
"""
train_and_test(1,cross_subject=False,cross_validation=True,clf='LDA',csp_n_components=3)
#测试csp_n_components
# scores=[]
# stds=[]
# for i in range(2,11,2):
#     print(f'csp_n_components={i}:')
#     score,std=train_and_test(2,cross_subject=False,cross_validation=True,clf='LDA',csp_n_components=i)
#     scores.append(score)
#     stds.append(std)
# print(scores)
# print(stds)