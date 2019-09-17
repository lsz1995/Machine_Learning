#数据预处理
import numpy as np
import pandas as pd
vg_df =pd.read_csv('./datasets/vgsales.csv',encoding='ISO-8859-1')


#离散值

# 离散值：   数据不是数值型 而是有几个种类   例如：房子朝向  东南  东

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
generes =np.unique(vg_df['Genre'])# 统计这个数值共有多少种类
# print(generes)
# ['Action' 'Adventure' 'Fighting' 'Misc' 'Platform' 'Puzzle' 'Racing'
#  'Role-Playing' 'Shooter' 'Simulation' 'Sports' 'Strategy']
# 共有十个值   可用  LabelEncoder 对其进行编码 转换成数值类型


gle = LabelEncoder()
genre_labels = gle.fit_transform(vg_df['Genre'])
genre_mappings = {index: label for index, label in enumerate(gle.classes_)}
# print(genre_labels)
vg_df['GenreLabel'] = genre_labels
# print(vg_df[['Name', 'Platform', 'Year', 'Genre', 'GenreLabel']].iloc[1:7]) # 取1 - 7  前6个




#MAP
poke_df = pd.read_csv('./datasets/Pokemon.csv', encoding='utf-8')
# print(poke_df.head())
poke_df = poke_df.sample(random_state=1, frac=1).reset_index(drop=True)
# print(np.unique(poke_df['Generation']))

#自己编写字典映射
gen_ord_map = {'Gen 1': 1, 'Gen 2': 2, 'Gen 3': 3,
               'Gen 4': 4, 'Gen 5': 5, 'Gen 6': 6}

poke_df['GenerationLabel'] = poke_df['Generation'].map(gen_ord_map)
print(poke_df[['Name', 'Generation', 'GenerationLabel']].iloc[4:10])


#OneHotEncoder
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


# transform and map pokemon generations
gen_le = LabelEncoder()
gen_labels = gen_le.fit_transform(poke_df['Generation'])
poke_df['Gen_Label'] = gen_labels

# transform and map pokemon legendary status
leg_le = LabelEncoder()
leg_labels = leg_le.fit_transform(poke_df['Legendary'])
poke_df['Lgnd_Label'] = leg_labels

poke_df_sub = poke_df[['Name', 'Generation', 'Gen_Label', 'Legendary', 'Lgnd_Label']]
print(poke_df_sub.iloc[4:10])



#  GET dummy

gen_dummy_features = pd.get_dummies(poke_df['Generation'], drop_first=False)# 不去掉第一行
print(pd.concat([poke_df[['Name', 'Generation']], gen_dummy_features], axis=1).iloc[4:10])


from sklearn.preprocessing import Binarizer #二值化预处理

from sklearn.preprocessing import Binarizer

# bn = Binarizer(threshold=0.9)大于0.9 =1   小于 0.9 =  0
# # pd_watched = bn.transform([popsong_df['listen_count']])[0]
# # popsong_df['pd_watched'] = pd_watched
# # popsong_df.head(10)