import pandas as pd

"""### 방문자 정보"""

df_place = pd.read_csv('/content/tn_visit_area_info_방문지정보_G.csv')

df_place

"""### 여행 정보"""

df_travel = pd.read_csv('/content/tn_travel_여행_G.csv')

df_travel

"""### 여행객 정보"""

df_traveler = pd.read_csv('/content/tn_traveller_master_여행객 Master_G.csv')


"""## 전부 합치기"""

df = pd.merge(df_place, df_travel, on='TRAVEL_ID', how='left')
df = pd.merge(df, df_traveler, on='TRAVELER_ID', how='left')


"""## 데이터셋 전처리"""

df_filter = df[~df['TRAVEL_MISSION_CHECK'].isnull()].copy()

df_filter.loc[:, 'TRAVEL_MISSION_INT'] = df_filter['TRAVEL_MISSION_CHECK'].str.split(';').str[0].astype(int)


"""## 데이터셋 전처리 2

- 필요한 열만 추출
- 비어있는 행 삭제
"""

df_filter = df_filter[[
    'GENDER',
    'AGE_GRP',
    'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4', 'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
    'TRAVEL_MOTIVE_1',
    'TRAVEL_COMPANIONS_NUM',
    'TRAVEL_MISSION_INT',
    'VISIT_AREA_NM',
    'DGSTFN',
]]

# df_filter.loc[:, 'GENDER'] = df_filter['GENDER'].map({'남': 0, '여': 1})

df_filter = df_filter.dropna()


"""## 데이터셋 전처리 3

- 범주형(categorical) 데이터 정의
- 범주형 데이터는 string or integer 형태여야 함
"""

categorical_features_names = [
    'GENDER',
    # 'AGE_GRP',
    'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4', 'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
    'TRAVEL_MOTIVE_1',
    # 'TRAVEL_COMPANIONS_NUM',
    'TRAVEL_MISSION_INT',
    'VISIT_AREA_NM',
    # 'DGSTFN',
]

df_filter[categorical_features_names[1:-1]] = df_filter[categorical_features_names[1:-1]].astype(int)


"""## 데이터셋 전처리 4

- 학습/테스트 데이터 나누기
"""

from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(df_filter, test_size=0.2, random_state=42)


# !pip install -q catboost

"""### 데이터 형태 정의"""

from catboost import CatBoostRegressor, Pool

train_pool = Pool(train_data.drop(['DGSTFN'], axis=1),
                  label=train_data['DGSTFN'],
                  cat_features=categorical_features_names)

test_pool = Pool(test_data.drop(['DGSTFN'], axis=1),
                 label=test_data['DGSTFN'],
                 cat_features=categorical_features_names)

"""### 모델 정의 및 학습"""

# model = CatBoostRegressor(
#     loss_function='RMSE',
#     eval_metric='MAE',
#     task_type='GPU',
#     depth=6,
#     learning_rate=0.01,
#     n_estimators=2000)

# model.fit(
#     train_pool,
#     eval_set=test_pool,
#     verbose=500,
#     plot=True)

# 모델을 저장할 파일 경로
model_save_path = 'catboost_model.cbm'

# 모델 저장
# model.save_model(model_save_path)

# 저장된 모델 불러오기
from catboost import CatBoostRegressor

model = CatBoostRegressor()
model.load_model('/content/catboost_model.cbm')

area_names = df_filter[['VISIT_AREA_NM']].drop_duplicates()

# 이 형식으로 값이 입력되어야합니다.
# traveler = {
#     'GENDER': '남',
#     'AGE_GRP': 20.0,
#     'TRAVEL_STYL_1': 1, 자연 도시
#     'TRAVEL_STYL_2': 2, 고정
#     'TRAVEL_STYL_3': 2, 고정
#     'TRAVEL_STYL_4': 3, 고정
#     'TRAVEL_STYL_5': 2, 휴양/휴식 체험활동
#     'TRAVEL_STYL_6': 2, 고정
#     'TRAVEL_STYL_7': 2, 고정
#     'TRAVEL_STYL_8': 2, 고정
#     'TRAVEL_MOTIVE_1': 8, 고정
#     'TRAVEL_COMPANIONS_NUM': 0.0, 고정
#     'TRAVEL_MISSION_INT': 3, 고정
# }

results = pd.DataFrame([], columns=['AREA', 'SCORE'])

for area in area_names['VISIT_AREA_NM']:
    input = list(traveler.values())
    input.append(area)

    score = model.predict(input)

    results = pd.concat([results, pd.DataFrame([[area, score]], columns=['AREA', 'SCORE'])])

# 점수를 기준으로 상위 20개 여행지 정렬
top_20_results = results.sort_values('SCORE', ascending=False).head(20)

# 상위 20개 여행지에서 'AREA'만 추출
top_20_areas = top_20_results['AREA']

# JSON 형식으로 저장
output_json_path = './top_20_travel_areas.json'
top_20_areas.to_json(output_json_path, orient='values', force_ascii=False)

print(f"상위 20개 추천 여행지 결과를 '{output_json_path}' 파일로 저장했습니다.")
