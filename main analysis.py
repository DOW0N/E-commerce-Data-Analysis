import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

def set_korean_font():
    font_path = 'C:/Windows/Fonts/malgun.ttf'  # 한글 폰트 경로 설정
    font_prop = fm.FontProperties(fname=font_path)
    plt.rc('font', family=font_prop.get_name())
    sns.set(font=font_prop.get_name())  # Seaborn에도 폰트 설정

set_korean_font()

# 데이터 불러오기
file_path = r"C:\src\ecommerce_dataset_updated.csv" 
ecommerce_df = pd.read_csv(file_path)

# 데이터의 상위 5개 샘플 확인
print(ecommerce_df.head())

# 결측치 처리
ecommerce_df = ecommerce_df.dropna()

# 'Purchase_Date'를 날짜 형식으로 변환 (날짜 형식 명시)
ecommerce_df['Purchase_Date'] = pd.to_datetime(ecommerce_df['Purchase_Date'], format='%d-%m-%Y')

# ===== 1. 가격 추세 분석 =====
monthly_price = ecommerce_df.groupby(ecommerce_df['Purchase_Date'].dt.to_period('M'))['Price (Rs.)'].mean()

# 월별 가격 추세 시각화
plt.figure(figsize=(12, 6))  #
monthly_price.plot(kind='line', color='b')
plt.title('월별 평균 가격 추세')  
plt.xlabel('월')  
plt.ylabel('평균 가격')  
plt.grid(True)

# x축 레이블이 겹치지 않게 회전
plt.xticks(rotation=45)

# 레이아웃 조정 (레이블이 짤리지 않게)
plt.tight_layout()

plt.show()

# ===== 2. 지불 방법 분포 =====
plt.figure(figsize=(10, 6))  # 그래프 크기 확장
sns.countplot(x='Payment_Method', data=ecommerce_df, palette='Set2')
plt.title('지불 방법 분포')  
plt.xlabel('지불 방법')  
plt.ylabel('구매 수')  
plt.xticks(rotation=45)

# 레이아웃 조정 (레이블이 짤리지 않게)
plt.tight_layout()

plt.show()

# ===== 3. 고객 구매 패턴 분석 =====
customer_purchase_count = ecommerce_df['User_ID'].value_counts()

# 상위 10명의 고객을 시각화
top_10_customers = customer_purchase_count.head(10)

plt.figure(figsize=(12, 6))  # 그래프 크기 확장
sns.barplot(x=top_10_customers.index, y=top_10_customers.values, palette='viridis')
plt.title('상위 10명의 고객 구매 횟수')  # 한글 제목
plt.xlabel('고객 ID')  # 한글 레이블
plt.ylabel('구매 횟수')  # 한글 레이블
plt.xticks(rotation=45)

# 레이아웃 조정 (레이블이 짤리지 않게)
plt.tight_layout()

plt.show()

# ===== 4. 예측 모델링 (회귀 분석) =====
# 'Discount (%)'와 'Quantity'를 독립 변수로, 'Final_Price(Rs.)'를 종속 변수로 설정
ecommerce_df['Quantity'] = 1  # 단순히 'Quantity'를 1로 설정 (예시, 실제 데이터에서 조정 필요)

# 독립 변수와 종속 변수 설정
X = ecommerce_df[['Discount (%)', 'Quantity']]
y = ecommerce_df['Final_Price(Rs.)']

# 데이터 분할 (훈련용, 테스트용)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 회귀 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 성능 평가
mse = mean_squared_error(y_test, y_pred)
print(f'평균 제곱 오차(MSE): {mse}')

# ===== 5. 예측 모델링 (분류 모델) =====
# 'Purchased'가 1이면 구매한 것으로, 0이면 구매하지 않은 것으로 설정 (가정)
ecommerce_df['Purchased'] = ecommerce_df['Final_Price(Rs.)'].apply(lambda x: 1 if x > 0 else 0)

# 독립 변수와 종속 변수 설정
X_class = ecommerce_df[['Discount (%)', 'Quantity']]
y_class = ecommerce_df['Purchased']

# 데이터 분할 (훈련용, 테스트용)
X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

# 분류 모델 학습 (Random Forest)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# 예측
y_pred_class = classifier.predict(X_test)

# 성능 평가
accuracy = accuracy_score(y_test, y_pred_class)
print(f'정확도: {accuracy}')
