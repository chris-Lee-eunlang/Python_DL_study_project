import numpy as np
import matplotlib.pyplot as plt

# PDF 출력
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')

#시그모이드 함수 그래프프
# xx = np.linspace(-6, 6, 500)
# yy = 1 / (np.exp(-xx) + 1)

# plt.figure(figsize=(6,6))
# plt.ylim(-3, 3)
# plt.xlim(-3, 3)
# plt.xticks(np.linspace(-3,3,13))
# plt.yticks(np.linspace(-3,3,13))
# plt.xlabel('x', fontsize=14)
# plt.ylabel('y', fontsize=14)
# plt.grid()
# plt.plot(xx, yy, c='b', label=r'$\dfrac{1}{1+\exp{(-x)}}$', lw=1)
# plt.plot(xx, xx, c='k', label=r'$y = x$', lw=1)
# plt.plot([-3,3], [0,0], c='k')
# plt.plot([0,0], [-3,3],c='k')
# plt.plot([-3,3],[1,1],linestyle='-.',c='k')
# plt.legend(fontsize=14)
# plt.show()

# 학습용 데이터 준비
from sklearn.datasets import load_iris
iris = load_iris()
x_org, y_org = iris.data, iris.target
print('원본 데이터', x_org.shape, y_org.shape)
# 데이터 추출
#   클래스 0, 1만
#   항목 sepal_length과 sepal_width만
x_data, y_data = iris.data[:100,:2], iris.target[:100]
print('대상 데이터', x_data.shape, y_data.shape)

# 더미 변수를 추가
x_data = np.insert(x_data, 0, 1.0, axis=1)
print('더미 변수를 추가 한 후', x_data.shape)

#원본 데이터의 크기
print(x_data.shape, y_data.shape)

#학습 데이터, 검증 데이터로 분할(셔플도 함께 실시)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
  x_data, y_data, train_size = 70, test_size = 30, 
  random_state = 42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
x = x_train
yt = y_train
print(yt[:5])
#시그모이드 함수
def sigmoid(x):
  return 1/(1+np.exp(-x))
#예측값 계산
def pred(x,w):
  return sigmoid(x @ w)

#교차 엔트로피 함수
def cross_entropy(yt, yp):
  #교차 엔트로피의 계산(이 단계에서는 벡터)
  ce1 = -(yt * np.log(yp)+(1-yt)*np.log(1-yp))
  #교차 엔트로피 벡터의 평균값을 계싼
  return(np.mean(ce1))

def classify(y):
  return np.where(y<0.5, 0, 1)

# 모델을 평가하는 함수
from sklearn.metrics import accuracy_score
def evaluate(xt, yt, w):
    
    # 예측값 계산
    yp = pred(xt, w)
    
    # 손실함수 값 계산
    loss = cross_entropy(yt, yp)
    
    # 예측값(확률값)을 0이나 1로 변환
    yp_b = classify(yp)
    
    # 정확도 산출
    score = accuracy_score(yt, yp_b)
    return loss, score




#초기화 처리

#표본 수
M = x.shape[0]
#입력 차원 수(더미 변수를 포함)
D = x.shape[1]

#반복 횟수
iters =10000

#학습률
alpha = 0.01

#초깃값
w = np.ones(D)

#평가 결과 기록(손실함수와 정확도)
history = np.zeros((0,3))

#반복 루프
for k in range(iters):
  #예측값 계산
  yp = pred(x,w)
  #오차 계산
  yd = yp-yt
  w = w-alpha *(x.T @ yd /M)
  
  #손실함수 계산
  if (k % 10 ==0):
    loss, score = evaluate(x_test, y_test, w)
    history = np.vstack((history, np.array([k, loss, score])))
    print("iter = %d loss %f score %f" % (k,loss,score))

#검증 데이터를 산점도용으로 준비
x_t0 = x_test[y_test == 0]
x_t1 = x_test[y_test == 1]
def b(x,w):
  return(-(w[0] + w[1]*x)/w[2])
#산점도 x1의 최솟값과 최댓값
x1 = np.asarray([x[:,1].min(), x[:,1].max()])
y1 = b(x1,w)
plt.figure(figsize=(6,6))
#산점도 표시
plt.scatter(x_t0[:,1], x_t0[:,2], marker = 'x', 
            c='b', s=50, label='class 0')
plt.scatter(x_t1[:,1], x_t1[:,2], marker ='o',
            c='k', s=50, label ='class 1')
#산점도에 결정경계 직선을 추가
plt.plot(x1, y1, c='b')
plt.xlabel('sepal_length', fontsize = 14)
plt.ylabel('sepal_width', fontsize =14)
plt.xticks(size = 16)
plt.yticks(size = 16)
plt.legend(fontsize = 16)
plt.show()

#학습곡선 표시(손실함수)
plt.figure(figsize = (6,4))
plt.plot(history[:,0], history[:,1], 'b')
plt.xlabel('iter',fontsize=14)
plt.ylabel('cost',fontsize=14)
plt.title('iter vs cost',fontsize = 14)
plt.show()

#학습곡선 표시 정확도
plt.figure(figsize =(6,4))
plt.plot(history[:,0],history[:,2],'b')
plt.xlabel('iter', fontsize =14)
plt.ylabel('accuracy', fontsize =14)
plt.title('iter vs accruacy', fontsize = 15)
plt.show()