# GCMC_recommendation

## Introduction

### 연구 목표

GCMC 논문을 이해하고 이를 구현한 소스 코드를 활용하여 실 고객 데이터 기반, 각 노드의 특징을 포함한 추천 시스템 환경을 구축하고 도커 이미지로 배포한다.

구체적으로는 다음과 같다.

1. GCMC 방법론의 이해 및 분석 수행
2. 도서 평점 데이터를 이용, 유저와 도서의 특징을 활용한 GCMC 추천시스템 구현
    - 사용자와 도서의 정보가 들어있는 테이블을 이용
    - 피처 엔지니어링 및 그래프 관점의 임베딩을 이용한 특징벡터를 생성해 유저, 도서 노드의 특징을 포함한 추천시스템 구현
3. 구현된 분석 환경을 도커의 이미지화
4. 정리된 코드를 Github에 업로드해 GCMC 추천시스템 환경 구축

### 연구 일정
<div style="text-align: center;">
  <img src="./images/wbs.png" /> <br>
</div>



## 논문 소개

### Idea

- Graph 구조의 방법론을 추천 시스템 중 Matrix Factorization에 투영
- user-item rating matrix를 graph 구조로 표현
- 데이터를 matrix가 아닌 graph로 표현한 것만 제외하면, 풀고자 하는 문제는 matrix factorization과 본질적으로 동일 (추천 시스템에서의 matrix completion 문제를 "link prediction on graph" 관점에서 해결)

### Intro
<details><summary> <b> 1. Framework </b> </summary>

user-item rating matrix를 user-item의 이분그래프로 표현하여 graph auto-encoder 프레임워크에 응용, 결과적으로 rating(edge, link)을 예측하는 시스템. matrix의 0점 부분을 예측한다는(채운다는) 의미에서 matrix completion이라고 할 수 있다.

</details>

<details><summary> <b> 2. User-item rating matrix </b> </summary>

user, item 수를 각각 Nu, Nv라 하고, user i가 item j에 메긴 rating을 r이라 할 때, rating matrix M은 다음과 같이 표현한다.

</details>

<details><summary> <b> 3. Rating matrix Graph Mapping </b> </summary>

user와 item 각각은 node로, rating은 edge로 표현된다. 
예를 들어 user1이 item1에 5점을 매긴 경우(M11=5) 그래프로 나타내면 user1, item1의 node를 연결하고, edge에 5점을 부여한다.

그래프로 기존의 Rating Matrix를 표현하는 방식은 다음과 같다.

W는 User와 Item의 집합을 나타내는 기호이고, E는 Edge 정보, 그리고 R은 Rating에 대한 정보를 의미한다. 각각은 Graph에서 Node, Edge, Link의 역할을 하게 된다. 

</details>

### Graph Auto Encoder
<details><summary> <b> 1. Framework </b> </summary>

- encoder 모델: user, item 각각에 대한 latent vector를 구한다. (matrix factorization에서 latent vector를 구하는 것과 동일). 논문에서의 encoder모델은 ​​graph convolutional network(gcn)이다.

- decoder 모델: latent vector로 원래의 rating을 예측한다. 일반적으로 u와 v의 내적을 사용하나 논문에서는 별도의 decoder 사용한다.
</details>

<details><summary> <b> 2. Encoder </b> </summary>
a) 개념
encoder 모델  f(.)는 다음과 같이 정의된다. 이는 앞서 user, item각각에 대한 latent vector를 구하는 것이라 하였다.





설명
표현
U, V
user latent vector, item latent vector로 이뤄진 matrix

E는 latent vector 차원
XN×D
user, item 각각에 대한 dense vector로 이뤄진 feature matrix

ANu×Nv
(or Mr)
graph adjacency matrix
edge가 있는 부분은 1 없는 부분은 0.
이를 각 rating에 대해 표현할 수도 있는데, r점에 대한 adjacency matrix Mr은 Mr∈0,1Nu×Nv로 정의됨.



정리하자면, encoder 모델은 feature matrix X와, adjacency matrix A(orMr)를 갖고 latent vector를 구하는 것이라 할 수 있다.

b) Graph convolutional encoder
		
논문이 제시한 encoder 모델은 graph convolutional network(gcn)이다.
본 논문에서는 1~5 사이의 Rating을 모두 다른 채널로 나누어 학습에 사용한다. 이는 파라미터의 관점에서 5개의 다른 shared parameter가 존재한다는 것을 의미하기도 한다. 
CNN에서 중심점의 근방 n개 (ex : 3x3 filter)만을 convolutional 하듯이, 네트워크에서도 node의 1-depth 만큼의 연결만을 들여다본다. 그리고 그 각각의 iteration을 shared parameter로 학습하는 것이다. 이것을 Local Graph Convolution 이라고 하며 shared 파라미터를 Edge-type Specific Parameter라고 한다. (WR로 표기하며, 후에 나오는 hidden layer ↔ encoder 간의 파라미터 W와는 별개이다)

이해하기 쉽게 풀어쓰자면 1점을 부여한 user-item의 모든 관계를 W1로 학습하여 generalization 하고, 2점을 부여한 user-item의 모든 관계를 W2로 학습하여 generalization 한다는 것이다. 그리고 이를 이용해 X와 WR 두 행렬로 연산된 결과를 활용하겠다는 것이다. 마치 Matrix Factorization에서 user-item rating 값을 latent factor로 generalization 하는 것과 유사하다는 것을 알 수 있다. 다만 GC에서는 MF처럼 내적의 연산결과를 추정하는 것이 아니라, 관계 자체를 generalization 한다는 것에서 조금 더 나은 아이디어라고 볼 수 있다. 그리고 User와 Item의 Densed Feature를 활용할 수 있다는 점에서 꽤나 큰 성능 차이를 보일 것이다.

Weight sharing(message)
message는 edge를 통해 두 노드가 weight를 주고받는다는 측면에서 생겨난 개념이다. user i와 item j가 rating edge r로 연결될 때 message는 아래와 같이 정의된다.

결국 user i에 대한 latent vector는 인접노드인 item j의 영향을 받으며,  두 노드는 서로 연결 되어있으니 반대도 성립한다.
식을 보면 알 수 있듯, 인접한 노드끼리는 같은 weight를 공유하고 있는 것이 핵심이다. 결국 학습을 거듭할수록 Wr 공간은 rating r을 갖는 user, item의 정보들로 표현이 될 것이다.
Message passing 단계를 거치면 아래의 그림과 같이 hi가 생겨나게 되는데, 이 결과가 바로 NN의 히든 레이어 역할을 하게 된다.

히든 레이어를 아래와 같이 한 번 더 통과시켜주면, output이 바로 임베딩 벡터가 된다.

 c) example
지금까지 개념적으로만 살펴봤다면 이제 구체적인 예를 들어본다. 해당 예시는 다음과 같이 user1이 item1과 item2에 각각 5점, 1점을 부여한 상황을 가정한다.

앞서 hi는 다음과 같이 정의된다 하였다. 여기서 Ni,R은 user i와 인접한 노드 중, rating = r을 만족하는 item 노드 집합을 나타낸다.

우리는 user1에 대해 보고있으니 i = 1이며, rating edge가 2개이므로 N1,1={2}, N1,5={1}가 존재한다. 이를 바탕으로 할 때, h1은 다음과 같다.

이어 message μ가 다음과 같이 계산된다고 하자.

이때 accum(.) = sum(.)으로 가정하면 h1은 다음과 같이 계산된다. 여기서 σ(.)는 activation function으로 논문에서는 relu를 사용했다.

결국 user 1에 대한 latent vector h1은 user1과 인접한 노드 item1, item2의 정보를 사용하여 계산된다는 것에 주목하자. 여기에 다른 item의 정보는 사용되지 않았다. 이는 cnn의 convolution이 영역을 한칸씩 옮기며 주변 탐색해 가는 원리랑 비슷하다.
또한 앞서 여기서는 adjacency matrix A를 사용하지 않는다고 했는데, 그 이유는 단순하다. A를 곱하는 이유는 (matrix 연산을 직접 해보면 알수 있듯) 인접 노드만 가져오기 위함인데, 여기서는 이미 인접 노드만 가져와서 계산을 하기때문에 굳이 고려할 필요가 없는 것이다.
https://untitledtblog.tistory.com/152 <GCN 기본개념>

</details>

<details><summary> <b> 3. Decoder </b> </summary>
a) 개념
decoder 모델 g(.)는 다음과 같이 정의된다. 이는 앞서 latent vector를 갖고 rating을 예측하는 것이라 하였다.


가장 Classic하게 평점을 예측하는 방법은 User 임베딩 벡터와 Item 임베딩 벡터를 바로 내적 계산하는 것이다.

https://yeomko.tistory.com/5  <MF 내적>

b) Bilinear decoder
GCMC에서는 아래의 softmax의 개념을 사용하여 각 점수(r=1,2,...R)를 확률표현으로 나타낸다.

연산 결과 rating에 따른 확률의 분포가 생성되고, 이 수식에는 trainable parameter인 Q가 사용된다(ExE dimension). 그리고 최종적인 rating prediction은 아래의 수식을 한 번 더 거쳐 완성된다.


즉, item j에 대한 user i의 rating은 rating을 가중평균해서 예측하며, 이때 가중치로 를 사용한다. 여기서 Qr은 encoder모델의 Wr과 같은 parameter matrix다.
</details>

### Model Training

<details><summary> <b> Detail </b> </summary>
a) Loss function

* I 함수: Indicator Function. [ ] 안이 참일 때 1, 그렇지 않을 때 0의 값 (일종의 Mask 역할로 평점 기록이 존재하는 Link에 대해서만 negative log likelihood를 적용) 결과적으로 negative likelihood L을 최소화 하는 parameter Θ = {W,Wr,Qr}를 찾는 것이 목표이다.


b) Node dropout
	
과적합(overfitting)의 위험을 줄이고 학습속도를 개선하는 등의 문제를 해결하기 위한 장치로 특정 Node에 대해 pdropout의 확률로 밖으로 나가는 모든 Message를 무작위로 drop out하는 방식으로 학습하는 것을 의미한다.
Message는 Dropout 단계 이후에 Rescaled된다. 


c) Mini batch
미니 배치는 User-Item Pair 총합에서 오직 고정된 수의 Contribution 만을 추출하여 학습에 사용한다는 것을 의미한다. 이렇게 함으로써 현재 Batch에서 존재하지 않는, 각 평점 Class의 User/Item 행을 제거할 수 있다.
효율적인 정규화, 모델 학습에 필요한 메모리 경감의 효과가 있다.

</details>

### Input Feature Representation and Side Information

<details><summary> <b> Detail </b> </summary>

Feature가 다른 User(Item)나 그들의 관심 사항을 제대로 구분할만한 충분한 정보를 갖고 있지 못한 경우 정보 병목현상이 발생할 수 있다. -> Side Information으로 활용한다.
Node i에 대해 User/Item Feature Vector xif 를 만들고, 이를 Dense Hidden Layer에 독립적인 채널을 통해 직접적으로 투입한다.

이 때 W1f,W2f 는 학습 가능한 Weight 행렬이며, User Weight Matrix와 Item Weight Matrix는 구분된다. Graph Convolutional Layer를 위한 Node Feature를 담고 있는 Input Feature Matrix X는 Graph에 있는 모든 Node에 대한 고유한 One-Hot 벡터와 함께 Identity Matrix로 선택된다.  논문에서 사용된 데이터셋에 한해서는, User/Item Content 정보는 제한적인 크기를 갖고 있기 때문에 이들을 Side Information으로 이용하고자 한다.
Side Information은 꼭 Per-Node Feature 벡터의 형태일 필요는 없다. 그래프 구조여도 되고, NLP 혹은 이미지 데이터일수도 있다. 물론 이 경우 위의 식은 RNN, CNN, GNN와 같이 미분 가능한 다른 모듈로 적절히 대체되어야 한다.
https://zzaebok.github.io/knowledge_g

</details>

### Optimization
<details><summary> <b> Detail </b> </summary>
신경망 학습의 목적은 손실 함수의 값을 가능한 한 낮추는 매개변수를 찾는 것이며, 이러한 문제를 푸는 것을 최적화(optimization)이라고 한다.
본 논문에서는 Adam 최적화 함수를 사용한다 Adam은 모멘텀과 AdaGrad를 융합한 방법이다.



구현 코드 구성 및 수정 사항
구현에 사용된 코드는 tanimutomo/gcmc (Link)이며, Pytorch 환경에서 구현되어 Customizing이 용이한 코드로 선정하였다.
파일 구성은 다음과 같다.

	
	수정된 사항은 다음과 같다. 

data 로드 - 특징벡터를 포함한 데이터를 url로 다운받아 불러올 수 있도록 코드 수정
weight 계산 - 특징벡터가 존재/존재하지 않는 경우 모두 config 파일에서 feature의 차원을 넘겨받아 weight 연산이 가능
accum 함수 - 메시지를 sum방식으로 accum하는 함수를 scatter_add를 이용해 수정
결과 도출&시각화 
test, train rmse를 그래프로 시각화하는 함수 작성
예측값과 실제값을 출력하는 함수 작성
라인 인수 - 분석 코드를 실행할 때 라인으로 인수를 받아 파라미터를 지정할 수 있도록 코드 작성
기타 불필요한 코드 정리, 변경된 환경에 맞게 코드가 실행될 수 있도록 패키지, 함수 수정

</details>

## Process
분석 방법론
Multi-modal

노드 테이블에 존재하는 텍스트, 이미지 등의 특징들을 함께 사용하기 위해 Multi-modal을 사용한다.

Graph Embedding

소셜네트워크, 지식그래프와 같이 유저, 아이템 각각의 그래프를 생성하고 centrality, 그래프 임베딩(gnn, node2vec 등)을 통해 특징 벡터를 추출하여 분석을 진행한다.

데이터 소개
데이터는 Kaggle의 Book Recommendation Dataset (Link)을 활용하였다. 데이터는 총 세가지로, 유저의 정보 데이터와 책의 정보 데이터, 유저가 책에 남긴 평점 데이터이다. 각 데이터의 컬럼명과 컬럼에 대한 설명, 데이터 간 관계를 시각화하면 다음과 같다. 
 
User Data
User Data는 총 278,858개의 레코드가 존재하며 int type의 User-ID, string type의 Location, float type의 Age로 구성되어 있다.

Book Data
Book Data의 경우 총 271,360개의 레코드가 존재하며 string type의 ISBN, Book-Title, Book-Author, Publisher, Image-URL-S, Image-URL-M, Image-URL-L, int type의 Year-Of-Publication으로 구성되어 있다.




Rating
Rating Data의 경우 총 1,149,780개의 레코드가 존재하며, User Data의 User-ID, Book Data의 ISBN, 0-10 사이의 int 값인 Book-Rating으로 구성되어 있다.


전처리 및 EDA
데이터의 전처리 및 EDA는 R&D의 목적성과 시간 관계상 kaggle의 다른 전처리와 EDA 예시를 참고하여 최대한 간결하게 진행했다. 	

User Data
User Data의 경우, 결측치는 다음과 같다.

User-ID          0
Location        92
Age         110762

어느 컬럼 중 하나라도 결측치가 존재하는 레코드 수는 ​​110,802이며, 이를 모두 제거한 User Data의 레코드 수는 168,056이다.

Book Data
Book Data의 경우, 결측치는 다음과 같다.

Book-Title             0
Book-Author            1
Year-Of-Publication    0
Publisher              2
Image-URL-S            0
Image-URL-M            0
Image-URL-L            3

결측치를 모두 ‘unknown’으로 대치한 뒤, 출판 연도가 0인 4,618개의 레코드를 제거했다. 이후, 값이 하나씩 밀려들어가 있는 레코드 값을 올바르게 대치해 준 Book Data의 레코드 수는 총 266,742이다.		
	
Rating
User Data와 Book Data의 전처리를 진행하면서 삭제된 User와 Book의 ID는 Rating에서도 제거해주어야 한다.

제거된 후 Rating의 총 레코드 수는 742,830이며, 평점이 0인 477,365건을 추가적으로 제거했다.
 
데이터가 지나치게 sparse해지는 것을 방지하기 위해 10권 이상의 책에 평점을 남긴 유저와 받은 평점이 10개 이상인 책만을 추출하여 최종적으로 265,465개의 레코드를 가진 Rating Data와 4353개의 User Data, 2160개의 Book Data를 추출하였다.
Feature Engineering
특징 벡터를 추출하기 위한 작업으로, 크게 Multimodal으로 특징 벡터를 추출하는 방식과  그래프 임베딩 아이디어를 이용하여 추출하는 방식으로 진행한다. 전자의 경우 유저의 특징 벡터와 책의 특징 벡터의 차원을 맞춰주기 위해 3가지 방식으로 스케일을 조정해 주었으며, 후자의 경우 2가지 아이디어로 특징 벡터를 추출했다. 

즉, 총 5가지의 특징 벡터를 포함해 GCMC를 수행하고 각 방식 간의 유의미한 차이가 있는지 확인하고자 한다.
	
User Data
User Data의 경우 Location과 Age 두 가지 컬럼이 존재하므로 Multimodal 개념을 적용하지 않고 Location 컬럼에 대한 크롤링만 진행한다. 

위치 정보 크롤링
User의 Location 컬럼의 경우,  ‘도시, 국가'의 형식의 문자열로 구성되어 있다. 그러나 아래의 ‘nyc, new york, usa’와 같이 같은 표현이 중복되기도 하며 일관적이지 않다는 것을 확인했다. 이를 구글맵에 검색해 위도와 경도를 추출하는 방식으로 가공한다면 지리적인 특성을 보다 반영하고 일관성을 유지할 수 있을 것이라고 판단해 크롤링을 진행하였다.

크롤링에 앞서, 문자열에서 ‘n/a’ 표현이나 불필요한 문장 부호를 제거해주었으며 크롤러는 구글맵에 가공된 문자열을 포함해 위치 정보를 검색하면 약 5초 뒤에 url이 변경되며 위도와 경도 정보를 포함하는 것을 이용해 제작하였다. (Link)		


Scaling
StandardScaler를 이용해 각 컬럼별 데이터의 스케일을 맞춰주는 작업을 진행했다.	

Book Data
Multimodal
Book Data에서 Book Title 컬럼은 Text Embedding을, Image-URL컬럼은 Image Embedding을, Book Author, Publisher는 Categorical Embedding을 진행하고 Publication_year의 경우 StandardScaler를 이용해 스케일을 조정해주었다.

Text Embedding
SBERT의 clip-ViT-B-32모델을 이용해 임베딩을 진행했다. 위 모델은 이미지와 텍스트를 같은 벡터 공간으로 임베딩하는데 유용하다. 텍스트 임베딩 벡터의 차원은 512이다.

Image Embedding
Image-URL컬럼의 경우 이미지를 다운로드 받을 수 있는 링크로 구성되어 있다. 이를 다운로드받아 리스트에 append하며 이미지가 존재하지 않는 경우 검정 이미지를 생성해 이를 append해주었다. 임베딩 결과는 텍스트 임베딩 벡터와 마찬가지로 한 이미지가 512개의 값으로 표현된다.

Categorical Embedding
Author와 Publisher의 경우 고유값 리스트를 생성한 뒤, binary encoder를 이용해 인코딩을 진행했다. binary encoder를 이용하면 one-hot encoding을 이용하는 것보다 dense한 벡터를 추출할 수 있다. 임베딩 결과 author는 8개, publisher는 10개의 이진 벡터를 추출했다.

Dimension Scaling
특징 벡터를 포함해 GCMC를 수행하기 위해서는 유저와 책의 특징 벡터의 열의 차원을 통일시켜주어야 한다. User의 특징 벡터의 경우 3, Book의 특징 벡터의 경우 1042이므로 이를 통일된 D로 스케일링하는 것을 목적으로 한다.

본 R&D에서는 PCA. Neural Network, Variational AutoEncoder의 세 가지 방식으로 차원을 통일하고자 한다.

PCA
PCA(Principal component analysis)는 고차원의 데이터를 저차원의 데이터로 환원시키는 데 사용되며 표본의 차이를 가장 잘 나타내는 성분들로 분해하는 기법이다.
PCA의 경우 User Data의 열 차원 수에 맞춰 주성분의 수를 3으로 설정하였다.

Variational AutoEncoder
VAE는 기존 데이터의 확률 분포를 유지하여 새로운 데이터를 생성하는 모델이다. VAE의 경우 PCA와 마찬가지로 User Data의 열 차원 수에 맞춰 주성분의 수를 3으로 설정하였다.

Neural Network
NN을 이용한 방식은 앞선 두가지 방식과는 다르게 User와 Book의 특징 벡터 차원을 9로 설정하였다. User의 경우 3에서 9차원으로 차원이 증가하며, Book의 경우 1042에서 9차원으로 축소된다. 


Graph Embedding
기존의 분석방법론에서 아이디어는 유저, 북 각각의 소셜네트워크나 지식그래프를 그래프 임베딩을 통해 side information으로 이용하여, feature engineering을 통해 구한 특징벡터와 함께 사용하는 것이었다. 
하지만 해당 데이터의 유저, 북 테이블을 이용하여 별도의 소셜 네트워크, 지식그래프를 형성하는 것은 불가하다고 판단하여 그래프 임베딩을 feature vector로 사용하는 2가지 아이디어를 이용해 분석을 진행한다.

Idea1


User/Book Feature를 이용하여 유사도를 계산한 후 fully connected 그래프를 형성한다. 그리고 유사도가 비슷한 노드가 가깝게 위치하도록 그래프 임베딩을 진행한다.

Idea 2


User/Book Feature를 이용하여 그래프를 형성한 후 node2vec을 이용하여  유사한 노드가 가깝게 위치하도록 그래프 임베딩을 진행한다.
	

분석 수행 

두 가지의 방법으로 데이터를 추출하여 분석을 수행한다.

10권 이상의 책에 평점을 남긴 유저 & 10명 이상의 유저가 평점을 남긴 책 데이터

edge가 많은 node를 추출하여 많은 연결성과 정보를 바탕으로 콜드 스타트 문제를 보완하여 학습시킨다.

Feature Vector가 없는 경우



Multi modal - PCA

		
Multi modal - VAE
		
Multi modal - Neural network



Graph Embedding Idea1



Graph Embedding Idea2




1~10점의 평점이 고르게 분포된 데이터

기존의 데이터는 점수별 불균형 문제를 고려하지 않았으므로 가장 레코드 수가 적은 1점의 레코드 수 870에 맞춰 총 8700개의 데이터를 추출했다. 점수가 고르게 분포된 데이터를 이용하여 클래스 불균형의 문제를 보완하여 학습시킨다.

Feature Vector가 없는 경우



Multi modal - PCA



Multi modal - VAE



Multi modal - Neural network



Graph Embedding Idea2




 결과 분석 및 비교

10권 이상의 책에 평점을 남긴 유저 & 10명 이상의 유저가 평점을 남긴 책 데이터

[성능] 대체적으로 모든 방법의 Test RMSE가 3점에서 시작되어 2점으로 끝나는 비슷한 성능을 보였음

Feature Engineering에서 그래프 임베딩을 통해 특징벡터를 추출하는 그래프 임베딩 아이디어2 방법이 가장 좋은 성능을 보임.
이는  node2vec을 이용하여 그래프의 구조적인 관점에서 유사한 노드를 가까운 위치에 임베딩 한 것이 결과에 긍정적인 영향을 미쳤다는 것을 확인할 수 있음

[과적합] rmse가 줄어들다가 어느순간부터 늘어나는 추세를 보이는 부분은 트레인셋에 오버피팅이 되었다고 생각할 수 있고 적절한 epoch설정이 필요로해 보임

[optimizer] epoch 0~200에서 급격하게 학습이 되는 것을 알 수 있고 이는 optimizer Adam의 특성에 부합한다고 볼 수 있음

[클래스 불균형] 예측값과 실제 값을 출력해 본 결과 대부분 5 이상의 값으로만 예측을 하며 낮은 점수에 대해 예측 성능이 낮은 것을 확인하였고, 본 데이터에 5-10점의 데이터가 많음에도 불구하고 클래스 불균형 문제를 고려하지 않은 결과라고 판단됨

1~10점의 평점이 고르게 분포된 데이터
[과적합] 트레인셋은 계속 학습이 되는 반면, 테스트셋은 성능이 점점 떨어지는 것을 보아 트레인셋에 과적합 되어가고 있다고 유추할 수 있음. 

[데이터] 클래스 불균형 문제를 해결했기 때문에 불균형한 A 데이터에 비해 성능이 향상될 것이라고 예측했으나 가정이 만족되지 않았음. 
이는  A의 데이터와 달리 데이터의 양 자체도 크게 감소하였으며  여러 유저에게 평점을 받은 아이템이 없어 그래프가 sparse해졌기 떄문에 edge에 대한 encoder에서의 학습이 부족해 결과가 좋지 않았다고 생각됨.

[cold start] sparse한 그래프에서 feature vector, 그래프 임베딩을 이용하면 cold start 문제를 어느정도 보완할 수 있을 것이라 생각하였지만 이를 이용한 분석에서도 좋지 않은 성능을 보임.
이는 로우 데이터를 잘 나타내는 Feature Engineering을 하지 못하였거나 데이터가 부족하여 결과가 좋지 않았다고 생각됨.

하이퍼 파라미터 튜닝

다음은 고도화를 위해 진행한 모델의 하이퍼파라미터 튜닝 결과이다.
하이퍼 파라미터 튜닝은 A(10권 이상의 책에 평점을 남긴 유저 & 10명 이상의 유저가 평점을 남긴 책 데이터) 데이터에 한해서 진행하였다.



Epoch
Learning Rate
Hidden Layer
Drop Out
RMSE
Default
1000
0.01
500, 75
0.7
1.95
Tuning 1
1000
0.1
500, 75
0.7
3.03
Tuning 2
1000
0.001
500, 75
0.7
1.79
Tuning 3
1000
0.0001
500, 75
0.7
2.04
Tuning 4
1000
0.01
250, 75
0.7
1.94
Tuning 5
1000
0.01
750, 75
0.7
1.90
Tuning 6
1000
0.01
1000, 100
0.7
1.89
Tuning 7
1000
0.01
500, 75
0.9
2.0
Tuning 8
1000
0.01
500, 75
0.1
1.87
Tuning 9
1000
0.01
500, 75
0
1.80
Best
150
0.01
500,75
0
1.67


이를 통해 Optimizer인 Adam의 특성상 Epoch가 150~200쯤에서 대부분의 학습이 이루어지며 Learning Rate는 값이 작을 수록 학습률이 높아지고 hidden layer의 수가 많을수록 학습이 잘 되며 Drop out이 작을 수록 학습이 잘 된다는 결론을 내릴 수 있었으며 최종 RMSE를 1.67로 모델의 성능을 더욱 향상 시킬 수 있었다.
Hidden Layer, Drop out과 같은 하이퍼파라미터는 성능과 계산량이 Trade Off 관계이므로 값이 클수록 학습이 잘되지만 시간이 오래 걸리기 때문에 지금보다 더욱 많은 데이터를 다룰 때에는 어느정도의 성능을 포기하더라도 계산량을 줄이는 등 적절한 튜닝이 필요함을 알 수 있었다.
Conclusion
결론
관계가 많은 첫번째 데이터에서 모든 방법이 전체적으로 학습이 잘 이루어지고, Graph Embedding Idea 2가 가장 좋은 성능을 가진 결과를 통해 기존의 논문에서 GCN을 이용해 주변의 정보를 Latent Vector로 generalization하여 분석을 수행하면 유저와 아이템의 Densed Feature를 사용할 수 있어 더 좋은 성능의 결과를 도출할 수 있을 것이라는 가설을 충족할 수 있었다.

반면에 클래스 불균형을 해결한 sparse한 두번째 데이터에서 학습이 거의 이루어지지 않은 결과를 통해 기존의 논문에서 Feature Engineering을 통해 Cold Start문제를 보완할 수 있을 것이라는 가설이 충족되지 않음을 알 수 있었다. 하지만 이는 Feature Engineering과정에서 Multimodal을 사용하는 과정과 Dimension Scaling 과정에서 적절한 하이퍼파라미터 튜닝을 통해 최종 Feature Vector가 로우 데이터를 잘 설명하는지 검증을 하는 과정이 생략되었고, 전체 데이터를 이용한 학습이 아닌 데이터의 일부를 추출하여 적은 양의 데이터로 학습을 하였기 때문에 아직 가설이 틀렸다고 말할 수 없는 부분이다. 추후에 이러한 과정 이후에도 학습이 제대로 이루어지지 않는다면 논문의 가설은 충족되지 않는다고 결론을 내릴 수 있을 것이다.

고도화 방안

Feature Engineering 과정에서의 하이퍼 파라미터 튜닝

Node2Vec
PCA
VAE
Neural Net

특징을 추출하는 과정에서 추출된 특징이 로우데이터를 잘 설명하도록 하이퍼파라미터 튜닝이 필요함

GCMC 모델에서의 하이퍼 파라미터 튜닝

Epoch
Learning Rate
Hidden Layer
Drop Out
Accumulation Function

각 데이터에 적합한 모델의 하이퍼파라미터 설정을 통해 향상된 결과를 도출할 수 있음

보다 많은 데이터를 수집해 지식 그래프 생성 
ex) 책 정보를 크롤링하여 장르 데이터 추가 수집/유저 간 연결망.. 

그래프 오버샘플링 방법 탐색

기존의 오버샘플링 방법은 새로운 관계를 생성해야하는 본 분석문제에 적합하지 않으므로 그래프적 관점의 오버샘플링 방법 탐색 필요

ex) Graph SMOTE

실 고객 데이터 기반의 GCMC 수행 
ex) 유저 간의 소셜네트워크, 아이템의 지식그래프를 별도로 생성 가능한 충분한 데이터

충분한 데이터를 통해 유저와 아이템 사이의 평점 그래프 외에도 유저 간의 소셜 네트워크, 아이템 간의 지식 그래프 등을 생성할 수 있다면 GNN, GCN, GraphSAGE 등 다양한 그래프 임베딩을 통해 그래프의 정보를 담아 Cold Start문제를 보완할 수 있을 것이라 예상됨
