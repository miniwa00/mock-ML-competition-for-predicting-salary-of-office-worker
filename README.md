# mock-ML-competition-for-predicting-salary-of-office-worker

### Overview
* * *
This is a mock machine learning competition held at Kookmin University's 'Machine Learning' course.  
Overall revisions were made based on my results, which placed 8th among 15 teams.
I used Jupyter Notebook on Visual Studio Code.

### Goal
* * *
This is a regression problem that requires predicting the salary of an office worker.

### Dataset
* * *
Employee data and salary from a specific company  
(Due to security issues, the company name is not disclosed.)  
  
**X_train**: 16570 rows × 11 columns  
**X_test**: 11048 rows × 11 columns

### EDA
* * *
- X_train은 다음과 같이 생겼습니다.
<img src="assets/X_train_head.PNG"/>

- y_train은 다음과 같이 생겼습니다.
<img src="assets/y_train_head.PNG"/>

- y_train의 분포는 다음과 같습니다.
<img src="assets/y_train_distribution.PNG"/>

- X_train의 info를 찍어보면 다음과 같습니다.
<img src="assets/X_train_info.PNG"/>

> _'직무태그'_ , _'근무형태'_ , _'어학시험'_ , _'대학성적'_ 컬럼에 결측치가 존재합니다.  
>  또한 수치형 변수들로 이루어진 _'대학성적'_ 을 제외한 모든 컬럼은 데이터타입이 'object'임을 확인할 수 있습니다.  

- 컬럼별 고유값의 개수는 다음과 같습니다.  
<img src="assets/X_train_columns_unique.PNG"/>



### Preprocess
* * *
개별 컬럼에 대해 전처리를 해보겠습니다.
먼저, X_train과 X_test에 같은 전처리를 해주기 위해 둘을 합치겠습니다.
<img src="assets/df_concat.PNG"/>
- _'직종'_ , _'세부직종'_ , _'출신대학'_ , _'자격증'_ 은 고유값 개수가 적고 결측치도 없어 전처리가 필요 없을 것 같습니다.

- _'어학시험'_ 컬럼을 살펴보겠습니다.
<img src="assets/foreign_language_missing_value.PNG"/>
<img src="assets/foreign_language_value_counts.PNG"/>

  > _'어학시험'_ 의 결측치는 '없음'으로 대체하는 것이 적절할 것 같습니다.

- _'근무경력'_ 컬럼을 살펴보겠습니다.
<img src="assets/work_year.PNG"/>

  > _'근무경력'_ 은 0년 00개월 형식입니다.  
  >  이를 모두 개월수로 바꾸어 수치형 변수로 바꿀 수 있을 것 같습니다.

  > 아주 잘 되었습니다.

- _'대학성적'_ 컬럼을 살펴보겠습니다.
  
  > 출신 대학이 분명히 있으니 성적도 있을텐데 왜 결측치가 있는지는 잘 모르겠습니다.
  > 어쨌든 평균으로 대체하는 것이 가장 합리적일 것 같습니다.

- _'근무형태'_ 컬럼을 살펴보겠습니다.
  
  > 근무형태가 결측치라는 것은 곧 신입이라는 뜻입니다.
  > '정규직, 계약직,'과 같이 끝에 ,가 있는 경우가 있습니다. 따라서 마지막의 ,를 제거해주겠습니다.

- _'대학전공'_ 컬럼을 살펴보겠습니다.
  
  > 같은 전공임에도 다르게 표현한 경우가 너무 많습니다.
  > 따라서 이들을 모두 수작업으로 통일시켜주겠습니다.

### Modeling
* * *


### What needs to improve?
* * *
