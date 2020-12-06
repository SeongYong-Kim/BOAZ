# Neural Machine Translation by Jointly Learning to Align and Translate



## Abstract

Encoder - Decoder 방식의 NMT(Neural Machine Translation)는 input seq/seq를 fixed length vector로 구성해야한다. 이 방법은 문장이 길어질 경우 성능이 급격하게 저하된다는 단점이 있다. 성능저하의 원인은 문장 전체의 정보를 압축하면서 단어 순서에 대한 정보가 사라지기 때문이다. 그렇다면 어떤 단어에 주목해야 하는지에 대한 정보가 주어진다면 번역이 더욱 제대로 이루어지지 않을까? 이 논문은 이런 점에 착안하였고 Alignment에 주목하여 Soft-serach를 이용한 번역 모델을 제시한다.



## Main Idea

![image-20201112111824943](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20201112111824943.png)

 Alignment는 **Output sentence의 단어가 Input sentence의 어느 부분에 주목해야 하는지에 대한 개념**으로 이를 계산하기 위해서는 Encoder에서 context vector를 계산하기 위해 사용되고 버려지는 hidden state와 Decoder에서의 hidden state를 이용한다. 이는 Decoder에 사용되는 수식을 통해 보다 쉽게 이해할 수 있다.
$$
p(y_t|y_1,⋯,y_t-1,x)=g(y_t-1,s_i,c_i)
$$

$$
s_i=f(s_i−1,y_i−1,c_i)
$$

 기존의 Decoder와의 차이점은 각각의 output 단어에 대한 conditional probability를 계산함에 있어서 하나의 context vector를 사용하는 것이 아니라 **각각 다른 context vector를 사용한다**는 것이다. 이 context vector는 Encoder에서의 hidden state와 Decoder에서의 hidden state 그리고 weight를 곱하여 계산된다.





