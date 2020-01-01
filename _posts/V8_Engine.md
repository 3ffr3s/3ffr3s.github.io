# V8 엔진
***
Chrome 및 안드로이드 브라우저에서 사용되는 c++로 만들어진 JavaSCript 엔진이다. V8은 JIT 컴파일러를 적용하여 JavaScript 코드를 실행할 때 컴파일하여 기계어 코드로 만든다. V8은 바이트코드나 다른 중간 코드를 생성하지 않는다.

#### Hidden class 
JavaScript는 프로토타입 기반 언어이다. JavaScript에는 class가 없고 object는 cloning 과정을 통해서 생성된다. JavaScript는 또한 동적으로 type된다. (JavaScript는 런타임에 데이터 타입을 변경할 수 있다.) type과 type 정보가 명시적이지 않고 실행 중에 (생성 이후에) object에 property들이 추가되거나 삭제될 수 있다. type과 property에 효율적으로 접근하는 것이 V8에게 어려운 과제였다. 대부분의 JavaScript 엔진이 하는 것처럼  object의 property를 저장하기 위해서 dictionary 같은 데이터 구조를 사용하고 동적으로 property의 위치를 탐색하는 대신에 V8은 런타임에 hidden class를 만든다. 이를 통해서 V8은 type 시스템을 내부적으로 표현하고 property에 접근하는 시간을 줄인다.
<br/>
Point 함수와 두 개의 Point object 생성을 예로 들어보자.

![Alt text](https://github.com/pwn3r/project_71/blob/master/week1/3ffr3s/hidden_class.JPG)

위의 예와 같은 레이아웃을 갖는다면 p와 q는 V8이 생성하는 hidden class에 속한다. 이 점은 hidden class를 사용하는 또다른 장점을 부각시킨다. V8은 property가 같은 object를 group화 할 수 있도록 한다. 이 때 p와 q는 동일한 optimized code를 사용한다. 
<br/>
위의 선언 이후에 q object에 z라는 property를 추가한다고 가정해보자. (동적으로 type되는 언어에서는 이런 행위가 가능하다.)
<br/>
V8은 위의 시나리오를 어떻게 해결할까? __V8은 생성자 함수가 property를 선언할 때마다 새로운 hidden class를 생성한다.__ 그리고 hidden class의 변화를 추적한다. 왜일까? 만약 두 개의 object가 생성되고 (p와 q) 두 번째 object의 생성 이후에 해당 object에 멤버가 추가된다면 V8은 마지막에 생성된 hidden class를 유지하고 (p object를 위해서) 새로운 멤버를 추가한 hidden class를 생성해야 하기 때문이다. (q object를 위한)   

![Alt text](https://github.com/pwn3r/project_71/blob/master/week1/3ffr3s/hidden_class2.JPG)

새로운 hidden class가 생성될 때 마다 이전 hidden class는 대신 사용될 hidden class로 업데이트 된다.  

###### Code optimization
V8이 각각의 property에 대해서 새로운 hidden class를 생성하기 때문에 hidden class 생성은 최소한이 되어야 한다. 이를 위해서 object 생성 이후에 property를 추가하는 것을 피해야 하고 항상 object 멤버들에 대한 초기화를 (생성을) 동일한 순서로 해야 한다. (hidden class가 여러 개 생성되는 것을 막기 위해서) 또한 모든 객체 멤버를 생성자 함수 안에서 초기화 해야 한다. (그래야 나중에 인스턴스가 멤버의 데이터 타입을 변경하지 않는다.)
<br/>
[업데이트]  Monomorphic operations은 동일한 hidden class를 갖는 object들에만 적용되는 연산이다. V8은 함수를 호출할 때 hidden class를 생성한다. 만약 우리가 서로 다른 parameter type을 갖는 함수를 여러 번 호출하면 V8은 다른 hidden class를 생성해야 한다. 따라서 polymorphic (다형적) 코드보다는 monomorphic (단형적) 코드를 사용해야 한다. 

#### V8이 어떻게 JavaScript 코드를 최적화 하는지에 대한 예시 
###### Tagged value
JavaScript object와 숫자들을 효율적으로 표현하기 위해서 V8은 object와 숫자를 32비트 값으로 나타낸다. 그리고 object인지 integer인지 구분하기 위해서 SMall Integer 또는 SMI라고 불리는 1bit 플래그를 사용한다. 따라서 숫자값이 31bit보다 크면 V8은 숫자를 감싸서 (box) double로 만들고 이 숫자를 저장할 새로운 object를 생성한다. 
<br/><br/>
__코드 최적화__ : JavaScript object를 만드는 비싼 boxing operation을 피하기 위해서 가능하면 31bit 짜리 signed 숫자를 사용해야 한다.
###### Arrays
V8은 array를 다루기 위해서 두 개의 서로 다른 method를 사용한다. 
- Fast elements : 키 집합이 매우 compact한 array를 위해 고안되었다. (키 값이 빈틈없이 채워진 array) array들은 매우 효율적으로 접근 가능한 linear storage buffer를 갖는다.
- Dictionary elements : 내부에 모든 element가 없는 sparse array를 위해서 고안되었다. Fast elements 보다 더 접근 비용이 비싼 hash table이다.

__코드 최적화__ : V8이 array를 다룰 때 Fast element를 사용하도록 해야 한다. 즉, 연속적인 key값을 갖지 않는 sparse array 사용을 피해야 한다. (array의 0번째 인덱스부터 시작하는 연속된 키 값 사용해야 한다.) 또한 큰 array를 미리 할당하는 것을 피해야 한다. 사용하면서 필요할 때 마다 늘리는 것이 더 좋다. 마지막으로 array에서 element 삭제를 하지 않아야 한다. element 삭제가 key set을 sparse하게 만들기 때문이다. 

#### V8이 어떻게 JavaScript 코드를 컴파일 하는가
V8은 두 가지 종류의 JIT 컴파일러를 갖는다.
- Full compiler : 모든 JavaScript에 대해서 좋은 코드를 생성하는 컴파일러이다. 하지만 좋은 JIT 코드를 생성하지는 못한다. 이 컴파일러의 목표는 코드를 빠르게 생성하는 것이다. 이 목표를 이루기 위해서 컴파일러는 어떠한 type 분석도 하지 않고 type에 대해서 아무 것도 모른다. 그 대신 Inline Caches 또는 IC 전략을 사용해서 프로그램을 실행하는 동안 변수의 데이터 type에 대한 정보를 구체화한다. IC는 매우 효율적이고 20배의 속도 향상을 가져온다. 
- Optimizing compiler : 대부분의 JavaScript 언어에 대해서 좋은 코드를 생성하는 컴파일러이다. 나중에 제공되며 hot function을 다시 컴파일한다. (hot 함수는 여러 번 실행되는 함수를 의미) Optimizing compiler는 Inline Cache에서 type들을 가져오고 어떻게 코드를 더 최적화 할 것인지 결정한다. 하지만 try/catch 블록같은 몇 몇 언어 feature는 아직 최적화를 지원하지 않는다. (try/catch 블록에 대한 해결책으로 함수 안에 non stable 코드를 작성하고 try 블럭에서 해당 함수를 호출하는 방법이 있다.)

__코드 최적화__ : V8은 de-optimization도 제공한다. Optimizing compiler는 Inline Cache로부터 서로 다른 type에 대한 최적화 가정을 한다. 만약 가정이 유효하지 않으면 de-optimization을 진행한다. 예를 들어서 생성된 hidden class가 예상한 것과 다르면 V8은 최적화 코드를 버리고 Inline Cache로 부터 type을 얻기 위해서 Full compiler로 복귀한다. 이 과정은 매우 느리기 때문에 함수들이 최적화 된 이후에는 함수를 (변수의 hidden class) 변경하지 않아야 한다. 
