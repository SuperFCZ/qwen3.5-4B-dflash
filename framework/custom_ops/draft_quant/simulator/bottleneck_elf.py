"""Read-only ELF and LLVM capability diagnostics. Never infer a load bias."""
import json
from pathlib import Path
import struct


def elf_layout(path):
    data=Path(path).read_bytes()
    def unpack(fmt,offset):
        size=struct.calcsize(fmt)
        if offset<0 or offset+size>len(data):raise ValueError('truncated ELF structure')
        return struct.unpack_from(fmt,data,offset)
    if len(data)<64 or data[:7]!=b'\x7fELF\x02\x01\x01':raise ValueError('requires ELF64 little-endian v1')
    kind,machine,version,entry,phoff,shoff,flags,ehsize,phsize,phnum,shsize,shnum,strings=unpack('<HHIQQQIHHHHHH',16)
    if version!=1 or ehsize!=64 or shsize!=64 or not shnum or strings>=shnum or (phnum and phsize!=56):
        raise ValueError('unsupported ELF header/extended section table')
    sections=[unpack('<IIQQQQIIQQ',shoff+i*shsize) for i in range(shnum)]
    name_section=sections[strings];offset,size=name_section[4:6]
    if offset+size>len(data):raise ValueError('truncated ELF string table')
    names=data[offset:offset+size];result=[]
    for s in sections:
        if s[0]>=len(names):raise ValueError('invalid ELF section name')
        stop=names.find(b'\0',s[0])
        if stop<0:raise ValueError('unterminated ELF section name')
        if s[1]!=8 and s[4]+s[5]>len(data):raise ValueError('section exceeds ELF file')
        result.append(dict(name=names[s[0]:stop].decode(),type=s[1],flags=s[2],address=s[3],offset=s[4],bytes=s[5]))
    loads=[]
    for i in range(phnum):
        ptype,pflags,offset,vaddr,paddr,filesz,memsz,align=unpack('<IIQQQQQQ',phoff+i*phsize)
        if ptype==1:
            if filesz>memsz or offset+filesz>len(data):raise ValueError('invalid LOAD segment')
            loads.append(dict(offset=offset,virtual_address=vaddr,physical_address=paddr,file_bytes=filesz,memory_bytes=memsz,flags=pflags,alignment=align))
    text=[s for s in result if s['name']=='.text']
    if len(text)!=1:raise ValueError('expected one .text section')
    return dict(status='ELF_LAYOUT_READ',elf_type=kind,machine=machine,machine_hex=hex(machine),entry=entry,
        text=text[0],load_segments=loads,debug_sections=[s for s in result if s['name'].startswith(('.debug','.rela.debug'))],
        executable_sections=[s for s in result if s['flags']&4],
        interpretation='ELF virtual addresses only; no runtime load-address correspondence is inferred.')


def address_space(layout,addresses):
    if layout.get('status')!='ELF_LAYOUT_READ' or layout.get('elf_type')!=2:
        return dict(status='UNRESOLVED_ADDRESS_SPACE',outside_count=len(addresses),reason='no supported ET_EXEC layout')
    def contains(address):
        return any(s['address']<=address<s['address']+s['bytes'] for s in layout['executable_sections']) and any(
            p['flags']&1 and p['virtual_address']<=address<p['virtual_address']+p['file_bytes'] for p in layout['load_segments'])
    outside=sorted(a for a in addresses if not contains(a))
    return dict(status='DIRECT_ELF_VIRTUAL_ADDRESSES' if addresses and not outside else 'UNRESOLVED_ADDRESS_SPACE',
        address_count=len(addresses),outside_count=len(outside),outside_samples=[hex(a) for a in outside[:20]],
        policy='Only unchanged addresses inside both executable section and executable LOAD are eligible. Never subtract a guessed base.')


def llvm_frames(text,addresses):
    # LLVM --output-style=JSON emits one object per stdin address. Match the
    # explicit Address keys; output order is not used as a PC association.
    decoder=json.JSONDecoder();objects=[];remaining=text.strip()
    while remaining:
        item,end=decoder.raw_decode(remaining);objects.extend(item if isinstance(item,list) else [item]);remaining=remaining[end:].lstrip()
    result={}
    for item in objects:
        address=item.get('Address')
        if not isinstance(address,str) or not address.startswith('0x'):raise ValueError('LLVM JSON has no explicit hexadecimal Address')
        number=int(address,16)
        if number in result:raise ValueError('duplicate LLVM address')
        frames=[]
        for frame in item.get('Symbol',[]):
            file,line=frame.get('FileName'),frame.get('Line')
            if isinstance(file,str) and file not in ('','??') and type(line) is int and line>0:
                frames.append(dict(file=file,line=line,function=frame.get('FunctionName','')))
        result[number]=frames
    if set(result)!=set(addresses):raise ValueError('LLVM returned another address set')
    return result
