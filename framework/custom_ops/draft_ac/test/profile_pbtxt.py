"""Streaming TextFormat structural reader; tensor strings are never materialized.

No GE schema import (which could load torch_npu). Unsupported/malformed syntax
raises with location instead of guessing keys. See protobuf Text Format spec.
"""
import re
import math

PAYLOAD={'data','tensor_content','weight_data'}
IDENT=re.compile(r'[A-Za-z_][A-Za-z_0-9]*\Z')
BARE=re.compile(r'''[^\s{}\[\]<>:;,"'#]+''')
BODY={q:re.compile(r'(?:\\.|[^'+q+r'\\\n])*') for q in ('"',"'")}


class Lexer:
    def __init__(self,stream):
        self.stream=stream;self.buf='';self.pos=0;self.line=1;self.column=1;self.look=None
    def error(self,message):
        raise ValueError(f'pbtxt line {self.line} column {self.column}: {message}')
    def char(self):
        if self.pos==len(self.buf):self.buf=self.stream.read(65536);self.pos=0
        return self.buf[self.pos:self.pos+1]
    def advance(self,n):
        piece=self.buf[self.pos:self.pos+n];self.pos+=n
        if '\n' in piece:self.line+=piece.count('\n');self.column=len(piece.rsplit('\n',1)[1])+1
        else:self.column+=n
    def string(self):
        quote=self.char();self.advance(1);parts=[];length=0
        while True:
            c=self.char()
            if not c:self.error('unterminated string')
            if c==quote:self.advance(1);break
            if c=='\n':self.error('unescaped newline in string')
            match=BODY[quote].match(self.buf,self.pos);piece=match.group()
            if piece:
                self.advance(len(piece))
            else:
                # A backslash can straddle a chunk boundary.
                if c!='\\':self.error('invalid string token')
                self.advance(1);escaped=self.char()
                if not escaped or escaped=='\n':self.error('unterminated escape')
                piece='\\'+escaped;self.advance(1)
            length+=len(piece)
            if length<=4096:parts.append(piece)
        if length>4096:return b'<omitted oversized string>'
        raw=''.join(parts);result=bytearray();i=0
        simple={'a':7,'b':8,'f':12,'n':10,'r':13,'t':9,'v':11,'?':63,'\\':92,"'":39,'"':34}
        while i<len(raw):
            if raw[i]!='\\':result.extend(raw[i].encode('utf-8'));i+=1;continue
            i+=1;c=raw[i];i+=1
            if c in simple:result.append(simple[c])
            elif c in '01234567':
                m=re.match('[0-7]{0,2}',raw[i:]);digits=c+m.group();i+=len(m.group())
                n=int(digits,8)
                if n>255:self.error('octal escape outside byte range')
                result.append(n)
            elif c in 'xuU':
                limit=2 if c=='x' else 4 if c=='u' else 8
                m=re.match('[0-9a-fA-F]{1,'+str(limit)+'}',raw[i:])
                if not m or (c!='x' and len(m.group())!=limit):self.error('invalid hex/unicode escape')
                i+=len(m.group());n=int(m.group(),16)
                try:result.extend(bytes([n]) if c=='x' else chr(n).encode('utf-8'))
                except (ValueError,UnicodeError):self.error('invalid Unicode scalar')
            else:self.error('unknown escape')
        return bytes(result)
    def next(self):
        while True:
            c=self.char()
            if not c:return ('EOF','')
            if c.isspace():self.advance(1);continue
            if c=='#':
                while self.char() and self.char()!='\n':
                    end=self.buf.find('\n',self.pos)
                    self.advance((end if end>=0 else len(self.buf))-self.pos)
                continue
            break
        if c in ('"',"'"):return ('STRING',self.string())
        if c in '{}[]<>:;,':self.advance(1);return (c,c)
        parts=[]
        while self.char():
            match=BARE.match(self.buf,self.pos)
            if not match:break
            part=match.group();parts.append(part);self.advance(len(part))
            if sum(map(len,parts))>4096:self.error('oversized bare token')
            if self.pos<len(self.buf):break
        if not parts:self.error('unexpected character')
        return ('BARE',''.join(parts))
    def peek(self):
        if self.look is None:self.look=self.next()
        return self.look
    def pop(self):
        value=self.peek();self.look=None;return value


def scalar(lex):
    kind,value=lex.pop()
    if kind=='STRING':
        while lex.peek()[0]=='STRING':
            other=lex.pop()[1];value=value+other if len(value)+len(other)<=4096 else b'<omitted oversized string>'
        # TextFormat concatenates adjacent byte literals before UTF-8 decoding.
        # Arbitrary non-UTF8 bytes remain visible as escapes, never as syntax.
        return value.decode('utf-8',errors='backslashreplace')
    if kind!='BARE':lex.error('expected scalar')
    if value in ('-','+'):
        kind,suffix=lex.pop()
        if kind!='BARE':lex.error('expected number after sign')
        value+=suffix
    if re.fullmatch(r'[+-]?(0[xX][0-9a-fA-F]+|[0-9]+)',value):
        unsigned=value.lstrip('+-');base=16 if unsigned.lower().startswith('0x') else 8 if len(unsigned)>1 and unsigned.startswith('0') else 10
        try:return int(value,base)
        except ValueError:lex.error('invalid integer')
    if re.fullmatch(r'[+-]?(?:(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?[fF]?)',value):
        number=float(value.rstrip('fF'))
        return number if math.isfinite(number) else value
    if IDENT.fullmatch(value) or value.lower() in ('-inf','+inf','-nan','+nan'):return value
    lex.error('invalid scalar token '+repr(value[:80]))


def read(stream):
    lex=Lexer(stream)
    def message(close='EOF',keep=True,depth=0):
        if depth>128:lex.error('message nesting exceeds 128')
        obj={}
        while lex.peek()[0]!=close:
            kind,name=lex.pop()
            if kind=='EOF':lex.error('unterminated message')
            if kind in (',',';'):continue
            if kind=='[':
                # Expanded Any/extension names are field names, not value lists.
                parts=[]
                while lex.peek()[0]!=']':
                    k,v=lex.pop()
                    if k!='BARE':lex.error('invalid extension/Any field name')
                    parts.append(v)
                lex.pop();name='['+''.join(parts)+']'
                if not parts:lex.error('empty extension field')
            elif kind!='BARE' or not IDENT.fullmatch(name):lex.error('expected field name, got '+repr(name[:80]))
            colon=lex.peek()[0]==':'
            if colon:lex.pop()
            retain=keep and name not in PAYLOAD
            def value():
                opening=lex.peek()[0]
                if opening in ('{','<'):
                    lex.pop();return message('}' if opening=='{' else '>',retain,depth+1)
                if not colon:lex.error('scalar field requires colon')
                return scalar(lex)
            items=[]
            if lex.peek()[0]=='[':
                lex.pop()
                while lex.peek()[0]!=']':
                    item=value()
                    if retain:items.append(item)
                    if lex.peek()[0]==',':lex.pop()
                    elif lex.peek()[0]!=']':lex.error('list requires comma or closing bracket')
                lex.pop()
            else:
                item=value()
                if retain:items.append(item)
            if retain:obj.setdefault(name,[]).extend(items)
        lex.pop();return obj
    return message()
