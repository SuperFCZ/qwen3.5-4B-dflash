"""Read target-bound dependency records; never invoke CMake, Make or a compiler."""
import json,os,re,shlex
from pathlib import Path
from layout import identity,verify,verify_tree


def absolute(value,cwd):
 p=Path(value)
 return Path(os.path.abspath(p if p.is_absolute() else cwd/p))


def make_words(line):
 # Deliberately limited Make dependency syntax. No expansion/evaluation/recipes.
 out=[];word='';i=0
 while i<len(line):
  c=line[i]
  if c=='\\':
   i+=1
   if i==len(line):raise ValueError('dangling Make escape')
   word+=line[i]
  elif c=='$':
   if i+1>=len(line) or line[i+1]!='$':raise ValueError('unsupported Make expansion')
   word+='$';i+=1
  elif c=='#':break
  elif c==':' or c.isspace():
   if word:out.append(word);word=''
   if c==':':out.append(None)
  else:word+=c
  i+=1
 if word:out.append(word)
 return out


def make_dependencies(text,cwd,obj):
 found=False;deps=set()
 for line in re.sub(r'\\\r?\n',' ',text).splitlines():
  if not line.strip() or line.lstrip().startswith('#'):continue
  if line.startswith('\t'):raise ValueError('recipe in dependency-only file')
  tokens=make_words(line)
  if tokens.count(None)!=1:raise ValueError('unsupported dependency rule')
  i=tokens.index(None);targets=tokens[:i]
  if not targets:raise ValueError('missing dependency target')
  if obj in {absolute(t,cwd).resolve() for t in targets}:
   found=True;deps.update(absolute(t,cwd) for t in tokens[i+1:])
 if not found or not deps:raise ValueError('no dependencies for exact probe object')
 return deps


def internal_dependencies(text,cwd,obj):
 current=None;deps=set();seen=0
 for line in text.splitlines():
  if not line.strip() or line.startswith('#'):continue
  if not line[0].isspace():
   current=absolute(line,cwd).resolve()
   if current==obj:seen+=1
  elif current==obj:deps.add(absolute(line.lstrip(),cwd))
 if seen!=1 or not deps:raise ValueError('missing/duplicate probe block in depend.internal')
 return deps


def compiled_dependencies(build,cann,source,reviewed):
 build=build.resolve(strict=True);cann=cann.resolve(strict=True);source=source.resolve(strict=True)
 expected=verify(reviewed).resolve(strict=True)
 if cann not in expected.parents:raise ValueError('reviewed platform header outside CANN_ROOT')
 evidence=[]
 def read(path):
  path=Path(path).resolve(strict=True)
  if build not in path.parents:raise ValueError('dependency evidence escapes build directory')
  item=identity(path)
  if item['bytes']>8*1024*1024:raise ValueError('dependency evidence exceeds 8 MiB')
  text=path.read_text(encoding='utf-8');verify(item);evidence.append(item);return text
 commands=json.loads(read(build/'compile_commands.json'))
 entries=[e for e in commands if absolute(e['file'],Path(e['directory'])).resolve()==source]
 if len(entries)!=1:raise ValueError('requires one exact probe compile command')
 entry=entries[0];cwd=Path(entry['directory']).resolve(strict=True)
 if cwd!=build:raise ValueError('unexpected probe compile working directory')
 argv=entry.get('arguments') or shlex.split(entry['command'])
 def operand(flag):
  if argv.count(flag)!=1 or argv.index(flag)+1>=len(argv):raise ValueError('ambiguous/missing compile flag '+flag)
  return argv[argv.index(flag)+1]
 if absolute(operand('-c'),cwd).resolve()!=source:raise ValueError('compile source does not match probe')
 obj=absolute(operand('-o'),cwd).resolve(strict=True)
 target=build/'CMakeFiles/stage4i_host.dir'
 if obj!=target/'probe.cpp.o' or not obj.is_file() or obj.stat().st_size==0:raise ValueError('wrong/missing probe object')
 if entry.get('output') and absolute(entry['output'],cwd).resolve()!=obj:raise ValueError('compile output field differs')
 if build not in obj.parents:raise ValueError('probe object escaped build directory')
 cache=read(build/'CMakeCache.txt')
 generators=re.findall(r'^CMAKE_GENERATOR:INTERNAL=(.*)$',cache,re.M)
 if len(generators)!=1:raise ValueError('missing/ambiguous CMake generator')
 generator=generators[0]
 dep=absolute(operand('-MF'),cwd) if '-MF' in argv else Path(str(obj)+'.d')
 if dep.is_file():
  deps=make_dependencies(read(dep),cwd,obj);route='COMPILER_DEPFILE'
 elif generator=='Unix Makefiles':
  # Actual CMake 3.16 server records; cross-check both exact-target lists.
  info=read(target/'DependInfo.cmake')
  blocks=re.findall(r'\bset\s*\(\s*CMAKE_DEPENDS_CHECK_CXX\b(.*?)\)',info,re.S)
  if len(blocks)!=1:raise ValueError('missing/ambiguous CMAKE_DEPENDS_CHECK_CXX')
  fields=shlex.split(blocks[0],comments=True)
  if len(fields)%2 or any('$' in x or ';' in x for x in fields):raise ValueError('unsupported source/object mapping')
  pairs=[(absolute(fields[i],cwd).resolve(),absolute(fields[i+1],cwd).resolve()) for i in range(0,len(fields),2)]
  if pairs.count((source,obj))!=1 or sum(o==obj for _,o in pairs)!=1:raise ValueError('DependInfo source/object mismatch')
  deps=make_dependencies(read(target/'depend.make'),cwd,obj)
  internal=internal_dependencies(read(target/'depend.internal'),cwd,obj)
  if deps!=internal:raise ValueError('Make/internal dependency lists disagree')
  route='CMAKE_UNIX_MAKEFILES_IMPLICIT_DEPENDENCIES'
 else:raise ValueError('no supported target-bound dependency evidence for '+generator)
 if source not in {p.resolve(strict=True) for p in deps}:raise ValueError('probe source missing from target dependencies')
 sdk={};aliases=[];platform_paths=[]
 for raw in sorted(deps):
  real=raw.resolve(strict=True)
  if cann in raw.parents and cann not in real.parents:raise ValueError('SDK dependency escapes CANN_ROOT: '+str(raw))
  if raw.name=='platform_ascendc.h':
   if real!=expected:raise ValueError('compiled platform header is not the reviewed header: '+str(raw))
   platform_paths.append(str(raw))
  if real==expected and cann in real.parents:platform_paths.append(str(raw))
  if cann in real.parents:
   item=identity(real);sdk[str(real)]=item;aliases.append(dict(path_as_recorded=str(raw),file=item))
 if not platform_paths or sdk.get(str(expected))!=reviewed:raise ValueError('reviewed platform header absent or identity mismatch')
 result=dict(status='COMPILED_PLATFORM_HEADER_IDENTITY_VERIFIED',route=route,generator=generator,
  compile_directory=str(cwd),source=identity(source),object=identity(obj),evidence=evidence,
  reviewed_header=reviewed,platform_dependency_paths=sorted(set(platform_paths)),
  sdk_aliases=aliases,headers=[sdk[k] for k in sorted(sdk)],
  scope='Target-specific generated build dependencies, not inferred from include search directories')
 verify_dependency_binding(result);return result


def verify_dependency_binding(binding):
 verify_tree(binding)
 for alias in binding['sdk_aliases']:
  if identity(alias['path_as_recorded'])!=alias['file']:raise ValueError('SDK dependency alias identity changed')
