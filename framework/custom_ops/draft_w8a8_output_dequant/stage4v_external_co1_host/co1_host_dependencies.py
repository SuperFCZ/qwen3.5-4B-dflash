"""Exact generated Op Host target dependencies, using the audited Make parsers."""
import re,shlex
from pathlib import Path
from co1_host_common import identity,verify,v
from raw_host_dependency_identity import make_dependencies,internal_dependencies,absolute


def collect(project,cann,reviewed):
 project=project.resolve(strict=True);cann=cann.resolve(strict=True)
 build=project/'build_out';target=build/'op_host/CMakeFiles/cust_optiling.dir'
 source=(project/'op_host/d_flash_external_co1_host_probe.cpp').resolve(strict=True)
 obj=(target/'d_flash_external_co1_host_probe.cpp.o').resolve(strict=True)
 if not obj.is_file() or obj.stat().st_size==0 or build not in obj.parents:raise ValueError('wrong/missing exact Op Host object')
 evidence=[]
 def read(p):
  p=p.resolve(strict=True)
  if build not in p.parents:raise ValueError('dependency evidence escaped build')
  item=identity(p)
  if item['bytes']>8*1024*1024:raise ValueError('dependency evidence too large')
  text=p.read_text();verify(item);evidence.append(item);return text
 cache=read(build/'CMakeCache.txt');gens=re.findall(r'^CMAKE_GENERATOR:INTERNAL=(.*)$',cache,re.M)
 if len(gens)!=1:raise ValueError('missing CMake generator')
 dep=Path(str(obj)+'.d')
 if dep.is_file():
  deps=make_dependencies(read(dep),build,obj);route='COMPILER_DEPFILE'
 elif gens==['Unix Makefiles']:
  blocks=re.findall(r'\bset\s*\(\s*CMAKE_DEPENDS_CHECK_CXX\b(.*?)\)',read(target/'DependInfo.cmake'),re.S)
  if len(blocks)!=1:raise ValueError('missing source/object mapping')
  fields=shlex.split(blocks[0],comments=True)
  if len(fields)%2 or any('$' in x or ';' in x for x in fields):raise ValueError('unsupported mapping')
  pairs=[(absolute(fields[i],build).resolve(),absolute(fields[i+1],build).resolve()) for i in range(0,len(fields),2)]
  if pairs.count((source,obj))!=1 or sum(o==obj for _,o in pairs)!=1:raise ValueError('Op Host source/object mismatch')
  deps=make_dependencies(read(target/'depend.make'),build,obj)
  if deps!=internal_dependencies(read(target/'depend.internal'),build,obj):raise ValueError('dependency record disagreement')
  route='CMAKE_MAKE_IMPLICIT_DEPENDENCIES'
 else:raise ValueError('no target-bound dependencies for generator '+gens[0])
 expected=verify(reviewed).resolve(strict=True)
 if cann not in expected.parents:raise ValueError('reviewed platform outside CANN')
 platform=[];aliases=[];source_seen=False
 for alias in sorted(deps):
  real=alias.resolve(strict=True)
  if alias.name=='platform_ascendc.h':
   if real!=expected:raise ValueError('compiled wrong platform header')
   platform.append(str(alias))
  if cann in alias.parents and cann not in real.parents:raise ValueError('SDK dependency escapes CANN')
  if real==source:source_seen=True
  if cann in real.parents or project in real.parents:aliases.append(dict(alias=str(alias),file=identity(real)))
 if not source_seen or not platform or identity(expected)!=reviewed:raise ValueError('missing source/reviewed platform dependency')
 result=dict(status='FORMAL_OP_HOST_DEPENDENCIES_BOUND',generator=gens[0],route=route,evidence=evidence,object=identity(obj),source=identity(source),reviewed=reviewed,platform_aliases=platform,dependencies=aliases)
 verify_binding(result);return result


def verify_binding(record):
 v.verify_tree(record)
 for item in record['dependencies']:
  if identity(item['alias'])!=item['file']:raise ValueError('compiled dependency alias changed')
