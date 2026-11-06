"""Fresh-process test harness; synthetic CMake evidence, no CANN/NPU invocation."""
import importlib,json,shlex,sys
from pathlib import Path
DIAG=Path(__file__).resolve().parents[1]
HISTORY=DIAG.parent/'stage4i'
REPO=DIAG.parents[3]
order,action,folder=sys.argv[1:]
if order=='history_first':
 sys.path.insert(0,str(HISTORY));importlib.import_module('run_stage4i')
elif order=='models_first':
 sys.path.insert(0,str(REPO));application_models=importlib.import_module('models')
sys.path.insert(0,str(DIAG))
if order=='validator_first':importlib.import_module('raw_host_dependency_identity')
host=importlib.import_module('raw_host')
validator=importlib.import_module('raw_host_dependency_identity')
legacy=importlib.import_module('dependency_identity')
assert Path(legacy.__file__).resolve()==(HISTORY/'dependency_identity.py').resolve()
assert host.compiled_dependencies is validator.compiled_dependencies
assert host.verify_dependency_binding is validator.verify_dependency_binding
assert host.compiled_dependencies is not legacy.compiled_dependencies
binding=host.bound_validator(host.sources())
assert binding['file']['path']==str((DIAG/'raw_host_dependency_identity.py').resolve())
for name in ('evidence','sdk_review','raw_resources'):
 assert Path(importlib.import_module(name).__file__).resolve()==(DIAG/(name+'.py')).resolve()
if order=='models_first':
 analyzer=importlib.import_module('analyze');numeric=importlib.import_module('stage4v_rounding_models')
 assert analyzer.analyze is numeric.analyze and sys.modules['models'] is application_models
 assert Path(application_models.__file__).resolve()==REPO/'models/__init__.py'
if action=='imports':
 print(json.dumps(binding));raise SystemExit(0)
if action=='wrong_function':
 host.compiled_dependencies=legacy.compiled_dependencies
 try:host.dependency_validator_identity()
 except ValueError as e:
  assert 'function origin' in str(e);print('REJECTED_FOREIGN_FUNCTION');raise SystemExit(0)
 raise AssertionError('foreign function accepted')
if action=='wrong_module':
 validator.__file__=legacy.__file__
 try:host.dependency_validator_identity()
 except ValueError as e:
  assert 'outside Diagnostics' in str(e);print('REJECTED_FOREIGN_MODULE');raise SystemExit(0)
 raise AssertionError('foreign module accepted')

root=Path(folder).resolve();root.mkdir()
cann=root/'sdk space';real=cann/'actual/platform_ascendc.h';real.parent.mkdir(parents=True);real.write_text('reviewed synthetic header')
alias=cann/'include/tiling/platform/platform_ascendc.h';alias.parent.mkdir(parents=True);alias.symlink_to(real)
reviewed=host.identity(real)
build=root/'build';target_name='stage4i_host' if action=='wrong_target' else 'raw_int32_host'
target=build/f'CMakeFiles/{target_name}.dir';target.mkdir(parents=True)
obj=target/'probe.cpp.o';obj.write_bytes(bytes(197280)) # NOT an ELF/compile success fixture
source=root/'source/probe.cpp';source.parent.mkdir();source.write_text('// synthetic compile dependency fixture')
obj_rel=f'CMakeFiles/{target_name}.dir/probe.cpp.o'
argv=['/usr/bin/c++','-I'+str(cann/'include'),'-O2','-std=gnu++17','-o',obj_rel,'-c',str(source)]
(build/'compile_commands.json').write_text(json.dumps([dict(directory=str(build),file=str(source),command=shlex.join(argv))]))
(build/'CMakeCache.txt').write_text('CMAKE_MAKE_PROGRAM:FILEPATH=/usr/bin/make\nCMAKE_GENERATOR:INTERNAL=Unix Makefiles\n')
(target/'DependInfo.cmake').write_text(f'set(CMAKE_DEPENDS_CHECK_CXX\n "{source}" "{obj}"\n)\n')
def escaped(path):return str(path).replace('\\','\\\\').replace(' ','\\ ').replace('$','$$').replace('#','\\#').replace(':','\\:')
paths=[source,alias]
(target/'depend.make').write_text('# CMake3.16 fixture\n'+''.join(obj_rel+': '+escaped(p)+'\n' for p in paths))
(target/'depend.internal').write_text('# CMake3.16 fixture\n'+obj_rel+'\n'+''.join(' '+str(p)+'\n' for p in paths))
if action=='depfile':
 Path(str(obj)+'.d').write_text(obj_rel+': '+escaped(source)+' \\\n '+escaped(alias)+'\n')
elif action=='tampered_header':real.write_text('tampered')
elif action=='disagree':(target/'depend.internal').write_text(obj_rel+'\n '+str(source)+'\n')
elif action=='missing':(target/'DependInfo.cmake').unlink()
elif action=='escaped_header':
 outside=root/'outside/platform_ascendc.h';outside.parent.mkdir();outside.write_text(real.read_text());alias.unlink();alias.symlink_to(outside)
elif action=='wrong_mapping':
 f=target/'DependInfo.cmake';f.write_text(f.read_text().replace('source/probe.cpp','source/other.cpp'))
negative=action in ('wrong_target','tampered_header','disagree','missing','escaped_header','wrong_mapping')
try:
 result=host.compiled_dependencies(build,cann,source,reviewed)
except (ValueError,OSError) as e:
 if not negative:raise
 if action=='wrong_target':assert 'wrong/missing probe object' in str(e)
 print('REJECTED '+action+': '+str(e));raise SystemExit(0)
assert not negative,'invalid evidence was accepted'
assert result['generator']=='Unix Makefiles'
assert result['object']['path']==str(obj) and result['object']['bytes']==197280
assert result['headers']==[reviewed]
assert result['route']==('COMPILER_DEPFILE' if action=='depfile' else 'CMAKE_UNIX_MAKEFILES_IMPLICIT_DEPENDENCIES')
if action!='depfile':
 assert {Path(x['path']).name for x in result['evidence']}=={'compile_commands.json','CMakeCache.txt','DependInfo.cmake','depend.make','depend.internal'}
host.verify_dependency_binding(result)
if action in ('tampered_after','retarget_after'):
 if action=='tampered_after':
  f=target/'depend.make';f.write_text(f.read_text()+'# changed after acceptance\n')
 else:
  other=cann/'other.h';other.write_text(real.read_text());alias.unlink();alias.symlink_to(other)
 try:host.verify_dependency_binding(result)
 except ValueError:
  print('REJECTED '+action);raise SystemExit(0)
 raise AssertionError('post-check alteration accepted')
print('ACCEPTED '+action)
