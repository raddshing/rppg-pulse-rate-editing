"""Original training backends; UBFC changes only the prescribed total-step boundary."""
from src.st30p.ops import ROOT
import ast
import copy


def initialize(name,config,seed):
    if name=='RhythmFormer':
        from src.st12c import rhythm_backend_v2 as backend
        args=(config,seed)
    else:
        from src.st9s import backend
        args=('FactorizePhys' if name=='FPm' else name,config,seed,'cuda:0')
    if config.get('dataset')!='UBFC':return backend.initialize(*args)
    # Full-step schedule is explicitly fixed at5580/4470 even when the final
    # UBFC epoch is partial. All other statements and defaults remain literal.
    path=ROOT/('src/st12c/rhythm_backend_v2.py' if name=='RhythmFormer' else 'src/st9s/backend.py')
    tree=ast.parse(path.read_text());node=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='initialize'))
    changes=0
    for n in ast.walk(node):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='OneCycleLR':
            n.keywords=[k for k in n.keywords if k.arg not in ['epochs','steps_per_epoch']]
            n.keywords.append(ast.keyword(arg='total_steps',value=ast.parse("int(settings['total_steps'])",mode='eval').body));changes+=1
    assert changes==1
    scope=dict(vars(backend));exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path)+':ST6_UBFC_total_steps','exec'),scope)
    return scope['initialize'](*args)


def step(name,model,criterion,optimizer,scheduler,x,y,epoch):
    if name=='RhythmFormer':
        from src.st12c.rhythm_backend_v2 import step as inherited
        return inherited(model,criterion,optimizer,scheduler,x,y,epoch)
    from src.st9s.backend import step as inherited
    return inherited('FactorizePhys' if name=='FPm' else name,model,criterion,optimizer,scheduler,x,y)
