"""Release one permitted core run after frozen inputs and corrected timing."""
import os
os.environ['ST9_NAMESPACE']='st9t'
from src.st9.gpu import *
import argparse


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True,choices=['PhysNet','FactorizePhys'])
    parser.add_argument('--arm',required=True);parser.add_argument('--seed',type=int,required=True);parser.add_argument('--gpu',type=int,choices=[0,1],default=1)
    args=parser.parse_args();model=args.model;arm=args.arm;seed=args.seed
    check('Only prescribed core training seeds',seed in [20261131,20261132,20261133])
    check('Only prescribed core arms',arm in (['0','i','vii','ii-h','iv-h'] if model=='PhysNet' else ['0','i','ii-h']))
    settings=REPORTS/'st9t_settings.json';check('Core training settings committed',sha(settings) in (ROOT/'reports/ST4_INSTR03_PREFIX.md').read_text())
    check('CPU training harness passed',json.loads((ROOT/'reports/st9s/st9s_harness_tests.json').read_text())['passed'])
    projection=ROOT/'reports/st9s/st9s_core_projection_corrected.json'
    check('Corrected core projection committed',sha(projection) in (ROOT/'reports/ST4_INSTR03_PREFIX.md').read_text())
    check('Core24 fits first-claim budget',json.loads(projection.read_text())['B_T_resource_gate'])
    run_id='st9t_'+model+'_'+arm+'_'+str(seed);request=dict(run_id=run_id,model=model,arm=arm,seed=seed,physical_GPU=args.gpu)
    check('Never repeat a core run',not (ROOT/'reports/st9/gpu_accounts'/(run_id+'.json')).exists())
    if arm not in ['0','vii']:
        if arm in ['ii-h','iv-h']:
            check('Both ED2-W gates passed before relevant training',
                json.loads((ROOT/'reports/st9e/st9e_component_summary.json').read_text())['E3R_a_passed'] and
                json.loads((ROOT/'reports/st9e/st9e_brange_selection.json').read_text())['gate_passed'])
        lag=ROOT/'reports/st9g'/('st9g_'+model+'_'+arm+'_lag_check.json')
        check('Arm/view stored tensor lag release',json.loads(lag.read_text())['training_allowed'])
        pool=ROOT/'outputs'/('st9g_pool_'+model+'_'+arm+'_manifest.csv')
        request.update(pool_manifest=str(pool),pool_manifest_sha256=sha(pool),lag_check_sha256=sha(lag))
    destination=REPORTS/(run_id+'_request.json');check('New fixed run request only',not destination.exists());save_json(destination,request)
    result=run_gpu(run_id,'B-T+L-dev',args.gpu,'repro','src.st9t.worker',args=[str(destination)])
    outcome=json.loads((REPORTS/(run_id+'_result.json')).read_text())
    if outcome['status']=='complete':
        check('Final research checkpoint retained unchanged',sha(outcome['checkpoint_path'])==outcome['checkpoint_sha256'])
    runs=rows('outputs/st9t_runs.csv') if (ROOT/'outputs/st9t_runs.csv').exists() else []
    row={k:outcome[k] for k in ['run_id','model','arm','seed','status','completed_steps','completed_epochs','final_loss','initial_weights_sha256','checkpoint_path','checkpoint_sha256','checkpoint_bytes','peak_GPU_allocated_bytes']}
    row.update(physical_GPU=args.gpu,GPU_seconds=result['GPU_seconds'],evaluation_calls=0,validation_calls=0);runs.append(row);write_rows('outputs/st9t_runs.csv',runs)
    counts=outcome['exposure'];n=counts['original']+counts['augmented']
    record=dict(run_id=run_id,model=model,arm=arm,seed=seed,original_exposures=counts['original'],augmented_exposures=counts['augmented'],
        high_original_exposures=counts['high_original'],high_augmented_exposures=counts['high_augmented'],
        high_original_fraction=counts['high_original']/counts['original'] if counts['original'] else None,
        high_augmented_fraction=counts['high_augmented']/counts['augmented'] if counts['augmented'] else None,
        high_total_fraction=(counts['high_original']+counts['high_augmented'])/n if n else None)
    exposure=rows('outputs/st9t_exposure.csv') if (ROOT/'outputs/st9t_exposure.csv').exists() else []
    exposure.append(record);write_rows('outputs/st9t_exposure.csv',exposure)
    return dict(**row,**{k:v for k,v in record.items() if k not in row},research_checkpoint_retained=bool(outcome['checkpoint_sha256']))


if __name__=='__main__':
    arguments=argparse.ArgumentParser(add_help=False);arguments.add_argument('--model');arguments.add_argument('--arm');arguments.add_argument('--seed');args,_=arguments.parse_known_args()
    finish('run_'+str(args.model)+'_'+str(args.arm)+'_'+str(args.seed),main,resource='GPU_training')
