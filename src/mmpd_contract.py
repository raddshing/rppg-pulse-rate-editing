"""Fixed instruction-02 metadata and approval checks; no prediction reads."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBSETS = ('S_full', 'S_simple', 'S_MM-E', 'S_MM+E')
INFO_ALL = dict(LIGHT=[1, 2, 3, 4], MOTION=[1, 2, 3, 4], EXERCISE=[1, 2],
                SKIN_COLOR=[3, 4, 5, 6], GENDER=[1, 2], GLASSER=[1, 2],
                HAIR_COVER=[1, 2], MAKEUP=[1, 2])


def read_csv(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def approved_metadata(root=ROOT):
    root = Path(root)
    paths = json.loads((root / 'paths.yaml').read_text())
    split_path = root / 'data/splits/mmpd_split_st1.json'
    receipt = json.loads((root / 'reports/st1_instruction_02_approval.json').read_text())
    if split_path.stat().st_mode & 0o222:
        raise ValueError('Instruction-02 split must be frozen read-only')
    digest = hashlib.sha256(split_path.read_bytes()).hexdigest()
    if receipt.get('split_artifact_sha256') != digest or receipt.get('split_independent_verification', {}).get('status') != 'passed':
        raise ValueError('Instruction-02 split approval or independent verification mismatch')
    split = json.loads(split_path.read_text())
    a, b = set(split['A']), set(split['B'])
    if a & b or len(a) != 11 or len(b) != 22 or len(split['A']) != 11 or len(split['B']) != 22:
        raise ValueError('Invalid instruction-02 split')
    index_path = root / 'data/index/mmpd_index_st1.csv'
    condition_path = root / 'data/index/mmpd_conditions_st1.csv'
    if hashlib.sha256(index_path.read_bytes()).hexdigest() != split.get('input_index_sha256'):
        raise ValueError('Current MMPD index hash differs from the approved split input')
    if (condition_path.stat().st_mode & 0o222 or
            hashlib.sha256(condition_path.read_bytes()).hexdigest() != receipt.get('mmpd_conditions_sha256')):
        raise ValueError('Current MMPD conditions differ from the frozen pre-result receipt')
    index = read_csv(index_path)
    conditions = read_csv(condition_path)
    indexed = {row['recording_id']: row for row in index}
    mapped = {row['recording_id']: row for row in conditions}
    if len(index) != 655 or len(indexed) != 655 or len(conditions) != 655 or set(indexed) != set(mapped):
        raise ValueError('Instruction-02 requires all 655 metadata records')
    if {row['subject_id'] for row in index} != a | b:
        raise ValueError('Split subject coverage differs from the index')
    for row in conditions:
        if row['subject_id'] != indexed[row['recording_id']]['subject_id'] or row['split'] != ('A' if row['subject_id'] in a else 'B'):
            raise ValueError('Condition identity differs from frozen metadata')
        codes = {key: int(row[key + '_code']) for key in ('light', 'motion', 'exercise', 'skin_color')}
        if int(row['original_skin_color']) != codes['skin_color']:
            raise ValueError('Subset skin code must equal the original recording label')
        skinlight = codes['light'] in (1, 2, 3) and codes['skin_color'] == 3
        expected = dict(S_full=True, S_simple=skinlight and codes['motion'] == 1 and codes['exercise'] == 2,
                        **{'S_MM-E': skinlight and codes['motion'] in (1, 2, 3, 4) and codes['exercise'] == 2,
                           'S_MM+E': skinlight and codes['motion'] in (1, 2, 3, 4)})
        for key, value in expected.items():
            if row[key] not in ('True', 'False') or (row[key] == 'True') != value:
                raise ValueError('Fixed subset metadata disagrees with original condition codes')
            row[key] = value
    return paths, split, indexed, mapped


def registered_run(run_id, root=ROOT, require_success=True):
    root=Path(root)
    instruction03=run_id.startswith('st1_mmpd_m03_')
    plan_path=root/('configs/st1_instruction03/m/plan.csv' if instruction03 else 'configs/st1_mmpd/plan.csv')
    planned = [row for row in read_csv(plan_path) if row['run_id'] == run_id]
    native_run_id=planned[0].get('native_run_id',run_id) if len(planned)==1 else run_id
    registered = [row for row in read_csv(root / 'runs/st1_runs.csv') if row['run_id'] == native_run_id]
    if len(planned)==1 and not registered and instruction03 and planned[0].get('signature')=='S5':
        plan=planned[0]
        manifest,parts=s5_batches_metadata(root)
        selected=[row for row in manifest['native_rows'] if row['canonical_run_id']==run_id]
        records=[registered_s5_part(row,root) for row in selected]
        if len(records)!=len(parts):raise ValueError('S5 canonical result requires every fixed successful batch')
        record=dict(records[0],run_id=run_id,config_path=plan['config_path'],config_sha256=plan['config_sha256'],
                    started_utc=min(r['started_utc'] for r in records),ended_utc=max(r['ended_utc'] for r in records),
                    source_native_run_ids=[r['run_id'] for r in records],derived_batch_alias=True)
        return plan,record
    if len(planned) != 1 or len(registered) != 1:
        raise ValueError('MMPD operation requires exactly one fixed-plan and registry row')
    plan, record = planned[0], registered[0]
    grouped_s8=instruction03 and plan.get('signature')=='S8'
    if instruction03:
        approval=json.loads((root/'reports/st1_instruction_03_approval.json').read_text())
        if not approval or not json.loads((root/'reports/instruction03/m0_parity.json').read_text()).get('C_allowed'):
            raise ValueError('Instruction03 M runs require approval and passed M0 parity')
        if plan.get('WP') not in ('MCPU','MGPU') or plan.get('signature') not in ('S8','S1','S3','S6','S5','S4','S2','S7'):
            raise ValueError('Instruction03 M account or signature differs from the fixed scope')
        if grouped_s8:
            if (native_run_id!='st1_mmpd_m03_s8_unsup7_s100' or plan['model'] not in ('CHROM','POS')
                    or record.get('model')!='UNSUP7' or plan['WP']!='MCPU' or plan['kind']!='OTHER'
                    or plan['native_mode']!='NAT-all' or plan.get('native_output_file')!=plan['model']+'_MMPD_outputs.pickle'):
                raise ValueError('S8 alias must name CHROM/POS from the single official seven-method CPU process')
        elif native_run_id!=run_id:
            raise ValueError('Only S8 may share a native process')
    if require_success and record['exit_code'] != '0':
        raise ValueError('MMPD operation requires successful registered inference')
    keys=('WP','kind','train_source','target','seed','config_path','checkpoint_path') if grouped_s8 else (
          'WP','kind','model','train_source','target','seed','config_path','checkpoint_path')
    if plan['target'] != 'MMPD' or any(plan[key] != record[key] for key in keys):
        raise ValueError('MMPD identity differs from the fixed plan')
    if plan['checkpoint_sha256'] != record['checkpoint_sha256']:
        raise ValueError('MMPD checkpoint hash differs from the fixed plan')
    config_path = Path(plan['config_path'])
    if not config_path.is_file() or hashlib.sha256(config_path.read_bytes()).hexdigest() != record.get('config_sha256'):
        raise ValueError('Current MMPD configuration differs from its successful registry hash')
    if plan['checkpoint_path']:
        checkpoint = Path(plan['checkpoint_path'])
        if not checkpoint.is_file() or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != plan['checkpoint_sha256']:
            raise ValueError('Current MMPD checkpoint differs from its planned registered hash')
    elif plan['checkpoint_sha256']:
        raise ValueError('An untrained MMPD method cannot carry a checkpoint hash')
    return plan, record


def s5_batches_metadata(root=ROOT):
    """Narrow fixed S5 disk-batch contract; metadata only, never predictions."""
    import yaml
    root=Path(root)
    paths,_,index,_=approved_metadata(root)
    manifest_path=root/'reports/instruction03/s5_batches.json'
    if manifest_path.is_symlink() or manifest_path.stat().st_mode & 0o222:
        raise ValueError('S5 batch declaration must be fixed read-only before native execution')
    manifest=json.loads(manifest_path.read_text())
    if manifest.get('signature')!='S5' or manifest['full_cache_estimated_bytes']+manifest['reserve_bytes']<=manifest['free_cache_bytes_at_partition']:
        raise ValueError('S5 batching requires the documented disk-space trigger')
    if not json.loads((root/'reports/st1_instruction_03_approval.json').read_text()) or not json.loads((root/'reports/instruction03/m0_parity.json').read_text())['C_allowed']:
        raise ValueError('S5 batches require instruction03 approval and passed M0')
    if not json.loads((root/'reports/instruction03/r2_completion.json').read_text())['first_group_complete']:
        raise ValueError('S5 is after the completed first group and R2')
    m6=json.loads((root/'reports/instruction03/m6_projection.json').read_text())
    if 'S5' not in m6['retained_signatures'] or 'S5' in m6['removed_signatures']:
        raise ValueError('S5 must be retained by the actual M6 decision')
    canonical={r['run_id']:r for r in read_csv(root/'configs/st1_instruction03/m/plan.csv') if r['signature']=='S5'}
    if len(canonical)!=2 or set(manifest['canonical_run_ids'])!=set(canonical):
        raise ValueError('S5 batches are limited to the two fixed RhythmFormer runs')
    subjects=sorted({r['subject_id'] for r in index.values()},key=lambda value:int(value[7:]))
    parts=manifest['parts'];count=len(parts);covered=[]
    if count<2 or count>32:raise ValueError('Invalid fixed S5 subject batch count')
    for i,part in enumerate(parts):
        key='batch'+str(i).zfill(2)
        if (part['number'],part['population_key'],part['BEGIN'],part['END'])!=(i,key,i/count,(i+1)/count):
            raise ValueError('S5 batch intervals differ from the fixed equal partition')
        selected=subjects[int(i/count*len(subjects)):int((i+1)/count*len(subjects))]
        expected=sorted(rid for rid,row in index.items() if row['subject_id'] in selected)
        if part['recording_ids']!=expected or part['expected_recordings']!=len(expected):
            raise ValueError('S5 batch coverage differs from official sorted-subject BEGIN/END')
        if Path(part['cache_root']).resolve()!=Path(paths['cache']).resolve()/'instruction03/m/S5'/key:
            raise ValueError('S5 cache is outside its own fresh batch')
        covered.extend(expected)
    if len(covered)!=655 or set(covered)!=set(index):raise ValueError('S5 batches must cover all655 exactly once')
    native=manifest['native_rows']
    if len(native)!=2*count or len({r['run_id'] for r in native})!=len(native):
        raise ValueError('S5 requires exactly two unique native calls per fixed batch')
    for row in native:
        plan=canonical.get(row['canonical_run_id']);part=parts[row['batch_number']]
        if not plan or (plan['model'],plan['native_mode'],plan['seed'],plan['population_key'],plan['WP'],plan['kind'],plan['target'],plan['label_type'],plan['chunk_len'])!=('RhythmFormer','NAT-all','100','all655','MGPU','RW','MMPD','Standardized','160'):
            raise ValueError('S5 model/native population is outside the fixed scope')
        if row['run_id']!=plan['run_id']+'_'+part['population_key'] or row['expected_recordings']!=part['expected_recordings']:
            raise ValueError('S5 native batch identity differs')
        if any(str(row[k])!=str(plan[k]) for k in ('WP','kind','model','train_source','target','seed','checkpoint_path','checkpoint_sha256','label_type','chunk_len')):
            raise ValueError('S5 scientific identity differs from its canonical fixed plan')
        if hashlib.sha256(Path(plan['config_path']).read_bytes()).hexdigest()!=row['canonical_config_sha256'] or row['canonical_config_sha256']!=plan['config_sha256']:
            raise ValueError('S5 canonical config changed after the batch declaration')
        config=Path(row['config_path'])
        if config.is_symlink() or config.stat().st_mode & 0o222 or hashlib.sha256(config.read_bytes()).hexdigest()!=row['config_sha256']:
            raise ValueError('S5 batch config hash/read-only status differs')
        expected={'BASE':[plan['config_path']], 'LOG':{'PATH':str(Path(paths['sealed_mmpd_b'])/row['run_id']/'native')},
                  'TEST':{'DATA':{'CACHED_PATH':part['cache_root'],'FILE_LIST_PATH':str(Path(part['cache_root'])/'filelists'),
                                  'BEGIN':part['BEGIN'],'END':part['END']}}}
        if yaml.safe_load(config.read_text())!=expected:
            raise ValueError('S5 batch override changes fields beyond BEGIN/END and storage')
    return manifest,parts


def registered_s5_part(row,root=ROOT):
    root=Path(root)
    records=[r for r in read_csv(root/'runs/st1_runs.csv') if r['run_id']==row['run_id']]
    if len(records)!=1 or records[0]['exit_code']!='0':
        raise ValueError('S5 batch requires exactly one actual successful native process')
    record=records[0]
    if any(str(row[k])!=str(record[k]) for k in ('WP','kind','model','train_source','target','seed','config_path','checkpoint_path','checkpoint_sha256','config_sha256')):
        raise ValueError('S5 actual batch registration differs from the fixed declaration')
    if hashlib.sha256(Path(row['checkpoint_path']).read_bytes()).hexdigest()!=row['checkpoint_sha256']:
        raise ValueError('S5 checkpoint changed')
    return record


def public_analysis_registry(root=ROOT):
    """Expose completed CHROM/POS method aliases without inventing process rows.

    Resource accounting always reads the actual registry directly. These two
    aliases are for public-A grid/metadata analysis of one S8 native process.
    """
    root=Path(root)
    records=read_csv(root/'runs/st1_runs.csv')
    plan_file=root/'configs/st1_instruction03/m/plan.csv'
    if not plan_file.is_file():return records
    for row in read_csv(plan_file):
        if row.get('signature') not in ('S8','S5'):continue
        completion=root/'outputs/st1_predictions'/row['run_id']/'mmpd_completion.json'
        if not completion.is_file():continue
        if json.loads(completion.read_text()).get('status')!='complete':continue
        plan,native=registered_run(row['run_id'],root)
        if any(r['run_id']==row['run_id'] for r in records):
            if row['signature']=='S5':continue
            raise ValueError('S8 analysis alias duplicates an observed process')
        alias=dict(native)
        for key in ('run_id','model','kind','WP','train_source','target','seed','config_path','checkpoint_path','checkpoint_sha256'):
            alias[key]=plan[key]
        alias.update(source_native_run_id=native['run_id'],derived_method_alias=True)
        records.append(alias)
    # Actual S5 BEGIN/END calls are accounting records, not full-cohort
    # scientific runs. Only their verified canonical merge enters analysis.
    batch_file=root/'reports/instruction03/s5_batches.json'
    if batch_file.is_file():
        partial_ids={r['run_id'] for r in json.loads(batch_file.read_text())['native_rows']}
        records=[r for r in records if r['run_id'] not in partial_ids]
    return records
