"""Effective source-budget DAG amendment; immutable S7 records remain historical."""
import copy
from .contracts import read_json
from capstone_lab.config import sha256_file

APPROVAL = 'configs/approvals/s8_foreground_amendment_v1.json'


def amend_dag(root, nodes, registry):
    a=read_json(root/APPROVAL)
    if a['status']!='APPROVED_BY_USER' or a['foreground_subsets']!={'A':'train748','M':'train748','H':'train748','L':'low187'}:
        raise ValueError('source policy approval mismatch')
    shared={'generate_generative_pool_6000','select_generative_random_3000','select_task_aware_3000'}
    original=copy.deepcopy(nodes)
    result=[]
    for n in original:
        name=n['node_id']
        if name in shared:
            for group,subset in [('M','train748'),('L','low187')]:
                clone=copy.deepcopy(n)
                clone['node_id']=name+'_'+group
                clone['foreground_subset']=subset
                clone['dependencies']=[d+'_'+group if d in shared else
                    'verify_train748_sources' if d=='verify_low187_sources' and group=='M' else d for d in clone['dependencies']]
                result.append(clone)
            continue
        n['dependencies']=[d+'_L' if d in shared and name.startswith('L') else d+'_M' if d in shared else d for d in n['dependencies']]
        if name.startswith('generate_A'):
            n['foreground_subset']='train748'
            n['dependencies']=['verify_train748_sources' if d=='verify_low187_sources' else d for d in n['dependencies']]
        if name.startswith('L1_seed'):
            n['dependencies']=['validate_L_A1_3000' if d=='validate_A1_3000' else d for d in n['dependencies']]
        if name=='generate_anydoor_pilot100':
            n['dependencies'].append('verify_train748_sources')
            n['required_review_budgets']=['train748','low187']
        result.append(n)
    result.extend([
        {'node_id':'verify_train748_sources','kind':'SOURCE_PROVENANCE','dependencies':['data_real748'],'execution_status':'VERIFIED_SOURCE_PROOF_RECHECK_ON_USE'},
        {'node_id':'generate_L_A1_3000','kind':'CPU_SYNTHESIS','dependencies':['gate_campaign_execution','verify_low187_sources'],'foreground_subset':'low187','execution_status':'PENDING_IMPLEMENTATION_AND_PREFLIGHT'},
        {'node_id':'validate_L_A1_3000','kind':'SYNTHESIS_VALIDATION','dependencies':['generate_L_A1_3000'],'execution_status':'PENDING_IMPLEMENTATION_AND_PREFLIGHT'}])
    ids={n['node_id'] for n in result}
    if len(ids)!=len(result) or any(not set(n['dependencies'])<=ids for n in result):
        raise ValueError('amended DAG has duplicate/missing nodes')
    done=set()
    while len(done)<len(ids):
        ready={n['node_id'] for n in result if set(n['dependencies'])<=done}-done
        if not ready: raise ValueError('amended DAG cycle')
        done.update(ready)
    records=copy.deepcopy(registry)
    for r in records:
        name=r['job_id']
        r['foreground_subset']='low187' if name.startswith('L') else 'train748'
        r['source_policy_override']=APPROVAL
        r['historical_fields_not_executable']=True
    return result,records,{APPROVAL:sha256_file(root/APPROVAL)}
