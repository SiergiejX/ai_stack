#!/usr/bin/env python3
import json
import os
import itertools

SRC = os.path.join(os.path.dirname(__file__), 'json.json')
OUT = os.path.join(os.path.dirname(__file__), 'converted_flow.json')

def collect(list_or_item):
    if list_or_item is None:
        return []
    if isinstance(list_or_item, list):
        return list_or_item
    return [list_or_item]

def main():
    with open(SRC, 'r', encoding='utf-8') as f:
        data = json.load(f)

    defs = data.get('definitions', {})
    processes = collect(defs.get('process', []))

    # Collect BPMN elements by id
    elements = {}
    flows = {}

    for proc in processes:
        # sequence flows
        for sf in collect(proc.get('sequenceFlow', [])):
            fid = sf.get('_id')
            src = sf.get('_sourceRef')
            tgt = sf.get('_targetRef')
            flows[fid] = {'source': src, 'target': tgt, 'name': sf.get('_name')}

        # common types to scan
        keys = ['startEvent', 'endEvent', 'task', 'userTask', 'serviceTask', 'scriptTask', 'exclusiveGateway', 'parallelGateway']
        for k in keys:
            for item in collect(proc.get(k, [])):
                iid = item.get('_id')
                if not iid:
                    continue
                elements[iid] = {
                    'bpmn_type': k,
                    'name': item.get('_name') or iid,
                }

        # also scan 'task' entries under other keys present in this BPMN
        # HACK: scan all dict items in process for objects with _id and _name
        for v in proc.values():
            for item in collect(v):
                if isinstance(item, dict) and item.get('_id'):
                    iid = item.get('_id')
                    if iid not in elements:
                        # try to infer type
                        btype = None
                        for t in ['task','activity','event','gateway']:
                            if t in item.get('_id','').lower():
                                btype = t
                                break
                        elements[iid] = {
                            'bpmn_type': btype or 'node',
                            'name': item.get('_name') or iid,
                        }

    # Build adjacency from flows
    adjacency = {nid: [] for nid in elements.keys()}
    for fid, info in flows.items():
        src = info['source']
        tgt = info['target']
        if src in adjacency:
            adjacency[src].append(tgt)

    # Map BPMN ids to Node-RED ids
    def nrid(idx):
        # produce deterministic hex id
        return hex(abs(hash(idx)) & 0xffffffff)[2:10]

    id_map = {bpmn_id: nrid(bpmn_id) for bpmn_id in elements.keys()}

    # Create Node-RED nodes
    nodes = []
    # layout helpers
    x = 80
    y_counters = {}

    for i, (bpmn_id, meta) in enumerate(elements.items()):
        targets = adjacency.get(bpmn_id, [])
        outputs = max(1, len(targets))
        # choose node type
        btype = meta.get('bpmn_type','')
        if 'start' in btype:
            ntype = 'inject'
        elif 'end' in btype:
            ntype = 'debug'
        elif 'gateway' in btype or 'exclusive' in btype:
            ntype = 'switch'
        else:
            ntype = 'function'

        # position per simple column layout by first char of id
        col = (i % 6)
        x = 120 + col*220
        y_counters.setdefault(col, 20)
        y = y_counters[col]
        y_counters[col] += 80

        node = {
            'id': id_map[bpmn_id],
            'type': ntype,
            'z': 'flow1',
            'name': meta.get('name'),
            'func': '// converted from BPMN element {}'.format(bpmn_id) if ntype=='function' else '',
            'outputs': outputs,
            'wires': [],
            'x': x,
            'y': y,
        }

        # prepare wires slots
        for tgt in targets:
            if tgt in id_map:
                node['wires'].append([id_map[tgt]])
            else:
                node['wires'].append([])

        # ensure wires length equals outputs
        if len(node['wires']) < outputs:
            while len(node['wires']) < outputs:
                node['wires'].append([])

        nodes.append(node)

    # write output
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(nodes, f, ensure_ascii=False, indent=2)

    print('Wrote', OUT)

if __name__ == '__main__':
    main()
