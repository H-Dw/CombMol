"""SDF-authoritative PLIP and PoseBusters; adapted from the user-owned EvoMolSteer profiling module.
Heavy atoms and the measured chemical graph are never optimized.
"""
import hashlib
import importlib.metadata
import logging
from pathlib import Path
import warnings
import numpy as np
from rdkit import Chem
from rdkit.Chem import QED, rdMolDescriptors
from .io import read, write, digest, value_hash as json_digest

def package_version(name):
    """An importable source installation may legitimately lack dist-info metadata."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        module = importlib.import_module(name)
        version = getattr(module, '__version__', None)
        return version if isinstance(version, str) else None


class PDBRepresentationError(ValueError):
    """The fixed pose cannot be encoded in PDB's Real(8.3) fields."""
    def __init__(self,reason,details):
        super().__init__(reason)
        self.reason=reason;self.details=details


def check_pdb_coordinates(xyz):
    xyz=np.asarray(xyz)
    if not np.isfinite(xyz).all():
        raise PDBRepresentationError('nonfinite_coordinates_for_pdb',{'nonfinite_values':int((~np.isfinite(xyz)).sum())})
    # Check the formatted width, including rounding at both asymmetric bounds.
    overflow=[(i,j) for i,row in enumerate(xyz) for j,x in enumerate(row) if len(f'{float(x):8.3f}')>8]
    if overflow:
        i,j=overflow[0]
        raise PDBRepresentationError('pdb_coordinate_field_overflow',{'atom_index':i,'axis':j,
            'value_A':float(xyz[i,j]),'overflow_values':len(overflow),'format':'Real(8.3)',
            'min_A':xyz.min(axis=0).tolist(),'max_A':xyz.max(axis=0).tolist()})


def graph(m):
    return {'elements':[a.GetSymbol() for a in m.GetAtoms()],
        'formal_charges':[a.GetFormalCharge() for a in m.GetAtoms()],
        'bonds':[[b.GetBeginAtomIdx(),b.GetEndAtomIdx(),str(b.GetBondType())] for b in m.GetBonds()]}


def validate_roundtrip(mol,perceived):
    """Compare chemistry and coordinates using a validation-only heavy view.

    OpenBabel may encode imine E/Z with an explicit H that RDKit's default
    RemoveHs retains. Removing that representation-only atom on a COPY is safe
    here: this check already ignores stereo labels and PLIP uses the unchanged
    hydrogenated OpenBabel graph and fixed 3D pose, not this comparison copy.
    """
    from scipy.optimize import linear_sum_assignment
    params=Chem.RemoveHsParameters();params.removeDefiningBondStereo=True
    perceived=Chem.RemoveHs(perceived,params)
    expected_smiles=Chem.MolToSmiles(mol,isomericSmiles=False)
    actual_smiles=Chem.MolToSmiles(perceived,isomericSmiles=False)
    if expected_smiles!=actual_smiles:
        raise ValueError(f'PDB round-trip changed ligand graph: {expected_smiles} != {actual_smiles}')
    a=np.array(mol.GetConformer().GetPositions());b=np.array(perceived.GetConformer().GetPositions())
    if len(a)!=len(b):raise ValueError('PLIP heavy-atom count changed')
    cost=((a[:,None]-b[None,:])**2).sum(-1)
    za=np.array([x.GetAtomicNum() for x in mol.GetAtoms()]);zb=np.array([x.GetAtomicNum() for x in perceived.GetAtoms()])
    cost+=1e6*(za[:,None]!=zb[None,:]);ii,jj=linear_sum_assignment(cost)
    if cost[ii,jj].max()>1e-4:raise ValueError('PLIP atom-coordinate mapping failed')
    mapping=dict(zip(map(int,ii),map(int,jj)))
    for i,j in mapping.items():
        x,y=mol.GetAtomWithIdx(i),perceived.GetAtomWithIdx(j)
        for name in ['GetFormalCharge','GetIsotope','GetNumRadicalElectrons']:
            if getattr(x,name)()!=getattr(y,name)():raise ValueError('PLIP atom chemistry changed: '+name)
        if x.GetTotalNumHs(includeNeighbors=True)!=y.GetTotalNumHs(includeNeighbors=True):
            raise ValueError('PLIP attached hydrogen count changed')
    for bond in mol.GetBonds():
        other=perceived.GetBondBetweenAtoms(mapping[bond.GetBeginAtomIdx()],mapping[bond.GetEndAtomIdx()])
        if other is None or other.GetBondType()!=bond.GetBondType():
            raise ValueError('PLIP coordinate-mapped bond changed')
    return float(np.sqrt(cost[ii,jj]).max())


def complex_pdb(mol,protein_file,destination):
    """Keep all heavy coordinates; write explicit hydrogen/bond/charge information."""
    m=Chem.Mol(mol);original=np.array(m.GetConformer().GetPositions())
    check_pdb_coordinates(original)
    h=Chem.AddHs(m,addCoords=True)
    check_pdb_coordinates(h.GetConformer().GetPositions())
    assert np.array_equal(original,np.array(h.GetConformer().GetPositions())[:len(original)])
    text=Path(protein_file).read_text()
    lines=[l for l in text.splitlines() if l.startswith(('ATOM  ','HETATM','TER   ','CONECT'))]
    atom_lines=[l for l in lines if l.startswith(('ATOM  ','HETATM'))]
    offset=max(int(l[6:11]) for l in atom_lines)+1
    serial_to_native={}
    for i,a in enumerate(h.GetAtoms()):
        info=Chem.AtomPDBResidueInfo()
        info.SetName((a.GetSymbol()+str(i+1)).rjust(4));info.SetResidueName('LIG')
        info.SetResidueNumber(1);info.SetChainId('Z');info.SetIsHeteroAtom(True);info.SetOccupancy(1.)
        a.SetMonomerInfo(info)
        if i<m.GetNumAtoms():serial_to_native[offset+i+1]=i
    added=[]
    for l in Chem.MolToPDBBlock(h).splitlines():
        if l.startswith(('ATOM  ','HETATM')):
            serial=int(l[6:11])+offset
            added.append(l[:6]+f'{serial:5d}'+l[11:])
        elif l.startswith('CONECT'):
            values=[int(l[i:i+5])+offset for i in range(6,len(l),5) if l[i:i+5].strip()]
            added.append('CONECT'+''.join(f'{v:5d}' for v in values))
    Path(destination).parent.mkdir(parents=True,exist_ok=True)
    Path(destination).write_text('\n'.join(lines+added+['END'])+'\n')
    return serial_to_native,h


def plip_profile(mol,protein_file,out):
    from plip.basic import config
    from plip.structure.preparation import PDBComplex,LigandFinder
    from plip.exchange.report import StructureReport,BindingSiteReport
    from openbabel import openbabel as ob,pybel
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    config.NOHYDRO=True;config.NOFIX=True;config.NOFIXFILE=True
    config.OUTPUTFOLDER=str(out);config.COMPRESS=False;config.MAXTHREADS=1
    logging.getLogger('plip').setLevel(logging.ERROR)
    serial_map,hydrogenated=complex_pdb(mol,protein_file,out/'complex.pdb')
    c=PDBComplex();c.output_path=str(out)+'/'
    c.load_pdb(str(out/'complex.pdb'))
    # PDB does not unambiguously preserve bond orders. OpenBabel can infer extra
    # bonds for a distorted generated pose even with CONECT records. Restore the
    # authoritative SDF graph (and deterministic explicit H) without moving any
    # atom, then let standard PLIP chemistry/interaction rules analyze that graph.
    # This repairs serialization, never the generated molecule or its geometry.
    start_serial=min(serial_map)
    serials={start_serial+i:i for i in range(hydrogenated.GetNumAtoms())}
    inverse={v:k for k,v in c.Mapper.proteinmap.items()}
    global_ids={i:inverse[s] for s,i in serials.items()}
    full_ids=set(global_ids.values());om=c.protcomplex.OBMol
    previous_bonds=[[bond.GetBeginAtomIdx(),bond.GetEndAtomIdx(),bond.GetBondOrder()]
        for bond in ob.OBMolBondIter(om)
        if bond.GetBeginAtomIdx() in full_ids or bond.GetEndAtomIdx() in full_ids]
    for a,b,order in previous_bonds:om.DeleteBond(om.GetBond(a,b))
    kekule=Chem.Mol(hydrogenated);Chem.Kekulize(kekule,clearAromaticFlags=False)
    for i,a in enumerate(hydrogenated.GetAtoms()):
        atom=om.GetAtom(global_ids[i]);atom.SetAtomicNum(a.GetAtomicNum());atom.SetFormalCharge(a.GetFormalCharge())
        atom.SetImplicitHCount(0);atom.SetSpinMultiplicity(0);atom.SetAromatic(a.GetIsAromatic())
    for bond in kekule.GetBonds():
        a,b=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
        om.AddBond(global_ids[a],global_ids[b],int(bond.GetBondTypeAsDouble()))
        om.GetBond(global_ids[a],global_ids[b]).SetAromatic(hydrogenated.GetBondBetweenAtoms(a,b).GetIsAromatic())
    om.SetAromaticPerceived(True)
    # Topology edits invalidate OpenBabel's chain-perception flag. Preserve the
    # explicit PDB residue labels instead of allowing it to recreate them.
    om.SetChainsPerceived(True)
    for i,expected in enumerate(hydrogenated.GetConformer().GetPositions()):
        atom=om.GetAtom(global_ids[i]);observed=np.array([atom.GetX(),atom.GetY(),atom.GetZ()])
        if np.linalg.norm(expected-observed)>1e-3:raise ValueError('Graph transfer moved a coordinate')
    finder=LigandFinder(c.protcomplex,c.altconf,c.modres,c.covalent,c.Mapper)
    c.ligands=finder.ligands;c.excluded=finder.excluded;c.atoms={a.idx:a for a in c.protcomplex}
    ligs=[l for l in c.ligands if l.hetid=='LIG' and l.chain=='Z' and int(l.position)==1]
    if len(ligs)!=1:raise ValueError('PLIP did not identify exactly LIG:Z:1')
    ligand=ligs[0]
    # Validate PDB/OpenBabel graph perception before trusting chemical roles.
    # PLIP's extracted ligand intentionally omits H without replenishing their
    # implicit counts. Validate the input graph with its explicit H, rather than
    # misclassifying that internal heavy-only view as a radical molecule.
    full=ob.OBMol();bits=ob.OBBitVec(c.protcomplex.OBMol.NumAtoms())
    for atom in ob.OBMolAtomIter(c.protcomplex.OBMol):
        residue=atom.GetResidue()
        if residue and residue.GetChain()=='Z' and residue.GetName().strip()=='LIG':bits.SetBitOn(atom.GetIdx())
    c.protcomplex.OBMol.CopySubstructure(full,bits,None,0)
    block=pybel.Molecule(full).write('sdf')
    perceived=Chem.MolFromMolBlock(block,removeHs=True,sanitize=True)
    if perceived is None:raise ValueError('PLIP ligand graph cannot be decoded')
    roundtrip_distance=validate_roundtrip(mol,perceived)
    c.analyze();site=c.interaction_sets['LIG:Z:1'];report=BindingSiteReport(site)
    StructureReport(c).write_xml()
    rows=[]
    for kind,name in [('hydrophobic','hydrophobic'),('hbond','hbond'),('saltbridge','saltbridge'),
        ('pistacking','pistacking'),('pication','pication'),('halogen','halogen'),('waterbridge','waterbridge'),('metal','metal')]:
        for values in getattr(report,name+'_info'):
            d=dict(zip(getattr(report,name+'_features'),values))
            role='hydrophobe' if kind=='hydrophobic' else kind
            if kind=='hydrophobic':ligidx=[int(d['LIGCARBONIDX'])];protidx=[int(d['PROTCARBONIDX'])]
            elif kind in ('hbond','waterbridge'):
                pd=bool(d['PROTISDON']);role='ligand_acceptor' if pd else 'ligand_donor'
                dk='DONORIDX' if kind=='hbond' else 'DONOR_IDX';ak='ACCEPTORIDX' if kind=='hbond' else 'ACCEPTOR_IDX'
                ligidx=[int(d[ak if pd else dk])];protidx=[int(d[dk if pd else ak])]
            elif kind in ('saltbridge','pistacking','pication'):
                ligidx=[int(x) for x in d['LIG_IDX_LIST'].split(',')]
                protidx=[int(x) for x in d['PROT_IDX_LIST'].split(',')]
                if kind=='saltbridge':role='ligand_negative' if d['PROTISPOS'] else 'ligand_positive'
                if kind=='pication':role='ligand_aromatic' if d['PROTCHARGED'] else 'ligand_positive'
            elif kind=='halogen':ligidx=[int(d['DON_IDX'])];protidx=[int(d['ACC_IDX'])];role='ligand_halogen_donor'
            else:
                ligidx=[int(x) for x in [d['METAL_IDX'],d['TARGET_IDX']] if int(x) in serial_map]
                protidx=[int(x) for x in [d['METAL_IDX'],d['TARGET_IDX']] if int(x) not in serial_map]
            if any(x not in serial_map for x in ligidx):raise ValueError('PLIP ligand serial absent from original atom map')
            core=[serial_map[x] for x in ligidx]
            fp=[kind,str(d['RESCHAIN']),str(d['RESNR']),str(d['RESTYPE']),role]
            rows.append({'type':kind,'role':role,'ligand_atoms':core,'ligand_pdb_serials':ligidx,
                'protein_pdb_serials':protidx,'fingerprint':fp,'details':d})
    core=sorted({i for r in rows for i in r['ligand_atoms']})
    thresholds={k:v for k,v in vars(config).items() if k.isupper() and any(x in k for x in ('DIST','ANGLE','OFFSET')) and isinstance(v,(float,int,str,bool))}
    result={'plip_version':config.__version__,'openbabel':ob.OBReleaseVersion(),'nohydro':True,'nofix':True,
        'protein_sha256':digest(protein_file),'source_graph':graph(mol),'roundtrip_graph_verified':True,
        'ligand_graph_source':'original SDF, transferred after PDB serialization; no geometric relaxation',
        'heavy_coords_roundtrip_max_A':roundtrip_distance,
        'roundtrip_validation':'heavy-graph-charge-hcounts-coordinates-v2',
        'pdb_serial_to_native':{str(k):v for k,v in serial_map.items()},'core_atoms':core,
        'core_coverage':len(core)/mol.GetNumHeavyAtoms(),'interactions':rows,
        'fingerprints':sorted({tuple(r['fingerprint']) for r in rows}), 'thresholds':thresholds}
    write(out/'interactions.json',result)
    return result


def quality(mol,protein_file):
    from posebusters import PoseBusters
    if mol is None:return {'valid_connected':False,'posebusters_pass':False,'severe_clash':False,'plip_evaluable':False}
    try:Chem.SanitizeMol(mol)
    except Exception as e:return {'valid_connected':False,'posebusters_pass':False,'severe_clash':False,'quality_error':str(e)}
    protein_xyz=np.array([[float(l[30:38]),float(l[38:46]),float(l[46:54])] for l in Path(protein_file).read_text().splitlines()
        if l.startswith(('ATOM  ','HETATM')) and l[76:78].strip().upper() not in ('H','D')])
    xyz=np.array(mol.GetConformer().GetPositions())
    dist=np.linalg.norm(xyz[:,None]-protein_xyz[None],axis=-1)
    result={'valid_connected':len(Chem.GetMolFrags(mol))==1,'min_protein_distance_A':float(dist.min()),
        'severe_clash':bool((dist<1.2).any()),'pairs_below_1_2A':int((dist<1.2).sum()),
        'diagnostic_sdf_smiles':Chem.MolToSmiles(mol),'qed':float(QED.qed(mol)),'heavy_atoms':mol.GetNumHeavyAtoms(),
        'formal_charge':sum(a.GetFormalCharge() for a in mol.GetAtoms()),'posebusters_version':package_version('posebusters')}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            frame=PoseBusters(config='dock').bust(mol,mol_cond=str(protein_file),full_report=False)
    except Exception as error:
        # Distorted but finite poses can overflow the evaluator's spatial grid.
        # Preserve known basic measurements and explicitly report missing PB;
        # an evaluator exception is neither a passing nor a failing PB result.
        result.update(posebusters_evaluable=False,posebusters_pass=None,posebusters_checks=None,
                      posebusters_error={'type':type(error).__name__,'message':str(error)})
        return result
    row=frame.iloc[0]
    checks={str(k):bool(v) for k,v in row.items() if isinstance(v,(bool,np.bool_))}
    if not checks:raise ValueError('PoseBusters returned no Boolean checks')
    result.update(posebusters_evaluable=True,posebusters_checks=checks,posebusters_pass=all(checks.values()))
    return result


def profile_task(task):
    """Worker cache is keyed by full source pose/protein, never SMILES alone."""
    mol=Chem.MolFromMolBlock(task['molblock'],removeHs=True,sanitize=True)
    if mol is None:raise ValueError('Could not load built source molecule')
    out=Path(task['out']);key=json_digest({'molblock':task['molblock'],'protein':digest(task['protein']),
        'plip':'3.0.1','pb':package_version('posebusters'),'hydrogenation':'rdkit-addHs-fixed-heavy-v1','policy':'nohydro-nofix-sdf-graph-dock-v2'})
    if (out/'complete.json').exists():
        saved=read(out/'complete.json')
        if saved['cache_key']!=key:raise ValueError('Profile cache source changed')
        return saved
    out.mkdir(parents=True,exist_ok=True)
    try:p=plip_profile(mol,task['protein'],out)
    except PDBRepresentationError as error:
        # Keep the source pose. Do not translate, clamp, relax, or count missing
        # interactions as a measured zero when serialization is impossible.
        p={'plip_evaluable':False,'reason':error.reason,'details':error.details,
           'source_graph':graph(mol),'source_coordinates_preserved':True,
           'core_atoms':None,'fingerprints':None,'interactions':None}
        write(out/'interactions.json',p)
    q=quality(mol,task['protein'])
    q['plip_evaluable']=p.get('plip_evaluable',True)
    if not q['plip_evaluable']:q['plip_unevaluable_reason']=p['reason']
    result={'cache_key':key,'node_id':task['node_id'],'profile_path':str(out/'interactions.json'),'quality':q}
    write(out/'complete.json',result)
    return result
