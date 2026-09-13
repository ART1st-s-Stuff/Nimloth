"""CPU-only render of completed feature tensors; no model or rollout execution."""
import argparse
from pathlib import Path
import torch
from rollout_features import plot


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    rows=[]
    for path in sorted(args.features_dir.glob('*_turn*.pt')):
        saved=torch.load(path,map_location='cpu',weights_only=True)
        prediction=saved['prediction'].float().numpy()
        target=saved['target'].float().numpy()
        rows.append(dict(**saved['metadata'],prediction=prediction,target=target,
                         error=1-saved['cosine'].float().numpy().reshape(target.shape[:2])))
    if not rows:
        raise ValueError('no saved feature tensors')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    plot(rows,args.output_dir)
    (args.output_dir/'COMPLETED').write_text('complete\n')

if __name__=='__main__': main()
