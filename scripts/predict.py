"""Apply a saved head adapter to an audited 100-return text input."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax

ROOT=Path(__file__).resolve().parents[1]


def predict(payload,adapter_dir):
    state=payload.get('input',payload).get('state')
    if not isinstance(state,str) or 'r_bp=' not in state:
        raise ValueError('Provide input.state with the documented r_bp= sequence')
    series=state.split('r_bp=',1)[1].split('\n',1)[0].split(',')
    if len(series)!=100:
        raise ValueError('Exactly 100 completed daily return observations are required')
    folder=Path(adapter_dir)
    config=json.loads((folder/'adapter_config.json').read_text())
    if config['model']=='laya':
        from model_features import LayaFeatures
        model=LayaFeatures(vulkan=True)
    else:
        from d3_features import D3Features
        model=D3Features(vulkan=True)
    features,tokens=model.extract([state])
    from train_evaluate import logits
    p=softmax(logits(features,np.load(folder/'weight.npy'),config.get('representation'))/config['additional_validation_temperature'],axis=1)[0]
    model.close()
    names=['large_down','down','flat','up','large_up']
    direction={'down':float(p[:2].sum()),'flat':float(p[2]),'up':float(p[3:].sum())}
    choice=max(direction,key=direction.get)
    answer={'horizon':'next session open-to-close; excludes overnight gap','five_bin_probabilities':dict(zip(names,map(float,p))),'direction_probabilities':direction,'choice':choice,'action':'abstain' if direction[choice]<.55 or choice=='flat' else choice,'tokens':tokens[0],'context_points':100,'model':config['model'],'base_revision':config['base_revision'],'news_coverage':payload.get('news_coverage_json','see supplied state'),'interpretation':'Research forecast; finite ranges are in trailing daily-return sigma, not current-price target ranges.'}
    nonflat_mass=direction['down']+direction['up']
    answer['conditional_nonflat_up_probability']=direction['up']/nonflat_mass if nonflat_mass>0 else None
    return answer


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--adapter',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args()
    answer=predict(json.loads(a.input.read_text()),a.adapter);text=json.dumps(answer,indent=2)
    if a.output:a.output.write_text(text)
    print(text)
