import pandas as pd
import numpy as np
from pyproj import Proj, transform
from tqdm import tqdm_notebook


def lat(x,y,z):
    return np.arcsin(z/np.sqrt(x**2 + y**2 + z**2))

def lon(x,y,z):
    return np.arctan2(y,x)


def get_data_2dprojection(lpmt_hits, spmt_hits, pos, true_info, edge_size=150, use_spmt=False, time=False):
    """
    Transfer data into 2d projection (moll) with 1 channel
    Params:
    ..., 
    edge_size : int - projection image size, 
    use_spmt : bool 
    
    Returns:
    data_lpmt: pd.DataFrame
    event_to_id: dict
    """
    channels = 1
    if time == True:
        channels = 3    
    pos['lat'] = list(map(lambda el: lat(el[0],el[1],el[2]), zip(pos['pmt_x'], pos['pmt_y'], pos['pmt_z'])))
    pos['lon'] = list(map(lambda el: lon(el[0],el[1],el[2]), zip(pos['pmt_x'], pos['pmt_y'], pos['pmt_z'])))
    p1 = Proj(proj='latlong')
    p2 = Proj(proj='moll')
    pos['mol0'] = list(map(lambda el: transform(p1,p2,el[0],el[1])[0], zip(pos['lat'], pos['lon'])))
    pos['mol1'] = list(map(lambda el: transform(p1,p2,el[0],el[1])[1], zip(pos['lat'], pos['lon'])))
    print("Make projection")
    if use_spmt:
        lpmt_hits = pd.concat([lpmt_hits, spmt_hits])
    merged_hits = pd.merge(lpmt_hits, pos, left_on='pmtID', right_on='pmt_id')
    merged_hits['mol0i'] = round(merged_hits['mol0'] / (pos['mol0'].max() - pos['mol0'].min()) * edge_size).astype(int)
    merged_hits['mol1i'] = round(merged_hits['mol1'] / (pos['mol1'].max() - pos['mol1'].min()) * edge_size).astype(int)
    n = len(lpmt_hits['event'].unique())
    data_lpmt = np.zeros((n, edge_size+1, edge_size+1, channels))
    event_to_id = {x:y for y, x in enumerate(sorted(merged_hits['event'].unique()))}
    print("Starting cycle...")
    for event, mol0i, mol1i in tqdm_notebook(zip(merged_hits['event'], merged_hits['mol0i'], merged_hits['mol1i'])):
        data_lpmt[event_to_id[event]][mol0i, mol1i] += 1
    return data_lpmt, event_to_id


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    