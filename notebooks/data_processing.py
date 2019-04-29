import pandas as pd
import numpy as np
from pyproj import Proj, transform
from tqdm import tqdm_notebook


def lat(x,y,z):
    return np.arcsin(z/np.sqrt(x**2 + y**2 + z**2))

def lon(x,y,z):
    return np.arctan2(y,x)


def get_data_2dprojection(lpmt_hits, spmt_hits, pos, true_info, edge_size=150, use_spmt=False, time=None):
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
    if time == 'min':
        channels = 2
    elif time == 'minmax':
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
    
    mol0min = -160000
    mol0max = 160000
    mol1min = -390000
    mol1max = 390000
    # Fit coordinates
    merged_hits['mol0i'] = round((merged_hits['mol0'] - mol0min) / (mol0max - mol0min) * edge_size).astype(int)
    merged_hits['mol1i'] = round((merged_hits['mol1'] - mol1min) / (mol1max - mol1min) * edge_size).astype(int)
    
    n = len(lpmt_hits['event'].unique())
    data_lpmt = np.zeros((n, edge_size+1, edge_size+1, channels))
    
    event_to_id = {x:y for y, x in enumerate(sorted(merged_hits['event'].unique()))}
    print("Starting cycle...")
    if time is None:
        for event, mol0i, mol1i in tqdm_notebook(zip(merged_hits['event'], merged_hits['mol0i'], merged_hits['mol1i'])):
            data_lpmt[event_to_id[event]][mol0i, mol1i] += 1
    elif time == 'min':
        EPS = 1e-7
        data_lpmt[:,:,:,1] = -EPS
        # Calculate min time for each event
        ev_min = merged_hits[['event', 'hitTime']].groupby('event').min()
        event2min = {id_: min_ for id_, min_ in zip(ev_min.index, ev_min.hitTime)}
        
        for event, mol0i, mol1i, time in tqdm_notebook(zip(merged_hits['event'], merged_hits['mol0i'], merged_hits['mol1i'], merged_hits['hitTime'])):
            data_lpmt[event_to_id[event]][mol0i, mol1i][0] += 1
            event_id = event_to_id[event]
            event_min = event2min[event]
            cur_min = data_lpmt[event_id][mol0i, mol1i][1]
            if cur_min < 0 or cur_min > (time - event_min):
                data_lpmt[event_id][mol0i, mol1i][1] = time - event_min
    elif time == 'minmax':
        # Add max and min for each cell
        for event, mol0i, mol1i, time in tqdm_notebook(zip(merged_hits['event'], merged_hits['mol0i'], merged_hits['mol1i'], merged_hits['hitTime'])):
            data_lpmt[event_to_id[event]][mol0i, mol1i][0] += 1
            cur_min = data_lpmt[event_to_id[event]][mol0i, mol1i][1]
            cur_max = data_lpmt[event_to_id[event]][mol0i, mol1i][2]
            if cur_min == 0:
                data_lpmt[event_to_id[event]][mol0i, mol1i][1] = time
            if cur_max == 0:
                data_lpmt[event_to_id[event]][mol0i, mol1i][2] = time
            if cur_min > time:
                data_lpmt[event_to_id[event]][mol0i, mol1i][1] = time
            if cur_max < time:
                data_lpmt[event_to_id[event]][mol0i, mol1i][2] = time
    return data_lpmt, event_to_id


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    