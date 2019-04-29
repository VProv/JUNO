import pandas as pd
import numpy as np
from pyproj import Proj, transform
from tqdm import tqdm_notebook


def lat(x,y,z):
    return np.arcsin(z/np.sqrt(x**2 + y**2 + z**2))

def lon(x,y,z):
    return np.arctan2(y,x)


def get_data_2dprojection(lpmt_hits, spmt_hits, pos, true_info, edge_size1=111, edge_size2=226, use_spmt=False, time=None, proj2='moll', limits=[-160000,160000,-390000,390000]):
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
    if time == 'min' or time == 'tomax':
        channels = 2
    pos['lat'] = list(map(lambda el: lat(el[0],el[1],el[2]), zip(pos['pmt_x'], pos['pmt_y'], pos['pmt_z'])))
    pos['lon'] = list(map(lambda el: lon(el[0],el[1],el[2]), zip(pos['pmt_x'], pos['pmt_y'], pos['pmt_z'])))
    p1 = Proj(proj='latlong')
    p2 = Proj(proj=proj2)
    pos['mol0'] = list(map(lambda el: transform(p1,p2,el[0],el[1])[0], zip(pos['lat'], pos['lon'])))
    pos['mol1'] = list(map(lambda el: transform(p1,p2,el[0],el[1])[1], zip(pos['lat'], pos['lon'])))
    print("Make projection")
    
    if use_spmt:
        lpmt_hits = pd.concat([lpmt_hits, spmt_hits])
    merged_hits = pd.merge(lpmt_hits, pos, left_on='pmtID', right_on='pmt_id')
    
    mol0min = limits[0]
    mol0max = limits[1]
    mol1min = limits[2]
    mol1max = limits[3]
    # Fit coordinates
    merged_hits['mol0i'] = round((merged_hits['mol0'] - mol0min) / (mol0max - mol0min) * (edge_size1 - 1)).astype(int)
    merged_hits['mol1i'] = round((merged_hits['mol1'] - mol1min) / (mol1max - mol1min) * (edge_size2 - 1)).astype(int)
    
    n = len(lpmt_hits['event'].unique())
    data_lpmt = np.zeros((n, edge_size1, edge_size2, channels))
    
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
        ev_len = merged_hits[['event', 'hitTime']].groupby('event').count()
        event2len = {id_: len_ for id_, len_ in zip(ev_len.index, ev_len.hitTime)}
        
        for event, mol0i, mol1i, time in tqdm_notebook(zip(merged_hits['event'], merged_hits['mol0i'], merged_hits['mol1i'], merged_hits['hitTime'])):
            event_id = event_to_id[event]
            data_lpmt[event_id][mol0i, mol1i][0] += 1 * 1000 / event2len[event]
            event_min = event2min[event]
            
            cur_min = data_lpmt[event_id][mol0i, mol1i][1] 
            if cur_min < 0 or cur_min > ((time - event_min)/100):
                data_lpmt[event_id][mol0i, mol1i][1] = (time - event_min)/100
    elif time == 'tomax':
        EPS = 1e-7
        data_lpmt[:,:,:,1] = -EPS
        # Calculate min time for each event
        ev_max = merged_hits[['event', 'hitTime']].groupby('event').max()
        event2max = {id_: max_ for id_, max_ in zip(ev_max.index, ev_max.hitTime)}
        ev_len = merged_hits[['event', 'hitTime']].groupby('event').count()
        event2len = {id_: len_ for id_, len_ in zip(ev_len.index, ev_len.hitTime)}
        
        for event, mol0i, mol1i, time in tqdm_notebook(zip(merged_hits['event'], merged_hits['mol0i'], merged_hits['mol1i'], merged_hits['hitTime'])):
            event_id = event_to_id[event]
            data_lpmt[event_id][mol0i, mol1i][0] += 1 * 1000 / event2len[event]
            event_max = event2max[event]
            cur_max = data_lpmt[event_id][mol0i, mol1i][1] 
            if cur_max < ((event_max - time)/100):
                data_lpmt[event_id][mol0i, mol1i][1] = (event_max - time) / 100
    return data_lpmt, event_to_id
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    