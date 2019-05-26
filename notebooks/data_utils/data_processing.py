import pandas as pd
import numpy as np
from pyproj import Proj, transform
from tqdm import tqdm_notebook


def lat(x,y,z):
    return np.arcsin(z/np.sqrt(x**2 + y**2 + z**2))

def lon(x,y,z):
    return np.arctan2(y,x)


def get_data_2dprojection(lpmt_hits, spmt_hits, pos, true_info, edge_size0=226, edge_size1=111, use_spmt=False, time=None, proj='sin'):
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
    if use_spmt:
        channels += 1
        spmt_index = 2
        
    latt  = np.array(list(map(lambda el: lat(el[0],el[1],el[2]), zip(pos['pmt_x'], pos['pmt_y'], pos['pmt_z']))))
    lontt = np.array(list(map(lambda el: lon(el[0],el[1],el[2]), zip(pos['pmt_x'], pos['pmt_y'], pos['pmt_z']))))
    
    if proj == 'sin':
        proj0min = -np.pi
        proj0max = np.pi
        proj1min = -np.pi/2
        proj1max = np.pi/2

        proj0 = lontt * np.cos(latt)
        proj1 = latt
    
    pos['proj0'] = np.round((proj0 - proj0min) / (proj0max - proj0min) * (edge_size0-1)).astype(int)
    pos['proj1'] = np.round((proj1 - proj1min) / (proj1max - proj1min) * (edge_size1-1)).astype(int)
        
    merged_hits = pd.merge(lpmt_hits, pos, left_on='pmtID', right_on='pmt_id')
    if use_spmt:
        merged_hits_s = pd.merge(spmt_hits, pos, left_on='pmtID', right_on='pmt_id')
        print("S_shape", merged_hits_s.shape)
    
    n = len(lpmt_hits['event'].unique())
    data_lpmt = np.zeros((n, edge_size0, edge_size1, channels))#, dtype='float32')
    
    event_to_id = {x:y for y, x in enumerate(sorted(merged_hits['event'].unique()))}
    
    print("Starting cycle...")
    if time is None:
        for event, mol0i, mol1i in tqdm_notebook(zip(merged_hits['event'], merged_hits['proj0'], merged_hits['proj1'])):
            data_lpmt[event_to_id[event]][mol0i, mol1i] += 1
    elif time == 'min':
        if not use_spmt:
            EPS = 1e-7
            data_lpmt[:,:,:,1] = -EPS
            # Calculate min time for each event
            ev_min = merged_hits[['event', 'hitTime']].groupby('event').min()
            event2min = {id_: min_ for id_, min_ in zip(ev_min.index, ev_min.hitTime)}
        
            for event, mol0i, mol1i, time in tqdm_notebook(zip(merged_hits['event'], merged_hits['proj0'], merged_hits['proj1'], merged_hits['hitTime'])):
                event_id = event_to_id[event]
                data_lpmt[event_id][mol0i, mol1i][0] += 1
                event_min = event2min[event]
                cur_min = data_lpmt[event_id][mol0i, mol1i][1]
                if cur_min < 0 or cur_min > (time - event_min):
                    data_lpmt[event_id][mol0i, mol1i][1] = time - event_min
        elif use_spmt:
            EPS = 1e-7
            data_lpmt[:,:,:,1] = -EPS
            data_lpmt[:,:,:,spmt_index] = -EPS

            # Calculate min time for each event
            min_merge = pd.concat([lpmt_hits, spmt_hits])
            ev_min = min_merge[['event', 'hitTime']].groupby('event').min()
            event2min = {id_: min_ for id_, min_ in zip(ev_min.index, ev_min.hitTime)}
            
            for event, mol0i, mol1i, time in tqdm_notebook(zip(merged_hits['event'], merged_hits['proj0'], merged_hits['proj1'], merged_hits['hitTime'])):
                event_id = event_to_id[event]
                data_lpmt[event_id][mol0i, mol1i][0] += 1
                event_min = event2min[event]
                cur_min = data_lpmt[event_id][mol0i, mol1i][1]
                if cur_min < 0 or cur_min > (time - event_min):
                    data_lpmt[event_id][mol0i, mol1i][1] = time - event_min
            
            for event, mol0i, mol1i, time in tqdm_notebook(zip(merged_hits_s['event'], merged_hits_s['proj0'], merged_hits_s['proj1'], merged_hits_s['hitTime'])):
                event_id = event_to_id[event]
                data_lpmt[event_id][mol0i, mol1i][spmt_index] += 1
                event_min = event2min[event]
                cur_min = data_lpmt[event_id][mol0i, mol1i][1]
                if cur_min < 0 or cur_min > (time - event_min):
                    data_lpmt[event_id][mol0i, mol1i][1] = time - event_min
            
    """
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
    """
    
    return data_lpmt, event_to_id