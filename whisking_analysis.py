# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 18:33:54 2021

@author: m.debritovanvelze
"""
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import statistics

import binary_calculations

        
def process_whisking(file_path, settings, rec_points=9000):
    """
    Marcel van Velze (m.debritovanvelze@icm-institute.org)
    2021.10.28
    
    Function that loads the output of facemap, resamples the data to the same 
    length as the suite2p data and outputs a dictionary with whisking data

    Parameters
    ----------
    file_path : .npy file
        File containing the analyzed data from facemap
    rec_points : int
        The number of frames in the suite2p data.
        Used to downsample the data if needed.
        The default is 9000.

    Returns
    -------
    'path_analysis_file': path to facemap file
    'original_trace': output of facemap
    'resampled_trace': whisking trace resampled to 2P data
    'filtered_trace': filtered whisking trace
    'normalized_trace': normalized traces, min max
    'original_binary_whisking': binary whisking of original data
    'binary_whisking': binary whisking with short events removed 
    'location_bouts': location of bouts
    'duration bouts': duration of bouts
    """
    
    
    whisking_data = np.load(file_path, allow_pickle=True).item()
    
    if 'motion' in whisking_data:
        whisker_motion = np.copy(whisking_data['motion'][1])
        len_rec = len(whisker_motion)
        
        # Down sample data to fit 2P data
        if len_rec != rec_points:
            if len_rec > rec_points:
                new_whisker_motion = signal.resample(whisker_motion, rec_points)
                #print('-- Whisker data resampled to {n_points}'.format(n_points = settings['N_samples']))    
            else:
                new_whisker_motion = signal.resample(whisker_motion, rec_points)
                #print('-- Whisker data resampled to {n_points}'.format(n_points = settings['N_samples'])) 
        else:
            new_whisker_motion = np.copy(whisker_motion)
            #print('No resampling needed')
        
        # Filter data using gaussian filter
        filtered_whisking = gaussian_filter1d(new_whisker_motion, settings['whisking']['sigma'])
        
        # Normalize data using minmax method
        normalized = (filtered_whisking-min(filtered_whisking))/(max(filtered_whisking)-min(filtered_whisking))
        #normalized = (filtered_whisking-min(filtered_whisking))/((np.percentile(filtered_whisking, 90))-min(filtered_whisking))
        
        # Create binary whisking
        binary_whisking = (normalized > (settings['whisking']['percentile']/100)) * 1
        bool_binary_whisking = (normalized > (settings['whisking']['percentile']/100))
        
        # Remove short events
        if settings['whisking']['remove short bouts'] == True:
            new_binary_whisking = binary_calculations.remove_short_events(binary_whisking, settings['fs'], settings['whisking']['whisk_min_duration'])
        
            # Join events
            if settings['whisking']['join bouts'] == True:
                new_binary_whisking = binary_calculations.remove_short_interevent_periods(new_binary_whisking, settings['fs'], settings['whisking']['whisk_max_inter_bout'])
        
        else: 
            new_binary_whisking = np.copy(binary_whisking)
        
        dic = { 'path_analysis_file': file_path,
                'original_trace': whisker_motion,
                'resampled_trace': new_whisker_motion,
                'filtered_trace': filtered_whisking,
                'normalized_trace': normalized,
                'original_binary_whisking': binary_whisking,
                'binary_whisking': new_binary_whisking,
                } 
            
        change = list(np.where(np.diff(new_binary_whisking,prepend=np.nan))[0])    
        if len(change) == 1:
            dic['duration bouts'] = []
            dic['location_bouts'] = []
            dic['percentage_whisking'] = np.count_nonzero(new_binary_whisking)/len(new_binary_whisking)
            dic['mean event duration'] = []
            dic['max event duration'] = []
                  
        else:
            delta, loc = binary_calculations.calc_event_duration(new_binary_whisking)
            dic['duration bouts'] = delta
            dic['location_bouts'] = loc
            dic['percentage_whisking'] = np.count_nonzero(new_binary_whisking)/len(new_binary_whisking)
            dic['mean event duration'] = statistics.mean(delta)/settings['fs']
            dic['max event duration'] = max(delta)/settings['fs']
        
        return dic
    return {}

def whisking_only(binary_whisking, binary_locomotion, settings):
    binary_whisking_only = []
    for w, r in zip(binary_whisking, binary_locomotion):
        if w == 1 and r == 0:
            binary_whisking_only.append(1)
        else:
            binary_whisking_only.append(0)
    binary = binary_calculations.remove_short_events(binary_whisking_only, settings['fs'], settings['whisking']['whisk_min_duration'])
    if np.count_nonzero(binary) < 1:
        return []
    else:
        delta, loc = binary_calculations.calc_event_duration(binary)
        dic = {'binary': binary,
               'bout_duration': delta,
               'bout_location': loc}
        return dic
       
       
       
       
       
       
       
       