"""Trajectory Hashing methods.

Methods to create hashes for each trajectory for fast comparison of trajectories.
"""
# Modules
import numpy as np
import pandas as pd

# Parameters


# Methods

# Classes
## Base
class TrajectoryHasherBase:
    """Base trajectory hasher class.
    
    Methods to implement:
    - _initialize
    - hash
    """
    # Constructor
    def __init__( self, name, **params ):
        # Parameters
        self.name = name
        self.params = params
        # Initialize
        self._initialize()
        # Return
        return
    # Initialize
    def _initialize( self ):
        Exception( '<initialize@TrajectoryHasherBase> : not implemented!' )
        return
    # Bucketize
    def hash( self, df_trajectory_processed ):
        Exception( '<hash@TrajectoryHasherBase> : not implemented!' )
        return # df_user_hashes
## Jacard Estimation
class TrajectoryHasherJacardEstimation(TrajectoryHasherBase):
    """Trajectory Hashing using Jacard Similarity Estimation.
    """
    # Constructor
    def __init__(
            self,
            name,
            n_hashes,
            **params
        ):
        # Parameters
        self.n_hashes = n_hashes
        super().__init__( name, **params )
        # Return
        return
    # Initialize
    def _initialize( self ):
        return
    # Hash
    def hash( self, df_trajectory_processed, df_type = 'pandas' ):
        # Assert Implemented Methods
        assert df_type in { 'pandas' }, 'hash@<TrajectoryHasherJacardEstimation>: df_type = "{}" is not implemented!'.format( df_type )
        # Hash
        if( df_type == 'pandas' ):
            # Bounds
            id_timestamp_min = df_trajectory_processed['id_timestamp'].min()
            id_timestamp_max = df_trajectory_processed['id_timestamp'].max()
            # Select ID TimeStamp for Hashes
            id_timestamps_selected = set(np.random.choice(
                np.arange( id_timestamp_min, id_timestamp_max+1 ),
                self.n_hashes,
                replace = False
            ))
            # Filter
            df_result = df_trajectory_processed[
                df_trajectory_processed['id_timestamp'].map(
                    lambda x:  x in id_timestamps_selected
                )
            ].copy().sort_values(['id_user', 'id_timestamp'])
            # Index Locations
            location_indices = []
            lat_lng_to_idx = dict()
            for lat, lng in zip( df_result['lat'], df_result['lng'] ):
                key = (lat,lng)
                if( not key in lat_lng_to_idx ):
                    lat_lng_to_idx[key] = len(lat_lng_to_idx)
                # Add New Index
                location_indices.append( lat_lng_to_idx[key] )
            df_result['location_indices'] = location_indices
            # Calculate Hashes
            df_hashes = None
            for i, id_timestamp in enumerate(id_timestamps_selected):
                if( df_hashes is None ):
                    df_hashes = df_result[
                        df_result['id_timestamp'] == id_timestamp
                    ][['id_user', 'location_indices']].copy()
                else:
                    df_hashes['location_indices'] = df_result[
                        df_result['id_timestamp'] == id_timestamp
                    ]['location_indices'].values
                # Rename New Column
                colname = 'hash_{}'.format( i )
                df_hashes.rename(columns={'location_indices':colname}, inplace = True)
        # Return
        return( df_hashes )