"""Trajectory Bucketing methods.

Methods to bucket trajectory data into fixed intervals.
"""
# Modules
import numpy as np
import pandas as pd

# Parameters
TIMESTAMP_BASE = 946684800

# Methods

# Classes
## Base
class TrajectoryBucketterBase:
    """Base trajectory bucketter class.
    
    Methods to implement:
    - _initialize
    - bucketize
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
        Exception( '<initialize@TrajectoryBucketterBase> : not implemented!' )
        return
    # Bucketize
    def bucketize( self, df_trajectory_raw_latlng ):
        Exception( '<bucketize@TrajectoryBucketterBase> : not implemented!' )
        return # df_trajectory_bucketed
## Time Mean
class TrajectoryBucketterMean( TrajectoryBucketterBase ):
    """Time Duration Averages over Buckets.
    
    Base TimeStamp: 946684800 -> 2020/01/01/ 00:00:00
    """
    # Constructor
    def __init__(
            self,
            name,
            bucket_size_seconds = 3600,
            **params
        ):
        # Parameters
        self.bucket_size_seconds = bucket_size_seconds
        # Super Initi
        super().__init__( name, **params )
        # Return
        return
    # Initialize
    def _initialize( self ):
        return
    # Bucketize
    def bucketize( self, df_trajectory_raw_latlng, df_type = 'pandas' ):
        # Assert Implemented Methods
        assert df_type in { 'pandas' }, 'bucketize@<TrajectoryBucketterMean>: df_type = "{}" is not implemented!'.format( df_type )
        # Bucketize
        if( df_type == 'pandas' ):
            ####################################################################
            # Re-Order
            df_result = df_trajectory_raw_latlng.copy().sort_values(['id_user', 'timestamp'])
            ####################################################################
            # Add Time Lagged Data
            df_result['id_user_next'] = np.roll(df_result['id_user'], -1)
            df_result['timestamp_next'] = np.roll(df_result['timestamp'], -1)
            ####################################################################
            # Duration Between Transitions
            df_result['duration_to_next'] = df_result['timestamp_next'] - df_result['timestamp']
            ####################################################################
            # Calculate Durations
            ## Durations
            _durations = df_result['duration_to_next'].values * (
                (df_result['id_user'] == df_result['id_user_next']).values.astype(np.float)
            ) / 2
            ### Last Row is Just rolled version of first row, so it should be removed!
            _durations[-1] = 0
            ## Assign
            df_result['duration_at_location'] = _durations + np.roll( _durations, 1 )
            ####################################################################
            # Add Time Index
            df_result['id_timestamp'] = (df_result['timestamp'] - TIMESTAMP_BASE) // self.bucket_size_seconds
            # Add Lat, Lng Weighted
            df_result['lat_weighted'] = df_result['lat'] * df_result['duration_at_location']
            df_result['lng_weighted'] = df_result['lng'] * df_result['duration_at_location']
            ####################################################################
            # Aggregate
            df_result = df_result.groupby(['id_user', 'id_timestamp']).agg({
                'lat_weighted': 'sum',
                'lng_weighted': 'sum',
                'duration_at_location': 'sum'
            })
            df_result['lat'] = df_result['lat_weighted'] / df_result['duration_at_location']
            df_result['lng'] = df_result['lng_weighted'] / df_result['duration_at_location']
            # Select & Re-Set Index
            df_result = df_result[['lat', 'lng']].reset_index()
            df_result = df_result[['id_user', 'id_timestamp', 'lat', 'lng']]
        # Return
        return( df_result )
        































