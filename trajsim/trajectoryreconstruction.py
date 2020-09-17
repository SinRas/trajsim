"""Trajectory Reconstruction methods.

Methods that get a non-regular-stepped trajectory and output a fixed-step trajectory.
"""
# Modules
import numpy as np
import pandas as pd

# Parameters

# Methods

# Classes
## Base
class TrajectoryReconstructorBase:
    """Base trajectory reconstruction class.
    
    Methods to implement:
    - _initialize
    - reconstruct
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
        Exception( '<initialize@TrajectoryReconstructorBase> : not implemented!' )
        return
    # Reconstruct
    def reconstruct( self, df_trajectory_bucketed ):
        Exception( '<reconstruct@TrajectoryReconstructorBase> : not implemented!' )
        return # df_trajectory_bucketed_reconstructed
## Linear Reconstructor
class TrajectoryReconstructorLinear( TrajectoryReconstructorBase ):
    """Reconstruct or Regularize Trajectory Linearly (e.g. Linear Imputation).
    
    The gap between timestamp ids is filled by linear interpolation of lat/lng
    before and after that.
    """
    # Constructor
    def __init__( self, name, **params ):
        super().__init__( name, **params )
        return
    # Initialize
    def _initialize( self ):
        return
    # Reconstruct
    def reconstruct( self, df_trajectory_bucketed, df_type = 'pandas' ):
        # Assert Implemented Methods
        assert df_type in { 'pandas' }, 'reconstruct@<TrajectoryReconstructorLinear>: df_type = "{}" is not implemented!'.format( df_type )
        # Reconstruct
        if( df_type == 'pandas' ):
            # Sort and Copy
            df_result = df_trajectory_bucketed.copy().sort_values(['id_user', 'id_timestamp'])
            df_result.reset_index( drop = True, inplace = True )
            # ID TimeStamp Bounds
            id_timestamp_min = df_result['id_timestamp'].min()
            id_timestamp_max = df_result['id_timestamp'].max()
            # New Data
            data = []
            _n = len(df_result)
            for i, (id_user, id_timestamp, lat, lng) in enumerate(zip(
                    df_result['id_user'], df_result['id_timestamp'],
                    df_result['lat'], df_result['lng']
                )):
                # First Row and Missing TimeStamps
                if( i == 0 and id_timestamp > id_timestamp_min ):
                    data.extend([
                        [id_user, t, lat, lng] for t in range(
                            id_timestamp_min, id_timestamp
                        )
                    ])
                # Get Next Row if Possible
                if( i < (_n-1) ):
                    # Next
                    id_user_next = df_result['id_user'][i+1]
                    id_timestamp_next = df_result['id_timestamp'][i+1]
                    lat_next = df_result['lat'][i+1]
                    lng_next = df_result['lng'][i+1]
                    # Linear Interpolation
                    if( id_user == id_user_next and (id_timestamp+1) < id_timestamp_next ):
                        # Line Slopes
                        slope_lat = ( lat_next - lat ) / ( id_timestamp_next - id_timestamp )
                        slope_lng = ( lng_next - lng ) / ( id_timestamp_next - id_timestamp )
                        # Update
                        for t in range( id_timestamp+1, id_timestamp_next ):
                            dt = ( t - id_timestamp )
                            data.append([
                                id_user, t, lat + dt * slope_lat, lng + dt * slope_lng
                            ])
                    elif( id_user != id_user_next ):
                        # New User Encountered
                        data.extend([
                            [id_user, t, lat, lng] for t in range(
                                id_timestamp, id_timestamp_max + 1
                            )
                        ])
                        data.extend([
                            [id_user_next, t, lat, lng] for t in range(
                                id_timestamp_min, id_timestamp_next
                            )
                        ])
                else:
                    # Last Row and Missing Following TimeStamps
                    if( id_timestamp < id_timestamp_max ):
                        data.extend([
                            [id_user, t, lat, lng] for t in range(
                                (id_timestamp+1), (id_timestamp_max+1)
                            )
                        ])
            # Append New Data
            df_result = pd.concat([
                df_result,
                pd.DataFrame(
                    data,
                    columns = [ 'id_user', 'id_timestamp', 'lat', 'lng' ]
                )
            ]).sort_values(['id_user', 'id_timestamp']).reset_index( drop = True )
        # Return
        return( df_result )