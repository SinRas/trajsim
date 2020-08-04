"""Trajectory Bucketing methods.

Methods to bucket trajectory data into fixed intervals.
"""
# Modules

# Parameters

# Methods

# Classes
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