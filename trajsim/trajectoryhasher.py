"""Trajectory Hashing methods.

Methods to create hashes for each trajectory for fast comparison of trajectories.
"""
# Modules

# Parameters

# Methods

# Classes
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