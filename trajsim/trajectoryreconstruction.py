"""Trajectory Reconstruction methods.

Methods that get a non-regular-stepped trajectory and output a fixed-step trajectory.
"""
# Modules

# Parameters

# Methods

# Classes
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