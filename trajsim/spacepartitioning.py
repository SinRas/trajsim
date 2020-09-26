"""Space Partitioning Submodule.

Classes here provide functionality to get a matrix of points,
then split them based on different criterion.
"""

# Modules
import numpy as np

# Parameters

# Methods

# Classes
## Base
class SpacePartitionerBase:
    """Base Space Partitioning Class providing following functionalities.

    - Constructor!
    - Partition Based on given Leaf Size Limit
    """
    # Constructor
    def __init__( self, name, **params ):
        # Parameters
        self.name = name
        self.params = params
        # Initialization
        self._initialize()
        # Return
        return
    # Initialize
    def _initialize( self ):
        Exception( '<initialize@SpacePartitionerBase> : not implemented!' )
        return
    # Partition Points
    def partition_points( self, points ):
        Exception( '<partition_points@SpacePartitionerBase> : not implemented!' )
        return
    # Partition Recursively
    def partition_recursive( self ):
        pass
    # Traverse Once
    def _traverse_once_dict( self, _dict, _splitting_limit ):
        """Traverse dictionary once.
        This function acts *INPLACE*.
        Check all partitions and split if number of samples is higher than `_splitting_limit`.
        
        Parameters:
        -----------
        `_dict`            : (dict) partitions dictionary. nested dictionary
            with keys `partitioner` and partition labels.
        `_splitting_limit` : (int) minimum number of samples required for
            splitting to happen.
        
        Output:
        -------
        `n_split_size_max` : (int) maximum size of splits remaining.
        """
        # N Split Size Max
        n_split_size_max = -1
        # Base
        if( '' in _dict and not isinstance( _dict[''], dict ) ):
            for _case in ['']:
                # Partition Point
                _func, _cs, _ps = self.partition_points( _dict[_case] )
                # Update Dictionary
                _dict[_case] = {
                  'partitioner': _func,
                }
                _dict[_case].update(dict(zip( _cs, _ps )))
                # N Split Size Max
                n_split_size_max = max( n_split_size_max, max([ len(x) for x in _ps ]) )
            # Return
            return( n_split_size_max )
        # Recurse
        for _case in [ x for x in _dict if x != 'partitioner' ]:
            if( isinstance( _dict[_case], dict ) ):
                n_split_size_max = max( n_split_size_max, self._traverse_once_dict( _dict[_case], _splitting_limit ) )
            else:
                # Few Points
                if( len( _dict[_case] ) <= _splitting_limit ):
                    n_split_size_max = max( n_split_size_max, len( _dict[_case] ) )
                    continue
                _func, _cs, _ps = self.partition_points( _dict[_case] )
                _dict[_case] = {
                    'partitioner': _func,
                }
                _dict[_case].update(dict(zip( _cs, _ps )))
                # N Split Size Max
                n_split_size_max = max( n_split_size_max, max([ len(x) for x in _ps ]) )
        # Return
        return( n_split_size_max )
    # Traverse Once
    def traverse_once_points( self, points, splitting_limit ):
        # Create Dictionary
        _dict = {
            '': points
        }
        # Traverse Dictionary
        n_split_size_max = self._traverse_once_dict( _dict, splitting_limit )
        # Return
        return( _dict, n_split_size_max )
    # Traverse Till Convergence
    def traverse_till_convergence_points( self, points, splitting_limit ):
        # Create Dictionary
        _dict = {
            '': points
        }
        # Traverse Dictionary
        n_split_size_max = splitting_limit + 1
        n_split_size_max_prev = -1
        while( (n_split_size_max >= splitting_limit) and ( n_split_size_max != n_split_size_max_prev ) ):
            n_split_size_max_prev = n_split_size_max
            n_split_size_max = self._traverse_once_dict( _dict, splitting_limit )
        # Store
        self.partitions = _dict
        # Return
        return( _dict, n_split_size_max )
    # Fit
    def fit( self, points, splitting_limit ):
        _ = self.traverse_till_convergence_points( points, splitting_limit )
        return
    # Label Point
    def _label_point( self, _point ):
        # Current Dictionary
        _partitions = self.partitions
        _dict = _partitions if not '' in _partitions else _partitions['']
        # While True!
        _label = ''
        while( 'partitioner' in _dict ):
            _c = _dict['partitioner']( _point )
            _label += _c
            _dict = _dict[_c]
        # Return
        return( _label )
## Binary Angles Partitioner
class BinaryPartitioner( SpacePartitionerBase ):
    """Partition Points by Hyperplane at Center of Mass to maximize Sum of Scores.
    
    Maximize: \sum_i \score_{i,h}
    s.t. ||h|| = 1
    """
    # Constructor
    def __init__(
            self,
            name = "BinaryPartitioner",
            **params
        ):
        """Partition the space and calculate argmax of sum of all points' angles with 
        the given hyperplane by choosing hyper plane direction.
        """
        # Parameters
        super().__init__( name = name, **params )
    # Initialize
    def _initialize( self ):
        """Create thetas and rays (unit vectors in direction of thetas).
        """
        # Thetas & Rays
        self.thetas = np.pi * np.arange(self.params['radians_resolution'])/self.params['radians_resolution']
        self.rays = np.vstack( (np.cos(self.thetas)[None,:], np.sin(self.thetas)[None,:]) )
        # Return
        return
    # Score Rays
    def _score_rays( self, _points ):
        Exception( '<_score_rays@BinaryPartitioner> : not implemented!' )
        return
    # Partition Points in Space
    def partition_points( self, _points ):
        # Center
        _center = np.mean( _points, axis = 0 )
        # Scores
        _scores = self._score_rays( _points )
        # Ray
        _ray = self.rays[:,np.argmax( _scores )]
        # Parts
        _indices_front = np.matmul( _points - _center, _ray ) >= 0
        _parts = [
        _points[ _indices_front, : ],
        _points[ ~_indices_front, : ]
        ]
        # New Partitioner
        def _partitioner( _point ):
            return( 'f' if np.sum( (_point-_center) * _ray ) >= 0 else 'b' )
        # Cases
        _cases = [ 'f', 'b' ]
        # Return
        return( _partitioner, _cases, _parts )
## Binary Angles Partitioner
class BinaryAnglesPartitioner( BinaryPartitioner ):
    """Partition Points by Hyperplane at Center of Mass to maximize Sum of Angles.
    
    Maximize: \sum_i \theta_{i,h}
    s.t. \theta_{i,h} = <p_i,h> / ||p_i||, ||h|| = 1
    """

    # Constructor
    def __init__(
            self,
            name = "BinaryAnglesPartitioner",
            radians_resolution = 180
        ):
        """Partition the space and calculate argmax of sum of all points' angles with 
        the given hyperplane by choosing hyper plane direction.
        
        Parameters:
        -----------
        `name`               : (str) Name of the class instance for reporting purposes
        `radians_resolution` : (int) Number of partitions per half circle
        """
        # Parameters
        super().__init__( name = name, radians_resolution = radians_resolution )
    # Score Rays
    def _score_rays( self, _points ):
        # Center
        _center = np.mean( _points, axis = 0 )
        # Angles
        _angles_signed = np.matmul( _points - _center, self.rays ) *\
            (1 / np.linalg.norm( _points - _center, axis = 1 )[:,None])
        # Scores
        _scores = np.sum( np.abs(_angles_signed), axis = 0 )
        # Return
        return( _scores )
## Binary Angles Balanced Partitioner
class BinaryAnglesBalancedPartitioner( BinaryPartitioner ):
    """Partition Points by Hyperplane at Center of Mass to maximize Sum of Angles.
    
    Maximize: (\sum_i \theta_{i,h}) / \sqrt{ n^2_+ + n^2_- }
    s.t. \theta_{i,h} = <p_i,h> / ||p_i||, ||h|| = 1
    """
    # Constructor
    def __init__(
            self,
            name = "BinaryAnglesBalancedPartitioner",
            radians_resolution = 180
        ):
        """Partition the space and calculate argmax of sum of all points' angles
        divided by splitting signed count with  the given hyperplane by
        choosing hyper plane direction.
        """
        # Parameters
        super().__init__( name = name, radians_resolution = radians_resolution )
    # Score Rays
    def _score_rays( self, _points ):
        # Center
        _center = np.mean( _points, axis = 0 )
        # Angles
        _angles_signed = np.matmul( _points - _center, self.rays ) *\
            (1 / np.linalg.norm( _points - _center, axis = 1 )[:,None])
        # Count Denominators
        _count_denominator = np.sqrt( np.sum( (_angles_signed >= 0 ), axis = 0 )**2 + np.sum( (_angles_signed <= 0 ), axis = 0 )**2 )
        # Scores
        _scores = np.sum( np.abs(_angles_signed), axis = 0 ) / _count_denominator
        # Return
        return( _scores )














# 
# ################################################################
# 
# #################################################################
# import random
# r = lambda: random.randint(0,255)
# labels = [ label_point(partitions, p) for p in points ]
# colormap = dict(zip(
#     list(set(labels)),
#     [ '#%02X%02X%02X' % (r(),r(),r()) for _ in set(labels) ]
# ))
# 
# 
# 





















