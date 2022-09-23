from enum import Enum
from typing import Dict, Optional

import numpy as np
from frozendict import frozendict
from recordclass import RecordClass


class QueryLengthUnit(Enum):
    BASE_PAIRS = 0
    BINS = 1
    PIXELS = 2


class ContigDirection(Enum):
    FORWARD = 1
    REVERSED = 0


class ScaffoldDirection(Enum):
    FORWARD = 1
    REVERSED = 0


class ContigHideType(Enum):
    AUTO_HIDDEN = 0
    AUTO_SHOWN = 1
    FORCED_HIDDEN = 2
    FORCED_SHOWN = 3


class ScaffoldBorders(RecordClass):
    start_contig_id: np.int64
    end_contig_id: np.int64
    
class NormalizationType(Enum):
    LINEAR = 0
    LOG2 = 1
    LOG10 = 2
    COOLER_BALANCE = 3

class ContigDescriptor(RecordClass):
    contig_id: np.int64
    contig_name: str
    direction: ContigDirection
    # contig_length_bp: int # is stored at resolution 1:0
    contig_length_at_resolution: frozendict  # Dict[np.int64, np.int64]
    scaffold_id: Optional[np.int64]
    presence_in_resolution: frozendict

    # TODO: Is shown in resolutions

    @staticmethod
    def make_contig_descriptor(
            contig_id: np.int64,
            contig_name: str,
            direction: ContigDirection,
            contig_length_bp: np.int64,
            contig_length_at_resolution: Dict[np.int64, np.int64],
            contig_presence_in_resolution: Dict[np.int64, ContigHideType],
            scaffold_id: Optional[np.int64] = None
    ) -> 'ContigDescriptor':
        assert (
                0 not in contig_length_at_resolution.keys()
        ), "There should be no resolution 1:0 as it is used internally to store contig length in base pairs"
        contig_length_at_resolution = frozendict({**contig_length_at_resolution, **{np.int64(0): contig_length_bp}})
        return ContigDescriptor(
            contig_id,
            contig_name,
            direction,
            contig_length_at_resolution,
            scaffold_id,
            frozendict({**contig_presence_in_resolution, **{np.int64(0): ContigHideType.FORCED_SHOWN}})
        )

    def __eq__(self, o: object) -> bool:
        if isinstance(o, ContigDescriptor):
            return (
                       self.contig_id,
                       self.direction,
                       self.contig_length_at_resolution
                   ) == (
                       o.contig_id,
                       o.direction,
                       o.contig_length_at_resolution
                   )
        return False


class StripeDescriptor(RecordClass):  # NamedTuple):
    stripe_id: np.int64
    stripe_length_bins: np.int64
    stripe_length_bp: np.int64
    contig_descriptor: ContigDescriptor
    bin_weights: Optional[np.ndarray]

    @staticmethod
    def make_stripe_descriptor(
            stripe_id: np.int64,
            stripe_length_bins: np.int64,
            stripe_length_bp: np.int64,
            contig_descriptor: ContigDescriptor,
            bin_weights: Optional[np.ndarray] = None
    ) -> 'StripeDescriptor':
        return StripeDescriptor(
            stripe_id,
            stripe_length_bins,
            stripe_length_bp,
            contig_descriptor,
            bin_weights
        )

    def __eq__(self, o: object) -> bool:
        if isinstance(o, StripeDescriptor):
            return (
                       self.stripe_id,
                       self.stripe_length_bins,
                       self.stripe_length_bp,
                       self.contig_descriptor,
                   ) == (
                       o.stripe_id,
                       o.stripe_length_bins,
                       o.stripe_length_bp,
                       o.contig_descriptor,
                   )
        return False


class ScaffoldDescriptor(RecordClass):
    scaffold_id: np.int64
    scaffold_name: str
    scaffold_borders: Optional[ScaffoldBorders]
    scaffold_direction: ScaffoldDirection
    spacer_length: int = 1000


class FinalizeRecordType(Enum):
    CONTIG_NOT_IN_SCAFFOLD = 0
    SCAFFOLD = 1
