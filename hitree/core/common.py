from enum import Enum
from typing import Dict, Optional

import numpy as np
from frozendict import frozendict
from recordclass import RecordClass


class LengthUnit(Enum):
    """
    Enum that describes length unit for coordinates.
    """

    BASE_PAIRS = 0
    PIXELS = 1


class ContigDirection(Enum):
    """
    Enum that describes contig's orientation in the model.
    """

    FORWARD = 1
    REVERSED = 0


class StripeDirection(Enum):
    """
    Enum that describes orientation of stripe in the model.
    """

    FORWARD = 1
    REVERSED = 0


class ScaffoldDirection(Enum):
    """
    Enum that describes orientation of scaffold in the model.
    """

    FORWARD = 1
    REVERSED = 0


class ContigHideType(Enum):
    """
    Describes the visibility of contig. Will be used in the next release.
    """

    AUTO_HIDDEN = 0
    AUTO_SHOWN = 1
    FORCED_HIDDEN = 2
    FORCED_SHOWN = 3


class ScaffoldBorders(RecordClass):
    """
    Describes bordering contigs of the scaffold.
    """

    start_contig_id: np.int64
    end_contig_id: np.int64


class StripeDescriptor(RecordClass):
    """
    Describes the stripe: a contiguous group of bins that belong to the same contig. It may be thought as a median
    hierarchy layer of nucleotides between bins and contigs.
    """

    stripe_id: np.int64
    stripe_length_bins: np.int64
    stripe_length_bp: np.int64
    direction: StripeDirection
    contig_id: np.int64

    @staticmethod
    def make_stripe_descriptor(
            stripe_id: np.int64,
            stripe_length_bins: np.int64,
            stripe_length_bp: np.int64,
            direction: StripeDirection,
            contig_id: np.int64,
    ) -> 'StripeDescriptor':
        return StripeDescriptor(
            stripe_id,
            stripe_length_bins,
            stripe_length_bp,
            direction,
            contig_id
        )

    def __eq__(self, o: object) -> bool:
        if isinstance(o, StripeDescriptor):
            o: StripeDescriptor = o
            return (
                       self.stripe_id,
                       self.stripe_length_bins,
                       self.stripe_length_bp,
                       self.direction,
                       self.contig_id,
                   ) == (
                       o.stripe_id,
                       o.stripe_length_bins,
                       o.stripe_length_bp,
                       o.direction,
                       o.contig_id,
                   )
        return False


class ContigDescriptor(RecordClass):
    """
    Describes contig stored in model. Note: contig names should be stored separately.
    """
    contig_id: np.int64
    direction: ContigDirection
    # contig_length_bp: int # is stored at resolution 1:0
    contig_length_at_resolution: frozendict  # Dict[np.int64, np.int64]
    scaffold_id: Optional[np.int64]
    presence_in_resolution: frozendict

    @staticmethod
    def make_contig_descriptor(
            contig_id: np.int64,
            direction: ContigDirection,
            contig_length_bp: np.int64,
            contig_length_at_resolution: Dict[np.int64, np.int64],
            contig_presence_in_resolution: Dict[np.int64, ContigHideType],
            scaffold_id: Optional[np.int64] = None
    ) -> 'ContigDescriptor':
        assert (
                0 not in contig_length_at_resolution.keys()
        ), "There should be no resolution 1:0 as it is used internally to store contig length in base pairs"
        contig_length_at_resolution = frozendict({**contig_length_at_resolution, **{0: contig_length_bp}})
        return ContigDescriptor(
            contig_id,
            direction,
            contig_length_at_resolution,
            scaffold_id,
            frozendict(contig_presence_in_resolution)
        )

    def __eq__(self, o: object) -> bool:
        if isinstance(o, ContigDescriptor):
            return (
                       self.contig_id,
                       self.direction,
                       self.contig_length_at_resolution,
                       self.presence_in_resolution,
                       self.scaffold_id
                   ) == (
                       o.contig_id,
                       o.direction,
                       o.contig_length_at_resolution,
                       o.presence_in_resolution,
                       o.scaffold_id
                   )
        return False


class ScaffoldDescriptor(RecordClass):
    """
    Describes scaffold in our model.
    """

    scaffold_id: np.int64
    scaffold_name: str
    scaffold_borders: Optional[ScaffoldBorders]
    scaffold_direction: ScaffoldDirection
    spacer_length: int = 1000


class FinalizeRecordType(Enum):
    """
    This enum is used to create finalization records for FASTAProcessor.
    """

    CONTIG_NOT_IN_SCAFFOLD = 0
    SCAFFOLD = 1
