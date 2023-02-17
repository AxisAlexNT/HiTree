from enum import Enum
import functools
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from frozendict import frozendict
from recordclass import RecordClass
from copy import deepcopy


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


class StripeDescriptor(RecordClass):
    stripe_id: np.int64
    stripe_length_bins: np.int64
    bin_weights: Optional[np.ndarray]

    @staticmethod
    def make_stripe_descriptor(
            stripe_id: np.int64,
            stripe_length_bins: np.int64,
            bin_weights: Optional[np.ndarray] = None
    ) -> 'StripeDescriptor':
        return StripeDescriptor(
            stripe_id,
            stripe_length_bins,
            bin_weights
        )

    def __eq__(self, o: object) -> bool:
        if isinstance(o, StripeDescriptor):
            return (
                self.stripe_id,
                self.stripe_length_bins,
            ) == (
                o.stripe_id,
                o.stripe_length_bins,
            )
        return False


class ATUDirection(Enum):
    FORWARD = 1
    REVERSED = 0


class ATUDescriptor(RecordClass):
    stripe_descriptor: StripeDescriptor
    start_index_in_stripe_incl: np.int64
    end_index_in_stripe_excl: np.int64
    direction: ATUDirection

    @staticmethod
    def make_atu_descriptor(
        stripe_descriptor: StripeDescriptor,
        start_index_in_stripe_incl: np.int64,
        end_index_in_stripe_excl: np.int64,
        direction: ATUDirection
    ) -> 'ATUDescriptor':
        assert (
            start_index_in_stripe_incl < end_index_in_stripe_excl
        ), f"All ATUs should have their start preceeding end ({start_index_in_stripe_incl} < {end_index_in_stripe_excl}), no empty ATUs are allowed, direction ({direction}) is controlled by ATUDirection"
        return ATUDescriptor(
            stripe_descriptor,
            start_index_in_stripe_incl,
            end_index_in_stripe_excl,
            direction
        )

    @staticmethod
    def clone_atu_descriptor(
        a: 'ATUDescriptor'
    ) -> 'ATUDescriptor':
        return ATUDescriptor(
            a.stripe_descriptor,
            deepcopy(a.start_index_in_stripe_incl),
            deepcopy(a.end_index_in_stripe_excl),
            deepcopy(a.direction)
        )

    def clone(self) -> 'ATUDescriptor':
        return ATUDescriptor.clone_atu_descriptor(self)

    def __eq__(self, o: object) -> bool:
        if isinstance(o, ATUDescriptor):
            return (
                self.stripe_descriptor,
                self.start_index_in_stripe_incl,
                self.end_index_in_stripe_excl,
                self.direction
            ) == (
                o.stripe_descriptor,
                o.start_index_in_stripe_incl,
                o.end_index_in_stripe_excl,
                o.direction
            )
        return False

    @staticmethod
    def merge(
        d1: 'ATUDescriptor',
        d2: 'ATUDescriptor'
    ) -> Tuple['ATUDescriptor', Optional['ATUDescriptor']]:
        if d1.stripe_descriptor.stripe_id == d2.stripe_descriptor.stripe_id and d1.direction == d2.direction:
            if d1.end_index_in_stripe_excl == d2.start_index_in_stripe_incl:
                assert (
                    d1.start_index_in_stripe_incl < d2.end_index_in_stripe_excl
                ), "L start < R end??"
                return ATUDescriptor.make_atu_descriptor(
                    d1.stripe_descriptor,
                    d1.start_index_in_stripe_incl,
                    d2.end_index_in_stripe_excl,
                    d1.direction
                ), None
            elif d2.end_index_in_stripe_excl == d1.start_index_in_stripe_incl:
                return ATUDescriptor.merge(d1=d2, d2=d1)
        return d1, d2

    @staticmethod
    def reduce(
        atus: Iterable['ATUDescriptor']
    ) -> List['ATUDescriptor']:
        if len(atus) == 0:
            return []

        def reduce_fn(merged: List[ATUDescriptor], atu: ATUDescriptor) -> List[ATUDescriptor]:
            assert (
                len(merged) > 0
            ), "At least one element must be added by initial condition"
            d1, d2 = ATUDescriptor.merge(merged[-1], atu)
            merged[-1] = d1
            if d2 is not None:
                merged.append(d2)
            return merged

        return functools.reduce(reduce_fn, atus[1:], [atus[0]])


class ScaffoldDescriptor(RecordClass):
    scaffold_id: np.int64
    scaffold_name: str
    scaffold_borders: Optional[ScaffoldBorders]
    scaffold_direction: ScaffoldDirection
    spacer_length: int = 1000


class ContigDescriptor(RecordClass):
    contig_id: np.int64
    contig_name: str
    # direction: ContigDirection
    contig_length_at_resolution: frozendict  # Dict[np.int64, np.int64]
    # scaffold_id: Optional[np.int64]
    presence_in_resolution: frozendict
    # TODO: Decide how mapping Contig -> ATU and ATU -> Stripe is organized by ATL
    # Should ATUs know their corresponding length in bins/bps?
    # This implementation is not useful in case contig split occurrs:
    atus: Dict[np.int64, List[ATUDescriptor]]
    atu_prefix_sum_length_bins: Dict[np.int64, np.ndarray]

    @staticmethod
    def make_contig_descriptor(
            contig_id: np.int64,
            contig_name: str,
            # direction: ContigDirection,
            contig_length_bp: np.int64,
            contig_length_at_resolution: Dict[np.int64, np.int64],
            contig_presence_in_resolution: Dict[np.int64, ContigHideType],
            atus: Dict[np.int64, List[ATUDescriptor]],
            # scaffold_id: Optional[np.int64] = None
    ) -> 'ContigDescriptor':
        assert (
            0 not in contig_length_at_resolution.keys()
        ), "There should be no resolution 1:0 as it is used internally to store contig length in base pairs"
        new_contig_length_at_resolution = frozendict(
            {**contig_length_at_resolution, **{np.int64(0): contig_length_bp}})
        return ContigDescriptor(
            contig_id=contig_id,
            contig_name=contig_name,
            # direction,
            contig_length_at_resolution=new_contig_length_at_resolution,
            # scaffold_id=scaffold_id,
            presence_in_resolution=frozendict({**contig_presence_in_resolution, **
                       {np.int64(0): ContigHideType.FORCED_SHOWN}}),
            atus=atus,
            atu_prefix_sum_length_bins={
                resolution: np.cumsum(
                    tuple(
                        map(
                            lambda atu: atu.end_index_in_stripe_excl - \
                            atu.start_index_in_stripe_incl, atus[resolution]
                        )
                    ), dtype=np.int64)
                for resolution in contig_length_at_resolution.keys()
            }
        )

    def __eq__(self, o: object) -> bool:
        if isinstance(o, ContigDescriptor):
            return (
                self.contig_id,
                # self.direction,
                self.contig_length_at_resolution,
                self.atus,
                self.atu_prefix_sum_length_bins
            ) == (
                o.contig_id,
                # o.direction,
                o.contig_length_at_resolution,
                o.atus,
                o.atu_prefix_sum_length_bins
            )
        return False


class FinalizeRecordType(Enum):
    CONTIG_NOT_IN_SCAFFOLD = 0
    SCAFFOLD = 1
