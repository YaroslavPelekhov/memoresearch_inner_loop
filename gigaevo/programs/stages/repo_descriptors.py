from __future__ import annotations

import json
from typing import Any

from gigaevo.programs.metrics.context import VALIDITY_KEY, MetricsContext
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.cache_handler import NO_CACHE
from gigaevo.programs.stages.common import FloatDictContainer, StageIO, StringContainer
from gigaevo.programs.stages.stage_registry import StageRegistry
from gigaevo.repo_harness.descriptors import (
    REPO_DESCRIPTORS_METADATA_KEY,
    apply_archive_roles,
    extract_repo_descriptors,
    initial_archive_roles,
)


class RepoDescriptorInputs(StageIO):
    metrics: FloatDictContainer | None
    reflection: StringContainer | None


@StageRegistry.register(
    description="Extract cheap repo-harness descriptors and preliminary archive roles"
)
class RepoDescriptorStage(Stage):
    """Attach deterministic repo descriptors before archive admission.

    This stage intentionally runs every refresh. Descriptor extraction is cheap,
    and lineage fields such as child_count can change even when benchmark and
    reflection outputs are input-cache hits.
    """

    InputsModel = RepoDescriptorInputs
    OutputModel = StringContainer
    cache_handler = NO_CACHE

    def __init__(
        self,
        *,
        metrics_context: MetricsContext,
        primary_key: str | None = None,
        validity_key: str = VALIDITY_KEY,
        min_quality_floor: float | None = None,
        regression_tolerance: float = 1.0e-9,
        enable_quarantine: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.metrics_context = metrics_context
        self.primary_key = primary_key
        self.validity_key = validity_key
        self.min_quality_floor = min_quality_floor
        self.regression_tolerance = regression_tolerance
        self.enable_quarantine = enable_quarantine

    async def compute(self, program: Program) -> StringContainer:
        descriptors = extract_repo_descriptors(
            program,
            metrics_context=self.metrics_context,
            primary_key=self.primary_key,
        )
        roles, reasons = initial_archive_roles(
            program,
            descriptors,
            validity_key=self.validity_key,
            min_quality_floor=self.min_quality_floor,
            regression_tolerance=self.regression_tolerance,
            enable_quarantine=self.enable_quarantine,
        )
        program.set_metadata(REPO_DESCRIPTORS_METADATA_KEY, descriptors)
        apply_archive_roles(
            program,
            roles,
            reasons=reasons,
            source="repo_descriptor_stage",
        )
        return StringContainer(
            data=json.dumps(
                {
                    "descriptors": descriptors,
                    "archive_roles": roles,
                    "archive_reasons": reasons,
                },
                sort_keys=True,
                default=str,
            )
        )
