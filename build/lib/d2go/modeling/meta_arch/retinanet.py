#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


from d2go.registry.builtin import META_ARCH_REGISTRY
from detectron2.modeling import RetinaNet as _RetinaNet


# Re-register D2's meta-arch in D2Go with updated APIs
@META_ARCH_REGISTRY.register()
class RetinaNet(_RetinaNet):
    def prepare_for_export(self, cfg, inputs, predictor_type):
        raise NotImplementedError
