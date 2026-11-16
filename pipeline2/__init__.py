"""Attempt-2 possession pipeline: better data layer + principled segmentation.

Variants:
  v1  attempt-1 features (pipeline.features) + new segmentation   (isolates segmenter gain)
  v2  full data layer (ball motion, physics fill, data-driven set pieces) + segmentation
  v3  v2 + broadcast-ATD ball fusion (if alignment gate passes)
"""
