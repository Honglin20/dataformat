"""SQ-Format comparison study (extension of experiments/fourbit).

Reuses the 4-bit study's ``part1.run_all`` / ``part2.run`` /
``reporter.generate_report`` pipeline by passing an SQ-Format–specific
:class:`experiments.fourbit.config.FourBitConfig`.  All study-specific
wiring (16 SQ-Format cells × 3 transforms, metrics set, Y-quantisation
and QuantizedMHA enabled) lives in :mod:`experiments.sqformat.config`.
"""
