"""Frozen whole-Tile analysis only; no fitted or device-derived admission Golden."""
import numpy as np
from vdeq_common import models,KEY_DOTS


def analyze(dots,words,actual,expected):
 summary,elements,bad=models.analyze(dots,words,actual,expected)
 for e in elements:
  m,n=e['row'],e['channel'];av=float(np.array([actual[m,n]],dtype='<u2').view('<f2')[0])
  ev=float(np.array([expected[m,n]],dtype='<u2').view('<f2')[0])
  e['absolute_error_vs_original_cpu']=abs(av-ev)
  e['ulp_vs_original_cpu']=int(abs(models.ordered(actual[m,n])-models.ordered(expected[m,n])))
  e['actual_value']=av;e['original_cpu_value']=ev
  e['scope']='FROZEN_ORIGINAL_FP16_OBSERVATION_NOT_NEW_VDEQ_PROBE_RESULT'
 keys=[e for e in elements if (e['row'],e['channel']) in KEY_DOTS]
 summary.update(raw_int32='INDEPENDENT_RAW_PATH_MATCH_DOES_NOT_IDENTIFY_ORIGINAL_FP16_L0C',
                new_vdeq16_execution='NOT_RUN',cause='UNRESOLVED',key_positions=keys)
 return summary,elements,bad
