"""Exact rational rounding models. Fits are observations, never hardware contracts."""
from fractions import Fraction
import numpy as np


def decode(bits,eb,fb):
 sign=-1 if bits>>(eb+fb) else 1
 exp=(bits>>fb)&((1<<eb)-1);mant=bits&((1<<fb)-1);bias=(1<<(eb-1))-1
 if exp==(1<<eb)-1:raise ValueError('nonfinite binary float')
 if exp==0:e=1-bias-fb
 else:mant|=1<<fb;e=exp-bias-fb
 return sign*Fraction(mant)*(Fraction(2)**e)


def round_bits(value,eb,fb,mode='rne',negative_zero=False):
 """RNE, toward-zero, directed rounding, or inexact round-to-odd (jamming)."""
 value=Fraction(value);neg=value<0 or value==0 and negative_zero;x=abs(value)
 sign=int(neg)<<(eb+fb);bias=(1<<(eb-1))-1;emin=1-bias;emax=bias
 if not x:return sign
 e=x.numerator.bit_length()-x.denominator.bit_length()
 if x<Fraction(2)**e:e-=1
 step=max(e-fb,emin-fb);scaled=x/(Fraction(2)**step);q,r=divmod(scaled.numerator,scaled.denominator)
 if mode=='rne':q+=int(2*r>scaled.denominator or 2*r==scaled.denominator and q%2)
 elif mode=='rtz':pass
 elif mode=='rup':q+=int(bool(r) and not neg)
 elif mode=='rdn':q+=int(bool(r) and neg)
 elif mode=='odd':q=q|1 if r else q
 else:raise ValueError('unknown rounding model')
 if not q:return sign
 if q>=(1<<(fb+1)):q>>=1;step+=1
 encoded_e=step+fb
 if encoded_e>emax:
  toward_finite=mode in ('rtz','odd') or mode=='rup' and neg or mode=='rdn' and not neg
  return sign|((((1<<eb)-2)<<fb)|((1<<fb)-1) if toward_finite else ((1<<eb)-1)<<fb)
 if q<(1<<fb):return sign|q
 return sign|((encoded_e+bias)<<fb)|(q-(1<<fb))


def half(value,mode='rne',negative_zero=False):return round_bits(value,5,10,mode,negative_zero)
def f32(value,mode='rne',negative_zero=False):return round_bits(value,8,23,mode,negative_zero)


def word_fields(word):
 word=int(word);m=word&0xffffffff
 return dict(word_hex=f'{word:016x}',multiplier_bits=f'{m:08x}',multiplier=float(decode(m,8,23)),
  n_raw=(word>>32)&15,n_documented_value=((word>>32)&15)+1,mcb=(word>>36)&1,
  offset_raw=(word>>37)&511,sign_bit46=(word>>46)&1,relu_word_bit47=(word>>47)&1,reserved=(word>>48)&65535,
  scope='CANN900 documented field interpretation; microarchitecture not inferred')

# Fixed suite, declared before reading device values. No model is a replacement Golden.
DEFINITIONS={
 'A_original': 'RN16(RN32(RN32(dot)*M)); original Stage4V contract',
 'B_even_toward_zero': 'e=sign(dot)*2*floor(abs(dot)/2); RN16(RN32(RN32(e)*M)); hypothesis',
 'C_exact_product': 'RN16(exact_integer_dot*exact_M); single final rounding reference, not a VDEQ claim',
 'C_f32_input_single_round': 'RN16(exact(RN32(dot))*exact_M); no FP32 product rounding, hypothesis',
 'C_cast_odd': 'ODD16(RN32(RN32(dot)*M)); documented Cast mode category, not documented VDEQ mode',
 'D_int_to_f32_rtz': 'RN16(RN32(RTZ32(dot)*M)); hypothetical conversion rounding, INT32 unchanged',
 'D_int_to_f32_odd': 'RN16(RN32(ODD32(dot)*M)); hypothetical conversion rounding, INT32 unchanged',
 'D_mul32_rtz': 'RN16(RTZ32(RN32(dot)*M)); hypothetical directed product rounding',
 'D_final16_rtz': 'RTZ16(RN32(RN32(dot)*M)); mathematical alternative, no claimed dav_2002 VDEQ setting',
 'D_mul32_floor': 'RN16(RDN32(RN32(dot)*M)); hypothetical product rounding',
 'D_negative_odd_plus1': 'dot+1 only for negative odd dot, then original A; post-hoc user hypothesis',
 'D_clear_twos_complement_lsb': 'dot & ~1, then original A; arithmetic floor to even, distinct from toward zero',
 'D_product_23bit_rtz': 'round product toward zero to23 significant bits (8 exponent bits), then RN16; hypothetical precision loss',
 'D_product_22bit_rtz': 'round product toward zero to22 significant bits (8 exponent bits), then RN16; hypothetical precision loss',
}


def predict(dot,scale_bits):
 d=int(dot);m=decode(int(scale_bits),8,23)
 if abs(d)>41290240 or m<0:raise ValueError('outside frozen dot/Scale scope')
 def original(v):
  x=decode(f32(v),8,23);prod=decode(f32(x*m,negative_zero=v<0 and m==0),8,23)
  return half(prod,negative_zero=v<0 and m==0)
 x=decode(f32(d),8,23);exact=Fraction(d)*m;product=x*m;rounded=decode(f32(product),8,23)
 negzero=d<0 and m==0
 even=(1 if d>=0 else -1)*(abs(d)//2*2)
 return {
  'A_original':original(d),'B_even_toward_zero':original(even),
  'C_exact_product':half(exact,negative_zero=negzero),
  'C_f32_input_single_round':half(product,negative_zero=negzero),
  'C_cast_odd':half(rounded,'odd',negzero),
  'D_int_to_f32_rtz':half(decode(f32(decode(f32(d,'rtz'),8,23)*m,negative_zero=negzero),8,23),negative_zero=negzero),
  'D_int_to_f32_odd':half(decode(f32(decode(f32(d,'odd'),8,23)*m,negative_zero=negzero),8,23),negative_zero=negzero),
  'D_mul32_rtz':half(decode(f32(product,'rtz',negzero),8,23),negative_zero=negzero),
  'D_final16_rtz':half(rounded,'rtz',negzero),
  'D_mul32_floor':half(decode(f32(product,'rdn',negzero),8,23),negative_zero=negzero),
  'D_negative_odd_plus1':original(d+1 if d<0 and d%2 else d),
  'D_clear_twos_complement_lsb':original(d&~1),
  'D_product_23bit_rtz':half(decode(round_bits(product,8,22,'rtz',negzero),8,22),negative_zero=negzero),
  'D_product_22bit_rtz':half(decode(round_bits(product,8,21,'rtz',negzero),8,21),negative_zero=negzero),
 }


def midpoint_info(dot,scale_bits):
 exact=Fraction(int(dot))*decode(int(scale_bits),8,23);mag=abs(exact)
 center_bits=min(half(mag),0x7bff);center=decode(center_bits,5,10)
 boundaries=[]
 if center_bits>0:boundaries.append((center_bits-1,center_bits,decode(center_bits-1,5,10),center))
 hi=decode(center_bits+1,5,10) if center_bits<0x7bff else Fraction(65536)
 boundaries.append((center_bits,center_bits+1,center,hi))
 low_bits,high_bits,lo,hi=min(boundaries,key=lambda x:abs(mag-(x[2]+x[3])/2))
 mid=(lo+hi)/2;distance=(mag-mid)/(hi-lo)
 x32=decode(f32(dot),8,23);p32=decode(f32(x32*decode(int(scale_bits),8,23)),8,23)
 mids=[(l+h)/2 for _,_,l,h in boundaries]
 return dict(status='FINITE_BRACKET',exact_product=float(exact),exact_product_hex=float(exact).hex(),fp32_product_hex=float(p32).hex(),fp32_product_bits=f'{f32(x32*decode(int(scale_bits),8,23)):08x}',
  lower_magnitude_bits=f'{low_bits:04x}',upper_magnitude_bits=f'{high_bits:04x}',
  upper_value_is_virtual_overflow=high_bits==0x7c00,
  midpoint_magnitude=float(mid),signed_distance_in_ulp=float(distance),
  distance_numerator=distance.numerator,distance_denominator=distance.denominator,
  exact_midpoint=mag in mids,fp32_product_hits_midpoint=abs(p32) in mids,
  midpoint_created_by_fp32=mag not in mids and abs(p32) in mids,int32_exact_in_fp32=x32==dot)


def ordered(bits):
 a=np.asarray(bits,dtype=np.uint16).astype(np.int64)
 return np.where(a&0x8000,0x8000-(a&0x7fff),0x8000+a)


def classify(dots,mask):
 d=np.asarray(dots,dtype=np.int64);mask=np.asarray(mask,bool)
 return {f'{sign}_{parity}':int(np.count_nonzero(mask&s&(d%2==p))) for sign,s in [('negative',d<0),('zero',d==0),('positive',d>0)] for parity,p in [('even',0),('odd',1)]}


def analyze(dots,words,actual_bits,expected_bits):
 dots=np.asarray(dots,dtype=np.int64);words=np.asarray(words,dtype='<u8')
 actual=np.asarray(actual_bits,dtype='<u2');expected=np.asarray(expected_bits,dtype='<u2')
 if dots.shape!=(16,256) or words.shape!=(256,) or actual.shape!=dots.shape or expected.shape!=dots.shape:raise ValueError('exact Tile0 shape required')
 if not np.isfinite(actual.view('<f2')).all() or not np.isfinite(expected.view('<f2')).all():raise ValueError('nonfinite active output; numerical model scope invalid')
 fields=[word_fields(w) for w in words]
 if any(x['mcb'] or x['n_raw'] or x['offset_raw'] or x['relu_word_bit47'] or x['reserved'] or x['sign_bit46']!=1 or int(x['multiplier_bits'],16)&0x1fff for x in fields):raise ValueError('not frozen Stage4V codec; inspect word identities')
 preds={n:np.empty(dots.shape,'<u2') for n in DEFINITIONS};elements=[]
 for row in range(16):
  for col in range(256):
   d=int(dots[row,col]);m=int(words[col]&np.uint64(0xffffffff));values=predict(d,m)
   for n,v in values.items():preds[n][row,col]=v
   elements.append(dict(row=row,channel=col,dot=d,negative=d<0,odd=bool(d%2),scale_word=fields[col]['word_hex'],
    actual_bits=f'{int(actual[row,col]):04x}',expected_bits=f'{int(expected[row,col]):04x}',
    predictions={n:f'{v:04x}' for n,v in values.items()},midpoint=midpoint_info(d,m)))
 if not np.array_equal(preds['A_original'],expected):raise ValueError('original expected file differs from independently reconstructed A; no model fitting permitted')
 original=actual!=expected;records={};groups={}
 for name,values in preds.items():
  bad=values!=actual;nonfinite=int(np.count_nonzero(~np.isfinite(values.view('<f2'))))
  records[name]=dict(definition=DEFINITIONS[name],bit_mismatches=int(bad.sum()),explained_original=int(np.count_nonzero(original&~bad)),
    original_unexplained=int(np.count_nonzero(original&bad)),new_errors_on_original_matches=int(np.count_nonzero(~original&bad)),
    explains_all_original=not np.any(original&bad),matches_entire_tile=not bad.any(),prediction_nonfinite=nonfinite,
    max_ulp=None if nonfinite else int(np.max(np.abs(ordered(values)-ordered(actual)))),
    ulp_histogram=None if nonfinite else {str(int(k)):int(v) for k,v in zip(*np.unique(np.abs(ordered(values)-ordered(actual)),return_counts=True))},
    mismatch_distribution=classify(dots,bad),interpretation='OBSERVATIONAL_MODEL_ONLY_NOT_HARDWARE_OR_BENCHMARK_ADMISSION')
  groups.setdefault(values.tobytes(),[]).append(name)
 bad_elements=[e for e in elements if e['actual_bits']!=e['expected_bits']]
 def midpoint_stats(rows):
  distances=[abs(e['midpoint']['signed_distance_in_ulp']) for e in rows if e['midpoint']['status']=='FINITE_BRACKET']
  return dict(count=len(rows),exact_midpoints=sum(e['midpoint'].get('exact_midpoint',False) for e in rows),
    fp32_created_midpoints=sum(e['midpoint'].get('midpoint_created_by_fp32',False) for e in rows),
    distance_abs_ulp_quantiles=None if not distances else {str(p):float(np.quantile(distances,p)) for p in (0,.25,.5,.75,1)},
    within_1_over_4096_ulp=sum(x<=1/4096 for x in distances))
 for name,values in preds.items():
  mask=(values!=actual).ravel();new_mask=(~original&(values!=actual)).ravel()
  records[name]['mismatch_midpoint_statistics']=midpoint_stats([e for e,m in zip(elements,mask) if m])
  records[name]['new_error_midpoint_statistics']=midpoint_stats([e for e,m in zip(elements,new_mask) if m])
  records[name]['new_error_distribution']=classify(dots,~original&(values!=actual))
 result=dict(status='TILE0_MODELS_ANALYZED_CAUSE_UNRESOLVED',count=4096,original_mismatches=int(original.sum()),
  original_match_count=int((~original).sum()),user_reported_six_confirmed=int(original.sum())==6,
  original_distribution=classify(dots,original),all_distribution=classify(dots,np.ones(dots.shape,bool)),
  original_max_ulp=int(np.max(np.abs(ordered(actual)-ordered(expected)))),
  original_toward_zero=int(np.count_nonzero(original&(np.abs(actual.view('<f2').astype(np.float64))<np.abs(expected.view('<f2').astype(np.float64))))),
  dot_abs_gt_2pow24=int(np.count_nonzero(np.abs(dots)>2**24)),dot_abs_gt_2pow25=int(np.count_nonzero(np.abs(dots)>2**25)),
  midpoint_statistics=dict(all=midpoint_stats(elements),original_mismatches=midpoint_stats(bad_elements),original_matches=midpoint_stats([e for e in elements if e['actual_bits']==e['expected_bits']])),
  models=records,indistinguishable_on_entire_tile=[x for x in groups.values() if len(x)>1],scale_fields=fields,
  conclusion='A fitting model does not identify the accumulator value or hardware rounding. Independent Raw INT32 and/or VDEQ evidence remains required.',
  benchmark_admission='PROHIBITED',replacement_golden='NOT_CREATED')
 return result,elements,bad_elements
