from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT/'framework/python'),str(ROOT),str(ROOT/'framework/custom_ops/draft_quant/test')]
from a2_common import checked_file,record,sha256,write_json
HERE=ROOT/'framework/custom_ops/draft_ac'
A_ROOT=HERE.parent/'draft_quant'; C_ROOT=HERE.parent/'draft_head'
