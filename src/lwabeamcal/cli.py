"""Command-line interface; no calibration fitting or hidden downloads."""
import argparse
import json
import sys
from .cache import ModelCache
from .core import FrozenBackground
from .fitsio import calibrate_fits


def main(argv=None):
    parser=argparse.ArgumentParser(prog='lwabeamcal',description='Subtract frozen FAST background, then divide by primary-beam gain.')
    parser.add_argument('--version',action='version',version='lwabeamcal 0.1.0')
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('model-info',help='Show exact adopted background model and caveats')
    info=sub.add_parser('cache-info',help='Validate a portable raw FAST cache')
    info.add_argument('cache')
    imp=sub.add_parser('import-audit',help='Convert a legacy TD audit to a portable raw FAST/gain cache')
    imp.add_argument('audit');imp.add_argument('--output',required=True);imp.add_argument('--overwrite',action='store_true')
    corr=sub.add_parser('correct',help='Correct spec_v2 FITS without changing input')
    corr.add_argument('input');corr.add_argument('--cache',required=True);corr.add_argument('--output',required=True)
    corr.add_argument('--beam',choices=['memo178','lwapb'],default='memo178')
    corr.add_argument('--input-unit',choices=['jy','sfu'],help='Explicit override for known mislabeled/missing BUNIT')
    corr.add_argument('--output-unit',choices=['jy','sfu'],default='jy')
    corr.add_argument('--experimental',action='store_true',help='Allow 15-40/85-90 MHz only where FAST has support')
    corr.add_argument('--mask-fits',help='Use original provider TD_MASK from exactly matching native axes')
    corr.add_argument('--off-window',nargs=2,type=float,action='append',metavar=('START_MJD','END_MJD'),help='Explicit off-point window; overrides cache intervals')
    corr.add_argument('--terrain-min-alt-deg',type=float,default=12.)
    corr.add_argument('--min-gain',type=float,default=None)
    corr.add_argument('--max-time-gap-seconds',type=float,default=None)
    corr.add_argument('--chunk-size',type=int,default=4096)
    corr.add_argument('--compute-gain',action='store_true',help='Compute Memo178 directly instead of using cached gains')
    corr.add_argument('--overwrite',action='store_true')
    args=parser.parse_args(argv)
    try:
        if args.command=='model-info': result=FrozenBackground().metadata()
        elif args.command=='cache-info':
            cache=ModelCache.load(args.cache)
            result=dict(metadata=cache.metadata,sha256=cache.sha256,shape=list(cache.fast_jy.shape),
                        frequency_range_mhz=[float(cache.frequencies_mhz[0]),float(cache.frequencies_mhz[-1])],
                        time_range_mjd=[float(cache.times_mjd[0]),float(cache.times_mjd[-1])],has_gain=cache.gain is not None)
        elif args.command=='import-audit':
            cache=ModelCache.from_td_audit(args.audit);cache.save(args.output,overwrite=args.overwrite)
            result=dict(output=args.output,metadata=cache.metadata)
        else:
            result=calibrate_fits(args.input,args.output,cache=args.cache,beam=args.beam,
                     input_unit=args.input_unit,output_unit=args.output_unit,experimental=args.experimental,
                     mask_fits=args.mask_fits,off_windows_mjd=args.off_window,
                     terrain_min_alt_deg=args.terrain_min_alt_deg,min_gain=args.min_gain,
                     max_time_gap_seconds=args.max_time_gap_seconds,chunk_size=args.chunk_size,
                     overwrite=args.overwrite,use_cached_gain=not args.compute_gain)
        print(json.dumps(result,sort_keys=True,indent=2,allow_nan=False))
        return 0
    except (ValueError,KeyError,TypeError,OSError) as error:
        print(f'lwabeamcal: error: {error}',file=sys.stderr)
        return 2
