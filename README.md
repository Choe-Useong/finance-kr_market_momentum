# finance-kr_market_momentum

Korean equity universe build + close download + quick momentum backtest utilities.

## Files
- `1_build_universe2.py`: Build universe with configurable screening (marcap/amount, top N or top %)
- `2_download_update_close.py`: Download/update close prices (YF with FDR fallback)
- `5_quick_backtest.py`: Quick grid backtest for rank momentum (supports EQUAL/RP weights)

## Quick start
1. Build universe
   - Edit settings in `1_build_universe2.py`
   - Run: `python 1_build_universe2.py`
2. Download close prices
   - Edit settings in `2_download_update_close.py`
   - Run: `python 2_download_update_close.py`
3. Run quick backtest
   - Edit settings in `5_quick_backtest.py`
   - Run: `python 5_quick_backtest.py`

## Notes
- Files are saved in UTF-8.
- If you see garbled Korean text in your editor, ensure UTF-8 (or UTF-8 BOM) encoding.
